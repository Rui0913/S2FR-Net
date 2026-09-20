"""Mamba-based feature fusion blocks."""

from __future__ import annotations

import math
import numbers

import torch
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from torch import nn

from layers import ConvBlock
from refinement import Refine


def _to_tokens(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, "b c h w -> b (h w) c")


def _to_image(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return rearrange(x, "b (h w) c -> b c h w", h=height, w=width)


class BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("Layer normalization expects a single feature dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(variance + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int | tuple[int, ...]) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        if len(normalized_shape) != 1:
            raise ValueError("Layer normalization expects a single feature dimension.")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(variance + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    """Layer normalization that accepts token sequences or 2D feature maps."""

    def __init__(self, channels: int, bias: bool = True) -> None:
        super().__init__()
        self.body = WithBiasLayerNorm(channels) if bias else BiasFreeLayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            height, width = x.shape[-2:]
            return _to_image(self.body(_to_tokens(x)), height, width)
        return self.body(x)


class PatchEmbed(nn.Module):
    """Convert a feature map to a sequence of patch tokens."""

    def __init__(
        self,
        input_channels: int,
        embedding_dim: int,
        patch_size: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels,
            embedding_dim,
            kernel_size=patch_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class PatchUnembed(nn.Module):
    """Convert a token sequence back to a 2D feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(
        self, x: torch.Tensor, spatial_size: tuple[int, int]
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.transpose(1, 2).reshape(
            batch_size, self.channels, spatial_size[0], spatial_size[1]
        )


def _square_spatial_size(token_count: int) -> tuple[int, int]:
    side = math.isqrt(token_count)
    if side * side != token_count:
        raise ValueError(f"Expected a square token grid, got {token_count} tokens.")
    return side, side


class CrossStateMamba(nn.Module):
    """Exchange modality tokens and perform cross-state Mamba fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.patch_unembed = PatchUnembed(channels)
        self.to_tokens = PatchEmbed(channels, channels)
        self.shallow_fusion = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.cross_mamba = Mamba(channels, bimamba_type="v3")
        self.ms_norm = LayerNorm(channels)
        self.pan_norm = LayerNorm(channels)
        self.depthwise_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

    def forward(
        self,
        ms: torch.Tensor,
        pan: torch.Tensor,
        ms_residual: torch.Tensor,
        pan_residual: torch.Tensor,
        fusion_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ms_residual = ms + ms_residual
        pan_residual = pan + pan_residual
        ms = self.ms_norm(ms_residual)
        pan = self.pan_norm(pan_residual)

        channels = ms.shape[-1]
        half = channels // 2
        ms_swapped = torch.cat([pan[:, :, :half], ms[:, :, half:]], dim=2)
        pan_swapped = torch.cat([ms[:, :, :half], pan[:, :, half:]], dim=2)

        spatial_size = _square_spatial_size(ms.shape[1])
        ms_swapped = self.patch_unembed(ms_swapped, spatial_size)
        pan_swapped = self.patch_unembed(pan_swapped, spatial_size)

        ms_swapped = self.shallow_fusion(
            torch.cat([ms_swapped, pan_swapped], dim=1)
        ) + ms_swapped
        pan_swapped = self.shallow_fusion(
            torch.cat([pan_swapped, ms_swapped], dim=1)
        ) + pan_swapped

        ms_swapped = self.to_tokens(ms_swapped)
        pan_swapped = self.to_tokens(pan_swapped)
        fusion_residual = ms_swapped + fusion_residual

        ms = self.ms_norm(fusion_residual)
        pan = self.pan_norm(pan_swapped)
        fused = self.cross_mamba(
            self.ms_norm(ms), extra_emb=self.pan_norm(pan)
        )

        spatial_size = _square_spatial_size(fused.shape[1])
        fused_image = self.patch_unembed(fused, spatial_size)
        fused = (self.depthwise_conv(fused_image) + fused_image).flatten(2).transpose(1, 2)
        return fused, pan, ms_residual, pan_residual, fusion_residual


class MambaFusionNet(nn.Module):
    """Three-stage Mamba fusion network for aligned feature maps."""

    def __init__(self, channels: int = 16, num_stages: int = 3) -> None:
        super().__init__()
        self.ms_to_tokens = PatchEmbed(channels, channels)
        self.pan_to_tokens = PatchEmbed(channels, channels)
        self.fusion_stages = nn.ModuleList(
            [CrossStateMamba(channels) for _ in range(num_stages)]
        )
        self.patch_unembed = PatchUnembed(channels)
        self.refine = Refine(channels, 3)
        self.skip_projection = ConvBlock(channels, 3)

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        spatial_size = ms.shape[-2:]
        ms_skip = ms
        ms_tokens = self.ms_to_tokens(ms)
        pan_tokens = self.pan_to_tokens(pan)

        ms_residual = torch.zeros_like(ms_tokens)
        pan_residual = torch.zeros_like(pan_tokens)
        fusion_residual = torch.zeros_like(ms_tokens)

        for stage in self.fusion_stages:
            (
                ms_tokens,
                pan_tokens,
                ms_residual,
                pan_residual,
                fusion_residual,
            ) = stage(
                ms_tokens,
                pan_tokens,
                ms_residual,
                pan_residual,
                fusion_residual,
            )

        fused_image = self.patch_unembed(ms_tokens, spatial_size)
        return self.refine(fused_image) + self.skip_projection(ms_skip)
