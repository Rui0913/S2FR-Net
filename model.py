"""Top-level CLIP-RWKV pan-sharpening model."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from clip_rwkv_fusion import CLIPRWKVFusion
from context_attention import ContextAwareAttentionModule
from layers import ConvBlock, ResNetBlock


def _haar_kernels() -> torch.Tensor:
    kernels = [
        [[0.5, 0.5], [0.5, 0.5]],
        [[-0.5, -0.5], [0.5, 0.5]],
        [[-0.5, 0.5], [-0.5, 0.5]],
        [[0.5, -0.5], [-0.5, 0.5]],
    ]
    return torch.tensor(kernels).unsqueeze(1)


class HaarDWT(nn.Module):
    """Differentiable 2D Haar wavelet transform."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", _haar_kernels())

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape
        reshaped = x.reshape(batch_size * channels, 1, height, width)
        output = functional.conv2d(
            reshaped, self.weight.to(dtype=x.dtype), stride=2
        )
        output = output.reshape(
            batch_size, channels, 4, height // 2, width // 2
        )
        return tuple(output[:, :, index] for index in range(4))


class HaarIWT(nn.Module):
    """Inverse 2D Haar wavelet transform."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", _haar_kernels())

    def forward(
        self,
        low_low: torch.Tensor,
        high_low: torch.Tensor,
        low_high: torch.Tensor,
        high_high: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, channels, height, width = low_low.shape
        stacked = torch.stack(
            [low_low, high_low, low_high, high_high], dim=2
        ).reshape(batch_size * channels, 4, height, width)
        output = functional.conv_transpose2d(
            stacked, self.weight.to(dtype=low_low.dtype), stride=2
        )
        return output.reshape(batch_size, channels, height * 2, width * 2)


def _channel_std(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected a 4D feature map, got shape {tuple(x.shape)}")
    mean = x.mean(dim=(2, 3), keepdim=True)
    variance = (x - mean).pow(2).mean(dim=(2, 3), keepdim=True)
    return variance.sqrt()


class SpatialFrequencyFusion(nn.Module):
    """Fuse two feature maps using spatial and channel attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden_channels = max(channels // 2, 1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, channels, 3, padding=1),
            nn.Sigmoid(),
        )
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_channels, channels * 2, 1),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Conv2d(channels * 2, channels, 3, padding=1)

    def forward(
        self, spatial_features: torch.Tensor, reference_features: torch.Tensor
    ) -> torch.Tensor:
        spatial_weights = self.spatial_attention(
            spatial_features - reference_features
        )
        spatial_residual = reference_features * spatial_weights + spatial_features
        combined = torch.cat([spatial_residual, reference_features], dim=1)
        channel_weights = self.channel_attention(
            _channel_std(combined) + self.average_pool(combined)
        )
        return self.output_projection(channel_weights * combined) + spatial_features


class DenseBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(output_channels, output_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AffineCouplingBlock(nn.Module):
    """Invertible affine coupling block."""

    def __init__(self, channels: int, clamp: float = 2.0) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("Affine coupling requires an even channel count.")
        self.split_channels = channels // 2
        self.first_transform = DenseBlock(
            self.split_channels, self.split_channels * 2
        )
        self.second_transform = DenseBlock(
            self.split_channels, self.split_channels * 2
        )
        self.clamp = clamp

    def _stable_scale(self, scale: torch.Tensor) -> torch.Tensor:
        return self.clamp * (torch.sigmoid(scale) * 2 - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = torch.split(x, self.split_channels, dim=1)
        scale, translation = self.first_transform(first).chunk(2, dim=1)
        transformed_second = second * torch.exp(self._stable_scale(scale)) + translation

        scale, translation = self.second_transform(transformed_second).chunk(2, dim=1)
        transformed_first = first * torch.exp(self._stable_scale(scale)) + translation
        return torch.cat([transformed_first, transformed_second], dim=1)

    def inverse(self, output: torch.Tensor) -> torch.Tensor:
        transformed_first, transformed_second = torch.split(
            output, self.split_channels, dim=1
        )
        scale, translation = self.second_transform(transformed_second).chunk(2, dim=1)
        first = (transformed_first - translation) * torch.exp(
            -self._stable_scale(scale)
        )

        scale, translation = self.first_transform(first).chunk(2, dim=1)
        second = (transformed_second - translation) * torch.exp(
            -self._stable_scale(scale)
        )
        return torch.cat([first, second], dim=1)


class MultispectralRefinementNet(nn.Module):
    """Refine multispectral features with invertible coupling blocks."""

    def __init__(
        self,
        input_channels: int = 3,
        feature_channels: int = 64,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv2d(
            input_channels, feature_channels, 3, padding=1
        )
        self.blocks = nn.ModuleList(
            [AffineCouplingBlock(feature_channels) for _ in range(num_blocks)]
        )
        self.output_projection = nn.Conv2d(
            feature_channels, input_channels, 3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(x)
        for block in self.blocks:
            features = block(features)
        return x + self.output_projection(features)


class PanSharpeningNet(nn.Module):
    """Complete CLIP-RWKV pan-sharpening network."""

    def __init__(
        self,
        device: torch.device | str,
        feature_channels: int = 32,
        context_channels: int = 16,
    ) -> None:
        super().__init__()
        self.clip_fusion = CLIPRWKVFusion(
            embedding_dim=feature_channels, num_layers=4, device=device
        )
        self.ms_refinement_1 = MultispectralRefinementNet(4, feature_channels, 4)
        self.ms_refinement_2 = MultispectralRefinementNet(4, feature_channels, 4)
        self.frequency_fusion = SpatialFrequencyFusion(3)

        self.four_to_three = ConvBlock(4, 3)
        self.six_to_three = ConvBlock(6, 3)
        self.image_to_context = ConvBlock(3, context_channels)
        self.pan_to_context = ConvBlock(1, context_channels)
        self.context_to_image = ConvBlock(context_channels, 3)

        self.context_attention = ContextAwareAttentionModule(context_channels)
        self.residual_block = ResNetBlock(context_channels)
        self.dwt = HaarDWT()
        self.iwt = HaarIWT()

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        clip_features = self.clip_fusion(ms, pan)
        upsampled_ms = functional.interpolate(
            ms, size=pan.shape[2:], mode="bicubic", align_corners=False
        )

        ms_ll, ms_hl, ms_lh, ms_hh = self.dwt(upsampled_ms)
        _, pan_hl, pan_lh, pan_hh = self.dwt(pan)
        fused_wavelet = self.iwt(
            ms_ll,
            pan_hl.repeat(1, 3, 1, 1),
            pan_lh.repeat(1, 3, 1, 1),
            pan_hh.repeat(1, 3, 1, 1),
        )

        refined_ms = torch.cat([upsampled_ms, pan], dim=1)
        refined_ms = self.ms_refinement_1(refined_ms)
        refined_ms = self.ms_refinement_2(refined_ms)
        refined_ms = self.ms_refinement_2(refined_ms)
        refined_ms = self.four_to_three(refined_ms)

        wavelet_features = self.frequency_fusion(fused_wavelet, refined_ms)
        clip_features = self.frequency_fusion(clip_features, refined_ms)
        fused_image = self.six_to_three(
            torch.cat([wavelet_features, clip_features], dim=1)
        ) + upsampled_ms

        context = self.image_to_context(fused_image)
        pan_context = self.pan_to_context(pan)
        wavelet_context = self.image_to_context(wavelet_features)
        context = self.context_attention(context, pan_context, wavelet_context)
        return self.context_to_image(self.residual_block(context))
