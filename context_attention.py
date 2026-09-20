"""Context-aware attention for multispectral and panchromatic features."""

from __future__ import annotations

import torch
from torch import nn

from layers import ConvBlock
from mamba_fusion import MambaFusionNet


class LayerNorm2d(nn.Module):
    """Apply layer normalization over channels of a 2D feature map."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None, None] * normalized + self.bias[:, None, None]


class SpatialMLP(nn.Module):
    """Point-wise MLP implemented with 1x1 convolutions."""

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or channels * 4
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ContextAwareAttentionModule(nn.Module):
    """Fuse two context stages with shared Mamba-based feature interaction."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pan_conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)
        self.ms_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.to_features = ConvBlock(3, channels)
        self.norm1 = LayerNorm2d(channels)
        self.mlp1 = SpatialMLP(channels)
        self.norm2 = LayerNorm2d(channels)
        self.mlp2 = SpatialMLP(channels)
        self.mamba_fusion = MambaFusionNet(channels=channels)

    def forward(
        self,
        ms_features: torch.Tensor,
        pan_features: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        pan_features = self.pan_conv(pan_features.unsqueeze(1)).squeeze(1)
        ms_features = self.ms_conv(ms_features)

        first_fusion = self.to_features(self.mamba_fusion(ms_features, pan_features))
        first_residual = first_fusion + pan_features
        first_output = first_residual + self.mlp1(self.norm1(first_residual))

        second_fusion = self.to_features(
            self.mamba_fusion(first_output, context_features)
        )
        second_residual = second_fusion + first_output
        return second_residual + self.mlp2(self.norm2(second_residual))
