"""Output refinement layers."""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Residual channel-attention block."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.process = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.process(x)
        weights = self.channel_gate(self.pool(residual))
        return x + weights * residual


class Refine(nn.Module):
    """Refine fused features and reconstruct the output image."""

    def __init__(self, feature_channels: int, output_channels: int) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)
        self.attention = ChannelAttention(feature_channels)
        self.output_conv = nn.Conv2d(feature_channels, output_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_conv(self.attention(self.input_conv(x)))
