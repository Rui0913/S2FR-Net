"""Reusable convolutional layers used by the fusion network."""

from __future__ import annotations

import torch
from torch import nn


def _activation(name: str | None) -> nn.Module | None:
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "prelu": lambda: nn.PReLU(init=0.5),
        "lrelu": lambda: nn.LeakyReLU(0.2, inplace=True),
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    if name is None:
        return None
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class ConvBlock(nn.Module):
    """Convolution followed by optional normalization and activation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        activation: str | None = "prelu",
        norm: str | None = None,
        pad_mode: str | None = None,
    ) -> None:
        super().__init__()

        if pad_mode not in (None, "reflection"):
            raise ValueError(f"Unsupported padding mode: {pad_mode}")
        if norm not in (None, "batch", "instance"):
            raise ValueError(f"Unsupported normalization: {norm}")

        self.padding = (
            nn.ReflectionPad2d(padding) if pad_mode == "reflection" else nn.Identity()
        )
        convolution_padding = 0 if pad_mode == "reflection" else padding
        self.conv = nn.Conv2d(
            input_size,
            output_size,
            kernel_size,
            stride,
            convolution_padding,
            bias=bias,
        )
        if norm == "batch":
            self.norm = nn.BatchNorm2d(output_size)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(output_size)
        else:
            self.norm = nn.Identity()
        self.activation = _activation(activation) or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(self.padding(x))))


class ResNetBlock(nn.Module):
    """Two-layer residual convolutional block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        scale: float = 1.0,
        activation: str | None = "prelu",
        norm: str | None = "batch",
        pad_mode: str | None = None,
    ) -> None:
        super().__init__()
        if pad_mode not in (None, "reflection"):
            raise ValueError(f"Unsupported padding mode: {pad_mode}")
        if norm not in (None, "batch", "instance"):
            raise ValueError(f"Unsupported normalization: {norm}")

        self.scale = scale
        self.padding = (
            nn.ReflectionPad2d(padding) if pad_mode == "reflection" else nn.Identity()
        )
        convolution_padding = 0 if pad_mode == "reflection" else padding
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            convolution_padding,
            bias=bias,
        )
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride,
            convolution_padding,
            bias=bias,
        )
        if norm == "batch":
            self.norm = nn.BatchNorm2d(channels)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(channels)
        else:
            self.norm = nn.Identity()
        self.activation = _activation(activation) or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.activation(self.norm(self.conv1(self.padding(x))))
        output = self.activation(self.norm(self.conv2(self.padding(output))))
        return x + output * self.scale
