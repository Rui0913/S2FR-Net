"""CLIP-guided RWKV feature fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from einops import rearrange
from torch import nn
from transformers import CLIPModel, CLIPProcessor


class CLIPSemanticGuidance(nn.Module):
    """Generate spatial and spectral guidance maps from frozen CLIP features."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        print(f"Loading CLIP guidance model: {model_name}")

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.clip_model.eval()
        self.clip_model.requires_grad_(False)

        spatial_prompt = [
            "High frequency details, edges, texture, sharp lines, structures"
        ]
        spectral_prompt = [
            "Smooth colors, spectral information, vegetation, water, flat regions"
        ]
        self.register_buffer(
            "spatial_text_embedding", self._encode_text(spatial_prompt), persistent=False
        )
        self.register_buffer(
            "spectral_text_embedding",
            self._encode_text(spectral_prompt),
            persistent=False,
        )

    def _encode_text(self, prompts: list[str]) -> torch.Tensor:
        inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            features = self.clip_model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    def train(self, mode: bool = True):
        """Keep the frozen CLIP backbone in evaluation mode."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def _extract_patch_features(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[2:] != (224, 224):
            image = functional.interpolate(
                image, size=(224, 224), mode="bilinear", align_corners=False
            )
        normalized = (image - 0.48145466) / 0.26862954

        with torch.no_grad():
            output = self.clip_model.vision_model(pixel_values=normalized)
            patch_tokens = output.last_hidden_state[:, 1:, :]
            patch_tokens = self.clip_model.visual_projection(patch_tokens)
            return patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)

    @staticmethod
    def _similarity_map(
        features: torch.Tensor,
        text_embedding: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        similarity = torch.matmul(features, text_embedding.t())
        grid_size = int(features.shape[1] ** 0.5)
        similarity = rearrange(
            similarity,
            "b (h w) c -> b c h w",
            h=grid_size,
            w=grid_size,
        )
        return torch.sigmoid(
            functional.interpolate(similarity, size=output_size, mode="bilinear")
        )

    def forward(
        self, pan: torch.Tensor, upsampled_ms: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_size = pan.shape[2:]
        pan_features = self._extract_patch_features(pan.repeat(1, 3, 1, 1))
        ms_features = self._extract_patch_features(upsampled_ms)
        spatial_weights = self._similarity_map(
            pan_features, self.spatial_text_embedding, output_size
        )
        spectral_weights = self._similarity_map(
            ms_features, self.spectral_text_embedding, output_size
        )
        return spatial_weights, spectral_weights


class RWKVSpatialMixing(nn.Module):
    """Linear global attention used for spatial feature mixing."""

    def __init__(self, embedding_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(embedding_dim)
        self.receptance = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.output = nn.Linear(embedding_dim, embedding_dim, bias=False)

    @staticmethod
    def _global_attention(
        receptance: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        key = key.softmax(dim=-2)
        context = torch.matmul(key.transpose(-1, -2), value)
        return torch.matmul(receptance, context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        residual = x
        tokens = self.norm(rearrange(x, "b c h w -> b (h w) c"))

        receptance = rearrange(
            self.receptance(tokens), "b l (h d) -> b l h d", h=self.num_heads
        )
        key = rearrange(
            self.key(tokens), "b l (h d) -> b l h d", h=self.num_heads
        )
        value = rearrange(
            self.value(tokens), "b l (h d) -> b l h d", h=self.num_heads
        )
        mixed = self._global_attention(receptance, key, value)
        mixed = rearrange(mixed, "b l h d -> b l (h d)")
        mixed = self.output(mixed)
        mixed = rearrange(
            mixed, "b (h w) c -> b c h w", h=height, w=width
        )
        return residual + mixed


class ChannelMixing(nn.Module):
    """RWKV-style gated channel feed-forward network."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.receptance = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim * 4, bias=False)
        self.value = nn.Linear(embedding_dim * 4, embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        residual = x
        tokens = self.norm(rearrange(x, "b c h w -> b (h w) c"))
        receptance = torch.sigmoid(self.receptance(tokens))
        key = torch.square(torch.relu(self.key(tokens)))
        output = receptance * self.value(key)
        output = rearrange(
            output, "b (h w) c -> b c h w", h=height, w=width
        )
        return residual + output


class SemanticModalityFusion(nn.Module):
    """Inject CLIP guidance into fused modality features."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embedding_dim, embedding_dim, 1),
            nn.GELU(),
        )
        self.guidance_projection = nn.Conv2d(2, embedding_dim, 1)
        self.output_projection = nn.Conv2d(embedding_dim, embedding_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        modality_features: torch.Tensor,
        spatial_weights: torch.Tensor,
        spectral_weights: torch.Tensor,
    ) -> torch.Tensor:
        combined = features + modality_features
        gated = combined * self.feature_gate(combined)
        guidance = torch.cat([spatial_weights, spectral_weights], dim=1)
        guidance = torch.sigmoid(self.guidance_projection(guidance))
        return self.output_projection(gated * (1 + guidance))


class BRWKVBlock(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.modality_fusion = SemanticModalityFusion(embedding_dim)
        self.spatial_mixing = RWKVSpatialMixing(embedding_dim)
        self.channel_mixing = ChannelMixing(embedding_dim)

    def forward(
        self,
        features: torch.Tensor,
        modality_features: torch.Tensor,
        spatial_weights: torch.Tensor,
        spectral_weights: torch.Tensor,
    ) -> torch.Tensor:
        features = self.modality_fusion(
            features, modality_features, spatial_weights, spectral_weights
        )
        features = self.spatial_mixing(features)
        return self.channel_mixing(features)


class CLIPRWKVFusion(nn.Module):
    """Fuse low-resolution MS and high-resolution PAN images."""

    def __init__(
        self,
        ms_channels: int = 3,
        pan_channels: int = 1,
        embedding_dim: int = 64,
        num_layers: int = 4,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.semantic_guidance = CLIPSemanticGuidance(device=device)
        input_channels = ms_channels + pan_channels
        self.input_projection = nn.Conv2d(input_channels, embedding_dim, 3, padding=1)
        self.modality_encoder = nn.Conv2d(input_channels, embedding_dim, 1)
        self.layers = nn.ModuleList(
            [BRWKVBlock(embedding_dim) for _ in range(num_layers)]
        )
        self.output_projection = nn.Conv2d(embedding_dim, ms_channels, 3, padding=1)

    def forward(self, ms: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        upsampled_ms = functional.interpolate(
            ms, size=pan.shape[2:], mode="bilinear", align_corners=False
        )
        spatial_weights, spectral_weights = self.semantic_guidance(
            pan, upsampled_ms
        )
        model_input = torch.cat([upsampled_ms, pan], dim=1)
        features = self.input_projection(model_input)
        modality_features = self.modality_encoder(model_input)

        for layer in self.layers:
            features = layer(
                features,
                modality_features,
                spatial_weights,
                spectral_weights,
            )

        return upsampled_ms + self.output_projection(features)
