"""Simple late-fusion baseline: concatenate then project."""

from __future__ import annotations

import torch
from torch import nn


class LateFusion(nn.Module):
    def __init__(self, num_modalities: int = 4, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, embed_dim), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.embed_dim = embed_dim

    def forward(self, modality_embeds: list[torch.Tensor]) -> torch.Tensor:
        return self.proj(torch.cat(modality_embeds, dim=-1))
