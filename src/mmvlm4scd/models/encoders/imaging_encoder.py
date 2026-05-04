"""Imaging encoder.

In production, this would wrap a CNN (e.g. ResNet-18) over peripheral
blood-smear patches. Here we accept pre-computed image embeddings so the
benchmark is reproducible without bundling an image archive. The encoder
is a small MLP head that adapts the embedding to the fusion dimension.
"""

from __future__ import annotations

import torch
from torch import nn


class ImagingEncoder(nn.Module):
    def __init__(self, input_dim: int = 64, embed_dim: int = 64,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
