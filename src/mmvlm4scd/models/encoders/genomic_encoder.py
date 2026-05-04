"""Encoder for HBB variant indicators + polygenic-like signal.

We project the binary variant block and the continuous polygenic-like
block separately, then concatenate to expose both effect types to fusion.
"""

from __future__ import annotations

import torch
from torch import nn


class GenomicEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64, dropout: float = 0.1,
                 split: int = 16):
        super().__init__()
        self.split = split
        self.variant_proj = nn.Sequential(
            nn.Linear(split, 64), nn.GELU(), nn.Dropout(dropout),
        )
        self.pgs_proj = nn.Sequential(
            nn.Linear(input_dim - split, 64), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128, embed_dim), nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.variant_proj(x[..., : self.split])
        p = self.pgs_proj(x[..., self.split:])
        return self.head(torch.cat([v, p], dim=-1))
