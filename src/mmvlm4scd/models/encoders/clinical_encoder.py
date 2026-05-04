"""MLP encoder for tabular clinical features."""

from __future__ import annotations

import torch
from torch import nn


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, embed_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
