"""GRU-based temporal encoder for monthly vitals/labs trajectories."""

from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64,
                 embed_dim: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.GELU(),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        _, h = self.gru(x)
        return self.head(h[-1])
