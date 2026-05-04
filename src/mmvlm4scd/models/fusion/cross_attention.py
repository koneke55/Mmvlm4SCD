"""Cross-attention fusion: clinical features query other modalities.

In SCD, the clinical record is the most reliable modality. We let it act
as the query stream while genomic / imaging / temporal embeddings serve
as keys and values. This biases fusion toward clinically-grounded
representations and is more robust when other modalities are missing.
"""

from __future__ import annotations

import torch
from torch import nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.norm_ff = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, query_embed: torch.Tensor,
                kv_embeds: list[torch.Tensor]) -> torch.Tensor:
        q = query_embed.unsqueeze(1)               # (B, 1, D)
        kv = torch.stack(kv_embeds, dim=1)         # (B, M, D)
        attn_out, _ = self.attn(q, kv, kv)
        h = self.norm(q + attn_out)
        h = self.norm_ff(h + self.ff(h))
        return h.squeeze(1)
