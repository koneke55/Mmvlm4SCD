"""Self-attention fusion across modality embeddings.

Treats each modality embedding as a token and applies a small Transformer
encoder block, then pools with a learned [CLS]-style query.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4,
                 ff_dim: int = 128, dropout: float = 0.1, num_modalities: int = 4):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls, std=0.02)
        self.modality_pos = nn.Parameter(torch.zeros(1, num_modalities + 1, embed_dim))
        nn.init.normal_(self.modality_pos, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.embed_dim = embed_dim

    def forward(self, modality_embeds: list[torch.Tensor]) -> torch.Tensor:
        # modality_embeds: list of (B, D); stack to (B, M, D)
        x = torch.stack(modality_embeds, dim=1)
        b = x.shape[0]
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.modality_pos[:, : x.shape[1] + 1]
        out = self.encoder(x)
        return out[:, 0]  # CLS token
