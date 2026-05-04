"""Top-level multimodal model with two prediction heads.

* Severity head     -> ordinal 3-class classifier (mild/moderate/severe)
* Survival head     -> continuous risk score for Cox partial-likelihood loss

The fusion strategy is selectable: ``attention``, ``cross``, or ``late``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from .encoders import (ClinicalEncoder, GenomicEncoder,
                       ImagingEncoder, TemporalEncoder)
from .fusion import AttentionFusion, CrossAttentionFusion, LateFusion


@dataclass
class ModelConfig:
    clinical_input_dim: int = 19      # set by preprocessor at runtime
    genomic_input_dim: int = 32
    imaging_input_dim: int = 64
    temporal_input_dim: int = 6
    embed_dim: int = 64
    num_severity_classes: int = 3
    fusion: Literal["attention", "cross", "late"] = "attention"
    dropout: float = 0.1
    head_hidden_dim: int = 64
    extra: dict = field(default_factory=dict)


class MultimodalSCDModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim

        self.clinical_enc = ClinicalEncoder(cfg.clinical_input_dim, embed_dim=d,
                                            dropout=cfg.dropout)
        self.genomic_enc = GenomicEncoder(cfg.genomic_input_dim, embed_dim=d,
                                          dropout=cfg.dropout)
        self.imaging_enc = ImagingEncoder(cfg.imaging_input_dim, embed_dim=d,
                                          dropout=cfg.dropout)
        self.temporal_enc = TemporalEncoder(cfg.temporal_input_dim, embed_dim=d,
                                            dropout=cfg.dropout)

        if cfg.fusion == "attention":
            self.fusion = AttentionFusion(embed_dim=d, dropout=cfg.dropout,
                                          num_modalities=4)
        elif cfg.fusion == "cross":
            self.fusion = CrossAttentionFusion(embed_dim=d, dropout=cfg.dropout)
        elif cfg.fusion == "late":
            self.fusion = LateFusion(num_modalities=4, embed_dim=d,
                                     dropout=cfg.dropout)
        else:
            raise ValueError(f"Unknown fusion: {cfg.fusion}")

        self.severity_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, cfg.num_severity_classes),
        )
        self.survival_head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden_dim), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden_dim, 1),
        )

    def encode(self, batch: dict) -> torch.Tensor:
        ec = self.clinical_enc(batch["clinical"])
        eg = self.genomic_enc(batch["genomic"])
        ei = self.imaging_enc(batch["imaging"])
        et = self.temporal_enc(batch["temporal"])
        if isinstance(self.fusion, CrossAttentionFusion):
            return self.fusion(ec, [eg, ei, et])
        return self.fusion([ec, eg, ei, et])

    def forward(self, batch: dict) -> dict:
        z = self.encode(batch)
        return {
            "embedding": z,
            "severity_logits": self.severity_head(z),
            "risk_score": self.survival_head(z).squeeze(-1),
        }
