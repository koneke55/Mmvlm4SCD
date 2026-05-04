"""Light-weight tensor augmentations used at training time.

These are intentionally simple: dropout-like masking on each modality
plus Gaussian jitter on numeric features. They make the multimodal
fusion robust to single-modality dropout, which is realistic for SCD
records where, e.g., genomic information is often missing.
"""

from __future__ import annotations

import torch


def random_modality_dropout(batch: dict, p: float = 0.10,
                            modalities=("clinical", "genomic", "imaging", "temporal")):
    """Zero out a whole modality with probability ``p`` per sample."""
    if not (0.0 <= p < 1.0):
        raise ValueError("p must be in [0, 1)")
    if p == 0.0:
        return batch
    out = dict(batch)
    n = batch["severity"].shape[0]
    for m in modalities:
        if m not in out:
            continue
        mask = (torch.rand(n) > p).float()
        x = out[m]
        view = mask.view([n] + [1] * (x.ndim - 1))
        out[m] = x * view
    return out


def numeric_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma
