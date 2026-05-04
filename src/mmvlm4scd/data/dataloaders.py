"""Helpers to build train/val/test DataLoaders."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .multimodal_dataset import MultimodalSCDDataset


def make_loaders(
    cohort: dict,
    clinical_features: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def _build(idx):
        return MultimodalSCDDataset(
            clinical=clinical_features[idx],
            genomic=cohort["genomic"][idx],
            imaging=cohort["imaging"][idx],
            temporal=cohort["temporal"][idx],
            severity=cohort["severity"][idx],
            survival_time=cohort["survival_time"][idx],
            survival_event=cohort["survival_event"][idx],
        )

    g = torch.Generator().manual_seed(0)
    train = DataLoader(_build(train_idx), batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, generator=g, drop_last=True)
    val = DataLoader(_build(val_idx), batch_size=batch_size, shuffle=False,
                     num_workers=num_workers)
    test = DataLoader(_build(test_idx), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)
    return train, val, test
