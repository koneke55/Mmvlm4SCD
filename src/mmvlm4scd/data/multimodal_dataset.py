"""PyTorch Dataset wrapping the multimodal SCD cohort."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MultimodalSCDDataset(Dataset):
    clinical: np.ndarray   # (N, D_clin)
    genomic: np.ndarray    # (N, D_gen)
    imaging: np.ndarray    # (N, D_img)
    temporal: np.ndarray   # (N, T, D_t)
    severity: np.ndarray   # (N,) int64
    survival_time: np.ndarray   # (N,) float32
    survival_event: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return self.severity.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "clinical": torch.as_tensor(self.clinical[i], dtype=torch.float32),
            "genomic": torch.as_tensor(self.genomic[i], dtype=torch.float32),
            "imaging": torch.as_tensor(self.imaging[i], dtype=torch.float32),
            "temporal": torch.as_tensor(self.temporal[i], dtype=torch.float32),
            "severity": torch.as_tensor(self.severity[i], dtype=torch.long),
            "survival_time": torch.as_tensor(self.survival_time[i], dtype=torch.float32),
            "survival_event": torch.as_tensor(self.survival_event[i], dtype=torch.long),
        }
