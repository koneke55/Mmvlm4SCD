"""Preprocessing utilities for tabular clinical features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


CLINICAL_NUMERIC: List[str] = [
    "age", "bmi", "transfusions_lifetime", "hb_g_dl", "hbf_pct",
    "wbc_k_ul", "platelets_k_ul", "ldh_u_l", "bilirubin_total_mg_dl",
    "crp_mg_l", "voc_rate_per_year",
]
CLINICAL_BINARY: List[str] = ["sex", "hydroxyurea", "acs_history", "stroke_history"]
CLINICAL_CATEGORICAL: List[str] = ["genotype"]


@dataclass
class StandardPreprocessor:
    numeric_cols: List[str] = field(default_factory=lambda: list(CLINICAL_NUMERIC))
    binary_cols: List[str] = field(default_factory=lambda: list(CLINICAL_BINARY))
    categorical_cols: List[str] = field(default_factory=lambda: list(CLINICAL_CATEGORICAL))
    means_: np.ndarray | None = None
    stds_: np.ndarray | None = None
    categories_: dict | None = None

    def fit(self, df: pd.DataFrame) -> "StandardPreprocessor":
        self.means_ = df[self.numeric_cols].to_numpy(dtype=np.float32).mean(axis=0)
        self.stds_ = df[self.numeric_cols].to_numpy(dtype=np.float32).std(axis=0) + 1e-6
        self.categories_ = {c: sorted(df[c].unique().tolist()) for c in self.categorical_cols}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.categories_ is None:
            raise RuntimeError("StandardPreprocessor must be fit before transform.")
        num = (df[self.numeric_cols].to_numpy(dtype=np.float32) - self.means_) / self.stds_
        bin_ = df[self.binary_cols].to_numpy(dtype=np.float32)
        cat_blocks = []
        for c in self.categorical_cols:
            cats = self.categories_[c]
            onehot = np.zeros((len(df), len(cats)), dtype=np.float32)
            for j, k in enumerate(cats):
                onehot[:, j] = (df[c].to_numpy() == k).astype(np.float32)
            cat_blocks.append(onehot)
        return np.concatenate([num, bin_] + cat_blocks, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def output_dim(self) -> int:
        if self.categories_ is None:
            raise RuntimeError("Call fit() first.")
        return (len(self.numeric_cols) + len(self.binary_cols)
                + sum(len(v) for v in self.categories_.values()))
