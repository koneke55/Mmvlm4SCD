"""Tests for Nigeria NDHS 2018 HR → multimodal cohort builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmvlm4scd.data.nigeria_dhs2018 import (
    build_cohort_from_nigeria_dhs2018_hr,
    genomic_block_from_codes,
    severity_from_sb113,
)


def test_severity_from_sb113_aa_vs_ss():
    codes = np.array([1, 2, 4, 5])
    out = severity_from_sb113(codes)
    assert out[0] == 0
    assert out[1] == 1
    assert out[2] == 2
    assert out[3] == 2


def test_genomic_block_shape():
    g = genomic_block_from_codes(np.array([1, 5, 2]))
    assert g.shape == (3, 32)


def test_build_cohort_from_minimal_stata(tmp_path):
    df = pd.DataFrame(
        {
            "sb113b": pd.Series([1, 2, 5], dtype=np.int64),
            "hc1": pd.Series([24, 36, 48], dtype=np.float64),
            "hc53": pd.Series([11.0, 10.5, 8.0], dtype=np.float64),
            "hc2": pd.Series([10.0, 12.0, 11.0], dtype=np.float64),
            "hc3": pd.Series([80.0, 85.0, 82.0], dtype=np.float64),
            "hc27": pd.Series([1, 2, 1], dtype=np.int64),
        }
    )
    path = tmp_path / "NGHR7BDT.dta"
    df.to_stata(path, write_index=False, version=118)
    cohort = build_cohort_from_nigeria_dhs2018_hr(path)
    assert len(cohort["clinical"]) == 3
    assert cohort["genomic"].shape == (3, 32)
    assert cohort["temporal"].shape == (3, 24, 6)
    assert set(np.unique(cohort["severity"]).tolist()) <= {0, 1, 2}
