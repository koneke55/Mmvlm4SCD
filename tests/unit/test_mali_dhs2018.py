"""Tests for Mali DHS 2018 HR cohort loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmvlm4scd.data.mali_dhs2018 import (
    anemia_severity_who_child,
    build_cohort_from_mali_dhs2018_hr,
    genomic_block_mali_hb_growth,
)


def test_anemia_severity_who_child_thresholds():
    h = np.array([12.0, 10.0, 7.5, np.nan])
    s = anemia_severity_who_child(h)
    assert list(s[:3]) == [0, 1, 2]
    assert s[3] == 1


@pytest.mark.parametrize(
    "hb,s",
    [
        (11.0, 0),
        (8.5, 1),
        (7.5, 2),
    ],
)
def test_anemia_single(hb: float, s: int):
    assert int(anemia_severity_who_child(np.array([hb]))[0]) == s


def test_genomic_block_padding():
    hb = np.ones(4, dtype=np.float32) * 10.0
    z = np.zeros(4, dtype=np.float32)
    g = genomic_block_mali_hb_growth(hb, z, z, z, z)
    assert g.shape == (4, 32)


def test_build_cohort_from_minimal_mlhr_stata(tmp_path):
    df = pd.DataFrame(
        {
            "hc56": pd.Series([11.0, 9.5, 12.0], dtype=np.float64),
            "hc1": pd.Series([24, 36, 48], dtype=np.float64),
            "hc53": pd.Series([11.0, 9.5, 12.0], dtype=np.float64),
            "hc70": pd.Series([-0.5, 0.0, -1.0], dtype=np.float64),
            "hc71": pd.Series([0.2, -0.4, 1.0], dtype=np.float64),
            "hc72": pd.Series([0.1, 0.0, -0.8], dtype=np.float64),
            "hc73": pd.Series([0.0, -0.6, 2.0], dtype=np.float64),
            "hc2": pd.Series([10.0, 12.0, 14.0], dtype=np.float64),
            "hc3": pd.Series([80.0, 88.0, 95.0], dtype=np.float64),
            "hc27": pd.Series([1, 2, 1], dtype=np.int64),
        }
    )
    path = tmp_path / "MLHR7ADT.dta"
    df.to_stata(path, write_index=False, version=118)
    cohort = build_cohort_from_mali_dhs2018_hr(path)
    assert len(cohort["clinical"]) == 3
    assert cohort["genomic"].shape == (3, 32)
    assert set(np.unique(cohort["severity"]).tolist()) <= {0, 1, 2}
