"""Unit tests for open West African reference-frequency helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from mmvlm4scd.data.west_africa_open import (
    EnsemblFetchError,
    mean_rs334_maf_west_africa,
)


def test_mean_rs334_maf_west_africa_synthetic_table():
    df = pd.DataFrame(
        {
            "population": ["1000GENOMES:phase_3:YRI"] * 2,
            "allele": ["A", "T"],
            "frequency": [0.14, 0.86],
            "allele_count": [30, 186],
        }
    )
    assert abs(mean_rs334_maf_west_africa(df) - 0.14) < 1e-9


def test_mean_rs334_maf_requires_two_alleles_per_pop():
    bad = pd.DataFrame(
        {
            "population": ["1000GENOMES:phase_3:YRI"],
            "allele": ["A"],
            "frequency": [1.0],
            "allele_count": [216],
        }
    )
    with pytest.raises(EnsemblFetchError):
        mean_rs334_maf_west_africa(bad)
