import numpy as np

from mmvlm4scd.data import (PUBLIC_SCD_DATASETS, StandardPreprocessor,
                            generate_synthetic_cohort, list_sources)
from mmvlm4scd.data.synthetic import SCDSyntheticConfig, split_indices


def test_registry_non_empty_and_filters():
    assert len(PUBLIC_SCD_DATASETS) >= 10
    open_sources = list_sources(access="open")
    assert all(s.access == "open" for s in open_sources)
    imaging = list_sources(modality="imaging")
    assert all(s.modality == "imaging" for s in imaging)


def test_synthetic_cohort_shapes():
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=300, seed=0))
    n = len(cohort["clinical"])
    assert n == 300
    assert cohort["genomic"].shape == (n, 32)
    assert cohort["imaging"].shape == (n, 64)
    assert cohort["temporal"].shape[0] == n and cohort["temporal"].shape[2] == 6
    assert cohort["severity"].shape == (n,)
    assert cohort["survival_time"].shape == (n,)
    assert cohort["survival_event"].shape == (n,)
    assert set(np.unique(cohort["severity"]).tolist()) <= {0, 1, 2}


def test_preprocessor_roundtrip():
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=128, seed=0))
    pre = StandardPreprocessor()
    x = pre.fit_transform(cohort["clinical"])
    assert x.dtype == np.float32
    assert x.shape[0] == 128
    assert x.shape[1] == pre.output_dim
    means = x.mean(axis=0)
    assert np.all(np.abs(means) < 1.0)


def test_split_indices_partition():
    a, b, c = split_indices(1000, seed=1)
    assert len(a) + len(b) + len(c) == 1000
    assert len(set(a.tolist()) & set(b.tolist())) == 0
    assert len(set(b.tolist()) & set(c.tolist())) == 0
