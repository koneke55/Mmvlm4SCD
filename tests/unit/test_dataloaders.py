"""Tests for data.dataloaders.make_loaders."""

import numpy as np

from mmvlm4scd.data import StandardPreprocessor, generate_synthetic_cohort
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import SCDSyntheticConfig, split_indices


def _build_loaders(batch_size=8, n=128):
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=n, seed=0))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(n, seed=0)
    return cohort, x, tr, va, te, make_loaders(cohort, x, tr, va, te,
                                               batch_size=batch_size)


def test_make_loaders_shapes_and_keys():
    cohort, x, tr, va, te, (trL, vaL, teL) = _build_loaders()
    batch = next(iter(trL))
    expected_keys = {"clinical", "genomic", "imaging", "temporal",
                     "severity", "survival_time", "survival_event"}
    assert expected_keys <= set(batch.keys())
    assert batch["clinical"].shape[1] == x.shape[1]
    assert batch["genomic"].shape[1] == cohort["genomic"].shape[1]
    assert batch["imaging"].shape[1] == cohort["imaging"].shape[1]
    assert batch["temporal"].shape[2] == cohort["temporal"].shape[2]


def test_loader_dtypes():
    _, _, _, _, _, (trL, vaL, teL) = _build_loaders()
    batch = next(iter(trL))
    assert batch["severity"].dtype.is_floating_point is False
    assert batch["clinical"].dtype.is_floating_point is True
    assert batch["temporal"].dtype.is_floating_point is True


def test_train_loader_drops_last_partial_batch():
    """Training loader has drop_last=True for stable batch norm / metrics."""
    n = 33
    _, _, _, _, _, (trL, _, _) = _build_loaders(batch_size=8, n=n)
    sizes = [b["severity"].shape[0] for b in trL]
    assert all(s == 8 for s in sizes)


def test_val_test_loaders_iterate_full_set():
    n = 100
    cohort, x, tr, va, te, (_, vaL, teL) = _build_loaders(batch_size=16, n=n)
    seen_v = sum(b["severity"].shape[0] for b in vaL)
    seen_t = sum(b["severity"].shape[0] for b in teL)
    assert seen_v == len(va)
    assert seen_t == len(te)


def test_split_partition_is_disjoint():
    a, b, c = split_indices(500, seed=42)
    s_a, s_b, s_c = set(a.tolist()), set(b.tolist()), set(c.tolist())
    assert not (s_a & s_b)
    assert not (s_b & s_c)
    assert not (s_a & s_c)
    assert len(s_a | s_b | s_c) == 500
