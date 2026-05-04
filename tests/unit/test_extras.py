"""Unit tests for evaluation.extras (modality dropout, fairness gap)."""

import numpy as np
import torch

from mmvlm4scd.data import StandardPreprocessor, generate_synthetic_cohort
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import SCDSyntheticConfig, split_indices
from mmvlm4scd.evaluation.extras import fairness_gap, modality_dropout_sweep
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel


def _tiny_setup(n=128, fusion="late"):
    torch.manual_seed(0); np.random.seed(0)
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=n, seed=0))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(n, seed=0)
    _, _, teL = make_loaders(cohort, x, tr, va, te, batch_size=32)
    model = MultimodalSCDModel(ModelConfig(
        clinical_input_dim=x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=32, fusion=fusion, dropout=0.0))
    return model, teL


def test_modality_dropout_sweep_returns_metrics_for_each_p():
    model, teL = _tiny_setup()
    res = modality_dropout_sweep(model, teL, probs=(0.0, 0.25), n_repeats=1, seed=0)
    assert set(res.keys()) == {0.0, 0.25}
    for p, m in res.items():
        assert {"accuracy", "f1_macro", "auroc_ovr", "c_index"} <= set(m.keys())


def test_modality_dropout_increases_or_preserves_loss():
    """Random untrained model: at least the metrics should remain numeric
    and within [0,1]/finite ranges across dropout levels."""
    model, teL = _tiny_setup()
    res = modality_dropout_sweep(model, teL, probs=(0.0, 0.5), n_repeats=1, seed=1)
    for m in res.values():
        for k in ("accuracy", "f1_macro", "auroc_ovr"):
            assert np.isnan(m[k]) or 0.0 <= m[k] <= 1.0
        assert np.isnan(m["c_index"]) or 0.0 <= m["c_index"] <= 1.0


def test_fairness_gap_basic():
    groups = {"A": {"auroc_ovr": 0.80, "c_index": 0.70},
              "B": {"auroc_ovr": 0.65, "c_index": 0.71},
              "C": {"auroc_ovr": 0.78, "c_index": 0.69}}
    out = fairness_gap(groups, metric="auroc_ovr")
    assert out["best"] == 0.80
    assert out["worst"] == 0.65
    assert abs(out["gap"] - 0.15) < 1e-9
    assert out["n_groups"] == 3


def test_fairness_gap_handles_nans():
    groups = {"A": {"auroc_ovr": float("nan")},
              "B": {"auroc_ovr": 0.7}}
    out = fairness_gap(groups, metric="auroc_ovr")
    assert out["best"] == 0.7
    assert out["worst"] == 0.7
    assert out["gap"] == 0.0
    assert out["n_groups"] == 1
