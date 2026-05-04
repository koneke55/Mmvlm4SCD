"""Unit tests for evaluation.clinical (DCA, ECE/MCE, per-class AUROC,
   sensitivity/specificity)."""

import numpy as np

from mmvlm4scd.evaluation.clinical import (decision_curve,
                                           expected_calibration_error,
                                           per_class_auroc,
                                           sens_spec_at_thresholds)


def _balanced_binary(n=200, seed=0, gap=2.5):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    p = rng.normal(loc=y * gap, scale=1.0, size=n)
    p = 1 / (1 + np.exp(-p))
    return p, y


def test_decision_curve_treat_all_at_zero_threshold_equals_prevalence():
    p, y = _balanced_binary()
    res = decision_curve(p, y, thresholds=[0.0001, 0.5])
    assert np.isclose(res["net_benefit_all"][0], y.mean(), atol=2e-2)
    assert "net_benefit_model" in res


def test_decision_curve_arrays_have_matching_length():
    p, y = _balanced_binary()
    res = decision_curve(p, y, thresholds=np.linspace(0.05, 0.95, 19))
    n = len(res["thresholds"])
    assert len(res["net_benefit_model"]) == n
    assert len(res["net_benefit_all"]) == n
    assert len(res["net_benefit_none"]) == n


def test_ece_perfect_calibration():
    n = 500
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=n)
    onehot = np.eye(3)[y]
    ece, mce, _ = expected_calibration_error(onehot, y, n_bins=10)
    assert ece < 1e-6
    assert mce < 1e-6


def test_ece_calibration_error_is_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    n = 300
    probs = rng.dirichlet(np.ones(3), size=n)
    y = rng.integers(0, 3, size=n)
    ece, mce, bins = expected_calibration_error(probs, y, n_bins=8)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert len(bins) == 8


def test_per_class_auroc_returns_one_value_per_class():
    rng = np.random.default_rng(0)
    n = 200
    probs = rng.dirichlet(np.ones(3), size=n)
    y = rng.integers(0, 3, size=n)
    res = per_class_auroc(probs, y)
    assert {"auroc_class0", "auroc_class1", "auroc_class2"} <= set(res.keys())
    for v in res.values():
        assert np.isnan(v) or 0.0 <= v <= 1.0


def test_per_class_auroc_perfect_separation_is_one():
    n = 60
    y = np.array([0] * 20 + [1] * 20 + [2] * 20)
    probs = np.zeros((n, 3))
    for i, c in enumerate(y):
        probs[i, c] = 0.95
        probs[i, (c + 1) % 3] = 0.05
    res = per_class_auroc(probs, y)
    for v in res.values():
        assert v >= 0.99


def test_sens_spec_at_thresholds_consistent_counts():
    p, y = _balanced_binary()
    res = sens_spec_at_thresholds(p, y, thresholds=[0.3, 0.5, 0.7])
    for t, m in res.items():
        assert m["tp"] + m["fn"] + m["tn"] + m["fp"] == len(y)
        assert 0.0 <= m["sensitivity"] <= 1.0
        assert 0.0 <= m["specificity"] <= 1.0
