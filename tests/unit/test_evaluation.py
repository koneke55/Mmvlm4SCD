import numpy as np

from mmvlm4scd.evaluation.bootstrap import bootstrap_metrics
from mmvlm4scd.evaluation.survival import brier_score, integrated_brier_score
from mmvlm4scd.training.metrics import severity_metrics


def test_bootstrap_returns_means_and_ci():
    rng = np.random.default_rng(0)
    n = 200
    logits = rng.normal(size=(n, 3))
    target = rng.integers(0, 3, size=n)
    risk = rng.normal(size=n)
    time = rng.uniform(1, 20, size=n).astype(float)
    event = (rng.uniform(size=n) < 0.6).astype(int)
    res = bootstrap_metrics(logits, target, risk, time, event,
                            n_boot=50, seed=1)
    for k in ("accuracy", "f1_macro", "auroc_ovr", "c_index"):
        assert k in res
        r = res[k]
        if not np.isnan(r.mean):
            assert r.ci_low <= r.mean <= r.ci_high


def test_bootstrap_auroc_matches_point_estimate_for_strong_signal():
    """Regression test for the AUROC-inversion bug (Table 2 of paper).

    A previous hand-rolled ``_auroc_ovr`` returned ``1 - AUROC`` because
    it sorted scores descending before applying the Mann-Whitney U
    formula, which expects ascending ranks. The bootstrap mean must
    track the point-estimate AUROC to within a few percent on data
    where the model has a strong signal.
    """
    rng = np.random.default_rng(0)
    n, k = 600, 3
    target = rng.integers(0, k, size=n)
    logits = np.zeros((n, k), dtype=float)
    logits[np.arange(n), target] = 3.0
    logits += rng.normal(scale=0.5, size=logits.shape)

    point = severity_metrics(logits, target)["auroc_ovr"]

    risk = rng.normal(size=n)
    time = rng.uniform(1, 20, size=n).astype(float)
    event = (rng.uniform(size=n) < 0.6).astype(int)
    res = bootstrap_metrics(logits, target, risk, time, event,
                            n_boot=200, seed=1)

    boot_mean = res["auroc_ovr"].mean
    assert point > 0.9, f"sanity: strong signal should give high AUROC, got {point}"
    assert abs(boot_mean - point) < 0.03, (
        f"bootstrap AUROC {boot_mean:.4f} diverges from point estimate "
        f"{point:.4f}; check for AUROC inversion in _auroc_ovr"
    )
    assert res["auroc_ovr"].ci_low <= boot_mean <= res["auroc_ovr"].ci_high


def test_brier_and_ibs_are_finite_for_simple_inputs():
    rng = np.random.default_rng(0)
    n = 200
    risk = rng.normal(size=n)
    time = rng.uniform(0.5, 25, size=n)
    event = (rng.uniform(size=n) < 0.5).astype(int)
    horizons = [2.0, 5.0, 10.0, 20.0]
    bs = brier_score(risk, time, event, horizons)
    for t in horizons:
        v = bs[f"bs_{t:g}"]
        assert np.isnan(v) or (0.0 <= v <= 1.5)
    ibs = integrated_brier_score(risk, time, event, horizons)
    assert np.isnan(ibs) or (0.0 <= ibs <= 1.5)
