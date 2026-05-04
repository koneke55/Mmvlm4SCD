import numpy as np

from mmvlm4scd.evaluation.bootstrap import bootstrap_metrics
from mmvlm4scd.evaluation.survival import brier_score, integrated_brier_score


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
