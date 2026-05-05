"""Smoke tests for visualization helpers: do they produce valid PNGs?"""

from pathlib import Path

import numpy as np

from mmvlm4scd.evaluation.visualization import (plot_brier_curve,
                                                plot_calibration_severity,
                                                plot_confusion,
                                                plot_decision_curve,
                                                plot_modality_importance,
                                                plot_per_class_roc,
                                                plot_robustness_curve,
                                                plot_training_curves)


def _png_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 100


def test_plot_training_curves(tmp_path: Path):
    history = [{"epoch": i, "train_loss": 1.0 - 0.1 * i,
                "accuracy": 0.5 + 0.05 * i, "auroc_ovr": 0.6 + 0.04 * i,
                "c_index": 0.55 + 0.03 * i, "f1_macro": 0.5}
               for i in range(5)]
    out = plot_training_curves(history, tmp_path / "curves.png")
    assert _png_ok(out)


def test_plot_confusion(tmp_path: Path):
    cm = np.array([[20, 3, 1], [4, 18, 2], [1, 2, 19]])
    out = plot_confusion(cm, ["mild", "moderate", "severe"], tmp_path / "cm.png")
    assert _png_ok(out)


def test_plot_modality_importance(tmp_path: Path):
    imp = {"clinical": 0.5, "genomic": 0.2, "imaging": 0.15, "temporal": 0.15}
    out = plot_modality_importance(imp, tmp_path / "mi.png")
    assert _png_ok(out)


def test_plot_calibration_severity(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 200
    probs = rng.dirichlet(np.ones(3), size=n)
    y = rng.integers(0, 3, size=n)
    out = plot_calibration_severity(probs, y, tmp_path / "cal.png")
    assert _png_ok(out)


def test_plot_brier_curve(tmp_path: Path):
    h = np.array([2.0, 5.0, 10.0, 20.0])
    bs = np.array([0.20, 0.18, 0.22, 0.30])
    out = plot_brier_curve(h, bs, ibs=0.22, out_path=tmp_path / "bs.png")
    assert _png_ok(out)


def test_plot_decision_curve(tmp_path: Path):
    t = np.linspace(0.05, 0.95, 19)
    nb_m = 0.2 - 0.2 * (t - 0.5) ** 2
    nb_a = 0.2 - 0.5 * t
    nb_n = np.zeros_like(t)
    out = plot_decision_curve(t, nb_m, nb_a, nb_n,
                              tmp_path / "dca.png", label="Mmvlm4SCD")
    assert _png_ok(out)


def test_plot_per_class_roc(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 3, size=n)
    p = rng.dirichlet(np.ones(3), size=n)
    for i, c in enumerate(y):
        p[i, c] += 0.5
        p[i] /= p[i].sum()
    out = plot_per_class_roc(p, y, ["mild", "moderate", "severe"],
                             tmp_path / "roc.png")
    assert _png_ok(out)


def test_plot_robustness_curve(tmp_path: Path):
    sweep = {0.0: {"auroc_ovr": 0.85, "c_index": 0.75},
             0.1: {"auroc_ovr": 0.83, "c_index": 0.74},
             0.25: {"auroc_ovr": 0.79, "c_index": 0.72},
             0.5: {"auroc_ovr": 0.71, "c_index": 0.68}}
    out = plot_robustness_curve(sweep, tmp_path / "robust.png")
    assert _png_ok(out)
