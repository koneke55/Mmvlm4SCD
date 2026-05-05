"""Bootstrap confidence intervals for severity + survival metrics.

We resample test indices with replacement ``n_boot`` times and compute
each metric on the resample, then report mean and percentile-based 95%
CI. This is the standard way to put non-parametric uncertainty bands
around a single test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

from ..training.metrics import _softmax, concordance_index_torch


@dataclass
class BootstrapResult:
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    samples: list


def _f1_macro(y, p) -> float:
    classes = np.unique(np.concatenate([y, p]))
    f1s = []
    for c in classes:
        tp = float(((y == c) & (p == c)).sum())
        fp = float(((y != c) & (p == c)).sum())
        fn = float(((y == c) & (p != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def _auroc_ovr(y: np.ndarray, proba: np.ndarray) -> float:
    """Macro one-vs-rest AUROC.

    Delegates to ``sklearn.metrics.roc_auc_score`` so the bootstrap
    estimate is consistent with the point-estimate AUROC reported by
    ``training.metrics.severity_metrics``. A previous hand-rolled
    implementation sorted scores in descending order before applying
    the Mann-Whitney U formula, which silently returned ``1 - AUROC``.
    """
    classes_present = np.unique(y)
    if classes_present.size < 2:
        return float("nan")
    try:
        active = proba[:, classes_present]
        active = active / active.sum(axis=1, keepdims=True)
        return float(
            roc_auc_score(y, active, multi_class="ovr",
                          average="macro", labels=classes_present)
        )
    except ValueError:
        return float("nan")


def bootstrap_metrics(severity_logits: np.ndarray,
                      severity_target: np.ndarray,
                      risk: np.ndarray,
                      time: np.ndarray,
                      event: np.ndarray,
                      n_boot: int = 500,
                      seed: int = 0,
                      ci: float = 0.95) -> Dict[str, BootstrapResult]:
    rng = np.random.default_rng(seed)
    n = severity_target.shape[0]
    proba = _softmax(severity_logits)
    pred = severity_logits.argmax(axis=1)

    out: Dict[str, List[float]] = {
        "accuracy": [],
        "f1_macro": [],
        "auroc_ovr": [],
        "c_index": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        out["accuracy"].append(float((pred[idx] == severity_target[idx]).mean()))
        out["f1_macro"].append(_f1_macro(severity_target[idx], pred[idx]))
        out["auroc_ovr"].append(_auroc_ovr(severity_target[idx], proba[idx]))
        try:
            c = concordance_index_torch(risk[idx], time[idx], event[idx])
        except Exception:
            c = float("nan")
        out["c_index"].append(c)

    a = (1 - ci) / 2
    results: Dict[str, BootstrapResult] = {}
    for k, vals in out.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            results[k] = BootstrapResult(k, float("nan"), float("nan"),
                                         float("nan"), [])
            continue
        lo, hi = np.quantile(arr, [a, 1 - a])
        results[k] = BootstrapResult(k, float(arr.mean()),
                                     float(lo), float(hi), arr.tolist())
    return results


def to_serialisable(results: Dict[str, BootstrapResult]) -> dict:
    return {k: {"mean": v.mean, "ci_low": v.ci_low, "ci_high": v.ci_high}
            for k, v in results.items()}
