"""Clinical-utility evaluation: decision-curve analysis, calibration,
   per-class ROC, sensitivity / specificity at operating points.

These complement the rank-based metrics (AUROC, C-index) with
threshold-sensitive numbers that map more directly to clinical
decisions: for a given decision threshold p, would using the model
yield more "net benefit" than treating everyone or no one?

References
----------
Vickers AJ, Elkin EB. *Decision curve analysis: a novel method for
evaluating prediction models.* Med Decis Making 26(6):565-574, 2006.
Naeini MP, Cooper GF, Hauskrecht M. *Obtaining well calibrated
probabilities using Bayesian binning.* AAAI 2015 (ECE).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


# --------------------------------------------------------------------- #
# Calibration: Expected and Maximum Calibration Error                   #
# --------------------------------------------------------------------- #

def expected_calibration_error(probs: np.ndarray, y: np.ndarray,
                               n_bins: int = 10) -> Tuple[float, float, list]:
    """Top-class ECE / MCE for multi-class outputs.

    Returns ``(ece, mce, bins)`` where ``bins`` is a list of dicts
    ``{conf, acc, count}`` for plotting.
    """
    probs = np.asarray(probs)
    y = np.asarray(y)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bins = []
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            bins.append({"conf": (lo + hi) / 2, "acc": float("nan"), "count": 0})
            continue
        c_mean = float(conf[mask].mean())
        a_mean = float(correct[mask].mean())
        gap = abs(c_mean - a_mean)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)
        bins.append({"conf": c_mean, "acc": a_mean, "count": int(mask.sum())})
    return float(ece), float(mce), bins


# --------------------------------------------------------------------- #
# Decision-curve analysis (binary "any-severe" framing)                 #
# --------------------------------------------------------------------- #

def decision_curve(probs_severe: np.ndarray, y_severe: np.ndarray,
                   thresholds: Iterable[float] | None = None
                   ) -> Dict[str, np.ndarray]:
    """Net-benefit curve over decision thresholds.

    Parameters
    ----------
    probs_severe : (n,) probability of the positive (severe) class.
    y_severe     : (n,) 0/1 ground truth.
    thresholds   : iterable of operating thresholds in (0,1).
                   Defaults to ``np.arange(0.05, 0.96, 0.05)``.
    """
    p = np.asarray(probs_severe).ravel()
    y = np.asarray(y_severe).ravel().astype(int)
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    thresholds = np.asarray(list(thresholds), dtype=float)
    n = len(y)
    prevalence = y.mean()

    nb_model = []
    nb_all = []
    nb_none = np.zeros_like(thresholds)
    for t in thresholds:
        if t >= 1.0:
            nb_model.append(0.0)
            nb_all.append(0.0)
            continue
        pred_pos = (p >= t)
        tp = int(((pred_pos == 1) & (y == 1)).sum())
        fp = int(((pred_pos == 1) & (y == 0)).sum())
        nb_m = tp / n - fp / n * (t / (1 - t))
        nb_a = prevalence - (1 - prevalence) * (t / (1 - t))
        nb_model.append(nb_m)
        nb_all.append(nb_a)
    return {
        "thresholds": thresholds,
        "net_benefit_model": np.asarray(nb_model),
        "net_benefit_all": np.asarray(nb_all),
        "net_benefit_none": nb_none,
        "prevalence": float(prevalence),
    }


# --------------------------------------------------------------------- #
# Per-class one-vs-rest AUROC (no sklearn version-skew dependency)       #
# --------------------------------------------------------------------- #

def _auroc_binary(score: np.ndarray, y: np.ndarray) -> float:
    score = np.asarray(score, dtype=float)
    y = np.asarray(y, dtype=int)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    s_sorted = score[order]
    i = 0
    n = len(score)
    rank = np.empty(n, dtype=float)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            rank[k] = avg
        i = j + 1
    ranks[order] = rank
    sum_pos_ranks = ranks[y == 1].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_class_auroc(probs: np.ndarray, y: np.ndarray,
                    classes: List[int] | None = None) -> Dict[str, float]:
    if classes is None:
        classes = list(range(probs.shape[1]))
    out: Dict[str, float] = {}
    for k in classes:
        out[f"auroc_class{k}"] = _auroc_binary(probs[:, k], (y == k).astype(int))
    return out


# --------------------------------------------------------------------- #
# Sensitivity / specificity at operating points                          #
# --------------------------------------------------------------------- #

def sens_spec_at_thresholds(probs_severe: np.ndarray, y_severe: np.ndarray,
                            thresholds: Iterable[float] = (0.3, 0.5, 0.7)
                            ) -> Dict[str, Dict[str, float]]:
    p = np.asarray(probs_severe).ravel()
    y = np.asarray(y_severe).ravel().astype(int)
    out: Dict[str, Dict[str, float]] = {}
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) else float("nan")
        npv = tn / (tn + fn) if (tn + fn) else float("nan")
        out[f"t={t:g}"] = {"sensitivity": sens, "specificity": spec,
                          "ppv": ppv, "npv": npv,
                          "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return out
