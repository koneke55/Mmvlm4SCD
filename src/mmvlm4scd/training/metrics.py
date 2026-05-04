"""Metrics: classification (CE/F1/AUROC) + survival (Harrell C-index)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)


def severity_metrics(logits: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    pred = logits.argmax(axis=1)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
    }
    try:
        proba = _softmax(logits)
        out["auroc_ovr"] = float(
            roc_auc_score(y, proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        out["auroc_ovr"] = float("nan")
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    out["confusion_matrix"] = cm.tolist()
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def concordance_index_torch(risk: torch.Tensor | np.ndarray,
                            time: torch.Tensor | np.ndarray,
                            event: torch.Tensor | np.ndarray) -> float:
    """Harrell's C-index for a continuous risk score.

    A higher risk should correspond to a shorter time-to-event among
    comparable pairs (i_event, j) where t_i < t_j.
    """
    risk = _to_np(risk).reshape(-1)
    time = _to_np(time).reshape(-1)
    event = _to_np(event).reshape(-1).astype(bool)

    n = time.shape[0]
    num = denom = 0
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if i == j or time[j] < time[i]:
                continue
            if time[j] == time[i] and event[j]:
                continue
            denom += 1
            if risk[i] > risk[j]:
                num += 1
            elif risk[i] == risk[j]:
                num += 0.5
    return float(num / denom) if denom else float("nan")


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_all_metrics(severity_logits: np.ndarray,
                        severity_target: np.ndarray,
                        risk: np.ndarray,
                        time: np.ndarray,
                        event: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out.update(severity_metrics(severity_logits, severity_target))
    out["c_index"] = concordance_index_torch(risk, time, event)
    return out
