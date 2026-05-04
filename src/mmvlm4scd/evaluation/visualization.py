"""Plotting utilities used by the experiment driver and the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})


def plot_training_curves(history: List[Dict], out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title("Training loss")
    ax[1].plot(epochs, [h.get("auroc_ovr", np.nan) for h in history],
               label="val AUROC (OvR)")
    ax[1].plot(epochs, [h.get("c_index", np.nan) for h in history],
               label="val C-index")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("metric")
    ax[1].set_ylim(0.4, 1.0); ax[1].legend(); ax[1].set_title("Validation metrics")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_confusion(cm: np.ndarray, classes: List[str], out_path: Path) -> Path:
    out_path = Path(out_path)
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Severity confusion matrix")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_km_by_risk(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                    out_path: Path) -> Path:
    """Kaplan-Meier curves stratified by predicted-risk tertile."""
    from lifelines import KaplanMeierFitter

    out_path = Path(out_path)
    q1, q2 = np.quantile(risk, [1 / 3, 2 / 3])
    grp = np.where(risk < q1, "low", np.where(risk < q2, "mid", "high"))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for g, color in [("low", "#2c7fb8"), ("mid", "#7fbc41"), ("high", "#c0392b")]:
        m = grp == g
        if m.sum() == 0:
            continue
        kmf = KaplanMeierFitter().fit(time[m], event[m], label=f"{g} risk (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, color=color, ci_show=False)
    ax.set_xlabel("time (years)"); ax.set_ylabel("survival probability")
    ax.set_title("Kaplan-Meier by predicted risk tertile")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_modality_importance(importance: Dict[str, float], out_path: Path) -> Path:
    out_path = Path(out_path)
    keys = list(importance.keys())
    vals = [importance[k] for k in keys]
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.barh(keys, vals, color="#404a99")
    ax.set_xlabel("|grad| L2 (mean over batches)")
    ax.set_title("Modality importance")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path


def plot_calibration_severity(logits: np.ndarray, y: np.ndarray, out_path: Path) -> Path:
    """Reliability diagram for the predicted top-class probability."""
    out_path = Path(out_path)
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p = p / p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(conf, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.plot(xs, ys, "o-", color="#404a99")
    for x, y_, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, y_), fontsize=7,
                    textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("predicted confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Severity calibration")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)
    return out_path
