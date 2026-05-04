"""Aggregate test-set evaluation: severity + survival + raw arrays."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..training.metrics import compute_all_metrics


@torch.no_grad()
def evaluate_model_full(model, loader, device: str = "cpu") -> Dict[str, np.ndarray | float]:
    model.eval().to(device)
    logits, sev, risk, t, e = [], [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        logits.append(out["severity_logits"].cpu().numpy())
        risk.append(out["risk_score"].cpu().numpy())
        sev.append(batch["severity"].cpu().numpy())
        t.append(batch["survival_time"].cpu().numpy())
        e.append(batch["survival_event"].cpu().numpy())
    logits = np.concatenate(logits)
    sev = np.concatenate(sev)
    risk = np.concatenate(risk)
    t = np.concatenate(t)
    e = np.concatenate(e)
    metrics = compute_all_metrics(logits, sev, risk, t, e)
    return {
        "metrics": metrics,
        "severity_logits": logits,
        "severity_target": sev,
        "risk": risk,
        "time": t,
        "event": e,
    }
