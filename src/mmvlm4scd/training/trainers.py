"""Minimal but correct multitask trainer.

Designed to run on CPU in a few minutes for a synthetic cohort while
leaving room for GPU scaling. Tracks per-epoch metrics on val and
returns the best-by-val checkpoint state dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn

from .losses import multitask_loss
from .metrics import compute_all_metrics


@dataclass
class TrainConfig:
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    alpha: float = 1.0  # CE weight
    beta: float = 0.5   # Cox weight
    early_stop_patience: int = 8
    select_metric: str = "auroc_ovr"  # higher is better
    log_every: int = 1
    device: str = "cpu"
    extra: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=cfg.epochs,
        )

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, Any]:
        self.model.eval()
        logits, sev, risk, t, e = [], [], [], [], []
        for batch in loader:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            out = self.model(batch)
            logits.append(out["severity_logits"].cpu().numpy())
            risk.append(out["risk_score"].cpu().numpy())
            sev.append(batch["severity"].cpu().numpy())
            t.append(batch["survival_time"].cpu().numpy())
            e.append(batch["survival_event"].cpu().numpy())
        return compute_all_metrics(
            np.concatenate(logits), np.concatenate(sev),
            np.concatenate(risk), np.concatenate(t), np.concatenate(e),
        )

    def fit(self, train_loader, val_loader) -> Dict[str, Any]:
        history: List[Dict[str, Any]] = []
        best_score = -np.inf
        best_state: Dict[str, torch.Tensor] | None = None
        bad_epochs = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            losses = []
            for batch in train_loader:
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
                out = self.model(batch)
                loss, comps = multitask_loss(
                    out["severity_logits"], batch["severity"],
                    out["risk_score"], batch["survival_time"], batch["survival_event"],
                    alpha=self.cfg.alpha, beta=self.cfg.beta,
                )
                self.optim.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optim.step()
                losses.append({k: float(v) for k, v in comps.items()})
            self.scheduler.step()

            train_loss = float(np.mean([d["total"] for d in losses]))
            val = self.evaluate(val_loader)
            entry = {"epoch": epoch, "train_loss": train_loss, **val}
            history.append(entry)

            score = val.get(self.cfg.select_metric, -np.inf)
            if isinstance(score, float) and not np.isnan(score) and score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if bad_epochs >= self.cfg.early_stop_patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"history": history, "best_score": best_score,
                "best_state": best_state}
