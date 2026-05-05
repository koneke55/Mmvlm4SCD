"""Robustness, fairness and external-cohort simulation evaluations.

* ``modality_dropout_sweep``: zero out each modality with probability p
  at test time and report degradation. Models the missing-at-random
  setting that real EHR data exhibits.
* ``fairness_gap``: compute the max-min gap in AUROC and C-index across
  subgroups -- a single scalar summary of equity.
* ``external_cohort_simulation``: draw a second synthetic cohort with a
  different seed and report performance, simulating distribution shift.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch

from ..training.metrics import compute_all_metrics


@torch.no_grad()
def modality_dropout_sweep(model, loader, device: str = "cpu",
                           probs: Iterable[float] = (0.0, 0.1, 0.25, 0.5),
                           n_repeats: int = 3, seed: int = 0
                           ) -> Dict[float, Dict[str, float]]:
    """Test-time dropout sweep over the four modalities."""
    rng = np.random.default_rng(seed)
    model.eval().to(device)
    out: Dict[float, Dict[str, float]] = {}
    modalities = ["clinical", "genomic", "imaging", "temporal"]
    for p in probs:
        agg = {"accuracy": [], "f1_macro": [], "auroc_ovr": [], "c_index": []}
        for r in range(n_repeats):
            logits, sev, risk, t, e = [], [], [], [], []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                if p > 0:
                    n = batch["severity"].shape[0]
                    for m in modalities:
                        x = batch[m]
                        mask = torch.as_tensor(rng.uniform(size=n) > p,
                                               dtype=x.dtype, device=device)
                        view = mask.view([n] + [1] * (x.ndim - 1))
                        batch[m] = x * view
                o = model(batch)
                logits.append(o["severity_logits"].cpu().numpy())
                risk.append(o["risk_score"].cpu().numpy())
                sev.append(batch["severity"].cpu().numpy())
                t.append(batch["survival_time"].cpu().numpy())
                e.append(batch["survival_event"].cpu().numpy())
            metrics = compute_all_metrics(np.concatenate(logits),
                                          np.concatenate(sev),
                                          np.concatenate(risk),
                                          np.concatenate(t),
                                          np.concatenate(e))
            for k in agg:
                agg[k].append(metrics[k])
        out[float(p)] = {k: float(np.nanmean(v)) for k, v in agg.items()}
    return out


def fairness_gap(group_metrics: Dict[str, Dict[str, float]],
                 metric: str = "auroc_ovr") -> Dict[str, float]:
    """Max-min subgroup gap on a chosen metric.

    Returns ``{best, worst, gap, n_groups}``.
    """
    vals = [m[metric] for m in group_metrics.values()
            if metric in m and not np.isnan(m[metric])]
    if not vals:
        return {"best": float("nan"), "worst": float("nan"),
                "gap": float("nan"), "n_groups": 0}
    return {"best": float(max(vals)), "worst": float(min(vals)),
            "gap": float(max(vals) - min(vals)), "n_groups": len(vals)}


def external_cohort_simulation(train_seed: int = 7, eval_seeds: List[int] = None,
                               n_eval: int = 1000, epochs: int = 20,
                               batch_size: int = 64, device: str = "cpu"
                               ) -> Dict[str, Dict[str, float]]:
    """Train on cohort drawn with ``train_seed``, test on cohorts drawn with
    different seeds (treated as 'external' distribution shifts).

    Returns ``{"seed_<n>": metrics, ...}``.
    """
    if eval_seeds is None:
        eval_seeds = [101, 202, 303]

    # Local imports to avoid circular dependency at module load time.
    from ..data import (StandardPreprocessor, generate_synthetic_cohort,
                        SCDSyntheticConfig)
    from ..data.dataloaders import make_loaders
    from ..data.synthetic import split_indices
    from ..models import ModelConfig, MultimodalSCDModel
    from ..training import Trainer, TrainConfig
    from .evaluators import evaluate_model_full

    train_cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=2000,
                                                                seed=train_seed))
    pre = StandardPreprocessor().fit(train_cohort["clinical"])
    clin_x = pre.transform(train_cohort["clinical"])
    tr, va, _ = split_indices(len(train_cohort["severity"]), seed=train_seed)
    trL, vaL, _ = make_loaders(train_cohort, clin_x, tr, va, np.array([0]),
                               batch_size=batch_size)
    model = MultimodalSCDModel(ModelConfig(
        clinical_input_dim=clin_x.shape[1],
        genomic_input_dim=train_cohort["genomic"].shape[1],
        imaging_input_dim=train_cohort["imaging"].shape[1],
        temporal_input_dim=train_cohort["temporal"].shape[2],
        embed_dim=64, fusion="attention", dropout=0.1))
    Trainer(model, TrainConfig(epochs=epochs, lr=1e-3, weight_decay=1e-4,
                               alpha=1.0, beta=0.5, device=device,
                               early_stop_patience=8)).fit(trL, vaL)

    out: Dict[str, Dict[str, float]] = {}
    for s in eval_seeds:
        ext = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=n_eval, seed=s))
        clin_x_ext = pre.transform(ext["clinical"])
        full_idx = np.arange(n_eval)
        _, _, teL = make_loaders(ext, clin_x_ext,
                                 full_idx[:1], full_idx[1:2], full_idx,
                                 batch_size=batch_size)
        ev = evaluate_model_full(model, teL, device=device)
        m = {k: v for k, v in ev["metrics"].items()
             if k != "confusion_matrix"}
        out[f"seed_{s}"] = m
    return out
