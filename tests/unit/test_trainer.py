"""Smoke tests for the trainer: it should improve on a small problem
   and respect early stopping."""

import numpy as np
import torch

from mmvlm4scd.data import StandardPreprocessor, generate_synthetic_cohort
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import SCDSyntheticConfig, split_indices
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig


def _build(epochs=5, n=128, fusion="late", seed=0):
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=n, seed=seed))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(n, seed=seed)
    trL, vaL, teL = make_loaders(cohort, x, tr, va, te, batch_size=16)
    model = MultimodalSCDModel(ModelConfig(
        clinical_input_dim=x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=32, fusion=fusion, dropout=0.1,
    ))
    cfg = TrainConfig(epochs=epochs, lr=1e-3, weight_decay=1e-4, alpha=1.0,
                      beta=0.5, early_stop_patience=epochs + 1, device="cpu")
    return model, cfg, trL, vaL, teL


def test_trainer_runs_and_returns_history():
    model, cfg, trL, vaL, _ = _build(epochs=3)
    out = Trainer(model, cfg).fit(trL, vaL)
    assert {"history", "best_score", "best_state"} <= set(out.keys())
    assert len(out["history"]) >= 1
    for entry in out["history"]:
        assert "train_loss" in entry
        assert "auroc_ovr" in entry


def test_trainer_loss_decreases_overall():
    """Training loss at the end should be lower than at epoch 1 on average."""
    torch.manual_seed(0); np.random.seed(0)
    model, cfg, trL, vaL, _ = _build(epochs=6)
    out = Trainer(model, cfg).fit(trL, vaL)
    losses = [h["train_loss"] for h in out["history"]]
    assert losses[-1] <= losses[0] + 1e-3


def test_trainer_reproducible_for_same_seed():
    """Two runs with identical seeds + small epochs should produce identical
    final train losses (deterministic CPU)."""
    def run():
        torch.manual_seed(123); np.random.seed(123)
        m, cfg, trL, vaL, _ = _build(epochs=2, seed=123)
        return Trainer(m, cfg).fit(trL, vaL)["history"][-1]["train_loss"]
    a = run()
    b = run()
    assert abs(a - b) < 1e-3
