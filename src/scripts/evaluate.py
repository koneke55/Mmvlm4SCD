"""Evaluate a (newly trained) model and dump test metrics + arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.evaluation import evaluate_model_full
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig
from mmvlm4scd.utils import auto_device, get_logger, load_config, save_json, set_seed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    log = get_logger()
    set_seed(cfg["train"].get("seed", 0))

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(
        n_patients=cfg["data"]["n_patients"],
        timesteps=cfg["data"]["timesteps"],
        seed=cfg["data"]["seed"],
    ))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_x = pre.transform(cohort["clinical"])
    tr_idx, va_idx, te_idx = split_indices(len(cohort["severity"]),
                                           seed=cfg["data"]["seed"])
    tr, va, te = make_loaders(cohort, clin_x, tr_idx, va_idx, te_idx,
                              batch_size=cfg["train"]["batch_size"])

    mcfg = ModelConfig(
        clinical_input_dim=clin_x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=cfg["model"]["embed_dim"],
        fusion=cfg["model"]["fusion"],
        dropout=cfg["model"]["dropout"],
    )
    model = MultimodalSCDModel(mcfg)
    trainer = Trainer(model, TrainConfig(
        epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        device=cfg["train"].get("device", auto_device()),
    ))
    trainer.fit(tr, va)
    res = evaluate_model_full(model, te, device=trainer.cfg.device)

    out = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    save_json(res["metrics"], out / "test_metrics.json")
    log.info("Test metrics: %s", res["metrics"])
    return res


if __name__ == "__main__":
    main()
