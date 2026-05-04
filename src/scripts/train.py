"""Train the multimodal SCD model from a YAML config.

Usage:
    python src/scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without `pip install -e .`.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig
from mmvlm4scd.utils import auto_device, get_logger, load_config, save_json, set_seed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger()
    log.info("Loaded config %s", args.config)

    set_seed(cfg["train"].get("seed", 0))

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(
        n_patients=cfg["data"]["n_patients"],
        timesteps=cfg["data"]["timesteps"],
        seed=cfg["data"]["seed"],
    ))

    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_x = pre.transform(cohort["clinical"])

    train_idx, val_idx, test_idx = split_indices(
        len(cohort["severity"]), seed=cfg["data"]["seed"])
    tr, va, te = make_loaders(cohort, clin_x, train_idx, val_idx, test_idx,
                              batch_size=cfg["train"]["batch_size"])

    model_cfg = ModelConfig(
        clinical_input_dim=clin_x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=cfg["model"]["embed_dim"],
        fusion=cfg["model"]["fusion"],
        dropout=cfg["model"]["dropout"],
        num_severity_classes=cfg["model"].get("num_severity_classes", 3),
    )
    model = MultimodalSCDModel(model_cfg)
    log.info("Model parameter count: %d", sum(p.numel() for p in model.parameters()))

    train_cfg = TrainConfig(
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        grad_clip=cfg["train"].get("grad_clip", 1.0),
        alpha=cfg["train"].get("alpha", 1.0),
        beta=cfg["train"].get("beta", 0.5),
        early_stop_patience=cfg["train"].get("early_stop_patience", 8),
        select_metric=cfg["train"].get("select_metric", "auroc_ovr"),
        device=cfg["train"].get("device", auto_device()),
    )
    trainer = Trainer(model, train_cfg)
    res = trainer.fit(tr, va)

    out = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    save_json(res["history"], out / "training_history.json")
    log.info("Best validation %s: %.4f", train_cfg.select_metric, res["best_score"])
    return res


if __name__ == "__main__":
    main()
