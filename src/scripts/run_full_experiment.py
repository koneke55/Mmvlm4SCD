"""End-to-end experiment driver: data -> train -> evaluate -> figures.

Outputs (under ``experiments/results/<experiment_name>/``):
    - training_history.json
    - test_metrics.json
    - figures/training_curves.png
    - figures/confusion.png
    - figures/km_by_risk.png
    - figures/modality_importance.png
    - figures/calibration.png
    - per_modality_ablation.json (when --ablate is passed)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.evaluation import (evaluate_model_full,
                                  gradient_modality_importance,
                                  plot_calibration_severity, plot_confusion,
                                  plot_km_by_risk, plot_modality_importance,
                                  plot_training_curves)
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig
from mmvlm4scd.utils import auto_device, get_logger, load_config, save_json, set_seed


def _build_data(cfg):
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(
        n_patients=cfg["data"]["n_patients"],
        timesteps=cfg["data"]["timesteps"],
        seed=cfg["data"]["seed"],
    ))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(len(cohort["severity"]), seed=cfg["data"]["seed"])
    return cohort, clin_x, tr, va, te


def _build_model(cfg, clin_dim, gen_dim, img_dim, t_dim):
    return MultimodalSCDModel(ModelConfig(
        clinical_input_dim=clin_dim,
        genomic_input_dim=gen_dim,
        imaging_input_dim=img_dim,
        temporal_input_dim=t_dim,
        embed_dim=cfg["model"]["embed_dim"],
        fusion=cfg["model"]["fusion"],
        dropout=cfg["model"]["dropout"],
        num_severity_classes=cfg["model"].get("num_severity_classes", 3),
    ))


def _train_one(model, tr, va, cfg):
    return Trainer(model, TrainConfig(
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        grad_clip=cfg["train"].get("grad_clip", 1.0),
        alpha=cfg["train"].get("alpha", 1.0),
        beta=cfg["train"].get("beta", 0.5),
        early_stop_patience=cfg["train"].get("early_stop_patience", 8),
        select_metric=cfg["train"].get("select_metric", "auroc_ovr"),
        device=cfg["train"].get("device", auto_device()),
    )).fit(tr, va)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny run for CI: 200 patients, 3 epochs.")
    p.add_argument("--ablate", action="store_true",
                   help="Also train per-modality ablations.")
    p.add_argument("--seeds", type=int, default=1,
                   help="Number of seeds for the main run.")
    args = p.parse_args(argv)
    log = get_logger()

    cfg = load_config(args.config)
    if args.smoke:
        cfg["data"]["n_patients"] = 200
        cfg["train"]["epochs"] = 3
        cfg["train"]["batch_size"] = 32

    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["train"].get("seed", 0))
    cohort, clin_x, tr_idx, va_idx, te_idx = _build_data(cfg)
    log.info("Cohort: %d patients, clinical=%d, genomic=%d, imaging=%d, temporal=%d",
             len(cohort["severity"]), clin_x.shape[1],
             cohort["genomic"].shape[1], cohort["imaging"].shape[1],
             cohort["temporal"].shape[2])

    tr_loader, va_loader, te_loader = make_loaders(
        cohort, clin_x, tr_idx, va_idx, te_idx,
        batch_size=cfg["train"]["batch_size"])

    seed_runs = []
    histories = []
    test_metrics_runs = []
    test_eval = None
    importance = None
    last_model = None
    for s in range(args.seeds):
        set_seed(s + cfg["train"].get("seed", 0))
        model = _build_model(cfg, clin_x.shape[1], cohort["genomic"].shape[1],
                             cohort["imaging"].shape[1], cohort["temporal"].shape[2])
        log.info("[seed %d] params=%d", s, sum(q.numel() for q in model.parameters()))
        train_res = _train_one(model, tr_loader, va_loader, cfg)
        histories.append(train_res["history"])

        ev = evaluate_model_full(model, te_loader,
                                 device=cfg["train"].get("device", "cpu"))
        test_metrics_runs.append(ev["metrics"])
        seed_runs.append({"seed": s, "best_val": train_res["best_score"],
                          "test": ev["metrics"]})
        if s == 0:
            test_eval = ev
            try:
                importance = gradient_modality_importance(
                    model, te_loader, device=cfg["train"].get("device", "cpu"))
            except Exception as exc:  # pragma: no cover
                log.warning("Importance failed: %s", exc)
                importance = None
        last_model = model

    save_json(seed_runs, out_dir / "seed_runs.json")
    save_json(histories[0], out_dir / "training_history.json")
    save_json(test_metrics_runs[0], out_dir / "test_metrics.json")

    if test_eval is not None:
        plot_training_curves(histories[0], fig_dir / "training_curves.png")
        plot_confusion(np.array(test_eval["metrics"]["confusion_matrix"]),
                       ["mild", "moderate", "severe"],
                       fig_dir / "confusion.png")
        plot_km_by_risk(test_eval["risk"], test_eval["time"], test_eval["event"],
                        fig_dir / "km_by_risk.png")
        plot_calibration_severity(test_eval["severity_logits"],
                                  test_eval["severity_target"],
                                  fig_dir / "calibration.png")
        if importance is not None:
            plot_modality_importance(importance, fig_dir / "modality_importance.png")
            save_json(importance, out_dir / "modality_importance.json")

    if last_model is not None:
        ckpt_dir = Path("experiments/checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(last_model.state_dict(), ckpt_dir / "best_model.pt")

    if args.ablate:
        ablation = {}
        modalities = ["clinical", "genomic", "imaging", "temporal"]
        for drop in modalities:
            log.info("Ablating modality: %s", drop)
            set_seed(cfg["train"].get("seed", 0))
            model = _build_model(cfg, clin_x.shape[1], cohort["genomic"].shape[1],
                                 cohort["imaging"].shape[1], cohort["temporal"].shape[2])
            tr2, va2, te2 = make_loaders(cohort, clin_x, tr_idx, va_idx, te_idx,
                                         batch_size=cfg["train"]["batch_size"])

            class _DropLoader:
                def __init__(self, base, drop):
                    self.base = base; self.drop = drop
                def __iter__(self):
                    for b in self.base:
                        b2 = dict(b); b2[self.drop] = torch.zeros_like(b2[self.drop]); yield b2
                def __len__(self):
                    return len(self.base)

            _train_one(model, _DropLoader(tr2, drop), _DropLoader(va2, drop), cfg)
            ev2 = evaluate_model_full(model, _DropLoader(te2, drop),
                                      device=cfg["train"].get("device", "cpu"))
            ablation[drop] = ev2["metrics"]
        save_json(ablation, out_dir / "per_modality_ablation.json")

    summary = {
        "experiment": cfg["experiment"]["name"],
        "n_patients": len(cohort["severity"]),
        "seeds": args.seeds,
        "best_val_per_seed": [r["best_val"] for r in seed_runs],
        "test_per_seed": test_metrics_runs,
    }
    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
