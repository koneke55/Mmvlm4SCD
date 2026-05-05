"""Clinical-utility, robustness and external-cohort evaluation driver.

Reads the test predictions saved by ``run_full_experiment.py`` (or
re-evaluates a checkpoint), then computes:

  * per-class one-vs-rest AUROC
  * decision-curve analysis (severe vs not)
  * sensitivity / specificity at p in {0.3, 0.5, 0.7}
  * expected and maximum calibration error (ECE / MCE)
  * test-time modality-dropout robustness sweep
  * external-cohort distribution-shift simulation
  * subgroup fairness gap (max-min on AUROC and C-index)

Outputs land in ``experiments/results/<exp>/clinical/`` as JSON +
figures; a ``clinical_summary.json`` aggregates the headline numbers
that the paper builder picks up automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.evaluation import (decision_curve, evaluate_model_full,
                                  expected_calibration_error,
                                  external_cohort_simulation, fairness_gap,
                                  modality_dropout_sweep, per_class_auroc,
                                  plot_decision_curve, plot_per_class_roc,
                                  plot_robustness_curve,
                                  sens_spec_at_thresholds)
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig
from mmvlm4scd.utils import auto_device, get_logger, load_config, save_json, set_seed


def _softmax(x):
    x = np.asarray(x)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _build_setup(cfg, smoke: bool):
    if smoke:
        cfg["data"]["n_patients"] = 400
        cfg["train"]["epochs"] = 6
        cfg["train"]["batch_size"] = 32
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(
        n_patients=cfg["data"]["n_patients"],
        timesteps=cfg["data"]["timesteps"],
        seed=cfg["data"]["seed"]))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(len(cohort["severity"]), seed=cfg["data"]["seed"])
    trL, vaL, teL = make_loaders(cohort, clin_x, tr, va, te,
                                 batch_size=cfg["train"]["batch_size"])
    model = MultimodalSCDModel(ModelConfig(
        clinical_input_dim=clin_x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=cfg["model"]["embed_dim"],
        fusion=cfg["model"]["fusion"],
        dropout=cfg["model"]["dropout"]))
    return cohort, pre, clin_x, tr, va, te, trL, vaL, teL, model


def _train(model, trL, vaL, cfg):
    Trainer(model, TrainConfig(
        epochs=cfg["train"]["epochs"], lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        alpha=cfg["train"].get("alpha", 1.0),
        beta=cfg["train"].get("beta", 0.5),
        early_stop_patience=cfg["train"].get("early_stop_patience", 8),
        device=cfg["train"].get("device", auto_device()))).fit(trL, vaL)


def _subgroup_predictions(model, cohort, clin_x, te_idx, device="cpu"):
    """Run inference on the test split and return arrays + subgroup masks."""
    from torch.utils.data import DataLoader
    from mmvlm4scd.data.multimodal_dataset import MultimodalSCDDataset

    ds = MultimodalSCDDataset(
        clinical=clin_x[te_idx], genomic=cohort["genomic"][te_idx],
        imaging=cohort["imaging"][te_idx], temporal=cohort["temporal"][te_idx],
        severity=cohort["severity"][te_idx],
        survival_time=cohort["survival_time"][te_idx],
        survival_event=cohort["survival_event"][te_idx])
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    ev = evaluate_model_full(model, loader, device=device)
    return ev


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--external-seeds", nargs="*", type=int, default=[101, 202, 303])
    args = p.parse_args(argv)
    log = get_logger()
    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 0))

    out_dir = Path(cfg["experiment"]["output_dir"]) / cfg["experiment"]["name"] / "clinical"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cohort, pre, clin_x, tr_idx, va_idx, te_idx, trL, vaL, teL, model = \
        _build_setup(cfg, args.smoke)

    log.info("[clinical] training on %d patients (%d epochs)",
             cfg["data"]["n_patients"], cfg["train"]["epochs"])
    _train(model, trL, vaL, cfg)
    ev = _subgroup_predictions(model, cohort, clin_x, te_idx,
                               device=cfg["train"].get("device", "cpu"))
    probs = _softmax(ev["severity_logits"])
    y = ev["severity_target"]

    # 1. per-class one-vs-rest AUROC
    pc_auroc = per_class_auroc(probs, y)
    plot_per_class_roc(probs, y, ["mild", "moderate", "severe"],
                       fig_dir / "per_class_roc.png")

    # 2. decision-curve analysis (severe = class 2)
    p_severe = probs[:, 2]
    y_severe = (y == 2).astype(int)
    dca = decision_curve(p_severe, y_severe)
    plot_decision_curve(dca["thresholds"], dca["net_benefit_model"],
                        dca["net_benefit_all"], dca["net_benefit_none"],
                        fig_dir / "decision_curve.png", label="Mmvlm4SCD")

    # 3. sensitivity/specificity at thresholds
    sens_spec = sens_spec_at_thresholds(p_severe, y_severe,
                                        thresholds=[0.3, 0.5, 0.7])

    # 4. ECE / MCE
    ece, mce, bins = expected_calibration_error(probs, y, n_bins=10)

    # 5. modality-dropout robustness sweep
    log.info("[clinical] robustness sweep ...")
    sweep = modality_dropout_sweep(model, teL,
                                   probs=(0.0, 0.1, 0.25, 0.5),
                                   n_repeats=3, seed=0)
    plot_robustness_curve(sweep, fig_dir / "robustness.png")

    # 6. fairness gap (genotype, age, sex)
    log.info("[clinical] subgroup AUROC / C-index ...")
    clin_df = cohort["clinical"].iloc[te_idx].reset_index(drop=True)
    subgroup_metrics = {}

    def _slice_ev(mask):
        if mask.sum() == 0:
            return None
        from mmvlm4scd.training.metrics import compute_all_metrics
        return compute_all_metrics(
            ev["severity_logits"][mask], ev["severity_target"][mask],
            ev["risk"][mask], ev["time"][mask], ev["event"][mask])

    for g in ["HbSS", "HbSC", "HbSbeta+", "HbSbeta0"]:
        m = _slice_ev((clin_df["genotype"] == g).to_numpy())
        if m is not None:
            subgroup_metrics[f"genotype:{g}"] = m
    for label, mask in [("age<18", (clin_df["age"] < 18).to_numpy()),
                        ("18<=age<40", ((clin_df["age"] >= 18) &
                                        (clin_df["age"] < 40)).to_numpy()),
                        ("age>=40", (clin_df["age"] >= 40).to_numpy())]:
        m = _slice_ev(mask)
        if m is not None:
            subgroup_metrics[f"age:{label}"] = m
    for label, mask in [("F", (clin_df["sex"] == "F").to_numpy()),
                        ("M", (clin_df["sex"] == "M").to_numpy())]:
        m = _slice_ev(mask)
        if m is not None:
            subgroup_metrics[f"sex:{label}"] = m
    fairness = {
        "auroc_ovr": fairness_gap(subgroup_metrics, metric="auroc_ovr"),
        "c_index":  fairness_gap(subgroup_metrics, metric="c_index"),
    }

    # 7. external-cohort simulation
    log.info("[clinical] external-cohort simulation seeds=%s", args.external_seeds)
    ext = external_cohort_simulation(
        train_seed=cfg["data"]["seed"], eval_seeds=list(args.external_seeds),
        n_eval=400 if args.smoke else 1000,
        epochs=4 if args.smoke else cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        device=cfg["train"].get("device", "cpu"))

    # ---- save artefacts ---------------------------------------------------
    save_json({k: float(v) for k, v in pc_auroc.items()},
              out_dir / "per_class_auroc.json")
    save_json({"thresholds": dca["thresholds"].tolist(),
               "net_benefit_model": dca["net_benefit_model"].tolist(),
               "net_benefit_all": dca["net_benefit_all"].tolist(),
               "net_benefit_none": dca["net_benefit_none"].tolist(),
               "prevalence": dca["prevalence"]},
              out_dir / "decision_curve.json")
    save_json(sens_spec, out_dir / "sens_spec.json")
    save_json({"ece": ece, "mce": mce, "bins": bins},
              out_dir / "calibration_error.json")
    save_json({str(k): v for k, v in sweep.items()},
              out_dir / "robustness.json")
    save_json(subgroup_metrics, out_dir / "subgroup_metrics.json")
    save_json(fairness, out_dir / "fairness_gap.json")
    save_json(ext, out_dir / "external_cohort.json")

    summary = {
        "per_class_auroc": {k: float(v) for k, v in pc_auroc.items()},
        "ece": ece, "mce": mce,
        "decision_curve_max_net_benefit": float(dca["net_benefit_model"].max()),
        "sens_spec": sens_spec,
        "robustness": {str(k): v for k, v in sweep.items()},
        "fairness_gap": fairness,
        "external_cohort": ext,
    }
    save_json(summary, out_dir / "clinical_summary.json")
    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
