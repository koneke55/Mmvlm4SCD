"""Subgroup analysis: stratify test metrics by genotype, age band, sex.

Loads (or trains) the default attention-fusion model, runs it on the
test set, then slices predictions by clinical subgroups and reports
severity AUROC + survival C-index for each subgroup.

Outputs:
    experiments/results/subgroups/subgroups.json
    experiments/results/subgroups/figures/subgroup_bars.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.evaluation import bootstrap_metrics, to_serialisable
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig
from mmvlm4scd.training.metrics import compute_all_metrics
from mmvlm4scd.utils import auto_device, get_logger, save_json, set_seed


def _get_predictions(model, loader, device):
    model.eval().to(device)
    logits, sev, risk, t, e = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            logits.append(out["severity_logits"].cpu().numpy())
            risk.append(out["risk_score"].cpu().numpy())
            sev.append(batch["severity"].cpu().numpy())
            t.append(batch["survival_time"].cpu().numpy())
            e.append(batch["survival_event"].cpu().numpy())
    return (np.concatenate(logits), np.concatenate(sev),
            np.concatenate(risk), np.concatenate(t), np.concatenate(e))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=auto_device())
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="experiments/results/subgroups")
    args = p.parse_args(argv)
    log = get_logger()

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=args.n,
                                                          seed=7))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(len(cohort["severity"]), seed=7)
    trL, vaL, teL = make_loaders(cohort, clin_x, tr, va, te,
                                 batch_size=args.batch_size)

    set_seed(args.seed)
    model = MultimodalSCDModel(ModelConfig(
        clinical_input_dim=clin_x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=64, fusion="attention", dropout=0.1,
    ))
    Trainer(model, TrainConfig(epochs=args.epochs, lr=1e-3, weight_decay=1e-4,
                               alpha=1.0, beta=0.5, device=args.device,
                               early_stop_patience=8)).fit(trL, vaL)

    logits, sev, risk, t, e = _get_predictions(model, teL, args.device)
    clinical_test = cohort["clinical"].iloc[te].reset_index(drop=True)

    overall = compute_all_metrics(logits, sev, risk, t, e)
    boot = bootstrap_metrics(logits, sev, risk, t, e, n_boot=300)

    groups = {}
    # Genotype
    for g in clinical_test["genotype"].unique():
        m = (clinical_test["genotype"] == g).to_numpy()
        if m.sum() < 20:
            continue
        groups[f"genotype={g}"] = compute_all_metrics(
            logits[m], sev[m], risk[m], t[m], e[m])
    # Age bands
    age_bins = [(0, 12, "0-12"), (12, 25, "12-25"), (25, 45, "25-45"),
                (45, 100, "45+")]
    for lo, hi, label in age_bins:
        m = ((clinical_test["age"] >= lo) & (clinical_test["age"] < hi)).to_numpy()
        if m.sum() < 20:
            continue
        groups[f"age={label}"] = compute_all_metrics(
            logits[m], sev[m], risk[m], t[m], e[m])
    # Sex
    for sex_val, label in [(0, "F"), (1, "M")]:
        m = (clinical_test["sex"] == sex_val).to_numpy()
        if m.sum() < 20:
            continue
        groups[f"sex={label}"] = compute_all_metrics(
            logits[m], sev[m], risk[m], t[m], e[m])

    summary = {
        "overall": overall,
        "bootstrap_overall": to_serialisable(boot),
        "groups": groups,
        "n_test": int(len(te)),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary, out / "subgroups.json")

    keys = list(groups.keys())
    aurocs = [groups[k]["auroc_ovr"] for k in keys]
    cs = [groups[k]["c_index"] for k in keys]

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    y = np.arange(len(keys))
    ax[0].barh(y, aurocs, color="#1a3d63")
    ax[0].axvline(overall["auroc_ovr"], ls="--", color="#666",
                  label=f"overall = {overall['auroc_ovr']:.3f}")
    ax[0].set_yticks(y); ax[0].set_yticklabels(keys, fontsize=8)
    ax[0].set_xlim(0.5, 1.0); ax[0].set_xlabel("Severity AUROC (OvR)")
    ax[0].legend(fontsize=8)
    ax[1].barh(y, cs, color="#7f3d63")
    ax[1].axvline(overall["c_index"], ls="--", color="#666",
                  label=f"overall = {overall['c_index']:.3f}")
    ax[1].set_yticks(y); ax[1].set_yticklabels(keys, fontsize=8)
    ax[1].set_xlim(0.5, 0.85); ax[1].set_xlabel("Survival C-index")
    ax[1].legend(fontsize=8)
    fig.suptitle("Subgroup performance (synthetic cohort, attention fusion)")
    fig.tight_layout(); fig.savefig(fig_dir / "subgroup_bars.png", dpi=200)
    plt.close(fig)

    print(json.dumps({"overall": overall,
                      "n_test": int(len(te)),
                      "groups": {k: {"auroc": v["auroc_ovr"],
                                     "c_index": v["c_index"]}
                                 for k, v in groups.items()}}, indent=2))


if __name__ == "__main__":
    main()
