"""Train each fusion strategy with multiple seeds and compare.

Outputs ``experiments/results/fusion_comparison/{summary,figures}/``:
    - ``per_fusion.json`` : per-fusion mean+/-std + bootstrap CIs on test set
    - ``fusion_bar.png``  : test AUROC and C-index by fusion strategy
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.evaluation import (bootstrap_metrics, evaluate_model_full,
                                  to_serialisable)
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig
from mmvlm4scd.utils import auto_device, get_logger, save_json, set_seed

FUSIONS = ("attention", "cross", "late")


def _train_eval(fusion: str, seed: int, cohort, clin_x, tr, va, te,
                epochs: int, batch_size: int, device: str):
    set_seed(seed)
    trL, vaL, teL = make_loaders(cohort, clin_x, tr, va, te,
                                 batch_size=batch_size)
    model = MultimodalSCDModel(ModelConfig(
        clinical_input_dim=clin_x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=64, fusion=fusion, dropout=0.1,
    ))
    Trainer(model, TrainConfig(epochs=epochs, lr=1e-3, weight_decay=1e-4,
                               alpha=1.0, beta=0.5, device=device,
                               early_stop_patience=8)).fit(trL, vaL)
    return evaluate_model_full(model, teL, device=device)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=auto_device())
    p.add_argument("--out", default="experiments/results/fusion_comparison")
    args = p.parse_args(argv)
    log = get_logger()

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=args.n,
                                                          seed=7))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(len(cohort["severity"]), seed=7)

    summary = {}
    for fusion in FUSIONS:
        log.info("=== fusion=%s ===", fusion)
        seed_metrics = []
        last_eval = None
        for s in range(args.seeds):
            ev = _train_eval(fusion, s, cohort, clin_x, tr, va, te,
                             args.epochs, args.batch_size, args.device)
            seed_metrics.append(ev["metrics"])
            last_eval = ev
            log.info("[%s seed=%d] auroc=%.3f c=%.3f",
                     fusion, s, ev["metrics"]["auroc_ovr"],
                     ev["metrics"]["c_index"])

        boot = bootstrap_metrics(last_eval["severity_logits"],
                                 last_eval["severity_target"],
                                 last_eval["risk"], last_eval["time"],
                                 last_eval["event"], n_boot=300, seed=0)
        summary[fusion] = {
            "seed_metrics": seed_metrics,
            "mean_auroc": float(np.mean([m["auroc_ovr"] for m in seed_metrics])),
            "std_auroc":  float(np.std([m["auroc_ovr"] for m in seed_metrics])),
            "mean_c_index": float(np.mean([m["c_index"] for m in seed_metrics])),
            "std_c_index":  float(np.std([m["c_index"] for m in seed_metrics])),
            "mean_accuracy": float(np.mean([m["accuracy"] for m in seed_metrics])),
            "mean_f1_macro": float(np.mean([m["f1_macro"] for m in seed_metrics])),
            "bootstrap": to_serialisable(boot),
        }

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_json(summary, out / "per_fusion.json")

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    aurocs = [summary[f]["mean_auroc"] for f in FUSIONS]
    a_err = [summary[f]["std_auroc"] for f in FUSIONS]
    cs = [summary[f]["mean_c_index"] for f in FUSIONS]
    c_err = [summary[f]["std_c_index"] for f in FUSIONS]
    x = np.arange(len(FUSIONS))
    ax[0].bar(x, aurocs, yerr=a_err, color="#1a3d63", capsize=4)
    ax[0].set_xticks(x); ax[0].set_xticklabels(FUSIONS)
    ax[0].set_ylim(0.5, 1.0)
    ax[0].set_ylabel("AUROC (OvR, macro)")
    ax[0].set_title("Severity")
    ax[1].bar(x, cs, yerr=c_err, color="#7f3d63", capsize=4)
    ax[1].set_xticks(x); ax[1].set_xticklabels(FUSIONS)
    ax[1].set_ylim(0.5, 0.85)
    ax[1].set_ylabel("Harrell C-index")
    ax[1].set_title("Survival")
    fig.suptitle("Test performance by fusion strategy")
    fig.tight_layout(); fig.savefig(out / "fusion_bar.png", dpi=200)
    plt.close(fig)

    print(json.dumps({f: {"mean_auroc": summary[f]["mean_auroc"],
                          "mean_c_index": summary[f]["mean_c_index"]}
                      for f in FUSIONS}, indent=2))


if __name__ == "__main__":
    main()
