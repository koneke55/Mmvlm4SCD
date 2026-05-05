"""Time-dependent survival evaluation: Brier + IBS at multiple horizons.

Reuses the trained best-model checkpoint produced by
``run_full_experiment.py`` (or trains one inline if missing) and reports
Brier(t) for t in {1,2,5,10,15,20} years plus the Integrated Brier Score
over [1,20]. Results are saved to
``experiments/results/survival_horizons/`` with a Brier(t) figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.dataloaders import make_loaders
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.evaluation import (brier_score, evaluate_model_full,
                                  integrated_brier_score, plot_brier_curve)
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.training import Trainer, TrainConfig
from mmvlm4scd.utils import auto_device, get_logger, save_json, set_seed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizons", type=str, default="1,2,5,10,15,20")
    p.add_argument("--device", default=auto_device())
    p.add_argument("--out", default="experiments/results/survival_horizons")
    args = p.parse_args(argv)
    log = get_logger()
    horizons = [float(t) for t in args.horizons.split(",")]

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=args.n,
                                                          seed=7))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(len(cohort["severity"]), seed=7)
    trL, vaL, teL = make_loaders(cohort, clin_x, tr, va, te,
                                 batch_size=args.batch_size)

    set_seed(0)
    model = MultimodalSCDModel(ModelConfig(
        clinical_input_dim=clin_x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
        embed_dim=64, fusion="attention", dropout=0.1,
    ))
    ckpt = ROOT / "experiments" / "checkpoints" / "best_model.pt"
    if ckpt.exists():
        log.info("Loading existing checkpoint %s", ckpt)
        try:
            model.load_state_dict(torch.load(ckpt, map_location=args.device))
        except Exception as exc:
            log.warning("Could not load checkpoint (%s); training from scratch", exc)
            ckpt = None
    if not ckpt or not ckpt.exists():
        Trainer(model, TrainConfig(epochs=args.epochs, lr=1e-3,
                                   weight_decay=1e-4, alpha=1.0, beta=0.5,
                                   device=args.device,
                                   early_stop_patience=8)).fit(trL, vaL)

    ev = evaluate_model_full(model, teL, device=args.device)
    bs = brier_score(ev["risk"], ev["time"], ev["event"], horizons)
    ibs = integrated_brier_score(ev["risk"], ev["time"], ev["event"], horizons)
    bs_vals = [bs[f"bs_{t:g}"] for t in horizons]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig_path = out / "brier_curve.png"
    plot_brier_curve(horizons, np.array(bs_vals), ibs, fig_path)
    save_json({"horizons": horizons, "brier": bs, "ibs": ibs,
               "n_test": int(len(te))}, out / "brier.json")
    print(json.dumps({"ibs": ibs, **bs}, indent=2))


if __name__ == "__main__":
    main()
