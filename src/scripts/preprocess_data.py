"""Materialise the synthetic cohort to disk as parquet/npz for downstream use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import generate_synthetic_cohort, SCDSyntheticConfig
from mmvlm4scd.utils import get_logger


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)
    log = get_logger()

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(
        n_patients=args.n, seed=args.seed))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cohort["clinical"].to_csv(out / "clinical.csv", index=False)
    np.savez_compressed(out / "modalities.npz",
                        genomic=cohort["genomic"],
                        imaging=cohort["imaging"],
                        temporal=cohort["temporal"],
                        severity=cohort["severity"],
                        survival_time=cohort["survival_time"],
                        survival_event=cohort["survival_event"])
    log.info("Wrote synthetic cohort to %s", out)


if __name__ == "__main__":
    main()
