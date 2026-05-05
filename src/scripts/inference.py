"""Per-patient inference helper.

Reads a small CSV of patient features and produces predicted severity
probabilities and a normalised survival risk score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mmvlm4scd.data import StandardPreprocessor, generate_synthetic_cohort, SCDSyntheticConfig
from mmvlm4scd.models import ModelConfig, MultimodalSCDModel
from mmvlm4scd.utils import get_logger


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clinical-csv", required=True)
    p.add_argument("--ckpt", required=False, default=None)
    args = p.parse_args(argv)
    get_logger()

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=512, seed=7))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    clin_in = pd.read_csv(args.clinical_csv)
    x = pre.transform(clin_in)

    mcfg = ModelConfig(
        clinical_input_dim=x.shape[1],
        genomic_input_dim=cohort["genomic"].shape[1],
        imaging_input_dim=cohort["imaging"].shape[1],
        temporal_input_dim=cohort["temporal"].shape[2],
    )
    model = MultimodalSCDModel(mcfg)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()

    n = x.shape[0]
    batch = {
        "clinical": torch.as_tensor(x, dtype=torch.float32),
        "genomic": torch.zeros(n, cohort["genomic"].shape[1], dtype=torch.float32),
        "imaging": torch.zeros(n, cohort["imaging"].shape[1], dtype=torch.float32),
        "temporal": torch.zeros(n, cohort["temporal"].shape[1],
                                cohort["temporal"].shape[2], dtype=torch.float32),
        "severity": torch.zeros(n, dtype=torch.long),
        "survival_time": torch.zeros(n, dtype=torch.float32),
        "survival_event": torch.zeros(n, dtype=torch.long),
    }
    with torch.no_grad():
        out = model(batch)
        probs = torch.softmax(out["severity_logits"], dim=1).numpy()
        risk = out["risk_score"].numpy()

    df = pd.DataFrame({
        "p_mild": probs[:, 0], "p_moderate": probs[:, 1], "p_severe": probs[:, 2],
        "risk": risk,
    })
    print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
