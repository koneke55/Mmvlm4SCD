"""Classical baselines on clinical features only.

Severity: multinomial logistic regression.
Survival: Cox proportional-hazards model (lifelines.CoxPHFitter).

Outputs go into ``experiments/results/baselines/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

from mmvlm4scd.data import (StandardPreprocessor, generate_synthetic_cohort,
                            SCDSyntheticConfig)
from mmvlm4scd.data.synthetic import split_indices
from mmvlm4scd.utils import save_json


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="experiments/results/baselines")
    args = p.parse_args(argv)

    cohort = generate_synthetic_cohort(SCDSyntheticConfig(
        n_patients=args.n, seed=args.seed))
    pre = StandardPreprocessor().fit(cohort["clinical"])
    x = pre.transform(cohort["clinical"])
    tr, va, te = split_indices(len(cohort["severity"]), seed=args.seed)

    # Severity: multinomial LR
    sev_model = LogisticRegression(max_iter=2000)
    sev_model.fit(x[tr], cohort["severity"][tr])
    sev_pred = sev_model.predict(x[te])
    sev_proba = sev_model.predict_proba(x[te])
    sev_metrics = {
        "accuracy": float(accuracy_score(cohort["severity"][te], sev_pred)),
        "f1_macro": float(f1_score(cohort["severity"][te], sev_pred, average="macro")),
        "auroc_ovr": float(roc_auc_score(cohort["severity"][te], sev_proba,
                                         multi_class="ovr", average="macro")),
        "confusion_matrix": confusion_matrix(cohort["severity"][te], sev_pred,
                                             labels=[0, 1, 2]).tolist(),
    }

    # Survival: Cox PH on standardised clinical numeric features only.
    feats = [c for c in cohort["clinical"].select_dtypes(include=[np.number]).columns]
    df = cohort["clinical"][feats].copy()
    df["time"] = cohort["survival_time"]
    df["event"] = cohort["survival_event"]
    sc = StandardScaler().fit(df[feats].iloc[tr])
    df_std = df.copy()
    df_std[feats] = sc.transform(df[feats])
    cph = CoxPHFitter(penalizer=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df_std.iloc[tr], duration_col="time", event_col="event")
    risk_test = cph.predict_partial_hazard(df_std.iloc[te]).to_numpy()
    c_idx = float(concordance_index(df["time"].iloc[te], -risk_test, df["event"].iloc[te]))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    save_json(sev_metrics, out / "logreg_severity.json")
    save_json({"c_index": c_idx,
               "n_train": int(len(tr)), "n_test": int(len(te))},
              out / "cox_survival.json")
    print(json.dumps({"severity": sev_metrics, "cox_c_index": c_idx}, indent=2))


if __name__ == "__main__":
    main()
