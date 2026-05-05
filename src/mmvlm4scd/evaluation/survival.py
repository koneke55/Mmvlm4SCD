"""Time-dependent survival metrics.

We turn a continuous risk score into individual survival curves by
mapping the population's risk percentiles into a Kaplan-Meier-derived
baseline survival, then evaluate Brier and Integrated Brier scores at a
set of horizons.

The implementation is intentionally lightweight (no scikit-survival
dependency). It accepts numpy arrays and returns a JSON-friendly dict.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
from lifelines import KaplanMeierFitter


def _survival_curves_from_risk(risk: np.ndarray, time: np.ndarray,
                               event: np.ndarray,
                               eval_times: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Turn risk -> per-subject S(t) on `eval_times` and a global S0 baseline.

    We compute a population KM curve as S0(t), then shift each subject's
    curve in log-cumulative-hazard space using a normalised risk score
    so higher risk -> lower S(t).
    """
    kmf = KaplanMeierFitter().fit(time, event)
    S0 = np.array([float(kmf.predict(t)) for t in eval_times])
    S0 = np.clip(S0, 1e-6, 1 - 1e-6)
    H0 = -np.log(S0)
    # Normalise risk to mean 0 -> exp(r) ~ relative hazard around baseline.
    r = risk - risk.mean()
    # Cap to avoid blowups
    r = np.clip(r, -3.0, 3.0)
    relH = np.exp(r)[:, None]      # (N, 1)
    surv = np.exp(-relH * H0[None, :])  # (N, T)
    return surv, S0


def brier_score(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                eval_times: Iterable[float]) -> Dict[str, float]:
    """Inverse-probability-of-censoring-weighted Brier score per horizon.

    For uncensored events at time t* <= t we use the indicator 1; for
    subjects censored before t we exclude them; otherwise indicator 0.
    Weighting follows Graf et al. (1999) using a KM estimate of the
    censoring distribution.
    """
    eval_times = np.asarray(list(eval_times), dtype=float)
    n = time.shape[0]

    # KM of censoring (event flipped)
    kmc = KaplanMeierFitter().fit(time, 1 - event)

    surv, _ = _survival_curves_from_risk(risk, time, event, eval_times)

    bs: Dict[str, float] = {}
    for j, t in enumerate(eval_times):
        S_t = np.clip(surv[:, j], 1e-6, 1 - 1e-6)
        Gt_indiv = np.array([float(kmc.predict(min(ti, t))) for ti in time])
        Gt_indiv = np.clip(Gt_indiv, 1e-3, 1.0)
        bs_terms = []
        weights = []
        for i in range(n):
            if time[i] <= t and event[i] == 1:
                w = 1.0 / Gt_indiv[i]
                bs_terms.append(w * (0.0 - S_t[i]) ** 2)
                weights.append(w)
            elif time[i] > t:
                Gt_t = float(kmc.predict(t))
                Gt_t = max(Gt_t, 1e-3)
                w = 1.0 / Gt_t
                bs_terms.append(w * (1.0 - S_t[i]) ** 2)
                weights.append(w)
        if bs_terms:
            bs[f"bs_{t:g}"] = float(np.sum(bs_terms) / np.sum(weights))
        else:
            bs[f"bs_{t:g}"] = float("nan")
    return bs


def integrated_brier_score(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                           horizons: Iterable[float]) -> float:
    horizons = np.asarray(list(horizons), dtype=float)
    bs = brier_score(risk, time, event, horizons)
    vals = np.array([bs[f"bs_{t:g}"] for t in horizons], dtype=float)
    valid = ~np.isnan(vals)
    if valid.sum() < 2:
        return float("nan")
    integrate = getattr(np, "trapezoid", None) or np.trapz  # numpy 1.x/2.x
    return float(integrate(vals[valid], horizons[valid]) /
                 (horizons[valid].max() - horizons[valid].min()))
