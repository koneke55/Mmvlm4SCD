"""Synthetic multimodal Sickle Cell Disease cohort generator.

The cohort is *not* a real patient population. It is a parametric simulator
whose marginal distributions are loosely calibrated to published SCD
literature (Hb levels, HbF %, white-cell counts, vaso-occlusive crisis
rates, age distributions, mortality hazards). Its purpose is reproducible
benchmarking of multimodal architectures when access to gated cohorts
(dbGaP, UK Biobank, MIMIC-IV) is not yet in place.

Outputs four aligned modalities for ``n`` patients:
    - clinical : pandas DataFrame with demographics + labs + comorbidities
    - genomic  : numpy array of HBB variant indicators + polygenic-like score
    - imaging  : numpy array simulating CNN embeddings of blood smears
    - temporal : numpy array of T x 6 vitals/labs trajectories

Plus two label sets:
    - severity : 0 mild, 1 moderate, 2 severe (ordinal)
    - survival : (time_years, event_observed) for time-to-death analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SCDSyntheticConfig:
    n_patients: int = 2000
    timesteps: int = 24             # ~24 monthly visits over 2 years
    image_embed_dim: int = 64
    genomic_dim: int = 32
    seed: int = 7
    censor_at_years: float = 25.0   # follow-up ceiling


_GENOTYPES = ["HbSS", "HbSC", "HbSbeta+", "HbSbeta0"]
_GENOTYPE_PROB = np.array([0.65, 0.22, 0.08, 0.05])


def _genotype_severity_offset(g: str) -> float:
    return {"HbSS": 1.0, "HbSC": -0.3, "HbSbeta+": -0.5, "HbSbeta0": 0.6}[g]


def _hazard_multiplier(g: str) -> float:
    # Relative mortality hazard across genotypes; HbSS highest, HbSbeta+ lowest.
    return {"HbSS": 1.0, "HbSC": 0.55, "HbSbeta+": 0.4, "HbSbeta0": 0.85}[g]


def generate_synthetic_cohort(
    cfg: SCDSyntheticConfig | None = None,
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Generate an aligned multimodal synthetic SCD cohort.

    Returns a dict with keys
        ``clinical``, ``genomic``, ``imaging``, ``temporal``,
        ``severity``, ``survival_time``, ``survival_event``.
    """
    cfg = cfg or SCDSyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_patients

    age = rng.gamma(shape=4.0, scale=6.0, size=n).clip(1, 70)
    sex = rng.integers(0, 2, size=n)  # 0=F, 1=M
    bmi = rng.normal(22.5, 3.5, size=n).clip(13, 45)
    hydroxyurea = (rng.uniform(size=n) < 0.55).astype(int)
    transfusion_history = rng.poisson(lam=1.2, size=n).clip(0, 12)

    genotype_idx = rng.choice(len(_GENOTYPES), size=n, p=_GENOTYPE_PROB)
    genotype = np.array(_GENOTYPES)[genotype_idx]
    geno_off = np.array([_genotype_severity_offset(g) for g in genotype])

    # Labs (mean +- sd loosely reflective of SCD literature).
    hb_g_dl = rng.normal(8.5 - 0.5 * geno_off, 1.3, size=n).clip(4, 14)
    hbf_pct = rng.normal(8 + 4 * hydroxyurea - 2 * (geno_off > 0), 4.5, size=n).clip(0.5, 35)
    wbc = rng.normal(11 + 0.6 * geno_off, 3.5, size=n).clip(3, 35)
    platelets = rng.normal(380 + 40 * geno_off, 110, size=n).clip(60, 900)
    ldh = rng.normal(420 + 90 * geno_off, 140, size=n).clip(150, 1500)
    bilirubin = rng.normal(2.1 + 0.6 * geno_off, 1.0, size=n).clip(0.2, 8)
    crp = np.exp(rng.normal(1.3 + 0.4 * geno_off, 0.6, size=n)).clip(0.1, 80)

    # Vaso-occlusive crisis (VOC) annual rate, increases severity.
    voc_rate = rng.poisson(lam=np.maximum(0.2, 1.6 + 0.9 * geno_off
                                          - 0.7 * hydroxyurea), size=n)
    acs_history = (rng.uniform(size=n) <
                   (0.10 + 0.07 * (geno_off > 0))).astype(int)
    stroke_history = (rng.uniform(size=n) <
                      (0.05 + 0.04 * (geno_off > 0))).astype(int)

    clinical = pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "genotype": genotype,
        "hydroxyurea": hydroxyurea,
        "transfusions_lifetime": transfusion_history,
        "hb_g_dl": hb_g_dl,
        "hbf_pct": hbf_pct,
        "wbc_k_ul": wbc,
        "platelets_k_ul": platelets,
        "ldh_u_l": ldh,
        "bilirubin_total_mg_dl": bilirubin,
        "crp_mg_l": crp,
        "voc_rate_per_year": voc_rate,
        "acs_history": acs_history,
        "stroke_history": stroke_history,
    })

    # Genomic: 16 HBB / modifier loci (binary) + 16-dim polygenic-like signal.
    variant_block = (rng.uniform(size=(n, 16)) < 0.12).astype(np.float32)
    # First 4 columns weakly correlated with genotype severity.
    variant_block[:, 0] = (geno_off > 0).astype(np.float32)
    pgs_block = rng.normal(geno_off[:, None], 1.0, size=(n, 16)).astype(np.float32)
    genomic = np.concatenate([variant_block, pgs_block], axis=1)
    assert genomic.shape[1] == cfg.genomic_dim

    # Imaging "embeddings": 64-d feature vector with a sickled-shape signal.
    sickled_signal = rng.normal(0.6 * geno_off + 0.2 * (voc_rate > 2), 0.8, size=n)
    base = rng.normal(0.0, 1.0, size=(n, cfg.image_embed_dim)).astype(np.float32)
    base[:, 0] += sickled_signal.astype(np.float32)
    base[:, 1] += (0.4 * np.log1p(ldh / 200)).astype(np.float32)
    imaging = base

    # Temporal: T x 6 vitals/labs trajectories (Hb, WBC, CRP, SpO2, HR, pain VAS).
    T = cfg.timesteps
    drift = rng.normal(geno_off[:, None] * 0.05, 0.05, size=(n, T))
    hb_t = (hb_g_dl[:, None] + drift +
            rng.normal(0, 0.5, size=(n, T))).astype(np.float32)
    wbc_t = (wbc[:, None] + 0.4 * drift * 5 +
             rng.normal(0, 1.5, size=(n, T))).astype(np.float32)
    crp_t = (np.log1p(crp[:, None]) + 0.5 * drift +
             rng.normal(0, 0.6, size=(n, T))).astype(np.float32)
    spo2_t = (98 - 1.5 * (geno_off[:, None] > 0) +
              rng.normal(0, 0.8, size=(n, T))).astype(np.float32)
    hr_t = (80 + 5 * geno_off[:, None] +
            rng.normal(0, 6, size=(n, T))).astype(np.float32)
    pain_t = (np.maximum(0, voc_rate[:, None] / 4 + geno_off[:, None]) +
              rng.normal(0, 1.0, size=(n, T))).clip(0, 10).astype(np.float32)
    temporal = np.stack([hb_t, wbc_t, crp_t, spo2_t, hr_t, pain_t], axis=-1)

    # Severity score (continuous) -> 3-class label by quantiles.
    severity_score = (
        0.40 * (10 - hb_g_dl) / 5
        + 0.35 * np.tanh((voc_rate - 2) / 2)
        + 0.20 * (geno_off + 0.3)
        + 0.15 * (ldh - 300) / 300
        + 0.10 * (acs_history + stroke_history)
        - 0.20 * (hbf_pct - 10) / 10
        - 0.10 * hydroxyurea
        + rng.normal(0, 0.25, size=n)
    )
    q1, q2 = np.quantile(severity_score, [0.40, 0.80])
    severity = np.where(severity_score < q1, 0,
                        np.where(severity_score < q2, 1, 2)).astype(np.int64)

    # Survival: Weibull with severity- and genotype-dependent hazard.
    haz_mult = np.array([_hazard_multiplier(g) for g in genotype])
    severity_mult = np.array([1.0, 1.6, 2.4])[severity]
    scale_years = 50.0 / (haz_mult * severity_mult * (1 + 0.05 * voc_rate))
    shape_k = 1.4
    u = rng.uniform(size=n)
    time_to_event = scale_years * (-np.log(1 - u)) ** (1 / shape_k)
    censor = rng.uniform(0.5, cfg.censor_at_years, size=n)
    survival_time = np.minimum(time_to_event, censor).astype(np.float32)
    survival_event = (time_to_event <= censor).astype(np.int64)

    return {
        "clinical": clinical,
        "genomic": genomic.astype(np.float32),
        "imaging": imaging.astype(np.float32),
        "temporal": temporal.astype(np.float32),
        "severity": severity,
        "survival_time": survival_time,
        "survival_event": survival_event,
    }


def split_indices(n: int, seed: int = 0,
                  fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15)
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    return (idx[:n_train],
            idx[n_train:n_train + n_val],
            idx[n_train + n_val:])
