"""Nigeria NDHS 2018 — Household Recode (NGHR*) child biomarkers + sickle genotype.

This uses **released microdata** from the Nigeria Standard DHS 2018 Household
Recode Stata dataset (typically ``NGHR7BDT.dta`` / ``NGHR7BDT.zip`` from
`<https://dhsprogram.com/>` — free registration required).

Relevant questionnaire variables (same naming as exported Stata):

    * ``sb113b`` — Result of genotype RDT: 1=AA, 2=AS, 3=AC, 4=SC, 5=SS, 6=Other.
    * ``hc1``  — child's age (months).
    * ``hc2``  — weight (kg).
    * ``hc3``  — height (cm).
    * ``hc53`` — hemoglobin (g/dl).
    * ``hc27`` — child's sex where present (common DHS coding: 1=male, 2=female).

Imaging / longitudinal vitals are not collected in NDHS → filled with zeros
(benchmark honesty). Survival is censored placeholders and Cox loss weight
should be set to ``beta=0`` in training when using this cohort.

Ref: Nigeria DHS 2018; Pullum TW. WP175 — sickle genotypes in NDHS 2018.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dhs_resolve import find_sb113_genotype_column, lower_columns_map, resolve_first_column

# Fallback column names (lowercased keys -> original names resolved at runtime).
_COL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "age_months": ("hc1",),
    "weight_kg": ("hc2",),
    "height_cm": ("hc3",),
    "hb_g_dl": ("hc53",),
    "child_sex": ("hc27", "hc30"),
}

SB113_LABELS = {1: "AA", 2: "AS", 3: "AC", 4: "SC", 5: "SS", 6: "OTHER"}


def severity_from_sb113(code: np.ndarray | pd.Series) -> np.ndarray:
    """3-class ordinal label for multimodal severity head."""
    x = np.asarray(code, dtype=np.float64).astype(np.int64)
    mild = np.zeros_like(x)
    moderate = np.ones_like(x) * 1
    severe = np.ones_like(x) * 2
    out = np.where(x == 1, mild,
                   np.where(np.isin(x, (2, 3)), moderate, severe))
    return out.astype(np.int64)


def genomic_block_from_codes(codes: np.ndarray, dim: int = 32) -> np.ndarray:
    """Fixed-size genomic vector from DHS genotype code (deterministic).

    Encodes approximate S-/C-variant dosage + one-hot-like expansion.
    """
    n = len(codes)
    c = np.asarray(codes, dtype=np.int64)
    s_dose = np.zeros(n, dtype=np.float32)
    s_dose = np.where(c == 2, 1.0, s_dose)  # AS
    s_dose = np.where(c == 4, 1.0, s_dose)  # SC
    s_dose = np.where(c == 5, 2.0, s_dose)  # SS
    c_dose = np.zeros(n, dtype=np.float32)
    c_dose = np.where(c == 3, 1.0, c_dose)  # AC
    c_dose = np.where(c == 4, 1.0, c_dose)  # SC
    onehot = np.zeros((n, 7), dtype=np.float32)
    for j in range(1, 7):
        onehot[:, j] = (c == j).astype(np.float32)
    base = np.column_stack(
        [s_dose / 2.0, c_dose / 2.0, onehot],
    ).astype(np.float32)
    if base.shape[1] >= dim:
        return base[:, :dim]
    pad = np.zeros((n, dim - base.shape[1]), dtype=np.float32)
    return np.concatenate([base, pad], axis=1)


def _default_clinical_numeric() -> dict[str, float]:
    # Mild pediatric priors where NDHS lacks SCD-specific labs (honest placeholders).
    return {
        "bmi": 16.5,
        "transfusions_lifetime": 0.0,
        "hbf_pct": 5.0,
        "wbc_k_ul": 10.0,
        "platelets_k_ul": 350.0,
        "ldh_u_l": 350.0,
        "bilirubin_total_mg_dl": 1.5,
        "crp_mg_l": 3.0,
        "voc_rate_per_year": 0.25,
        "hydroxyurea": 0,
        "acs_history": 0,
        "stroke_history": 0,
    }


def build_cohort_from_nigeria_dhs2018_hr(
    stata_path: str | Path,
    *,
    timesteps: int = 24,
    image_embed_dim: int = 64,
    max_patients: int | None = None,
    random_seed: int = 42,
    read_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load NGHR*.dta rows with valid ``sb113b`` into the synthetic cohort schema."""
    stata_path = Path(stata_path)
    if not stata_path.is_file():
        raise FileNotFoundError(stata_path)

    read_kwargs = read_kwargs or {}
    df = pd.read_stata(stata_path, **read_kwargs)

    cmap = lower_columns_map(df.columns)
    gcn = find_sb113_genotype_column(df.columns)
    if gcn is None:
        keys = sorted(cmap.keys())
        raise ValueError(
            "Could not resolve genotype column (expected sb113b). "
            f"Try another export; columns sample: {keys[:25]} ..."
        )

    am = resolve_first_column(cmap, _COL_CANDIDATES["age_months"])
    wk = resolve_first_column(cmap, _COL_CANDIDATES["weight_kg"])
    hc = resolve_first_column(cmap, _COL_CANDIDATES["height_cm"])
    hb_col = resolve_first_column(cmap, _COL_CANDIDATES["hb_g_dl"])
    sx_col = resolve_first_column(cmap, _COL_CANDIDATES["child_sex"])

    work = pd.DataFrame(
        {
            "genotype_code": pd.to_numeric(df[gcn], errors="coerce"),
        }
    )
    if am is not None:
        work["age_months"] = pd.to_numeric(df[am], errors="coerce")
    else:
        work["age_months"] = np.nan
    if wk is not None:
        work["weight_kg"] = pd.to_numeric(df[wk], errors="coerce")
    else:
        work["weight_kg"] = np.nan
    if hc is not None:
        work["height_cm"] = pd.to_numeric(df[hc], errors="coerce")
    else:
        work["height_cm"] = np.nan
    if hb_col is not None:
        work["hb_g_dl"] = pd.to_numeric(df[hb_col], errors="coerce")
    else:
        work["hb_g_dl"] = np.nan
    if sx_col is not None:
        work["child_sex"] = pd.to_numeric(df[sx_col], errors="coerce")
    else:
        work["child_sex"] = np.nan

    work = work.dropna(subset=["genotype_code"])
    work = work[work["genotype_code"].between(1, 6)]
    work = work.reset_index(drop=True)

    if max_patients is not None and len(work) > max_patients:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(work), size=max_patients, replace=False)
        work = work.iloc[idx].reset_index(drop=True)

    n = len(work)
    if n == 0:
        raise ValueError("No rows with valid sb113b codes 1–6.")

    age_years = (work["age_months"].fillna(36.0) / 12.0).clip(0.1, 18.0)

    wt = work["weight_kg"]
    ht = work["height_cm"] / 100.0
    bmi = (wt / (ht ** 2)).where(ht > 0.2)
    bmi = bmi.where(bmi.notna() & np.isfinite(bmi), np.nan)

    defaults = _default_clinical_numeric()
    hb_meas = work["hb_g_dl"].where(work["hb_g_dl"].between(3.0, 20.0))

    if hb_meas.notna().any():
        med = float(np.nanmedian(hb_meas.to_numpy(dtype=np.float64)))
        hb_final = hb_meas.fillna(med)
    else:
        hb_final = hb_meas.fillna(11.0)

    sex_raw = work["child_sex"]
    sex_bin = np.zeros(n, dtype=np.int64)
    if sex_raw.notna().any():
        # DHS convention: 1=male → 1, else female → 0
        male = sex_raw.fillna(-1).astype(float).to_numpy().astype(np.int64) == 1
        sex_bin = male.astype(np.int64)

    row_genotype_label = np.array(
        [SB113_LABELS.get(int(k), "?") for k in work["genotype_code"].to_numpy()],
        dtype=object,
    )

    clin = pd.DataFrame(
        {
            "age": age_years.astype(np.float32),
            "sex": sex_bin,
            "bmi": bmi.fillna(defaults["bmi"]).astype(np.float32),
            "genotype": row_genotype_label,
            "hydroxyurea": defaults["hydroxyurea"],
            "transfusions_lifetime": float(defaults["transfusions_lifetime"]),
            "hb_g_dl": hb_final.astype(np.float32),
            "hbf_pct": float(defaults["hbf_pct"]),
            "wbc_k_ul": float(defaults["wbc_k_ul"]),
            "platelets_k_ul": float(defaults["platelets_k_ul"]),
            "ldh_u_l": float(defaults["ldh_u_l"]),
            "bilirubin_total_mg_dl": float(defaults["bilirubin_total_mg_dl"]),
            "crp_mg_l": float(defaults["crp_mg_l"]),
            "voc_rate_per_year": float(defaults["voc_rate_per_year"]),
            "acs_history": defaults["acs_history"],
            "stroke_history": defaults["stroke_history"],
        },
    )

    geno_codes = work["genotype_code"].astype(np.int64).to_numpy()
    genomic = genomic_block_from_codes(geno_codes)

    temporal = np.zeros((n, timesteps, 6), dtype=np.float32)
    imaging = np.zeros((n, image_embed_dim), dtype=np.float32)

    severity = severity_from_sb113(geno_codes)
    survival_time = np.full(n, 25.0, dtype=np.float32)
    survival_event = np.zeros(n, dtype=np.int64)

    return {
        "clinical": clin,
        "genomic": genomic,
        "imaging": imaging,
        "temporal": temporal,
        "severity": severity,
        "survival_time": survival_time,
        "survival_event": survival_event,
        "meta": {
            "source": "nigeria_dhs2018_hr",
            "n_rows": n,
            "note": "Imaging/temporal zero-filled; survival censored — use beta=0 in TrainConfig.",
        },
    }
