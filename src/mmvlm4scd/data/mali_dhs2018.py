"""Mali EDS‑VI / DHS 2018 — Household Recode ``MLHR*`` child hematology/nutrition.

Released microdata (`MLHR7ADT.zip` → ``MLHR7ADT.dta``) from `<https://dhsprogram.com/>`
(free registration). Unlike Nigeria NDHS 2018, Mali 2018 **did not administer**
sickle genotype RDT; the Household children block (``RECH6``) provides
anthropometry (**``hc1``–``hc3``**) and hemoglobin (**``hc53``** / **`hc56`**
altitude-adjusted) plus WHO ``hc70``–``hc73`` z-scores when calculated.

Training labels default to **3-class anemia tiers** from validated Hb (g/dl).
``genomic`` is a deterministic hematology–growth proxy (not GWAS). Imaging and
temporal modalities are zeros. Use ``beta=0`` in ``TrainConfig``.

Ref: INSTAT Mali & ICF 2019. *Mali Demographic and Health Survey 2018.*

Catalog: `<https://microdata.worldbank.org/catalog/3526>`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dhs_resolve import lower_columns_map, resolve_first_column
from .nigeria_dhs2018 import _default_clinical_numeric


def anemia_severity_who_child(hb_g_dl: np.ndarray) -> np.ndarray:
    """Hb (g/dl) → 3 anemia classes (public-health surrogate, **not** SCD genotype).

    Aligned to common screening bands for young children with DHS Hb:

        * 0 — Hb ≥ 11.0 (non-anemic)
        * 1 — 8.0 ≤ Hb < 11.0 (mild/moderate band)
        * 2 — Hb < 8.0 (severe alarm band)
    """
    x = np.asarray(hb_g_dl, dtype=np.float64)
    out = np.where(x >= 11.0, 0, np.where(x >= 8.0, 1, 2)).astype(np.int64)
    bad = ~np.isfinite(x)
    out[bad] = 1
    return out


def genomic_block_mali_hb_growth(
    hb: np.ndarray,
    haz: np.ndarray,
    waz: np.ndarray,
    whz: np.ndarray,
    bmiz: np.ndarray,
    *,
    dim: int = 32,
) -> np.ndarray:
    """32-d proxy features for ``GenomicEncoder`` (Hb + WHO z-scores / 10)."""
    n = len(hb)
    h = np.clip(np.asarray(hb, dtype=np.float32) / 15.0, 0.0, 1.5)
    zs = []
    for z in (haz, waz, whz, bmiz):
        arr = np.nan_to_num(np.asarray(z, dtype=np.float32), nan=0.0) / 10.0
        zs.append(arr.reshape(-1, 1))
    core = np.concatenate([h.reshape(-1, 1)] + zs, axis=1)
    out = np.zeros((n, dim), dtype=np.float32)
    k = min(dim, core.shape[1])
    out[:, :k] = core[:, :k]
    return out


def build_cohort_from_mali_dhs2018_hr(
    stata_path: str | Path,
    *,
    timesteps: int = 24,
    image_embed_dim: int = 64,
    max_patients: int | None = None,
    random_seed: int = 42,
    read_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load ``MLHR*.dta`` child rows with measured Hb."""
    stata_path = Path(stata_path)
    if not stata_path.is_file():
        raise FileNotFoundError(stata_path)

    read_kwargs = read_kwargs or {}
    df = pd.read_stata(stata_path, **read_kwargs)

    cmap = lower_columns_map(df.columns)
    am = resolve_first_column(cmap, ("hc1",))
    wk = resolve_first_column(cmap, ("hc2",))
    hc = resolve_first_column(cmap, ("hc3",))
    hb_col = resolve_first_column(cmap, ("hc56", "hc53"))
    h70 = resolve_first_column(cmap, ("hc70",))
    h71 = resolve_first_column(cmap, ("hc71",))
    h72 = resolve_first_column(cmap, ("hc72",))
    h73 = resolve_first_column(cmap, ("hc73",))
    sx_col = resolve_first_column(cmap, ("hc27", "hc30"))

    if hb_col is None:
        raise ValueError(
            "No hemoglobin column (hc53/hc56). "
            f"Sample columns: {[str(c) for c in df.columns[:22]]}"
        )

    work = pd.DataFrame({"hb_g_dl": pd.to_numeric(df[hb_col], errors="coerce")})

    def _grab(colnm: str | None) -> pd.Series:
        if colnm is None:
            return pd.Series(np.nan, index=df.index)
        return pd.to_numeric(df[colnm], errors="coerce")

    work["age_months"] = _grab(am)
    work["weight_kg"] = _grab(wk)
    work["height_cm"] = _grab(hc)
    work["hc70"] = _grab(h70)
    work["hc71"] = _grab(h71)
    work["hc72"] = _grab(h72)
    work["hc73"] = _grab(h73)
    work["child_sex"] = _grab(sx_col)

    work = work[work["hb_g_dl"].between(3.0, 22.0)]
    work = work.reset_index(drop=True)

    if max_patients is not None and len(work) > max_patients:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(work), size=max_patients, replace=False)
        work = work.iloc[idx].reset_index(drop=True)

    n = len(work)
    if n == 0:
        raise ValueError("No child rows passed Hb QC (3 ≤ hb_g_dl ≤ 22).")

    age_years = (work["age_months"].fillna(36.0) / 12.0).clip(0.1, 18.0)

    wt = work["weight_kg"]
    ht = work["height_cm"] / 100.0
    bmi = (wt / (ht ** 2)).where(ht > 0.2)
    bmi = bmi.where(bmi.notna() & np.isfinite(bmi), np.nan)

    defaults = _default_clinical_numeric()
    hb_meas = work["hb_g_dl"].astype(np.float32)

    sex_raw = work["child_sex"]
    sex_bin = np.zeros(n, dtype=np.int64)
    if sex_raw.notna().any():
        male = sex_raw.fillna(-1).astype(float).to_numpy().astype(np.int64) == 1
        sex_bin = male.astype(np.int64)

    clin = pd.DataFrame(
        {
            "age": age_years.astype(np.float32),
            "sex": sex_bin,
            "bmi": bmi.fillna(defaults["bmi"]).astype(np.float32),
            "genotype": np.full(n, "MALI_EDS2018_UNGENOTYPED", dtype=object),
            "hydroxyurea": defaults["hydroxyurea"],
            "transfusions_lifetime": float(defaults["transfusions_lifetime"]),
            "hb_g_dl": hb_meas,
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

    genomic = genomic_block_mali_hb_growth(
        hb_meas.to_numpy(),
        work["hc70"].to_numpy(),
        work["hc71"].to_numpy(),
        work["hc72"].to_numpy(),
        work["hc73"].to_numpy(),
    )

    temporal = np.zeros((n, timesteps, 6), dtype=np.float32)
    imaging = np.zeros((n, image_embed_dim), dtype=np.float32)

    severity = anemia_severity_who_child(hb_meas.to_numpy(dtype=np.float64))
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
            "source": "mali_dhs2018_hr",
            "n_rows": n,
            "label": "anemia_tertile_from_hb",
            "note": "No sb113 in Mali 2018 HR; genomic = Hb + WHO z-scores. "
            "Imaging/temporal zero; use Cox beta=0.",
        },
    }
