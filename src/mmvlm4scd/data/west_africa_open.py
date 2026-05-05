"""Open, API-addressable West African reference data for SCD-related priors.

This module fetches **population-level** allele frequencies for rs334 (HBB
Glu7Val / sickle mutation) from the Ensembl Variation REST API for 1000
Genomes Phase 3 West African panels (YRI, ESN, GWD, MSL).

It does **not** ship individual-level patient records (those live behind
cohort-specific DUAs). The returned table is suitable for calibrating
synthetic benchmarking cohorts or for reporting provenance in notebooks.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

ENSEMBL_RS334_URL = (
    "https://rest.ensembl.org/variation/human/rs334?pops=1;content-type=application/json"
)

# 1000 Genomes Phase 3 West African population suffixes (Nigeria, Gambia,
# Sierra Leone in the reference panel).
WEST_AFRICA_1KG_PHASE3_SUFFIXES: tuple[str, ...] = (":YRI", ":ESN", ":GWD", ":MSL")


class EnsemblFetchError(RuntimeError):
    """Raised when the Ensembl rs334 fetch fails or payload is malformed."""


def _is_west_africa_phase3(population: str) -> bool:
    return population.startswith("1000GENOMES:phase_3:") and any(
        population.endswith(suf) for suf in WEST_AFRICA_1KG_PHASE3_SUFFIXES
    )


def fetch_rs334_west_africa_1000g_phase3(
    timeout: float = 60.0,
    user_agent: str = "mmvlm4scd/0.1.1 (https://github.com/sickle-cell-research; "
    "academic use)",
) -> pd.DataFrame:
    """Download rs334 population frequencies and keep West African 1KG panels.

    Returns a tidy table with columns
        ``population``, ``allele``, ``frequency``, ``allele_count``.
    """
    req = Request(
        ENSEMBL_RS334_URL,
        headers={
            "Accept": "application/json",
            "User-Agent": user_agent,
        },
        method="GET",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 - curated HTTPS endpoint
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:  # pragma: no cover - network
        raise EnsemblFetchError(f"Ensembl HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:  # pragma: no cover - network
        raise EnsemblFetchError(f"Ensembl network error: {exc.reason}") from exc

    try:
        payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EnsemblFetchError("Ensembl response was not valid JSON") from exc

    rows = payload.get("populations") or []
    out: list[dict[str, Any]] = []
    for row in rows:
        pop = row.get("population") or ""
        if not _is_west_africa_phase3(pop):
            continue
        out.append(
            {
                "population": pop,
                "allele": row.get("allele"),
                "frequency": float(row.get("frequency", 0.0)),
                "allele_count": int(row.get("allele_count", 0)),
            }
        )
    if not out:
        raise EnsemblFetchError(
            "No West African 1000 Genomes Phase 3 populations found in Ensembl "
            "response (API shape may have changed)."
        )
    return pd.DataFrame(out).sort_values("population").reset_index(drop=True)


def mean_rs334_maf_west_africa(df: pd.DataFrame | None = None) -> float:
    """Mean per-population minor allele frequency of rs334 across panels.

    For each ``population`` we take ``min(f, 1-f)`` over the two alleles,
    then average across the four West African Phase 3 cohorts.
    """
    frame = df if df is not None else fetch_rs334_west_africa_1000g_phase3()
    mafs: list[float] = []
    for _, sub in frame.groupby("population", sort=False):
        freqs = sub["frequency"].astype(float).tolist()
        if len(freqs) < 2:  # pragma: no cover - defensive
            continue
        mafs.append(float(min(freqs)))
    if not mafs:
        raise EnsemblFetchError("Could not derive MAF values from frequency table.")
    return float(sum(mafs) / len(mafs))
