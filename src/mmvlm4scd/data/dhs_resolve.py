"""Minimal column lookups for exported DHS/Stata Household Recode files."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def lower_columns_map(columns: pd.Index) -> dict[str, str]:
    """Map lowercased Stata symbol -> original column identifier."""
    return {str(c).lower(): str(c) for c in columns}


def resolve_first_column(lmap: dict[str, str], candidates: Iterable[str]) -> str | None:
    for c in candidates:
        k = str(c).lower()
        if k in lmap:
            return lmap[k]
    return None


def find_column_substring(lmap: dict[str, str], needle: str) -> str | None:
    nl = needle.lower()
    for key, orig in lmap.items():
        if nl in key:
            return orig
    return None


def find_sb113_genotype_column(columns: pd.Index) -> str | None:
    """Nigeria NDHS sickle genotype RDT (``sb113b``) if present."""
    lmap = lower_columns_map(columns)
    resolved = resolve_first_column(lmap, ("sb113b",))
    if resolved is not None:
        return resolved
    return find_column_substring(lmap, "sb113b")
