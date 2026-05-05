"""HIPAA Safe-Harbor de-identification utilities.

Three responsibilities:

1. ``assert_no_phi_columns(df)``: refuse to load a frame that contains
   columns matching HIPAA Safe-Harbor identifiers.
2. ``hash_patient_id(...)``: produce stable, non-reversible patient
   identifiers from source-specific IDs + a project salt.
3. ``shift_dates(...)`` and ``cap_age(...)``: helpers for date-shifting
   and age capping.
4. ``scan_text_for_phi(...)``: regex scanner used by the pre-commit
   hook to refuse commits that smuggle PHI into git.

This module is intentionally **conservative**: it errs on the side of
refusing data rather than silently passing PHI through.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import re
from typing import Iterable, List, Tuple

import pandas as pd


# --------------------------------------------------------------------- #
# Forbidden column names                                                #
# --------------------------------------------------------------------- #

# Case-insensitive substring match. If a substring appears in any
# column name, the loader refuses the frame.
_FORBIDDEN_COL_SUBSTRINGS = (
    "name", "first_name", "last_name", "middle_name",
    "address", "street", "zip", "postcode", "postal_code",
    "phone", "fax", "email", "e_mail",
    "ssn", "social_security",
    "mrn", "medical_record_number",
    "license", "licence", "drivers_license",
    "vin", "vehicle",
    "ip_address", "ipv4", "ipv6", "url",
    "fingerprint", "biometric", "voiceprint",
    "photo", "photograph", "face_image",
    "dob", "date_of_birth", "birth_date",
)

# A few exact column names that are technically allowed elsewhere but
# are PHI in clinical contexts.  We block them and require the loader
# to remap.
_FORBIDDEN_COL_EXACT = {"address1", "address2"}


def assert_no_phi_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any column name resembles PHI."""
    bad: List[str] = []
    for col in df.columns:
        c = str(col).lower()
        if c in _FORBIDDEN_COL_EXACT:
            bad.append(col)
            continue
        for sub in _FORBIDDEN_COL_SUBSTRINGS:
            if sub in c:
                bad.append(col)
                break
    if bad:
        raise ValueError(
            f"frame contains PHI-looking columns: {sorted(set(bad))}. "
            "Drop or remap before loading.")


# --------------------------------------------------------------------- #
# Patient-ID hashing                                                    #
# --------------------------------------------------------------------- #

def hash_patient_id(source: str, original_id: str, project_salt: str,
                    digest_len: int = 16) -> str:
    """Stable, non-reversible patient ID.

    Uses HMAC-SHA256 with the project salt as the key. Output is the
    first ``digest_len`` hex characters prefixed with the source
    abbreviation so two cohorts with colliding raw IDs cannot collide
    in the unified namespace.
    """
    if not project_salt or len(project_salt) < 16:
        raise ValueError(
            "project_salt must be at least 16 characters; treat it like "
            "a secret -- never commit it.")
    msg = f"{source}::{original_id}".encode("utf-8")
    key = project_salt.encode("utf-8")
    h = hmac.new(key, msg, hashlib.sha256).hexdigest()[:digest_len]
    return f"{source[:6]}-{h}"


# --------------------------------------------------------------------- #
# Date / age helpers                                                    #
# --------------------------------------------------------------------- #

def cap_age(age_years: float, ceiling: int = 90) -> float:
    """HIPAA Safe-Harbor: ages > 89 collapse to 90."""
    if age_years is None:
        return None
    return float(min(age_years, ceiling))


def shift_dates(dates: Iterable[_dt.date], shift_days: int
                ) -> List[_dt.date]:
    """Apply a constant shift (per-patient) to a list of dates.

    Per HIPAA Safe-Harbor, dates more granular than the year are PHI;
    the standard work-around is to shift every date for a given patient
    by the same random number of days (drawn once and stored only on the
    secured workstation, never in the public repo).
    """
    return [d + _dt.timedelta(days=shift_days) for d in dates]


def relative_days_from_index(dates: Iterable[_dt.date],
                             index_date: _dt.date) -> List[int]:
    """Convert a series of dates to integer days relative to the index
    encounter; the absolute calendar date never leaves the function."""
    return [(d - index_date).days for d in dates]


# --------------------------------------------------------------------- #
# Text scanner (pre-commit hook)                                        #
# --------------------------------------------------------------------- #

# Regex patterns for PHI that might show up by accident in code, docs,
# notebooks, or commit messages.
_PHI_PATTERNS: List[Tuple[str, str]] = [
    # SSN: 3-2-4
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    # US phone numbers
    ("phone_us", r"\b(\+?1[\s\-\.])?\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}\b"),
    # Email
    ("email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    # Common DOB formats
    ("dob_iso",
     r"\bDOB\s*[:=]?\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"),
    # Medical record numbers labelled MRN
    ("mrn_labeled", r"\bMRN[:\s#]+[A-Z0-9]{4,}\b"),
    # ZIP codes labelled
    ("zip_labeled",
     r"\bZIP\s*[:=]?\s*\d{5}(?:-\d{4})?\b"),
]


def scan_text_for_phi(text: str) -> List[Tuple[str, str]]:
    """Return a list of ``(label, match)`` tuples for any PHI found.

    Used by the pre-commit hook to block commits.
    """
    findings: List[Tuple[str, str]] = []
    for label, pat in _PHI_PATTERNS:
        for m in re.finditer(pat, text):
            findings.append((label, m.group(0)))
    return findings


def scan_path_for_phi(path: str, max_bytes: int = 1_000_000
                      ) -> List[Tuple[str, str, int]]:
    """Scan a file path. Returns ``(label, match, line_no)`` tuples."""
    out: List[Tuple[str, str, int]] = []
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    text = data.decode("utf-8", errors="replace")
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, match in scan_text_for_phi(line):
            out.append((label, match, line_no))
    return out
