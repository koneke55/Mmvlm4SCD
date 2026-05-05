"""Sickle Cell Society of Sri Lanka cohort loader (registered)."""

from __future__ import annotations

from .base import StubLoader


class SCSSriLankaLoader(StubLoader):
    source = "scs_sri_lanka"
    access = "registered"
    todo = ("liaison through the Sickle Cell Society of Sri Lanka; "
            "smaller cohort but distinct ancestry and care system; "
            "Region.SOUTH_ASIA, Ancestry.SRI_LANKAN")
