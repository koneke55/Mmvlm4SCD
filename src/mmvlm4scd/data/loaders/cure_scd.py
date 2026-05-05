"""NHLBI CuRe-SCD Data Hub loader (registered)."""

from __future__ import annotations

from .base import StubLoader


class CureSCDLoader(StubLoader):
    source = "cure_scd"
    access = "registered"
    todo = ("apply through the Cure Sickle Cell Initiative data access "
            "committee; once granted, populate all four modality blocks "
            "of the unified schema")
