"""AIIMS New Delhi SCD Registry loader (DUA, India)."""

from __future__ import annotations

from .base import StubLoader


class AIIMSDelhiLoader(StubLoader):
    source = "aiims_delhi"
    access = "dua-required"
    todo = ("AIIMS Institutional Ethics Committee approval; tertiary "
            "centre data with paired clinical, biochemical and HBB/HBA "
            "genotypes; useful for adult-onset complication studies on "
            "the Arab-Indian background")
