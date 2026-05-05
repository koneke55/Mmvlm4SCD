"""ICMR-NIRTH Jabalpur Tribal SCD cohort loader (DUA, India)."""

from __future__ import annotations

from .base import StubLoader


class ICMRNIRTHJabalpurLoader(StubLoader):
    source = "icmr_nirth_jabalpur"
    access = "dua-required"
    todo = ("MoU + Tribal Health Bureau approval; longitudinal central-"
            "Indian tribal SCD cohort; central reference for the "
            "Arab-Indian haplotype background; ensure CARE Principles "
            "for Indigenous Data Governance are honoured before any "
            "secondary-use derivation")
