"""UK Biobank SCD subset loader (DUA-required).

Provides the most diverse multimodal signal (imaging, biochemistry,
genotype array) for external validation.
"""

from __future__ import annotations

from .base import StubLoader


class UKBiobankSCDLoader(StubLoader):
    source = "ukbb"
    access = "dua-required"
    todo = ("UK Biobank application + GDPR DPIA; restrict to ICD-10 "
            "D57.* coded participants; map the four modalities through "
            "the unified schema")
