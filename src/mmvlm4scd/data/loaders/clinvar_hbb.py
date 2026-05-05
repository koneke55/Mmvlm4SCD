"""ClinVar HBB pathogenic-variant catalogue (open)."""

from __future__ import annotations

from .base import StubLoader


class ClinVarHBBLoader(StubLoader):
    source = "clinvar_hbb"
    access = "open"
    todo = ("download the HBB-restricted ClinVar VCF, build the "
            "pathogenic-variant indicator block of the Genomic schema")
