"""MalariaGEN HBB / globin variant data loader (open, Africa)."""

from __future__ import annotations

from .base import StubLoader


class MalariaGENHBBLoader(StubLoader):
    source = "malariagen_hbb"
    access = "open"
    todo = ("download MalariaGEN African population-genomics releases, "
            "extract HBB variant calls and HbS/HbC/HbE allele "
            "frequencies, populate the Genomic block and assign the most "
            "likely HbHaplotype per population")
