"""NHLBI TOPMed SCD-relevant cohorts loader (DUA-required).

Replaces the polygenic-like block in the Genomic schema with a real
PRS computed on TOPMed WGS.
"""

from __future__ import annotations

from .base import StubLoader


class TOPMedSCDLoader(StubLoader):
    source = "topmed"
    access = "dua-required"
    todo = ("TOPMed application via dbGaP; coordinate with the relevant "
            "working group; emit Genomic.polygenic_score with prs_source "
            "= `topmed_v1`")
