"""H3Africa SCD bioinformatics archive loader (DUA, pan-African)."""

from __future__ import annotations

from .base import StubLoader


class H3AfricaSCDLoader(StubLoader):
    source = "h3africa_scd"
    access = "dua-required"
    todo = ("apply via H3ABioNet; SCD-specific sub-projects include "
            "modifier-gene studies (BCL11A, HMIP-2, MYB) on African "
            "haplotype backgrounds; ingest into Genomic.bcl11a_genotype "
            "and Genomic.hmip2_genotype")
