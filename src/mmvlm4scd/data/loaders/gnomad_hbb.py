"""gnomAD v4 HBB allele-frequency loader (open)."""

from __future__ import annotations

from .base import StubLoader


class GnomADHBBLoader(StubLoader):
    source = "gnomad_hbb"
    access = "open"
    todo = ("query gnomAD v4 HBB allele frequencies, use them to weight "
            "the polygenic-like score in the Genomic schema")
