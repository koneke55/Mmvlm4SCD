"""GEO GSE53441 -- whole-blood transcriptome in SCD vs controls (open)."""

from __future__ import annotations

from .base import StubLoader


class GEOGSE53441Loader(StubLoader):
    source = "geo_gse53441"
    access = "open"
    todo = ("download GSE53441 series matrix, summarise differential "
            "expression into a 32-dim embedding consistent with the "
            "GenomicEncoder input shape")
