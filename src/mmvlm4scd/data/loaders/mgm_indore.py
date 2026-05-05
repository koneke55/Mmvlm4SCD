"""MGM Medical College Indore SCD programme loader (DUA, India)."""

from __future__ import annotations

from .base import StubLoader


class MGMIndoreLoader(StubLoader):
    source = "mgm_indore"
    access = "dua-required"
    todo = ("MGM IEC approval; western-India regional SCD programme; "
            "useful for treatment-effect heterogeneity (hydroxyurea) "
            "and pregnancy-outcome studies")
