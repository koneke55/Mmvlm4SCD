"""University College Hospital, Ibadan SCD cohort loader (DUA, Nigeria)."""

from __future__ import annotations

from .base import StubLoader


class UCHIbadanLoader(StubLoader):
    source = "uch_ibadan"
    access = "dua-required"
    todo = ("UI/UCH Health Research Ethics Committee; complements LUTH for "
            "South-Western Nigerian phenotypic diversity; supports HbF "
            "modifier and treatment-response studies on the Benin "
            "haplotype")
