"""dbGaP phs001514 (Sickle Cell Disease Implementation Consortium) loader.

DUA-gated. Phase 3 priority. Provides the most detailed
clinical-phenotype block and pre-registered severity labels.
"""

from __future__ import annotations

from .base import StubLoader


class DbGaPSCDICLoader(StubLoader):
    source = "dbgap_phs001514"
    access = "dua-required"
    todo = ("submit dbGaP DAR for phs001514 with a Data Use Statement "
            "matching docs/preregistration.md; load the registry tables "
            "into the Clinical and Outcomes blocks of the unified schema")
