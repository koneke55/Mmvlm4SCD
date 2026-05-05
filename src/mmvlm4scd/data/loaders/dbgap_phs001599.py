"""dbGaP phs001599 (Walk-PHaSST) loader.

Provides survival labels and pulmonary-hypertension hemodynamics.
Phase 3 priority alongside SCDIC.
"""

from __future__ import annotations

from .base import StubLoader


class DbGaPWalkPhasstLoader(StubLoader):
    source = "dbgap_phs001599"
    access = "dua-required"
    todo = ("submit dbGaP DAR for phs001599; map TRV / 6MWT / NT-proBNP "
            "into the Clinical block and the death endpoint into Outcomes")
