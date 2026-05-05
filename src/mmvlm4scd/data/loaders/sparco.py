"""SickleInAfrica / SPARCO Registry loader (DUA-required, Africa).

The Sickle Pan-African Research Consortium operates a harmonised
electronic SCD registry across Tanzania (Muhimbili), Ghana (Kumasi,
Korle-Bu) and Nigeria (Lagos, Ibadan), with Cameroon and Mali
expanding. This is the *flagship* multi-site African SCD cohort and
the primary target for Phase 3 of the African branch of
``docs/real_data_roadmap.md``.

Implementation notes:

1. Data access is mediated by SickleInAfrica leadership and the host
   country's IRB; an MoU with the local PI is required *before* any
   data are extracted.
2. CARE Principles for Indigenous Data Governance and the H3Africa
   Ethics Working Group guidance apply: participant communities have
   a say in secondary use.
3. Records are stratified by ``Region.AFRICA`` and the appropriate
   country code; ``HbHaplotype`` is typically Benin (West Africa) or
   Bantu/Cameroon (Central / East Africa).
"""

from __future__ import annotations

from .base import StubLoader


class SPARCORegistryLoader(StubLoader):
    source = "sparco"
    access = "dua-required"
    todo = ("sign MoU + IRB approvals at host country (Tanzania NIMR / "
            "Ghana GHS-ERC / Nigeria NHREC), then map the harmonised "
            "registry tables to the unified PatientRecord schema with "
            "Region.AFRICA and the appropriate HbHaplotype")
