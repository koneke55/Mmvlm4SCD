"""National Sickle Cell Anaemia Elimination Mission (India) loader.

NSCAEM is India's 2023 national programme targeting screening of ~70 M
people in 17 SCD-endemic states, predominantly tribal populations.
The dominant HbS background here is the Arab-Indian haplotype, which
exhibits high baseline HbF and a distinctive severity profile.

Access is registered + state-level MoUs.
"""

from __future__ import annotations

from .base import StubLoader


class NSCAEMIndiaLoader(StubLoader):
    source = "nscaem_india"
    access = "registered"
    todo = ("obtain ICMR + State Health Society approvals; ingest the "
            "district-level screening + follow-up data, including "
            "HemeChip / SickleSCAN point-of-care results; default "
            "HbHaplotype.ARAB_INDIAN; use Ancestry.SOUTH_ASIAN_TRIBAL "
            "where the screened population is tribal")
