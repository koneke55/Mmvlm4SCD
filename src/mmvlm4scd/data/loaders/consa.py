"""CONSA -- Consortium on Newborn Screening in Africa loader (registered).

CONSA's distinctive value is *under-5 SCD survival data with
intervention exposure* across Liberia, Ghana, Tanzania, Uganda, DRC,
Kenya, Madagascar, Zambia and Nigeria. Newborn-screening + early-life
follow-up is the modality where Sub-Saharan Africa contributes signal
that no Western cohort has at scale.
"""

from __future__ import annotations

from .base import StubLoader


class CONSALoader(StubLoader):
    source = "consa"
    access = "registered"
    todo = ("submit through the CONSA coordinating centre; obtain ethics "
            "approvals at each contributing site; emit pediatric "
            "PatientRecords with explicit CareSettingTier and "
            "HydroxyureaAccess fields, since both vary widely across CONSA "
            "sites and drive survival outcomes")
