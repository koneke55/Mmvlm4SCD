"""Muhimbili Sickle Cohort (Tanzania) loader -- registered, Africa.

The Muhimbili University of Health and Allied Sciences (MUHAS) cohort
is the largest single-site East African SCD cohort with longitudinal
phenotype, biospecimen archive and well-published modifier-gene work.
Most studies tag patients to the Bantu/Cameroon haplotype background.
"""

from __future__ import annotations

from .base import StubLoader


class MuhimbiliMSCLoader(StubLoader):
    source = "muhimbili_msc"
    access = "registered"
    todo = ("apply through MUHAS / Muhimbili Sickle Cell Programme; "
            "obtain Tanzania NIMR ethics clearance; ingest longitudinal "
            "haematology + clinical events into the PatientRecord schema; "
            "default Region.AFRICA, Country='Tanzania', HbHaplotype.BANTU_CAR "
            "where unspecified")
