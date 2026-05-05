"""MIMIC-IV Sickle Cell Disease subset loader (registered, PhysioNet).

Implementation notes for Phase 2:

1. PhysioNet credentialed-researcher status required.
2. Identify SCD patients via ``diagnoses_icd`` join on ``icd_code IN
   ('D570', 'D571', 'D572', 'D574', 'D578')`` (ICD-10).
3. For each patient, build:
   - ``Clinical``: demographics from ``patients`` + first labs from
     ``labevents`` joined to ``d_labitems``.
   - ``Temporal``: longitudinal labs (Hb, WBC, LDH) and vitals
     (HR, SpO2, SBP) sampled at irregular times -- emit a
     ``TemporalPoint`` per observation, keep ``observed_mask``
     accurate.
   - ``Outcomes.event_observed``: True iff ``patients.dod`` is set.
4. Drop any column that fails ``assert_no_phi_columns``.
5. Hash patient IDs via ``hash_patient_id(source, subject_id, salt)``.
"""

from __future__ import annotations

from .base import StubLoader


class MimicIVSCDLoader(StubLoader):
    source = "mimiciv"
    access = "registered"
    todo = ("complete PhysioNet DUA, then wire SQL queries against the "
            "MIMIC-IV `hosp` and `icu` schemas, restricted to ICD-10 "
            "codes D57.0/D57.1/D57.2/D57.4/D57.8")
