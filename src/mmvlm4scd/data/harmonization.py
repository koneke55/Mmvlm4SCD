"""Unified PatientRecord schema for real-data loaders.

The model never sees a source-specific format: every loader returns
a list of ``PatientRecord`` instances, which a single collator
converts into the tensor batches that ``MultimodalSCDModel`` expects.

The schema is enforced by ``PatientRecord.validate()`` and by an
explicit allow-list of fields per modality block. New fields require
a schema-version bump (``SCHEMA_VERSION``).

See ``docs/data_harmonization.md`` for the human-readable contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


SCHEMA_VERSION = "1.0.0"

ALLOWED_SOURCES = {
    "synthetic",
    # North America / Europe
    "mimiciv",
    "dbgap_phs001514",
    "dbgap_phs001599",
    "ukbb",
    "cure_scd",
    "topmed",
    # Open foundations
    "erythrocytes_idb",
    "kaggle_sickle_rbc",
    "geo_gse53441",
    "geo_gse35007",
    "clinvar_hbb",
    "gnomad_hbb",
    # Sub-Saharan Africa cohorts and resources
    "sparco",
    "muhimbili_msc",
    "consa",
    "luth_nigeria",
    "uch_ibadan",
    "ghana_kath_korlebu",
    "malariagen_hbb",
    "h3africa_scd",
    # South Asia cohorts and resources
    "nscaem_india",
    "icmr_nirth_jabalpur",
    "aiims_delhi",
    "lokbiradari_hemalkasa",
    "mgm_indore",
    "scs_sri_lanka",
}

ALLOWED_SPLITS = {"train", "val", "test", "external"}


class Sex(str, Enum):
    F = "F"
    M = "M"
    UNKNOWN = "unknown"


class Ancestry(str, Enum):
    # West African (predominant SCD origins for Africa + diaspora)
    WEST_AFRICAN = "west_african"
    EAST_AFRICAN = "east_african"
    CENTRAL_AFRICAN = "central_african"
    SOUTHERN_AFRICAN = "southern_african"
    NORTH_AFRICAN = "north_african"
    AFRICAN_AMERICAN = "african_american"
    AFRO_CARIBBEAN = "afro_caribbean"
    AFRO_LATINO = "afro_latino"
    # South Asian groupings relevant to the Arab-Indian / tribal HbS background
    SOUTH_ASIAN_TRIBAL = "south_asian_tribal"
    SOUTH_ASIAN_NON_TRIBAL = "south_asian_non_tribal"
    SRI_LANKAN = "sri_lankan"
    # Other SCD-relevant groups
    MIDDLE_EASTERN = "middle_eastern"
    HISPANIC_LATINO = "hispanic_latino"
    MIXED = "mixed"
    OTHER = "other"
    UNKNOWN = "unknown"

    # Backwards-compatible alias for code that used the coarse "african".
    @classmethod
    def AFRICAN(cls) -> "Ancestry":  # noqa: N802 - public alias kept for tests
        return cls.WEST_AFRICAN


class Region(str, Enum):
    """Geographic region of the cohort, mirroring DataSourceSpec.region."""
    AFRICA = "africa"
    SOUTH_ASIA = "south_asia"
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST = "middle_east"
    OTHER = "other"
    UNKNOWN = "unknown"


class HbHaplotype(str, Enum):
    """HbS haplotype background. Drives baseline HbF, severity and
    response to hydroxyurea; ignoring it across geographies is the most
    common source of unfair severity predictions in SCD.

    Refs: Pagnier et al. 1984; Lapoumeroulie et al. 1992;
    Kulozik et al. 1986 (Arab-Indian).
    """
    BENIN = "benin"
    BANTU_CAR = "bantu_car"  # Central African Republic / Bantu
    SENEGAL = "senegal"
    CAMEROON = "cameroon"
    ARAB_INDIAN = "arab_indian"  # India + Eastern Saudi Arabia
    ATYPICAL = "atypical"
    UNKNOWN = "unknown"


class CareSettingTier(str, Enum):
    """Resource-stratified care setting -- crucial for transferability
    between high-resource (US/UK) and low/middle-income care contexts.

    Maps loosely to WHO health-system tiers + hydroxyurea availability.
    """
    TERTIARY_HIC = "tertiary_high_income"          # US/UK academic centre
    TERTIARY_LMIC = "tertiary_lmic"                # Indian / African tertiary centre
    SECONDARY_LMIC = "secondary_lmic"              # district hospital
    PRIMARY_LMIC = "primary_lmic"                  # primary clinic
    COMMUNITY = "community_clinic"                 # NGO / mission / community
    UNKNOWN = "unknown"


class HydroxyureaAccess(str, Enum):
    """Whether hydroxyurea is reliably available at the patient's care
    setting. Treatment effect size differs dramatically across these
    tiers, so the model and its evaluations must be stratified."""
    ROUTINE = "routine"           # available with no out-of-pocket barrier
    AVAILABLE = "available"       # available but with cost / supply issues
    INTERMITTENT = "intermittent" # supply gaps, frequent interruptions
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class Genotype(str, Enum):
    HBSS = "HbSS"
    HBSC = "HbSC"
    HBSBETA_PLUS = "HbSbeta+"
    HBSBETA_ZERO = "HbSbeta0"
    HBS_OTHER = "HbSother"
    UNKNOWN = "unknown"


class SeverityLabel(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


@dataclass
class Clinical:
    age_years: Optional[float] = None
    sex: Sex = Sex.UNKNOWN
    ancestry: Ancestry = Ancestry.UNKNOWN
    genotype: Genotype = Genotype.UNKNOWN
    region: Region = Region.UNKNOWN
    country: Optional[str] = None  # ISO-3166 alpha-2 or full country name
    care_setting: CareSettingTier = CareSettingTier.UNKNOWN
    hydroxyurea_access: HydroxyureaAccess = HydroxyureaAccess.UNKNOWN
    hb_g_dl: Optional[float] = None
    hbf_pct: Optional[float] = None
    wbc_k_uL: Optional[float] = None
    plt_k_uL: Optional[float] = None
    ldh_u_l: Optional[float] = None
    bili_total_mg_dl: Optional[float] = None
    crp_mg_l: Optional[float] = None
    bmi_kg_m2: Optional[float] = None
    acs_history: Optional[bool] = None
    stroke_history: Optional[bool] = None
    hydroxyurea_ever: Optional[bool] = None
    chronic_transfusion: Optional[bool] = None
    voc_per_year: Optional[float] = None


@dataclass
class Genomic:
    hbb_variants: List[str] = field(default_factory=list)
    polygenic_score: Optional[float] = None
    prs_source: Optional[str] = None
    hbs_haplotype: HbHaplotype = HbHaplotype.UNKNOWN
    # Co-inherited modifiers known to alter SCD phenotype:
    alpha_thal_status: Optional[str] = None  # e.g. "alpha-/alpha", "-alpha3.7"
    bcl11a_genotype: Optional[str] = None    # rs1427407 etc.
    hmip2_genotype: Optional[str] = None     # HBS1L-MYB intergenic


@dataclass
class Imaging:
    image_embedding: Optional[np.ndarray] = None  # (D,) float32
    image_count: int = 0
    imaging_source: Optional[str] = None


@dataclass
class TemporalPoint:
    delta_days: int
    hr_bpm: Optional[float] = None
    spo2_pct: Optional[float] = None
    sbp_mmHg: Optional[float] = None
    hb_g_dl: Optional[float] = None
    wbc_k_uL: Optional[float] = None
    pain_vas: Optional[float] = None
    observed_mask: Dict[str, bool] = field(default_factory=dict)


@dataclass
class Temporal:
    timeline: List[TemporalPoint] = field(default_factory=list)


@dataclass
class Outcomes:
    severity_label: SeverityLabel = SeverityLabel.UNKNOWN
    severity_score: Optional[float] = None
    time_to_event_years: Optional[float] = None
    event_observed: Optional[bool] = None
    cause_of_death: Optional[str] = None


# --------------------------------------------------------------------- #
# PatientRecord                                                         #
# --------------------------------------------------------------------- #

# Patterns that look like a date of birth and must NEVER appear in
# ``patient_id``.  This is one of the cheap pre-flight checks on top of
# the full PHI scanner in ``deidentification.py``.
_DOBLIKE_RE = (
    r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"
    r"|\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b"
)


@dataclass
class PatientRecord:
    """Source-agnostic patient representation."""

    patient_id: str
    source: str
    cohort_split: str
    clinical: Clinical = field(default_factory=Clinical)
    genomic: Genomic = field(default_factory=Genomic)
    imaging: Imaging = field(default_factory=Imaging)
    temporal: Temporal = field(default_factory=Temporal)
    outcomes: Outcomes = field(default_factory=Outcomes)
    schema_version: str = SCHEMA_VERSION

    # ----------------------------- validation -------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on any contract violation."""
        import re

        if not self.patient_id or not isinstance(self.patient_id, str):
            raise ValueError("patient_id must be a non-empty string")
        if re.search(_DOBLIKE_RE, self.patient_id):
            raise ValueError("patient_id looks like a date of birth")
        if self.source not in ALLOWED_SOURCES:
            raise ValueError(f"unknown source: {self.source}")
        if self.cohort_split not in ALLOWED_SPLITS:
            raise ValueError(f"unknown cohort_split: {self.cohort_split}")

        major = self.schema_version.split(".", 1)[0]
        if major != SCHEMA_VERSION.split(".", 1)[0]:
            raise ValueError(
                f"schema major mismatch: record {self.schema_version} "
                f"runtime {SCHEMA_VERSION}")

        c = self.clinical
        if c.age_years is not None:
            if c.age_years < 0 or c.age_years > 90:
                raise ValueError(
                    "age_years must be in [0, 90]; HIPAA Safe-Harbor "
                    "requires ages > 89 to be capped at 90.")
        if not isinstance(c.sex, Sex):
            raise ValueError("clinical.sex must be a Sex enum value")
        if not isinstance(c.genotype, Genotype):
            raise ValueError("clinical.genotype must be a Genotype enum value")

        i = self.imaging
        if i.image_embedding is not None:
            if not isinstance(i.image_embedding, np.ndarray):
                raise ValueError("image_embedding must be a numpy array")
            if i.image_embedding.dtype != np.float32:
                raise ValueError("image_embedding must be float32")
            if i.image_embedding.ndim != 1:
                raise ValueError("image_embedding must be 1-D")

        prev = -10**9
        for p in self.temporal.timeline:
            if p.delta_days <= prev:
                raise ValueError(
                    "temporal.timeline must be strictly increasing in "
                    "delta_days")
            prev = p.delta_days

        o = self.outcomes
        if o.event_observed is True and (o.time_to_event_years is None or
                                          o.time_to_event_years <= 0):
            raise ValueError(
                "event_observed=True requires positive time_to_event_years")
        if o.severity_label is not SeverityLabel.UNKNOWN \
                and o.severity_score is not None \
                and not np.isfinite(o.severity_score):
            raise ValueError(
                "non-unknown severity_label requires finite severity_score")

    # ----------------------------- helpers ---------------------------

    def has_modality(self, name: str) -> bool:
        if name == "clinical":
            return self.clinical.age_years is not None
        if name == "genomic":
            return (bool(self.genomic.hbb_variants)
                    or self.genomic.polygenic_score is not None)
        if name == "imaging":
            return self.imaging.image_embedding is not None
        if name == "temporal":
            return bool(self.temporal.timeline)
        raise ValueError(f"unknown modality: {name}")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # numpy arrays -> lists for JSON serialisability
        ie = self.imaging.image_embedding
        if ie is not None:
            d["imaging"]["image_embedding"] = ie.tolist()
        # enum values -> their string equivalents
        d["clinical"]["sex"] = self.clinical.sex.value
        d["clinical"]["ancestry"] = self.clinical.ancestry.value
        d["clinical"]["genotype"] = self.clinical.genotype.value
        d["clinical"]["region"] = self.clinical.region.value
        d["clinical"]["care_setting"] = self.clinical.care_setting.value
        d["clinical"]["hydroxyurea_access"] = \
            self.clinical.hydroxyurea_access.value
        d["genomic"]["hbs_haplotype"] = self.genomic.hbs_haplotype.value
        d["outcomes"]["severity_label"] = self.outcomes.severity_label.value
        return d


# --------------------------------------------------------------------- #
# Cohort-level helpers                                                  #
# --------------------------------------------------------------------- #

def validate_cohort(records: Iterable[PatientRecord]) -> None:
    """Validate every record and assert patient-id uniqueness."""
    seen = set()
    for r in records:
        r.validate()
        if r.patient_id in seen:
            raise ValueError(f"duplicate patient_id: {r.patient_id}")
        seen.add(r.patient_id)


def cohort_modality_coverage(records: Iterable[PatientRecord]
                             ) -> Dict[str, float]:
    """Per-modality fraction of records that have data.

    Useful for reporting in the manuscript and for the modality-aware
    DataLoader to know which modalities are present at all.
    """
    records = list(records)
    n = max(len(records), 1)
    out = {}
    for m in ("clinical", "genomic", "imaging", "temporal"):
        out[m] = sum(r.has_modality(m) for r in records) / n
    return out


def cohort_geographic_breakdown(records: Iterable[PatientRecord]
                                ) -> Dict[str, Dict[str, int]]:
    """Patients by region, country, haplotype and care setting.

    Reported alongside any geographic-transfer claim so that reviewers
    can see who is actually in the training versus evaluation data.
    """
    records = list(records)
    region: Dict[str, int] = {}
    country: Dict[str, int] = {}
    haplotype: Dict[str, int] = {}
    care: Dict[str, int] = {}
    for r in records:
        region[r.clinical.region.value] = \
            region.get(r.clinical.region.value, 0) + 1
        c = r.clinical.country or "unknown"
        country[c] = country.get(c, 0) + 1
        haplotype[r.genomic.hbs_haplotype.value] = \
            haplotype.get(r.genomic.hbs_haplotype.value, 0) + 1
        care[r.clinical.care_setting.value] = \
            care.get(r.clinical.care_setting.value, 0) + 1
    return {"region": region, "country": country,
            "hbs_haplotype": haplotype, "care_setting": care}
