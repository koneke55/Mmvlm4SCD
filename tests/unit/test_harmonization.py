"""Unit tests for the unified PatientRecord schema."""

import numpy as np
import pytest

from mmvlm4scd.data.harmonization import (Ancestry, CareSettingTier,
                                          Clinical, Genomic, Genotype,
                                          HbHaplotype, HydroxyureaAccess,
                                          Imaging, Outcomes,
                                          PatientRecord, Region,
                                          SCHEMA_VERSION, SeverityLabel,
                                          Sex, Temporal, TemporalPoint,
                                          cohort_geographic_breakdown,
                                          cohort_modality_coverage,
                                          validate_cohort)


def _good_record(pid: str = "synth-abc123") -> PatientRecord:
    return PatientRecord(
        patient_id=pid, source="synthetic", cohort_split="train",
        clinical=Clinical(age_years=42.0, sex=Sex.F,
                          ancestry=Ancestry.WEST_AFRICAN,
                          genotype=Genotype.HBSS,
                          region=Region.AFRICA, country="Nigeria",
                          care_setting=CareSettingTier.TERTIARY_LMIC,
                          hydroxyurea_access=HydroxyureaAccess.AVAILABLE,
                          hb_g_dl=8.5),
        genomic=Genomic(hbb_variants=["NM_000518.5:c.20A>T"],
                        polygenic_score=0.3, prs_source="gnomad_v4",
                        hbs_haplotype=HbHaplotype.BENIN),
        imaging=Imaging(image_embedding=np.zeros(512, dtype=np.float32),
                        image_count=1, imaging_source="pbs_microscopy"),
        temporal=Temporal(timeline=[TemporalPoint(delta_days=-30, hr_bpm=78.0),
                                    TemporalPoint(delta_days=0, hr_bpm=82.0)]),
        outcomes=Outcomes(severity_label=SeverityLabel.MODERATE,
                          severity_score=1.2,
                          time_to_event_years=12.5,
                          event_observed=True),
    )


def test_good_record_validates():
    _good_record().validate()


def test_schema_version_is_semver_string():
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    for p in parts:
        assert p.isdigit()


def test_age_must_be_capped_at_90():
    r = _good_record()
    r.clinical.age_years = 92.0
    with pytest.raises(ValueError):
        r.validate()


def test_negative_age_rejected():
    r = _good_record()
    r.clinical.age_years = -1.0
    with pytest.raises(ValueError):
        r.validate()


def test_dob_like_patient_id_rejected():
    r = _good_record(pid="patient-1985-04-12")
    with pytest.raises(ValueError):
        r.validate()


def test_unknown_source_rejected():
    r = _good_record()
    r.source = "not_a_real_source"
    with pytest.raises(ValueError):
        r.validate()


def test_unknown_cohort_split_rejected():
    r = _good_record()
    r.cohort_split = "elsewhere"
    with pytest.raises(ValueError):
        r.validate()


def test_event_with_zero_time_rejected():
    r = _good_record()
    r.outcomes.event_observed = True
    r.outcomes.time_to_event_years = 0.0
    with pytest.raises(ValueError):
        r.validate()


def test_image_embedding_dtype_enforced():
    r = _good_record()
    r.imaging.image_embedding = np.zeros(512, dtype=np.float64)
    with pytest.raises(ValueError):
        r.validate()


def test_temporal_must_be_strictly_increasing():
    r = _good_record()
    r.temporal.timeline = [TemporalPoint(delta_days=0),
                           TemporalPoint(delta_days=0)]
    with pytest.raises(ValueError):
        r.validate()


def test_has_modality_flags():
    r = _good_record()
    for m in ("clinical", "genomic", "imaging", "temporal"):
        assert r.has_modality(m) is True
    r2 = PatientRecord(patient_id="p1", source="synthetic",
                       cohort_split="train")
    for m in ("clinical", "genomic", "imaging", "temporal"):
        assert r2.has_modality(m) is False


def test_to_dict_serialises_enums_and_array():
    d = _good_record().to_dict()
    assert d["clinical"]["sex"] == "F"
    assert d["clinical"]["genotype"] == "HbSS"
    assert d["clinical"]["region"] == "africa"
    assert d["clinical"]["care_setting"] == "tertiary_lmic"
    assert d["clinical"]["hydroxyurea_access"] == "available"
    assert d["genomic"]["hbs_haplotype"] == "benin"
    assert d["outcomes"]["severity_label"] == "moderate"
    assert isinstance(d["imaging"]["image_embedding"], list)


def test_geographic_breakdown_counts_by_region_and_haplotype():
    a = _good_record("p1")
    b = _good_record("p2")
    b.clinical.region = Region.SOUTH_ASIA
    b.clinical.country = "India"
    b.clinical.care_setting = CareSettingTier.COMMUNITY
    b.genomic.hbs_haplotype = HbHaplotype.ARAB_INDIAN
    breakdown = cohort_geographic_breakdown([a, b])
    assert breakdown["region"]["africa"] == 1
    assert breakdown["region"]["south_asia"] == 1
    assert breakdown["country"]["Nigeria"] == 1
    assert breakdown["country"]["India"] == 1
    assert breakdown["hbs_haplotype"]["benin"] == 1
    assert breakdown["hbs_haplotype"]["arab_indian"] == 1
    assert breakdown["care_setting"]["tertiary_lmic"] == 1
    assert breakdown["care_setting"]["community_clinic"] == 1


def test_ancestry_alias_for_african_returns_west_african():
    """Backwards-compatible alias used by older test suites."""
    assert Ancestry.AFRICAN() is Ancestry.WEST_AFRICAN


def test_haplotype_enum_covers_classical_backgrounds():
    expected = {"benin", "bantu_car", "senegal", "cameroon",
                "arab_indian", "atypical", "unknown"}
    assert {h.value for h in HbHaplotype} == expected


def test_validate_cohort_detects_duplicates():
    a = _good_record("synth-abc123")
    b = _good_record("synth-abc123")
    with pytest.raises(ValueError):
        validate_cohort([a, b])


def test_modality_coverage_fractions():
    full = _good_record("p1")
    bare = PatientRecord(patient_id="p2", source="synthetic",
                         cohort_split="train")
    cov = cohort_modality_coverage([full, bare])
    for m in ("clinical", "genomic", "imaging", "temporal"):
        assert cov[m] == 0.5
