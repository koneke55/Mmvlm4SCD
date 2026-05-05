"""Unit tests for HIPAA-Safe-Harbor de-identification helpers."""

import datetime as dt

import pandas as pd
import pytest

from mmvlm4scd.data.deidentification import (assert_no_phi_columns, cap_age,
                                             hash_patient_id,
                                             relative_days_from_index,
                                             scan_path_for_phi,
                                             scan_text_for_phi,
                                             shift_dates)


def test_assert_no_phi_columns_passes_clean_frame():
    df = pd.DataFrame({"age_years": [45], "hb_g_dl": [9.0],
                       "genotype": ["HbSS"]})
    assert_no_phi_columns(df)


@pytest.mark.parametrize("col", [
    "patient_name", "first_name", "address", "zip_code", "phone",
    "email_address", "ssn", "mrn", "date_of_birth", "dob",
    "ip_address", "drivers_license", "fingerprint", "address1",
])
def test_assert_no_phi_columns_rejects_phi(col: str):
    df = pd.DataFrame({col: ["x"], "age_years": [42]})
    with pytest.raises(ValueError):
        assert_no_phi_columns(df)


def test_hash_patient_id_is_deterministic_and_namespaced():
    a = hash_patient_id("mimiciv", "subject-1", "this_is_a_long_salt_xyz")
    b = hash_patient_id("mimiciv", "subject-1", "this_is_a_long_salt_xyz")
    c = hash_patient_id("mimiciv", "subject-2", "this_is_a_long_salt_xyz")
    assert a == b
    assert a != c
    assert a.startswith("mimici-")


def test_hash_patient_id_rejects_short_salt():
    with pytest.raises(ValueError):
        hash_patient_id("mimiciv", "subject-1", "short")


def test_cap_age_collapses_above_89():
    assert cap_age(85) == 85
    assert cap_age(89) == 89
    assert cap_age(91) == 90
    assert cap_age(120) == 90
    assert cap_age(None) is None


def test_shift_dates_applies_constant_offset():
    base = [dt.date(2020, 1, 1), dt.date(2021, 6, 15)]
    out = shift_dates(base, shift_days=30)
    assert out == [dt.date(2020, 1, 31), dt.date(2021, 7, 15)]


def test_relative_days_from_index_zero_at_index():
    base = [dt.date(2020, 1, 1), dt.date(2020, 1, 4), dt.date(2020, 2, 1)]
    out = relative_days_from_index(base, dt.date(2020, 1, 1))
    assert out == [0, 3, 31]


@pytest.mark.parametrize("text, label", [
    ("contact 555-12-3456 SSN", "ssn"),
    ("Reach me at john.doe@example.com please", "email"),
    ("DOB: 1984-05-12", "dob_iso"),
    ("MRN: AB12345", "mrn_labeled"),
    ("ZIP: 02139-1234", "zip_labeled"),
    ("call (415) 555-1212", "phone_us"),
])
def test_scan_text_for_phi_finds_labels(text: str, label: str):
    found = scan_text_for_phi(text)
    assert any(lab == label for lab, _ in found), found


def test_scan_text_for_phi_clean_string_returns_empty():
    assert scan_text_for_phi("just regular code without identifiers") == []


def test_scan_path_for_phi_reports_line_numbers(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("clean line\n"
                 "contact me at jane@example.com today\n"
                 "another clean line\n"
                 "patient SSN 123-45-6789\n")
    findings = scan_path_for_phi(str(p))
    by_label = {lab: line for lab, _, line in findings}
    assert by_label["email"] == 2
    assert by_label["ssn"] == 4
