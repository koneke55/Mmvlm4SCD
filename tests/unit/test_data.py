import numpy as np

from mmvlm4scd.data import (PUBLIC_SCD_DATASETS, StandardPreprocessor,
                            generate_synthetic_cohort, list_sources)
from mmvlm4scd.data.synthetic import SCDSyntheticConfig, split_indices, _genotype_prob_for_config


def test_registry_non_empty_and_filters():
    assert len(PUBLIC_SCD_DATASETS) >= 10
    open_sources = list_sources(access="open")
    assert all(s.access == "open" for s in open_sources)
    imaging = list_sources(modality="imaging")
    assert all(s.modality == "imaging" for s in imaging)


def test_registry_geographic_filters():
    """Africa + South Asia entries must exist and be filterable."""
    africa = list_sources(region="africa")
    south_asia = list_sources(region="south_asia")
    assert len(africa) >= 5
    assert len(south_asia) >= 5
    # Africa + South Asia together must outweigh the US/UK reference set.
    na_eu = (list_sources(region="north_america")
             + list_sources(region="europe"))
    assert len(africa) + len(south_asia) >= len(na_eu)
    nigeria = list_sources(country="Nigeria")
    assert len(nigeria) >= 1
    india = list_sources(country="India")
    assert len(india) >= 3


def test_registry_includes_priority_african_and_indian_sources():
    """Spot-check the flagship cohorts the roadmap depends on."""
    names = {s.name for s in PUBLIC_SCD_DATASETS}
    assert any("SickleInAfrica" in n or "SPARCO" in n for n in names)
    assert any("Muhimbili" in n for n in names)
    assert any("CONSA" in n for n in names)
    assert any("NSCAEM" in n for n in names)
    assert any("NIRTH" in n for n in names)
    assert any("AIIMS" in n for n in names)


def test_synthetic_cohort_shapes():
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=300, seed=0))
    n = len(cohort["clinical"])
    assert n == 300
    assert cohort["genomic"].shape == (n, 32)
    assert cohort["imaging"].shape == (n, 64)
    assert cohort["temporal"].shape[0] == n and cohort["temporal"].shape[2] == 6
    assert cohort["severity"].shape == (n,)
    assert cohort["survival_time"].shape == (n,)
    assert cohort["survival_event"].shape == (n,)
    assert set(np.unique(cohort["severity"]).tolist()) <= {0, 1, 2}


def test_preprocessor_roundtrip():
    cohort = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=128, seed=0))
    pre = StandardPreprocessor()
    x = pre.fit_transform(cohort["clinical"])
    assert x.dtype == np.float32
    assert x.shape[0] == 128
    assert x.shape[1] == pre.output_dim
    means = x.mean(axis=0)
    assert np.all(np.abs(means) < 1.0)


def test_west_africa_maf_reweights_genotypes():
    p0 = _genotype_prob_for_config(SCDSyntheticConfig(seed=0))
    p1 = _genotype_prob_for_config(SCDSyntheticConfig(seed=0, west_africa_rs334_maf=0.13))
    assert abs(p0.sum() - 1.0) < 1e-9
    assert abs(p1.sum() - 1.0) < 1e-9
    assert p1[0] >= p0[0]
    assert p1[0] + p1[1] >= p0[0] + p0[1]


def test_west_africa_maf_more_hbss_in_large_cohort():
    base = generate_synthetic_cohort(SCDSyntheticConfig(n_patients=8000, seed=42))
    wa = generate_synthetic_cohort(
        SCDSyntheticConfig(n_patients=8000, seed=42, west_africa_rs334_maf=0.14)
    )
    hbss_base = (base["clinical"]["genotype"].values == "HbSS").mean()
    hbss_wa = (wa["clinical"]["genotype"].values == "HbSS").mean()
    assert hbss_wa >= hbss_base


def test_split_indices_partition():
    a, b, c = split_indices(1000, seed=1)
    assert len(a) + len(b) + len(c) == 1000
    assert len(set(a.tolist()) & set(b.tolist())) == 0
    assert len(set(b.tolist()) & set(c.tolist())) == 0
