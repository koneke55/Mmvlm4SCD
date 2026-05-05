"""Tests for DHS/Stata column helpers."""

from pandas import Index

from mmvlm4scd.data.dhs_resolve import (
    find_sb113_genotype_column,
    lower_columns_map,
    resolve_first_column,
)


def test_lower_and_resolve():
    lmap = lower_columns_map(Index(["HC1", "SB113B", "Hc53"]))
    assert resolve_first_column(lmap, ("hc1",)) == "HC1"
    assert resolve_first_column(lmap, ("missing",)) is None


def test_find_sb113():
    assert find_sb113_genotype_column(Index(["v1", "sb113b"])) == "sb113b"
    assert find_sb113_genotype_column(Index(["v1", "x_sb113b_tail"])) == "x_sb113b_tail"
    assert find_sb113_genotype_column(Index(["hc53",])) is None
