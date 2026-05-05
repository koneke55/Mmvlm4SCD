"""Komfo Anokye / Korle-Bu SCD programmes loader (DUA, Ghana)."""

from __future__ import annotations

from .base import StubLoader


class GhanaKATHKorleBuLoader(StubLoader):
    source = "ghana_kath_korlebu"
    access = "dua-required"
    todo = ("Ghana Health Service Ethics Review Committee approval; site "
            "MoU with KATH (Kumasi) and Korle-Bu (Accra); these are the "
            "two pillars of the Ghanaian newborn-screening + hydroxyurea "
            "programme")
