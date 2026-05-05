"""Kaggle Sickle RBC dataset loader (open)."""

from __future__ import annotations

from .base import StubLoader


class KaggleSickleRBCLoader(StubLoader):
    source = "kaggle_sickle_rbc"
    access = "open"
    todo = ("download via `kaggle datasets download sandeshb/sickle-cell`, "
            "run the Phase-1 CNN encoder, emit PatientRecords with imaging "
            "embeddings; this set is also used as a CNN augmentation pool")
