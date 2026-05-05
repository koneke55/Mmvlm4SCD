"""Base classes for real-data loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..harmonization import PatientRecord, validate_cohort


class DataAccessError(RuntimeError):
    """Raised when a credentialed loader is invoked without
    a fulfilled DUA / IRB / credential.

    The loader does not attempt to download anything; the user is
    expected to provision access through the official portal first
    (see ``docs/data_access_checklist.md``).
    """


@dataclass
class LoaderConfig:
    root: Path
    project_salt: str = ""
    cohort_split: str = "train"
    max_patients: int | None = None


class BaseLoader(ABC):
    """Common contract for every Mmvlm4SCD real-data loader."""

    #: Source key, must match ``ALLOWED_SOURCES`` in
    #: ``harmonization.py``.
    source: str = "synthetic"

    #: One of ``open``, ``registered``, ``dua-required``.
    access: str = "open"

    def __init__(self, root: str | Path, project_salt: str = "",
                 cohort_split: str = "train",
                 max_patients: int | None = None):
        self.cfg = LoaderConfig(root=Path(root),
                                project_salt=project_salt,
                                cohort_split=cohort_split,
                                max_patients=max_patients)

    # ------------------------------------------------------------------
    # Subclasses implement ``_load_records``; ``load`` adds the safety
    # net (validation, PHI-check, optional truncation).
    # ------------------------------------------------------------------

    @abstractmethod
    def _load_records(self) -> List[PatientRecord]:
        """Return the unvalidated list of records."""

    def load(self) -> List[PatientRecord]:
        """Validated, length-truncated list of records."""
        recs = self._load_records()
        if self.cfg.max_patients is not None:
            recs = recs[: self.cfg.max_patients]
        validate_cohort(recs)
        return recs


class StubLoader(BaseLoader):
    """Concrete loader that always raises ``DataAccessError``.

    Used as a placeholder until a real loader is implemented for
    a given source.  Subclasses must set ``source`` and ``access``.
    """

    todo: str = "implement loader"

    def _load_records(self) -> List[PatientRecord]:
        raise DataAccessError(
            f"{self.__class__.__name__}: data access not configured. "
            f"{self.todo}. See docs/data_access_checklist.md.")
