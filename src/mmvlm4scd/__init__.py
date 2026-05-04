"""Mmvlm4SCD: Multimodal Model for Sickle Cell Disease.

Public re-exports kept intentionally small. Heavy submodules are imported
lazily by callers.
"""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("mmvlm4scd")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "Sambou Kone"

__all__ = ["__version__", "__author__"]
