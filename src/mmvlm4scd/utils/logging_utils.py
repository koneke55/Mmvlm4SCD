"""Tiny logger so callers don't need to import logging directly."""

from __future__ import annotations

import logging


def get_logger(name: str = "mmvlm4scd", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(level)
    return logger
