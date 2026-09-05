"""Compatibility re-export for the canonical evaluation normalizer."""

from src.evaluation.answer_normalization import *  # noqa: F403
from src.evaluation.answer_normalization import __all__ as _canonical_all

__all__ = _canonical_all
