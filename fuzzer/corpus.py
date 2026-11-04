"""
Corpus shim: re-exports from core.corpus.

The Corpus class now lives in fuzzer/core/corpus.py.
This module preserves existing import paths.
"""

from __future__ import annotations

from .core.corpus import (
    Corpus,
    CorpusEntry,
    _compute_pressure,
    _NOVELTY_SCORE_FLOOR,
    _ERROR_NOVELTY_SCORE,
    _PREEMPTION_NOVELTY_SCORE,
    _PREEMPTION_COUNT_CAP,
    _SWAP_NOVELTY_SCORE,
    _SWAP_COUNT_CAP,
    _KV_NOVELTY_SCORE,
    _RUN_COUNT_DECAY,
)

__all__ = [
    "Corpus",
    "CorpusEntry",
    "_compute_pressure",
    "_NOVELTY_SCORE_FLOOR",
    "_ERROR_NOVELTY_SCORE",
    "_PREEMPTION_NOVELTY_SCORE",
    "_PREEMPTION_COUNT_CAP",
    "_SWAP_NOVELTY_SCORE",
    "_SWAP_COUNT_CAP",
    "_KV_NOVELTY_SCORE",
    "_RUN_COUNT_DECAY",
]
