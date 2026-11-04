"""
Oracles: determine whether a scenario run constitutes a bug finding.

This module is a shim — the Oracle class lives in fuzzer.core.oracles.
All symbols are re-exported here to keep existing import paths working.
"""

from __future__ import annotations

from .core.oracles import Oracle
from .core.types import (
    BugClass,
    OracleResult,
    _FATAL_PATTERNS,
    _CORRUPT_RE,
    MIN_LATENCY_SAMPLE,
    _LATENCY_SPIKE_MIN_MS,
)

__all__ = [
    "Oracle",
    "BugClass",
    "OracleResult",
    "_FATAL_PATTERNS",
    "_CORRUPT_RE",
    "MIN_LATENCY_SAMPLE",
    "_LATENCY_SPIKE_MIN_MS",
]
