"""Tests for ``fuzzer/adapters/sglang/strategies.py``.

Phase 3 of the adapter / strategy split refactor.

Two concerns, expressed as exact assertions:

* ``TestFingerprint24845`` — the structural predicate that defines the
  #24845 trigger shape (concurrent shared-prefix sends + abort event).
  Necessary-but-not-sufficient; we test both the matches and the
  non-matches so a future refactor that loosens or tightens the
  predicate fails loudly.

* ``TestDefaultExplorationStrategyScoreMultiplier`` — the runtime
  integration: a Strategy whose ``score_multiplier`` returns the
  #24845 penalty (0.5) on matching corpus entries and 1.0 elsewhere.
  This is the value the Phase 1 corpus hook actually applies, so this
  is the load-bearing wiring between the Strategy layer and the
  corpus selection layer.
"""

from types import SimpleNamespace

import pytest

from fuzzer.adapters.sglang.strategies import (
    DefaultExplorationStrategy,
    fingerprint_sglang_24845,
)
from fuzzer.adapters.sglang.capabilities import SGLANG_24845
from fuzzer.trace import ActionType, Event, Trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send(request_id, family=None, offset=0):
    return Event(
        action=ActionType.SEND,
        offset_ms=offset,
        request_id=request_id,
        prompt_len=64,
        max_tokens=16,
        stream=True,
        temperature=0.0,
        prompt_family_id=family,
    )


def _cancel(request_id, offset=10):
    return Event(action=ActionType.CANCEL, offset_ms=offset, request_id=request_id)


def _disconnect(request_id, offset=10):
    return Event(action=ActionType.DISCONNECT, offset_ms=offset, request_id=request_id)


def _entry(trace):
    """Mimic a CorpusEntry — only ``.trace`` is read by score_multiplier."""
    return SimpleNamespace(trace=trace)


# ---------------------------------------------------------------------------
# Fingerprint predicate
# ---------------------------------------------------------------------------

class TestFingerprint24845:
    """Necessary condition: every #24845 instance matches.
    Not-sufficient condition: matching traces may or may not actually
    trigger the bug — the predicate is intentionally permissive enough
    that a corpus penalty steers exploration without auto-classifying
    findings (see the v3 doc review)."""

    def test_shared_family_plus_cancel_matches(self):
        trace = Trace(seed=0, events=[
            _send("r1", family="fam0", offset=0),
            _send("r2", family="fam0", offset=10),
            _cancel("r1", offset=20),
        ])
        assert fingerprint_sglang_24845(trace) is True

    def test_shared_family_plus_disconnect_matches(self):
        """DISCONNECT is part of the trigger surface alongside CANCEL —
        the may_18 ablation showed disconnect behaves equivalently."""
        trace = Trace(seed=0, events=[
            _send("r1", family="fam0", offset=0),
            _send("r2", family="fam0", offset=10),
            _disconnect("r1", offset=20),
        ])
        assert fingerprint_sglang_24845(trace) is True

    def test_no_shared_family_does_not_match(self):
        """Each SEND in its own family → no shared-prefix attractor."""
        trace = Trace(seed=0, events=[
            _send("r1", family="famA", offset=0),
            _send("r2", family="famB", offset=10),
            _cancel("r1", offset=20),
        ])
        assert fingerprint_sglang_24845(trace) is False

    def test_no_abort_event_does_not_match(self):
        """Shared-prefix family but no cancel/disconnect — concurrent
        normal traffic, not the abort-driven trigger."""
        trace = Trace(seed=0, events=[
            _send("r1", family="fam0", offset=0),
            _send("r2", family="fam0", offset=10),
        ])
        assert fingerprint_sglang_24845(trace) is False

    def test_single_send_in_family_does_not_match(self):
        """Only 1 SEND sharing a family → no concurrent prefix usage."""
        trace = Trace(seed=0, events=[
            _send("r1", family="fam0", offset=0),
            _send("r2", family=None,   offset=10),
            _cancel("r1", offset=20),
        ])
        assert fingerprint_sglang_24845(trace) is False

    def test_sends_with_no_family_id_do_not_match(self):
        """SEND events without prompt_family_id are not part of any
        shared-prefix family."""
        trace = Trace(seed=0, events=[
            _send("r1", family=None, offset=0),
            _send("r2", family=None, offset=10),
            _cancel("r1", offset=20),
        ])
        assert fingerprint_sglang_24845(trace) is False


# ---------------------------------------------------------------------------
# Strategy.score_multiplier — Phase 1 ↔ Phase 3 integration
# ---------------------------------------------------------------------------

class TestDefaultExplorationStrategyScoreMultiplier:
    """The runtime wiring: ``Strategy.score_multiplier`` returns the
    #24845 penalty (0.5) for matching corpus entries and 1.0 otherwise.
    The runner passes this callable to ``Corpus.select()`` via the
    Phase 1 hook; this is the load-bearing integration between Strategy
    and corpus selection."""

    def test_attractor_trace_returns_known_bug_penalty(self):
        """Trace matching the #24845 fingerprint → multiplier == 0.5,
        which is the value held in ``fingerprint_penalties[SGLANG_24845]``."""
        trace = Trace(seed=0, events=[
            _send("r1", family="fam0", offset=0),
            _send("r2", family="fam0", offset=10),
            _cancel("r1", offset=20),
        ])
        s = DefaultExplorationStrategy()
        assert s.score_multiplier(_entry(trace)) == 0.5

    def test_attractor_trace_via_disconnect_returns_penalty(self):
        """The disconnect form of the trigger must also down-weight —
        otherwise disconnect-driven corpus entries would dominate."""
        trace = Trace(seed=0, events=[
            _send("r1", family="fam0", offset=0),
            _send("r2", family="fam0", offset=10),
            _disconnect("r1", offset=20),
        ])
        s = DefaultExplorationStrategy()
        assert s.score_multiplier(_entry(trace)) == 0.5

    def test_non_attractor_trace_returns_one(self):
        trace = Trace(seed=0, events=[
            _send("r1", family="famA", offset=0),
            _send("r2", family="famB", offset=10),
        ])
        s = DefaultExplorationStrategy()
        assert s.score_multiplier(_entry(trace)) == 1.0

    def test_entry_with_none_trace_returns_one(self):
        """Defensive: a malformed entry without a trace shouldn't crash;
        the strategy degrades to no-bias rather than raising."""
        s = DefaultExplorationStrategy()
        assert s.score_multiplier(_entry(None)) == 1.0

    def test_none_entry_returns_one(self):
        """Defensive: ``None`` entry returns 1.0 rather than raising."""
        s = DefaultExplorationStrategy()
        assert s.score_multiplier(None) == 1.0

    def test_penalty_is_sourced_from_fingerprint_penalties_dict(self):
        """The 0.5 multiplier must come from ``fingerprint_penalties``,
        not a hardcoded constant. This makes the penalty tunable per
        campaign without touching the Strategy class."""
        s = DefaultExplorationStrategy()
        assert s.fingerprint_penalties[SGLANG_24845] == 0.5

        # Tune the penalty; the multiplier reflects the change.
        s.fingerprint_penalties[SGLANG_24845] = 0.1
        trace = Trace(seed=0, events=[
            _send("r1", family="fam0", offset=0),
            _send("r2", family="fam0", offset=10),
            _cancel("r1", offset=20),
        ])
        assert s.score_multiplier(_entry(trace)) == 0.1
