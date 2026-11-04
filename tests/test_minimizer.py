"""Tests for fuzzer/minimizer.py — no GPU or server required."""

import pytest
from fuzzer.trace import ActionType, Event, Trace
from fuzzer.minimizer import minimize, minimize_with_oracle


def make_cancel_event(offset_ms: int, tag: str) -> Event:
    return Event(
        action=ActionType.CANCEL,
        offset_ms=offset_ms,
        request_id=tag,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(offset_ms: int, tag: str) -> Event:
    """Make a SEND event. We use request_id as an opaque tag for testing."""
    return Event(
        action=ActionType.SEND,
        offset_ms=offset_ms,
        request_id=tag,
        prompt_len=64,
        max_tokens=32,
    )


def make_trace(*tags: str) -> Trace:
    events = [make_event(i * 100, tag) for i, tag in enumerate(tags)]
    return Trace(seed=1, events=events)


# ---------------------------------------------------------------------------
# minimize()
# ---------------------------------------------------------------------------

class TestMinimize:
    async def test_empty_trace_returns_immediately(self):
        t = Trace(seed=1)
        result = await minimize(t, predicate=lambda _: (_ for _ in ()).throw(
            AssertionError("predicate should not be called")))
        assert result.events == []

    async def test_single_matching_event_returns_it(self):
        t = make_trace("A")

        async def predicate(trace: Trace) -> bool:
            return any(e.request_id == "A" for e in trace.events)

        result = await minimize(t, predicate)
        assert len(result.events) == 1
        assert result.events[0].request_id == "A"

    async def test_minimizes_to_single_required_event(self):
        # Only event "C" is needed to trigger the predicate
        t = make_trace("A", "B", "C", "D", "E")

        async def predicate(trace: Trace) -> bool:
            return any(e.request_id == "C" for e in trace.events)

        result = await minimize(t, predicate)
        ids = [e.request_id for e in result.events]
        assert "C" in ids
        # `<= len(t.events)` could not fail; the point of the test is that the
        # trace actually shrank.  (minimize() is a greedy pass, not full ddmin,
        # so it is not guaranteed to reach exactly ["C"].)
        assert len(result.events) < len(t.events)

    async def test_result_is_subset_of_original(self):
        t = make_trace("A", "B", "C", "D")
        original_ids = {e.request_id for e in t.events}

        async def predicate(trace: Trace) -> bool:
            return len(trace.events) >= 2

        result = await minimize(t, predicate)
        for e in result.events:
            assert e.request_id in original_ids

    async def test_predicate_never_true_returns_original_length(self):
        """If predicate never fires, minimize can't shrink anything."""
        t = make_trace("A", "B", "C")

        async def predicate(trace: Trace) -> bool:
            return False

        result = await minimize(t, predicate)
        # Repaired: this asserted `len(result.events) >= 0`, which cannot fail.
        # The behaviour that matters is that an unsatisfiable predicate leaves
        # the trace completely untouched, order included -- a minimizer that
        # returned an empty trace here would silently destroy every finding
        # whose predicate has gone stale.
        assert [e.request_id for e in result.events] == ["A", "B", "C"]

    async def test_all_events_required_not_reduced(self):
        """Predicate requires ALL events to be present."""
        t = make_trace("A", "B", "C")

        async def predicate(trace: Trace) -> bool:
            ids = {e.request_id for e in trace.events}
            return {"A", "B", "C"}.issubset(ids)

        result = await minimize(t, predicate)
        assert len(result.events) == 3

    async def test_preserves_seed(self):
        t = make_trace("A", "B", "C")

        async def predicate(trace: Trace) -> bool:
            return True

        result = await minimize(t, predicate)
        assert result.seed == t.seed

    async def test_result_events_maintain_relative_order(self):
        """The minimized trace should preserve event ordering."""
        t = make_trace("A", "B", "C", "D")

        async def predicate(trace: Trace) -> bool:
            ids = [e.request_id for e in trace.events]
            # Predicate requires B before D
            return "B" in ids and "D" in ids and ids.index("B") < ids.index("D")

        result = await minimize(t, predicate)
        ids = [e.request_id for e in result.events]
        assert "B" in ids and "D" in ids
        assert ids.index("B") < ids.index("D")


# ---------------------------------------------------------------------------
# minimize() — min_length and required_action_types
# ---------------------------------------------------------------------------

class TestMinimizeConstraints:
    async def test_min_length_prevents_over_reduction(self):
        """Predicate accepts single event, but min_length=2 prevents it."""
        t = make_trace("A", "B", "C")

        async def predicate(trace: Trace) -> bool:
            return True  # always fires

        result = await minimize(t, predicate, min_length=2)
        assert len(result.events) >= 2

    async def test_min_length_1_allows_single_event(self):
        """With min_length=1, reduction to 1 event is allowed."""
        t = make_trace("A", "B", "C")

        async def predicate(trace: Trace) -> bool:
            return any(e.request_id == "A" for e in trace.events)

        result = await minimize(t, predicate, min_length=1)
        assert len(result.events) == 1

    async def test_required_cancel_preserved(self):
        """A CANCEL event must survive minimization if the original had one."""
        send   = make_event(0, "s1")
        cancel = make_cancel_event(100, "s1")
        extra  = make_event(200, "extra")

        t = Trace(seed=1, events=[send, cancel, extra])

        async def predicate(trace: Trace) -> bool:
            return len(trace.events) >= 1  # fires for any non-empty trace

        result = await minimize(t, predicate,
                                min_length=1,
                                required_action_types={ActionType.CANCEL})
        action_types = {e.action for e in result.events}
        assert ActionType.CANCEL in action_types

    async def test_required_action_not_in_original_not_enforced(self):
        """If original has no DISCONNECT, constraint is not applied."""
        t = make_trace("A", "B", "C")  # only SEND events

        async def predicate(trace: Trace) -> bool:
            return any(e.request_id == "A" for e in trace.events)

        # DISCONNECT required but original has none — should not block reduction
        result = await minimize(t, predicate,
                                min_length=1,
                                required_action_types={ActionType.DISCONNECT})
        assert any(e.request_id == "A" for e in result.events)


# ---------------------------------------------------------------------------
# minimize_with_oracle()
# ---------------------------------------------------------------------------

class TestMinimizeWithOracle:
    async def test_wrapper_reduces_correctly(self):
        t = make_trace("A", "B", "C", "D")

        async def run_fn(trace):
            return {"ids": [e.request_id for e in trace.events]}

        def oracle_fn(run_output) -> bool:
            return "C" in run_output["ids"]

        result = await minimize_with_oracle(t, run_fn, oracle_fn)
        ids = [e.request_id for e in result.events]
        assert "C" in ids

    async def test_empty_trace_predicate_returns_false(self):
        """minimize_with_oracle's inner predicate short-circuits on empty."""
        t = Trace(seed=1)
        call_count = {"n": 0}

        async def run_fn(trace):
            call_count["n"] += 1
            return {}

        def oracle_fn(_) -> bool:
            return True

        await minimize_with_oracle(t, run_fn, oracle_fn)
        assert call_count["n"] == 0   # predicate never called for empty trace
