"""Tests for speculative decoding mutators."""
import random
import pytest

from fuzzer.trace import Trace, Event, ActionType
from fuzzer.adapters.vllm_spec.mutators import (
    cancel_during_spec,
    spec_shared_prefix_race,
)


def _send(offset_ms=0, max_tokens=128, stream=True, prefix_len=64) -> Event:
    return Event(
        action    = ActionType.SEND,
        offset_ms = offset_ms,
        prompt_len = 128,
        max_tokens = max_tokens,
        stream     = stream,
        prefix_len = prefix_len,
        temperature = 0.0,
    )


def _trace(*events) -> Trace:
    return Trace(seed=0, events=list(events))


class TestCancelDuringSpec:
    def test_adds_cancel_event(self):
        t = _trace(_send(offset_ms=0, max_tokens=128, stream=True))
        rng = random.Random(42)
        result = cancel_during_spec(t, rng)
        cancels = [e for e in result.events if e.action == ActionType.CANCEL]
        assert len(cancels) == 1

    def test_cancel_targets_streaming_long_request(self):
        short = _send(max_tokens=16, stream=True)   # too short
        long  = _send(max_tokens=128, stream=True)  # eligible
        t = _trace(short, long)
        rng = random.Random(42)
        result = cancel_during_spec(t, rng)
        cancels = [e for e in result.events if e.action == ActionType.CANCEL]
        assert cancels[0].request_id == long.request_id

    @pytest.mark.parametrize("base_offset", [0, 250, 4000])
    def test_cancel_offset_in_flat_50_350ms_window(self, base_offset):
        """The cancel lands 50–350 ms after its own SEND, not at a fraction of
        estimated wall time (mutators.py:41-45).  The window is relative to the
        send's offset, so it must track a non-zero base offset."""
        ev = _send(offset_ms=base_offset, max_tokens=100, stream=True)
        t = _trace(ev)
        rng = random.Random(42)
        result = cancel_during_spec(t, rng)
        cancel = next(e for e in result.events if e.action == ActionType.CANCEL)
        assert base_offset + 50 <= cancel.offset_ms <= base_offset + 350

    def test_cancel_offset_window_holds_across_seeds(self):
        """randint(50, 350) is inclusive at both ends; sweep seeds so a
        drifted bound cannot hide behind one lucky draw."""
        for seed in range(200):
            t = _trace(_send(offset_ms=0, max_tokens=100, stream=True))
            result = cancel_during_spec(t, random.Random(seed))
            cancel = next(e for e in result.events if e.action == ActionType.CANCEL)
            assert 50 <= cancel.offset_ms <= 350, f"seed {seed}"

    def test_noop_when_no_eligible_events(self):
        t = _trace(_send(max_tokens=32, stream=True))   # too short
        rng = random.Random(42)
        result = cancel_during_spec(t, rng)
        cancels = [e for e in result.events if e.action == ActionType.CANCEL]
        assert len(cancels) == 0

    def test_noop_for_non_streaming(self):
        t = _trace(_send(max_tokens=128, stream=False))
        rng = random.Random(42)
        result = cancel_during_spec(t, rng)
        cancels = [e for e in result.events if e.action == ActionType.CANCEL]
        assert len(cancels) == 0

    def test_events_sorted_by_offset(self):
        t = _trace(_send(offset_ms=0, max_tokens=128, stream=True))
        rng = random.Random(42)
        result = cancel_during_spec(t, rng)
        offsets = [e.offset_ms for e in result.events]
        assert offsets == sorted(offsets)

    def test_original_trace_not_mutated(self):
        t = _trace(_send(max_tokens=128, stream=True))
        original_len = len(t.events)
        rng = random.Random(42)
        cancel_during_spec(t, rng)
        assert len(t.events) == original_len


class TestSpecSharedPrefixRace:
    def test_adds_2_to_4_events(self):
        t = _trace(_send(offset_ms=100))
        rng = random.Random(0)
        result = spec_shared_prefix_race(t, rng)
        new_sends = [e for e in result.events
                     if e.action == ActionType.SEND and e.origin == "mutated"]
        assert 2 <= len(new_sends) <= 4

    def test_all_new_events_share_same_offset(self):
        t = _trace(_send(offset_ms=200))
        rng = random.Random(1)
        result = spec_shared_prefix_race(t, rng)
        new_sends = [e for e in result.events
                     if e.action == ActionType.SEND and e.origin == "mutated"]
        assert len({e.offset_ms for e in new_sends}) == 1

    def test_prefix_len_at_least_256(self):
        t = _trace(_send(offset_ms=0))
        rng = random.Random(2)
        result = spec_shared_prefix_race(t, rng)
        new_sends = [e for e in result.events
                     if e.action == ActionType.SEND and e.origin == "mutated"]
        for ev in new_sends:
            assert ev.prefix_len >= 256

    def test_distinct_prompt_family_ids(self):
        t = _trace(_send(offset_ms=0))
        rng = random.Random(3)
        result = spec_shared_prefix_race(t, rng)
        new_sends = [e for e in result.events
                     if e.action == ActionType.SEND and e.origin == "mutated"]
        fam_ids = [e.prompt_family_id for e in new_sends]
        assert len(set(fam_ids)) == len(fam_ids)

    def test_events_sorted_by_offset(self):
        t = _trace(_send(offset_ms=500))
        rng = random.Random(4)
        result = spec_shared_prefix_race(t, rng)
        offsets = [e.offset_ms for e in result.events]
        assert offsets == sorted(offsets)

    def test_original_trace_not_mutated(self):
        t = _trace(_send(offset_ms=0))
        original_len = len(t.events)
        rng = random.Random(5)
        spec_shared_prefix_race(t, rng)
        assert len(t.events) == original_len
