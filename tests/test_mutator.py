"""Tests for fuzzer/mutator.py — no GPU or server required."""

import copy
import random
import pytest

from fuzzer.trace import ActionType, Event, Trace
from fuzzer.mutator import (
    jitter_timing,
    scale_prompt_lengths,
    scale_max_tokens,
    inject_cancel,
    inject_disconnect,
    inject_retry,
    change_prefix_overlap,
    add_burst,
    toggle_stream,
    remove_random_event,
    cuda_graph_thrash,
    mutate_sampling_params,
    splice,
    mutate,
    MUTATIONS,
    MAX_TOKENS_CAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = random.Random(0)


def make_base_trace() -> Trace:
    return Trace(
        seed=1,
        events=[
            Event(action=ActionType.SEND, offset_ms=0,   request_id="r1",
                  prompt_len=128, max_tokens=64, stream=True, prefix_len=0),
            Event(action=ActionType.SEND, offset_ms=200, request_id="r2",
                  prompt_len=256, max_tokens=128, stream=True, prefix_len=0),
            Event(action=ActionType.CANCEL, offset_ms=400, request_id="r1"),
        ],
    )


def make_prefixed_trace() -> Trace:
    """Like make_base_trace() but with a *non-zero* prefix_len.

    This is the state the fuzzer is actually in most of the time: once
    change_prefix_overlap has run, every SEND carries prefix_len > 0.  The
    zero-prefix base trace makes the trace-validity invariant vacuous.
    """
    return Trace(
        seed=1,
        events=[
            Event(action=ActionType.SEND, offset_ms=0,   request_id="r1",
                  prompt_len=128, max_tokens=64, stream=True, prefix_len=64),
            Event(action=ActionType.SEND, offset_ms=200, request_id="r2",
                  prompt_len=256, max_tokens=128, stream=True, prefix_len=64),
            Event(action=ActionType.CANCEL, offset_ms=400, request_id="r1"),
        ],
    )


def fresh_rng() -> random.Random:
    return random.Random(42)


def assert_trace_is_valid(t: Trace) -> None:
    """Structural post-condition every mutator must preserve.

    prefix_len is the number of *shared* leading tokens; the request must keep
    at least one unique suffix token, so 0 <= prefix_len < prompt_len.  A trace
    violating this is sent to the server anyway and silently becomes a
    different request than the one recorded.
    """
    for ev in t.events:
        if ev.action != ActionType.SEND:
            continue
        if ev.prefix_len is None or ev.prompt_len is None:
            continue
        assert 0 <= ev.prefix_len < ev.prompt_len, (
            f"{ev.request_id}: prefix_len={ev.prefix_len} "
            f"not in [0, prompt_len={ev.prompt_len})"
        )


# ---------------------------------------------------------------------------
# General invariants
# ---------------------------------------------------------------------------

# Mutators that rewrite prompt_len without rescaling prefix_len.  These are
# PRODUCTION BUGS in fuzzer/core/mutator.py, not test problems:
#   * scale_prompt_lengths (core/mutator.py:35-45) multiplies prompt_len by
#     0.25/0.5/2.0/4.0 and never touches prefix_len, so any down-scale leaves
#     prefix_len >= prompt_len.
#   * cuda_graph_thrash rewrites prompt_len to a CUDA-graph bucket
#     (16/32/64/128 or 1024+) and likewise never touches prefix_len.
# strict=True so that fixing either one turns this test red and the entry can
# be deleted rather than silently rotting.
_PREFIX_RESCALE_BUGS = {
    "scale_prompt_lengths": "rescales prompt_len but not prefix_len",
    "cuda_graph_thrash":    "rebuckets prompt_len but not prefix_len",
}


def _mutation_params():
    out = []
    for fn in MUTATIONS:
        marks = []
        reason = _PREFIX_RESCALE_BUGS.get(fn.__name__)
        if reason:
            marks.append(pytest.mark.xfail(strict=True, reason=reason))
        out.append(pytest.param(fn, marks=marks, id=fn.__name__))
    return out


class TestMutationInvariants:
    """Every mutator must leave its input alone, return a fresh Trace, and
    produce a structurally valid trace."""

    @pytest.mark.parametrize("fn", _mutation_params())
    def test_mutation_invariants(self, fn):
        original = make_prefixed_trace()
        before   = copy.deepcopy(original)

        result = fn(original, fresh_rng())

        # 1. the input trace is untouched (deep equality, not just length/seed)
        assert original == before, f"{fn.__name__} mutated its input in place"
        # 2. a distinct object came back
        assert result is not original
        # 3. that object is a usable Trace
        assert isinstance(result, Trace)
        assert isinstance(result.events, list)
        # 4. and it is structurally valid (see D5)
        assert_trace_is_valid(result)


def test_mutations_registry_is_exactly_this_list():
    """One exact-list assert replaces nine per-mutator `in MUTATIONS` tests.

    Strictly stronger: it also catches additions, removals and reordering
    (mutate() indexes into this list, so order is part of the contract)."""
    assert [fn.__name__ for fn in MUTATIONS] == [
        "jitter_timing",
        "scale_prompt_lengths",
        "scale_max_tokens",
        "inject_cancel",
        "inject_disconnect",
        "inject_retry",
        "change_prefix_overlap",
        "add_burst",
        "toggle_stream",
        "remove_random_event",
        "stagger_arrivals",
        "burst_idle_burst",
        "cuda_graph_thrash",
        "mutate_sampling_params",
        "inject_chat_turn",
        "inject_unicode_stress",
        "inject_multi_system_message",
        "inject_rtl_system_message",
        "retry_storm",
        "inject_mixed_endpoints",
        "inject_stop_sequences",
        "inject_max_prefix",
        "memory_pressure",
        "chunk_boundary_align",
        "kv_block_boundary",
        "min_tokens_conflict",
        "compose_mask_restrictors",
        "invert_priority_order",
        "inject_prompt_embeds",
        "malformed_constraint",
        "inject_cache_reset",
        "inject_structured_output",
    ]


# ---------------------------------------------------------------------------
# jitter_timing
# ---------------------------------------------------------------------------

class TestJitterTiming:
    def test_all_offsets_non_negative(self):
        t = make_base_trace()
        result = jitter_timing(t, fresh_rng())
        assert all(e.offset_ms >= 0 for e in result.events)

    def test_offsets_change(self):
        # With a fixed seed, at least some offsets should differ from 0
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=1000, request_id="r1",
                  prompt_len=128, max_tokens=64)
        ])
        result = jitter_timing(t, fresh_rng())
        # offset was 1000, jitter is ±20% so range is [800, 1200]
        assert 800 <= result.events[0].offset_ms <= 1200

    def test_zero_offset_stays_zero(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=32)
        ])
        result = jitter_timing(t, fresh_rng())
        assert result.events[0].offset_ms == 0


# ---------------------------------------------------------------------------
# scale_prompt_lengths
# ---------------------------------------------------------------------------

class TestScalePromptLengths:
    def test_prompt_len_at_least_one(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=1, max_tokens=64)
        ])
        # Even scaling down from 1 should stay >= 1
        for _ in range(20):
            result = scale_prompt_lengths(t, random.Random())
            assert result.events[0].prompt_len >= 1

    def test_non_send_events_unaffected(self):
        t = make_base_trace()
        result = scale_prompt_lengths(t, fresh_rng())
        cancel = next(e for e in result.events if e.action == ActionType.CANCEL)
        assert cancel.prompt_len is None


# ---------------------------------------------------------------------------
# scale_max_tokens
# ---------------------------------------------------------------------------

class TestScaleMaxTokensCap:
    def test_cap_enforced_on_large_seed(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=512)
        ])
        results = [scale_max_tokens(t, random.Random(s)) for s in range(40)]
        assert all(ev.max_tokens <= MAX_TOKENS_CAP
                   for result in results
                   for ev in result.events
                   if ev.action == ActionType.SEND)

    def test_cap_does_not_affect_small_values(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=16)
        ])
        for _ in range(20):
            result = scale_max_tokens(t, random.Random())
            for ev in result.events:
                if ev.action == ActionType.SEND:
                    assert 1 <= ev.max_tokens <= MAX_TOKENS_CAP

    def test_floor_still_one(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=1)
        ])
        for _ in range(20):
            result = scale_max_tokens(t, random.Random())
            assert result.events[0].max_tokens >= 1


# ---------------------------------------------------------------------------
# inject_cancel
# ---------------------------------------------------------------------------

class TestInjectCancel:
    def test_adds_one_cancel_event(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=128, max_tokens=64)
        ])
        result = inject_cancel(t, fresh_rng())
        cancels = [e for e in result.events if e.action == ActionType.CANCEL]
        assert len(cancels) == 1

    def test_cancel_targets_existing_send(self):
        t = make_base_trace()
        result = inject_cancel(t, fresh_rng())
        send_ids   = {e.request_id for e in t.events if e.action == ActionType.SEND}
        cancel_ids = {e.request_id for e in result.events if e.action == ActionType.CANCEL}
        # All cancel IDs in the result must come from an existing SEND
        new_cancel_ids = cancel_ids - {
            e.request_id for e in t.events if e.action == ActionType.CANCEL
        }
        assert new_cancel_ids.issubset(send_ids)

    def test_cancel_offset_after_send(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=100, request_id="r1",
                  prompt_len=128, max_tokens=64)
        ])
        result = inject_cancel(t, fresh_rng())
        cancel = next(e for e in result.events if e.action == ActionType.CANCEL)
        assert cancel.offset_ms > 100

    def test_no_sends_returns_unchanged(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.WAIT, offset_ms=0, duration_ms=100)
        ])
        result = inject_cancel(t, fresh_rng())
        assert len(result.events) == len(t.events)


# ---------------------------------------------------------------------------
# inject_disconnect
# ---------------------------------------------------------------------------

class TestInjectDisconnect:
    def test_cancel_becomes_disconnect(self):
        t = make_base_trace()  # already has a CANCEL
        result = inject_disconnect(t, fresh_rng())
        disconnects = [e for e in result.events if e.action == ActionType.DISCONNECT]
        assert len(disconnects) >= 1

    def test_no_cancels_falls_back_to_inject_cancel(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=128, max_tokens=64)
        ])
        result = inject_disconnect(t, fresh_rng())
        # Should have added either a CANCEL or DISCONNECT
        lifecycle = [e for e in result.events
                     if e.action in (ActionType.CANCEL, ActionType.DISCONNECT)]
        assert len(lifecycle) >= 1


# ---------------------------------------------------------------------------
# change_prefix_overlap
# ---------------------------------------------------------------------------

class TestChangePrefixOverlap:
    def test_all_sends_get_same_raw_prefix(self):
        # All sends should receive the same capped prefix_len value
        t = make_base_trace()
        result = change_prefix_overlap(t, fresh_rng())
        sends = [e for e in result.events if e.action == ActionType.SEND]
        prefix_lens = {e.prefix_len for e in sends}
        assert len(prefix_lens) == 1  # all the same

    def test_prefix_len_less_than_prompt_len(self):
        t = make_base_trace()
        for _ in range(30):
            result = change_prefix_overlap(t, random.Random())
            for e in result.events:
                if e.action == ActionType.SEND:
                    assert e.prefix_len < (e.prompt_len or 1)

    def test_prefix_len_non_negative(self):
        t = make_base_trace()
        for _ in range(30):
            result = change_prefix_overlap(t, random.Random())
            for e in result.events:
                if e.action == ActionType.SEND:
                    assert e.prefix_len >= 0


# ---------------------------------------------------------------------------
# inject_retry
# ---------------------------------------------------------------------------

class TestInjectRetry:
    def test_adds_retry_send_after_cancel(self):
        t = make_base_trace()  # has a CANCEL for r1
        result = inject_retry(t, fresh_rng())
        retry_sends = [e for e in result.events
                       if e.action == ActionType.SEND and e.request_id.endswith("_retry")]
        assert len(retry_sends) >= 1

    def test_retry_offset_after_cancel(self):
        t = make_base_trace()
        result = inject_retry(t, fresh_rng())
        cancel = next(e for e in t.events if e.action == ActionType.CANCEL)
        retry  = next(e for e in result.events
                      if e.action == ActionType.SEND and e.request_id.endswith("_retry"))
        assert retry.offset_ms > cancel.offset_ms

    def test_retry_copies_original_prompt_len(self):
        t = make_base_trace()
        result = inject_retry(t, fresh_rng())
        original_send = next(e for e in t.events
                             if e.action == ActionType.SEND and e.request_id == "r1")
        retry = next(e for e in result.events
                     if e.action == ActionType.SEND and e.request_id == "r1_retry")
        assert retry.prompt_len == original_send.prompt_len

    def test_retry_copies_max_tokens(self):
        t = make_base_trace()
        result = inject_retry(t, fresh_rng())
        original_send = next(e for e in t.events
                             if e.action == ActionType.SEND and e.request_id == "r1")
        retry = next(e for e in result.events
                     if e.action == ActionType.SEND and e.request_id == "r1_retry")
        assert retry.max_tokens == original_send.max_tokens

    def test_no_cancel_falls_back_to_inject_cancel(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=128, max_tokens=64)
        ])
        result = inject_retry(t, fresh_rng())
        lifecycle = [e for e in result.events
                     if e.action in (ActionType.CANCEL, ActionType.SEND)]
        assert len(lifecycle) >= 2  # original send + cancel added by fallback


# ---------------------------------------------------------------------------
# add_burst
# ---------------------------------------------------------------------------

class TestAddBurst:
    def test_doubles_send_count(self):
        t = make_base_trace()
        orig_sends = len([e for e in t.events if e.action == ActionType.SEND])
        result = add_burst(t, fresh_rng())
        new_sends = len([e for e in result.events if e.action == ActionType.SEND])
        assert new_sends == orig_sends * 2

    def test_burst_request_ids_have_suffix(self):
        t = make_base_trace()
        result = add_burst(t, fresh_rng())
        burst_ids = [e.request_id for e in result.events
                     if e.action == ActionType.SEND and e.request_id.endswith("_b")]
        assert len(burst_ids) > 0


# ---------------------------------------------------------------------------
# toggle_stream
# ---------------------------------------------------------------------------

class TestToggleStream:
    def test_stream_is_flipped(self):
        t = make_base_trace()
        orig_streams = [e.stream for e in t.events if e.action == ActionType.SEND]
        result = toggle_stream(t, fresh_rng())
        new_streams = [e.stream for e in result.events if e.action == ActionType.SEND]
        assert new_streams == [not s for s in orig_streams]

    def test_double_toggle_restores_original(self):
        t = make_base_trace()
        result = toggle_stream(toggle_stream(t, fresh_rng()), fresh_rng())
        orig = [e.stream for e in t.events if e.action == ActionType.SEND]
        back = [e.stream for e in result.events if e.action == ActionType.SEND]
        assert orig == back


# ---------------------------------------------------------------------------
# remove_random_event
# ---------------------------------------------------------------------------

class TestRemoveRandomEvent:
    def test_removes_exactly_one_event(self):
        t = make_base_trace()
        result = remove_random_event(t, fresh_rng())
        assert len(result.events) == len(t.events) - 1

    def test_single_event_trace_unchanged(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=32)
        ])
        result = remove_random_event(t, fresh_rng())
        assert len(result.events) == 1


# ---------------------------------------------------------------------------
# mutate() dispatcher
# ---------------------------------------------------------------------------

class TestMutate:
    def test_does_not_mutate_original(self):
        t = make_base_trace()
        original_seed = t.seed
        mutate(t, fresh_rng())
        assert t.seed == original_seed


# ---------------------------------------------------------------------------
# cuda_graph_thrash
# ---------------------------------------------------------------------------

class TestCudaGraphThrash:
    SMALL_BUCKETS = {16, 32, 64, 128}
    LARGE_BUCKETS = {1024, 2048, 4096, 8192}

    def _make_multi_send(self, n: int) -> Trace:
        return Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=i * 100,
                  request_id=f"r{i}", prompt_len=256, max_tokens=64)
            for i in range(n)
        ])

    def test_prompt_lens_alternate_small_large(self):
        # Swept over seeds: this subsumes the old "all lens are from a known
        # bucket" test, which asserted only set membership.
        for seed in range(10):
            t = self._make_multi_send(6)
            result = cuda_graph_thrash(t, random.Random(seed))
            sends = sorted(
                [ev for ev in result.events if ev.action == ActionType.SEND],
                key=lambda e: e.offset_ms,
            )
            for i, ev in enumerate(sends):
                if i % 2 == 0:
                    assert ev.prompt_len in self.SMALL_BUCKETS
                else:
                    assert ev.prompt_len in self.LARGE_BUCKETS

    def test_single_send_gets_small_bucket(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=256, max_tokens=64)
        ])
        result = cuda_graph_thrash(t, fresh_rng())
        sends = [ev for ev in result.events if ev.action == ActionType.SEND]
        assert sends[0].prompt_len in self.SMALL_BUCKETS

    def test_non_send_events_untouched(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=256, max_tokens=64),
            Event(action=ActionType.CANCEL, offset_ms=100, request_id="r1"),
        ])
        result = cuda_graph_thrash(t, fresh_rng())
        cancel = next(ev for ev in result.events if ev.action == ActionType.CANCEL)
        assert cancel.prompt_len is None


# ---------------------------------------------------------------------------
# mutate_sampling_params
# ---------------------------------------------------------------------------

class TestMutateSamplingParams:
    def test_sets_temperature_on_sends(self):
        t = make_base_trace()
        result = mutate_sampling_params(t, fresh_rng())
        sends = [e for e in result.events if e.action == ActionType.SEND]
        # At least one SEND should have temperature set
        assert any(e.temperature is not None for e in sends)

    def test_no_sends_returns_unchanged_length(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.WAIT, offset_ms=0, duration_ms=100)
        ])
        result = mutate_sampling_params(t, fresh_rng())
        assert len(result.events) == len(t.events)

    def test_n_completions_none_when_one(self):
        # Run many times; when n=1 is sampled, n_completions must be None
        t = make_base_trace()
        for seed in range(30):
            result = mutate_sampling_params(t, random.Random(seed))
            for ev in result.events:
                if ev.action == ActionType.SEND:
                    assert ev.n_completions != 1

    def test_temperature_in_valid_range(self):
        t = make_base_trace()
        for seed in range(30):
            result = mutate_sampling_params(t, random.Random(seed))
            for ev in result.events:
                if ev.action == ActionType.SEND and ev.temperature is not None:
                    assert 0.0 <= ev.temperature <= 2.0


# ---------------------------------------------------------------------------
# splice
# ---------------------------------------------------------------------------

def make_trace_b() -> Trace:
    """A second distinct trace for crossover tests."""
    return Trace(
        seed=2,
        events=[
            Event(action=ActionType.SEND, offset_ms=0,   request_id="b1",
                  prompt_len=64, max_tokens=32, stream=False, prefix_len=0),
            Event(action=ActionType.SEND, offset_ms=300, request_id="b2",
                  prompt_len=128, max_tokens=64, stream=True, prefix_len=0),
            Event(action=ActionType.CANCEL, offset_ms=500, request_id="b2"),
        ],
    )


class TestSplice:
    def test_returns_new_trace(self):
        a = make_base_trace()
        b = make_trace_b()
        result = splice(a, b, fresh_rng())
        assert result is not a
        assert result is not b

    def test_does_not_modify_either_parent(self):
        a = make_base_trace()
        b = make_trace_b()
        orig_a_len = len(a.events)
        orig_b_len = len(b.events)
        splice(a, b, fresh_rng())
        assert len(a.events) == orig_a_len
        assert len(b.events) == orig_b_len

    def test_merged_events_sorted_by_offset(self):
        a = make_base_trace()
        b = make_trace_b()
        result = splice(a, b, fresh_rng())
        offsets = [e.offset_ms for e in result.events]
        assert offsets == sorted(offsets)

    def test_orphaned_cancel_removed(self):
        """A CANCEL whose SEND was in the discarded half must not appear."""
        a = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="a1",
                  prompt_len=64, max_tokens=32),
        ])
        # b has a cancel for b1; if b1's SEND is not in the second half, cancel is orphaned
        b = Trace(seed=2, events=[
            Event(action=ActionType.SEND,   offset_ms=0,   request_id="b1",
                  prompt_len=64, max_tokens=32),
            Event(action=ActionType.CANCEL, offset_ms=100, request_id="b1"),
        ])
        result = splice(a, b, fresh_rng())
        send_ids = {e.request_id for e in result.events if e.action == ActionType.SEND}
        cancel_ids = {e.request_id for e in result.events if e.action == ActionType.CANCEL}
        # Every cancel must have a matching SEND
        assert cancel_ids.issubset(send_ids)

    def test_splice_same_trace_with_itself_is_valid(self):
        """Splicing a trace with itself should not crash."""
        t = make_base_trace()
        result = splice(t, t, fresh_rng())
        assert isinstance(result.events, list)


# ---------------------------------------------------------------------------
# mutate() with other (crossover)
# ---------------------------------------------------------------------------

class TestMutateWithOther:
    def test_other_provided_produces_valid_trace(self):
        a = make_base_trace()
        b = make_trace_b()
        result, ops = mutate(a, fresh_rng(), n=2, other=b)
        assert isinstance(result, Trace)
        assert isinstance(result.events, list)
        assert len(ops) == 2

    def test_does_not_mutate_original_with_other(self):
        a = make_base_trace()
        b = make_trace_b()
        orig_seed = a.seed
        mutate(a, fresh_rng(), n=2, other=b)
        assert a.seed == orig_seed


# ---------------------------------------------------------------------------
# mutate() operator-name reporting
# ---------------------------------------------------------------------------

class TestMutateOperatorNames:
    """The verbose logging in runner._emit_verbose_log relies on mutate()
    returning the list of operator names that were applied. These tests pin
    the contract so that contract changes will surface as test failures."""

    def test_returns_tuple_of_trace_and_str_list(self):
        t = make_base_trace()
        ret = mutate(t, fresh_rng(), n=2)
        assert isinstance(ret, tuple) and len(ret) == 2
        result, ops = ret
        assert isinstance(result, Trace)
        assert isinstance(ops, list)
        assert all(isinstance(op, str) for op in ops)

    def test_ops_length_matches_n(self):
        t = make_base_trace()
        for n in (1, 2, 5):
            _, ops = mutate(t, fresh_rng(), n=n)
            assert len(ops) == n

    def test_ops_drawn_from_provided_registry(self):
        # When a single-operator registry is provided, every applied op must
        # be that operator's __name__.
        t = make_base_trace()
        single = [jitter_timing]
        _, ops = mutate(t, fresh_rng(), n=4, mutations=single)
        assert ops == ["jitter_timing"] * 4

    def test_splice_op_name_when_other_supplied(self):
        # Force the splice branch by setting RNG so splice_prob (0.20) is
        # always taken: if rng.random() returns < splice_prob, splice fires.
        # Use a deterministic RNG and single-op registry to make the splice
        # branch the only possible outcome at low pressure.
        a = make_base_trace()
        b = make_trace_b()

        class _AlwaysSplice(random.Random):
            # rng.random() < 0.20 (splice_prob with pressure=0) → True for 0.0
            def random(self):  # type: ignore[override]
                return 0.0
            def choices(self, *a, **kw):  # not used in splice branch
                return [None]
            def randint(self, lo, hi):
                return 1234

        _, ops = mutate(a, _AlwaysSplice(), n=1, other=b, other_pressure=0.0)
        assert ops == ["splice"]

    def test_directed_splice_op_name_under_high_pressure(self):
        a = make_base_trace()
        b = make_trace_b()

        class _AlwaysSplice(random.Random):
            def random(self):  # type: ignore[override]
                return 0.0
            def choices(self, *a, **kw):
                return [None]
            def randint(self, lo, hi):
                return 1234

        _, ops = mutate(a, _AlwaysSplice(), n=1, other=b, other_pressure=5.0)
        assert ops == ["directed_splice"]


# ---------------------------------------------------------------------------
# inject_unicode_stress
# ---------------------------------------------------------------------------

from fuzzer.mutator import (
    inject_unicode_stress,
    inject_multi_system_message,
    inject_rtl_system_message,
    retry_storm,
    inject_mixed_endpoints,
)


_VALID_STRESS_TYPES = {"cjk", "emoji", "combining", "rtl_mixed"}


def make_chat_trace() -> Trace:
    """A trace with a chat SEND event."""
    return Trace(
        seed=3,
        events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="c1",
                  max_tokens=64, stream=True,
                  messages=[{"role": "system", "content_len": 32},
                             {"role": "user", "content_len": 64}]),
        ],
    )


class TestInjectUnicodeStress:
    def test_sets_unicode_stress_on_sends(self):
        t = make_base_trace()
        result = inject_unicode_stress(t, fresh_rng())
        sends = [ev for ev in result.events if ev.action == ActionType.SEND]
        assert any(ev.unicode_stress is not None for ev in sends)

    def test_unicode_stress_valid_type(self):
        t = make_base_trace()
        for seed in range(20):
            result = inject_unicode_stress(t, random.Random(seed))
            for ev in result.events:
                if ev.action == ActionType.SEND and ev.unicode_stress is not None:
                    assert ev.unicode_stress in _VALID_STRESS_TYPES

    def test_no_sends_returns_unchanged(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.WAIT, offset_ms=0, duration_ms=100)
        ])
        result = inject_unicode_stress(t, fresh_rng())
        assert len(result.events) == len(t.events)


# ---------------------------------------------------------------------------
# inject_multi_system_message
# ---------------------------------------------------------------------------

class TestInjectMultiSystemMessage:
    def test_adds_system_message_to_chat_event(self):
        t = make_chat_trace()
        orig_msg_count = len(t.events[0].messages)
        result = inject_multi_system_message(t, fresh_rng())
        chat_sends = [ev for ev in result.events
                      if ev.action == ActionType.SEND and ev.messages is not None]
        assert len(chat_sends[0].messages) == orig_msg_count + 1
        assert chat_sends[0].messages[0]["role"] == "system"

    def test_fallback_when_no_chat_events(self):
        t = make_base_trace()  # only completions events (no messages)
        result = inject_multi_system_message(t, fresh_rng())
        # Should not crash — falls back to inject_chat_turn or returns something valid
        assert isinstance(result, Trace)


# ---------------------------------------------------------------------------
# Phase 3: retry_storm
# ---------------------------------------------------------------------------

class TestRetryStorm:
    def test_adds_cancel_and_storm_send(self):
        t = make_base_trace()
        result = retry_storm(t, fresh_rng())
        cancels = [ev for ev in result.events if ev.action == ActionType.CANCEL]
        storm_sends = [ev for ev in result.events
                       if ev.action == ActionType.SEND and "_storm" in ev.request_id]
        assert len(cancels) >= 1
        assert len(storm_sends) >= 1

    def test_prefix_len_within_bounds(self):
        t = make_base_trace()
        for seed in range(20):
            result = retry_storm(t, random.Random(seed))
            for ev in result.events:
                if ev.action == ActionType.SEND and "_storm" in ev.request_id:
                    if ev.prefix_len is not None and ev.prompt_len is not None:
                        assert ev.prefix_len >= 0
                        assert ev.prefix_len < ev.prompt_len


# ---------------------------------------------------------------------------
# Phase 3: inject_mixed_endpoints
# ---------------------------------------------------------------------------

class TestInjectMixedEndpoints:
    def _all_completions_trace(self) -> Trace:
        return make_base_trace()  # only completions events

    def _all_chat_trace(self) -> Trace:
        return Trace(seed=4, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="c1",
                  max_tokens=64, stream=True,
                  messages=[{"role": "system", "content_len": 32},
                             {"role": "user", "content_len": 64}]),
            Event(action=ActionType.SEND, offset_ms=100, request_id="c2",
                  max_tokens=32, stream=True,
                  messages=[{"role": "user", "content_len": 48}]),
        ])

    def test_all_completions_gets_chat_event(self):
        t = self._all_completions_trace()
        result = inject_mixed_endpoints(t, fresh_rng())
        chat_sends = [ev for ev in result.events
                      if ev.action == ActionType.SEND and ev.messages is not None]
        assert len(chat_sends) >= 1

    def test_all_chat_gets_completions_event(self):
        t = self._all_chat_trace()
        result = inject_mixed_endpoints(t, fresh_rng())
        comp_sends = [ev for ev in result.events
                      if ev.action == ActionType.SEND and ev.messages is None]
        assert len(comp_sends) >= 1

    def test_already_mixed_unchanged_structure(self):
        # A trace already having both chat and completions events → returned as-is
        t = Trace(seed=5, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=32),
            Event(action=ActionType.SEND, offset_ms=100, request_id="c1",
                  max_tokens=32,
                  messages=[{"role": "user", "content_len": 32}]),
        ])
        result = inject_mixed_endpoints(t, fresh_rng())
        assert isinstance(result, Trace)



# ---------------------------------------------------------------------------
# compose_mask_restrictors (mask-composition surface)
# ---------------------------------------------------------------------------

class TestComposeMaskRestrictors:
    """Every draw must produce a *valid* trace that actually composes two or
    more independent ``-inf`` writers, and must never produce a combination the
    server rejects up front — a 400 is not a finding."""

    def _all_draws(self, n=400):
        from fuzzer.adapters.vllm_base.mutators import compose_mask_restrictors
        base = make_prefixed_trace()
        out = []
        for seed in range(n):
            t = compose_mask_restrictors(base, random.Random(seed))
            assert_trace_is_valid(t)
            assert t == compose_mask_restrictors(base, random.Random(seed)), \
                "not deterministic for a fixed rng seed"
            out.append(t)
        return out

    def _touched(self, t):
        from fuzzer.core.types import mask_restrictors_of
        return [ev for ev in t.events
                if ev.action == ActionType.SEND and mask_restrictors_of(ev)]

    def test_every_draw_composes_at_least_two_writers(self):
        from fuzzer.core.types import mask_restrictors_of
        for t in self._all_draws():
            touched = self._touched(t)
            assert len(touched) == 1, "exactly one SEND is restricted"
            assert len(mask_restrictors_of(touched[0])) >= 2

    def test_input_is_untouched_and_result_is_fresh(self):
        from fuzzer.adapters.vllm_base.mutators import compose_mask_restrictors
        base = make_prefixed_trace()
        before = copy.deepcopy(base)
        out = compose_mask_restrictors(base, fresh_rng())
        assert base == before
        assert out is not base

    def test_logprobs_are_always_requested(self):
        """The sentinel predicate reads the sampled token's own logprob and the
        top-k beside it; without them the oracle has nothing to decide on."""
        for t in self._all_draws():
            ev = self._touched(t)[0]
            assert ev.logprobs is not None and ev.logprobs >= 5

    def test_decoding_is_always_greedy(self):
        """The defect is ``argmax`` over an all-``-inf`` row.  Above
        temperature 0 the row goes through softmax instead."""
        for t in self._all_draws():
            assert self._touched(t)[0].temperature == 0.0
            assert t.is_deterministic()

    def test_min_tokens_never_exceeds_max_tokens(self):
        """``_verify_args`` (sampling_params.py:592) rejects that outright."""
        for t in self._all_draws():
            ev = self._touched(t)[0]
            if ev.min_tokens:
                assert ev.max_tokens is not None
                assert ev.min_tokens <= ev.max_tokens

    def test_allowlists_are_never_empty(self):
        """An empty ``allowed_token_ids`` is a 400
        (sampling_params.py:866-871), which is noise rather than a result."""
        for t in self._all_draws():
            ev = self._touched(t)[0]
            if ev.allowed_token_ids is not None:
                assert len(ev.allowed_token_ids) >= 1
                assert all(isinstance(i, int) and i > 0
                           for i in ev.allowed_token_ids)

    def test_token_ids_stay_in_range_of_any_vocabulary(self):
        """No tokenizer is available, so ids must be small enough that no real
        model can call them out of range."""
        for t in self._all_draws():
            ev = self._touched(t)[0]
            ids = list(ev.allowed_token_ids or []) + list(ev.stop_token_ids or [])
            assert all(1 <= i < 1024 for i in ids), ids

    def test_token_id_zero_is_never_allowed(self):
        """0 is what ``argmax`` returns for an all-``-inf`` row; allowing it
        would let the degenerate token look legal."""
        for t in self._all_draws():
            ev = self._touched(t)[0]
            assert 0 not in (ev.allowed_token_ids or [])

    def test_bad_words_are_non_empty_strings(self):
        """``_verify_args`` (sampling_params.py:633-638) rejects an empty
        bad word."""
        for t in self._all_draws():
            ev = self._touched(t)[0]
            for w in (ev.bad_words or []):
                assert isinstance(w, str) and w

    def test_logit_bias_keys_are_strings_and_values_within_the_clamp(self):
        for t in self._all_draws():
            ev = self._touched(t)[0]
            for k, v in (ev.logit_bias or {}).items():
                assert isinstance(k, str) and k.lstrip("-").isdigit()
                assert -100.0 <= v <= 100.0

    def test_the_provable_recipe_puts_the_allowlist_inside_the_stop_set(self):
        """``allowlist_x_min_tokens`` is the one recipe that is empty by
        construction on every model — and it is what the allowlist-escape
        predicate decides."""
        seen = False
        for t in self._all_draws():
            ev = self._touched(t)[0]
            if ev.stop_token_ids:
                seen = True
                assert ev.allowed_token_ids
                assert set(ev.allowed_token_ids) <= set(ev.stop_token_ids)
                assert ev.min_tokens
        assert seen, "the provable recipe was never drawn"

    def test_all_recipes_are_reachable(self):
        from fuzzer.adapters.vllm_base.mutators import MASK_COMPOSE_RECIPES
        from fuzzer.core.types import mask_restrictors_of
        shapes = set()
        for t in self._all_draws():
            shapes.add(frozenset(mask_restrictors_of(self._touched(t)[0])))
        # one distinct restrictor set per recipe, at minimum
        assert len(shapes) >= len(MASK_COMPOSE_RECIPES)

    def test_prompt_derived_ids_are_process_stable(self):
        """md5, not ``hash()``: PYTHONHASHSEED would otherwise make the same
        trace restrict a different window on every replay."""
        from fuzzer.adapters.vllm_base.mutators import _prompt_derived_ids
        ev = Event(action=ActionType.SEND, offset_ms=0, request_id="stable",
                   prompt_len=32, max_tokens=16)
        assert (_prompt_derived_ids(ev, random.Random(1), 3)
                == _prompt_derived_ids(ev, random.Random(2), 3))

    def test_no_sends_is_a_noop(self):
        from fuzzer.adapters.vllm_base.mutators import compose_mask_restrictors
        t = Trace(seed=1, events=[Event(action=ActionType.WAIT, offset_ms=0,
                                        duration_ms=10)])
        assert compose_mask_restrictors(t, fresh_rng()).events == t.events


# ---------------------------------------------------------------------------
# invert_priority_order (scheduler priority surface)
# ---------------------------------------------------------------------------

def _priority_trace(n: int = 6, spacing: int = 200,
                    with_cancel: bool = False) -> Trace:
    events = [
        Event(action=ActionType.SEND, offset_ms=i * spacing,
              request_id=f"r{i}", prompt_len=256, max_tokens=64,
              stream=True, prefix_len=64)
        for i in range(n)
    ]
    if with_cancel:
        events.append(Event(action=ActionType.CANCEL,
                            offset_ms=n * spacing + 50, request_id="r0"))
    return Trace(seed=5, events=events)


class TestInvertPriorityOrder:
    def _mutate(self, trace=None, seed=42):
        from fuzzer.adapters.vllm_base.mutators import invert_priority_order
        return invert_priority_order(trace or _priority_trace(),
                                     random.Random(seed))

    def test_is_deterministic_for_a_given_rng_seed(self):
        base = _priority_trace()
        for seed in (0, 1, 7, 99):
            assert self._mutate(base, seed) == self._mutate(base, seed)

    def test_priority_is_inverted_against_arrival_order(self):
        """The point of the mutator.  If priority ASCENDED with arrival, the
        correct schedule and the FCFS schedule would coincide and the workload
        would be blind to a priority-queue bug -- which is exactly the state
        the 401-entry / zero-finding campaign was in."""
        out = self._mutate()
        sends = sorted(out.send_events(), key=lambda e: e.offset_ms)
        pri = [e.priority for e in sends]
        assert all(p is not None for p in pri)
        assert pri == sorted(pri, reverse=True), pri
        assert len(set(pri)) == len(pri), "ties would not force a reordering"
        # lower value = higher priority, so the LAST arrival must win
        assert sends[-1].priority == 0
        assert sends[0].priority == len(sends) - 1

    def test_arrivals_are_packed_so_priority_can_bind(self):
        """Priority only arbitrates among requests waiting simultaneously.  A
        trace whose requests arrive after the queue has drained is served in
        arrival order whatever its priorities say."""
        from fuzzer.adapters.vllm_base.mutators import (
            _PRIORITY_ARRIVAL_WINDOW_MS)
        out = self._mutate(_priority_trace(n=6, spacing=500))
        offs = [e.offset_ms for e in out.send_events()]
        assert max(offs) - min(offs) <= _PRIORITY_ARRIVAL_WINDOW_MS

    def test_packing_never_moves_a_send_later(self):
        """The packing is what keeps CANCEL/DISCONNECT ordering safe: a SEND
        that only ever moves earlier cannot overtake an abort that referenced
        it."""
        base = _priority_trace(n=6, spacing=500)
        out = self._mutate(base)
        for before, after in zip(base.send_events(), out.send_events()):
            assert after.offset_ms <= before.offset_ms

    def test_packing_is_skipped_when_the_trace_has_lifecycle_events(self):
        """A trace carrying a CANCEL has load-bearing timing this mutator has
        no view of, so it gets the priorities but not the repacking."""
        base = _priority_trace(n=4, spacing=500, with_cancel=True)
        out = self._mutate(base)
        assert [e.offset_ms for e in out.events] == \
               [e.offset_ms for e in base.events]
        assert all(e.priority is not None for e in out.send_events())

    def test_already_packed_arrivals_are_left_alone(self):
        base = _priority_trace(n=4, spacing=5)
        out = self._mutate(base)
        assert [e.offset_ms for e in out.send_events()] == \
               [e.offset_ms for e in base.send_events()]

    def test_amplification_lands_on_the_highest_priority_requests(self):
        """The scheduler is supposed to admit these first, so mis-ordering
        them is wrong by the largest wall-clock margin."""
        seen_amplified = False
        for seed in range(24):
            out = self._mutate(seed=seed)
            sends = sorted(out.send_events(), key=lambda e: e.priority)
            amped = [e for e in sends
                     if e.prompt_logprobs is not None
                     or (e.n_completions or 1) > 1]
            assert amped, f"seed {seed} amplified nothing"
            seen_amplified = True
            # contiguous prefix of the priority-sorted list == the best ones
            assert amped == sends[:len(amped)], seed
        assert seen_amplified

    def test_amplifiers_never_build_a_request_vllm_would_reject(self):
        """Two hard 400s in vLLM's request models:
        ``prompt_logprobs`` with ``stream=True``
        (chat_completion/protocol.py:805-810), and ``n > 1`` under greedy
        sampling (sampling_params.py:640-644).  A mutator that emits either
        produces a rejected request instead of a schedule, which is how a
        whole surface goes quiet without anyone noticing."""
        for seed in range(40):
            for ev in self._mutate(seed=seed).send_events():
                if ev.prompt_logprobs is not None:
                    assert ev.prompt_logprobs > 0, seed
                    assert ev.stream is False, seed
                if (ev.n_completions or 1) > 1:
                    assert ev.temperature is not None and ev.temperature > 0, seed
                    assert ev.seed is not None, seed

    def test_both_amplification_cells_are_reachable(self):
        from fuzzer.adapters.vllm_base.mutators import _PRIORITY_N
        cells = set()
        for seed in range(40):
            out = self._mutate(seed=seed)
            if any(e.prompt_logprobs is not None for e in out.send_events()):
                cells.add("prompt_logprobs")
            if any((e.n_completions or 1) > 1 for e in out.send_events()):
                cells.add("n")
        assert cells == {"prompt_logprobs", "n"}
        assert all(n > 1 for n in _PRIORITY_N)

    def test_the_prompt_logprobs_cell_keeps_the_trace_greedy(self):
        """Half of the traces this mutator makes must stay determinism-probe
        eligible, or the priority surface costs us the determinism oracle."""
        greedy = [self._mutate(seed=s) for s in range(40)]
        assert any(t.is_deterministic() for t in greedy)
        for t in greedy:
            if t.is_deterministic():
                assert all((e.n_completions or 1) == 1
                           for e in t.send_events())

    def test_prompt_len_and_prefix_len_are_untouched(self):
        base = _priority_trace()
        out = self._mutate(base)
        assert [(e.prompt_len, e.prefix_len) for e in out.send_events()] == \
               [(e.prompt_len, e.prefix_len) for e in base.send_events()]
        assert_trace_is_valid(out)

    def test_single_send_is_a_noop(self):
        """One request cannot be scheduled out of order with respect to
        itself, so a priority on it says nothing."""
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r0",
                  prompt_len=32, max_tokens=16, prefix_len=0)])
        out = self._mutate(t)
        assert out.events == t.events
        assert out.send_events()[0].priority is None

    def test_no_sends_is_a_noop(self):
        t = Trace(seed=1, events=[Event(action=ActionType.WAIT, offset_ms=0,
                                        duration_ms=10)])
        assert self._mutate(t).events == t.events

    def test_input_is_not_mutated_in_place(self):
        base = _priority_trace()
        before = copy.deepcopy(base)
        self._mutate(base)
        assert base == before
