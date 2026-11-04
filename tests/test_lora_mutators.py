"""Tests for LoRA-specific mutation strategies."""

from __future__ import annotations

import copy
import random

import pytest
from fuzzer.adapters.vllm_lora.mutators import (
    switch_adapter,
    concurrent_adapter_burst,
    same_prefix_different_adapters,
    adapter_thrash,
    mixed_base_lora,
    inject_same_prompt_pair,
    LORA_MUTATIONS,
)
from fuzzer.trace import Trace, Event, ActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trace(events: list[Event], seed: int = 1) -> Trace:
    return Trace(seed=seed, events=events)


def send(request_id: str, adapter_name: str | None = None,
         prompt_len: int = 64, prefix_len: int = 0,
         offset_ms: int = 0) -> Event:
    return Event(
        action=ActionType.SEND,
        offset_ms=offset_ms,
        request_id=request_id,
        prompt_len=prompt_len,
        max_tokens=32,
        prefix_len=prefix_len,
        adapter_name=adapter_name,
    )


def send_chat(request_id: str, adapter_name: str | None = None,
              offset_ms: int = 0) -> Event:
    """Chat completions SEND event (messages set, prompt_len/prefix_len None)."""
    return Event(
        action=ActionType.SEND,
        offset_ms=offset_ms,
        request_id=request_id,
        messages=[{"role": "user", "content_len": 32}],
        max_tokens=32,
        adapter_name=adapter_name,
    )


def cancel(request_id: str, offset_ms: int = 10) -> Event:
    return Event(
        action=ActionType.CANCEL,
        offset_ms=offset_ms,
        request_id=request_id,
    )


RNG = random.Random(42)


# ---------------------------------------------------------------------------
# switch_adapter
# ---------------------------------------------------------------------------

def test_switch_adapter_noop_fewer_than_two_adapters():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_a")])
    result = switch_adapter(trace, RNG)
    for ev in result.events:
        if ev.action == ActionType.SEND:
            assert ev.adapter_name == "lora_a"


def test_switch_adapter_noop_no_adapters():
    trace = make_trace([send("r1", None), send("r2", None)])
    result = switch_adapter(trace, RNG)
    # unchanged
    for ev in result.events:
        assert ev.adapter_name is None


def test_switch_adapter_reassigns_from_present_adapters():
    trace = make_trace([
        send("r1", "lora_a"),
        send("r2", "lora_b"),
        send("r3", "lora_a"),
    ])
    rng = random.Random(0)
    result = switch_adapter(trace, rng)
    allowed = {"lora_a", "lora_b"}
    for ev in result.events:
        if ev.action == ActionType.SEND and ev.adapter_name is not None:
            assert ev.adapter_name in allowed


def test_switch_adapter_does_not_mutate_original():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_b")])
    original_names = [ev.adapter_name for ev in trace.events]
    switch_adapter(trace, random.Random(7))
    assert [ev.adapter_name for ev in trace.events] == original_names


def test_switch_adapter_preserves_non_send_events():
    trace = make_trace([
        send("r1", "lora_a"),
        cancel("r1"),
        send("r2", "lora_b"),
    ])
    result = switch_adapter(trace, RNG)
    cancel_events = [ev for ev in result.events if ev.action == ActionType.CANCEL]
    assert len(cancel_events) == 1


def test_switch_adapter_skips_none_adapter_sends():
    """SEND events with adapter_name=None should not be reassigned."""
    trace = make_trace([
        send("r1", "lora_a"),
        send("r2", "lora_b"),
        send("r3", None),   # base model request — must stay None
    ])
    result = switch_adapter(trace, RNG)
    none_sends = [ev for ev in result.events
                  if ev.action == ActionType.SEND and ev.request_id == "r3"]
    assert none_sends[0].adapter_name is None


# ---------------------------------------------------------------------------
# concurrent_adapter_burst
# ---------------------------------------------------------------------------

def test_concurrent_adapter_burst_adds_events():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_b")])
    original_len = len(trace.events)
    result = concurrent_adapter_burst(trace, random.Random(1))
    assert len(result.events) > original_len


def test_concurrent_adapter_burst_size_4_to_8():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_b")])
    original_len = len(trace.events)
    result = concurrent_adapter_burst(trace, random.Random(5))
    added = len(result.events) - original_len
    assert 4 <= added <= 8


def test_concurrent_adapter_burst_noop_single_adapter():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_a")])
    original_len = len(trace.events)
    result = concurrent_adapter_burst(trace, RNG)
    assert len(result.events) == original_len


def test_concurrent_adapter_burst_noop_no_sends():
    trace = make_trace([cancel("r1"), cancel("r2")])
    original_len = len(trace.events)
    result = concurrent_adapter_burst(trace, RNG)
    assert len(result.events) == original_len


def test_concurrent_adapter_burst_request_ids_unique():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_b")])
    result = concurrent_adapter_burst(trace, random.Random(9))
    ids = [ev.request_id for ev in result.events if ev.action == ActionType.SEND]
    # burst events have ids like burst_0..burst_N; originals are r1, r2
    burst_ids = [i for i in ids if i.startswith("burst_")]
    assert len(burst_ids) == len(set(burst_ids))


def test_concurrent_adapter_burst_events_sorted_by_offset():
    trace = make_trace([
        send("r1", "lora_a", offset_ms=100),
        send("r2", "lora_b", offset_ms=200),
    ])
    result = concurrent_adapter_burst(trace, random.Random(3))
    offsets = [ev.offset_ms for ev in result.events]
    assert offsets == sorted(offsets)


def test_concurrent_adapter_burst_alternates_adapters():
    """Burst events should use both adapters (alternating)."""
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_b")])
    result = concurrent_adapter_burst(trace, random.Random(2))
    burst_adapters = {
        ev.adapter_name
        for ev in result.events
        if ev.action == ActionType.SEND and ev.request_id.startswith("burst_")
    }
    assert len(burst_adapters) >= 2


def test_concurrent_adapter_burst_prefix_len_positive():
    trace = make_trace([
        send("r1", "lora_a", prompt_len=32, prefix_len=0),
        send("r2", "lora_b", prompt_len=32, prefix_len=0),
    ])
    result = concurrent_adapter_burst(trace, random.Random(11))
    burst_events = [ev for ev in result.events
                    if ev.action == ActionType.SEND
                    and ev.request_id.startswith("burst_")]
    for ev in burst_events:
        assert ev.prefix_len >= 1


# ---------------------------------------------------------------------------
# same_prefix_different_adapters
# ---------------------------------------------------------------------------

def test_same_prefix_different_adapters_sets_shared_prefix():
    trace = make_trace([
        send("r1", "lora_a", prompt_len=32, prefix_len=5),
        send("r2", "lora_b", prompt_len=32, prefix_len=3),
    ])
    result = same_prefix_different_adapters(trace, random.Random(0))
    sends = {ev.request_id: ev for ev in result.events if ev.action == ActionType.SEND}
    assert sends["r1"].prefix_len == sends["r2"].prefix_len


def test_same_prefix_different_adapters_noop_single_adapter():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_a")])
    result = same_prefix_different_adapters(trace, RNG)
    # If noop, events unchanged
    assert len(result.events) == len(trace.events)


def test_same_prefix_different_adapters_prompt_len_gte_prefix_plus_one():
    trace = make_trace([
        send("r1", "lora_a", prompt_len=16, prefix_len=0),
        send("r2", "lora_b", prompt_len=24, prefix_len=0),
    ])
    result = same_prefix_different_adapters(trace, random.Random(1))
    for ev in result.events:
        if ev.action == ActionType.SEND:
            assert (ev.prompt_len or 0) > (ev.prefix_len or 0)


def test_same_prefix_different_adapters_does_not_mutate_original():
    trace = make_trace([
        send("r1", "lora_a", prompt_len=32, prefix_len=5),
        send("r2", "lora_b", prompt_len=32, prefix_len=3),
    ])
    original = [(ev.prefix_len, ev.prompt_len) for ev in trace.events]
    same_prefix_different_adapters(trace, random.Random(0))
    assert [(ev.prefix_len, ev.prompt_len) for ev in trace.events] == original


def test_same_prefix_different_adapters_prefix_at_least_one():
    trace = make_trace([
        send("r1", "lora_a", prompt_len=8, prefix_len=0),
        send("r2", "lora_b", prompt_len=8, prefix_len=0),
    ])
    result = same_prefix_different_adapters(trace, random.Random(5))
    for ev in result.events:
        if ev.action == ActionType.SEND and ev.adapter_name is not None:
            assert (ev.prefix_len or 0) >= 1


# ---------------------------------------------------------------------------
# LORA_MUTATIONS list
# ---------------------------------------------------------------------------

def test_lora_mutations_registry_is_exactly_this_list():
    """One exact-list assert replaces three membership/callable tests.

    Strictly stronger: the test it replaces was named `..._contains_all_five`
    and still passed after inject_same_prompt_pair became the sixth entry."""
    assert [m.__name__ for m in LORA_MUTATIONS] == [
        "switch_adapter",
        "concurrent_adapter_burst",
        "same_prefix_different_adapters",
        "adapter_thrash",
        "mixed_base_lora",
        "inject_same_prompt_pair",
    ]


def test_lora_mutations_return_trace():
    trace = make_trace([send("r1", "lora_a"), send("r2", "lora_b")])
    for mut in LORA_MUTATIONS:
        result = mut(trace, random.Random(0))
        assert isinstance(result, Trace)


# ---------------------------------------------------------------------------
# adapter_thrash
# ---------------------------------------------------------------------------

class TestAdapterThrash:
    def test_adds_alternating_requests(self):
        trace = make_trace([send("r1", "lora_a", offset_ms=0), send("r2", "lora_b", offset_ms=10)])
        result = adapter_thrash(trace, random.Random(0))
        thrash_events = [ev for ev in result.events
                         if ev.action == ActionType.SEND and ev.request_id.startswith("thrash_")]
        assert len(thrash_events) >= 6

    def test_alternates_adapter_names(self):
        trace = make_trace([send("r1", "lora_a", offset_ms=0), send("r2", "lora_b", offset_ms=10)])
        result = adapter_thrash(trace, random.Random(0))
        thrash_events = sorted(
            [ev for ev in result.events if ev.request_id.startswith("thrash_")],
            key=lambda e: e.request_id,
        )
        adapters_used = {ev.adapter_name for ev in thrash_events}
        assert adapters_used == {"lora_a", "lora_b"}

    def test_thrash_events_have_zero_prefix(self):
        trace = make_trace([send("r1", "lora_a", offset_ms=0), send("r2", "lora_b", offset_ms=10)])
        result = adapter_thrash(trace, random.Random(0))
        for ev in result.events:
            if ev.request_id.startswith("thrash_"):
                assert (ev.prefix_len or 0) == 0

    def test_noop_with_single_adapter(self):
        trace = make_trace([send("r1", "lora_a"), send("r2", "lora_a")])
        result = adapter_thrash(trace, random.Random(0))
        assert result is trace  # no change when only one adapter


# ---------------------------------------------------------------------------
# mixed_base_lora
# ---------------------------------------------------------------------------

class TestMixedBaseLora:
    def test_some_adapter_names_set_to_none(self):
        events = [send(f"r{i}", "lora_a") for i in range(6)]
        trace = make_trace(events)
        result = mixed_base_lora(trace, random.Random(0))
        none_count = sum(
            1 for ev in result.events
            if ev.action == ActionType.SEND and ev.adapter_name is None
        )
        assert none_count >= 1

    def test_at_least_one_lora_request_remains(self):
        events = [send(f"r{i}", "lora_a") for i in range(6)]
        trace = make_trace(events)
        result = mixed_base_lora(trace, random.Random(0))
        lora_count = sum(
            1 for ev in result.events
            if ev.action == ActionType.SEND and ev.adapter_name is not None
        )
        assert lora_count >= 1

    def test_demotes_roughly_one_third(self):
        events = [send(f"r{i}", "lora_a") for i in range(9)]
        trace = make_trace(events)
        result = mixed_base_lora(trace, random.Random(42))
        none_count = sum(
            1 for ev in result.events
            if ev.action == ActionType.SEND and ev.adapter_name is None
        )
        # max(1, 9 // 3) = 3
        assert none_count == 3

    def test_noop_with_no_adapters(self):
        trace = make_trace([send("r1", None), send("r2", None)])
        result = mixed_base_lora(trace, random.Random(0))
        assert result is trace


# ---------------------------------------------------------------------------
# inject_same_prompt_pair
# ---------------------------------------------------------------------------

class TestInjectSamePromptPair:

    def test_noop_fewer_than_two_adapters(self):
        """Only one adapter present — no-op."""
        trace = make_trace([send("r1", "lora_a"), send("r2", "lora_a")])
        result = inject_same_prompt_pair(trace, random.Random(0))
        assert result is trace

    def test_noop_all_chat_events(self):
        """All sends are chat events — completions subset is empty — must not crash."""
        trace = make_trace([
            send_chat("r1", "lora_a"),
            send_chat("r2", "lora_b"),
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        assert result is trace

    def test_noop_only_one_adapter_has_completions(self):
        """lora_b only has chat sends — completions by_adapter has one key — must not crash."""
        trace = make_trace([
            send("r1", "lora_a"),
            send_chat("r2", "lora_b"),
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        assert result is trace

    def test_assigns_same_family_id_to_two_adapters(self):
        """Normal case: two completions sends with different adapters get same family_id."""
        trace = make_trace([
            send("r1", "lora_a", prompt_len=64, prefix_len=8),
            send("r2", "lora_b", prompt_len=64, prefix_len=8),
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        sends = [ev for ev in result.events if ev.action == ActionType.SEND]
        family_ids = {ev.prompt_family_id for ev in sends if ev.prompt_family_id}
        assert len(family_ids) == 1, "both sends should share one family_id"
        adapters = {ev.adapter_name for ev in sends if ev.prompt_family_id}
        assert adapters == {"lora_a", "lora_b"}

    def test_aligns_prompt_len_and_prefix_len(self):
        """Shared pair must have identical prompt_len and prefix_len."""
        trace = make_trace([
            send("r1", "lora_a", prompt_len=32, prefix_len=4),
            send("r2", "lora_b", prompt_len=64, prefix_len=8),
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        paired = [ev for ev in result.events
                  if ev.action == ActionType.SEND and ev.prompt_family_id]
        assert len(paired) == 2
        assert paired[0].prompt_len == paired[1].prompt_len
        assert paired[0].prefix_len == paired[1].prefix_len

    def test_chat_events_not_stamped(self):
        """Mixed trace: chat events must not receive a prompt_family_id."""
        trace = make_trace([
            send("r1", "lora_a"),
            send("r2", "lora_b"),
            send_chat("r3", "lora_a"),
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        chat_ev = next(ev for ev in result.events if getattr(ev, "messages", None))
        assert chat_ev.prompt_family_id is None

    def test_sets_logprobs_5_on_paired_events(self):
        """inject_same_prompt_pair must set logprobs=5 on both stamped events."""
        trace = make_trace([
            send("r1", "lora_a", prompt_len=64),
            send("r2", "lora_b", prompt_len=64),
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        paired = [ev for ev in result.events
                  if ev.action == ActionType.SEND and ev.prompt_family_id]
        assert len(paired) == 2
        for ev in paired:
            assert ev.logprobs == 5, (
                f"expected logprobs=5 on paired event {ev.request_id}, got {ev.logprobs}"
            )

    def test_logprobs_not_set_on_unpaired_events(self):
        """Events that are NOT stamped with a family_id keep their original logprobs."""
        trace = make_trace([
            send("r1", "lora_a", prompt_len=64),
            send("r2", "lora_b", prompt_len=64),
            send("r3", "lora_a", prompt_len=64),  # extra — not always the chosen one
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        unpaired = [ev for ev in result.events
                    if ev.action == ActionType.SEND and not ev.prompt_family_id]
        # At least one event should have no family_id; its logprobs must be unchanged
        for ev in unpaired:
            assert ev.logprobs is None

    def test_logprobs_not_clobbered_on_chat_events(self):
        """Chat events that bypass the mutation keep logprobs=None."""
        trace = make_trace([
            send("r1", "lora_a"),
            send("r2", "lora_b"),
            send_chat("r3", "lora_a"),
        ])
        result = inject_same_prompt_pair(trace, random.Random(0))
        chat_ev = next(ev for ev in result.events if getattr(ev, "messages", None))
        assert chat_ev.logprobs is None
