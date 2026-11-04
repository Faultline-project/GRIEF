"""
LoRA-specific mutation strategies.

These mutations engineer traces that stress cross-adapter KV isolation:
concurrent requests with different adapters sharing the same prefix.
"""

from __future__ import annotations

import copy
import hashlib
import random


def _stable_family_id(prompt_len: int, prefix_len: int) -> str:
    """Deterministic family id from shape — process-stable (no Python hash())."""
    key = f"fam:{prompt_len}:{prefix_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]

from ...trace import Trace, Event, ActionType


def _lora_adapters(trace: Trace) -> list[str]:
    """Return the distinct non-None adapter_name values present in the trace."""
    return list({
        ev.adapter_name
        for ev in trace.events
        if ev.action == ActionType.SEND and ev.adapter_name is not None
    })


def switch_adapter(trace: Trace, rng: random.Random) -> Trace:
    """
    Randomly reassign adapter_name on each SEND event, choosing from the
    adapters already present in the trace. Explores different adapter
    combinations without changing timing or prompt structure.

    No-op if the trace has fewer than 2 distinct adapters.
    """
    adapters = _lora_adapters(trace)
    if len(adapters) < 2:
        return trace

    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND and ev.adapter_name is not None:
            ev.adapter_name = rng.choice(adapters)
    return t


def concurrent_adapter_burst(trace: Trace, rng: random.Random) -> Trace:
    """
    Insert a burst of 4–8 SEND events at the same offset_ms, alternating
    between the adapters present in the trace, all using the same prefix_len.

    Directly targets the KV race: two adapters competing for the same prefix
    cache entry at the same time.

    No-op if the trace has fewer than 2 distinct adapters or no SEND events.
    """
    adapters = _lora_adapters(trace)
    send_events = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if len(adapters) < 2 or not send_events:
        return trace

    t = copy.deepcopy(trace)
    template = rng.choice(send_events)
    burst_size = rng.randint(4, 8)
    # Pick a shared prefix_len that's at least half the prompt length
    shared_prefix = max(1, (template.prefix_len or 0),
                        (template.prompt_len or 16) // 2)
    burst_offset = template.offset_ms

    for i in range(burst_size):
        ev = copy.deepcopy(template)
        ev.request_id   = f"burst_{i}"
        ev.offset_ms    = burst_offset
        ev.prefix_len   = shared_prefix
        ev.adapter_name = adapters[i % len(adapters)]
        ev.origin       = "burst"
        t.events.append(ev)

    t.events.sort(key=lambda e: e.offset_ms)
    return t


def same_prefix_different_adapters(trace: Trace, rng: random.Random) -> Trace:
    """
    Pick two existing SEND events with different adapter_name values and
    force them to share the same prefix_len. Ensures the trace has the
    shared-prefix + different-adapter pattern that the isolation oracle needs.

    No-op if fewer than 2 distinct adapters are present.
    """
    adapters = _lora_adapters(trace)
    if len(adapters) < 2:
        return trace

    t = copy.deepcopy(trace)
    send_events = [ev for ev in t.events if ev.action == ActionType.SEND
                   and ev.adapter_name is not None]

    # Group by adapter
    by_adapter: dict[str, list[Event]] = {}
    for ev in send_events:
        by_adapter.setdefault(ev.adapter_name, []).append(ev)

    # Pick one event from each of two different adapters
    a_name, b_name = rng.sample(list(by_adapter.keys()), 2)
    ev_a = rng.choice(by_adapter[a_name])
    ev_b = rng.choice(by_adapter[b_name])

    # Use the smaller prompt_len as the shared prefix baseline
    min_len = min(ev_a.prompt_len or 16, ev_b.prompt_len or 16)
    shared  = max(1, min_len - 1)   # near-full prefix (prompt_len - 1)

    ev_a.prefix_len = shared
    ev_b.prefix_len = shared
    ev_a.prompt_len = max(ev_a.prompt_len or 16, shared + 1)
    ev_b.prompt_len = max(ev_b.prompt_len or 16, shared + 1)

    return t


def adapter_thrash(trace: Trace, rng: random.Random) -> Trace:
    """Rapid alternating adapter sends at 5–20ms intervals with no shared prefix.

    Generates a tight sequence of N requests alternating between adapters.
    The zero prefix_len isolates adapter switching cost from KV sharing effects,
    targeting adapter load/unload races and stale-pointer conditions.
    """
    adapters = _lora_adapters(trace)
    sends    = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if len(adapters) < 2 or not sends:
        return trace
    t        = copy.deepcopy(trace)
    template = rng.choice(sends)
    n        = rng.randint(6, 12)
    interval = rng.randint(5, 20)
    base_ms  = template.offset_ms
    for i in range(n):
        ev              = copy.deepcopy(template)
        ev.request_id   = f"thrash_{i}"
        ev.offset_ms    = base_ms + i * interval
        ev.adapter_name = adapters[i % len(adapters)]
        ev.prefix_len   = 0
        ev.origin       = "thrash"
        t.events.append(ev)
    t.events.sort(key=lambda e: e.offset_ms)
    return t


def mixed_base_lora(trace: Trace, rng: random.Random) -> Trace:
    """Demote ~1/3 of LoRA requests to base model (adapter_name=None).

    Mixes base-model requests into a LoRA burst to test scheduler handling
    of heterogeneous adapter batches. The demoted requests share the same
    prompt shape as their LoRA siblings, isolating the adapter-presence
    variable from prompt-structure effects.
    """
    adapters = _lora_adapters(trace)
    if not adapters:
        return trace
    t = copy.deepcopy(trace)
    sends = [
        ev for ev in t.events
        if ev.action == ActionType.SEND and ev.adapter_name is not None
    ]
    if not sends:
        return t
    n_demote = max(1, len(sends) // 3)
    for ev in rng.sample(sends, min(n_demote, len(sends))):
        ev.adapter_name = None
    return t


def inject_same_prompt_pair(trace: Trace, rng: random.Random) -> Trace:
    """
    Pick two SEND events with different adapters, align their prompt shape,
    and stamp them with the same prompt_family_id.

    This ensures the LoRA isolation oracle can compare their outputs — without
    a shared prompt_family_id, _build_prompt generates distinct prompts per
    request_id, making output comparison meaningless.

    Strategy:
      - Pick one event from each of two distinct adapters
      - Use the smaller prompt_len as the shared shape
      - Set prefix_len = shared_prompt_len - 1 (near-full prefix for maximum KV sharing)
      - Assign prompt_family_id = stable_hash(prompt_len, prefix_len)

    No-op if fewer than 2 distinct adapters are present.
    """
    adapters = _lora_adapters(trace)
    if len(adapters) < 2:
        return trace

    t = copy.deepcopy(trace)
    # Chat events have non-trivial message structure that isn't normalized by
    # this mutation; including them would manufacture the false-positive where
    # two requests share a prompt_family_id but different message shapes.
    send_events = [ev for ev in t.events if ev.action == ActionType.SEND
                   and ev.adapter_name is not None
                   and ev.messages is None]  # completions-only

    by_adapter: dict[str, list] = {}
    for ev in send_events:
        by_adapter.setdefault(ev.adapter_name, []).append(ev)

    if len(by_adapter) < 2:
        return trace  # no-op: all completions events share one adapter, or none exist

    a_name, b_name = rng.sample(list(by_adapter.keys()), 2)
    ev_a = rng.choice(by_adapter[a_name])
    ev_b = rng.choice(by_adapter[b_name])

    # Align on the smaller prompt_len so both events are syntactically valid
    shared_prompt_len = min(ev_a.prompt_len or 16, ev_b.prompt_len or 16)
    shared_prefix_len = max(1, shared_prompt_len - 1)
    family_id = _stable_family_id(shared_prompt_len, shared_prefix_len)

    ev_a.prompt_len = shared_prompt_len
    ev_b.prompt_len = shared_prompt_len
    ev_a.prefix_len = shared_prefix_len
    ev_b.prefix_len = shared_prefix_len
    ev_a.prompt_family_id = family_id
    ev_b.prompt_family_id = family_id
    # logprobs=5 enables the fast-path tiebreaker in check_cross_adapter_isolation:
    # same text + different logprobs → greedy collision (skip);
    # same text + same logprobs → high-confidence candidate (score 7, queued for probe).
    # 5 values gives a richer distributional view than the minimum of 1.
    ev_a.logprobs = 5
    ev_b.logprobs = 5

    return t


LORA_MUTATIONS: list = [
    switch_adapter,
    concurrent_adapter_burst,
    same_prefix_different_adapters,
    adapter_thrash,
    mixed_base_lora,
    inject_same_prompt_pair,
]


def make_inject_lora_adapters(adapter_names: list) -> object:
    """Return a mutation that bootstraps LoRA exploration from plain traces.

    All other LoRA mutations silently no-op when no adapter_name is present
    in the trace (guard: ``if len(adapters) < 2: return trace``).  This
    mutation stamps adapter names onto events that don't have one, allowing
    root seeds to evolve into LoRA traces through mutation.
    """
    def inject_lora_adapters(trace: Trace, rng: random.Random) -> Trace:
        sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
        if not sends:
            return trace
        t = copy.deepcopy(trace)
        sends = [ev for ev in t.events if ev.action == ActionType.SEND]
        if rng.random() < 0.30:
            name = rng.choice(adapter_names)
            for ev in sends:
                if ev.adapter_name is None:
                    ev.adapter_name = name
        else:
            for i, ev in enumerate(sends):
                if ev.adapter_name is None:
                    ev.adapter_name = adapter_names[i % len(adapter_names)]
        return t

    inject_lora_adapters.__name__ = "inject_lora_adapters"
    return inject_lora_adapters
