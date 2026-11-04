"""
SGLang LoRA mutation strategies.

Primary target: same-request correctness under concurrent load.
  - same_adapter_high_concurrency: many requests, same adapter, shared prefix
    → stresses batch-level adapter slot assignment
  - preemption_stress_burst: long + short requests mixed with multiple adapters
    → triggers memory pressure / preemption + resume paths

Secondary (legacy cross-adapter KV frame, kept as secondary signal):
  - switch_adapter, concurrent_adapter_burst, same_prefix_different_adapters,
    adapter_thrash, mixed_base_lora, inject_same_prompt_pair
"""

from __future__ import annotations

import copy
import hashlib
import random

from ...trace import Trace, Event, ActionType


def _stable_family_id(prompt_len: int, prefix_len: int) -> str:
    key = f"fam:{prompt_len}:{prefix_len}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _lora_adapters(trace: Trace) -> list[str]:
    return list({
        ev.adapter_name
        for ev in trace.events
        if ev.action == ActionType.SEND and ev.adapter_name is not None
    })


def switch_adapter(trace: Trace, rng: random.Random) -> Trace:
    """Randomly reassign adapter_name on each SEND event."""
    adapters = _lora_adapters(trace)
    if len(adapters) < 2:
        return trace
    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND and ev.adapter_name is not None:
            ev.adapter_name = rng.choice(adapters)
    return t


def concurrent_adapter_burst(trace: Trace, rng: random.Random) -> Trace:
    """Insert 4–8 concurrent SEND events alternating between adapters with shared prefix."""
    adapters = _lora_adapters(trace)
    send_events = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if len(adapters) < 2 or not send_events:
        return trace

    t = copy.deepcopy(trace)
    template     = rng.choice(send_events)
    burst_size   = rng.randint(4, 8)
    shared_prefix = max(1, (template.prefix_len or 0),
                        (template.prompt_len or 16) // 2)
    burst_offset  = template.offset_ms

    for i in range(burst_size):
        ev              = copy.deepcopy(template)
        ev.request_id   = f"burst_{i}"
        ev.offset_ms    = burst_offset
        ev.prefix_len   = shared_prefix
        ev.adapter_name = adapters[i % len(adapters)]
        ev.origin       = "burst"
        t.events.append(ev)

    t.events.sort(key=lambda e: e.offset_ms)
    return t


def same_prefix_different_adapters(trace: Trace, rng: random.Random) -> Trace:
    """Force two events with different adapters to share the same prefix_len."""
    adapters = _lora_adapters(trace)
    if len(adapters) < 2:
        return trace

    t = copy.deepcopy(trace)
    send_events = [ev for ev in t.events if ev.action == ActionType.SEND
                   and ev.adapter_name is not None]

    by_adapter: dict[str, list[Event]] = {}
    for ev in send_events:
        by_adapter.setdefault(ev.adapter_name, []).append(ev)

    a_name, b_name = rng.sample(list(by_adapter.keys()), 2)
    ev_a = rng.choice(by_adapter[a_name])
    ev_b = rng.choice(by_adapter[b_name])

    min_len = min(ev_a.prompt_len or 16, ev_b.prompt_len or 16)
    shared  = max(1, min_len - 1)

    ev_a.prefix_len = shared
    ev_b.prefix_len = shared
    ev_a.prompt_len = max(ev_a.prompt_len or 16, shared + 1)
    ev_b.prompt_len = max(ev_b.prompt_len or 16, shared + 1)

    return t


def adapter_thrash(trace: Trace, rng: random.Random) -> Trace:
    """Rapid alternating adapter sends at 5–20ms intervals with no shared prefix."""
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
    """Demote ~1/3 of LoRA requests to base model (adapter_name=None)."""
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

    This ensures the isolation oracle can compare their outputs.
    Only completions events (messages=None) are eligible.
    """
    adapters = _lora_adapters(trace)
    if len(adapters) < 2:
        return trace

    t = copy.deepcopy(trace)
    send_events = [ev for ev in t.events if ev.action == ActionType.SEND
                   and ev.adapter_name is not None
                   and ev.messages is None]

    by_adapter: dict[str, list] = {}
    for ev in send_events:
        by_adapter.setdefault(ev.adapter_name, []).append(ev)

    if len(by_adapter) < 2:
        return trace

    a_name, b_name = rng.sample(list(by_adapter.keys()), 2)
    ev_a = rng.choice(by_adapter[a_name])
    ev_b = rng.choice(by_adapter[b_name])

    shared_prompt_len = min(ev_a.prompt_len or 16, ev_b.prompt_len or 16)
    shared_prefix_len = max(1, shared_prompt_len - 1)
    family_id = _stable_family_id(shared_prompt_len, shared_prefix_len)

    ev_a.prompt_len       = shared_prompt_len
    ev_b.prompt_len       = shared_prompt_len
    ev_a.prefix_len       = shared_prefix_len
    ev_b.prefix_len       = shared_prefix_len
    ev_a.prompt_family_id = family_id
    ev_b.prompt_family_id = family_id
    ev_a.logprobs         = 5
    ev_b.logprobs         = 5

    return t


def same_adapter_high_concurrency(trace: Trace, rng: random.Random) -> Trace:
    """N=6–12 requests, same adapter, shared prefix, 0–10 ms stagger.

    Stresses batch-level adapter slot assignment and same-adapter concurrent
    prefix races (finding_02400 variant with LoRA).
    """
    adapters = _lora_adapters(trace)
    sends    = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if not adapters or not sends:
        return trace

    t        = copy.deepcopy(trace)
    template = rng.choice(sends)
    adapter  = rng.choice(adapters)
    n        = rng.randint(6, 12)
    shared_prefix = max(1, (template.prefix_len or 0),
                        (template.prompt_len or 16) // 2)

    for i in range(n):
        ev              = copy.deepcopy(template)
        ev.request_id   = f"sahc_{i}"
        ev.offset_ms    = template.offset_ms + rng.randint(0, 10)
        ev.prefix_len   = shared_prefix
        ev.adapter_name = adapter
        ev.origin       = "same_adapter_high_concurrency"
        t.events.append(ev)

    t.events.sort(key=lambda e: e.offset_ms)
    return t


def preemption_stress_burst(trace: Trace, rng: random.Random) -> Trace:
    """Add 2–3 long-prompt requests (512–1024 tokens) interleaved with short ones.

    Mixing very long and short requests with multiple adapters creates memory
    pressure that forces SGLang to preempt long requests and swap adapter state.

    Gate: requires >= 2 adapters AND sends that are already concurrent (offset
    spread <= 500 ms). Sequential traces (spread > 500 ms) are skipped — adding
    long requests to sequential traffic does not create meaningful preemption.
    """
    adapters = _lora_adapters(trace)
    sends    = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if len(adapters) < 2 or len(sends) < 2:
        return trace

    offsets = sorted(ev.offset_ms for ev in sends)
    if offsets[-1] - offsets[0] > 500:
        return trace  # sequential trace, not concurrent

    t        = copy.deepcopy(trace)
    template = rng.choice(sends)
    n_long   = rng.randint(2, 3)
    base_ms  = template.offset_ms

    for i in range(n_long):
        ev              = copy.deepcopy(template)
        ev.request_id   = f"pre_long_{i}"
        ev.offset_ms    = base_ms + i * rng.randint(5, 15)
        ev.prompt_len   = rng.randint(512, 1024)
        ev.prefix_len   = rng.randint(0, 128)
        ev.max_tokens   = rng.randint(64, 128)
        ev.adapter_name = adapters[i % len(adapters)]
        ev.origin       = "preemption_stress"
        t.events.append(ev)

    t.events.sort(key=lambda e: e.offset_ms)
    return t


LORA_MUTATIONS: list = [
    same_adapter_high_concurrency,
    preemption_stress_burst,
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

    Without this, LoRA-path coverage is stranded on the initial LoRA seeds
    and never grows from the larger pool of root seeds.
    """
    def inject_lora_adapters(trace: Trace, rng: random.Random) -> Trace:
        sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
        if not sends:
            return trace
        t = copy.deepcopy(trace)
        sends = [ev for ev in t.events if ev.action == ActionType.SEND]
        # 30% chance: stamp all events with the same adapter (single-adapter load test)
        # 70% chance: alternate adapters across events (cross-adapter isolation shape)
        if rng.random() < 0.50:
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
