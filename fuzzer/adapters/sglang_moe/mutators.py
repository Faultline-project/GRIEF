"""
SGLang MoE batch-isolation mutation strategies.

These mutations target the compute path of fused MoE kernels by varying batch
composition, background traffic, and arrival timing.  All mutations are
trace-level and backend-agnostic.
"""

from __future__ import annotations

import copy
import random
import uuid

from ...trace import Trace, Event, ActionType


def concurrent_expert_stress(trace: Trace, rng: random.Random) -> Trace:
    """Add 2–4 background SEND events around a designated target request."""
    sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if not sends:
        return trace

    t         = copy.deepcopy(trace)
    target_ev = rng.choice(sends)
    base_offset = min(ev.offset_ms for ev in sends)

    n = rng.randint(2, 4)
    for _ in range(n):
        fam_id = uuid.uuid4().hex[:12]
        while fam_id == target_ev.prompt_family_id:
            fam_id = uuid.uuid4().hex[:12]
        bg_ev = Event(
            action           = ActionType.SEND,
            offset_ms        = base_offset + rng.randint(0, 20),
            request_id       = str(uuid.uuid4())[:8],
            prompt_len       = rng.randint(64, 128),
            max_tokens       = rng.randint(32, 64),
            stream           = True,
            temperature      = 0.0,
            prompt_family_id = fam_id,
            origin           = "mutated",
        )
        t.events.append(bg_ev)
    t.events.sort(key=lambda e: e.offset_ms)
    return t


def jitter_offsets(trace: Trace, rng: random.Random) -> Trace:
    """Apply ±5–20ms random jitter to each SEND event's offset_ms."""
    sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if not sends:
        return trace

    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND:
            delta = rng.randint(5, 20) * rng.choice([-1, 1])
            ev.offset_ms = max(0, ev.offset_ms + delta)

    t.events.sort(key=lambda e: e.offset_ms)

    send_events = [ev for ev in t.events if ev.action == ActionType.SEND]
    for i in range(1, len(send_events)):
        prev = send_events[i - 1]
        curr = send_events[i]
        if curr.offset_ms <= prev.offset_ms:
            curr.offset_ms = prev.offset_ms + 1

    t.events.sort(key=lambda e: e.offset_ms)
    return t


def batch_size_step(trace: Trace, rng: random.Random) -> Trace:
    """Double or halve the number of background SEND events."""
    sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if len(sends) < 2:
        return trace

    t           = copy.deepcopy(trace)
    target_send = next(ev for ev in t.events if ev.action == ActionType.SEND)
    background  = [ev for ev in t.events
                   if ev.action == ActionType.SEND and ev is not target_send]

    if rng.random() < 0.5:
        new_events = []
        for ev in background:
            dup                  = copy.deepcopy(ev)
            dup.request_id       = str(uuid.uuid4())[:8]
            dup.prompt_family_id = uuid.uuid4().hex[:12]
            dup.origin           = "mutated"
            new_events.append(dup)
        t.events.extend(new_events)
    else:
        to_drop = set(id(ev) for ev in background[::2])
        t.events = [ev for ev in t.events if id(ev) not in to_drop]

    t.events.sort(key=lambda e: e.offset_ms)
    return t


MOE_MUTATIONS: list = [
    concurrent_expert_stress,
    jitter_offsets,
    batch_size_step,
]
