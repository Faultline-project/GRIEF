"""
MoE batch-isolation mutation strategies.

These mutations target the compute path of fused MoE kernels: vLLM processes
all requests in a batch together through FusedMoE (Triton), so cross-request
interference at the expert-dispatch level is possible.  Mutations vary batch
composition, background traffic, and arrival timing to stress this path.
"""

from __future__ import annotations

import copy
import random
import uuid

from ...trace import Trace, Event, ActionType


def concurrent_expert_stress(trace: Trace, rng: random.Random) -> Trace:
    """
    Add N=2-4 background SEND events around a designated target request.

    - Target: one existing SEND (preserves its prompt_family_id)
    - Background events: distinct generated family_ids (not the target's family)
    - Short prompts (64-128 tokens) and outputs (32-64 tokens) for fast dispatch
    - Tight offsets: within 0-20ms of the first existing event
    - temperature=0.0, stream=True

    Tests: does other concurrent traffic perturb the target's output?
    No-op if trace has no eligible SEND events.
    """
    sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if not sends:
        return trace

    t = copy.deepcopy(trace)
    target_ev = rng.choice(sends)
    base_offset = min(ev.offset_ms for ev in sends)

    n = rng.randint(2, 4)
    for _ in range(n):
        fam_id = uuid.uuid4().hex[:12]
        # Ensure background family differs from target
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
    """
    Apply ±5-20ms random jitter to each SEND event's offset_ms.

    After jittering, events are sorted by offset_ms and a ≥1ms separation
    floor is enforced between consecutive events (push forward if needed).

    Tests: does which requests land in the same fused dispatch affect outputs?
    No structural change — same requests, slightly different timing.
    """
    sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if not sends:
        return trace

    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND:
            delta = rng.randint(5, 20) * rng.choice([-1, 1])
            ev.offset_ms = max(0, ev.offset_ms + delta)

    # Sort all events by offset_ms
    t.events.sort(key=lambda e: e.offset_ms)

    # Enforce ≥1ms separation between consecutive SEND events
    send_events = [ev for ev in t.events if ev.action == ActionType.SEND]
    for i in range(1, len(send_events)):
        prev = send_events[i - 1]
        curr = send_events[i]
        if curr.offset_ms <= prev.offset_ms:
            curr.offset_ms = prev.offset_ms + 1

    # Re-sort after adjustment (non-SEND events stay at original positions)
    t.events.sort(key=lambda e: e.offset_ms)
    return t


def batch_size_step(trace: Trace, rng: random.Random) -> Trace:
    """
    Double or halve the number of background SEND events.

    - Target: the first SEND event — never removed or duplicated
    - Double: duplicate each non-target SEND with a new request_id and generated family_id
    - Halve:  drop every other non-target SEND

    Tests: does batch size (more or fewer background requests) affect the
    target's output through expert dispatch density changes?
    No-op if fewer than 2 SEND events (need at least one background request).
    """
    sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
    if len(sends) < 2:
        return trace

    t = copy.deepcopy(trace)
    target_send = next(ev for ev in t.events if ev.action == ActionType.SEND)
    background = [ev for ev in t.events if ev.action == ActionType.SEND and ev is not target_send]

    if rng.random() < 0.5:
        # Double: duplicate each background send
        new_events = []
        for ev in background:
            dup = copy.deepcopy(ev)
            dup.request_id = str(uuid.uuid4())[:8]
            dup.prompt_family_id = uuid.uuid4().hex[:12]
            dup.origin = "mutated"
            new_events.append(dup)
        t.events.extend(new_events)
    else:
        # Halve: drop every other background send
        to_drop = set(id(ev) for ev in background[::2])
        t.events = [ev for ev in t.events if id(ev) not in to_drop]

    t.events.sort(key=lambda e: e.offset_ms)
    return t


MOE_MUTATIONS: list = [
    concurrent_expert_stress,
    jitter_offsets,
    batch_size_step,
]
