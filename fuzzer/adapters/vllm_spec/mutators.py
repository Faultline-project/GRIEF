"""
Speculative decoding mutation strategies.

These mutations target the spec rollback path and shared-prefix KV races
that speculative decoding introduces.  Per-request spec toggle fields do not
exist in the vLLM API (Step 0 verified), so mutations work at the trace level
only — controlling concurrency, timing, and prefix overlap.
"""

from __future__ import annotations

import copy
import random
import uuid

from ...trace import Trace, Event, ActionType


def cancel_during_spec(trace: Trace, rng: random.Random) -> Trace:
    """
    Inject a CANCEL event 50–350 ms after one randomly chosen long streaming
    SEND event (stream=True, max_tokens >= 64).

    The flat millisecond window replaced an earlier "30–60 % of expected wall
    time" estimate, which assumed ~20 ms/token and overshot the speculative
    verification window entirely on small fast models.

    Biases toward requests that are likely mid-speculation when cancelled,
    exercising the rollback path under cancellation pressure.

    No-op if the trace has no eligible SEND events.
    """
    eligible = [
        ev for ev in trace.events
        if ev.action == ActionType.SEND
        and ev.stream
        and (ev.max_tokens or 0) >= 64
    ]
    if not eligible:
        return trace

    t = copy.deepcopy(trace)
    ev = rng.choice(eligible)

    # Target the spec verification window directly.
    # A flat 50–350 ms range after the send fires early enough to interrupt
    # an in-progress speculative batch regardless of model speed, unlike
    # a fraction of estimated wall time which overshoots on small fast models.
    cancel_offset = ev.offset_ms + rng.randint(50, 350)

    cancel = Event(
        action     = ActionType.CANCEL,
        offset_ms  = cancel_offset,
        request_id = ev.request_id,
        origin     = "mutated",
    )
    t.events.append(cancel)
    t.events.sort(key=lambda e: e.offset_ms)
    return t


def spec_shared_prefix_race(trace: Trace, rng: random.Random) -> Trace:
    """
    Insert 2–4 concurrent SEND events that share the same prefix_len
    (>= 256 tokens) but have distinct prompt_family_ids, all landing at the
    same offset_ms.

    Forces multiple requests into the same KV block allocation window
    simultaneously, exercising rollback on shared prefix blocks — the
    primary corruption surface for speculative decoding with prefix caching.
    """
    n = rng.randint(2, 4)
    base_offset = rng.choice(
        [ev.offset_ms for ev in trace.events if ev.action == ActionType.SEND]
    ) if any(ev.action == ActionType.SEND for ev in trace.events) else 0

    prefix_len  = rng.choice([256, 512, 1024])
    prompt_len  = prefix_len + rng.randint(16, 64)
    max_tokens  = rng.randint(32, 128)

    t = copy.deepcopy(trace)
    for _ in range(n):
        fam_id = uuid.uuid4().hex[:12]
        ev = Event(
            action           = ActionType.SEND,
            offset_ms        = base_offset,
            request_id       = str(uuid.uuid4())[:8],
            prompt_len       = prompt_len,
            max_tokens       = max_tokens,
            stream           = True,
            prefix_len       = prefix_len,
            temperature      = 0.0,
            prompt_family_id = fam_id,
            origin           = "mutated",
        )
        t.events.append(ev)
    t.events.sort(key=lambda e: e.offset_ms)
    return t


SPEC_MUTATIONS: list = [
    cancel_during_spec,
    spec_shared_prefix_race,
]
