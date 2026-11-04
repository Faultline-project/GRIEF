"""
SGLang speculative decoding mutation strategies.

These mutations target the spec rollback path and shared-prefix KV races
that speculative decoding introduces.  All mutations are trace-level and
backend-agnostic (identical logic to vllm_spec.mutators).
"""

from __future__ import annotations

import copy
import random
import uuid

from ...trace import Trace, Event, ActionType


def cancel_during_spec(trace: Trace, rng: random.Random) -> Trace:
    """
    Inject a CANCEL event targeting a long streaming SEND, biased to fire
    mid-speculation (50–350ms after the send).

    No-op if the trace has no eligible SEND events (stream=True, max_tokens >= 64).
    """
    eligible = [
        ev for ev in trace.events
        if ev.action == ActionType.SEND
        and ev.stream
        and (ev.max_tokens or 0) >= 64
    ]
    if not eligible:
        return trace

    t  = copy.deepcopy(trace)
    ev = rng.choice(eligible)
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
    Insert 2–4 concurrent SEND events that share the same prefix_len (≥256
    tokens) but have distinct prompt_family_ids, all at the same offset_ms.

    Forces multiple requests into the same KV block allocation window
    simultaneously, exercising rollback on shared prefix blocks.
    """
    n = rng.randint(2, 4)
    base_offset = rng.choice(
        [ev.offset_ms for ev in trace.events if ev.action == ActionType.SEND]
    ) if any(ev.action == ActionType.SEND for ev in trace.events) else 0

    prefix_len = rng.choice([256, 512, 1024])
    prompt_len = prefix_len + rng.randint(16, 64)
    max_tokens = rng.randint(32, 128)

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


def spec_mixed_chunk_race(trace: Trace, rng: random.Random) -> Trace:
    """
    Build a decode-then-large-prefill pattern that stresses speculative state
    transitions when chunked prefill and running decode overlap.

    Phase 1: 3–5 short streaming decode requests at t=0–60ms (prompt ≤ 512,
             max_tokens=64) to populate the running decode batch under spec.
    Phase 2: one large prefill request at t=150–300ms (prompt ≥ 3072) that
             is large enough to be chunked, arriving while phase-1 requests
             are still decoding.

    Preserves any existing events from the seed so the mutation composes.
    """
    t = copy.deepcopy(trace)

    n_decode = rng.randint(3, 5)
    decode_prompt = rng.choice([256, 384, 512])
    prefix_len    = rng.choice([64, 128])
    family_id     = uuid.uuid4().hex[:12]

    for i in range(n_decode):
        ev = Event(
            action           = ActionType.SEND,
            offset_ms        = i * rng.randint(10, 20),
            request_id       = f"mcd_dec_{i}",
            prompt_len       = decode_prompt,
            max_tokens       = 64,
            stream           = True,
            prefix_len       = prefix_len,
            temperature      = 0.0,
            prompt_family_id = family_id,
            origin           = "mutated",
        )
        t.events.append(ev)

    large_offset = rng.randint(150, 300)
    large_prompt = rng.choice([3072, 4096, 5120])
    ev = Event(
        action           = ActionType.SEND,
        offset_ms        = large_offset,
        request_id       = "mcd_prefill",
        prompt_len       = large_prompt,
        max_tokens       = rng.randint(16, 64),
        stream           = True,
        prefix_len       = 0,
        temperature      = 0.0,
        prompt_family_id = uuid.uuid4().hex[:12],
        origin           = "mutated",
    )
    t.events.append(ev)
    t.events.sort(key=lambda e: e.offset_ms)
    return t


def spec_budget_exhaust(trace: Trace, rng: random.Random) -> Trace:
    """
    Reduce max_tokens on a subset of SEND events to values smaller than the
    speculative draft budget, while preserving timing overlap so the affected
    requests remain concurrent with normal-budget requests.

    Targets short-budget requests finishing mid-speculative-verification:
    some requests in the batch complete before verification is done, changing
    the effective batch size in ways the spec decode path may not expect.

    Strategy:
    - Pick 2–3 eligible SEND events (stream=True, max_tokens >= 16).
    - Set their max_tokens to a small value in {1, 2, 3}.
    - Tighten their offset_ms to within 30ms of each other so they coexist
      in the same speculative batch.
    - Leave remaining requests at their original max_tokens so the batch
      contains a mix of short- and normal-budget requests.
    """
    eligible = [
        i for i, ev in enumerate(trace.events)
        if ev.action == ActionType.SEND
        and ev.stream
        and (ev.max_tokens or 0) >= 16
    ]
    if len(eligible) < 2:
        return trace

    t = copy.deepcopy(trace)
    n_shrink = min(rng.randint(2, 3), len(eligible))
    chosen   = rng.sample(eligible, n_shrink)

    # Anchor all chosen events near the earliest one so they overlap
    anchor_ms = min(t.events[i].offset_ms for i in chosen)
    for i in chosen:
        t.events[i].max_tokens = rng.choice([1, 2, 3])
        t.events[i].offset_ms  = anchor_ms + rng.randint(0, 30)

    t.events.sort(key=lambda e: e.offset_ms)
    return t


SPEC_MUTATIONS: list = [
    cancel_during_spec,
    spec_shared_prefix_race,
    spec_mixed_chunk_race,
    spec_budget_exhaust,
]
