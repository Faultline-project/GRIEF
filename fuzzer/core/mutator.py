"""
Core generic mutations and mutate() dispatcher.

Generic mutations are mode-agnostic and work with any target adapter.
vLLM-specific mutations live in fuzzer/adapters/vllm_base/mutators.py.
"""

from __future__ import annotations

import copy
import random
from typing import Callable, Optional

from ..trace import Trace, Event, ActionType


MutationFn = Callable[[Trace, random.Random], Trace]

MAX_TOKENS_CAP  = 256   # output token ceiling; seed files are not affected
MAX_BURST_EXTRA = 5     # add_burst adds at most this many extra requests

# ---------------------------------------------------------------------------
# Individual generic mutation strategies
# ---------------------------------------------------------------------------

def jitter_timing(trace: Trace, rng: random.Random) -> Trace:
    """Shift event offsets by ±20% to find timing-sensitive races."""
    t = copy.deepcopy(trace)
    for ev in t.events:
        delta = int(ev.offset_ms * rng.uniform(-0.2, 0.2))
        ev.offset_ms = max(0, ev.offset_ms + delta)
    return t


def scale_prompt_lengths(trace: Trace, rng: random.Random) -> Trace:
    """Scale all prompt lengths by a random factor (0.25x – 4x)."""
    t      = copy.deepcopy(trace)
    factor = rng.choice([0.25, 0.5, 2.0, 4.0])
    for ev in t.events:
        if ev.action == ActionType.SEND and ev.prompt_len:
            ev.prompt_len = max(1, int(ev.prompt_len * factor))
        elif ev.action == ActionType.SEND and ev.messages:
            for msg in ev.messages:
                msg["content_len"] = max(1, int(msg.get("content_len", 16) * factor))
    return t


def scale_max_tokens(trace: Trace, rng: random.Random) -> Trace:
    """Scale max_tokens to stress decode-phase KV pressure."""
    t      = copy.deepcopy(trace)
    factor = rng.choice([0.1, 0.5, 2.0, 8.0])
    for ev in t.events:
        if ev.action == ActionType.SEND and ev.max_tokens:
            ev.max_tokens = min(MAX_TOKENS_CAP, max(1, int(ev.max_tokens * factor)))
    return t


def inject_cancel(trace: Trace, rng: random.Random) -> Trace:
    """Pick a random SEND and insert a CANCEL shortly after it."""
    t     = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    target = rng.choice(sends)
    delay  = rng.randint(50, 500)   # cancel 50–500 ms after send
    cancel = Event(
        action=ActionType.CANCEL,
        offset_ms=target.offset_ms + delay,
        request_id=target.request_id,
    )
    t.events.append(cancel)
    return t


def inject_disconnect(trace: Trace, rng: random.Random) -> Trace:
    """Replace a random CANCEL with a DISCONNECT (TCP close instead)."""
    t       = copy.deepcopy(trace)
    cancels = [ev for ev in t.events if ev.action == ActionType.CANCEL]
    if not cancels:
        return inject_cancel(t, rng)   # fall back to injecting a cancel first
    target        = rng.choice(cancels)
    target.action = ActionType.DISCONNECT
    return t


def change_prefix_overlap(trace: Trace, rng: random.Random) -> Trace:
    """Randomly set prefix_len on all SEND events to stress KV sharing."""
    t          = copy.deepcopy(trace)
    raw_prefix = rng.choice([0, 32, 128, 512, 1024])
    for ev in t.events:
        if ev.action == ActionType.SEND:
            # prefix_len must be < prompt_len so there are unique suffix tokens
            ev.prefix_len = min(raw_prefix, max(0, (ev.prompt_len or 1) - 1))
    return t


def add_burst(trace: Trace, rng: random.Random) -> Trace:
    """Add a burst of up to MAX_BURST_EXTRA extra SEND events at a random offset."""
    t          = copy.deepcopy(trace)
    burst_time = rng.randint(0, max(1, t.duration_ms()))
    sends      = [ev for ev in trace.events if ev.action == ActionType.SEND]
    picks      = rng.sample(sends, min(MAX_BURST_EXTRA, len(sends)))
    for original in picks:
        burst_ev = copy.deepcopy(original)
        burst_ev.offset_ms  = burst_time + rng.randint(0, 50)
        burst_ev.request_id = original.request_id + "_b"
        burst_ev.origin     = "burst"
        t.events.append(burst_ev)
    return t


def toggle_stream(trace: Trace, rng: random.Random) -> Trace:
    """Flip stream=True/False on all SEND events."""
    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND:
            ev.stream = not ev.stream
    return t


def remove_random_event(trace: Trace, rng: random.Random) -> Trace:
    """Drop one random event (used during minimization too)."""
    t = copy.deepcopy(trace)
    if len(t.events) > 1:
        idx = rng.randrange(len(t.events))
        t.events.pop(idx)
    return t


def inject_retry(trace: Trace, rng: random.Random) -> Trace:
    """After a CANCEL/DISCONNECT, re-send the same request to stress KV block reuse."""
    t       = copy.deepcopy(trace)
    cancels = [ev for ev in t.events
               if ev.action in (ActionType.CANCEL, ActionType.DISCONNECT)]
    if not cancels:
        return inject_cancel(t, rng)
    target = rng.choice(cancels)
    original_send = next(
        (ev for ev in t.events
         if ev.action == ActionType.SEND and ev.request_id == target.request_id),
        None,
    )
    if original_send is None:
        return t
    retry_delay = rng.randint(50, 200)
    retry = copy.deepcopy(original_send)
    retry.offset_ms  = target.offset_ms + retry_delay
    retry.request_id = target.request_id + "_retry"
    retry.origin     = "retry"
    t.events.append(retry)
    return t


def stagger_arrivals(trace: Trace, rng: random.Random) -> Trace:
    """Spread SEND events uniformly over a 100–800ms window.

    Produces mid-prefill interleaving that simultaneous bursts cannot:
    each request arrives while earlier ones are still in the prefill phase,
    forcing the scheduler to handle overlapping prefill + decode batches.
    """
    t = copy.deepcopy(trace)
    sends = sorted(
        [ev for ev in t.events if ev.action == ActionType.SEND],
        key=lambda e: e.offset_ms,
    )
    if len(sends) < 2:
        return t
    window_ms = rng.randint(100, 800)
    base_ms   = sends[0].offset_ms
    step      = window_ms / max(len(sends) - 1, 1)
    for i, ev in enumerate(sends):
        ev.offset_ms = int(base_ms + i * step)
    return t


def burst_idle_burst(trace: Trace, rng: random.Random) -> Trace:
    """Insert an idle gap mid-trace to trigger KV eviction then reuse.

    Splits events at the temporal midpoint and shifts the second half
    forward by 400–800ms. The server drains and partially evicts KV state
    during the idle window; the second burst then re-enters the same prefix
    cache entries, targeting eviction races and stale-block reuse bugs.
    """
    t = copy.deepcopy(trace)
    if len(t.events) < 2:
        return t
    mid_ms = t.duration_ms() // 2
    gap_ms = rng.randint(400, 800)
    for ev in t.events:
        if ev.offset_ms > mid_ms:
            ev.offset_ms += gap_ms
    return t


# ---------------------------------------------------------------------------
# Splice operations (different signature — two parents)
# ---------------------------------------------------------------------------

def splice(trace_a: Trace, trace_b: Trace, rng: random.Random) -> Trace:
    """Crossover: first segment of trace_a + second segment of trace_b.

    Split points are chosen at 30–70% of each trace's duration so both halves
    are non-trivial.  The second segment is rebased in time to start immediately
    after the first segment ends.  Orphaned CANCEL/DISCONNECT events (whose
    SEND was in the discarded half) are removed to keep the trace structurally
    valid.

    Note: splice is NOT in MUTATIONS (different signature — two parents).
    It is called by mutate() when an `other` trace is supplied.
    """
    t_a = copy.deepcopy(trace_a)
    t_b = copy.deepcopy(trace_b)
    dur_a   = max(trace_a.duration_ms(), 1)
    dur_b   = max(trace_b.duration_ms(), 1)
    split_a = int(rng.uniform(0.3, 0.7) * dur_a)
    split_b = int(rng.uniform(0.3, 0.7) * dur_b)

    first_half  = [e for e in t_a.events if e.offset_ms <= split_a]
    second_half = [e for e in t_b.events if e.offset_ms >= split_b]

    # Rebase second half: shift so it starts right after split_a
    offset = split_a + 1 - split_b
    for e in second_half:
        e.offset_ms  = max(0, e.offset_ms + offset)
        e.request_id = e.request_id + "_s"   # avoid ID collisions with first half

    merged = first_half + second_half
    merged.sort(key=lambda e: e.offset_ms)

    # Remove orphaned CANCEL/DISCONNECT events whose SEND was not included
    send_ids = {e.request_id for e in merged if e.action == ActionType.SEND}
    merged   = [e for e in merged
                if e.action == ActionType.SEND or e.request_id in send_ids]

    result        = copy.deepcopy(trace_a)
    result.events = merged
    result.seed   = rng.randint(0, 2**31)
    return result


def directed_splice(trace_a: Trace, trace_b: Trace, rng: random.Random) -> Trace:
    """Semantics-aware crossover: prefix-building half of A + eviction half of B.

    Split point in A is chosen to maximise prefix-sharing events (high
    prefix_len relative to prompt_len) in the first half.  Split point in B
    starts just before its largest requests.  Trace B's events are rebased so
    they arrive while A's requests are still in-flight — the overlap window is
    randomised (10–150 ms) to explore the race window without creating a dead
    zone that would let the server drain and evict the KV state we just built.

    Falls back to regular splice() if either trace has too few events.
    """
    if len(trace_a.events) < 2 or len(trace_b.events) < 2:
        return splice(trace_a, trace_b, rng)

    t_a = copy.deepcopy(trace_a)
    t_b = copy.deepcopy(trace_b)

    sends_a = sorted([e for e in t_a.events if e.action == ActionType.SEND],
                     key=lambda e: e.offset_ms)
    sends_b = sorted([e for e in t_b.events if e.action == ActionType.SEND],
                     key=lambda e: e.offset_ms)

    if not sends_a or not sends_b:
        return splice(trace_a, trace_b, rng)

    # --- Split A: maximise prefix-sharing events in the first half -----------
    def prefix_ratio(ev: Event) -> float:
        if not ev.prompt_len:
            return 0.0
        return (ev.prefix_len or 0) / ev.prompt_len

    min_idx = max(1, len(sends_a) // 3)
    max_idx = max(min_idx + 1, int(len(sends_a) * 0.7))
    best_split_idx, best_score = min_idx, -1.0
    for idx in range(min_idx, max_idx + 1):
        score = sum(prefix_ratio(e) for e in sends_a[:idx])
        if score > best_score:
            best_score, best_split_idx = score, idx

    # The last SEND in A's first half — eviction must arrive while it's live
    last_send_a_time = sends_a[best_split_idx - 1].offset_ms
    split_a_time     = last_send_a_time

    # --- Split B: start from the eviction-heavy (large-prompt) portion -------
    # Use the temporal midpoint; large requests in B tend to cluster later
    split_b_time = sends_b[len(sends_b) // 2].offset_ms if len(sends_b) > 1 else 0

    first_half  = [e for e in t_a.events if e.offset_ms <= split_a_time]
    second_half = [e for e in t_b.events if e.offset_ms >= split_b_time]
    if not second_half:
        second_half = list(t_b.events)

    # --- Rebase B so it overlaps with A's last send ---------------------------
    # overlap_ms: positive = eviction starts BEFORE A's last send finishes
    overlap_ms = rng.randint(10, 150)
    min_second = min(e.offset_ms for e in second_half)
    rebase_offset = last_send_a_time + overlap_ms - min_second
    for e in second_half:
        e.offset_ms  = max(0, e.offset_ms + rebase_offset)
        e.request_id = e.request_id + "_ds"

    merged = first_half + second_half
    merged.sort(key=lambda e: e.offset_ms)

    # Remove orphaned CANCEL/DISCONNECT whose SEND was discarded
    send_ids = {e.request_id for e in merged if e.action == ActionType.SEND}
    merged   = [e for e in merged
                if e.action == ActionType.SEND or e.request_id in send_ids]

    result        = copy.deepcopy(trace_a)
    result.events = merged
    result.seed   = rng.randint(0, 2**31)
    return result


# ---------------------------------------------------------------------------
# Generic mutation registry
# ---------------------------------------------------------------------------

MUTATIONS: list[MutationFn] = [
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
    stagger_arrivals,
    burst_idle_burst,
]


def mutate(trace: Trace, rng: random.Random, n: int = 1,
           other: Optional[Trace] = None,
           other_pressure: float = 0.0,
           mutations: Optional[list] = None,
           weights: Optional[list] = None) -> "tuple[Trace, list[str]]":
    """Apply n randomly chosen mutation strategies sequentially.

    If `other` is provided, splice probability scales with other_pressure:
      - base 20%, up to 40% at pressure ≥ 10
    When other_pressure is high (other trace stressed the scheduler heavily),
    directed_splice is used instead of random splice to maximise the chance
    of building KV state then immediately evicting it.

    If `mutations` is provided, it overrides the default MUTATIONS list.

    Returns
    -------
    (Trace, list[str])
        The mutated trace and the ordered list of operator names that were
        applied (length == n). Operator names are the function __name__ for
        registry mutations, or "splice" / "directed_splice" for crossover.
    """
    _mutations = mutations if mutations is not None else MUTATIONS
    result = trace
    ops_applied: list[str] = []
    splice_prob = min(0.40, 0.20 + other_pressure * 0.02)
    for _ in range(n):
        if other is not None and rng.random() < splice_prob:
            if other_pressure > 3.0:
                result = directed_splice(result, other, rng)
                ops_applied.append("directed_splice")
            else:
                result = splice(result, other, rng)
                ops_applied.append("splice")
        else:
            fn     = rng.choices(_mutations, weights=weights, k=1)[0]
            result = fn(result, rng)
            ops_applied.append(fn.__name__)
    result.seed = rng.randint(0, 2**31)
    return result, ops_applied
