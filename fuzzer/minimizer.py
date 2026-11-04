"""
Minimizer: delta-debugging on Trace event sequences.

Given a bug-triggering Trace and a predicate (does this trace still trigger
the bug?), reduces the trace to the smallest sequence of events that still
satisfies the predicate.

Algorithm: two-phase minimization.

Phase 1 — Greedy forward pass (O(n) oracle calls per sweep):
    Iterate through events one by one; try dropping each.  Repeat until a
    full sweep removes nothing.  This converges quickly when individual
    events can be dropped independently, which is the common case for
    concurrent-request traces where many sends are redundant.

Phase 2 — ddmin with complement strategy (O(n log n) calls):
    Standard delta-debugging on the residual after the greedy pass.  Each
    granularity level now also tries keeping each chunk (removing its
    complement) before moving on, doubling the success rate for bugs that
    require a minimum positive set of events.

Post-processing — offset normalization:
    Compact offset_ms so the first event starts at 0ms, preserving relative
    gaps.  Makes minimized traces easier to read and reproduce.
"""

from __future__ import annotations

import asyncio
import copy
from typing import Callable, Awaitable, Optional, Set

from .trace import Trace, ActionType


Predicate = Callable[[Trace], Awaitable[bool]]


def _normalize_offsets(events: list) -> list:
    """Shift all offset_ms values so the earliest event starts at 0ms."""
    if not events:
        return events
    offsets = [e.offset_ms for e in events if e.offset_ms is not None]
    if not offsets:
        return events
    base = min(offsets)
    if base == 0:
        return events
    result = []
    for e in events:
        ev = copy.copy(e)
        if ev.offset_ms is not None:
            ev.offset_ms -= base
        result.append(ev)
    return result


async def _greedy_pass(
    events: list,
    trace: Trace,
    predicate: Predicate,
    is_valid: Callable[[list], bool],
) -> list:
    """
    Repeatedly sweep the event list and drop each event that is individually
    removable without breaking the predicate.  Continues until a full sweep
    makes no removals.  O(n) oracle calls per sweep, typically 2-3 sweeps.
    """
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(events):
            candidate_events = events[:i] + events[i + 1:]
            if not is_valid(candidate_events):
                i += 1
                continue
            candidate = copy.deepcopy(trace)
            candidate.events = candidate_events
            if await predicate(candidate):
                events = candidate_events
                changed = True
                # Don't advance i: the next event shifted into position i.
            else:
                i += 1
    return events


async def _ddmin_pass(
    events: list,
    trace: Trace,
    predicate: Predicate,
    is_valid: Callable[[list], bool],
) -> list:
    """
    Standard ddmin with complement-first strategy.  For each granularity
    level, tries both removing the chunk AND keeping only the chunk
    (complement removal) before moving to the next granularity.
    """
    granularity = 2
    while granularity <= len(events):
        chunk_size = len(events) // granularity
        reduced = False

        for i in range(granularity):
            start = i * chunk_size
            end   = start + chunk_size

            # Try removing this chunk
            without_chunk = events[:start] + events[end:]
            if is_valid(without_chunk):
                candidate = copy.deepcopy(trace)
                candidate.events = without_chunk
                if await predicate(candidate):
                    events      = without_chunk
                    granularity = max(granularity - 1, 2)
                    reduced     = True
                    break

            # Try keeping only this chunk (complement removal)
            only_chunk = events[start:end]
            if is_valid(only_chunk):
                candidate = copy.deepcopy(trace)
                candidate.events = only_chunk
                if await predicate(candidate):
                    events      = only_chunk
                    granularity = max(granularity - 1, 2)
                    reduced     = True
                    break

        if not reduced:
            if granularity == len(events):
                break
            granularity = min(granularity * 2, len(events))

    return events


async def minimize(trace: Trace,
                   predicate: Predicate,
                   min_length: int = 2,
                   required_action_types: Optional[Set[ActionType]] = None) -> Trace:
    """
    Return the smallest sub-trace of `trace` for which predicate returns True.

    predicate should re-run the scenario and check whether the bug fires.

    min_length: never reduce below this many events (default 2).
    required_action_types: action types that must survive if they existed in
        the original trace (e.g. CANCEL/DISCONNECT — the interesting part).
    """
    events = list(trace.events)

    if len(events) == 0:
        return trace

    # Which action types must be preserved (intersect with what original has)
    must_keep: Set[ActionType] = set()
    if required_action_types:
        original_actions = {e.action for e in events}
        must_keep = set(required_action_types) & original_actions

    def _is_valid_subset(subset: list) -> bool:
        if len(subset) < min_length:
            return False
        if must_keep:
            subset_actions = {e.action for e in subset}
            if not must_keep.issubset(subset_actions):
                return False
        return True

    # Phase 1: greedy forward pass — handles independent removals efficiently
    events = await _greedy_pass(events, trace, predicate, _is_valid_subset)

    # Phase 2: ddmin with complement — handles cases requiring a minimum
    # positive set that the greedy pass can't reduce
    events = await _ddmin_pass(events, trace, predicate, _is_valid_subset)

    # Post-processing: compact offsets
    events = _normalize_offsets(events)

    result        = copy.deepcopy(trace)
    result.events = events
    return result


async def minimize_with_oracle(
    trace: Trace,
    run_fn: Callable[[Trace], Awaitable[tuple]],
    oracle_fn: Callable[[tuple], bool],
    min_length: int = 2,
    required_action_types: Optional[Set[ActionType]] = None,
) -> Trace:
    """
    Convenience wrapper: builds the predicate from run + oracle callables.
    """
    async def predicate(t: Trace) -> bool:
        if not t.events:
            return False
        results = await run_fn(t)
        return oracle_fn(results)

    return await minimize(trace, predicate,
                          min_length=min_length,
                          required_action_types=required_action_types)
