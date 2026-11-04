"""
SGLang-specific mutations targeting RadixAttention cache internals.

These mutations are designed to stress the radix tree's prefix matching,
node splitting, eviction, and lock reference management — all of which
lack synchronization primitives in SGLang's implementation.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from ...trace import Trace, Event, ActionType


# ---------------------------------------------------------------------------
# Mutation implementations
# ---------------------------------------------------------------------------

def prefix_boundary_align(trace: Trace, rng: random.Random) -> Trace:
    """Set prefix_len to exact page/block boundaries to trigger node splits.

    The radix cache splits nodes when a prefix match ends partway through
    a stored key. Setting prefix_len to common block sizes (1, 8, 16, 32,
    64, 128) hits the split boundary condition directly.

    Bug surface: _split_node(), hash value splitting at page boundary,
    off-by-one in split_len // page_size.
    """
    trace = copy.deepcopy(trace)
    boundaries = [1, 8, 16, 32, 64, 128, 256]
    for event in trace.events:
        if event.action == ActionType.SEND and event.prompt_len and rng.random() < 0.5:
            boundary = rng.choice(boundaries)
            if boundary < (event.prompt_len or 16):
                event.prefix_len = boundary
    return trace


def prefix_fork_stress(trace: Trace, rng: random.Random) -> Trace:
    """Multiple requests share a long prefix then fork to different suffixes.

    Forces the radix tree to create a shared prefix node, then split it
    when subsequent requests extend the prefix differently. Tests node
    splitting under concurrent access.

    Bug surface: _insert_helper() partial match + split, concurrent
    _match_prefix_helper() reading while split is in progress.
    """
    trace = copy.deepcopy(trace)
    sends = [e for e in trace.events if e.action == ActionType.SEND]
    if len(sends) < 2:
        return trace

    # Pick a shared prefix length
    shared_prefix = rng.choice([32, 64, 128, 256])

    # First request: long prompt with shared prefix
    base = rng.choice(sends)
    base.prefix_len = shared_prefix
    base.prompt_len = max(base.prompt_len or 128, shared_prefix + 64)

    # Add 2-4 forking requests: same prefix_len but DIFFERENT family_ids
    # so _build_prompt() generates different suffixes after the shared prefix.
    # This forces the radix tree to split at the prefix boundary.
    base_family = f"fork_{rng.randint(0, 9999)}"
    base.prompt_family_id = f"{base_family}_base"

    for i in range(rng.randint(2, 4)):
        fork = Event(
            action=ActionType.SEND,
            offset_ms=base.offset_ms + rng.randint(5, 50),
            request_id=f"{base.request_id}_fork{i}",
            prompt_len=shared_prefix + rng.randint(32, 256),
            max_tokens=base.max_tokens or 32,
            stream=True,
            prefix_len=shared_prefix,
            prompt_family_id=f"{base_family}_fork{i}",
        )
        trace.events.append(fork)
    return trace


def eviction_race(trace: Trace, rng: random.Random) -> Trace:
    """Create memory pressure to trigger eviction during concurrent prefill.

    Injects large-prompt fill requests before the existing trace events,
    then appends reuse requests after a gap. The original trace events
    are preserved (including adapter_name, messages, etc.) so this
    mutation works correctly in LoRA/Spec campaigns.

    Bug surface: evict() running concurrently with match_prefix(),
    lock_ref underflow, evictable_leaves set inconsistency.
    """
    trace = copy.deepcopy(trace)

    shared_prefix = rng.choice([64, 128, 256])
    family_id = f"evict_{rng.randint(0, 9999)}"

    # Compute time offset: insert fill events before existing trace
    min_offset = min((e.offset_ms for e in trace.events), default=0)

    # Phase 1: fill cache with large prompts that share a prefix family
    fill_events = []
    for i in range(rng.randint(3, 6)):
        fill_events.append(Event(
            action=ActionType.SEND,
            offset_ms=max(0, min_offset - 200 + i * rng.randint(5, 20)),
            request_id=f"fill_{i}",
            prompt_len=rng.choice([1024, 2048, 4096]),
            max_tokens=rng.randint(16, 64),
            stream=True,
            prefix_len=shared_prefix,
            prompt_family_id=family_id,
        ))

    # Phase 2: after existing trace, send requests that reuse the same prefix
    max_offset = max((e.offset_ms for e in trace.events), default=0)
    gap_ms = rng.randint(200, 500)
    reuse_events = []
    for i in range(rng.randint(2, 4)):
        reuse_events.append(Event(
            action=ActionType.SEND,
            offset_ms=max_offset + gap_ms + i * rng.randint(5, 15),
            request_id=f"reuse_{i}",
            prompt_len=shared_prefix + rng.randint(64, 256),
            max_tokens=32,
            stream=True,
            prefix_len=shared_prefix,
            prompt_family_id=family_id,
        ))

    # Prepend fill, keep original trace, append reuse
    trace.events = fill_events + trace.events + reuse_events
    return trace


def lock_ref_stress(trace: Trace, rng: random.Random) -> Trace:
    """Rapid cancel + retry to stress lock reference counting.

    The radix cache uses lock_ref counters (not OS locks) to protect
    nodes from eviction. Rapid cancel/retry sequences may cause
    inc_lock_ref/dec_lock_ref to race, potentially leading to underflow
    or nodes being evicted while still referenced.

    Bug surface: inc_lock_ref/dec_lock_ref race, lock_ref underflow,
    eviction of locked nodes.
    """
    trace = copy.deepcopy(trace)
    sends = [e for e in trace.events if e.action == ActionType.SEND]
    if not sends:
        return trace

    original_target = rng.choice(sends)
    # Remove original from trace — it will be re-added via new_events
    trace.events = [e for e in trace.events if e is not original_target]

    current = original_target
    new_events = [current]

    # Rapid cancel + retry cycle (3-5 iterations)
    t = current.offset_ms
    for i in range(rng.randint(3, 5)):
        t += rng.randint(10, 50)
        new_events.append(Event(
            action=ActionType.CANCEL,
            offset_ms=t,
            request_id=current.request_id,
            stream=current.stream,
        ))
        t += rng.randint(5, 20)
        # Preserve all fields from original target (including messages for chat)
        retry = copy.deepcopy(original_target)
        retry.action = ActionType.SEND
        retry.offset_ms = t
        retry.request_id = f"{original_target.request_id}_retry{i}"
        new_events.append(retry)
        current = retry  # next cancel targets the retry

    trace.events = trace.events + new_events
    return trace


def partial_prefix_mismatch(trace: Trace, rng: random.Random) -> Trace:
    """Send requests with overlapping but not identical prefixes.

    Request A has prefix_len=100, request B has prefix_len=80. Both share
    the first 80 tokens but A has 20 more prefix tokens. This forces a
    partial match in the radix tree — B matches a prefix of A's cached
    node, triggering a node split mid-traversal.

    Bug surface: _match_prefix_helper() partial match, _split_node()
    during concurrent traversal.
    """
    trace = copy.deepcopy(trace)
    sends = [e for e in trace.events if e.action == ActionType.SEND]
    if len(sends) < 2:
        return trace

    family_id = f"partial_{rng.randint(0, 9999)}"
    long_prefix = rng.choice([64, 128, 256])
    short_prefix = long_prefix - rng.choice([8, 16, 32])
    if short_prefix < 1:
        short_prefix = 1

    # Request A: longer prefix, arrives first
    a = sends[0]
    a.prefix_len = long_prefix
    a.prompt_len = max(a.prompt_len or 128, long_prefix + 32)
    a.prompt_family_id = family_id

    # Request B: shorter prefix, arrives shortly after.
    # sends[1] is guaranteed to exist (len(sends) >= 2 checked above).
    b = sends[1]
    b.prefix_len = short_prefix
    b.prompt_len = max(b.prompt_len or 128, short_prefix + 32)
    b.prompt_family_id = family_id
    b.offset_ms = a.offset_ms + rng.randint(10, 50)
    b.request_id = f"{a.request_id}_partial"
    return trace


def concurrent_same_prefix(trace: Trace, rng: random.Random) -> Trace:
    """Multiple requests with identical prefix_len arrive simultaneously.

    Tests the radix cache's handling of concurrent lookups for the same
    prefix. All requests should get the same cached node, but if the
    first request's insert hasn't completed when the second arrives,
    the second may create a duplicate node or miss the cache.

    Bug surface: concurrent _match_prefix_helper() + _insert_helper()
    for the same key, duplicate nodes in tree.
    """
    trace = copy.deepcopy(trace)
    sends = [e for e in trace.events if e.action == ActionType.SEND]
    if not sends:
        return trace

    base = rng.choice(sends)
    prefix = rng.choice([32, 64, 128])
    family_id = f"same_{rng.randint(0, 9999)}"

    # 4-8 requests all at the same offset, same prefix
    new_events = []
    for i in range(rng.randint(4, 8)):
        ev = Event(
            action=ActionType.SEND,
            offset_ms=base.offset_ms + rng.randint(0, 5),  # near-simultaneous
            request_id=f"same_pfx_{i}",
            prompt_len=prefix + rng.randint(32, 256),
            max_tokens=base.max_tokens or 32,
            stream=True,
            prefix_len=prefix,
            prompt_family_id=family_id,
        )
        new_events.append(ev)

    trace.events.extend(new_events)
    return trace


# ---------------------------------------------------------------------------
# Mutation registry
# ---------------------------------------------------------------------------

SGLANG_MUTATIONS = [
    prefix_boundary_align,
    prefix_fork_stress,
    eviction_race,
    lock_ref_stress,
    partial_prefix_mismatch,
    concurrent_same_prefix,
]

SGLANG_MUTATION_WEIGHTS = {
    prefix_boundary_align:    3.0,   # directly targets known bug (finding_03330)
    prefix_fork_stress:       2.5,   # tree split under concurrent access
    eviction_race:            2.0,   # memory pressure + cache reuse
    lock_ref_stress:          2.0,   # cancel/retry + reference counting
    partial_prefix_mismatch:  2.0,   # partial match node split
    concurrent_same_prefix:   1.5,   # duplicate node creation
}
