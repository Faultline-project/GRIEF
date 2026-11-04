"""
KVEventCollector — subscribe to vLLM's KV cache event stream over ZMQ and
provide structured probe summaries for LoRA isolation detection.

Requires the vLLM server to be started with:
    --kv-events-config '{"enable_kv_cache_events": true, "publisher": "zmq", "endpoint": "tcp://*:5557"}'

If the server doesn't have events enabled, or pyzmq / vllm are not importable,
the collector silently sets _enabled=False and all calls return minimal/empty results.

Architecture:
  - KVEventCollector owns: normalization, ownership tracking, probe-window queries,
    summary construction, fingerprint generation.
  - Core runner owns: collector lifecycle (start/stop), timing markers.
  - Adapters own: semantic interpretation of KVProbeSummary fields.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import hashlib
import json
import math
import time
from typing import List, Optional

_TOPIC          = ""            # vLLM 0.18.0 publishes with empty topic
_BUFFER_SIZE    = 2000   # max events retained in memory
_DRAIN_TIMEOUT  = 0.01   # seconds — asyncio wait_for timeout per drain cycle
_TOKEN_CAP      = 64     # max token_ids stored per BlockStored event
_RAW_EVENTS_CAP = 200    # max raw events included in a probe summary
_EVICTED_CAP    = 50     # max conflicting/evicted hashes in a summary
_REUSE_TOO_FAST_MS = 10.0  # remove→store gap shorter than this (ms) is suspicious


# ---------------------------------------------------------------------------
# JSON boundary
# ---------------------------------------------------------------------------

def _json_safe(value):
    """Recursively replace non-finite floats (inf / -inf / NaN) with ``None``.

    ``KVProbeSummary`` is splatted verbatim into every finding report, and
    ``json.dumps`` renders ``float('inf')`` as a bare ``Infinity`` token — which
    is *not* valid JSON (RFC 8259 has no such literal, and ``allow_nan=False``
    rejects it).  Every non-Python consumer of a finding file therefore failed
    to parse it; 30+ archived findings on disk carry ``"reuse_too_fast_min_gap_ms":
    Infinity`` today.

    The sentinel is stripped **at the serialisation boundary only**, not on the
    dataclass field.  In memory ``inf`` is the correct value: it is the identity
    for the running ``min()`` in :meth:`summarize_probe`, and every reader
    compares it numerically (``analyze_kv_anomalies`` below,
    ``core/runner.py:1616`` ``if gap < float('inf')``), so a ``None`` on the
    field would turn each of those into a ``TypeError``.  ``null`` is also the
    natural JSON spelling of what the field means when no event was observed.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# KVProbeSummary — structured result from summarize_probe()
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class KVProbeSummary:
    """Structured summary of KV cache activity within a probe time window.

    Fields
    ------
    start_t / end_t
        Wall-clock timestamps (time.time()) bounding the probe window.
    stored_count
        Total block-hash occurrences in BlockStored events (not deduplicated).
    removed_count
        Total block-hash occurrences in BlockRemoved events.
    cleared_count
        Number of AllBlocksCleared events in the window.
    distinct_blocks
        Number of distinct block_hashes seen in BlockStored events.
    per_lora_counts
        Mapping from lora_name (or lora_id string / "unknown") to
        {"stored": int, "removed": int}.  Note: BlockRemoved events in vLLM
        0.18.0 do not carry lora_id, so "removed" counts are attributed under
        "unknown" — this is a v1 limitation of the upstream schema.
    cross_lora_reused_blocks
        Number of block_hashes that appeared in BlockStored events under two or
        more distinct lora_ids within this window.  A non-zero value is a
        strong structural indicator of a cache-key isolation defect.
    conflicting_hashes
        List of hex strings for the conflicting block_hashes (capped at 50).
    reuse_too_fast_count
        Number of block_hashes that were seen in a BlockRemoved event and then
        again in a BlockStored event within _REUSE_TOO_FAST_MS milliseconds.
        A very short remove→store gap suggests the block was not properly
        invalidated before being reused.
    reuse_too_fast_min_gap_ms
        Shortest observed remove→store gap in ms across all flagged blocks.
        float('inf') in memory when reuse_too_fast_count == 0 (it is the
        identity for the running min, and readers compare it numerically);
        ``to_dict()`` renders it as JSON ``null``, since ``Infinity`` is not
        valid JSON.  See :func:`_json_safe`.
    fingerprint
        16-char hex hash of the summary structure, order-insensitive.
        Useful for dedup and clustering of similar probe windows.
    raw_event_sample
        Last ≤200 events in the window for forensic inspection.
    """

    start_t:                  float
    end_t:                    float
    stored_count:             int
    removed_count:            int
    cleared_count:            int
    distinct_blocks:          int
    per_lora_counts:          dict          # lora_name_or_id -> {"stored": int, "removed": int}
    cross_lora_reused_blocks: int
    conflicting_hashes:       List[str]
    reuse_too_fast_count:     int   = 0
    reuse_too_fast_min_gap_ms: float = float('inf')
    hash_content_conflict_count: int  = 0
    hash_content_conflicts:  List[dict] = dataclasses.field(default_factory=list)
    # [{hash, first_tokens, conflict_tokens}] — capped at 10; direct content violation
    orphan_child_count:       int   = 0
    # blocks whose parent_block_hash was not stored earlier in this window (weak signal)
    fingerprint:              str   = ""
    raw_event_sample:         list  = dataclasses.field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe dict of the summary.

        Non-finite floats are mapped to ``None`` so the result survives
        ``json.dumps(..., allow_nan=False)`` — this dict is splatted into every
        finding report (``snapshot_forensics``, ``core/runner.py`` structural
        repro), and a bare ``Infinity`` there is unparseable outside Python.
        """
        return _json_safe(dataclasses.asdict(self))


# ---------------------------------------------------------------------------
# KVEventCollector
# ---------------------------------------------------------------------------

class KVEventCollector:
    """
    Campaign-scoped subscriber for vLLM KV cache block events.

    Lifecycle:
        await collector.start()   # once, before the fuzzing loop
        ...
        await collector.stop()    # once, in a finally block after the loop

    During probes (probe-aligned window):
        start_t = time.time()
        ...run probe...
        end_t = time.time()
        summary = collector.summarize_probe(start_t, end_t)

    At KV_CORRUPTION finding time (backwards-compatible legacy path):
        forensics = collector.snapshot_forensics(window_s=30.0)
    """

    def __init__(self, endpoint: str = "tcp://localhost:5557") -> None:
        self._endpoint = endpoint
        self._buf: collections.deque[dict] = collections.deque(maxlen=_BUFFER_SIZE)
        self._task:     Optional[asyncio.Task] = None
        self._enabled:  bool = False
        self._zmq_ctx  = None
        self._sock     = None

        # lora_id -> lora_name mapping, populated from BlockStored events
        self._lora_id_to_name: dict = {}

        # Global block ownership: block_hash -> {lora_id, lora_name, first_seen_t, last_seen_t}
        # First-seen semantics: owner is never overwritten; last_seen_t is updated on all events.
        # Cleared entirely on AllBlocksCleared.
        self._block_owner: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to the ZMQ endpoint and start the background drain task.

        Never raises — any failure silently disables the collector.
        """
        try:
            import zmq
            import zmq.asyncio as zaio
            self._zmq_ctx = zaio.Context()
            self._sock    = self._zmq_ctx.socket(zmq.SUB)
            self._sock.connect(self._endpoint)
            self._sock.setsockopt_string(zmq.SUBSCRIBE, _TOPIC)
        except Exception as exc:
            print(f"[kv_events] ZMQ setup failed ({exc}) — forensics disabled")
            self._enabled = False
            return
        self._task = asyncio.create_task(self._drain_loop(), name="kv_event_drain")
        print(f"[kv_events] subscribed to {self._endpoint}")

    #: How long stop() waits for the drain task to acknowledge its cancel
    #: before giving up on it.  The task only ever awaits a socket recv, so a
    #: cancel it does not honour within seconds means it never will.
    _STOP_TASK_TIMEOUT_S: float = 5.0

    async def stop(self, final: bool = False) -> None:
        """Cancel the drain task and close the ZMQ socket.

        Every step is bounded, because this runs inside ``Runner.run``'s
        ``finally`` and used to hang there:

        * ``await self._task`` was unbounded — a drain task that does not
          honour its cancel blocks shutdown forever.  Now bounded by
          ``_STOP_TASK_TIMEOUT_S``, with ``shield`` so the timeout abandons
          the wait instead of re-cancelling a task mid-teardown.
        * ``Context.term()`` is a blocking C call.  pyzmq's asyncio socket
          finishes closing via event-loop callbacks, which cannot run while
          term() blocks the loop thread — calling it from a coroutine right
          after ``close()`` is a self-deadlock (observed on the cluster: a slot that
          had already dumped its ledger still needed the 120 s SIGKILL grace,
          rc=137).  An ``await asyncio.sleep(0)`` between the two lets those
          callbacks run first, and ``final=True`` skips term() altogether:
          at process shutdown the OS reclaims the context anyway.
        """
        if self._task is not None:
            task, self._task = self._task, None
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task),
                                       timeout=self._STOP_TASK_TIMEOUT_S)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception:
                pass
        if self._sock is not None:
            sock, self._sock = self._sock, None
            try:
                sock.close(linger=0)
            except Exception:
                pass
            # Yield once so pyzmq's close callbacks are serviced before any
            # blocking call below.
            await asyncio.sleep(0)
        if self._zmq_ctx is not None:
            ctx, self._zmq_ctx = self._zmq_ctx, None
            if not final:
                try:
                    ctx.term()
                except Exception:
                    pass

    # ──────────────────────────────────────────────────────────────────────────
    # Probe-aligned summary (primary interface)
    # ──────────────────────────────────────────────────────────────────────────

    def summarize_probe(self, start_t: float, end_t: float) -> KVProbeSummary:
        """Return a structured summary of KV activity in [start_t, end_t].

        This is the primary interface for probe-aligned analysis.  Pass
        time.time() timestamps recorded before and after a structured probe
        to get a window that is precisely scoped to that probe's activity.

        Returns a zero-value KVProbeSummary (with enabled=False indicator in
        raw_event_sample) if the collector is not enabled.
        """
        if not self._enabled:
            return KVProbeSummary(
                start_t=start_t, end_t=end_t,
                stored_count=0, removed_count=0, cleared_count=0,
                distinct_blocks=0, per_lora_counts={},
                cross_lora_reused_blocks=0, conflicting_hashes=[],
                reuse_too_fast_count=0, reuse_too_fast_min_gap_ms=float('inf'),
                hash_content_conflict_count=0, hash_content_conflicts=[],
                orphan_child_count=0,
                fingerprint="", raw_event_sample=[],
            )

        window = [e for e in self._buf if start_t <= e["t"] <= end_t]

        stored_hashes: set = set()
        # lora_key -> {"stored": int, "removed": int}
        per_lora: dict = {}
        # block_hash -> set of lora_ids seen in this window (for cross-lora detection)
        window_owners: dict = {}
        # block_hash -> most recent BlockRemoved timestamp in this window
        last_removed_t: dict = {}
        stored_raw_count = 0
        removed_raw_count = 0
        cleared = 0
        reuse_too_fast_count = 0
        reuse_too_fast_min_gap_ms = float('inf')
        # Content-based oracle tracking
        hash_to_first_tokens: dict = {}      # hash → tuple(token_ids) for first occurrence
        hash_content_conflicts_list: list = []
        hash_content_conflict_count = 0
        parent_hashes_seen: set = set()      # hashes stored so far in this window
        orphan_child_count = 0

        for e in window:
            etype = e["type"]
            if etype == "BlockStored":
                hashes = e.get("block_hashes", [])
                stored_raw_count += len(hashes)
                stored_hashes.update(hashes)
                lora_id   = e.get("lora_id")
                lora_name = e.get("lora_name") or (
                    self._lora_id_to_name.get(lora_id) if lora_id is not None else None
                )
                key = lora_name or (str(lora_id) if lora_id is not None else "unknown")
                bucket = per_lora.setdefault(key, {"stored": 0, "removed": 0})
                bucket["stored"] += len(hashes)
                if lora_id is not None:
                    for h in hashes:
                        window_owners.setdefault(h, set()).add(lora_id)
                # Check for suspiciously fast remove→store cycles
                for h in hashes:
                    prev_t = last_removed_t.get(h)
                    if prev_t is not None:
                        gap_ms = (e["t"] - prev_t) * 1000.0
                        if gap_ms <= _REUSE_TOO_FAST_MS:
                            reuse_too_fast_count += 1
                            if gap_ms < reuse_too_fast_min_gap_ms:
                                reuse_too_fast_min_gap_ms = gap_ms

                # Content-based oracle: detect same hash stored with different token_ids
                block_size    = max(e.get("block_size", 16) or 16, 1)
                token_ids_raw = e.get("token_ids", [])
                ph            = e.get("parent_block_hash")
                # Orphan check: parent referenced but not yet stored in this window
                if ph is not None and ph not in parent_hashes_seen:
                    orphan_child_count += 1
                for i, h in enumerate(hashes):
                    parent_hashes_seen.add(h)
                    start_tok = i * block_size
                    block_toks = tuple(token_ids_raw[start_tok:start_tok + block_size])
                    if not block_toks:
                        continue  # truncated by _TOKEN_CAP — skip conservatively
                    if h in hash_to_first_tokens:
                        if (hash_to_first_tokens[h] != block_toks
                                and len(hash_content_conflicts_list) < 10):
                            hash_content_conflicts_list.append({
                                "hash":            hex(h) if isinstance(h, int) else str(h),
                                "first_tokens":    list(hash_to_first_tokens[h]),
                                "conflict_tokens": list(block_toks),
                            })
                            hash_content_conflict_count += 1
                    else:
                        hash_to_first_tokens[h] = block_toks

            elif etype == "BlockRemoved":
                hashes = e.get("block_hashes", [])
                removed_raw_count += len(hashes)
                # BlockRemoved in vLLM 0.18.0 does not carry lora_id — count under "unknown"
                per_lora.setdefault("unknown", {"stored": 0, "removed": 0})
                per_lora["unknown"]["removed"] += len(hashes)
                # Record removal time for reuse-too-fast detection
                for h in hashes:
                    last_removed_t[h] = e["t"]

            elif etype == "AllBlocksCleared":
                cleared += 1
                # Clear removal timestamps — post-clear stores are not suspicious
                last_removed_t.clear()
                # Post-clear content tracking resets — conflicts after this are fresh
                parent_hashes_seen.clear()
                hash_to_first_tokens.clear()

        conflicting = [h for h, ids in window_owners.items() if len(ids) > 1]
        conflicting_hashes = [hex(h) for h in conflicting[:_EVICTED_CAP]]

        fp = self._build_fingerprint(
            stored_count=len(stored_hashes),
            removed_count=removed_raw_count,
            cleared_count=cleared,
            cross_lora_reused_blocks=len(conflicting),
            reuse_too_fast_count=reuse_too_fast_count,
            per_lora=per_lora,
            hash_content_conflict_count=hash_content_conflict_count,
            orphan_child_count=orphan_child_count,
        )

        return KVProbeSummary(
            start_t=start_t,
            end_t=end_t,
            stored_count=stored_raw_count,
            removed_count=removed_raw_count,
            cleared_count=cleared,
            distinct_blocks=len(stored_hashes),
            per_lora_counts=per_lora,
            cross_lora_reused_blocks=len(conflicting),
            conflicting_hashes=conflicting_hashes,
            reuse_too_fast_count=reuse_too_fast_count,
            reuse_too_fast_min_gap_ms=reuse_too_fast_min_gap_ms,
            hash_content_conflict_count=hash_content_conflict_count,
            hash_content_conflicts=hash_content_conflicts_list,
            orphan_child_count=orphan_child_count,
            fingerprint=fp,
            raw_event_sample=window[-_RAW_EVENTS_CAP:],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy forensics snapshot (backwards-compatible wrapper)
    # ──────────────────────────────────────────────────────────────────────────

    def snapshot_forensics(self, window_s: float = 30.0) -> dict:
        """Return a forensics dict for embedding in a finding JSON.

        Thin wrapper around summarize_probe() that adds legacy field aliases
        so existing KV_CORRUPTION finding consumers continue to work.

        Legacy fields preserved:
            kv_events_enabled     bool
            kv_events_window_s    float
            kv_events_count       int    (stored + removed + cleared event counts)
            kv_events_type_counts dict
            kv_cleared_fired      bool
            kv_evicted_hashes     list   (hashes stored AND removed in window)
            kv_raw_events         list
        """
        if not self._enabled:
            return {"kv_events_enabled": False}

        end_t   = time.time()
        start_t = end_t - window_s
        summary = self.summarize_probe(start_t, end_t)

        # Reconstruct legacy type_counts for backwards compatibility
        window = summary.raw_event_sample  # already the last ≤200 in window
        type_counts: dict = {}
        for e in window:
            t = e["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        # kv_evicted_hashes: blocks both stored AND removed in the window
        # (this is the original TOCTOU aliasing signal, separate from cross-lora conflicts)
        stored_set:  set = set()
        removed_set: set = set()
        for e in self._buf:
            if e["t"] < start_t:
                continue
            if e["type"] == "BlockStored":
                stored_set.update(e.get("block_hashes", []))
            elif e["type"] == "BlockRemoved":
                removed_set.update(e.get("block_hashes", []))
        evicted = sorted(stored_set & removed_set)[:_EVICTED_CAP]

        # Count event records (not hash occurrences) to preserve original semantics
        event_count = sum(1 for e in self._buf if e["t"] >= start_t)

        return {
            "kv_events_enabled":     True,
            "kv_events_window_s":    window_s,
            # New structured fields from KVProbeSummary
            **summary.to_dict(),
            # Legacy alias fields
            "kv_events_count":       event_count,
            "kv_events_type_counts": type_counts,
            "kv_cleared_fired":      summary.cleared_count > 0,
            "kv_evicted_hashes":     evicted,
            "kv_raw_events":         summary.raw_event_sample,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Public read-only accessors
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def lora_id_to_name(self) -> dict:
        """Read-only view of the accumulated lora_id → lora_name mapping."""
        return dict(self._lora_id_to_name)

    @property
    def block_owner(self) -> dict:
        """Read-only view of the global block ownership map."""
        return dict(self._block_owner)

    def get_block_snapshot(self, start_t: float, end_t: float) -> dict:
        """Return {hash: token_ids_tuple} for BlockStored events in [start_t, end_t].

        Only includes blocks with non-empty token_ids (skips entries truncated by
        _TOKEN_CAP).  Returns {} when the collector is not enabled.

        Used for cross-run token content comparison in the structural repro loop.
        First-seen semantics: if the same hash appears multiple times in the window,
        the first token_ids snapshot is kept.
        """
        if not self._enabled:
            return {}
        snapshot: dict = {}
        for e in self._buf:
            if e["t"] < start_t or e["t"] > end_t:
                continue
            if e["type"] != "BlockStored":
                continue
            hashes        = e.get("block_hashes", [])
            token_ids_raw = e.get("token_ids", [])
            block_size    = max(e.get("block_size", 16) or 16, 1)
            for i, h in enumerate(hashes):
                if h in snapshot:
                    continue  # first-seen semantics
                start_tok = i * block_size
                toks = tuple(token_ids_raw[start_tok:start_tok + block_size])
                if toks:
                    snapshot[h] = toks
        return snapshot

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        assert self._sock is not None
        try:
            while True:
                try:
                    parts = await asyncio.wait_for(
                        self._sock.recv_multipart(),
                        timeout=_DRAIN_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    continue
                if len(parts) < 3:
                    continue
                for rec in self._decode_batch(parts[2]):
                    self._buf.append(rec)
                    self._update_ownership(rec)
                if not self._enabled:
                    self._enabled = True
                    print("[kv_events] receiving block events")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[kv_events] drain loop error ({exc}) — forensics disabled")
            self._enabled = False

    def _update_ownership(self, rec: dict) -> None:
        """Update global block ownership state from a freshly decoded event."""
        etype = rec["type"]
        if etype == "BlockStored":
            now       = rec["t"]
            lora_id   = rec.get("lora_id")
            lora_name = rec.get("lora_name")
            if lora_id is not None and lora_name is not None:
                self._lora_id_to_name[lora_id] = lora_name
            for h in rec.get("block_hashes", []):
                existing = self._block_owner.get(h)
                if existing is None:
                    self._block_owner[h] = {
                        "lora_id":      lora_id,
                        "lora_name":    lora_name,
                        "first_seen_t": now,
                        "last_seen_t":  now,
                    }
                else:
                    # First-seen semantics: preserve original owner, update recency
                    existing["last_seen_t"] = now

        elif etype == "BlockRemoved":
            now = rec["t"]
            for h in rec.get("block_hashes", []):
                if h in self._block_owner:
                    self._block_owner[h]["last_seen_t"] = now

        elif etype == "AllBlocksCleared":
            # Full cache flush — reset all ownership; conflicts after this are fresh
            self._block_owner.clear()

    @staticmethod
    def _decode_batch(raw: bytes) -> list[dict]:
        try:
            import msgspec
            import msgspec.msgpack
        except ImportError:
            return []

        # Vendor vLLM 0.18.0 KV event types locally so this works without
        # the host having vllm 0.18.0 installed.  These definitions must match
        # exactly what the container publishes (array_like=True, tag=True for
        # tagged union discrimination).
        try:
            from vllm.distributed.kv_events import (
                KVEventBatch, BlockStored, BlockRemoved, AllBlocksCleared,
            )
        except Exception:
            # Vendored fallback — mirrors vllm/distributed/kv_events.py v0.18.0
            class KVCacheEvent(msgspec.Struct, array_like=True,
                               omit_defaults=True, gc=False, tag=True):
                pass

            class BlockStored(KVCacheEvent):  # type: ignore[no-redef]
                block_hashes:      list
                parent_block_hash: object
                token_ids:         list
                block_size:        int
                lora_id:           object
                medium:            object
                lora_name:         object
                extra_keys:        object = None

            class BlockRemoved(KVCacheEvent):  # type: ignore[no-redef]
                block_hashes: list
                medium:       object

            class AllBlocksCleared(KVCacheEvent):  # type: ignore[no-redef]
                pass

            class EventBatch(msgspec.Struct, array_like=True,
                             omit_defaults=True, gc=False):
                ts:                 float
                events:             list
                data_parallel_rank: object = None

            from typing import Union
            class KVEventBatch(EventBatch):  # type: ignore[no-redef]
                events: list[Union[BlockStored, BlockRemoved, AllBlocksCleared]]

        # Try vLLM typed decode first; cache the result so subsequent
        # messages skip the failing path (format won't change mid-stream).
        if not hasattr(KVEventCollector, '_decode_mode'):
            KVEventCollector._decode_mode = "unknown"

        if KVEventCollector._decode_mode != "sglang":
            try:
                batch = msgspec.msgpack.decode(raw, type=KVEventBatch)
                KVEventCollector._decode_mode = "vllm"
            except Exception:
                batch = None
        else:
            batch = None

        # SGLang uses a different tag encoding (string tag as first list element)
        # that msgspec's tag=True cannot decode. Fall back to raw decode +
        # manual parsing for SGLang-emitted events.
        if batch is None:
            try:
                decoded = msgspec.msgpack.decode(raw)
                if isinstance(decoded, (list, tuple)) and len(decoded) >= 2:
                    KVEventCollector._decode_mode = "sglang"
                    return KVEventCollector._decode_raw_events(decoded[1])
            except Exception:
                return []
            return []

        now = time.time()
        result: list[dict] = []
        for ev in batch.events:
            if isinstance(ev, BlockStored):
                result.append({
                    "type":              "BlockStored",
                    "t":                 now,
                    "block_hashes":      list(ev.block_hashes),
                    "parent_block_hash": ev.parent_block_hash,
                    # Truncate token_ids to keep finding JSON sizes sane.
                    "token_ids":         list(ev.token_ids[:_TOKEN_CAP]),
                    "block_size":        ev.block_size,
                    "lora_id":           ev.lora_id,
                    "lora_name":         getattr(ev, "lora_name", None),
                    "medium":            ev.medium,
                })
            elif isinstance(ev, BlockRemoved):
                result.append({
                    "type":         "BlockRemoved",
                    "t":            now,
                    "block_hashes": list(ev.block_hashes),
                    "medium":       ev.medium,
                    # lora_id not emitted by vLLM 0.18.0 on remove events
                })
            elif isinstance(ev, AllBlocksCleared):
                result.append({
                    "type": "AllBlocksCleared",
                    "t":    now,
                })
        return result

    @staticmethod
    def _decode_raw_events(events_list: list) -> list[dict]:
        """Parse raw-decoded event lists (SGLang format).

        Each event is [tag, ...fields] where tag is a string like "BlockStored".
        SGLang BlockStored: [tag, block_hashes, parent_hash, token_ids, block_size, lora_id, medium]
        SGLang BlockRemoved: [tag, block_hashes, medium]
        SGLang AllBlocksCleared: [tag]
        """
        now = time.time()
        result: list[dict] = []
        for ev in events_list:
            if not isinstance(ev, (list, tuple)) or len(ev) < 1:
                continue
            tag = ev[0]
            if tag == "BlockStored" and len(ev) >= 6:
                result.append({
                    "type":              "BlockStored",
                    "t":                 now,
                    "block_hashes":      list(ev[1]),
                    "parent_block_hash": ev[2],
                    "token_ids":         list(ev[3][:_TOKEN_CAP]) if ev[3] else [],
                    "block_size":        ev[4],
                    "lora_id":           ev[5],
                    "lora_name":         None,
                    "medium":            ev[6] if len(ev) > 6 else None,
                })
            elif tag == "BlockRemoved" and len(ev) >= 2:
                result.append({
                    "type":         "BlockRemoved",
                    "t":            now,
                    "block_hashes": list(ev[1]),
                    "medium":       ev[2] if len(ev) > 2 else None,
                })
            elif tag == "AllBlocksCleared":
                result.append({
                    "type": "AllBlocksCleared",
                    "t":    now,
                })
        return result

    @staticmethod
    def _build_fingerprint(
        stored_count: int,
        removed_count: int,
        cleared_count: int,
        cross_lora_reused_blocks: int,
        reuse_too_fast_count: int,
        per_lora: dict,
        hash_content_conflict_count: int = 0,
        orphan_child_count: int = 0,
    ) -> str:
        """Build an order-insensitive fingerprint of the probe summary structure.

        Does NOT hash raw event order or exact timestamps — avoids scheduler jitter.
        Suitable for dedup and clustering of structurally similar probe windows.

        **Every counter is log2-bucketed before hashing.**  Until 2026-08-23
        this hashed the raw counters plus the raw per-LoRA dict, which made the
        fingerprint very nearly unique per trace: it changed whenever a single
        extra block was stored.  `Corpus.add()` treats it as an exact-match
        novelty key, so it fired on roughly every other iteration and pinned
        the entry score at _KV_NOVELTY_SCORE — 5686 of 10185 persisted corpus
        entries sit at exactly 1.5 because of it
        (an internal analysis note).  An exact-match signal
        over an unbounded key space is always novel, which is exactly as
        useless as never novel.

        Bucketing to `n.bit_length()` (0, 1, 2-3, 4-7, 8-15, ...) collapses
        that to a key space comparable to the scheduler fingerprint: 100 and
        120 stored blocks share a bucket (7), 100 and 1000 do not (7 vs 10).
        The per-LoRA sub-counts are bucketed the same way; the LoRA *ids* are
        kept because which adapter touched a block is structural, not
        quantitative.

        The raw counters are untouched on KVProbeSummary itself and still reach
        `collection_forensics` in every finding — only the hash input is
        coarsened.
        """
        def _bucket(n) -> int:
            """log2 bucket of a non-negative count: 0→0, 1→1, 2-3→2, 4-7→3, ..."""
            try:
                n = int(n)
            except (TypeError, ValueError):
                return 0
            return n.bit_length() if n > 0 else 0

        canonical = {
            "stored":          _bucket(stored_count),
            "removed":         _bucket(removed_count),
            "cleared":         _bucket(cleared_count),
            "cross_lora":      _bucket(cross_lora_reused_blocks),
            "reuse_too_fast":  _bucket(reuse_too_fast_count),
            "hash_conflicts":  _bucket(hash_content_conflict_count),
            "orphan_children": _bucket(orphan_child_count),
            "per_lora": {
                str(k): {sk: _bucket(sv) for sk, sv in sorted(v.items())}
                for k, v in sorted(per_lora.items(), key=lambda kv: str(kv[0]))
            },
        }
        return hashlib.md5(
            json.dumps(canonical, sort_keys=True).encode()
        ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Structural anomaly analysis (dependency-free — returns plain dicts)
# ---------------------------------------------------------------------------

# Score ladder for generic block-lifecycle anomalies:
#   reuse_too_fast (< 2 ms):   5.0  → structural repro queue
#   reuse_too_fast (2–10 ms):  3.0  → corpus novelty only
#
# Note: cross_lora_reuse (score 6.0) is LoRA-specific and is scored by
# VllmLoRAAdapter.analyze_kv_summary() via adapters/vllm_lora/oracles.py.
_REUSE_FAST_HARD_SCORE        = 5.0   # gap < 2 ms
_REUSE_FAST_SOFT_SCORE        = 3.0   # gap 2–10 ms
_REUSE_FAST_HARD_THRESHOLD_MS = 2.0


def analyze_kv_anomalies(summary: KVProbeSummary) -> list:
    """Score generic block-lifecycle anomalies from a trace-window probe summary.

    Returns a list of anomaly dicts. Each dict has:
        anomaly_type  str    "reuse_too_fast"
        count         int    number of affected blocks
        score         float  triage score (see ladder above)
        detail        str    human-readable description
        metadata      dict   anomaly-specific fields for finding JSON

    Returns an empty list if the summary has no anomalies or the collector
    was not enabled (fingerprint == "").

    Intentionally has no imports from fuzzer.core — keeps kv_events.py
    usable without the full fuzzer package installed.

    Adapter-specific anomalies (e.g. cross_lora_reuse) are NOT scored here.
    They are returned by adapter.analyze_kv_summary(summary) and merged by
    the runner:
        kv_anomalies = analyze_kv_anomalies(summary)
        if adapter is not None:
            kv_anomalies += adapter.analyze_kv_summary(summary)
    """
    if not summary.fingerprint:
        # Collector was not enabled; summary is a zero-value placeholder
        return []

    results = []

    # --- Suspiciously fast remove→store cycle ---------------------------
    if summary.reuse_too_fast_count > 0:
        gap = summary.reuse_too_fast_min_gap_ms
        score = (
            _REUSE_FAST_HARD_SCORE if gap < _REUSE_FAST_HARD_THRESHOLD_MS
            else _REUSE_FAST_SOFT_SCORE
        )
        results.append({
            "anomaly_type": "reuse_too_fast",
            "count":  summary.reuse_too_fast_count,
            "score":  score,
            "detail": (
                f"{summary.reuse_too_fast_count} block(s) removed then re-stored "
                f"within {gap:.2f} ms (threshold {_REUSE_TOO_FAST_MS} ms) "
                f"in trace window [{summary.start_t:.3f}, {summary.end_t:.3f}]"
            ),
            # This metadata dict is embedded verbatim in the finding JSON, so
            # the gap goes through the same boundary sanitiser as to_dict().
            "metadata": {
                "reuse_too_fast_count":      summary.reuse_too_fast_count,
                "reuse_too_fast_min_gap_ms": _json_safe(gap),
            },
        })

    # --- Same block_hash stored with different token_ids (content violation) ----
    # Score 8.0 — above structural repro threshold (5.0), so it auto-enters the
    # k-run confirmation loop.  Stronger than timing-based reuse_too_fast because
    # it is a direct content inconsistency, not just suspicious timing.
    if summary.hash_content_conflict_count > 0:
        results.append({
            "anomaly_type": "hash_content_conflict",
            "count":  summary.hash_content_conflict_count,
            "score":  8.0,
            "detail": (
                f"{summary.hash_content_conflict_count} block(s) stored with "
                f"conflicting token_ids for the same block_hash in window "
                f"[{summary.start_t:.3f}, {summary.end_t:.3f}]"
            ),
            "metadata": {
                "hash_content_conflict_count": summary.hash_content_conflict_count,
                "conflicts_sample":            summary.hash_content_conflicts[:3],
            },
        })

    # --- Child block references parent not stored in this window (weak signal) --
    # Score 1.5 — below structural repro threshold; useful only as corpus novelty.
    # False-fires are expected: parent may have been created before the window.
    if summary.orphan_child_count > 0:
        results.append({
            "anomaly_type": "orphan_child_block",
            "count":  summary.orphan_child_count,
            "score":  1.5,
            "detail": (
                f"{summary.orphan_child_count} block(s) referenced a parent_block_hash "
                f"not yet stored in this window (may be pre-existing — weak signal)"
            ),
            "metadata": {"orphan_child_count": summary.orphan_child_count},
        })

    return results


# ---------------------------------------------------------------------------
# Cross-run snapshot comparison
# ---------------------------------------------------------------------------

def compare_block_snapshots(snap1: dict, snap2: dict) -> list:
    """Compare two {hash: token_ids_tuple} snapshots from get_block_snapshot().

    Returns a list of conflict dicts for hashes present in both snapshots with
    non-empty, differing token_ids.

    Conservative: empty tuples (from _TOKEN_CAP truncation) are skipped so that
    blocks whose full token content was not captured don't produce false conflicts.

    Typical use: call after each of k structural-repro runs, then compare
    snap[0] vs snap[-1] — any conflict means the same block was filled with
    different content across runs, confirming reproducible corruption.
    """
    conflicts = []
    for h, toks1 in snap1.items():
        toks2 = snap2.get(h)
        if toks2 is None or not toks1 or not toks2:
            continue
        if toks1 != toks2:
            conflicts.append({
                "hash":        hex(h) if isinstance(h, int) else str(h),
                "run1_tokens": list(toks1[:16]),
                "run2_tokens": list(toks2[:16]),
            })
    return conflicts
