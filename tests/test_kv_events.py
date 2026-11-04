"""Tests for the KV events pipeline: KVProbeSummary, summarize_probe(),
ownership tracking, fingerprinting, and snapshot_forensics() compatibility."""

import json
import time
import collections

import pytest

from fuzzer.kv_events import KVEventCollector, KVProbeSummary


# ---------------------------------------------------------------------------
# Helpers — build a collector with pre-loaded events (no ZMQ needed)
# ---------------------------------------------------------------------------

def _make_collector(*events: dict) -> KVEventCollector:
    """Return a KVEventCollector with _enabled=True and a pre-filled buffer."""
    c = KVEventCollector.__new__(KVEventCollector)
    c._endpoint = "tcp://localhost:5557"
    c._buf = collections.deque(events, maxlen=2000)
    c._task = None
    c._enabled = True
    c._zmq_ctx = None
    c._sock = None
    c._lora_id_to_name = {}
    c._block_owner = {}
    # Replay ownership from the events already in the buffer
    for e in events:
        c._update_ownership(e)
    return c


def _stored(block_hashes, lora_id=None, lora_name=None, t=1.0):
    return {
        "type": "BlockStored",
        "t": t,
        "block_hashes": block_hashes,
        "parent_block_hash": None,
        "token_ids": [],
        "block_size": 16,
        "lora_id": lora_id,
        "lora_name": lora_name,
        "medium": "GPU",
    }


def _removed(block_hashes, t=2.0):
    return {
        "type": "BlockRemoved",
        "t": t,
        "block_hashes": block_hashes,
        "medium": "GPU",
    }


def _cleared(t=3.0):
    return {"type": "AllBlocksCleared", "t": t}


# ---------------------------------------------------------------------------
# A. lora_name preservation in decoded events
# ---------------------------------------------------------------------------

class TestLoraNamesPreserved:
    def test_lora_id_to_name_mapping_updated_on_stored(self):
        c = _make_collector(_stored([1001], lora_id=42, lora_name="lora_a"))
        assert c._lora_id_to_name.get(42) == "lora_a"

    def test_lora_id_to_name_not_updated_when_lora_name_is_none(self):
        c = _make_collector(_stored([1001], lora_id=42, lora_name=None))
        assert 42 not in c._lora_id_to_name

    def test_lora_id_to_name_property_returns_copy(self):
        c = _make_collector(_stored([1001], lora_id=1, lora_name="x"))
        mapping = c.lora_id_to_name
        mapping[99] = "tampered"
        assert 99 not in c._lora_id_to_name

    def test_second_stored_same_id_updates_mapping(self):
        c = _make_collector(
            _stored([1001], lora_id=1, lora_name="old_name", t=1.0),
            _stored([1002], lora_id=1, lora_name="new_name", t=2.0),
        )
        # Most recent wins for lora_id_to_name (last write wins for the mapping)
        assert c._lora_id_to_name.get(1) == "new_name"


# ---------------------------------------------------------------------------
# B. Global block ownership tracking
# ---------------------------------------------------------------------------

class TestBlockOwnership:
    def test_first_stored_creates_owner_entry(self):
        c = _make_collector(_stored([0xABCD], lora_id=1, lora_name="lora_a", t=1.0))
        entry = c._block_owner.get(0xABCD)
        assert entry is not None
        assert entry["lora_id"] == 1
        assert entry["lora_name"] == "lora_a"

    def test_second_stored_same_hash_same_lora_does_not_overwrite_first_seen(self):
        c = _make_collector(
            _stored([0xABCD], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([0xABCD], lora_id=1, lora_name="lora_a", t=5.0),
        )
        entry = c._block_owner[0xABCD]
        assert entry["first_seen_t"] == 1.0
        assert entry["last_seen_t"] == 5.0

    def test_removed_updates_last_seen_but_does_not_delete_entry(self):
        c = _make_collector(
            _stored([0xABCD], lora_id=1, lora_name="lora_a", t=1.0),
            _removed([0xABCD], t=3.0),
        )
        assert 0xABCD in c._block_owner
        assert c._block_owner[0xABCD]["last_seen_t"] == 3.0

    def test_all_blocks_cleared_empties_ownership_map(self):
        c = _make_collector(
            _stored([0x1111, 0x2222], lora_id=1, lora_name="lora_a", t=1.0),
            _cleared(t=2.0),
        )
        assert c._block_owner == {}


# ---------------------------------------------------------------------------
# C. summarize_probe() — disabled collector
# ---------------------------------------------------------------------------

class TestSummarizeProbeDisabled:
    def test_disabled_collector_returns_zero_summary(self):
        c = KVEventCollector.__new__(KVEventCollector)
        c._endpoint = "tcp://localhost:5557"
        c._buf = collections.deque(maxlen=2000)
        c._task = None
        c._enabled = False
        c._zmq_ctx = None
        c._sock = None
        c._lora_id_to_name = {}
        c._block_owner = {}

        s = c.summarize_probe(0.0, 10.0)
        assert s.stored_count == 0
        assert s.removed_count == 0
        assert s.cleared_count == 0
        assert s.cross_lora_reused_blocks == 0
        assert s.conflicting_hashes == []
        assert s.raw_event_sample == []


# ---------------------------------------------------------------------------
# D. summarize_probe() — empty window
# ---------------------------------------------------------------------------

class TestSummarizeProbeEmptyWindow:
    def test_no_events_in_window_returns_zeros(self):
        # Events at t=1.0 are outside window [5.0, 9.0]
        c = _make_collector(
            _stored([0x1111], lora_id=1, lora_name="lora_a", t=1.0),
        )
        s = c.summarize_probe(5.0, 9.0)
        assert s.stored_count == 0
        assert s.removed_count == 0
        assert s.distinct_blocks == 0
        assert s.cross_lora_reused_blocks == 0
        assert s.per_lora_counts == {}


# ---------------------------------------------------------------------------
# E. summarize_probe() — single-adapter activity
# ---------------------------------------------------------------------------

class TestSummarizeProbeOneAdapter:
    def test_counts_stored_correctly(self):
        c = _make_collector(
            _stored([0x1, 0x2, 0x3], lora_id=1, lora_name="lora_a", t=1.0),
        )
        s = c.summarize_probe(0.5, 1.5)
        assert s.stored_count == 3
        assert s.distinct_blocks == 3
        assert s.cross_lora_reused_blocks == 0

    def test_per_lora_stored_count(self):
        c = _make_collector(
            _stored([0x1, 0x2], lora_id=1, lora_name="lora_a", t=1.0),
        )
        s = c.summarize_probe(0.5, 1.5)
        assert s.per_lora_counts.get("lora_a", {}).get("stored") == 2

    def test_removed_count_under_unknown_key(self):
        # BlockRemoved has no lora_id → counted under "unknown"
        c = _make_collector(
            _stored([0x1, 0x2], lora_id=1, lora_name="lora_a", t=1.0),
            _removed([0x1, 0x2], t=1.5),
        )
        s = c.summarize_probe(0.5, 2.0)
        assert s.removed_count == 2
        assert s.per_lora_counts.get("unknown", {}).get("removed") == 2

    def test_no_cross_lora_conflict_single_adapter(self):
        c = _make_collector(
            _stored([0xAA, 0xBB], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([0xCC], lora_id=1, lora_name="lora_a", t=1.5),
        )
        s = c.summarize_probe(0.5, 2.0)
        assert s.cross_lora_reused_blocks == 0
        assert s.conflicting_hashes == []


# ---------------------------------------------------------------------------
# F. summarize_probe() — cross-adapter conflict (THE KEY SIGNAL)
# ---------------------------------------------------------------------------

class TestSummarizeProbeCrossAdapter:
    def test_same_hash_two_lora_ids_is_conflict(self):
        H = 0xDEADBEEF
        c = _make_collector(
            _stored([H], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([H], lora_id=2, lora_name="lora_b", t=1.5),
        )
        s = c.summarize_probe(0.5, 2.0)
        assert s.cross_lora_reused_blocks == 1
        assert hex(H) in s.conflicting_hashes

    def test_multiple_conflicting_hashes(self):
        c = _make_collector(
            _stored([0xA1, 0xA2, 0xA3], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([0xA1, 0xA2, 0xA4], lora_id=2, lora_name="lora_b", t=1.5),
        )
        s = c.summarize_probe(0.5, 2.0)
        # 0xA1 and 0xA2 appear under both lora_ids — 2 conflicts
        assert s.cross_lora_reused_blocks == 2

    def test_conflict_only_when_both_in_window(self):
        # First stored at t=1.0 is inside window; second at t=10.0 is outside
        H = 0xBEEF
        c = _make_collector(
            _stored([H], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([H], lora_id=2, lora_name="lora_b", t=10.0),
        )
        s = c.summarize_probe(0.5, 2.0)   # window ends at 2.0
        # Only one event in window → no conflict
        assert s.cross_lora_reused_blocks == 0

    def test_lora_id_fallback_from_mapping(self):
        # lora_name is None in the event but lora_id is in the mapping
        c = _make_collector()
        c._lora_id_to_name[1] = "lora_a"
        c._lora_id_to_name[2] = "lora_b"
        H = 0xF00D
        # Manually inject events without lora_name
        e1 = _stored([H], lora_id=1, lora_name=None, t=1.0)
        e2 = _stored([H], lora_id=2, lora_name=None, t=1.5)
        c._buf.append(e1)
        c._buf.append(e2)
        s = c.summarize_probe(0.5, 2.0)
        # Both lora_ids are distinct → conflict
        assert s.cross_lora_reused_blocks == 1


# ---------------------------------------------------------------------------
# G. Time window filtering
# ---------------------------------------------------------------------------

class TestTimeWindowFiltering:
    def test_only_events_in_window_counted(self):
        # 2 events at t=1.0 (inside), 3 events at t=5.0 (outside window [0.5, 2.0])
        c = _make_collector(
            _stored([0x1, 0x2], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([0x3, 0x4, 0x5], lora_id=1, lora_name="lora_a", t=5.0),
        )
        s = c.summarize_probe(0.5, 2.0)
        assert s.stored_count == 2
        assert s.distinct_blocks == 2

    def test_window_boundaries_are_inclusive(self):
        c = _make_collector(
            _stored([0xAA], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([0xBB], lora_id=1, lora_name="lora_a", t=2.0),
        )
        s = c.summarize_probe(1.0, 2.0)
        assert s.stored_count == 2

    def test_all_blocks_cleared_in_window_counted(self):
        c = _make_collector(
            _cleared(t=1.5),
            _cleared(t=1.8),
        )
        s = c.summarize_probe(1.0, 2.0)
        assert s.cleared_count == 2

    def test_raw_event_sample_bounded(self):
        # Insert 250 events — sample should be capped at 200
        events = [_stored([i], lora_id=1, lora_name="lora_a", t=float(i))
                  for i in range(1, 251)]
        c = _make_collector(*events)
        s = c.summarize_probe(0.5, 300.0)
        assert len(s.raw_event_sample) <= 200


# ---------------------------------------------------------------------------
# H. Fingerprint stability and sensitivity
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_same_structure_same_fingerprint(self):
        c = _make_collector(
            _stored([0x1, 0x2], lora_id=1, lora_name="lora_a", t=1.0),
        )
        s1 = c.summarize_probe(0.5, 2.0)
        s2 = c.summarize_probe(0.5, 2.0)
        assert s1.fingerprint == s2.fingerprint
        assert len(s1.fingerprint) == 16

    def test_cross_lora_conflict_changes_fingerprint(self):
        H = 0xFACE
        c_clean = _make_collector(
            _stored([H], lora_id=1, lora_name="lora_a", t=1.0),
        )
        c_dirty = _make_collector(
            _stored([H], lora_id=1, lora_name="lora_a", t=1.0),
            _stored([H], lora_id=2, lora_name="lora_b", t=1.5),
        )
        s_clean = c_clean.summarize_probe(0.5, 2.0)
        s_dirty = c_dirty.summarize_probe(0.5, 2.0)
        assert s_clean.fingerprint != s_dirty.fingerprint

    def test_cleared_count_changes_fingerprint(self):
        c1 = _make_collector(_cleared(t=1.0))
        c2 = _make_collector(_cleared(t=1.0), _cleared(t=1.5))
        s1 = c1.summarize_probe(0.5, 2.0)
        s2 = c2.summarize_probe(0.5, 2.0)
        assert s1.fingerprint != s2.fingerprint

    def test_fingerprint_not_sensitive_to_exact_timestamps(self):
        # Same structural counts but different event timestamps → same fingerprint
        # (fingerprint only hashes counts/ratios, not raw timestamps)
        c1 = _make_collector(_stored([0xAA], lora_id=1, lora_name="lora_a", t=1.0))
        c2 = _make_collector(_stored([0xBB], lora_id=1, lora_name="lora_a", t=99.9))
        s1 = c1.summarize_probe(0.5, 2.0)
        s2 = c2.summarize_probe(50.0, 150.0)
        # Both have: stored=1, removed=0, cleared=0, cross_lora=0, per_lora={lora_a:{stored:1}}
        assert s1.fingerprint == s2.fingerprint


# ---------------------------------------------------------------------------
# H2. Fingerprint counter bucketing (step 8 of the dedup/novelty plan)
# ---------------------------------------------------------------------------

class TestFingerprintBucketing:
    """The fingerprint hashes log2 buckets of its counters, not the raw counts.

    Before this change the fingerprint moved whenever a single extra block was
    stored, so `Corpus.add()`'s exact-match `novel_kv` signal fired on roughly
    every other iteration and pinned entries at _KV_NOVELTY_SCORE
    (an internal analysis note).
    """

    def _summary_with_n_blocks(self, n: int):
        c = _make_collector(
            _stored(list(range(1, n + 1)), lora_id=1, lora_name="lora_a", t=1.0)
        )
        return c.summarize_probe(0.5, 2.0)

    # Note: the interior sample points (100-vs-120 collapse, 100-vs-1000
    # separation, and the per-LoRA variant of the same collapse) were dropped —
    # test_bucket_boundaries_are_powers_of_two below asserts both directions at
    # the actual bucket edges, which subsumes all three.  The helper stores
    # every block under lora_a, so the per-LoRA sub-counts are bucketed by that
    # test too: the 64-vs-127 equality needs both the top-level and per-LoRA
    # counts to collapse.

    def test_raw_counters_are_not_bucketed(self):
        """Only the hash input is coarsened — collection_forensics still gets
        the exact counts."""
        s100 = self._summary_with_n_blocks(100)
        s120 = self._summary_with_n_blocks(120)
        assert s100.stored_count == 100
        assert s120.stored_count == 120
        assert s100.distinct_blocks == 100
        assert s120.distinct_blocks == 120

    def test_bucket_boundaries_are_powers_of_two(self):
        # 63 → bucket 6, 64 → bucket 7, 127 → bucket 7, 128 → bucket 8.
        fp = {n: self._summary_with_n_blocks(n).fingerprint
              for n in (63, 64, 127, 128)}
        assert fp[63] != fp[64]
        assert fp[64] == fp[127]
        assert fp[127] != fp[128]

    def test_key_space_stays_small_over_a_sweep(self):
        """A sweep of 1..256 block counts yields ~log2 distinct fingerprints,
        not 256 of them — that is the whole point of the change."""
        fps = {self._summary_with_n_blocks(n).fingerprint
               for n in range(1, 257)}
        assert len(fps) <= 12


# ---------------------------------------------------------------------------
# I. snapshot_forensics() backwards compatibility
# ---------------------------------------------------------------------------

class TestSnapshotForensicsBackwardsCompat:
    def test_enabled_collector_returns_kv_events_enabled_true(self):
        c = _make_collector(_stored([0x1], lora_id=1, lora_name="lora_a", t=time.time()))
        result = c.snapshot_forensics(window_s=30.0)
        assert result["kv_events_enabled"] is True

    def test_disabled_collector_returns_kv_events_enabled_false(self):
        c = KVEventCollector.__new__(KVEventCollector)
        c._endpoint = "tcp://localhost:5557"
        c._buf = collections.deque(maxlen=2000)
        c._task = None
        c._enabled = False
        c._zmq_ctx = None
        c._sock = None
        c._lora_id_to_name = {}
        c._block_owner = {}
        result = c.snapshot_forensics()
        assert result["kv_events_enabled"] is False

    def test_legacy_field_kv_cleared_fired(self):
        c = _make_collector(_cleared(t=time.time()))
        result = c.snapshot_forensics(window_s=30.0)
        assert "kv_cleared_fired" in result
        assert result["kv_cleared_fired"] is True

    def test_legacy_field_kv_evicted_hashes_present(self):
        now = time.time()
        c = _make_collector(
            _stored([0x1234], lora_id=1, lora_name="lora_a", t=now),
            _removed([0x1234], t=now + 0.1),
        )
        result = c.snapshot_forensics(window_s=30.0)
        assert "kv_evicted_hashes" in result
        assert 0x1234 in result["kv_evicted_hashes"]

    def test_legacy_field_kv_events_count(self):
        now = time.time()
        c = _make_collector(
            _stored([0x1, 0x2], lora_id=1, lora_name="lora_a", t=now),
            _removed([0x1], t=now + 0.1),
            _cleared(t=now + 0.2),
        )
        result = c.snapshot_forensics(window_s=30.0)
        assert "kv_events_count" in result
        # 3 event records in window (not hash-occurrence count)
        assert result["kv_events_count"] == 3

    def test_new_fields_also_present(self):
        c = _make_collector(_stored([0x1], lora_id=1, lora_name="lora_a", t=time.time()))
        result = c.snapshot_forensics(window_s=30.0)
        # KVProbeSummary fields should be present via to_dict()
        assert "cross_lora_reused_blocks" in result
        assert "per_lora_counts" in result
        assert "fingerprint" in result


# ---------------------------------------------------------------------------
# J. KVProbeSummary.to_dict() is JSON-serializable
# ---------------------------------------------------------------------------

class TestKVProbeSummarySerializable:
    """to_dict() must survive ``allow_nan=False`` — i.e. emit *valid* JSON.

    Regression.  ``reuse_too_fast_min_gap_ms`` defaults to ``float('inf')`` (it
    is the identity for the running min in summarize_probe), and the summary is
    splatted into every finding report.  ``json.dumps`` renders that as a bare
    ``Infinity`` token, which no JSON parser outside Python accepts; 30+
    archived findings on disk carry ``"reuse_too_fast_min_gap_ms": Infinity``.

    The three tests this class used to hold all called plain ``json.dumps``,
    which happily emits ``Infinity`` — so none of them could see the bug.
    ``allow_nan=False`` is the whole point: it is what makes these tests fail
    against the unfixed ``to_dict``.
    """

    def test_default_summary_round_trips_as_strict_json(self):
        # No reuse_too_fast events were observed → the field is inf in memory.
        s = KVProbeSummary(
            start_t=1.0, end_t=2.0,
            stored_count=5, removed_count=3,
            cleared_count=0, distinct_blocks=5,
            per_lora_counts={"lora_a": {"stored": 5, "removed": 0}},
            cross_lora_reused_blocks=1,
            conflicting_hashes=["0xdeadbeef"],
            fingerprint="abc123def456abcd",
            raw_event_sample=[],
        )
        assert s.reuse_too_fast_min_gap_ms == float('inf')   # unchanged in memory

        d = s.to_dict()
        assert d["reuse_too_fast_min_gap_ms"] is None        # null on the wire
        assert d["conflicting_hashes"] == ["0xdeadbeef"]

        blob = json.dumps(d, allow_nan=False)   # raises on Infinity/NaN
        assert "Infinity" not in blob and "NaN" not in blob
        assert json.loads(blob) == d            # and it parses back

    def test_real_gap_is_preserved_not_nulled(self):
        """Only the non-finite sentinel is replaced — real gaps survive."""
        c = _make_collector(
            _removed([0xABCD], t=1.0),
            _stored([0xABCD], lora_id=1, t=1.005),
        )
        d = c.summarize_probe(0.5, 2.0).to_dict()
        assert abs(d["reuse_too_fast_min_gap_ms"] - 5.0) < 0.1
        json.dumps(d, allow_nan=False)

    def test_finding_report_payload_round_trips(self):
        """The shapes that actually reach a finding JSON, end to end.

        ``snapshot_forensics`` splats ``to_dict()`` into the report, and
        ``analyze_kv_anomalies`` embeds its own copy of the gap in the anomaly
        metadata — both are written by ``runner.py``'s ``json.dumps``.
        """
        now = time.time()
        c = _make_collector(_stored([0x1], lora_id=1, lora_name="lora_a", t=now))
        forensics = c.snapshot_forensics(window_s=30.0)
        assert forensics["reuse_too_fast_min_gap_ms"] is None
        json.dumps(forensics, allow_nan=False, default=str)

        # A summary that *did* flag reuse still carries its gap into metadata.
        anomalies = analyze_kv_anomalies(_summary(reuse_count=1, min_gap=1.5))
        assert anomalies[0]["metadata"]["reuse_too_fast_min_gap_ms"] == 1.5
        json.dumps(anomalies, allow_nan=False, default=str)


# ---------------------------------------------------------------------------
# reuse_too_fast detection
# ---------------------------------------------------------------------------

class TestReuseTooFast:

    def test_reuse_too_fast_detected(self):
        """Remove at t=1.0, store same hash at t=1.005 (5ms gap) → flagged."""
        H = 0xABCD
        c = _make_collector(
            _removed([H], t=1.0),
            _stored([H], lora_id=1, t=1.005),
        )
        s = c.summarize_probe(0.5, 2.0)
        assert s.reuse_too_fast_count == 1
        assert abs(s.reuse_too_fast_min_gap_ms - 5.0) < 0.1

    def test_reuse_not_flagged_when_gap_large(self):
        """Remove at t=1.0, store at t=1.100 (100ms gap) → not flagged."""
        H = 0xABCD
        c = _make_collector(
            _removed([H], t=1.0),
            _stored([H], lora_id=1, t=1.100),
        )
        s = c.summarize_probe(0.5, 2.0)
        assert s.reuse_too_fast_count == 0
        assert s.reuse_too_fast_min_gap_ms == float('inf')

    def test_reuse_too_fast_min_gap_tracks_minimum(self):
        """Three pairs at 3ms, 7ms, 1ms → min_gap_ms ≈ 1.0."""
        H1, H2, H3 = 0x1, 0x2, 0x3
        c = _make_collector(
            _removed([H1], t=1.000),
            _stored([H1], lora_id=1, t=1.003),   # 3ms
            _removed([H2], t=2.000),
            _stored([H2], lora_id=1, t=2.007),   # 7ms
            _removed([H3], t=3.000),
            _stored([H3], lora_id=1, t=3.001),   # 1ms
        )
        s = c.summarize_probe(0.5, 4.0)
        assert s.reuse_too_fast_count == 3
        assert abs(s.reuse_too_fast_min_gap_ms - 1.0) < 0.1

    def test_store_without_prior_remove_not_flagged(self):
        """A store with no preceding remove in the window is never suspicious."""
        H = 0xABCD
        c = _make_collector(_stored([H], lora_id=1, t=1.0))
        s = c.summarize_probe(0.5, 2.0)
        assert s.reuse_too_fast_count == 0

    def test_clear_resets_removal_tracking(self):
        """After AllBlocksCleared, a following store is not considered a fast reuse."""
        H = 0xABCD
        c = _make_collector(
            _removed([H], t=1.0),
            _cleared(t=1.001),          # clear between remove and store
            _stored([H], lora_id=1, t=1.002),  # would be 2ms gap, but clear resets
        )
        s = c.summarize_probe(0.5, 2.0)
        assert s.reuse_too_fast_count == 0

    def test_reuse_too_fast_out_of_window_not_counted(self):
        """Remove outside the probe window does not contribute to the count."""
        H = 0xABCD
        c = _make_collector(
            _removed([H], t=0.1),          # outside window [1.0, 3.0]
            _stored([H], lora_id=1, t=1.005),  # inside window
        )
        s = c.summarize_probe(1.0, 3.0)
        assert s.reuse_too_fast_count == 0


# ---------------------------------------------------------------------------
# analyze_kv_anomalies
# ---------------------------------------------------------------------------

import math as _math
from fuzzer.kv_events import analyze_kv_anomalies


def _summary(cross_lora=0, reuse_count=0, min_gap=_math.inf):
    """Build a minimal KVProbeSummary for anomaly analysis tests."""
    return KVProbeSummary(
        start_t=0.0, end_t=1.0,
        stored_count=10, removed_count=5, cleared_count=0,
        distinct_blocks=8, per_lora_counts={},
        cross_lora_reused_blocks=cross_lora,
        conflicting_hashes=["0xabc"] * cross_lora,
        reuse_too_fast_count=reuse_count,
        reuse_too_fast_min_gap_ms=min_gap,
        fingerprint="abcdef1234567890",   # non-empty → collector was enabled
        raw_event_sample=[],
    )


class TestAnalyzeKvAnomalies:

    def test_clean_summary_returns_empty(self):
        assert analyze_kv_anomalies(_summary()) == []

    def test_disabled_collector_returns_empty(self):
        """Empty fingerprint means collector was not enabled."""
        s = _summary(cross_lora=2)
        s = KVProbeSummary(
            start_t=0.0, end_t=1.0, stored_count=0, removed_count=0,
            cleared_count=0, distinct_blocks=0, per_lora_counts={},
            cross_lora_reused_blocks=2, conflicting_hashes=[],
            fingerprint="",   # disabled
            raw_event_sample=[],
        )
        assert analyze_kv_anomalies(s) == []

    def test_reuse_too_fast_under_2ms_score(self):
        results = analyze_kv_anomalies(_summary(reuse_count=1, min_gap=1.5))
        assert len(results) == 1
        r = results[0]
        assert r["anomaly_type"] == "reuse_too_fast"
        assert r["score"] == 5.0

    def test_reuse_too_fast_2_to_10ms_score(self):
        results = analyze_kv_anomalies(_summary(reuse_count=1, min_gap=5.0))
        assert len(results) == 1
        r = results[0]
        assert r["anomaly_type"] == "reuse_too_fast"
        assert r["score"] == 3.0


# ---------------------------------------------------------------------------
# Scope: analyze_kv_anomalies is generic-only; cross_lora lives in adapter
# ---------------------------------------------------------------------------

class TestAnalyzeKvAnomaliesScope:

    def test_cross_lora_reuse_not_in_generic_analysis(self):
        """cross_lora_reused_blocks is NOT scored by the generic function."""
        s = _summary(cross_lora=2)
        result = analyze_kv_anomalies(s)
        assert "cross_lora_reuse" not in [r["anomaly_type"] for r in result]
        assert result == []
        # ...and it stays out even when a generic anomaly *is* scored.
        both = analyze_kv_anomalies(_summary(cross_lora=3, reuse_count=1,
                                             min_gap=1.0))
        assert [r["anomaly_type"] for r in both] == ["reuse_too_fast"]


# ---------------------------------------------------------------------------
# Fingerprint includes reuse_too_fast_count
# ---------------------------------------------------------------------------

class TestFingerprintWithReuseTooFast:

    def test_fingerprint_changes_with_reuse_too_fast(self):
        """Two summaries differing only in reuse_too_fast_count → different fps."""
        H = 0x1234
        c_no_reuse = _make_collector(_stored([H], lora_id=1, t=1.0))
        c_reuse = _make_collector(
            _removed([H], t=1.0),
            _stored([H], lora_id=1, t=1.005),
        )
        s_no = c_no_reuse.summarize_probe(0.5, 2.0)
        s_re = c_reuse.summarize_probe(0.5, 2.0)
        assert s_no.reuse_too_fast_count == 0
        assert s_re.reuse_too_fast_count == 1
        assert s_no.fingerprint != s_re.fingerprint


# ---------------------------------------------------------------------------
# stop() must always terminate — it runs inside Runner.run's finally
# ---------------------------------------------------------------------------

class _FakeSock:
    def __init__(self):
        self.closed_linger = None

    def close(self, linger=None):
        self.closed_linger = linger


class _FakeCtx:
    def __init__(self):
        self.term_calls = 0

    def term(self):
        self.term_calls += 1


class TestStopIsBounded:
    """The drain task await used to be unbounded and term() was called from
    inside the coroutine right after close() — pyzmq's async close needs loop
    callbacks that a blocking term() prevents from running."""

    @pytest.mark.asyncio
    async def test_stop_returns_when_task_ignores_cancel(self):
        import asyncio

        state = {"swallow": True}

        async def _stubborn():
            # A wedged drain loop: swallows its cancel and keeps going.
            while state["swallow"]:
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass

        c = _make_collector()
        c._STOP_TASK_TIMEOUT_S = 0.2
        task = asyncio.create_task(_stubborn())
        c._task = task
        await asyncio.sleep(0)

        t0 = time.monotonic()
        await c.stop()
        elapsed = time.monotonic() - t0

        # Bounded by _STOP_TASK_TIMEOUT_S, not by the task (which is still
        # running: stop() abandons it rather than waiting forever).
        assert elapsed < 2.0
        assert c._task is None
        assert not task.done()

        state["swallow"] = False           # let the leftover task retire
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    @pytest.mark.asyncio
    async def test_stop_closes_socket_and_terms_context(self):
        c = _make_collector()
        c._sock, c._zmq_ctx = _FakeSock(), _FakeCtx()
        sock, ctx = c._sock, c._zmq_ctx

        await c.stop()

        assert sock.closed_linger == 0
        assert ctx.term_calls == 1
        assert c._sock is None and c._zmq_ctx is None

    @pytest.mark.asyncio
    async def test_final_stop_skips_term(self):
        """At process shutdown term() buys nothing and can self-deadlock."""
        c = _make_collector()
        c._sock, c._zmq_ctx = _FakeSock(), _FakeCtx()
        sock, ctx = c._sock, c._zmq_ctx

        await c.stop(final=True)

        assert sock.closed_linger == 0
        assert ctx.term_calls == 0
        assert c._sock is None and c._zmq_ctx is None

    @pytest.mark.asyncio
    async def test_stop_survives_a_raising_socket_close(self):
        class _Boom(_FakeSock):
            def close(self, linger=None):
                raise RuntimeError("socket already closed")

        c = _make_collector()
        c._sock, c._zmq_ctx = _Boom(), _FakeCtx()
        await c.stop()                       # must not raise
        assert c._sock is None
