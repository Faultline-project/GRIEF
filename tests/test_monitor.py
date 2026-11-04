"""Tests for fuzzer/monitor.py — no GPU or server required."""

import re
import time
from pathlib import Path

import pytest

from fuzzer import monitor
from fuzzer.monitor import MetricSnapshot, MonitorSession, MockMonitor, SchedulerFingerprint


# ---------------------------------------------------------------------------
# SchedulerFingerprint
# ---------------------------------------------------------------------------

class TestSchedulerFingerprint:
    def test_no_activity_gives_zero_bucket(self):
        session = make_session(snap_full(running=0, waiting=0))
        fp = session.fingerprint()
        assert fp.queue_depth_bucket == 0

    def test_bucket_boundary_4_is_bucket_1(self):
        session = make_session(snap_full(waiting=4))
        fp = session.fingerprint()
        assert fp.queue_depth_bucket == 1

    def test_bucket_boundary_5_is_bucket_2(self):
        session = make_session(snap_full(waiting=5))
        fp = session.fingerprint()
        assert fp.queue_depth_bucket == 2

    def test_bucket_boundary_19_is_bucket_2(self):
        session = make_session(snap_full(waiting=19))
        fp = session.fingerprint()
        assert fp.queue_depth_bucket == 2

    def test_bucket_boundary_20_is_bucket_3(self):
        session = make_session(snap_full(waiting=20))
        fp = session.fingerprint()
        assert fp.queue_depth_bucket == 3

    def test_no_preemptions_had_preemption_false(self):
        session = make_session(snap_full(preemptions=0), snap_full(preemptions=0))
        fp = session.fingerprint()
        assert fp.had_preemption is False

    def test_preemption_delta_positive_had_preemption_true(self):
        session = make_session(snap_full(preemptions=5), snap_full(preemptions=8))
        fp = session.fingerprint()
        assert fp.had_preemption is True

    def test_no_preemption_metric_had_preemption_false(self):
        session = make_session(snap_full())  # preemptions=None
        fp = session.fingerprint()
        assert fp.had_preemption is False

    def test_fingerprint_is_hashable(self):
        session = make_session(snap_full(running=2, waiting=3))
        fp = session.fingerprint()
        s = {fp}  # must be hashable
        assert fp in s

    def test_identical_sessions_same_fingerprint(self):
        s1 = make_session(snap_full(running=2, waiting=3, preemptions=0, cpu_cache_pct=0.0))
        s2 = make_session(snap_full(running=2, waiting=3, preemptions=0, cpu_cache_pct=0.0))
        assert s1.fingerprint() == s2.fingerprint()

    def test_different_bucket_different_fingerprint(self):
        s1 = make_session(snap_full(waiting=1))    # bucket 1
        s2 = make_session(snap_full(waiting=20))   # bucket 3
        assert s1.fingerprint() != s2.fingerprint()

    def test_empty_session_fingerprint_all_defaults(self):
        session = MonitorSession()
        fp = session.fingerprint()
        assert fp == SchedulerFingerprint(
            had_preemption=False, had_swap=False, queue_depth_bucket=0,
            kv_peak_bucket=0, preemption_count_bucket=0,
            memory_delta_bucket=0, finished_count_bucket=0,
            prefix_cache_hit_bucket=0,
        )


# ---------------------------------------------------------------------------
# MonitorSession
# ---------------------------------------------------------------------------

def make_session(*snapshots: MetricSnapshot) -> MonitorSession:
    s = MonitorSession()
    s.snapshots = list(snapshots)
    return s


def snap(mem_used: float = 4000.0, mem_free: float = 20000.0,
         running: int = 0, waiting: int = 0,
         t: float = None) -> MetricSnapshot:
    return MetricSnapshot(
        timestamp=t if t is not None else time.time(),
        gpu_memory_used_mb=mem_used,
        gpu_memory_free_mb=mem_free,
        requests_running=running,
        requests_waiting=waiting,
    )


def snap_full(running: int = 0, waiting: int = 0,
              preemptions: int = None,
              cpu_cache_pct: float = None,
              t: float = None) -> MetricSnapshot:
    return MetricSnapshot(
        timestamp=t if t is not None else time.time(),
        requests_running=running,
        requests_waiting=waiting,
        preemptions_total=preemptions,
        cpu_cache_usage_pct=cpu_cache_pct,
    )


class TestMonitorSession:
    def test_empty_session_accessors_all_abstain(self):
        """One test for the eight `return max(vals) if vals else None` bodies.

        Replaces eight separate "empty session -> None" tests spread over five
        classes, and covers three accessors that had no empty-session test at
        all (peak_kv_cache_pct, finished_count_delta, preemption_delta)."""
        s = MonitorSession()
        assert s.baseline_memory_mb() is None
        assert s.peak_memory_mb() is None
        assert s.final_memory_mb() is None
        assert s.memory_leaked_mb() is None
        assert s.peak_swapped() is None
        assert s.peak_running() is None
        assert s.min_prefix_cache_hit_rate() is None
        assert s.peak_kv_cache_pct() is None
        assert s.finished_count_delta() is None
        assert s.preemption_delta() is None
        # ...and the two boolean accessors abstain to False, not None.
        assert s.had_swap() is False
        assert s.requests_ever_stuck() is False

    def test_baseline_memory_first_snapshot(self):
        session = make_session(snap(4000), snap(4100), snap(4050))
        assert session.baseline_memory_mb() == 4000.0

    def test_peak_memory_returns_max(self):
        session = make_session(snap(4000), snap(5000), snap(4500))
        assert session.peak_memory_mb() == 5000.0

    def test_final_memory_returns_last(self):
        session = make_session(snap(4000), snap(5000), snap(4200))
        assert session.final_memory_mb() == 4200.0

    def test_memory_leaked_mb_below_tolerance_returns_zero(self):
        # delta = 4050 - 4000 = 50 MB, tolerance = 100 MB → no leak
        session = make_session(snap(4000), snap(4050))
        assert session.memory_leaked_mb(tolerance_mb=100.0) == 0.0

    def test_memory_leaked_mb_above_tolerance_returns_delta(self):
        # delta = 4200 - 4000 = 200 MB, tolerance = 100 MB → 200 MB leak
        session = make_session(snap(4000), snap(4200))
        assert session.memory_leaked_mb(tolerance_mb=100.0) == 200.0

    def test_memory_leaked_mb_missing_gpu_values_returns_none(self):
        s = MetricSnapshot(timestamp=time.time())  # no gpu fields
        session = make_session(s)
        assert session.memory_leaked_mb() is None

    def test_requests_ever_stuck_false_when_waiting_stays_zero(self):
        session = make_session(
            snap(running=5, waiting=0),
            snap(running=3, waiting=0),
            snap(running=0, waiting=0),
        )
        assert not session.requests_ever_stuck()

    def test_requests_ever_stuck_false_when_never_idle(self):
        # waiting > 0 but running is also always > 0 (normal operation)
        session = make_session(
            snap(running=5, waiting=2),
            snap(running=3, waiting=1),
        )
        assert not session.requests_ever_stuck()

    def test_requests_ever_stuck_false_initial_snapshot_before_requests(self):
        # Regression: monitor fires a snapshot before any requests arrive
        # (running=0, waiting=0). Then requests queue normally.
        # This must NOT be treated as a scheduler hang.
        session = make_session(
            snap(running=0, waiting=0),   # initial snapshot — no requests yet
            snap(running=1, waiting=1),   # large prompt running, small waiting
            snap(running=1, waiting=0),   # small admitted
            snap(running=0, waiting=0),   # all done
        )
        assert not session.requests_ever_stuck()


# ---------------------------------------------------------------------------
# MockMonitor
# ---------------------------------------------------------------------------

class TestMockMonitor:
    async def test_start_resets_session(self):
        m = MockMonitor()
        await m.snapshot()
        await m.start()
        session = await m.stop()
        assert len(session.snapshots) == 0

    async def test_snapshot_increments_call_count(self):
        m = MockMonitor()
        await m.start()
        await m.snapshot()
        await m.snapshot()
        session = await m.stop()
        assert len(session.snapshots) == 2

    async def test_no_leak_mode_stable_memory(self):
        m = MockMonitor(inject_leak=False)
        await m.start()
        for _ in range(5):
            await m.snapshot()
        session = await m.stop()
        mems = [s.gpu_memory_used_mb for s in session.snapshots]
        assert all(v == 4000.0 for v in mems)

    async def test_inject_leak_mode_growing_memory(self):
        m = MockMonitor(inject_leak=True)
        await m.start()
        for _ in range(3):
            await m.snapshot()
        session = await m.stop()
        mems = [s.gpu_memory_used_mb for s in session.snapshots]
        assert mems == sorted(mems)   # strictly increasing
        assert mems[-1] > mems[0]

    async def test_inject_leak_triggers_leak_oracle(self):
        m = MockMonitor(inject_leak=True)
        await m.start()
        for _ in range(20):           # 20 × 10 MB = 200 MB above baseline
            await m.snapshot()
        session = await m.stop()
        leaked = session.memory_leaked_mb(tolerance_mb=100.0)
        assert leaked is not None
        assert leaked > 0

    async def test_no_leak_does_not_trigger_oracle(self):
        m = MockMonitor(inject_leak=False)
        await m.start()
        for _ in range(10):
            await m.snapshot()
        session = await m.stop()
        leaked = session.memory_leaked_mb(tolerance_mb=100.0)
        assert leaked == 0.0


# ---------------------------------------------------------------------------
# New SchedulerFingerprint fields (B1)
# ---------------------------------------------------------------------------

class TestSchedulerFingerprintNewFields:
    def test_kv_peak_bucket_edges(self):
        """Both sides of every kv_peak_bucket edge (monitor.py:414-418).

        Replaces five tests that sampled 10/35/60/90/97 — all *interior* points,
        so not one of the four thresholds (20/50/80/95) was pinned at all.
        """
        for pct, bucket in [
            (0.0, 0), (19.999, 0), (20.0, 1),      # edge 20
            (49.999, 1), (50.0, 2),                # edge 50
            (79.999, 2), (80.0, 3),                # edge 80
            (94.999, 3), (95.0, 4), (100.0, 4),    # edge 95
        ]:
            s = MetricSnapshot(timestamp=time.time(), kv_cache_usage_pct=pct)
            assert make_session(s).fingerprint().kv_peak_bucket == bucket, pct

    def test_kv_peak_bucket_none_when_no_data(self):
        session = MonitorSession()
        assert session.peak_kv_cache_pct() is None
        assert session.fingerprint().kv_peak_bucket == 0

    def test_preemption_count_bucket_edges(self):
        """Both sides of every preemption_count_bucket edge (monitor.py:422-425)."""
        for delta, bucket in [(0, 0), (1, 1), (2, 1), (3, 2),
                              (10, 2), (11, 3), (100, 3)]:
            session = make_session(snap_full(preemptions=0),
                                   snap_full(preemptions=delta))
            assert session.fingerprint().preemption_count_bucket == bucket, delta

    def test_finished_count_delta_helper(self):
        s1 = MetricSnapshot(timestamp=time.time(), requests_finished=10)
        s2 = MetricSnapshot(timestamp=time.time(), requests_finished=17)
        session = make_session(s1, s2)
        assert session.finished_count_delta() == 7

    def test_finished_count_delta_single_snapshot_returns_none(self):
        s = MetricSnapshot(timestamp=time.time(), requests_finished=5)
        session = make_session(s)
        assert session.finished_count_delta() is None

    def test_finished_count_bucket_edges(self):
        """Both sides of every finished_count_bucket edge (monitor.py:437-440).

        Bucket 2 (6-20) had no test at all before this."""
        for delta, bucket in [(0, 0), (1, 1), (5, 1), (6, 2),
                              (20, 2), (21, 3), (100, 3)]:
            s1 = MetricSnapshot(timestamp=time.time(), requests_finished=0)
            s2 = MetricSnapshot(timestamp=time.time(), requests_finished=delta)
            assert make_session(s1, s2).fingerprint().finished_count_bucket \
                   == bucket, delta

    def test_peak_kv_cache_pct_returns_max(self):
        s1 = MetricSnapshot(timestamp=time.time(), kv_cache_usage_pct=30.0)
        s2 = MetricSnapshot(timestamp=time.time(), kv_cache_usage_pct=75.0)
        session = make_session(s1, s2)
        assert session.peak_kv_cache_pct() == 75.0

    # -----------------------------------------------------------------
    # Drain state / KV baseline
    # -----------------------------------------------------------------

    @staticmethod
    def _kv(pct=None, running=None, waiting=None, mem=None):
        return MetricSnapshot(timestamp=time.time(), kv_cache_usage_pct=pct,
                              requests_running=running, requests_waiting=waiting,
                              gpu_memory_used_mb=mem)

    def test_engine_drained_true_false_and_unknown(self):
        assert make_session(self._kv(running=2, waiting=0),
                            self._kv(running=0, waiting=0)).engine_drained() is True
        assert make_session(self._kv(running=0, waiting=0),
                            self._kv(running=1, waiting=4)).engine_drained() is False
        # No queue telemetry anywhere: "never looked", not "idle".
        assert make_session(self._kv(pct=1.0),
                            self._kv(pct=2.0)).engine_drained() is None
        assert MonitorSession().engine_drained() is None

    def test_final_queue_state_skips_snapshots_without_queue_data(self):
        session = make_session(self._kv(running=3, waiting=5),
                               self._kv(pct=10.0))  # no queue fields
        assert session.final_queue_state() == (3, 5)

    def test_kv_final_drained_reads_the_snapshot_that_supplied_final(self):
        """Not merely the last snapshot: the one the number came from."""
        session = make_session(
            self._kv(pct=0.0, running=0, waiting=0),
            self._kv(pct=70.0, running=4, waiting=0),   # supplies final
            self._kv(running=0, waiting=0),             # idle but no KV
        )
        assert session.kv_cache_final_pct() == 70.0
        assert session.kv_cache_final_drained() is False

    def test_kv_final_drained_none_without_queue_data(self):
        session = make_session(self._kv(pct=0.0), self._kv(pct=40.0))
        assert session.kv_cache_final_drained() is None

    def test_kv_baseline_skips_leading_snapshot_without_kv(self):
        """The missing None-guard: its `final` counterpart always had one.

        A session whose first sample missed the KV scrape used to report
        `kv_cache_baseline_pct() is None`, which made the oracle announce
        "no kv cache data" even though every later sample carried it.
        """
        session = make_session(self._kv(running=0, waiting=0),   # no kv_pct
                               self._kv(pct=3.0, running=0, waiting=0),
                               self._kv(pct=9.0, running=0, waiting=0))
        assert session.kv_cache_baseline_pct() == 3.0

    def test_kv_baseline_anchors_to_an_idle_snapshot(self):
        session = make_session(
            self._kv(pct=60.0, running=5, waiting=2),   # busy — not a baseline
            self._kv(pct=4.0,  running=0, waiting=0),   # first idle reading
            self._kv(pct=40.0, running=0, waiting=0),
        )
        assert session.kv_cache_baseline_pct() == 4.0

    def test_kv_baseline_falls_back_when_never_idle(self):
        """Fallback can only understate the delta, never manufacture one."""
        session = make_session(self._kv(pct=60.0, running=5, waiting=2),
                               self._kv(pct=62.0, running=1, waiting=0))
        assert session.kv_cache_baseline_pct() == 60.0

    def test_kv_baseline_none_when_no_kv_anywhere(self):
        assert make_session(self._kv(running=0)).kv_cache_baseline_pct() is None
        assert MonitorSession().kv_cache_baseline_pct() is None

    def test_fingerprint_str_includes_all_fields(self):
        session = MonitorSession()
        s = str(session.fingerprint())
        assert "kv=" in s
        assert "pc=" in s
        assert "md=" in s
        assert "fc=" in s
        assert "hit=" in s


# ---------------------------------------------------------------------------
# requests_ever_stuck() — 3-second duration threshold (Change 1)
# ---------------------------------------------------------------------------

class TestRequestsEverStuckDuration:
    """Verifies the new duration-based scheduler hang detection.

    A streak of 2 consecutive 1-second snapshots (old behaviour) must NOT
    fire; only a continuous stuck condition lasting ≥ 3 seconds fires.
    """

    def test_2s_stuck_does_not_fire(self):
        # Snapshots at t=0,1,2,3: stuck starts at t=2, only 1 s has elapsed by t=3
        session = make_session(
            snap(running=3, waiting=0, t=0.0),
            snap(running=3, waiting=0, t=1.0),
            snap(running=0, waiting=2, t=2.0),   # stuck starts
            snap(running=0, waiting=2, t=3.0),   # 1 s elapsed — below threshold
        )
        assert not session.requests_ever_stuck()

    def test_exactly_3s_does_not_fire(self):
        # threshold is >=, so exactly 3.0 s fires
        session = make_session(
            snap(running=1, waiting=0, t=0.0),
            snap(running=0, waiting=1, t=1.0),   # stuck starts
            snap(running=0, waiting=1, t=4.0),   # exactly 3.0 s elapsed — fires
        )
        assert session.requests_ever_stuck()

    def test_4s_stuck_fires(self):
        session = make_session(
            snap(running=2, waiting=0, t=0.0),
            snap(running=0, waiting=2, t=1.0),   # stuck starts
            snap(running=0, waiting=2, t=5.0),   # 4 s elapsed — fires
        )
        assert session.requests_ever_stuck()

    def test_waiting_drops_to_zero_clears_timer(self):
        # Waiting queue empties briefly, then fills again — timer must restart.
        session = make_session(
            snap(running=2, waiting=0, t=0.0),
            snap(running=0, waiting=2, t=1.0),   # stuck starts
            snap(running=0, waiting=0, t=2.0),   # queue drained — timer cleared
            snap(running=0, waiting=2, t=3.0),   # stuck starts again
            snap(running=0, waiting=2, t=5.0),   # only 2 s — no fire
        )
        assert not session.requests_ever_stuck()

    def test_custom_min_stuck_seconds(self):
        # With min_stuck_seconds=1.0, a 2-second gap fires.
        session = make_session(
            snap(running=1, waiting=0, t=0.0),
            snap(running=0, waiting=1, t=1.0),   # stuck starts
            snap(running=0, waiting=1, t=2.0),   # 1 s elapsed → fires at 1.0
        )
        assert session.requests_ever_stuck(min_stuck_seconds=1.0)

    def test_never_had_running_does_not_fire(self):
        # saw_running never becomes True — stuck flag can never start.
        session = make_session(
            snap(running=0, waiting=3, t=0.0),
            snap(running=0, waiting=3, t=5.0),
        )
        assert not session.requests_ever_stuck()


# ---------------------------------------------------------------------------
# New MetricSnapshot fields + MonitorSession helpers
# ---------------------------------------------------------------------------

class TestPeakSwapped:
    def test_returns_max(self):
        s1 = MetricSnapshot(timestamp=time.time(), requests_swapped=2)
        s2 = MetricSnapshot(timestamp=time.time(), requests_swapped=7)
        s3 = MetricSnapshot(timestamp=time.time(), requests_swapped=3)
        session = make_session(s1, s2, s3)
        assert session.peak_swapped() == 7

    def test_all_none_returns_none(self):
        s1 = MetricSnapshot(timestamp=time.time())
        session = make_session(s1)
        assert session.peak_swapped() is None


class TestHadSwap:
    def test_all_zero_returns_false(self):
        s1 = MetricSnapshot(timestamp=time.time(), requests_swapped=0)
        s2 = MetricSnapshot(timestamp=time.time(), requests_swapped=0)
        assert not make_session(s1, s2).had_swap()

    def test_any_positive_returns_true(self):
        s1 = MetricSnapshot(timestamp=time.time(), requests_swapped=0)
        s2 = MetricSnapshot(timestamp=time.time(), requests_swapped=1)
        assert make_session(s1, s2).had_swap()

    def test_none_values_ignored(self):
        s1 = MetricSnapshot(timestamp=time.time())  # requests_swapped=None
        assert not make_session(s1).had_swap()


class TestMinPrefixCacheHitRate:
    def test_returns_min(self):
        s1 = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.9)
        s2 = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.3)
        s3 = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.7)
        assert make_session(s1, s2, s3).min_prefix_cache_hit_rate() == pytest.approx(0.3)

    def test_none_values_ignored(self):
        s1 = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.5)
        s2 = MetricSnapshot(timestamp=time.time())  # no hit rate
        assert make_session(s1, s2).min_prefix_cache_hit_rate() == pytest.approx(0.5)


class TestPrefixCacheHitBucket:
    def test_no_data_gives_bucket_0(self):
        session = MonitorSession()
        assert session.fingerprint().prefix_cache_hit_bucket == 0

    def test_exactly_80_gives_bucket_1(self):
        s = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.80)
        assert make_session(s).fingerprint().prefix_cache_hit_bucket == 1

    def test_exactly_50_gives_bucket_2(self):
        s = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.50)
        assert make_session(s).fingerprint().prefix_cache_hit_bucket == 2

    def test_low_hit_rate_gives_bucket_3(self):
        s = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.20)
        assert make_session(s).fingerprint().prefix_cache_hit_bucket == 3

    def test_min_of_multiple_snapshots_used(self):
        # High in first snapshot, low in second → min is low → bucket 3
        s1 = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.95)
        s2 = MetricSnapshot(timestamp=time.time(), gpu_prefix_cache_hit_rate=0.10)
        assert make_session(s1, s2).fingerprint().prefix_cache_hit_bucket == 3


class TestBackwardCompatFingerprint:
    def test_field_count_is_8_v1_dims_plus_version_plus_3_v2_dims(self):
        """The schema is append-only: 8 v1 dimensions, then version, then 3 v2 ones.

        The v1 dimensions keep their original *positions*, which is what lets a
        persisted 7- or 8-element tuple pad forward without any remapping.
        """
        import dataclasses
        from fuzzer.monitor import SchedulerFingerprint
        names = [f.name for f in dataclasses.fields(SchedulerFingerprint)]
        assert names[:8] == [
            "had_preemption", "had_swap", "queue_depth_bucket",
            "kv_peak_bucket", "preemption_count_bucket", "memory_delta_bucket",
            "finished_count_bucket", "prefix_cache_hit_bucket",
        ]
        assert names[8:] == [
            "version", "ttft_spread_bucket", "concurrency_peak_bucket",
            "error_class_bucket",
        ]
        assert len(names) == 12

    def test_had_cpu_swap_still_works_as_fallback(self):
        # MonitorSession.had_cpu_swap() is kept for backward compat
        s = MetricSnapshot(timestamp=time.time(), cpu_cache_usage_pct=0.5)
        session = make_session(s)
        assert session.had_cpu_swap() is True


# ---------------------------------------------------------------------------
# Fingerprint v2 — the three workload-derived dimensions (Workstream 1b)
#
# The v1 fingerprint has 6 constant dimensions on every campaign measured so far
# (an internal analysis note).  These three are derived from
# the TraceResult rather than from the 1 Hz telemetry poll, so they move under
# the existing burst/timing mutators.
# ---------------------------------------------------------------------------

from fuzzer.core.types import RequestResult, TraceResult
from fuzzer.monitor import (
    ERR_CLASS_4XX, ERR_CLASS_5XX, ERR_CLASS_CONN, ERR_CLASS_MIXED,
    ERR_CLASS_NONE, FP_VERSION_V1, FP_VERSION_V2, build_fingerprint,
    concurrency_peak_bucket, error_class_bucket, peak_concurrency,
    ttft_spread_bucket,
)


def req(rid="r", sent=0.0, first=None, completed=None, cancelled=None,
        error=None) -> RequestResult:
    return RequestResult(request_id=rid, sent_at=sent, first_token_at=first,
                         completed_at=completed, cancelled_at=cancelled,
                         error=error)


def reqs_with_ttfts(*ttft_ms) -> list:
    """Requests whose ttft_ms() are exactly the given values."""
    return [req(rid=f"r{i}", sent=0.0, first=t / 1000.0)
            for i, t in enumerate(ttft_ms)]


def tresult(requests, started=0.0, finished=10.0) -> TraceResult:
    return TraceResult(trace_seed=1, started_at=started, finished_at=finished,
                       requests=requests)


class TestTtftSpreadBucket:
    def test_ratio_just_under_2x_is_bucket_1(self):
        # 6 samples: p50 = index ceil(.5*6)-1 = 2 -> 100; p99 -> 199 => 1.99x
        assert ttft_spread_bucket(reqs_with_ttfts(100, 100, 100, 100, 100, 199)) == 1

    def test_ratio_2x_is_bucket_2(self):
        assert ttft_spread_bucket(reqs_with_ttfts(100, 100, 100, 100, 100, 200)) == 2

    def test_ratio_8x_is_bucket_4(self):
        assert ttft_spread_bucket(reqs_with_ttfts(100, 100, 100, 100, 100, 800)) == 4

    def test_ratio_is_capped_at_5(self):
        assert ttft_spread_bucket(reqs_with_ttfts(100, 100, 100, 100, 100, 100000)) == 5

    def test_requests_without_a_first_token_are_ignored(self):
        # 5 timed + 3 never-answered: still 5 usable samples, all equal.
        rs = reqs_with_ttfts(*([50] * 5)) + [req(rid=f"x{i}") for i in range(3)]
        assert ttft_spread_bucket(rs) == 1

    def test_min_sample_matches_the_oracles_constant(self):
        from fuzzer.core.types import MIN_LATENCY_SAMPLE
        # 5 identical samples is exactly enough; 4 is not.
        assert ttft_spread_bucket(reqs_with_ttfts(*([7] * MIN_LATENCY_SAMPLE))) == 1
        assert ttft_spread_bucket(reqs_with_ttfts(*([7] * (MIN_LATENCY_SAMPLE - 1)))) == 0


class TestPeakConcurrency:
    def test_sequential_requests_never_overlap(self):
        rs = [req("a", sent=0.0, completed=1.0),
              req("b", sent=1.0, completed=2.0),
              req("c", sent=2.0, completed=3.0)]
        assert peak_concurrency(tresult(rs)) == 1

    def test_fully_overlapping_requests(self):
        rs = [req(f"r{i}", sent=0.0, completed=5.0) for i in range(4)]
        assert peak_concurrency(tresult(rs)) == 4

    def test_partial_overlap_takes_the_max(self):
        rs = [req("a", sent=0.0, completed=3.0),
              req("b", sent=1.0, completed=2.0),
              req("c", sent=1.5, completed=4.0),
              req("d", sent=9.0, completed=9.5)]
        assert peak_concurrency(tresult(rs)) == 3

    def test_cancelled_request_closes_at_cancel(self):
        rs = [req("a", sent=0.0, cancelled=1.0),
              req("b", sent=1.0, completed=2.0)]
        assert peak_concurrency(tresult(rs)) == 1

    def test_never_completed_request_stays_open_to_trace_end(self):
        rs = [req("hung", sent=0.0), req("b", sent=5.0, completed=6.0)]
        assert peak_concurrency(tresult(rs, finished=10.0)) == 2

    def test_no_requests_is_zero(self):
        assert peak_concurrency(tresult([])) == 0

    def test_none_result_is_zero(self):
        assert peak_concurrency(None) == 0


class TestConcurrencyPeakBucket:
    def test_log2_staircase(self):
        cases = {1: 1, 2: 2, 3: 2, 4: 3, 7: 3, 8: 4, 15: 4, 16: 5, 64: 5}
        for n, bucket in cases.items():
            rs = [req(f"r{i}", sent=0.0, completed=5.0) for i in range(n)]
            assert concurrency_peak_bucket(MonitorSession(), tresult(rs)) == bucket, n

    def test_zero_requests_is_bucket_zero(self):
        assert concurrency_peak_bucket(MonitorSession(), tresult([])) == 0

    def test_polled_gauge_wins_when_it_is_higher(self):
        # The client saw 1 in flight, the scheduler reported 9 running: the
        # server was queueing work the client cannot see, so take the max.
        s = make_session(MetricSnapshot(timestamp=1.0, requests_running=9))
        rs = [req("a", sent=0.0, completed=1.0)]
        assert concurrency_peak_bucket(s, tresult(rs)) == 4

    def test_client_sweep_wins_when_the_poll_missed_the_peak(self):
        # The 1 Hz poll caught a single snapshot at 1 running; the trace itself
        # had 6 requests in flight simultaneously.  This is the failure mode
        # that made finished_count_delta useless.
        s = make_session(MetricSnapshot(timestamp=1.0, requests_running=1))
        rs = [req(f"r{i}", sent=0.0, completed=0.2) for i in range(6)]
        assert concurrency_peak_bucket(s, tresult(rs)) == 3

    def test_no_result_falls_back_to_the_gauge(self):
        s = make_session(MetricSnapshot(timestamp=1.0, requests_running=3))
        assert concurrency_peak_bucket(s, None) == 2


class TestErrorClassBucket:
    def test_no_errors(self):
        assert error_class_bucket([req("a"), req("b")]) == ERR_CLASS_NONE

    def test_4xx(self):
        e = "Client error '400 Bad Request' for url 'http://localhost:8000/v1/completions'"
        assert error_class_bucket([req("a", error=e)]) == ERR_CLASS_4XX

    def test_5xx(self):
        e = "Server error '503 Service Unavailable' for url 'http://localhost:8000/v1/completions'"
        assert error_class_bucket([req("a", error=e)]) == ERR_CLASS_5XX

    def test_connection_reset(self):
        assert error_class_bucket(
            [req("a", error="[Errno 104] Connection reset by peer")]) == ERR_CLASS_CONN

    def test_connection_refused(self):
        assert error_class_bucket(
            [req("a", error="All connection attempts failed")]) == ERR_CLASS_CONN

    def test_unrecognised_exception_counts_as_transport(self):
        # No HTTP status ever arrived, which is exactly what separates this
        # family from 4xx/5xx.
        assert error_class_bucket(
            [req("a", error="something odd happened")]) == ERR_CLASS_CONN

    def test_mixed_families(self):
        rs = [req("a", error="Client error '422 Unprocessable Entity' for url 'x'"),
              req("b", error="Server error '500 Internal Server Error' for url 'x'")]
        assert error_class_bucket(rs) == ERR_CLASS_MIXED

    def test_two_errors_of_the_same_family_are_not_mixed(self):
        rs = [req("a", error="Client error '400 Bad Request' for url 'x'"),
              req("b", error="Client error '404 Not Found' for url 'x'")]
        assert error_class_bucket(rs) == ERR_CLASS_4XX

    def test_a_port_number_is_not_mistaken_for_a_status(self):
        # 'http://localhost:5557' must not read as a 5xx.
        assert error_class_bucket(
            [req("a", error="[Errno 104] Connection reset by peer "
                            "(http://localhost:5557)")]) == ERR_CLASS_CONN


class TestBuildFingerprint:
    def _busy(self):
        s = make_session(
            MetricSnapshot(timestamp=1.0, requests_waiting=6, requests_running=2),
            MetricSnapshot(timestamp=2.0, requests_waiting=6, requests_running=2),
        )
        rs = reqs_with_ttfts(100, 100, 100, 100, 100, 800)
        for i, r in enumerate(rs):
            r.completed_at = 5.0
        return s, tresult(rs)

    def test_v1_is_exactly_session_fingerprint(self):
        s, r = self._busy()
        assert build_fingerprint(s, r, version=FP_VERSION_V1) == s.fingerprint()

    def test_v1_leaves_the_new_dimensions_at_zero(self):
        s, r = self._busy()
        fp = build_fingerprint(s, r, version=FP_VERSION_V1)
        assert fp.version == FP_VERSION_V1
        assert (fp.ttft_spread_bucket, fp.concurrency_peak_bucket,
                fp.error_class_bucket) == (0, 0, 0)

    def test_v2_keeps_every_v1_dimension(self):
        s, r = self._busy()
        v1 = s.fingerprint()
        v2 = build_fingerprint(s, r, version=FP_VERSION_V2)
        for f in ("had_preemption", "had_swap", "queue_depth_bucket",
                  "kv_peak_bucket", "preemption_count_bucket",
                  "memory_delta_bucket", "finished_count_bucket",
                  "prefix_cache_hit_bucket"):
            assert getattr(v1, f) == getattr(v2, f), f

    def test_v2_fills_the_new_dimensions(self):
        s, r = self._busy()
        fp = build_fingerprint(s, r, version=FP_VERSION_V2)
        assert fp.version == FP_VERSION_V2
        assert fp.ttft_spread_bucket == 4        # 800/100 = 8x
        assert fp.concurrency_peak_bucket == 3   # 6 in flight
        assert fp.error_class_bucket == 0

    def test_v2_without_a_result_is_all_unknown(self):
        s, _ = self._busy()
        fp = build_fingerprint(s, None, version=FP_VERSION_V2)
        assert (fp.ttft_spread_bucket, fp.error_class_bucket) == (0, 0)
        assert fp.version == FP_VERSION_V2

    def test_v2_moves_where_v1_is_constant(self):
        """The point of the exercise: two traces v1 cannot tell apart."""
        s = make_session(MetricSnapshot(timestamp=1.0, requests_waiting=0,
                                        requests_running=1))
        flat = reqs_with_ttfts(*([100] * 6))
        spread = reqs_with_ttfts(100, 100, 100, 100, 100, 3200)
        assert build_fingerprint(s, tresult(flat), 1) == \
               build_fingerprint(s, tresult(spread), 1)
        assert build_fingerprint(s, tresult(flat), 2) != \
               build_fingerprint(s, tresult(spread), 2)

    def test_v2_is_hashable(self):
        s, r = self._busy()
        fp = build_fingerprint(s, r, version=FP_VERSION_V2)
        assert fp in {fp}

    def test_v1_str_is_unchanged(self):
        """The offline novelty replay parses this format out of campaign logs."""
        s, r = self._busy()
        assert str(build_fingerprint(s, r, version=FP_VERSION_V1)) == \
               "pre=n swap=n q=2 kv=0 pc=0 md=0 fc=0 hit=0"

    def test_v2_str_appends_the_new_dimensions(self):
        s, r = self._busy()
        text = str(build_fingerprint(s, r, version=FP_VERSION_V2))
        assert text.startswith("pre=n swap=n q=2 kv=0 pc=0 md=0 fc=0 hit=0 ")
        assert " ts=4" in text and " cc=3" in text and " ec=0" in text
        assert text.endswith(" v=2")


class TestPeakRunning:
    def test_returns_max(self):
        s = make_session(MetricSnapshot(timestamp=1.0, requests_running=2),
                         MetricSnapshot(timestamp=2.0, requests_running=9),
                         MetricSnapshot(timestamp=3.0, requests_running=1))
        assert s.peak_running() == 9



# ---------------------------------------------------------------------------
# Prometheus metric names — the class of bug that made four fingerprint
# dimensions constant-zero on every run we have ever done.
#
# fuzzer/monitor.py used to scrape vllm:gpu_cache_usage_perc,
# vllm:num_requests_finished, vllm:gpu_prefix_cache_hit_rate,
# vllm:cpu_cache_usage_perc and vllm:num_requests_swapped.  None of those exist
# in vLLM v1.  Every scrape quietly produced None and kv_peak_bucket /
# finished_count_bucket / prefix_cache_hit_bucket / had_swap were pinned at
# 0/0/0/False -- with no error, no warning, and nothing in the test suite that
# could notice, because every test built MetricSnapshots by hand.
#
# These tests close that hole: they check our scrape targets against the real
# exported sample names of the two SIF images we run, checked in under
# tests/baselines/.  Regenerate those files from the images (the header of each
# has the command) whenever we move to a new vLLM.
# ---------------------------------------------------------------------------

class TestScrapedMetricNamesExist:
    BASELINE_DIR = Path(__file__).parent / "baselines"

    @classmethod
    def _names(cls, version: str) -> set[str]:
        path = cls.BASELINE_DIR / f"vllm_metric_names_{version}.txt"
        return {
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }

    def test_baselines_are_present_and_populated(self):
        for v in ("0.19.0", "0.27.1"):
            assert len(self._names(v)) > 100, f"{v} baseline looks truncated"

    @pytest.mark.parametrize("version", ["0.19.0", "0.27.1"])
    def test_every_scraped_name_exists_in_both_images(self, version):
        real = self._names(version)
        missing = [n for n in monitor.SCRAPED_METRIC_NAMES if n not in real]
        assert not missing, (
            f"fuzzer/monitor.py scrapes names vLLM {version} does not export: "
            f"{missing}"
        )

    def test_offload_name_exists_in_0_27(self):
        real = self._names("0.27.1")
        assert all(n in real for n in monitor.SCRAPED_METRIC_NAMES_0_27_ONLY)

    def test_the_dead_v0_names_are_absent_from_both_images(self):
        """Guards the guard: if these ever showed up, the test above is vacuous."""
        dead = {
            "vllm:gpu_cache_usage_perc",
            "vllm:num_requests_finished",
            "vllm:gpu_prefix_cache_hit_rate",
            "vllm:cpu_cache_usage_perc",
            "vllm:num_requests_swapped",
        }
        for v in ("0.19.0", "0.27.1"):
            assert not (dead & self._names(v))

    def test_no_vllm_metric_literal_anywhere_in_fuzzer_is_unknown(self):
        """Scan the package, not just the constants, so an inline name is caught."""
        known = self._names("0.19.0") | self._names("0.27.1")
        pkg = Path(monitor.__file__).parent
        # Bare metric *base* names are legitimate too (a scrape may match a
        # family and sum its samples), so accept a literal that is either an
        # exported sample name or the base of one.
        bases = {n.rsplit("_", 1)[0] for n in known} | known
        offenders = []
        for py in sorted(pkg.rglob("*.py")):
            for lineno, line in enumerate(py.read_text().splitlines(), 1):
                for m in re.finditer(r'["\'](vllm:[a-z][A-Za-z0-9_]*)["\']', line):
                    name = m.group(1)
                    if name not in known and name not in bases:
                        offenders.append(f"{py}:{lineno}: {name}")
        assert not offenders, (
            "these Prometheus names are exported by neither vLLM image:\n  "
            + "\n  ".join(offenders)
        )


class TestParsePrometheusText:
    BODY = "\n".join([
        "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.",
        "# TYPE vllm:kv_cache_usage_perc gauge",
        'vllm:kv_cache_usage_perc{engine="0",model_name="Qwen"} 0.998',
        'vllm:num_requests_running{engine="0"} 12.0',
        'vllm:num_requests_waiting{engine="0"} 7.0',
        'vllm:num_requests_waiting_by_reason{engine="0",reason="kv"} 999.0',
        'vllm:num_preemptions_total{engine="0"} 41.0',
        'vllm:num_preemptions_created{engine="0"} 1787771369.0',
        'vllm:request_success_total{engine="0",finished_reason="stop"} 30.0',
        'vllm:request_success_total{engine="0",finished_reason="length"} 12.0',
        'vllm:prefix_cache_hits_total{engine="0"} 800.0',
        'vllm:prefix_cache_queries_total{engine="0"} 1000.0',
        'vllm:external_prefix_cache_hits_total{engine="0"} 5.0',
        'vllm:time_to_first_token_seconds_sum{engine="0"} 3.5',
        'vllm:time_to_first_token_seconds_count{engine="0"} 20.0',
        'vllm:kv_offload_cpu_cache_usage_perc{engine="0"} 0.25',
    ])

    def test_comments_are_skipped(self):
        parsed = monitor.parse_prometheus_text(self.BODY)
        assert not any(k.startswith("#") for k in parsed)

    def test_names_are_matched_exactly_not_by_prefix(self):
        """0.27.1 exports num_requests_waiting_by_reason; a startswith() scrape
        would have read the breakdown series as the gauge."""
        parsed = monitor.parse_prometheus_text(self.BODY)
        assert parsed["vllm:num_requests_waiting"] == [7.0]
        assert parsed["vllm:num_requests_waiting_by_reason"] == [999.0]

    def test_counter_created_series_is_a_separate_name(self):
        parsed = monitor.parse_prometheus_text(self.BODY)
        assert parsed["vllm:num_preemptions_total"] == [41.0]
        assert "vllm:num_preemptions_created" in parsed

    def test_repeated_labels_collect_every_value(self):
        parsed = monitor.parse_prometheus_text(self.BODY)
        assert sorted(parsed["vllm:request_success_total"]) == [12.0, 30.0]

    def test_unlabelled_sample(self):
        parsed = monitor.parse_prometheus_text("vllm:num_requests_running 3.0")
        assert parsed["vllm:num_requests_running"] == [3.0]

    def test_malformed_lines_are_ignored(self):
        parsed = monitor.parse_prometheus_text("garbage\nvllm:x{a=\"b\"} notanumber\n")
        assert parsed.get("vllm:x") is None


class TestFillFromMetrics:
    def _snap(self):
        snap = MetricSnapshot(timestamp=0.0)
        monitor.NvmlMonitor._fill_from_metrics(
            snap, monitor.parse_prometheus_text(TestParsePrometheusText.BODY))
        return snap

    def test_kv_usage_is_scaled_to_percent(self):
        """vLLM's gauge is a fraction ('1 means 100 percent usage'); every
        consumer -- kv_peak_bucket's 20/50/80/95, runner.py's '5 percentage
        points' -- reads this field as 0-100."""
        assert self._snap().kv_cache_usage_pct == pytest.approx(99.8)

    def test_kv_peak_bucket_is_top_bucket_at_99_percent(self):
        """The whole point: 99.8% KV usage used to fingerprint as kv=0."""
        s = self._snap()
        assert make_session(s, s).fingerprint().kv_peak_bucket == 4

    def test_finished_count_sums_request_success_over_finish_reasons(self):
        assert self._snap().requests_finished == 42

    def test_preemptions_ignores_the_created_series(self):
        assert self._snap().preemptions_total == 41

    def test_waiting_is_the_gauge_not_the_by_reason_breakdown(self):
        assert self._snap().requests_waiting == 7

    def test_prefix_counters_and_cumulative_rate(self):
        s = self._snap()
        assert s.prefix_cache_hits_total == 800.0
        assert s.prefix_cache_queries_total == 1000.0
        assert s.gpu_prefix_cache_hit_rate == pytest.approx(0.8)

    def test_external_prefix_counter_does_not_leak_into_ours(self):
        assert self._snap().prefix_cache_hits_total == 800.0

    def test_offload_cpu_usage_is_scaled_to_percent(self):
        assert self._snap().cpu_cache_usage_pct == pytest.approx(25.0)

    def test_swap_stays_none_because_v1_has_no_swapping(self):
        assert self._snap().requests_swapped is None

    def test_ttft_histogram(self):
        s = self._snap()
        assert s.ttft_histogram_sum == pytest.approx(3.5)
        assert s.ttft_histogram_count == 20

    def test_empty_body_leaves_everything_none(self):
        snap = MetricSnapshot(timestamp=0.0)
        monitor.NvmlMonitor._fill_from_metrics(snap, {})
        assert snap.kv_cache_usage_pct is None
        assert snap.requests_finished is None


class TestIntervalPrefixCacheHitRate:
    """The rate is trace-local (delta/delta), not the server-lifetime ratio."""

    def test_uses_deltas_not_cumulative_ratio(self):
        # Lifetime ratio stays ~80%, but this trace hit 10 of 100 queries.
        s = make_session(
            MetricSnapshot(timestamp=0.0, prefix_cache_hits_total=800.0,
                           prefix_cache_queries_total=1000.0),
            MetricSnapshot(timestamp=1.0, prefix_cache_hits_total=810.0,
                           prefix_cache_queries_total=1100.0),
        )
        assert s.min_prefix_cache_hit_rate() == pytest.approx(0.1)
        assert s.fingerprint().prefix_cache_hit_bucket == 3   # low

    def test_intervals_without_queries_are_skipped_not_counted_as_zero(self):
        s = make_session(
            MetricSnapshot(timestamp=0.0, prefix_cache_hits_total=0.0,
                           prefix_cache_queries_total=0.0),
            MetricSnapshot(timestamp=1.0, prefix_cache_hits_total=0.0,
                           prefix_cache_queries_total=0.0),
            MetricSnapshot(timestamp=2.0, prefix_cache_hits_total=90.0,
                           prefix_cache_queries_total=100.0),
        )
        assert s.min_prefix_cache_hit_rate() == pytest.approx(0.9)

    def test_min_over_several_intervals(self):
        s = make_session(
            MetricSnapshot(timestamp=0.0, prefix_cache_hits_total=0.0,
                           prefix_cache_queries_total=0.0),
            MetricSnapshot(timestamp=1.0, prefix_cache_hits_total=90.0,
                           prefix_cache_queries_total=100.0),
            MetricSnapshot(timestamp=2.0, prefix_cache_hits_total=110.0,
                           prefix_cache_queries_total=200.0),
        )
        assert s.min_prefix_cache_hit_rate() == pytest.approx(0.2)

    def test_falls_back_to_the_rate_field_when_counters_absent(self):
        s = make_session(
            MetricSnapshot(timestamp=0.0, gpu_prefix_cache_hit_rate=0.42))
        assert s.min_prefix_cache_hit_rate() == pytest.approx(0.42)

    def test_single_snapshot_has_no_interval(self):
        s = make_session(
            MetricSnapshot(timestamp=0.0, prefix_cache_hits_total=1.0,
                           prefix_cache_queries_total=2.0))
        assert s.min_prefix_cache_hit_rate() is None


class TestStopTakesAFinalSnapshot:
    """A ~0.9 s determinism-probe trace used to collect one sample at a 1 s poll,
    so every session delta diffed that sample against itself."""

    def test_stop_brackets_the_trace(self):
        import asyncio

        m = monitor.NvmlMonitor(metrics_url="http://127.0.0.1:1/metrics")
        values = iter([{"vllm:num_preemptions_total": [3.0]},
                       {"vllm:num_preemptions_total": [9.0]}])

        async def fake_scrape():
            return next(values, {"vllm:num_preemptions_total": [9.0]})

        m._scrape = fake_scrape
        m._read_gpu_memory = lambda: (None, None)

        async def run():
            m._session = MonitorSession()
            await m.snapshot()          # start-of-trace sample
            return await m.stop()       # no poll loop ran; stop() must sample

        session = asyncio.run(run())
        assert len(session.snapshots) == 2
        assert session.preemption_delta() == 6

    def test_stop_survives_an_unreachable_server(self):
        import asyncio

        m = monitor.NvmlMonitor(metrics_url="http://127.0.0.1:1/metrics")
        m._read_gpu_memory = lambda: (None, None)

        async def boom():
            raise RuntimeError("connection refused")

        m._scrape = boom
        # _scrape's own except: clause is bypassed here, so this exercises
        # stop()'s guard.
        session = asyncio.run(m.stop())
        assert session.snapshots == []


class TestPollIntervalDefault:
    def test_default_lands_at_least_one_sample_inside_a_probe_trace(self):
        """A determinism-probe trace is ~0.9 s.  The trace's *endpoints* are
        guaranteed by NvmlMonitor.stop(); the poll rate only has to be short
        enough that something lands in between."""
        assert monitor.DEFAULT_POLL_INTERVAL_S < 0.9

    def test_default_is_not_so_short_that_it_perturbs_the_server(self):
        """A /metrics render is ~3.8 ms and runs synchronously on the API
        server's asyncio loop -- the loop that streams tokens, whose timings
        feed ttft_spread_bucket.  Below ~0.25 s we start measuring ourselves;
        see fuzzer/monitor.py:DEFAULT_POLL_INTERVAL_S for the numbers."""
        assert monitor.DEFAULT_POLL_INTERVAL_S >= 0.25

    def test_nvml_monitor_uses_the_shared_default(self):
        m = monitor.NvmlMonitor()
        assert m.poll_interval_s == monitor.DEFAULT_POLL_INTERVAL_S

    def test_cli_default_matches(self, monkeypatch):
        import sys
        from fuzzer.__main__ import parse_args
        monkeypatch.setattr(sys, "argv", ["python3 -m fuzzer"])
        assert parse_args().poll_interval == monitor.DEFAULT_POLL_INTERVAL_S


class TestFingerprintVersion3:
    def test_latest_is_v3(self):
        assert monitor.FP_VERSION_LATEST == monitor.FP_VERSION_V3 == 3

    def test_v3_str_carries_the_version(self):
        s = make_session(MetricSnapshot(timestamp=0.0, requests_waiting=1))
        assert str(monitor.build_fingerprint(
            s, None, version=monitor.FP_VERSION_V3)).endswith(" v=3")

    def test_requesting_v2_still_stamps_v2(self):
        s = make_session(MetricSnapshot(timestamp=0.0, requests_waiting=1))
        fp = monitor.build_fingerprint(s, None, version=monitor.FP_VERSION_V2)
        assert fp.version == monitor.FP_VERSION_V2

    def test_had_swap_is_constant_false_without_a_swap_gauge(self):
        """v1 has no swapping; the dimension is held, not dropped or repurposed
        onto the kv_offload gauge."""
        s = make_session(MetricSnapshot(timestamp=0.0, cpu_cache_usage_pct=42.0))
        assert s.fingerprint().had_swap is False
