"""Tests for fuzzer/oracles.py — no GPU or server required."""

import time
import pytest

from fuzzer.executor import TraceResult, RequestResult
from fuzzer.monitor import MonitorSession, MetricSnapshot
from fuzzer.oracles import Oracle, BugClass, OracleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session(*snapshots: MetricSnapshot) -> MonitorSession:
    s = MonitorSession()
    s.snapshots = list(snapshots)
    return s


def snap(mem_used: float = 4000.0, running: int = 0, waiting: int = 0,
         t: float = None) -> MetricSnapshot:
    return MetricSnapshot(
        timestamp=t if t is not None else time.time(),
        gpu_memory_used_mb=mem_used,
        requests_running=running,
        requests_waiting=waiting,
    )


def make_result(*requests: RequestResult, seed: int = 1) -> TraceResult:
    return TraceResult(
        trace_seed=seed,
        started_at=time.time() - 1.0,
        finished_at=time.time(),
        requests=list(requests),
    )


def completed_request(rid: str = "r1", ttft_s: float = 0.1) -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=rid,
        sent_at=now - 1.0,
        first_token_at=now - 1.0 + ttft_s,
        completed_at=now,
        tokens_received=10,
    )


def hung_request(rid: str = "r1") -> RequestResult:
    """A request with no completed_at or cancelled_at."""
    return RequestResult(
        request_id=rid,
        sent_at=time.time() - 60.0,
        tokens_received=0,
    )


def post_cancel_request(rid: str = "r1") -> RequestResult:
    """Tokens received AFTER the cancel timestamp."""
    t = time.time()
    return RequestResult(
        request_id=rid,
        sent_at=t - 2.0,
        cancelled_at=t - 1.5,
        first_token_at=t - 1.0,   # first token AFTER cancel
        last_token_at=t - 0.6,    # last token also AFTER cancel — what the oracle checks
        completed_at=t,
        tokens_received=5,
    )


# ---------------------------------------------------------------------------
# Memory leak oracle
# ---------------------------------------------------------------------------

class TestMemoryLeakOracle:
    oracle = Oracle(memory_leak_tolerance_mb=100.0)

    def test_no_gpu_data_passes(self):
        session = MonitorSession()  # empty
        r = self.oracle._check_memory_leak(session)
        assert r.passed

    def test_stable_memory_passes(self):
        session = make_session(snap(4000), snap(4000), snap(4000))
        r = self.oracle._check_memory_leak(session)
        assert r.passed

    def test_small_growth_within_tolerance_passes(self):
        # 50 MB growth, tolerance 100 MB
        session = make_session(snap(4000), snap(4050))
        r = self.oracle._check_memory_leak(session)
        assert r.passed

    def test_large_growth_fails(self):
        # 200 MB growth, tolerance 100 MB
        session = make_session(snap(4000), snap(4200))
        r = self.oracle._check_memory_leak(session)
        assert not r.passed
        assert r.bug_class == BugClass.MEMORY_LEAK

    def test_score_scales_with_leak_size(self):
        small = Oracle(memory_leak_tolerance_mb=0.0)._check_memory_leak(
            make_session(snap(4000), snap(4100))
        )
        large = Oracle(memory_leak_tolerance_mb=0.0)._check_memory_leak(
            make_session(snap(4000), snap(5000))
        )
        assert large.score > small.score


# ---------------------------------------------------------------------------
# Post-cancel tokens oracle
# ---------------------------------------------------------------------------

class TestPostCancelTokensOracle:
    oracle = Oracle()

    def test_clean_completion_passes(self):
        result = make_result(completed_request("r1"))
        r = self.oracle._check_post_cancel_tokens(result)
        assert r.passed

    def test_cancel_with_no_tokens_passes(self):
        req = RequestResult(
            request_id="r1", sent_at=time.time() - 1,
            cancelled_at=time.time() - 0.5,
            tokens_received=0,
        )
        result = make_result(req)
        r = self.oracle._check_post_cancel_tokens(result)
        assert r.passed

    def test_tokens_before_cancel_passes(self):
        t = time.time()
        req = RequestResult(
            request_id="r1", sent_at=t - 2.0,
            first_token_at=t - 1.8,    # token BEFORE cancel
            cancelled_at=t - 1.0,
            tokens_received=3,
        )
        result = make_result(req)
        r = self.oracle._check_post_cancel_tokens(result)
        assert r.passed

    def test_tokens_after_cancel_fails(self):
        result = make_result(post_cancel_request("r1"))
        r = self.oracle._check_post_cancel_tokens(result)
        assert not r.passed
        assert r.bug_class == BugClass.LIFECYCLE_ERROR

    def test_score_proportional_to_count(self):
        one   = make_result(post_cancel_request("r1"))
        two   = make_result(post_cancel_request("r1"), post_cancel_request("r2"))
        r_one = self.oracle._check_post_cancel_tokens(one)
        r_two = self.oracle._check_post_cancel_tokens(two)
        assert r_two.score > r_one.score


# ---------------------------------------------------------------------------
# Hung requests oracle
# ---------------------------------------------------------------------------

class TestHungRequestsOracle:
    oracle = Oracle(request_timeout_s=30.0)

    def test_completed_request_passes(self):
        result = make_result(completed_request())
        r = self.oracle._check_hung_requests(result)
        assert r.passed

    def test_hung_request_fails(self):
        result = make_result(hung_request())
        r = self.oracle._check_hung_requests(result)
        assert not r.passed
        assert r.bug_class == BugClass.LIFECYCLE_ERROR

    def test_cancelled_request_is_not_hung(self):
        req = RequestResult(
            request_id="r1", sent_at=time.time() - 1,
            cancelled_at=time.time() - 0.5,
        )
        result = make_result(req)
        r = self.oracle._check_hung_requests(result)
        assert r.passed


# ---------------------------------------------------------------------------
# Scheduler hang oracle
# ---------------------------------------------------------------------------

class TestSchedulerHangOracle:
    oracle = Oracle()

    def test_normal_drain_passes(self):
        session = make_session(
            snap(running=3, waiting=0),
            snap(running=0, waiting=0),
        )
        r = self.oracle._check_scheduler_hang(session)
        assert r.passed

    def test_waiting_after_idle_fails(self):
        # Need timestamps spanning ≥ 3 s of continuous stuck time to fire.
        session = make_session(
            snap(running=3, waiting=0, t=0.0),
            snap(running=0, waiting=0, t=1.0),
            snap(running=0, waiting=2, t=2.0),   # stuck starts
            snap(running=0, waiting=2, t=5.1),   # 3.1 s — fires
        )
        r = self.oracle._check_scheduler_hang(session)
        assert not r.passed
        assert r.bug_class == BugClass.SCHEDULER_HANG


# ---------------------------------------------------------------------------
# Latency spike oracle
# ---------------------------------------------------------------------------

def non_streaming_request(rid: str = "r1", total_s: float = 5.0) -> RequestResult:
    """Non-streaming request: first_token_at == completed_at (full completion)."""
    now = time.time()
    return RequestResult(
        request_id=rid,
        sent_at=now - total_s,
        first_token_at=now,   # set at completion, not at first token
        completed_at=now,
        tokens_received=1,
        stream=False,
    )


class TestLatencySpikeOracle:
    oracle = Oracle(latency_ttft_threshold_ms=1000.0)

    def test_fast_requests_pass(self):
        # Use MIN_LATENCY_SAMPLE requests so the guard doesn't skip the check
        requests = [completed_request(f"r{i}", ttft_s=0.1) for i in range(5)]
        result = make_result(*requests)
        r = self.oracle._check_latency_spike(result)
        assert r.passed

    def test_slow_p99_fails(self):
        # 10 requests all slow (6 seconds each) — clearly above 1s threshold
        requests = [completed_request(f"r{i}", ttft_s=6.0) for i in range(10)]
        result = make_result(*requests)
        r = self.oracle._check_latency_spike(result)
        assert not r.passed
        assert r.bug_class == BugClass.LATENCY_SPIKE

    def test_no_requests_passes(self):
        result = make_result()
        r = self.oracle._check_latency_spike(result)
        assert r.passed

    def test_too_few_samples_passes(self):
        """Fewer than MIN_LATENCY_SAMPLE streaming requests → skip latency check."""
        from fuzzer.oracles import MIN_LATENCY_SAMPLE
        # 4 slow requests — under sample guard
        requests = [completed_request(f"r{i}", ttft_s=6.0)
                    for i in range(MIN_LATENCY_SAMPLE - 1)]
        result = make_result(*requests)
        r = self.oracle._check_latency_spike(result)
        assert r.passed
        assert "too few samples" in r.detail

    def test_exactly_min_samples_uses_oracle(self):
        """Exactly MIN_LATENCY_SAMPLE samples → oracle fires normally."""
        from fuzzer.oracles import MIN_LATENCY_SAMPLE
        requests = [completed_request(f"r{i}", ttft_s=6.0)
                    for i in range(MIN_LATENCY_SAMPLE)]
        result = make_result(*requests)
        r = self.oracle._check_latency_spike(result)
        assert not r.passed

    def test_p99_index_correct_for_100_requests(self):
        """p99 of 100 requests is index 98 (not 99); a single outlier at index 99 must not fire."""
        # 99 fast + 1 slow: with correct formula the p99 is fast → passes
        requests = [completed_request(f"r{i}", ttft_s=0.05) for i in range(99)]
        requests.append(completed_request("slow", ttft_s=6.0))
        result = make_result(*requests)
        r = self.oracle._check_latency_spike(result)
        assert r.passed   # slow is p100, not p99

    def test_non_streaming_requests_excluded(self):
        """stream=False TTFT is total completion time — must not trigger oracle."""
        result = make_result(non_streaming_request("r1", total_s=10.0))
        r = self.oracle._check_latency_spike(result)
        assert r.passed

    def test_only_streaming_requests_checked(self):
        """Mix of streaming (fast) and non-streaming (slow): only streaming checked."""
        result = make_result(
            completed_request("r1", ttft_s=0.1),
            non_streaming_request("r2", total_s=20.0),
        )
        r = self.oracle._check_latency_spike(result)
        assert r.passed

    def test_relative_threshold_used_when_baseline_set(self):
        """When baseline_ttft_ms is set, threshold = baseline × multiplier."""
        oracle = Oracle(baseline_ttft_ms=50.0, latency_multiplier=3.0,
                        baseline_prompt_len=32)
        # All 5 requests have TTFT = 2000ms >> 50ms*3 = 150ms threshold
        requests = [completed_request(f"r{i}", ttft_s=2.0) for i in range(5)]
        result = make_result(*requests)
        r = oracle._check_latency_spike(result)
        assert not r.passed
        assert r.bug_class == BugClass.LATENCY_SPIKE

    def test_relative_threshold_passes_below_multiplier(self):
        oracle = Oracle(baseline_ttft_ms=500.0, latency_multiplier=3.0,
                        baseline_prompt_len=32)
        # TTFT = 1000ms < 500ms*3=1500ms threshold for 5 requests → passes
        requests = [completed_request(f"r{i}", ttft_s=1.0) for i in range(5)]
        result = make_result(*requests)
        r = oracle._check_latency_spike(result)
        assert r.passed

    def test_p99_with_baseline_correct_index(self):
        """p99 index fix also applies in baseline mode."""
        oracle = Oracle(baseline_ttft_ms=50.0, latency_multiplier=3.0,
                        baseline_prompt_len=32)
        # 99 fast (50ms) + 1 extremely slow (10s); with correct p99 index → passes
        requests = [completed_request(f"r{i}", ttft_s=0.05) for i in range(99)]
        requests.append(completed_request("slow", ttft_s=10.0))
        result = make_result(*requests)
        r = oracle._check_latency_spike(result)
        assert r.passed

    # --- Concurrency-aware threshold (burst queuing should not fire) ----------

    def _burst_request(self, rid: str, send_offset_s: float,
                       ttft_s: float) -> RequestResult:
        """Request sent at a fixed offset from a common start time."""
        t0 = 1_000_000.0   # fixed epoch so offsets are deterministic
        return RequestResult(
            request_id=rid,
            sent_at=t0 + send_offset_s,
            first_token_at=t0 + send_offset_s + ttft_s,
            completed_at=t0 + send_offset_s + ttft_s + 0.1,
            tokens_received=10,
            stream=True,
        )

    def test_burst_queuing_not_flagged(self):
        """30 requests sent simultaneously: last one is naturally ~30x slower.
        The concurrency-aware threshold must absorb this and not fire."""
        oracle = Oracle(baseline_ttft_ms=50.0, latency_multiplier=3.0,
                        baseline_prompt_len=32)
        n = 30
        # All sent at t=0; each takes 50ms baseline × its queue position
        requests = [
            self._burst_request(f"r{i}", send_offset_s=0.0,
                                ttft_s=(i + 1) * 0.05)
            for i in range(n)
        ]
        result = make_result(*requests)
        r = oracle._check_latency_spike(result)
        assert r.passed, (
            f"burst queuing falsely flagged as latency spike: {r.detail}"
        )

    def test_genuine_spike_above_concurrency_fires(self):
        """Burst of 10 requests (all concurrent, in_flight=10): 5 fast + 5 spiked.
        Threshold per request = 50 * 3 * 10 = 1500 ms.
        Spiked TTFT = 10 000 ms → 6.7x threshold.  p99 lands on a spiked
        request → must fire despite the high concurrency."""
        oracle = Oracle(baseline_ttft_ms=50.0, latency_multiplier=3.0,
                        baseline_prompt_len=32)
        # All sent simultaneously so in_flight=10 for every request.
        fast  = [self._burst_request(f"f{i}", send_offset_s=0.0, ttft_s=0.05)
                 for i in range(5)]
        slow  = [self._burst_request(f"s{i}", send_offset_s=0.0, ttft_s=10.0)
                 for i in range(5)]
        result = make_result(*(fast + slow))
        r = oracle._check_latency_spike(result)
        assert not r.passed
        assert r.bug_class == BugClass.LATENCY_SPIKE

    def test_concurrency_detail_included(self):
        """Finding detail must include concurrency so it's visible in the report."""
        oracle = Oracle(baseline_ttft_ms=50.0, latency_multiplier=3.0,
                        baseline_prompt_len=32)
        requests = [
            self._burst_request("r0", send_offset_s=0.0, ttft_s=0.05),
            self._burst_request("slow", send_offset_s=0.001, ttft_s=10.0),
        ]
        result = make_result(*requests)
        r = oracle._check_latency_spike(result)
        if not r.passed:
            assert "concurrency=" in r.detail


# ---------------------------------------------------------------------------
# KV block leak oracle
# ---------------------------------------------------------------------------

def snap_kv(mem_used: float = 4000.0, kv_pct: float = 0.0,
            running: int = 0, waiting: int = 0) -> MetricSnapshot:
    return MetricSnapshot(
        timestamp=time.time(),
        gpu_memory_used_mb=mem_used,
        kv_cache_usage_pct=kv_pct,
        requests_running=running,
        requests_waiting=waiting,
    )


class TestKvBlockLeakOracle:
    oracle = Oracle(kv_block_leak_pct=5.0)

    def test_stable_kv_usage_passes(self):
        session = make_session(snap_kv(kv_pct=10.0), snap_kv(kv_pct=10.0))
        r = self.oracle._check_kv_block_leak(session)
        assert r.passed

    def test_kv_usage_grows_beyond_threshold_fails(self):
        session = make_session(snap_kv(kv_pct=10.0), snap_kv(kv_pct=20.0))
        r = self.oracle._check_kv_block_leak(session)
        assert not r.passed
        assert r.bug_class == BugClass.KV_BLOCK_LEAK

    def test_small_kv_growth_within_threshold_passes(self):
        session = make_session(snap_kv(kv_pct=10.0), snap_kv(kv_pct=13.0))
        r = self.oracle._check_kv_block_leak(session)
        assert r.passed

    def test_no_kv_data_passes(self):
        session = make_session(snap(4000), snap(4000))  # no kv_pct set
        r = self.oracle._check_kv_block_leak(session)
        assert r.passed


# ---------------------------------------------------------------------------
# KV block leak — drain gate
#
# Regression suite for the false-positive class: every kv_block_leak finding
# GRIEF ever produced came from reading the final KV percentage of an engine
# that was still generating, because the client had given up at
# trace_timeout_s.  These tests pin the three outcomes apart: FIRES on a
# drained leak, DOES NOT FIRE when not drained, and the second is visibly
# "could not measure" rather than a pass.
# ---------------------------------------------------------------------------

def snap_no_queue(kv_pct: float) -> MetricSnapshot:
    """A snapshot with KV data but no running/waiting telemetry at all."""
    return MetricSnapshot(timestamp=time.time(), kv_cache_usage_pct=kv_pct)


class TestKvBlockLeakDrainGate:
    oracle = Oracle(kv_block_leak_pct=5.0)

    # --- must NOT fire ---------------------------------------------------

    def test_busy_engine_does_not_file(self):
        """The exact historical false positive: 3 running, 7 waiting, high KV."""
        session = make_session(
            snap_kv(kv_pct=0.0),
            snap_kv(kv_pct=89.5, running=3, waiting=7),
        )
        r = self.oracle._check_kv_block_leak(session)
        assert r.passed
        assert r.bug_class is None

    def test_queued_but_not_running_does_not_file(self):
        """waiting>0 with running==0 is still not drained."""
        session = make_session(
            snap_kv(kv_pct=0.0),
            snap_kv(kv_pct=55.3, running=0, waiting=9),
        )
        assert self.oracle._check_kv_block_leak(session).passed

    def test_no_queue_telemetry_does_not_file(self):
        """Absent running/waiting is 'never looked', not 'idle'."""
        session = make_session(snap_no_queue(0.0), snap_no_queue(60.0))
        r = self.oracle._check_kv_block_leak(session)
        assert r.passed

    def test_drain_is_read_from_the_snapshot_that_supplied_final(self):
        """A later idle snapshot without KV data cannot launder a busy reading.

        kv_cache_final_pct() comes from the last KV-bearing snapshot; the gate
        has to interrogate *that* snapshot, not merely the last one of any kind.
        """
        session = make_session(
            snap_kv(kv_pct=0.0),
            snap_kv(kv_pct=70.0, running=4, waiting=0),   # supplies `final`
            snap(4000, running=0, waiting=0),             # idle, but no KV
        )
        assert self.oracle._check_kv_block_leak(session).passed

    # --- must STILL fire on a genuine leak -------------------------------

    def test_drained_session_with_retained_kv_still_files(self):
        """The class is gated, not deleted."""
        session = make_session(
            snap_kv(kv_pct=0.0, running=0, waiting=0),
            snap_kv(kv_pct=45.0, running=2, waiting=1),   # mid-trace, busy
            snap_kv(kv_pct=30.0, running=0, waiting=0),   # drained, 30% held
        )
        r = self.oracle._check_kv_block_leak(session)
        assert not r.passed
        assert r.bug_class == BugClass.KV_BLOCK_LEAK
        assert r.score == pytest.approx(3.0)
        assert "running=0 waiting=0" in r.detail

    def test_single_retained_block_scale_still_below_threshold(self):
        """Sanity on the units: 0.1% (one block of 1152) is not a finding."""
        session = make_session(snap_kv(kv_pct=0.0), snap_kv(kv_pct=0.1))
        assert self.oracle._check_kv_block_leak(session).passed

    # --- "could not measure" must be distinguishable from a pass ---------

    def test_not_drained_is_unmeasured_not_a_clean_pass(self):
        busy = make_session(snap_kv(kv_pct=0.0),
                            snap_kv(kv_pct=80.0, running=3, waiting=7))
        clean = make_session(snap_kv(kv_pct=0.0), snap_kv(kv_pct=0.0))

        r_busy  = self.oracle._check_kv_block_leak(busy)
        r_clean = self.oracle._check_kv_block_leak(clean)

        # Both "pass" in the sense of filing nothing …
        assert r_busy.passed and r_clean.passed
        # … but only one of them was actually measured.
        assert r_busy.unmeasured is not None
        assert r_clean.unmeasured is None
        assert "not drained" in r_busy.unmeasured
        assert "running=3" in r_busy.unmeasured and "waiting=7" in r_busy.unmeasured
        assert "not measurable" in r_busy.detail

    def test_missing_telemetry_reasons_are_distinct(self):
        no_kv    = self.oracle._check_kv_block_leak(
            make_session(snap(4000), snap(4000)))
        no_queue = self.oracle._check_kv_block_leak(
            make_session(snap_no_queue(0.0), snap_no_queue(60.0)))
        busy     = self.oracle._check_kv_block_leak(
            make_session(snap_kv(kv_pct=0.0),
                         snap_kv(kv_pct=80.0, running=1, waiting=0)))
        reasons = {no_kv.unmeasured, no_queue.unmeasured, busy.unmeasured}
        assert len(reasons) == 3
        assert all(r and r.startswith("kv_block_leak: ") for r in reasons)

    # --- evaluate() plumbing ---------------------------------------------

    def test_evaluate_reports_gap_without_filing_a_finding(self):
        result  = make_result(completed_request("r1"))
        session = make_session(snap_kv(kv_pct=0.0),
                               snap_kv(kv_pct=80.0, running=3, waiting=7))
        findings = self.oracle.evaluate(result, session)
        assert not any(f.bug_class == BugClass.KV_BLOCK_LEAK for f in findings)
        gaps = [g.unmeasured for g in self.oracle.last_measurement_gaps]
        assert any(g.startswith("kv_block_leak: engine not drained") for g in gaps)

    def test_evaluate_clean_drained_session_reports_no_gap(self):
        result  = make_result(completed_request("r1"))
        session = make_session(snap_kv(kv_pct=0.0), snap_kv(kv_pct=0.0))
        self.oracle.evaluate(result, session)
        assert [g.unmeasured for g in self.oracle.last_measurement_gaps] == []

    def test_gaps_are_reset_between_evaluations(self):
        result = make_result(completed_request("r1"))
        self.oracle.evaluate(result, make_session(
            snap_kv(kv_pct=0.0), snap_kv(kv_pct=80.0, running=3, waiting=7)))
        assert self.oracle.last_measurement_gaps
        self.oracle.evaluate(result, make_session(
            snap_kv(kv_pct=0.0), snap_kv(kv_pct=0.0)))
        assert self.oracle.last_measurement_gaps == []

    def test_gap_is_not_returned_as_a_finding(self):
        """Gaps must stay out of the list that feeds corpus scoring."""
        result  = make_result(completed_request("r1"))
        session = make_session(snap_kv(kv_pct=0.0),
                               snap_kv(kv_pct=80.0, running=3, waiting=7))
        findings = self.oracle.evaluate(result, session)
        assert all(f.unmeasured is None for f in findings)


# ---------------------------------------------------------------------------
# Memory leak — same drain gate
# ---------------------------------------------------------------------------

class TestMemoryLeakDrainGate:
    oracle = Oracle(memory_leak_tolerance_mb=100.0)

    def test_busy_engine_does_not_file(self):
        session = make_session(snap(4000), snap(4500, running=2, waiting=3))
        r = self.oracle._check_memory_leak(session)
        assert r.passed
        assert r.unmeasured is not None
        assert "not drained" in r.unmeasured
        assert "not measurable" in r.detail

    def test_drained_growth_still_files(self):
        session = make_session(snap(4000), snap(4500, running=0, waiting=0))
        r = self.oracle._check_memory_leak(session)
        assert not r.passed
        assert r.bug_class == BugClass.MEMORY_LEAK
        assert "running=0 waiting=0" in r.detail

    def test_no_queue_telemetry_is_unmeasured(self):
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=time.time(), gpu_memory_used_mb=4000.0),
            MetricSnapshot(timestamp=time.time(), gpu_memory_used_mb=4500.0),
        ]
        r = self.oracle._check_memory_leak(session)
        assert r.passed
        assert r.unmeasured == "memory_leak: no queue telemetry"

    def test_drained_stable_memory_is_a_real_pass(self):
        session = make_session(snap(4000), snap(4000))
        r = self.oracle._check_memory_leak(session)
        assert r.passed
        assert r.unmeasured is None


# ---------------------------------------------------------------------------
# Determinism oracle (static)
# ---------------------------------------------------------------------------

class TestDeterminismOracle:
    def test_identical_outputs_pass(self):
        r = Oracle.check_determinism("The answer is 42.", "The answer is 42.")
        assert r.passed

    def test_different_outputs_fail(self):
        r = Oracle.check_determinism(
            "pre127 pre128 pre129 pre130 pre131",
            "000000 111111 222222 333333 444444",
        )
        assert not r.passed
        assert r.bug_class == BugClass.KV_CORRUPTION
        assert r.probe_metadata["similarity"] < 0.3

    def test_high_similarity_still_fails(self):
        # Any divergence returns passed=False; FP filtering is done by caller
        r = Oracle.check_determinism("The answer is 42.", "The answer is 43.")
        assert not r.passed
        assert r.probe_metadata["similarity"] > 0.5

    def test_mid_similarity_fails(self):
        a = "alpha beta gamma delta epsilon zeta eta theta"
        b = "alpha beta gamma delta omega psi chi phi"
        r = Oracle.check_determinism(a, b)
        assert not r.passed
        assert 0.0 < r.probe_metadata["similarity"] < 1.0

    def test_empty_strings_identical_pass(self):
        r = Oracle.check_determinism("", "")
        assert r.passed

    def test_score_is_high_severity(self):
        # Use fully disjoint strings to guarantee a hard finding
        r = Oracle.check_determinism("aaa bbb ccc ddd eee", "xxx yyy zzz qqq rrr")
        assert r.score >= 10.0


# ---------------------------------------------------------------------------
# Stability oracle (static)
# ---------------------------------------------------------------------------

class TestStabilityOracle:
    # --- Passing cases (no real instability) --------------------------------

    def test_identical_outputs_pass(self):
        outputs = ["The answer is 42."] * 3
        r = Oracle.check_stability(outputs, outputs)
        assert r.passed

    def test_low_target_divergence_passes(self):
        # 1/3 diverge → rate=0.33, below min_target_rate=0.6
        target   = ["The answer is 42.", "The answer is 42.", "The answer is 43."]
        baseline = ["The answer is 42."] * 3
        r = Oracle.check_stability(target, baseline)
        assert r.passed

    def test_high_rate_but_low_delta_passes(self):
        # Both target and baseline are noisy — delta is ~0
        noisy = ["a", "b", "c"]  # diverge_rate = 0.67 each
        r = Oracle.check_stability(noisy, noisy)
        assert r.passed  # delta = 0, fails Rule B

    def test_empty_target_passes(self):
        r = Oracle.check_stability([], ["The answer is 42."])
        assert r.passed

    def test_empty_baseline_passes(self):
        r = Oracle.check_stability(["The answer is 42."], [])
        assert r.passed

    def test_single_element_each_passes(self):
        # Single-element lists → diverge_rate = 0.0 for both
        r = Oracle.check_stability(["a"], ["b"])
        assert r.passed

    def test_noisy_baseline_suppresses_finding(self):
        # target_rate=0.8, baseline_rate=0.67 → delta=0.13 < 0.4, no fire
        target   = ["a", "b", "c", "d", "a"]
        baseline = ["a", "b", "c"]
        r = Oracle.check_stability(target, baseline)
        assert r.passed

    # --- Failing cases (validated instability) ------------------------------

    def test_high_divergence_stable_baseline_fires(self):
        # target: 3/5 differ from modal "foo" → rate=0.6
        # baseline: 0/3 differ → rate=0.0; delta=0.6 >= 0.4
        target   = ["foo", "bar", "baz", "qux", "foo"]
        baseline = ["foo"] * 3
        r = Oracle.check_stability(target, baseline)
        assert not r.passed
        assert r.bug_class == BugClass.KV_CORRUPTION

    def test_score_capped_at_10(self):
        target   = ["a", "b", "c", "d", "a"]  # rate=0.6, delta=0.6 → score=6.0
        baseline = ["a"] * 3
        r = Oracle.check_stability(target, baseline)
        assert not r.passed
        assert r.score <= 10.0

    def test_score_proportional_to_delta(self):
        # All 3 target outputs different → modal ambiguous, rate=0.67
        # baseline stable → delta=0.67; score=min(10, 6.7)
        target   = ["x", "y", "z"]
        baseline = ["x"] * 3
        r = Oracle.check_stability(target, baseline)
        assert not r.passed
        assert r.score == pytest.approx(min(10.0, 0.666 * 10.0), abs=0.5)

    def test_custom_thresholds_respected(self):
        # With lenient thresholds, fires at lower divergence
        target   = ["a", "b", "a"]   # 1/3 differ → rate=0.33
        baseline = ["a"] * 3
        r = Oracle.check_stability(
            target, baseline,
            min_target_rate=0.3,
            min_rate_delta=0.3,
        )
        assert not r.passed

    def test_detail_contains_rates(self):
        target   = ["foo", "bar", "baz", "qux", "foo"]
        baseline = ["foo"] * 3
        r = Oracle.check_stability(target, baseline)
        assert not r.passed
        assert "target_rate=" in r.detail
        assert "baseline_rate=" in r.detail
        assert "delta=" in r.detail

    def test_check_determinism_unchanged(self):
        # Regression guard: check_determinism still uses exact equality
        assert Oracle.check_determinism("x", "x").passed
        assert not Oracle.check_determinism("x", "y").passed


# ---------------------------------------------------------------------------
# Cache-hit proxy oracle (static)
# ---------------------------------------------------------------------------

class TestCacheHitProxyOracle:
    def test_similar_ttft_passes(self):
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=100.0, probe_ttft_ms=150.0)
        assert r.passed

    def test_ttft_regression_fails(self):
        # probe 5× slower than warm → likely cold cache miss
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=100.0, probe_ttft_ms=600.0)
        assert not r.passed
        assert r.bug_class == BugClass.CACHE_EVICTION

    def test_custom_multiplier(self):
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=200.0,
                                          probe_ttft_ms=350.0,
                                          multiplier=2.0)
        assert r.passed  # 350 < 200×2 = 400

    def test_score_proportional_to_regression(self):
        small = Oracle.check_cache_hit_proxy(100.0, 250.0)
        large = Oracle.check_cache_hit_proxy(100.0, 1000.0)
        assert large.score > small.score

    # --- 500 ms absolute floor (Change 4) ------------------------------------

    def test_above_multiplier_but_below_500ms_floor_passes(self):
        # warm=15ms, 4×=60ms threshold, probe=80ms → above 4× but below 500ms → pass
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=15.0, probe_ttft_ms=80.0)
        assert r.passed

    def test_above_500ms_floor_fails(self):
        # warm=15ms, probe=600ms → above both 4× (60ms) and 500ms floor → fail
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=15.0, probe_ttft_ms=600.0)
        assert not r.passed
        assert r.bug_class == BugClass.CACHE_EVICTION

    def test_large_warm_baseline_multiplier_dominates_floor(self):
        # warm=200ms, 4×=800ms > 500ms floor; probe=850ms → above multiplier → fail
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=200.0, probe_ttft_ms=850.0)
        assert not r.passed

    def test_large_warm_below_multiplier_passes(self):
        # warm=200ms, 4×=800ms, probe=750ms → below multiplier threshold → pass
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=200.0, probe_ttft_ms=750.0)
        assert r.passed

    def test_probe_exactly_500ms_with_tiny_warm_passes(self):
        # probe=500ms is NOT strictly > 500ms floor → pass
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=10.0, probe_ttft_ms=500.0)
        assert r.passed

    def test_probe_just_above_500ms_floor_fails(self):
        r = Oracle.check_cache_hit_proxy(warm_ttft_ms=10.0, probe_ttft_ms=500.1)
        assert not r.passed


# ---------------------------------------------------------------------------
# Oracle.evaluate() integration
# ---------------------------------------------------------------------------

class TestOracleEvaluate:
    oracle = Oracle(memory_leak_tolerance_mb=100.0,
                    latency_ttft_threshold_ms=5000.0,
                    request_timeout_s=30.0)

    def test_clean_run_returns_single_passing_result(self):
        result  = make_result(completed_request("r1"))
        session = make_session(snap(4000), snap(4000))
        findings = self.oracle.evaluate(result, session)
        assert len(findings) == 1
        assert findings[0].passed

    def test_memory_leak_detected(self):
        result  = make_result(completed_request("r1"))
        session = make_session(snap(4000), snap(4500))  # 500 MB growth
        findings = self.oracle.evaluate(result, session)
        assert any(not f.passed and f.bug_class == BugClass.MEMORY_LEAK
                   for f in findings)

    def test_hung_request_detected(self):
        result  = make_result(hung_request("r1"))
        session = make_session(snap(4000), snap(4000))
        findings = self.oracle.evaluate(result, session)
        assert any(not f.passed and f.bug_class == BugClass.LIFECYCLE_ERROR
                   for f in findings)

    def test_multiple_bugs_all_reported(self):
        result  = make_result(hung_request("r1"), post_cancel_request("r2"))
        session = make_session(snap(4000), snap(4500))  # also a memory leak
        findings = self.oracle.evaluate(result, session)
        bug_classes = {f.bug_class for f in findings if not f.passed}
        assert BugClass.MEMORY_LEAK     in bug_classes
        assert BugClass.LIFECYCLE_ERROR in bug_classes


# ---------------------------------------------------------------------------
# Token overflow oracle
# ---------------------------------------------------------------------------

class TestTokenOverflowOracle:
    oracle = Oracle()

    def _req_with_max(self, rid: str, tokens_received: int,
                      max_tokens_requested: int) -> RequestResult:
        return RequestResult(
            request_id=rid,
            sent_at=time.time() - 1.0,
            completed_at=time.time(),
            tokens_received=tokens_received,
            max_tokens_requested=max_tokens_requested,
        )

    def test_within_budget_passes(self):
        result = make_result(self._req_with_max("r1", 10, 64))
        r = self.oracle._check_token_overflow(result)
        assert r.passed

    def test_exactly_at_budget_passes(self):
        result = make_result(self._req_with_max("r1", 64, 64))
        r = self.oracle._check_token_overflow(result)
        assert r.passed

    def test_exceeds_budget_fails(self):
        result = make_result(self._req_with_max("r1", 65, 64))
        r = self.oracle._check_token_overflow(result)
        assert not r.passed
        assert r.bug_class == BugClass.LIFECYCLE_ERROR

    def test_no_max_tokens_requested_passes(self):
        req = RequestResult(request_id="r1", sent_at=time.time(),
                            tokens_received=100)
        result = make_result(req)
        r = self.oracle._check_token_overflow(result)
        assert r.passed

    def test_score_scales_with_count(self):
        one = make_result(self._req_with_max("r1", 65, 64))
        two = make_result(self._req_with_max("r1", 65, 64),
                          self._req_with_max("r2", 65, 64))
        r1 = self.oracle._check_token_overflow(one)
        r2 = self.oracle._check_token_overflow(two)
        assert r2.score > r1.score


# ---------------------------------------------------------------------------
# Error rate oracle — fatal-pattern filter (Change 2)
# ---------------------------------------------------------------------------

def req_error(rid: str, error: str) -> RequestResult:
    return RequestResult(request_id=rid, sent_at=time.time(),
                         completed_at=time.time(), error=error)


#: Errors that must NOT fire the liveness oracle.  Each of the three marked
#: ones contains a 5xx *digit run* somewhere in its text, which is how the old
#: bare-substring match against ``_FATAL_PATTERNS`` mis-filed them.
BENIGN_ERRORS = [
    "400 Bad Request context length exceeded",
    "400 Bad Request: max_tokens 8500 exceeds context length",   # "8500" ⊃ "500"
    "429 Too Many Requests",
    "429 Too Many Requests (retry after 5040ms)",                # "5040" ⊃ "504"
    "Request timed out after 500 s",                             # "500 s"
    # httpx's raise_for_status() rendering — the shape executor.py:396 stores.
    "Client error '400 Bad Request' for url 'http://localhost:8000/v1/completions'",
    "Error code: 400 - {'message': 'max_tokens 8500 is too large'}",
]


class TestFatalErrorOracle:
    """_check_error_rate fires on fatal server errors regardless of rate.

    A 5xx *response*, or a transport failure where no status ever came back,
    fires on a single occurrence.  Every 4xx (400 context length, 429 rate
    limit) and every timeout is ignored — those are the expected outcome of
    fuzzing an inference server, not evidence that it is unhealthy.
    """
    oracle = Oracle()

    def test_no_errors_passes(self):
        result = make_result(completed_request("r1"), completed_request("r2"))
        assert self.oracle._check_error_rate(result).passed

    def test_http_400_ignored(self):
        """4xx / timeouts pass — *including* when their text contains a 5xx run.

        Regression guard.  ``_check_error_rate`` used to test
        ``any(p in r.error for p in _FATAL_PATTERNS)`` with
        ``_FATAL_PATTERNS = ("500", "502", "503", "504", ...)``, a bare
        substring match, so "max_tokens 8500", "retry after 5040ms" and
        "timed out after 500 s" each filed a LIVENESS_FAILURE finding.  These
        are the commonest errors under fuzzing load, so the false positives
        dominated the ledger.  The literals below are chosen to fail against
        that implementation: keep the embedded digit runs.
        """
        for err in BENIGN_ERRORS:
            r = self.oracle._check_error_rate(make_result(req_error("r1", err)))
            assert r.passed, f"benign error filed as a finding: {err!r}"
        # ...and all of them at once, alongside a healthy request.
        mixed = make_result(
            *[req_error(f"r{i}", e) for i, e in enumerate(BENIGN_ERRORS)],
            completed_request("ok"),
        )
        assert self.oracle._check_error_rate(mixed).passed

    def test_http_500_fires(self):
        """Any 5xx fires, in both spellings the client can produce."""
        for err in ("500 Internal Server Error",
                    "502 Bad Gateway",
                    "Server error '503 Service Unavailable' "
                    "for url 'http://localhost:8000/v1/completions'",
                    "Error code: 500 - {'object': 'error'}"):
            r = self.oracle._check_error_rate(make_result(req_error("r1", err)))
            assert not r.passed, err
            assert r.bug_class == BugClass.LIVENESS_FAILURE

    def test_connection_refused_fires(self):
        """Transport failures fire: no status came back at all."""
        for err in ("Connection refused",
                    "[Errno 111] Connection refused",
                    "Connection reset by peer",
                    "ConnectionRefusedError: [Errno 111]",
                    "ConnectError: failed to connect"):
            r = self.oracle._check_error_rate(make_result(req_error("r1", err)))
            assert not r.passed, err
            assert r.bug_class == BugClass.LIVENESS_FAILURE

    def test_one_502_among_many_400s_fires(self):
        reqs = [req_error(f"r{i}", "400 Bad Request") for i in range(9)]
        reqs.append(req_error("fatal", "502 Bad Gateway"))
        result = make_result(*reqs)
        r = self.oracle._check_error_rate(result)
        assert not r.passed
        assert "502" in r.detail
        # The nine 400s must not inflate the score: 1 fatal request, not 10.
        assert r.score == 5.0


# ---------------------------------------------------------------------------
# Abnormal length oracle
# ---------------------------------------------------------------------------

def req_with_tokens(rid: str = "r1", tokens_received: int = 10,
                    max_tokens_requested: int = 64,
                    error: str = None,
                    cancelled: bool = False) -> RequestResult:
    now = time.time()
    r = RequestResult(
        request_id=rid,
        sent_at=now - 1.0,
        completed_at=now if not cancelled else None,
        cancelled_at=now if cancelled else None,
        tokens_received=tokens_received,
        max_tokens_requested=max_tokens_requested,
    )
    if error:
        r.error = error
    return r


# ---------------------------------------------------------------------------
# Repetition oracle
# ---------------------------------------------------------------------------

def req_with_text(rid: str = "r1", text: str = "", error: str = None,
                  cancelled: bool = False) -> RequestResult:
    now = time.time()
    r = RequestResult(
        request_id=rid,
        sent_at=now - 1.0,
        completed_at=now if not cancelled else None,
        cancelled_at=now if cancelled else None,
        tokens_received=len(text.split()),
        output_text=text,
    )
    if error:
        r.error = error
    return r


class TestRepetitionOracle:
    oracle = Oracle()

    def test_clean_output_passes(self):
        text = " ".join(f"word{i}" for i in range(30))
        result = make_result(req_with_text("r1", text))
        findings = self.oracle._check_repetition(result)
        assert all(f.passed for f in findings)

    def test_repetitive_output_fires(self):
        text = "a " * 20
        result = make_result(req_with_text("r1", text))
        findings = self.oracle._check_repetition(result)
        assert any(not f.passed and f.bug_class == BugClass.REPETITION for f in findings)

    def test_short_output_skipped(self):
        text = " ".join(f"w{i}" for i in range(5))
        result = make_result(req_with_text("r1", text))
        findings = self.oracle._check_repetition(result)
        assert all(f.passed for f in findings)

    def test_repetition_late_in_output_fires(self):
        clean = " ".join(f"word{i}" for i in range(50))
        repeated = " ".join(["rep"] * 20)
        text = clean + " " + repeated
        result = make_result(req_with_text("r1", text))
        findings = self.oracle._check_repetition(result)
        assert any(not f.passed and f.bug_class == BugClass.REPETITION for f in findings)

    def test_errored_request_skipped(self):
        text = "a " * 25
        result = make_result(req_with_text("r1", text, error="500 Internal Server Error"))
        findings = self.oracle._check_repetition(result)
        assert all(f.passed for f in findings)


# ---------------------------------------------------------------------------
# Char corruption oracle
# ---------------------------------------------------------------------------

class TestCharCorruptionOracle:
    oracle = Oracle()

    def test_clean_ascii_passes(self):
        result = make_result(req_with_text("r1", "Hello, world! This is clean text."))
        findings = self.oracle._check_char_corruption(result)
        assert all(f.passed for f in findings)

    def test_replacement_char_fires(self):
        result = make_result(req_with_text("r1", "Hello \ufffd world"))
        findings = self.oracle._check_char_corruption(result)
        assert any(not f.passed and f.bug_class == BugClass.CHAR_CORRUPTION for f in findings)

    def test_null_byte_fires(self):
        result = make_result(req_with_text("r1", "Hello\x00world"))
        findings = self.oracle._check_char_corruption(result)
        assert any(not f.passed and f.bug_class == BugClass.CHAR_CORRUPTION for f in findings)

    def test_newline_tab_cr_pass(self):
        result = make_result(req_with_text("r1", "line1\nline2\ttabbed\rcarriage"))
        findings = self.oracle._check_char_corruption(result)
        assert all(f.passed for f in findings)

    def test_vertical_tab_fires(self):
        result = make_result(req_with_text("r1", "Hello\x0bworld"))
        findings = self.oracle._check_char_corruption(result)
        assert any(not f.passed and f.bug_class == BugClass.CHAR_CORRUPTION for f in findings)

    def test_del_fires(self):
        result = make_result(req_with_text("r1", "Hello\x7fworld"))
        findings = self.oracle._check_char_corruption(result)
        assert any(not f.passed and f.bug_class == BugClass.CHAR_CORRUPTION for f in findings)

    def test_cjk_unicode_passes(self):
        result = make_result(req_with_text("r1", "\u4e2d\u6587 text is fine"))
        findings = self.oracle._check_char_corruption(result)
        assert all(f.passed for f in findings)

    def test_errored_request_skipped(self):
        result = make_result(req_with_text("r1", "Hello\ufffdworld",
                                           error="500 Internal Server Error"))
        findings = self.oracle._check_char_corruption(result)
        assert all(f.passed for f in findings)


# ---------------------------------------------------------------------------
# Feature failure oracle
# ---------------------------------------------------------------------------

def req_with_features(rid: str = "r1",
                      logprobs_requested: int = None,
                      n_completions_requested: int = None,
                      response_has_logprobs: bool = None,
                      response_n_choices: int = None,
                      error: str = None,
                      cancelled: bool = False,
                      stream: bool = False) -> RequestResult:
    now = time.time()
    r = RequestResult(
        request_id=rid,
        sent_at=now - 1.0,
        completed_at=now if not cancelled else None,
        cancelled_at=now if cancelled else None,
        tokens_received=10,
        stream=stream,
        logprobs_requested=logprobs_requested,
        n_completions_requested=n_completions_requested,
        response_has_logprobs=response_has_logprobs,
        response_n_choices=response_n_choices,
    )
    if error:
        r.error = error
    return r


class TestFeatureFailureOracle:
    oracle = Oracle()

    def test_logprobs_requested_absent_fires(self):
        result = make_result(req_with_features("r1", logprobs_requested=5,
                                               response_has_logprobs=False))
        findings = self.oracle._check_feature_failure(result)
        assert any(not f.passed and f.bug_class == BugClass.FEATURE_FAILURE for f in findings)

    def test_logprobs_requested_present_passes(self):
        result = make_result(req_with_features("r1", logprobs_requested=5,
                                               response_has_logprobs=True))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)

    def test_logprobs_not_requested_passes(self):
        result = make_result(req_with_features("r1", logprobs_requested=None,
                                               response_has_logprobs=False))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)

    def test_logprobs_streaming_skipped(self):
        # Streaming logprobs presence is unreliable — should not fire
        result = make_result(req_with_features("r1", logprobs_requested=20,
                                               response_has_logprobs=False, stream=True))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)

    def test_n_completions_wrong_fires(self):
        result = make_result(req_with_features("r1", n_completions_requested=4,
                                               response_n_choices=2))
        findings = self.oracle._check_feature_failure(result)
        assert any(not f.passed and f.bug_class == BugClass.FEATURE_FAILURE for f in findings)

    def test_n_completions_correct_passes(self):
        result = make_result(req_with_features("r1", n_completions_requested=4,
                                               response_n_choices=4))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)

    def test_n_completions_none_passes(self):
        result = make_result(req_with_features("r1", n_completions_requested=None,
                                               response_n_choices=1))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)

    def test_n_completions_streaming_skipped(self):
        # Streaming n>1 choice-counting is unreliable — should not fire
        result = make_result(req_with_features("r1", n_completions_requested=4,
                                               response_n_choices=1, stream=True))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)

    def test_cancelled_skipped(self):
        result = make_result(req_with_features("r1", logprobs_requested=5,
                                               response_has_logprobs=False,
                                               cancelled=True))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)

    def test_errored_skipped(self):
        result = make_result(req_with_features("r1", logprobs_requested=5,
                                               response_has_logprobs=False,
                                               error="500 Internal Server Error"))
        findings = self.oracle._check_feature_failure(result)
        assert all(f.passed for f in findings)




# ---------------------------------------------------------------------------
# Structured-output exact oracle (Workstream 2)
# ---------------------------------------------------------------------------

_FLAT_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["name", "count"],
    "additionalProperties": False,
}

_VALID_JSON = '{"name": "a", "count": 1}'


def _schema_trace(*, rid="r1", **ev_kwargs):
    from fuzzer.trace import ActionType, Event, Trace
    kw = dict(action=ActionType.SEND, offset_ms=0, request_id=rid,
              prompt_len=32, max_tokens=64, stream=True, temperature=0.0)
    kw.update(ev_kwargs)
    return Trace(seed=1, events=[Event(**kw)])


def _json_schema_trace(schema=_FLAT_SCHEMA, rid="r1"):
    return _schema_trace(
        rid=rid,
        response_format={"type": "json_schema",
                         "json_schema": {"name": "t", "schema": schema}})


def _schema_request(rid="r1", text=_VALID_JSON, finish_reason="stop",
                    error=None, cancelled_at=None, **kwargs) -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=rid, sent_at=now - 1.0, first_token_at=now - 0.9,
        completed_at=now, cancelled_at=cancelled_at, output_text=text,
        finish_reason=finish_reason, error=error, tokens_received=8,
        **kwargs,
    )


def _check(trace, *requests):
    return [f for f in Oracle()._check_schema_violation(make_result(*requests), trace)
            if not f.passed]


class TestSchemaViolationOracle:
    """The one oracle with no FP statistics: the engine guarantees the grammar,
    so `finish_reason == "stop"` plus a failing validation is exact."""

    # -- fires --------------------------------------------------------------

    def test_fires_on_stop_plus_unparseable_json(self):
        bugs = _check(_json_schema_trace(), _schema_request(text='{"name": "a"'))
        assert len(bugs) == 1
        assert bugs[0].bug_class is BugClass.SCHEMA_VIOLATION
        assert bugs[0].probe_metadata["failure"] == "json_parse"
        assert bugs[0].probe_metadata["constraint"] == "json_schema"
        assert bugs[0].probe_metadata["request_id"] == "r1"
        assert bugs[0].score > 0

    def test_fires_on_parseable_but_schema_invalid(self):
        # valid JSON, but "count" is missing → only the schema check can see it
        bugs = _check(_json_schema_trace(), _schema_request(text='{"name": "a"}'))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["failure"] == "json_schema"

    def test_fires_on_prose_around_the_object(self):
        bugs = _check(_json_schema_trace(),
                      _schema_request(text=_VALID_JSON + " and that is all"))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["failure"] == "json_parse"
        assert "trailing content" in bugs[0].probe_metadata["reason"]

    def test_guided_json_field_is_honoured(self):
        bugs = _check(_schema_trace(guided_json=_FLAT_SCHEMA),
                      _schema_request(text="not json"))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["constraint"] == "json_schema"

    def test_regex_constraint(self):
        trace = _schema_trace(guided_regex=r"[0-9]{3}")
        assert _check(trace, _schema_request(text="123")) == []
        bugs = _check(trace, _schema_request(text="12a"))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["failure"] == "regex"

    def test_choice_constraint(self):
        trace = _schema_trace(guided_choice=["yes", "no"])
        assert _check(trace, _schema_request(text="yes")) == []
        assert _check(trace, _schema_request(text=" no\n")) == []   # stripped
        bugs = _check(trace, _schema_request(text="maybe"))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["failure"] == "choice"

    def test_json_object_mode_checks_parseability_only(self):
        trace = _schema_trace(response_format={"type": "json_object"})
        # schema-invalid but parseable → silent (json_object promises no schema)
        assert _check(trace, _schema_request(text='{"whatever": 1}')) == []
        bugs = _check(trace, _schema_request(text="{oops"))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["failure"] == "json_parse"

    # -- silent on valid ----------------------------------------------------

    def test_silent_on_valid_json(self):
        assert _check(_json_schema_trace(), _schema_request()) == []

    def test_silent_when_no_constraint_declared(self):
        assert _check(_schema_trace(), _schema_request(text="not json at all")) == []

    def test_silent_when_trace_is_none(self):
        assert _check(None, _schema_request(text="not json")) == []

    def test_silent_when_request_has_no_matching_event(self):
        assert _check(_json_schema_trace(rid="other"),
                      _schema_request(text="not json")) == []

    # -- FP guard 1: truncation --------------------------------------------

    @pytest.mark.parametrize("reason", ["length", "error", None])
    def test_silent_when_not_finished_with_stop(self, reason):
        assert _check(_json_schema_trace(),
                      _schema_request(text='{"name": "a"',
                                      finish_reason=reason)) == []

    # -- FP guard 2: server rejection --------------------------------------

    def test_silent_on_4xx_and_recorded_as_capability_gap(self):
        from fuzzer.core import oracles as _o
        before = sum(_o._CAPABILITY_GAPS.values())
        bugs = _check(_json_schema_trace(),
                      _schema_request(text=None,
                                      error="400 Bad Request: unsupported $ref"))
        assert bugs == []
        assert sum(_o._CAPABILITY_GAPS.values()) == before + 1
        assert any("400" in k for k in _o.capability_gaps())

    def test_unsatisfiable_schema_is_a_capability_gap_not_a_bug(self):
        pytest.importorskip("jsonschema")
        trace = _json_schema_trace(schema={"type": "not-a-type"})
        bugs = _check(trace, _schema_request(text='{"name": "a"}'))
        assert bugs == []

    def test_silent_on_cancelled_request(self):
        assert _check(_json_schema_trace(),
                      _schema_request(text="{oops", cancelled_at=time.time())) == []

    # -- FP guard 3: trailing whitespace -----------------------------------

    @pytest.mark.parametrize("text", [
        _VALID_JSON + "\n",
        _VALID_JSON + "   ",
        "  " + _VALID_JSON + " \n\t",
    ])
    def test_trailing_whitespace_is_accepted(self, text):
        assert _check(_json_schema_trace(), _schema_request(text=text)) == []

    # -- FP guard 4: nothing generated -------------------------------------

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_output_is_not_a_violation(self, text):
        assert _check(_json_schema_trace(), _schema_request(text=text)) == []

    # -- FP guard 5: n>1 join artifact -------------------------------------

    def test_n_gt_1_segments_are_validated_separately(self):
        joined = "\x1f".join([_VALID_JSON, _VALID_JSON])
        assert _check(_json_schema_trace(), _schema_request(text=joined)) == []

    def test_n_gt_1_reports_the_offending_segment_only(self):
        joined = "\x1f".join([_VALID_JSON, "{broken"])
        bugs = _check(_json_schema_trace(), _schema_request(text=joined))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["sequence_index"] == 1
        assert bugs[0].probe_metadata["output"] == "{broken"

    # -- signature ----------------------------------------------------------

    @pytest.mark.parametrize("kwargs,text,expected", [
        (dict(), '{"name": "a"', "sv_json_parse"),
        (dict(), '{"name": "a"}', "sv_json_schema"),
        (dict(guided_regex="[0-9]+"), "abc", "sv_regex"),
        (dict(guided_choice=["yes"]), "no", "sv_choice"),
    ])
    def test_subtypes_do_not_collapse(self, kwargs, text, expected):
        from fuzzer.core.finding_sig import oracle_subtype
        trace = _json_schema_trace() if not kwargs else _schema_trace(**kwargs)
        bugs = _check(trace, _schema_request(text=text))
        assert len(bugs) == 1
        assert oracle_subtype(bugs[0].bug_class, bugs[0].detail,
                              bugs[0].probe_metadata) == expected

    def test_four_failure_modes_give_four_signatures(self):
        from fuzzer.core.finding_sig import mechanism_signature
        sigs = set()
        for kwargs, text in [(dict(), '{"name": "a"'),
                             (dict(), '{"name": "a"}'),
                             (dict(guided_regex="[0-9]+"), "abc"),
                             (dict(guided_choice=["yes"]), "no")]:
            trace = _json_schema_trace() if not kwargs else _schema_trace(**kwargs)
            bugs = _check(trace, _schema_request(text=text))
            sigs.add(mechanism_signature(trace, bugs))
        assert len(sigs) == 4


# ---------------------------------------------------------------------------
# Degenerate composed logits mask (mask-composition surface)
# ---------------------------------------------------------------------------

SENTINEL = -9999.0


def _mask_request(rid="r1", *, logprobs=None, top=None, tokens=None,
                  restrictors=None, error=None, finish_reason="stop",
                  output_text="!!!", cancelled_at=None) -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=rid, sent_at=now - 1.0, first_token_at=now - 0.9,
        completed_at=now, cancelled_at=cancelled_at, stream=False,
        output_text=output_text, finish_reason=finish_reason, error=error,
        tokens_received=len(tokens or []) or 3,
        output_logprobs=logprobs, output_top_logprobs=top, output_tokens=tokens,
        mask_restrictors=restrictors,
    )


def _degenerate_request(rid="r1", *, position=1, restrictors=None,
                        **overrides) -> RequestResult:
    """The measured shape: at *position* the sampled token carries the sentinel
    and it is in the top-k at the sentinel, while ``<|im_end|>`` sits at 0.0."""
    tokens = ["{", "!", "!"]
    logprobs = [-0.1] * len(tokens)
    top = [{"{": -0.1, "a": -2.0}] * len(tokens)
    logprobs[position] = SENTINEL
    top[position] = {"!": SENTINEL, "<|im_end|>": 0.0, "#": SENTINEL}
    kw = dict(logprobs=logprobs, top=top, tokens=tokens,
              restrictors=restrictors if restrictors is not None else
              {"structured_output": "json_schema", "min_tokens": 64})
    kw.update(overrides)
    return _mask_request(rid, **kw)


class TestSchemaViolationStopReasonGuard:
    """Regression: `finish_reason == "stop"` is NOT proof the FSM accepted.

    Three findings in the 7352699_tight15_s7 campaign were filed against
    request `p7`, whose SEND (recovered from
    `corpus/seed_471613229.json`, and byte-identical in seeds 2014234619 /
    538992729) was::

        min_tokens=32, logprobs=5, max_tokens=64,
        bad_words=['{', '"', ':', '}'],
        stop=['In ', 'Second', '\\n\\n'],
        response_format={'type': 'json_schema', ...}

    `bad_words` banned the quote, so under the JSON grammar whitespace was the
    only legal continuation; `min_tokens` held stop-string checking off below
    the floor and the pre-floor text is permanently outside every later search
    window (`v1/engine/detokenizer.py:117-123,340`), so literal `\\n\\n`
    accumulated *inside* the returned string.  The first post-floor `\\n\\n`
    then truncated the text and overwrote the finish reason
    (`detokenizer.py:138-141`, `v1/engine/output_processor.py:659-661`),
    yielding `{\\n\\n  \\n\\n\\n...` with `finish_reason == "stop"` and
    `stop_reason == "\\n\\n"`.

    The grammar could not have terminated: `}` was in `bad_words` and EOS is
    outside an unfinished object's bitmask, so the stop string is the only
    reachable route to `"stop"` here.
    """

    #: The p7 output, as stored on all three findings (excerpt).
    _P7_OUTPUT = "{\n\n  " + "\n" * 60 + "  "

    def _p7_trace(self):
        return _schema_trace(
            rid="p7", min_tokens=32, logprobs=5,
            stop=["In ", "Second", "\n\n"],
            bad_words=["{", "\"", ":", "}"],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "mask_compose",
                                             "schema": _FLAT_SCHEMA}})

    def test_the_three_campaign_findings_are_suppressed(self):
        """A stop *string* truncated the output → not the FSM's doing."""
        bugs = _check(self._p7_trace(),
                      _schema_request(rid="p7", text=self._P7_OUTPUT,
                                      finish_reason="stop",
                                      stop_reason="\n\n",
                                      stop_reason_reported=True,
                                      had_stop_sequences=True))
        assert bugs == []

    def test_same_output_without_a_stop_reason_still_fires(self):
        """Not a blanket suppression: identical text, genuine EOS → filed."""
        bugs = _check(self._p7_trace(),
                      _schema_request(rid="p7", text=self._P7_OUTPUT,
                                      finish_reason="stop",
                                      stop_reason=None,
                                      stop_reason_reported=True,
                                      had_stop_sequences=True))
        assert len(bugs) == 1
        assert bugs[0].bug_class is BugClass.SCHEMA_VIOLATION

    def test_requested_stop_token_id_is_suppressed(self):
        """`stop_token_ids` fires before the length cap (sched/utils.py:108-117)."""
        trace = _schema_trace(
            rid="r1", stop_token_ids=[13],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "t", "schema": _FLAT_SCHEMA}})
        bugs = _check(trace,
                      _schema_request(text='{"name": "a"', stop_reason=13,
                                      stop_reason_reported=True))
        assert bugs == []

    def test_generation_config_eos_id_is_not_suppressed(self):
        """The false-negative trap the guard must avoid.

        `sampling_params.py:661-674` silently folds the checkpoint's *secondary*
        EOS ids into `stop_token_ids`.  Qwen2.5-0.5B-Instruct declares
        `eos_token_id: [151645, 151643]` with primary 151645, so a voluntary
        `<|endoftext|>` (151643) reports an int the request never asked for.
        That is a real accepting-state stop and must still be judged.
        """
        bugs = _check(_json_schema_trace(),
                      _schema_request(text='{"name": "a"', stop_reason=151643,
                                      stop_reason_reported=True))
        assert len(bugs) == 1
        assert bugs[0].bug_class is BugClass.SCHEMA_VIOLATION

    def test_backend_that_never_reports_stop_reason_is_suppressed_when_stops_set(self):
        """No stop_reason field + stop strings in play → cannot prove voluntary.

        This is the only reader of `RequestResult.had_stop_sequences`.
        """
        trace = _schema_trace(
            rid="r1", stop=["\n\n"],
            response_format={"type": "json_schema",
                             "json_schema": {"name": "t", "schema": _FLAT_SCHEMA}})
        bugs = _check(trace,
                      _schema_request(text='{"name": "a"',
                                      stop_reason=None,
                                      stop_reason_reported=False,
                                      had_stop_sequences=True))
        assert bugs == []

    def test_backend_that_never_reports_stop_reason_still_fires_without_stops(self):
        """No stop condition was in play at all → nothing to explain it away."""
        bugs = _check(_json_schema_trace(),
                      _schema_request(text='{"name": "a"',
                                      stop_reason=None,
                                      stop_reason_reported=False,
                                      had_stop_sequences=False))
        assert len(bugs) == 1

    def test_default_result_is_unchanged(self):
        """A RequestResult nobody populated keeps the pre-fix behaviour."""
        bugs = _check(_json_schema_trace(), _schema_request(text='{"name": "a"'))
        assert len(bugs) == 1

    @pytest.mark.parametrize("stop_reason", ["\n\n", "In ", "Second", True])
    def test_str_and_bool_stop_reasons_never_pass(self, stop_reason):
        bugs = _check(_json_schema_trace(),
                      _schema_request(text='{"name": "a"',
                                      stop_reason=stop_reason,
                                      stop_reason_reported=True))
        assert bugs == []


def _dm(*requests):
    return [f for f in Oracle()._check_degenerate_mask(make_result(*requests))
            if not f.passed]


class TestDegenerateMaskSentinel:
    """The generic predicate: the *sampled* token carries the -inf sentinel
    while a rival at the same position is finite."""

    def test_fires_on_the_measured_shape(self):
        bugs = _dm(_degenerate_request())
        assert len(bugs) == 1
        assert bugs[0].bug_class is BugClass.DEGENERATE_MASK
        pm = bugs[0].probe_metadata
        assert pm["predicate"] == "sentinel"
        assert pm["position"] == 1
        assert pm["sampled_token"] == "!"
        assert pm["sampled_logprob"] == SENTINEL
        assert pm["best_alternative_logprob"] == 0.0
        assert pm["restrictors"] == ["min_tokens", "structured_output"]
        assert pm["pre_logprob_restrictors"] == ["structured_output"]
        assert bugs[0].score > 0

    def test_reports_the_first_offending_position_only(self):
        r = _degenerate_request(position=1)
        r.output_logprobs[2] = SENTINEL
        r.output_top_logprobs[2] = {"!": SENTINEL, "b": -0.5}
        bugs = _dm(r)
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["position"] == 1

    def test_fires_without_any_declared_restrictor(self):
        """Deliberately generic: a sampled token the pre-sampler mask had
        already excluded is a defect whatever produced it, including a
        composition the fuzzer never asked for."""
        bugs = _dm(_degenerate_request(restrictors={}))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["restrictors"] == []

    # -- FP guard 1: a legitimately banned token is not a bug ---------------

    def test_silent_when_a_banned_token_was_not_the_one_sampled(self):
        """The normal appearance of a *working* mask: rivals at the sentinel,
        a legal token actually sampled."""
        r = _mask_request(
            logprobs=[-0.1, -0.2],
            top=[{"{": -0.1, "!": SENTINEL}, {"a": -0.2, "!": SENTINEL}],
            tokens=["{", "a"])
        assert _dm(r) == []

    # -- FP guard 2: the sentinel could be a real logprob -------------------

    def test_silent_on_a_near_sentinel_value(self):
        """-9999.0 is a ``max()`` clamp, so the test is exact equality."""
        r = _degenerate_request()
        r.output_logprobs[1] = -9998.7
        r.output_top_logprobs[1] = {"!": -9998.7, "<|im_end|>": 0.0}
        assert _dm(r) == []

    def test_silent_when_the_sampled_token_is_absent_from_the_topk(self):
        """ChatCompletionLogProb.logprob defaults to -9999.0 and the chat
        serialiser omits the kwarg exactly when the sampled token is missing
        from that step's top-k — -9999.0 with no -inf behind it."""
        r = _degenerate_request()
        r.output_top_logprobs[1] = {"<|im_end|>": 0.0, "#": -1.0}
        assert _dm(r) == []

    def test_silent_when_the_topk_entry_is_finite(self):
        r = _degenerate_request()
        r.output_top_logprobs[1] = {"!": -3.0, "<|im_end|>": 0.0}
        assert _dm(r) == []

    def test_silent_when_no_rival_is_finite(self):
        """A row whose alternatives are all at the sentinel is a missing top-k
        or a different defect, not evidence of a composition."""
        r = _degenerate_request()
        r.output_top_logprobs[1] = {"!": SENTINEL, "#": SENTINEL}
        assert _dm(r) == []

    def test_silent_when_the_only_rival_is_below_the_floor(self):
        r = _degenerate_request()
        r.output_top_logprobs[1] = {"!": SENTINEL, "#": -50000.0}
        assert _dm(r) == []

    @pytest.mark.parametrize("bad", [float("nan"), float("-inf")])
    def test_silent_on_non_finite_rivals(self, bad):
        r = _degenerate_request()
        r.output_top_logprobs[1] = {"!": SENTINEL, "#": bad}
        assert _dm(r) == []

    # -- FP guard 3: no logprobs, no verdict --------------------------------

    @pytest.mark.parametrize("missing", ["logprobs", "top", "tokens"])
    def test_silent_without_the_logprob_data(self, missing):
        r = _degenerate_request()
        setattr(r, {"logprobs": "output_logprobs", "top": "output_top_logprobs",
                    "tokens": "output_tokens"}[missing], None)
        assert _dm(r) == []

    # -- FP guard 4: cancelled / errored ------------------------------------

    def test_silent_on_cancelled(self):
        assert _dm(_degenerate_request(cancelled_at=time.time())) == []

    def test_silent_on_a_4xx(self):
        """A 400 is the server rejecting the combination up front — the honest
        outcome, and a capability gap rather than a defect."""
        assert _dm(_degenerate_request(
            error="400 Bad Request: allowed_token_ids is not None and empty!")) == []

    # -- FP guard 6: alignment ----------------------------------------------

    def test_does_not_read_past_the_shortest_list(self):
        r = _degenerate_request(position=2)
        r.output_top_logprobs = r.output_top_logprobs[:2]
        assert _dm(r) == []


class TestDegenerateMaskGrammarGuard:
    """The post-0.27 symptom: the downstream guard's HTTP 500."""

    _COMPOSED = {"structured_output": "json_schema", "min_tokens": 64}

    def _r(self, error, restrictors=None):
        return _mask_request(
            error=error, output_text=None, finish_reason=None,
            restrictors=self._COMPOSED if restrictors is None else restrictors)

    def test_fires_on_a_500_over_a_composed_mask(self):
        bugs = _dm(self._r("Error code: 500 - {'object': 'error'}"))
        assert len(bugs) == 1
        assert bugs[0].bug_class is BugClass.DEGENERATE_MASK
        pm = bugs[0].probe_metadata
        assert pm["predicate"] == "grammar_guard"
        assert pm["http_status"] == 500
        assert pm["log_corroborated"] is None

    @pytest.mark.parametrize("line,kind", [
        ("Failed to advance FSM for request chatcmpl-abc for tokens 0. "
         "Please file an issue.", "fsm"),
        ("Unexpected: grammar rejected tokens [0] for request chatcmpl-abc. "
         "Terminating request.", "reject"),
    ])
    def test_server_log_line_corroborates(self, line, kind):
        bugs = _dm(self._r("Error code: 500 - " + line))
        assert len(bugs) == 1
        assert bugs[0].probe_metadata["log_corroborated"] == kind

    def test_grammar_guard_signature_is_exact(self):
        from fuzzer.core.oracles import grammar_guard_signature
        assert grammar_guard_signature("") is None
        assert grammar_guard_signature("grammar rejected tokens for request") is None
        assert grammar_guard_signature(
            "Unexpected: grammar rejected tokens [7] for request r. "
            "Terminating request.") == "reject"

    # -- FP guards ----------------------------------------------------------

    @pytest.mark.parametrize("error", [
        "Client error '400 Bad Request' for url 'http://x/v1/completions'",
        "Server error '503 Service Unavailable' for url 'http://x/v1/completions'",
        "429 Too Many Requests (retry after 5040ms)",
        "Connection refused",
    ])
    def test_silent_on_anything_that_is_not_a_500(self, error):
        assert _dm(self._r(error)) == []

    def test_silent_on_a_500_with_only_one_restrictor(self):
        assert _dm(self._r("Error code: 500 - {}",
                           restrictors={"min_tokens": 64})) == []

    def test_silent_on_a_500_with_no_grammar(self):
        """The guard lives inside the xgrammar backend, so a 500 on a request
        with no structured output cannot be it."""
        assert _dm(self._r("Error code: 500 - {}",
                           restrictors={"min_tokens": 64,
                                        "allowed_token_ids": [1]})) == []

    def test_silent_on_a_500_with_no_restrictors_at_all(self):
        assert _dm(self._r("Error code: 500 - {}", restrictors={})) == []

    def test_the_same_500_still_files_a_liveness_failure(self):
        """Deliberate: the availability defect is real and the two sub-types
        keep the ledger entries apart."""
        result = make_result(self._r("Error code: 500 - {}"))
        classes = {f.bug_class for f in Oracle().evaluate(result, make_session())}
        assert BugClass.DEGENERATE_MASK in classes
        assert BugClass.LIVENESS_FAILURE in classes


class TestDegenerateMaskAllowlistEscape:
    """The processor-only composition the reported logprobs cannot show."""

    _COMPOSED = {"allowed_token_ids": [5], "stop_token_ids": [5],
                 "min_tokens": 16}

    def _r(self, tokens, restrictors=None, **kw):
        return _mask_request(tokens=tokens, output_text="".join(tokens),
                             restrictors=(self._COMPOSED if restrictors is None
                                          else restrictors), **kw)

    def test_fires_when_more_distinct_tokens_than_allowlist_ids(self):
        bugs = _dm(self._r(["!", "a", "b"]))
        assert len(bugs) == 1
        pm = bugs[0].probe_metadata
        assert pm["predicate"] == "allowlist_escape"
        assert pm["allowed_token_ids"] == [5]
        assert pm["min_tokens"] == 16
        assert pm["distinct_tokens_in_window"] == 3

    def test_silent_when_the_allowlist_could_account_for_the_output(self):
        """One id renders as one string, so a single repeated string is not
        proof of anything."""
        assert _dm(self._r(["!", "!", "!"])) == []

    def test_multi_id_allowlist_needs_more_distinct_than_ids(self):
        r = self._r(["a", "b"], restrictors={"allowed_token_ids": [1, 2, 3],
                                             "stop_token_ids": [1, 2, 3],
                                             "min_tokens": 8})
        assert _dm(r) == []
        r = self._r(["a", "b", "c", "d"],
                    restrictors={"allowed_token_ids": [1, 2, 3],
                                 "stop_token_ids": [1, 2, 3], "min_tokens": 8})
        assert len(_dm(r)) == 1

    def test_only_the_window_below_the_floor_counts(self):
        """Above ``min_tokens`` the ban lifts and the allowed id is legal
        again, so tokens there prove nothing."""
        r = self._r(["!", "!", "a", "b", "c"],
                    restrictors={"allowed_token_ids": [5],
                                 "stop_token_ids": [5], "min_tokens": 2})
        assert _dm(r) == []

    # -- FP guards ----------------------------------------------------------

    def test_silent_when_the_allowlist_is_not_inside_the_stop_set(self):
        """Without that containment the ban is not provable, so neither is the
        escape."""
        assert _dm(self._r(["a", "b"],
                           restrictors={"allowed_token_ids": [5],
                                        "stop_token_ids": [6],
                                        "min_tokens": 16})) == []

    @pytest.mark.parametrize("drop", ["allowed_token_ids", "stop_token_ids",
                                      "min_tokens"])
    def test_silent_when_a_leg_of_the_composition_is_missing(self, drop):
        restrictors = dict(self._COMPOSED)
        restrictors.pop(drop)
        assert _dm(self._r(["a", "b"], restrictors=restrictors)) == []

    def test_silent_without_token_strings(self):
        r = self._r(["a", "b"])
        r.output_tokens = None
        assert _dm(r) == []

    @pytest.mark.parametrize("reason", ["error", None])
    def test_silent_when_the_request_did_not_claim_success(self, reason):
        assert _dm(self._r(["a", "b"], finish_reason=reason)) == []

    def test_silent_on_cancelled(self):
        assert _dm(self._r(["a", "b"], cancelled_at=time.time())) == []


class TestDegenerateMaskWiring:
    def test_runs_inside_evaluate(self):
        findings = Oracle().evaluate(make_result(_degenerate_request()),
                                     make_session())
        assert any(f.bug_class is BugClass.DEGENERATE_MASK for f in findings)

    def test_clean_traces_stay_clean(self):
        findings = Oracle().evaluate(make_result(completed_request()),
                                     make_session())
        assert not any(f.bug_class is BugClass.DEGENERATE_MASK
                       for f in findings)

    def test_no_restrictors_no_logprobs_is_a_noop(self):
        assert _dm(_mask_request()) == []


class TestSchemaViolationAdapterWiring:
    """The oracle only ever runs through evaluate_extra — that is the one hook
    the runner passes a Trace to."""

    def _adapter(self):
        from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
        return VllmBaseAdapter()

    def test_evaluate_extra_returns_the_finding(self):
        out = self._adapter().evaluate_extra(
            make_result(_schema_request(text="{broken")), make_session(),
            trace=_json_schema_trace())
        assert len(out) == 1
        assert out[0].bug_class is BugClass.SCHEMA_VIOLATION

    def test_evaluate_extra_without_trace_is_a_noop(self):
        assert self._adapter().evaluate_extra(
            make_result(_schema_request(text="{broken")), make_session()) == []

    def test_evaluate_extra_drops_passing_results(self):
        assert self._adapter().evaluate_extra(
            make_result(_schema_request()), make_session(),
            trace=_json_schema_trace()) == []
