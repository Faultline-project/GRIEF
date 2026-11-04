"""
Monitor: collects system-level metrics from the inference server and GPU.

Two implementations:
  - NvmlMonitor   real GPU metrics via pynvml + vLLM /metrics scraping
  - MockMonitor   returns synthetic data; used for testing without a GPU
"""

from __future__ import annotations

import dataclasses
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

# Fingerprint schema versions.
#   1 — the historical 8-dimension scheduler fingerprint (unchanged).
#   2 — adds ttft_spread_bucket / concurrency_peak_bucket / error_class_bucket,
#       three *workload*-derived dimensions that move under the burst/timing
#       mutators, while keeping all 8 v1 dimensions (see build_fingerprint).
#   3 — same eleven dimensions as v2, but four of them are *measured* for the
#       first time.  Every fingerprint we ever recorded scraped Prometheus names
#       that vLLM v1 does not export (``vllm:gpu_cache_usage_perc``,
#       ``vllm:num_requests_finished``, ``vllm:gpu_prefix_cache_hit_rate``,
#       ``vllm:num_requests_swapped``), so ``kv_peak_bucket``,
#       ``finished_count_bucket``, ``prefix_cache_hit_bucket`` and ``had_swap``
#       were pinned at 0/0/0/False in *all* archived v1 and v2 fingerprints.
#       See ``SchedulerFingerprint`` for what this does to comparability.
FP_VERSION_V1: int = 1
FP_VERSION_V2: int = 2
FP_VERSION_V3: int = 3
FP_VERSION_LATEST: int = FP_VERSION_V3

# ---------------------------------------------------------------------------
# Prometheus metric names.
#
# GROUND TRUTH: these were read out of the two SIF images we actually run
# (vllm_0.19.0.sif and vllm_0.27.1.sif) by enumerating
# the ``vllm:*`` literals in ``vllm/**/*.py`` together with their
# Gauge/Counter/Histogram constructor.  The full exported-sample-name lists are
# checked in at tests/baselines/vllm_metric_names_{0.19.0,0.27.1}.txt and
# tests/test_monitor.py asserts that every name below appears in both.
#
# vLLM v1 (both images) exports NONE of the v0 names this module used to scrape:
#   vllm:gpu_cache_usage_perc      -> vllm:kv_cache_usage_perc   (Gauge)
#   vllm:num_requests_finished     -> no equivalent; the count of finished
#                                     requests is sum(vllm:request_success_total)
#                                     over its finished_reason label.
#   vllm:gpu_prefix_cache_hit_rate -> no equivalent; v1 exports the two raw
#                                     counters vllm:prefix_cache_{hits,queries}
#                                     and the *rate* is ours to compute.
#   vllm:cpu_cache_usage_perc      -> no equivalent (no CPU KV pool in v1).
#   vllm:num_requests_swapped      -> no equivalent; v1 has no swapping at all,
#                                     preemption is recompute-only.
_M_KV_USAGE       = "vllm:kv_cache_usage_perc"          # Gauge, FRACTION 0..1
_M_RUNNING        = "vllm:num_requests_running"         # Gauge
_M_WAITING        = "vllm:num_requests_waiting"         # Gauge
_M_PREEMPTIONS    = "vllm:num_preemptions_total"        # Counter
_M_SUCCESS        = "vllm:request_success_total"        # Counter (per reason)
_M_PREFIX_HITS    = "vllm:prefix_cache_hits_total"      # Counter (tokens)
_M_PREFIX_QUERIES = "vllm:prefix_cache_queries_total"   # Counter (tokens)
_M_TTFT_SUM       = "vllm:time_to_first_token_seconds_sum"
_M_TTFT_COUNT     = "vllm:time_to_first_token_seconds_count"
# 0.27.1 only, and only when a CPU-offload KV connector is configured.  This is
# the closest v1 relative of the dead vllm:cpu_cache_usage_perc; it is recorded
# on the snapshot but deliberately NOT folded into the fingerprint (see
# SchedulerFingerprint.had_swap).
_M_OFFLOAD_CPU_USAGE = "vllm:kv_offload_cpu_cache_usage_perc"  # Gauge, 0..1

# Telemetry poll period, in seconds.
#
# Was 1.0, which is longer than a determinism-probe trace (~0.9 s): such a trace
# collected a single sample and every session delta diffed it against itself.
# The structural half of that is fixed in ``NvmlMonitor.stop()``, which now
# brackets the trace with an end-of-trace snapshot, so the poll rate only
# governs *within-trace resolution* (peak KV usage, peak queue depth) — peak
# concurrency already has an exact client-side sweep (``peak_concurrency``).
#
# Why 0.5 and not the 0.2 that was proposed.  Measured in the 0.19 SIF: a vLLM
# v1 /metrics body is ~500 samples / ~46 KB and ``generate_latest`` takes
# **3.8 ms**.  It is rendered *synchronously on the API server's asyncio event
# loop* — ``vllm/entrypoints/serve/instrumentator/metrics.py:41`` mounts
# ``prometheus_client.make_asgi_app`` — the same loop that forwards SSE token
# chunks.  (It is at least the cheap path: we launch via
# ``python3 -m vllm.entrypoints.openai.api_server``, which never calls
# ``setup_multiprocess_prometheus()`` — that lives in
# ``entrypoints/cli/serve.py:244`` — so ``PROMETHEUS_MULTIPROC_DIR`` is unset
# and the registry is the in-process one, not a MultiProcessCollector mmap scan.)
#
#   1.0 s -> 0.38 % of the event loop, a 3.8 ms stall every 1000 ms
#   0.5 s -> 0.76 %,                    a 3.8 ms stall every  500 ms
#   0.2 s -> 1.90 %,                    a 3.8 ms stall every  200 ms
#
# Inter-token latency for the 0.5B model we fuzz is ~5-15 ms, so at 5 Hz a
# meaningful fraction of an ITL is injected into the stream five times a second
# — and TTFT spread is itself a fingerprint dimension (``ttft_spread_bucket``).
# Perturbing the fingerprint's own input to buy resolution on another dimension
# is the wrong trade, so the rate is doubled rather than quintupled and the
# missing samples are recovered structurally instead.
DEFAULT_POLL_INTERVAL_S: float = 0.5

SCRAPED_METRIC_NAMES: tuple[str, ...] = (
    _M_KV_USAGE, _M_RUNNING, _M_WAITING, _M_PREEMPTIONS, _M_SUCCESS,
    _M_PREFIX_HITS, _M_PREFIX_QUERIES, _M_TTFT_SUM, _M_TTFT_COUNT,
)
# Not in the 0.19 image, so it is asserted against 0.27.1 only.
SCRAPED_METRIC_NAMES_0_27_ONLY: tuple[str, ...] = (_M_OFFLOAD_CPU_USAGE,)


def parse_prometheus_text(text: str) -> dict[str, list[float]]:
    """Map *exact* sample name -> every value exported under it.

    Exact-name keying matters: vLLM 0.27.1 exports both
    ``vllm:num_requests_waiting`` and ``vllm:num_requests_waiting_by_reason``,
    and the old ``line.startswith(...)`` scrape would have read the breakdown
    series as if it were the gauge.  Multiple values per name are normal — one
    per ``engine`` label under data parallelism, one per ``finished_reason``
    for ``vllm:request_success_total`` — so callers decide whether to sum or max.
    """
    out: dict[str, list[float]] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        # "name{labels} value" or "name value"; labels may contain spaces.
        brace = line.find("{")
        if brace != -1:
            name  = line[:brace]
            close = line.rfind("}")
            rest  = line[close + 1:] if close > brace else ""
        else:
            name, _, rest = line.partition(" ")
        try:
            value = float(rest.split()[-1])
        except (IndexError, ValueError):
            continue    # unparseable — do not create an empty entry for `name`
        out.setdefault(name, []).append(value)
    return out


def _sum(parsed: dict[str, list[float]], name: str) -> Optional[float]:
    vals = parsed.get(name)
    return sum(vals) if vals else None


def _max(parsed: dict[str, list[float]], name: str) -> Optional[float]:
    vals = parsed.get(name)
    return max(vals) if vals else None

# Minimum number of observed TTFTs before a p99/median ratio means anything.
# Reuses the oracle's own minimum sample size so the two never drift.
# (fuzzer.core.types imports nothing from the package, so this cannot cycle.)
from .core.types import MIN_LATENCY_SAMPLE as _MIN_TTFT_SAMPLE  # noqa: E402

# error_class_bucket values.
ERR_CLASS_NONE:  int = 0
ERR_CLASS_4XX:   int = 1
ERR_CLASS_5XX:   int = 2
ERR_CLASS_CONN:  int = 3
ERR_CLASS_MIXED: int = 4

_RE_HTTP_STATUS = re.compile(r"\b([45])\d\d\b")


def _log2_bucket(n: int, cap: int = 5) -> int:
    """log2-style bucket, the same staircase kv_events.py uses for its counters.

    ``n.bit_length()`` clamped to *cap*: 0→0, 1→1, 2-3→2, 4-7→3, 8-15→4, 16+→5.
    """
    if n is None or n <= 0:
        return 0
    return min(cap, int(n).bit_length())


def _percentile(sorted_vals: list[float], q: float) -> Optional[float]:
    """Nearest-rank percentile (no interpolation) so the bucket is deterministic."""
    if not sorted_vals:
        return None
    idx = int(math.ceil(q * len(sorted_vals))) - 1
    return sorted_vals[max(0, min(len(sorted_vals) - 1, idx))]


def ttft_spread_bucket(requests: list,
                       min_sample: int = _MIN_TTFT_SAMPLE) -> int:
    """log2 bucket of (p99 TTFT / median TTFT) across a trace's requests, 0-5.

    Head-of-line blocking, preemption and chunked-prefill boundaries all show up
    as a *spread* between the fastest and slowest first token in one batch, which
    is invisible to every v1 dimension (they are all scheduler gauges polled at
    1 Hz).  Values come from ``RequestResult.ttft_ms()``
    (``fuzzer/core/types.py``), i.e. exact client-side timings, not the monitor's
    1 Hz poll.

    Returns 0 when fewer than *min_sample* requests produced a first token —
    a p99/median over 2 samples is noise.  0 therefore means "flat or unknown",
    the same convention v1's ``kv_peak_bucket`` uses for missing data; a defined
    ratio always lands in 1..5 (1 = ratio < 2x, 5 = ratio >= 16x).
    """
    ttfts = sorted(
        t for t in (r.ttft_ms() for r in requests)
        if t is not None and t > 0.0
    )
    if len(ttfts) < min_sample:
        return 0
    p50 = _percentile(ttfts, 0.50)
    p99 = _percentile(ttfts, 0.99)
    if not p50 or p50 <= 0 or p99 is None:
        return 0
    return _log2_bucket(int(p99 / p50))


def peak_concurrency(result) -> int:
    """Exact peak in-flight request count, swept from the trace's own timings.

    The monitor polls ``vllm:num_requests_running`` at ``poll_interval_s``, so a trace
    shorter than a few seconds can miss its own peak entirely — the same
    blind spot that made ``finished_count_delta`` useless.  This sweep uses the
    client-side send/complete timestamps already on every ``RequestResult``, so
    it is exact regardless of trace length.  A request that never completed is
    treated as open until ``TraceResult.finished_at``.
    """
    if result is None:
        return 0
    events: list[tuple[float, int]] = []
    trace_end = getattr(result, "finished_at", None)
    for r in getattr(result, "requests", None) or []:
        start = getattr(r, "sent_at", None)
        if start is None:
            continue
        end = r.completed_at or r.cancelled_at or trace_end or start
        if end is None or end < start:
            end = start
        events.append((start, 1))
        events.append((end, -1))
    # Close (-1) before open (+1) on a tie: two requests that touch but do not
    # overlap must not read as concurrency 2.
    events.sort(key=lambda e: (e[0], e[1]))
    cur = peak = 0
    for _t, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def concurrency_peak_bucket(session: "MonitorSession", result) -> int:
    """log2 bucket, 0-5, of max in-flight requests during the trace.

    Takes the max of the polled scheduler gauge (``requests_running``, which can
    see server-side queuing the client cannot) and the exact client-side sweep
    (``peak_concurrency``, which cannot be missed by the 1 Hz poll).
    """
    polled = session.peak_running() if session is not None else None
    swept  = peak_concurrency(result)
    return _log2_bucket(max(polled or 0, swept or 0))


def _classify_error(msg: str) -> int:
    """Map one ``RequestResult.error`` string to a 4xx / 5xx / transport class."""
    low = msg.lower()
    if "client error" in low:
        return ERR_CLASS_4XX
    if "server error" in low:
        return ERR_CLASS_5XX
    m = _RE_HTTP_STATUS.search(msg)
    if m:
        return ERR_CLASS_4XX if m.group(1) == "4" else ERR_CLASS_5XX
    # Anything without an HTTP status never got a reply: it is a transport
    # failure (reset / refused / timeout / protocol error — httpx spells these
    # "Connection reset", "ConnectError", "ReadTimeout", "RemoteProtocolError",
    # "All connection attempts failed", ...).  Unrecognised exception strings
    # land here too — deliberately, since "no status" is exactly what
    # distinguishes them from the 4xx/5xx families.
    #
    # This used to test the message against a `_CONN_MARKERS` tuple first and
    # return ERR_CLASS_CONN either way, so the guard could not change the
    # outcome; the marker list is gone rather than kept as decoration.
    return ERR_CLASS_CONN


def error_class_bucket(requests: list) -> int:
    """0 = no errors, 1 = 4xx only, 2 = 5xx only, 3 = transport only, 4 = mixed.

    Source: ``RequestResult.error`` (``fuzzer/executor.py`` stores ``str(exc)``,
    so httpx's "Client error '400 Bad Request' ..." / "Server error '503 ...'" /
    "[Errno 104] Connection reset by peer" spellings are what we classify).
    Deliberately coarse — the *identity* of an error already has its own novelty
    signal (``Corpus.error_novelty`` over ``error_signature``); this dimension
    only says which failure family the trace provoked.
    """
    classes = {
        _classify_error(r.error)
        for r in requests
        if getattr(r, "error", None)
    }
    if not classes:
        return ERR_CLASS_NONE
    if len(classes) > 1:
        return ERR_CLASS_MIXED
    return classes.pop()


@dataclass
class MetricSnapshot:
    timestamp:            float
    gpu_memory_used_mb:   Optional[float] = None   # from NVML
    gpu_memory_free_mb:   Optional[float] = None
    # 0-100 percent.  vLLM exports vllm:kv_cache_usage_perc as a FRACTION
    # ("1 means 100 percent usage", loggers.py), so the scrape multiplies by 100
    # — every consumer here and in runner.py treats this field as percent
    # (kv_peak_bucket's 20/50/80/95 thresholds, the idle-drift oracle's "5
    # percentage points").  Feeding it the raw fraction pinned kv_peak_bucket
    # at 0 even at 99.8% KV usage.
    kv_cache_usage_pct:   Optional[float] = None   # vllm:kv_cache_usage_perc x100
    requests_running:     Optional[int]   = None   # vllm:num_requests_running
    requests_waiting:     Optional[int]   = None   # vllm:num_requests_waiting
    requests_finished:    Optional[int]   = None   # sum(vllm:request_success_total)
    preemptions_total:    Optional[int]   = None   # vllm:num_preemptions_total
    # vllm:kv_offload_cpu_cache_usage_perc x100 (0.27.1 + offload connector only).
    # v1 has no vllm:cpu_cache_usage_perc; this is the nearest live relative.
    cpu_cache_usage_pct:  Optional[float] = None
    # Always None against vLLM v1: there is no swapping and no
    # vllm:num_requests_swapped gauge.  Field kept so persisted sessions and the
    # sglang adapters (which do have a swap notion) still round-trip.
    requests_swapped:          Optional[int]   = None
    # Cumulative hit rate = prefix_cache_hits_total / prefix_cache_queries_total.
    # Server-lifetime, therefore history-dependent — prefer the trace-local
    # interval rate from the two raw counters below.
    gpu_prefix_cache_hit_rate: Optional[float] = None
    prefix_cache_hits_total:   Optional[float] = None   # vllm:prefix_cache_hits_total
    prefix_cache_queries_total: Optional[float] = None  # vllm:prefix_cache_queries_total
    ttft_histogram_sum:        Optional[float] = None   # vllm:time_to_first_token_seconds_sum
    ttft_histogram_count:      Optional[int]   = None   # vllm:time_to_first_token_seconds_count


@dataclass(frozen=True)
class SchedulerFingerprint:
    """Multi-dimensional summary of scheduler behaviour during one trace.

    frozen=True makes it hashable so it can live in a set for novelty tracking.

    **v1** (``version == 1``) is the historical 8-dimension fingerprint: up to
    5 × 4 × 4 × 4 × 4 × 2 × 2 × 4 ≈ 20,480 distinct values, of which the offline
    replay (an internal analysis note) only ever observed
    **8**, with 6 of the 8 dimensions constant across 10,185 corpus entries.

    **v2** (``version == 2``) keeps every v1 dimension and adds three
    *workload*-derived ones — ``ttft_spread_bucket``, ``concurrency_peak_bucket``
    and ``error_class_bucket`` — which are computed from the ``TraceResult``
    rather than from the 1 Hz telemetry poll, and which do move under the
    existing burst/timing mutators.  The v1 dimensions that were measured
    constant are *not* deleted: `Corpus` down-weights invariant dimensions
    towards ``_DIM_WEIGHT_FLOOR`` and re-probes periodically, so a dimension that
    starts moving under a new server config (``preemption_count_bucket`` under
    ``--max-num-seqs 8``, say) recovers its weight instead of being permanently
    unobservable.  See `Corpus._recompute_dim_weights`.

    **v3** (``version == 3``) has the *same eleven dimensions* as v2 — no field
    was added, removed or reordered.  What changed is that four of them are now
    fed by metric names vLLM actually exports.  Until this fix the monitor
    scraped ``vllm:gpu_cache_usage_perc``, ``vllm:num_requests_finished``,
    ``vllm:gpu_prefix_cache_hit_rate`` and ``vllm:num_requests_swapped``, none
    of which exist in vLLM v1 (verified in both SIF images), so every scrape
    silently produced ``None`` and ``kv_peak_bucket`` / ``finished_count_bucket``
    / ``prefix_cache_hit_bucket`` / ``had_swap`` were **constant 0/0/0/False in
    every fingerprint we have ever recorded** — including the archived v1 and v2
    ones (all 174 fingerprint lines of the tight15 run read
    ``kv=0 hit=0 fc=0 swap=n`` while the server log showed 99.8% KV usage).

    *Comparability with archived runs.*  Because the repair lives in the
    telemetry rather than in the bucketing, ``session.fingerprint()`` is shared
    by all three versions and there is no way to keep emitting the old values
    short of keeping the bug.  Treat this as a schema break: **an archived
    ``fpv=1``/``fpv=2`` fingerprint is not comparable to anything produced after
    this fix**, because its zeros mean "never measured", not "measured as low".
    New runs must be labelled ``v=3``; when re-analysing an old corpus, compare
    only the seven dimensions that were genuinely live
    (``had_preemption``, ``queue_depth_bucket``, ``preemption_count_bucket``,
    ``memory_delta_bucket`` and the three v2 workload dimensions), or replay the
    corpus rather than trusting the stored tuple.  ``FP_VERSION_LATEST`` is
    ``3``; ``--fingerprint-version 1|2`` still selects the v1/v2 *dimension set*
    (that is what the A/B arms vary) and now carries repaired values too, so an
    A/B against an older ``fpv=2`` job is invalid — rerun both arms.

    ``had_swap`` is deliberately **retained and held at a documented constant
    False** on vLLM v1 rather than dropped.  Dropping it would change the
    dataclass arity and the ``str()`` rendering that
    the offline novelty replay parses, and would make every persisted
    7/8/11-element tuple in the corpus unpaddable
    (``Corpus._pad_fingerprint_tuple``).  It is *not* repurposed to mean
    "KV offloaded to CPU" — ``cpu_cache_usage_pct`` records
    ``vllm:kv_offload_cpu_cache_usage_perc`` for the offload campaign, and
    folding that into ``had_swap`` would silently redefine the dimension.
    ``Corpus._recompute_dim_weights`` already drives an invariant dimension's
    weight to the floor, so a permanently-False dimension costs selection
    pressure, not correctness.

    Field order is append-only: ``version`` and the three v2 dimensions all
    carry defaults, so a v1 fingerprint constructed positionally with the
    original 8 arguments is byte-identical to the pre-v2 one, and persisted
    7/8-element tuples pad forward (``Corpus._pad_fingerprint_tuple``).
    """
    had_preemption:          bool
    # vLLM v1 has no swapping (preemption is recompute-only) and no
    # vllm:num_requests_swapped gauge: constant False there.  Still live for the
    # sglang adapters.
    had_swap:                bool
    queue_depth_bucket:      int    # 0=none, 1=1–4, 2=5–19, 3=20+
    kv_peak_bucket:          int    # 0=<20%, 1=20–50%, 2=50–80%, 3=80–95%, 4=95%+
    preemption_count_bucket: int    # 0=none, 1=1–2, 2=3–10, 3=11+
    memory_delta_bucket:     int    # 0=<10 MB, 1=10–50 MB, 2=50–200 MB, 3=200 MB+
    finished_count_bucket:   int    # 0=0, 1=1–5, 2=6–20, 3=21+
    prefix_cache_hit_bucket: int = 0  # 0=unknown, 1=high(≥80%), 2=medium(50–79%), 3=low(<50%)
    # --- v2 ---------------------------------------------------------------
    version:                 int = FP_VERSION_V1
    ttft_spread_bucket:      int = 0  # log2(p99 TTFT / median TTFT), 0=unknown/flat … 5=>=16x
    concurrency_peak_bucket: int = 0  # log2(max in-flight requests), 0=none … 5=16+
    error_class_bucket:      int = 0  # 0=none 1=4xx 2=5xx 3=transport 4=mixed

    def __str__(self) -> str:
        base = (
            f"pre={'Y' if self.had_preemption else 'n'}"
            f" swap={'Y' if self.had_swap else 'n'}"
            f" q={self.queue_depth_bucket}"
            f" kv={self.kv_peak_bucket}"
            f" pc={self.preemption_count_bucket}"
            f" md={self.memory_delta_bucket}"
            f" fc={self.finished_count_bucket}"
            f" hit={self.prefix_cache_hit_bucket}"
        )
        if self.version < FP_VERSION_V2:
            # Byte-identical to the pre-v2 rendering, so the `[novel ...]` log
            # lines that the offline novelty replay parses keep their format.
            return base
        return (f"{base}"
                f" ts={self.ttft_spread_bucket}"
                f" cc={self.concurrency_peak_bucket}"
                f" ec={self.error_class_bucket}"
                f" v={self.version}")


@dataclass
class MonitorSession:
    """All snapshots collected during one scenario run."""
    snapshots: list[MetricSnapshot] = field(default_factory=list)

    def baseline_memory_mb(self) -> Optional[float]:
        """First snapshot — GPU memory before the scenario started."""
        return self.snapshots[0].gpu_memory_used_mb if self.snapshots else None

    def peak_memory_mb(self) -> Optional[float]:
        vals = [s.gpu_memory_used_mb for s in self.snapshots
                if s.gpu_memory_used_mb is not None]
        return max(vals) if vals else None

    def final_memory_mb(self) -> Optional[float]:
        for s in reversed(self.snapshots):
            if s.gpu_memory_used_mb is not None:
                return s.gpu_memory_used_mb
        return None

    def memory_leaked_mb(self, tolerance_mb: float = 50.0) -> Optional[float]:
        """
        Positive value = memory grew beyond tolerance after scenario drained.
        None if baseline was never recorded.
        """
        base  = self.baseline_memory_mb()
        final = self.final_memory_mb()
        if base is None or final is None:
            return None
        delta = final - base
        return delta if delta > tolerance_mb else 0.0

    # ------------------------------------------------------------------
    # Drain state.
    #
    # "Did the engine actually finish?" is a *precondition* for every
    # end-of-trace metric, not a nicety.  ``Runner`` calls
    # ``monitor.start() / executor.run() / monitor.stop()`` back to back with no
    # drain barrier, and ``AsyncExecutor.run`` client-cancels still-running
    # tasks once ``trace_timeout_s`` (90 s) elapses and returns while the server
    # keeps generating.  A terminal metric read at that instant describes a busy
    # engine, not a settled one, so any oracle that subtracts it from a
    # pre-trace value is measuring "was the engine still working when the client
    # gave up".  The predicate below is the same one
    # ``Runner._update_idle_windows`` uses for the cross-iteration detectors
    # (which agreed with vLLM's own logs and never fired).
    #
    # Why gate the *oracle* on the recorded drain state rather than make
    # ``Monitor.stop()`` poll-until-idle before its final scrape:
    #
    #   * A blocking drain wait perturbs the thing being measured.  It runs on
    #     every iteration, and on a timeout-heavy workload it becomes the
    #     common path — so campaign throughput, and with it the concurrency and
    #     preemption behaviour the fuzzer exists to explore, would shift.
    #   * It would silently move fingerprints.  Extra snapshots taken while
    #     idling change ``finished_count_delta``, ``preemption_delta`` and
    #     ``memory_delta_bucket``, breaking comparability with every archived
    #     corpus — the same schema break ``SchedulerFingerprint`` v3 documents.
    #   * It would have to be implemented, and stay correct, in each ``Monitor``
    #     subclass rather than once.
    #   * Recording the state is retroactive: the drain verdict can be
    #     recomputed from any stored ``MonitorSession``, which is what makes
    #     re-adjudicating historical findings possible at all.  A barrier only
    #     helps runs that happen after it lands.
    #
    # And the gate belongs where the claim is made: the oracle is the thing
    # that says "after scenario drained", so the oracle is the thing that has
    # to check it.
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_drained(s: MetricSnapshot) -> Optional[bool]:
        """``True``/``False`` drained, or ``None`` when *s* carries no queue data.

        ``None`` is not "drained": a snapshot with neither ``requests_running``
        nor ``requests_waiting`` is silent about the engine, and callers that
        gate on drain must treat that as *unmeasured* rather than idle.
        """
        if s.requests_running is None and s.requests_waiting is None:
            return None
        return (s.requests_running or 0) == 0 and (s.requests_waiting or 0) == 0

    def _last_snapshot_with_kv(self) -> "Optional[MetricSnapshot]":
        for s in reversed(self.snapshots):
            if s.kv_cache_usage_pct is not None:
                return s
        return None

    def final_queue_state(self) -> Optional[tuple[int, int]]:
        """``(running, waiting)`` from the last snapshot that reported either."""
        for s in reversed(self.snapshots):
            if s.requests_running is not None or s.requests_waiting is not None:
                return (s.requests_running or 0, s.requests_waiting or 0)
        return None

    def engine_drained(self) -> Optional[bool]:
        """Was the engine idle at the end of the session?

        ``None`` when the session recorded no queue telemetry at all — the
        honest answer being "we never looked", which is distinct from "no".
        """
        q = self.final_queue_state()
        return None if q is None else q == (0, 0)

    def kv_cache_baseline_pct(self) -> Optional[float]:
        """KV cache usage in the earliest *idle* snapshot that carries it.

        This used to be a bare ``self.snapshots[0].kv_cache_usage_pct``, which
        had two problems.  It had no ``None``-guard (its ``final`` counterpart
        does), so a session whose first sample missed the scrape reported "no kv
        cache data" even when every later sample had it.  And it was not
        anchored to an idle engine, so on a server that starts a trace with work
        still in flight the "baseline" was mid-trace occupancy.

        The subtraction is deliberately *kept* rather than dropped.  On vLLM v1
        this value is ~0 by construction — the poll loop's first sample lands
        before the trace's first request, and every one of the 164 idle samples
        in the three campaign arms read exactly 0.0% — so subtracting it is
        currently decorative.  But the quantity the oracle claims to report is
        *what this trace retained*, and dropping the baseline would turn the
        check into a test of absolute occupancy, which fires on KV inherited
        from a previous iteration.  That trades one false-positive class for
        another.  Anchoring the baseline to a drain-verified idle sample makes
        the subtraction mean what the detail string says it means.

        Falls back to the first kv-bearing snapshot when no idle one exists,
        which can only *understate* the delta (an occupied baseline is larger),
        so it cannot manufacture a finding.
        """
        first_kv: Optional[float] = None
        for s in self.snapshots:
            if s.kv_cache_usage_pct is None:
                continue
            if first_kv is None:
                first_kv = s.kv_cache_usage_pct
            if self._snapshot_drained(s) is not False:
                return s.kv_cache_usage_pct
        return first_kv

    def kv_cache_final_pct(self) -> Optional[float]:
        """KV cache usage at the end of the scenario."""
        s = self._last_snapshot_with_kv()
        return s.kv_cache_usage_pct if s is not None else None

    def kv_cache_final_drained(self) -> Optional[bool]:
        """Was the engine idle in the snapshot ``kv_cache_final_pct`` came from?

        Deliberately tied to *that* snapshot rather than to the last snapshot of
        any kind: the number an oracle divides by has to be the number whose
        provenance was checked.  ``None`` when that snapshot has no queue data.
        """
        s = self._last_snapshot_with_kv()
        return None if s is None else self._snapshot_drained(s)

    def requests_ever_stuck(self, min_stuck_seconds: float = 3.0) -> bool:
        """True if the waiting queue was non-empty continuously for ≥ min_stuck_seconds
        after the running queue drained to zero.

        A 2-snapshot streak (~2s) is insufficient: normal CUDA graph swapping causes
        100–500ms gaps where running=0, waiting>0 at every large batch boundary.
        Requiring a 3-second continuous duration rules out these transient gaps.

        The timer resets whenever running > 0 (scheduler is active again).
        """
        saw_running  = False
        stuck_since: Optional[float] = None
        for s in self.snapshots:
            if s.requests_running is not None and s.requests_running > 0:
                saw_running  = True
                stuck_since  = None
            elif saw_running and s.requests_running == 0:
                if s.requests_waiting is not None and s.requests_waiting > 0:
                    if stuck_since is None:
                        stuck_since = s.timestamp
                    elif s.timestamp - stuck_since >= min_stuck_seconds:
                        return True
                else:
                    stuck_since = None
        return False

    def preemption_delta(self) -> Optional[int]:
        """Increase in preemptions_total across this session.
        Returns None if the metric was never recorded."""
        vals = [s.preemptions_total for s in self.snapshots
                if s.preemptions_total is not None]
        if not vals:
            return None
        return vals[-1] - vals[0]

    def had_cpu_swap(self) -> bool:
        """True if cpu_cache_usage_pct was ever > 0 (blocks spilled to CPU).

        Kept as fallback for sessions collected before requests_swapped was added.
        """
        return any(
            s.cpu_cache_usage_pct is not None and s.cpu_cache_usage_pct > 0.0
            for s in self.snapshots
        )

    def peak_swapped(self) -> Optional[int]:
        """Maximum num_requests_swapped observed during this session."""
        vals = [s.requests_swapped for s in self.snapshots
                if s.requests_swapped is not None]
        return max(vals) if vals else None

    def had_swap(self) -> bool:
        """True if any requests were ever swapped to CPU (direct gauge, not proxy)."""
        return any(
            s.requests_swapped is not None and s.requests_swapped > 0
            for s in self.snapshots
        )

    def min_prefix_cache_hit_rate(self) -> Optional[float]:
        """Minimum prefix cache hit rate over this trace — drops signal eviction.

        vLLM v1 exports no hit-*rate* gauge, only the two cumulative counters
        ``vllm:prefix_cache_{hits,queries}_total``.  Their plain ratio is the
        rate over the *server's whole lifetime*, which drifts with campaign
        position and would make two identical traces fingerprint differently
        depending on when they ran.  So the rate is computed **per adjacent
        snapshot pair** — Δhits / Δqueries — which is trace-local and
        reproducible, and the minimum over those intervals is returned.

        Pairs with Δqueries == 0 (no prefill happened in that interval) carry no
        information and are skipped, rather than counted as a 0% hit rate.

        Falls back to the per-snapshot ``gpu_prefix_cache_hit_rate`` field when
        the raw counters are absent — sessions persisted before the counters
        existed, and the sglang adapters, which report a rate directly.
        """
        pairs = [s for s in self.snapshots
                 if s.prefix_cache_hits_total is not None
                 and s.prefix_cache_queries_total is not None]
        rates: list[float] = []
        for prev, cur in zip(pairs, pairs[1:]):
            d_q = cur.prefix_cache_queries_total - prev.prefix_cache_queries_total
            if d_q <= 0:
                continue
            d_h = cur.prefix_cache_hits_total - prev.prefix_cache_hits_total
            rates.append(max(0.0, min(1.0, d_h / d_q)))
        if rates:
            return min(rates)
        vals = [s.gpu_prefix_cache_hit_rate for s in self.snapshots
                if s.gpu_prefix_cache_hit_rate is not None]
        return min(vals) if vals else None

    def peak_running(self) -> Optional[int]:
        """Maximum requests_running observed. None if never recorded.

        Polled at ``poll_interval_s`` (``DEFAULT_POLL_INTERVAL_S``), so a short trace can
        under-observe its own peak — ``concurrency_peak_bucket`` therefore takes
        the max of this and the exact client-side sweep in ``peak_concurrency``.
        """
        vals = [s.requests_running for s in self.snapshots
                if s.requests_running is not None]
        return max(vals) if vals else None

    def peak_waiting_queue(self) -> Optional[int]:
        """Maximum requests_waiting observed. None if never recorded."""
        vals = [s.requests_waiting for s in self.snapshots
                if s.requests_waiting is not None]
        return max(vals) if vals else None

    def peak_kv_cache_pct(self) -> Optional[float]:
        """Maximum KV cache usage percentage observed."""
        vals = [s.kv_cache_usage_pct for s in self.snapshots
                if s.kv_cache_usage_pct is not None]
        return max(vals) if vals else None

    def finished_count_delta(self) -> Optional[int]:
        """Increase in requests_finished across this session."""
        vals = [s.requests_finished for s in self.snapshots
                if s.requests_finished is not None]
        if len(vals) < 2:
            return None
        return vals[-1] - vals[0]

    def fingerprint(self) -> SchedulerFingerprint:
        """Summarise scheduler behaviour into a multi-dimensional hashable fingerprint."""
        delta  = self.preemption_delta()
        peak_q = self.peak_waiting_queue()

        # queue_depth_bucket
        if peak_q is None or peak_q == 0: q_bucket = 0
        elif peak_q < 5:                  q_bucket = 1
        elif peak_q < 20:                 q_bucket = 2
        else:                             q_bucket = 3

        # kv_peak_bucket
        kv_peak = self.peak_kv_cache_pct()
        if kv_peak is None or kv_peak < 20:  kv_bucket = 0
        elif kv_peak < 50:                   kv_bucket = 1
        elif kv_peak < 80:                   kv_bucket = 2
        elif kv_peak < 95:                   kv_bucket = 3
        else:                                kv_bucket = 4

        # preemption_count_bucket
        pre_delta = delta or 0
        if pre_delta == 0:    pre_bucket = 0
        elif pre_delta <= 2:  pre_bucket = 1
        elif pre_delta <= 10: pre_bucket = 2
        else:                 pre_bucket = 3

        # memory_delta_bucket
        baseline_mem = self.baseline_memory_mb()
        final_mem    = self.final_memory_mb()
        mem_delta    = (final_mem - baseline_mem) if (
            baseline_mem is not None and final_mem is not None) else 0.0
        if mem_delta < 10:    mem_bucket = 0
        elif mem_delta < 50:  mem_bucket = 1
        elif mem_delta < 200: mem_bucket = 2
        else:                 mem_bucket = 3

        # finished_count_bucket
        fin_delta = self.finished_count_delta()
        if fin_delta is None or fin_delta == 0: fin_bucket = 0
        elif fin_delta <= 5:                    fin_bucket = 1
        elif fin_delta <= 20:                   fin_bucket = 2
        else:                                   fin_bucket = 3

        # had_swap: requests_swapped only.  Constant False against vLLM v1 by
        # design — see SchedulerFingerprint's docstring for why the dimension is
        # held rather than dropped or repurposed onto the kv_offload gauge.
        had_swap_val = self.had_swap()

        # prefix_cache_hit_bucket
        min_hit = self.min_prefix_cache_hit_rate()
        if min_hit is None:       hit_bucket = 0
        elif min_hit >= 0.80:     hit_bucket = 1
        elif min_hit >= 0.50:     hit_bucket = 2
        else:                     hit_bucket = 3

        return SchedulerFingerprint(
            had_preemption=delta is not None and delta > 0,
            had_swap=had_swap_val,
            queue_depth_bucket=q_bucket,
            kv_peak_bucket=kv_bucket,
            preemption_count_bucket=pre_bucket,
            memory_delta_bucket=mem_bucket,
            finished_count_bucket=fin_bucket,
            prefix_cache_hit_bucket=hit_bucket,
        )


def build_fingerprint(session: MonitorSession,
                      result=None,
                      version: int = FP_VERSION_LATEST) -> SchedulerFingerprint:
    """Fingerprint one trace at the requested schema *version*.

    ``version <= 1`` returns exactly ``session.fingerprint()`` — the v1 path is
    untouched, including the object it returns.

    ``version >= 2`` keeps all eight v1 dimensions and fills the three
    workload-derived ones from *result* (a ``TraceResult``).  ``session`` alone
    cannot produce them: TTFT spread and the error class live on the individual
    ``RequestResult``s, which the 1 Hz telemetry poll never sees.  When *result*
    is None the three new dimensions fall back to 0 (= unknown), which is the
    same convention the v1 dimensions use for missing telemetry.

    ``version >= 3`` has the same dimension set as v2 and differs only in the
    stamped ``version``: v3 asserts that the KV / finished / prefix-hit / swap
    dimensions came from metric names vLLM actually exports.  The *values* are
    repaired for v1 and v2 requests too — the fix is in the scrape, not the
    bucketing — so a post-fix ``fpv=2`` tuple is not comparable to an archived
    one.  See ``SchedulerFingerprint``.
    """
    base = session.fingerprint()
    if version < FP_VERSION_V2:
        return base
    requests = list(getattr(result, "requests", None) or []) if result is not None else []
    return dataclasses.replace(
        base,
        version=min(int(version), FP_VERSION_LATEST),
        ttft_spread_bucket=ttft_spread_bucket(requests),
        concurrency_peak_bucket=concurrency_peak_bucket(session, result),
        error_class_bucket=error_class_bucket(requests),
    )


class Monitor(ABC):
    @abstractmethod
    async def start(self) -> None:
        """Begin background polling."""
        ...

    @abstractmethod
    async def stop(self) -> MonitorSession:
        """Stop polling and return the collected session."""
        ...

    @abstractmethod
    async def snapshot(self) -> MetricSnapshot:
        """Take a single immediate snapshot."""
        ...

    @abstractmethod
    async def peek(self) -> MetricSnapshot:
        """Single out-of-band snapshot; does not append to the active session."""
        ...


# ---------------------------------------------------------------------------
# Real monitor (requires pynvml + running vLLM)
# ---------------------------------------------------------------------------

class NvmlMonitor(Monitor):
    """
    Polls GPU memory via NVML and vLLM /metrics endpoint.
    Requires: pip install pynvml httpx
    """

    def __init__(self,
                 metrics_url: str  = "http://localhost:8000/metrics",
                 poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
                 gpu_index: int = 0,
                 local_gpu: bool = True):
        self.metrics_url     = metrics_url
        self.poll_interval_s = poll_interval_s
        self.gpu_index       = gpu_index
        self.local_gpu       = local_gpu
        self._session        = MonitorSession()
        self._task           = None
        self._nvml_handle    = None
        self._client         = None    # reused httpx.AsyncClient (keep-alive)

    async def start(self) -> None:
        import asyncio
        self._session = MonitorSession()
        self._task    = asyncio.create_task(self._poll_loop())

    async def stop(self) -> MonitorSession:
        """Cancel the poll loop, but take one final snapshot first.

        Without this the session's last sample is whatever the 1 Hz loop
        happened to land on, so a trace shorter than the poll interval —
        the determinism probe runs ~0.9 s — collects a *single* sample and
        ``preemption_delta()`` / ``finished_count_delta()`` end up diffing that
        sample against itself, i.e. structurally 0.  Bracketing the trace with
        an explicit end-of-trace snapshot makes both deltas well-defined at any
        poll rate, and costs one extra scrape per trace rather than 5x the
        scrape rate for the whole campaign.
        """
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except __import__("asyncio").CancelledError:
                pass
            self._task = None
        try:
            await self.snapshot()
        except Exception:
            pass
        return self._session

    async def aclose(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except __import__("asyncio").CancelledError:
                pass
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _read_gpu_memory(self) -> tuple[Optional[float], Optional[float]]:
        """(used_mb, free_mb) via NVML; the device handle is cached across polls."""
        if not self.local_gpu:
            return None, None
        try:
            import pynvml
            if self._nvml_handle is None:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return mem.used / 1024**2, mem.free / 1024**2
        except Exception:
            self._nvml_handle = None
            return None, None

    async def _scrape(self) -> dict[str, list[float]]:
        """GET /metrics and parse it. Returns {} on any failure.

        The client is reused so a poll is one request on a kept-alive
        connection rather than a fresh TCP handshake per sample.
        """
        try:
            import httpx
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=2.0)
            resp = await self._client.get(self.metrics_url)
            return parse_prometheus_text(resp.text)
        except Exception:
            self._client = None
            return {}

    @staticmethod
    def _fill_from_metrics(snap: MetricSnapshot,
                           parsed: dict[str, list[float]]) -> None:
        """Populate *snap* from a parsed /metrics body.

        Aggregation across label sets: gauges that describe the whole engine are
        summed over the ``engine`` label (one series per DP rank), except KV
        usage where the peak is the meaningful number; counters are summed over
        every label combination — ``vllm:request_success_total`` in particular
        has one series per ``finished_reason``, so the finished-request count is
        the sum, not any single series.
        """
        kv = _max(parsed, _M_KV_USAGE)
        if kv is not None:
            snap.kv_cache_usage_pct = kv * 100.0     # gauge is a fraction 0..1
        cpu = _max(parsed, _M_OFFLOAD_CPU_USAGE)
        if cpu is not None:
            snap.cpu_cache_usage_pct = cpu * 100.0
        for field_name, name in ((("requests_running"), _M_RUNNING),
                                 (("requests_waiting"), _M_WAITING),
                                 (("requests_finished"), _M_SUCCESS),
                                 (("preemptions_total"), _M_PREEMPTIONS)):
            val = _sum(parsed, name)
            if val is not None:
                setattr(snap, field_name, int(val))
        hits    = _sum(parsed, _M_PREFIX_HITS)
        queries = _sum(parsed, _M_PREFIX_QUERIES)
        if hits is not None and queries is not None:
            snap.prefix_cache_hits_total    = hits
            snap.prefix_cache_queries_total = queries
            if queries > 0:
                snap.gpu_prefix_cache_hit_rate = hits / queries
        ttft_sum   = _sum(parsed, _M_TTFT_SUM)
        ttft_count = _sum(parsed, _M_TTFT_COUNT)
        if ttft_sum is not None:
            snap.ttft_histogram_sum = ttft_sum
        if ttft_count is not None:
            snap.ttft_histogram_count = int(ttft_count)

    async def snapshot(self) -> MetricSnapshot:
        snap = MetricSnapshot(timestamp=time.time())
        snap.gpu_memory_used_mb, snap.gpu_memory_free_mb = self._read_gpu_memory()
        self._fill_from_metrics(snap, await self._scrape())
        self._session.snapshots.append(snap)
        return snap

    async def peek(self) -> MetricSnapshot:
        """Single out-of-band snapshot; not appended to the active session."""
        snap = MetricSnapshot(timestamp=time.time())
        snap.gpu_memory_used_mb, snap.gpu_memory_free_mb = self._read_gpu_memory()
        self._fill_from_metrics(snap, await self._scrape())
        return snap

    async def _poll_loop(self) -> None:
        import asyncio
        while True:
            await self.snapshot()
            await asyncio.sleep(self.poll_interval_s)


# ---------------------------------------------------------------------------
# Mock monitor (for testing without a GPU)
# ---------------------------------------------------------------------------

class MockMonitor(Monitor):
    """
    Returns synthetic metrics that simulate a healthy server.
    Pass inject_leak=True to simulate a memory leak for oracle testing.
    """

    def __init__(self, inject_leak: bool = False):
        self.inject_leak = inject_leak
        self._session    = MonitorSession()
        self._call_count = 0

    async def start(self) -> None:
        self._session    = MonitorSession()
        self._call_count = 0

    async def stop(self) -> MonitorSession:
        return self._session

    async def snapshot(self) -> MetricSnapshot:
        self._call_count += 1
        leak = 10.0 * self._call_count if self.inject_leak else 0.0
        snap = MetricSnapshot(
            timestamp=time.time(),
            gpu_memory_used_mb=4000.0 + leak,
            gpu_memory_free_mb=20000.0 - leak,
            # 0-100 percent, matching NvmlMonitor (which scales vLLM's 0..1
            # gauge by 100).  The mock used to emit 0.1..0.9 here, i.e. the raw
            # fraction on a field every consumer reads as a percentage — the
            # same scale confusion that pinned kv_peak_bucket at 0 in the real
            # monitor, so the mock could not have caught it.
            kv_cache_usage_pct=min(10.0 * self._call_count, 90.0),
            requests_running=max(0, 5 - self._call_count),
            requests_waiting=0,
            requests_finished=self._call_count * 2,
        )
        self._session.snapshots.append(snap)
        return snap

    async def peek(self) -> MetricSnapshot:
        return MetricSnapshot(
            timestamp=time.time(),
            gpu_memory_used_mb=4000.0,
            kv_cache_usage_pct=0.0,
            requests_running=0,
            requests_waiting=0,
        )


class EmptyMonitor(Monitor):
    """Explicitly unavailable telemetry; never invent healthy measurements."""
    async def start(self):
        pass

    async def stop(self):
        return MonitorSession()

    async def snapshot(self):
        return MetricSnapshot(timestamp=time.time())

    async def peek(self):
        return await self.snapshot()
