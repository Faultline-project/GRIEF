"""
Core Runner: the main fuzzing loop (mode-agnostic).

The Runner class accepts an optional TargetAdapter for mode-specific behavior.
Without an adapter, it falls through to the vLLM-specific inline logic for
backward compatibility.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, deque
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Prompt length used for baseline TTFT measurement — must match what's passed
# to the oracle so it can normalize per-request thresholds by prompt length.
_BASELINE_PROMPT_LEN = 256

# Cross-iteration idle-window leak detection constants.
_IDLE_WINDOW_SIZE  = 50     # readings before baseline is established
_MEM_LEAK_DRIFT_PCT = 5.0   # % KV cache drift above idle baseline to fire MEMORY_LEAK
_KV_LEAK_DRIFT_PCT = 10.0   # % above idle baseline to fire KV_BLOCK_LEAK
_KV_FLUSH_EVERY    = 50     # flush prefix cache every N iterations

from ..corpus    import Corpus
from ..executor  import error_signature, Executor, PythonExecutor, TraceResult
from ..minimizer import minimize_with_oracle
from ..monitor   import (Monitor, MonitorSession, NvmlMonitor,
                         SchedulerFingerprint, build_fingerprint)
from ..mutator   import mutate
from ..oracles   import BugClass, Oracle, OracleResult
from ..trace     import Trace, Event, ActionType
from .adapter    import TargetAdapter
from .finding_sig import (SIG_VERSION, log2_bucket, mechanism_signature,
                          normalize_server_flags)


def _score_multiplier_for_adapter(adapter):
    """Return the Strategy.score_multiplier callable when the adapter is
    new-shape (Capabilities + Strategy), otherwise None for legacy adapters.

    The same callable reaches both the main and splice corpus.select() calls,
    so known-bug-shaped traces cannot still dominate as splice parents.
    Legacy adapters → None → corpus selection is bit-exact preserved.
    """
    if adapter is None:
        return None
    strategy = getattr(adapter, "strategy", None)
    if strategy is None:
        return None
    return getattr(strategy, "score_multiplier", None)


@dataclass
class RunnerConfig:
    events_file:             str   = ""
    base_url:                str   = "http://localhost:8000"
    model_name:              str   = ""      # sent in flush request; auto-discovered if empty
    seeds_dir:               str   = "./seeds"
    adapter_seeds_dir:       Optional[str] = None
    corpus_dir:              str   = "./corpus"
    findings_dir:            str   = "./findings"
    max_iterations:          int   = 10_000
    mutations_per_seed:      int   = 2
    seed:                    int   = 42
    minimize_findings:       bool  = True
    poll_interval_s:         float = 1.0
    trace_timeout_s:         float = 60.0
    settle_s:                float = 0.5
    probe_determinism_every:         int   = 5
    probe_cache_hit_every:           int   = 100
    #: Three-arm relational (prefix-cache equivalence) probe cadence.
    #: 0 = off (default).  When > 0, every Nth iteration re-runs the trace with
    #: TWO differently-salted twins beside every greedy SEND — one to route
    #: around the prefix cache, one as the negative control that says whether
    #: the salt itself is the confound; see ``Runner._prefix_equivalence_probe``.
    #: Deliberately independent of ``probe_determinism_every`` — both probes
    #: run, so the relational probe's survivor rate can be compared against the
    #: determinism probe's.
    #:
    #: COST.  A firing re-runs the whole trace and sends 3 requests per greedy
    #: SEND where the trace sent 1.  Measured against the 134 distinct traces of
    #: the novelty_ab campaign (mean 5.94 SENDs/trace, 5.41 of them greedy), at
    #: this knob's campaign value of 5:
    #:     baseline            29.7 SENDs per 5 iterations
    #:     two-arm probe      +11.4  (+38%)
    #:     three-arm probe    +16.8  (+56%)
    #:     the control arm      +5.4  (+18%)
    #: So the negative control costs ~18% more request volume.  It replaces a
    #: planned separate campaign slot running with prefix caching globally off,
    #: which would have cost a whole slot instead.
    prefix_equiv_probe_every:        int   = 0
    campaign_id:             str   = ""
    server_flags:            str   = ""
    kv_events_endpoint:      str   = ""
    kv_events_window_s:      float = 30.0

    # Recheck the reduced trace without replacing the original evidence.
    reverify_minimized:      bool  = False

    # Mechanism-signature ledger (fuzzer/core/finding_sig.py): how many
    # exemplar finding JSONs one signature writes before later occurrences are
    # counted only.  The ledger itself counts every occurrence regardless.
    exemplars_per_signature: int   = 5

    # Campaign-mode metadata (record-keeping only — no behavior changes).
    # Populated by run_one_config.sh; empty/-1 sentinel values mean "not set".
    campaign_name:           str   = ""
    config_id:               str   = ""
    config_file:             str   = ""
    mode:                    str   = ""
    gpu_type:                str   = ""
    slot_id:                 str   = ""
    rotation_index:          int   = -1

    # Verbose per-iteration logging — emits one line per iteration with
    # mutation operators applied, trace complexity, and KV forensics.
    # Intended for offline campaign analysis (e.g. paper figures); not for
    # routine fuzzing.
    verbose:                 bool  = False

    # Soft-suspicion batch reproduction (behavioral divergence)
    soft_repro_k:              int   = 3      # runs per candidate
    soft_repro_batch_size:     int   = 20     # enqueue trigger
    soft_repro_max_candidates: int   = 15     # after family dedup
    soft_repro_budget_s:       float = 300.0  # max wall time per batch (5 min)
    soft_repro_interval_min:   float = 30.0   # time-based trigger
    soft_repro_cooldown_s:     float = 1800.0 # cooldown after rejection (30 min)

    # Structural KV anomaly batch reproduction
    structural_repro_k:              int   = 3      # runs per candidate
    structural_repro_batch_size:     int   = 10     # enqueue trigger
    structural_repro_budget_s:       float = 180.0  # max wall time per batch (3 min)
    structural_repro_interval_min:   float = 20.0   # time-based trigger
    structural_repro_cooldown_s:     float = 900.0  # cooldown after rejection (15 min)

    # Isolation oracle calibration (off by default)
    # When True, runs a calibration phase before the main loop that empirically
    # mines prompt families where the loaded adapters produce measurably different
    # logprob distributions in solo runs. The top-K discriminating families are
    # seeded into the corpus so the isolation oracle has reliable comparison targets.
    path_calibration:                bool  = False
    calibration_n_candidates:        int   = 64     # candidate prompt families to probe
    calibration_top_k:               int   = 16     # families kept as corpus seeds
    calibration_n_repeats:           int   = 2      # runs per candidate (stability scoring)

    # Confirmation ladder for output-divergence candidates (off by default;
    # see fuzzer/confirm/ladder.py).  When on, the determinism and
    # prefix-equivalence probes ENQUEUE their divergences instead of filing
    # them; _drain_confirmations replays each one on the stock server (solo
    # twice) and on a second, batch-invariant server (batched twice, solo
    # once) and files only the verdicts in ladder.FILED_VERDICTS.
    confirm_ladder:                  bool  = False
    confirm_bi_url:                  str   = ""     # the VLLM_BATCH_INVARIANT=1 server
    confirm_bi_log:                  str   = ""     # its log file: BI engagement proof
    confirm_bi_backend:              str   = ""     # backend override for the allowlist
    confirm_gpu_arch:                str   = ""     # GPU arch override for the allowlist
    confirm_every:                   int   = 5      # drain cadence, in iterations
    confirm_max_per_drain:           int   = 2      # candidates per drain
    confirm_budget_s:                float = 120.0  # wall-clock budget per drain
    confirm_queue_max:               int   = 64     # overflow drops the lowest-priority one
    confirm_sig_cooldown_after:      int   = 5      # BATCH_VARIANCE verdicts before a
                                                    # never-confirmed signature stops enqueueing
    confirm_feedback:                bool  = True   # verdicts adjust corpus weight
    confirm_feedback_delta:          float = 2.0
    # "ladder": FP-tie candidates are confirmed at low priority.  "log": they
    # are logged (verdict TIE_LOGGED) and never replayed, so the ladder's time
    # goes to candidates the logit screen did not already explain.
    confirm_ties:                    str   = "ladder"
    # Hidden size for Event.prompt_embeds (0 = send those requests as text).
    prompt_embeds_dim:               int   = 0
    # Append every oracle report (finding shape, with its trace) to
    # findings/reports.jsonl, so offline matchers can attribute each one.
    log_reports:                     bool  = False


@dataclass
class SoftCandidate:
    """A trace that triggered an exact-output mismatch in _determinism_probe.

    Held in _soft_queue until _run_soft_repro_batch() validates or suppresses it.
    Never persisted to disk.
    """
    trace:              Trace
    iteration:          int
    trigger_request_id: str
    fingerprint:        Optional[SchedulerFingerprint]
    timestamp:          float


@dataclass
class StructuralCandidate:
    """A trace that produced a KV structural anomaly (cross-lora reuse or fast reuse).

    Held in _structural_queue until _run_structural_repro_batch() confirms or
    suppresses it. Never persisted to disk.
    """
    trace:         Trace
    iteration:     int
    anomaly_types: list          # e.g. ["cross_lora_reuse"]
    top_score:     float
    summary:       object        # KVProbeSummary snapshot at trigger time
    fingerprint:   Optional[SchedulerFingerprint]
    timestamp:     float


@dataclass
class RecentTraceRecord:
    """One recently executed trace kept in memory for liveness forensics."""
    iteration: int
    timestamp: float
    trace: Trace


class Runner:

    def __init__(self,
                 config:   RunnerConfig,
                 adapter:  Optional[TargetAdapter] = None,
                 executor: Executor | None = None,
                 monitor:  Monitor  | None = None):
        from ..events import EventWriter
        self.events = EventWriter(config.events_file)
        self.config   = config
        self.adapter  = adapter
        self.executor = executor or PythonExecutor(
            trace_timeout_s=config.trace_timeout_s,
            adapter=adapter,
        )
        self.monitor  = monitor  or NvmlMonitor(
            metrics_url=f"{config.base_url}/metrics",
            poll_interval_s=config.poll_interval_s,
        )
        if hasattr(self.executor, "model_name"):
            self.executor.model_name = config.model_name
        self.oracle   = Oracle()
        self.corpus   = Corpus(config.seeds_dir, config.corpus_dir,
                               adapter_seeds_dir=config.adapter_seeds_dir)
        self.rng      = random.Random(config.seed)
        self._campaign_start: Optional[float] = None
        if config.prompt_embeds_dim and hasattr(self.executor, "prompt_embeds_dim"):
            self.executor.prompt_embeds_dim = config.prompt_embeds_dim
        # Separate, independently seeded stream for the prefix-equivalence
        # probe's cache_salt values.  Derived from config.seed so a rerun with
        # the same seed produces the same salts (reproducible), but drawn from
        # its own Random so enabling the probe does not shift the mutation
        # sequence self.rng drives.
        self._pfx_rng = random.Random(config.seed ^ 0x9E3779B9)
        # Confirmation ladder state.  Its salts come from their own stream so
        # turning the ladder on does not shift the prefix-equiv probe's salts.
        self._confirm_rng = random.Random(config.seed ^ 0x5C0FFEE5)
        self._confirm_queue: list = []          # ladder.Candidate
        self._confirm_seq: int = 0
        self._confirm_stats: Counter = Counter()
        #   mech_sig -> Counter(verdict -> n)
        self._confirm_sig_hist: dict[str, Counter] = {}
        self._confirm_guard: dict = {}
        # One-shot latch so the "running without a vacuity control" warning is
        # printed once per campaign rather than once per finding.
        self._pfx_vacuity_warned = False
        self.findings_dir = Path(config.findings_dir)
        self.findings_dir.mkdir(parents=True, exist_ok=True)
        self._cache_probe_warm_ttft: Optional[float] = None
        self._baseline_window:       list[float]     = []
        self._cold_start_baseline:   Optional[float] = None
        self._mem_idle_window:       list[float]     = []
        self._kv_idle_window:        list[float]     = []
        self._campaign_id:           str             = ""
        self._version_info:          dict            = {}
        self._collector = None   # AdapterCollector — set by adapter.make_collector()

        # Isolation probe scheduler state
        self._iso_cooldown_remaining: int = 0
        self._ISO_COOLDOWN_M:         int = 50
        self._ISO_DRAIN_N:            int = 200
        self._ISO_SIG_EXPIRY_T:       int = 500
        self._ISO_REENQUEUE_CAP:      int = 1
        self._iso_iteration:          int = 0
        # max-heap entries: (negated_score, enqueued_at_iter, probe_metadata_dict)
        self._iso_queue:    list = []
        self._iso_sig_seen: dict = {}  # sig -> enqueue count
        self._iso_sig_last: dict = {}  # sig -> last drain iteration

        # Finding dedup: per-mechanism ledger.  Never drops a finding
        # silently — every occurrence is counted; only the JSONs past the
        # K-th exemplar of a signature are suppressed.  In-memory and per
        # slot: it is dumped once at the end of the run and merged across
        # slots offline.
        #   sig -> {count, first_iter, last_iter, max_score, exemplars, classes}
        self._sig_ledger: dict[str, dict] = {}

        # Monotonic per-run counter appended to every finding filename.  The
        # (iteration, seed) pair is NOT unique: _determinism_probe files one
        # finding per diverging request of the same trace in the same
        # iteration, so a name built from those two alone silently overwrote
        # every finding after the first (measured: 65 filed, 44 on disk).
        self._finding_seq: int = 0

        # Measurement gaps: oracles that declined to judge because a
        # precondition of their measurement was not met (see
        # ``OracleResult.unmeasured``).  These are NOT passes, and the whole
        # kv_block_leak false-positive class came from an unmeasured quantity
        # being reported as a measurement — so they are counted here, printed
        # on first sighting of each distinct reason, attached to any finding
        # filed in the same iteration, and dumped in the signature ledger.
        #   reason -> occurrences
        self._measurement_gaps: Counter = Counter()
        self._gap_reasons_seen: set[str] = set()
        #: Gaps observed in the current iteration, consumed by _handle_finding.
        self._iter_measurement_gaps: list[str] = []

        # Liveness circuit breaker: abort after N consecutive health failures
        # so a crashed server doesn't silently consume the whole time budget.
        self._LIVENESS_ABORT_THRESHOLD: int   = 10
        self._consecutive_liveness_failures:  int   = 0
        self._RECENT_TRACE_BUFFER_SIZE: int = 10
        self._recent_traces = deque(maxlen=self._RECENT_TRACE_BUFFER_SIZE)

        # Soft-suspicion queue — never persisted, reset each campaign run
        self._soft_queue:          list  = []
        self._soft_cooldown:       dict  = {}   # sig -> expiry timestamp
        self._last_soft_repro_at:  float = time.time()
        self._last_fingerprint:    Optional[SchedulerFingerprint] = None

        # Structural KV anomaly queue — never persisted, reset each campaign run
        self._structural_queue:         list  = []
        self._structural_cooldown:      dict  = {}   # sig -> expiry timestamp
        self._last_structural_repro_at: float = time.time()

        # Ask the adapter to create a collector (returns None if not applicable).
        # Must be called after adapter is set, before mutation list is built.
        if self.adapter is not None:
            self._collector = self.adapter.make_collector(config)

        # Append any adapter-required server flags (skip if already present)
        if self.adapter is not None:
            _extra = self.adapter.server_flags_extra()
            if _extra and _extra not in self.config.server_flags:
                import dataclasses as _dc
                self.config = _dc.replace(
                    self.config,
                    server_flags=f"{self.config.server_flags} {_extra}".strip(),
                )

        # Build combined mutation list if adapter provided
        if self.adapter is not None:
            from .mutator import MUTATIONS as _BASE_MUTATIONS
            self._all_mutations = list(_BASE_MUTATIONS) + list(self.adapter.extra_mutations())
            # Build aligned weights array; adapter can override per-mutation weight
            weight_map = getattr(self.adapter, "mutation_weights", lambda: {})()
            self._mutation_weights = [
                weight_map.get(fn.__name__, 1.0) for fn in self._all_mutations
            ]
        else:
            self._all_mutations = None  # will use default MUTATIONS in mutate()
            self._mutation_weights = None

    async def _discover_server_version(self) -> dict:
        """Fetch server version info via the adapter, or fall back to vLLM defaults."""
        if self.adapter is not None:
            return await self.adapter.discover_version(self.config.base_url)

        info: dict = {}

        # 1. HTTP /version endpoint
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.config.base_url}/version", timeout=5.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    info["vllm_version"] = data.get("version", "unknown")
        except Exception:
            pass

        # 2. importlib.metadata fallback
        if "vllm_version" not in info:
            try:
                from importlib.metadata import version as _pkg_version
                info["vllm_version"] = _pkg_version("vllm")
            except Exception:
                info["vllm_version"] = "unknown"

        # 3. Commit hash — try embedded suffix first
        try:
            import vllm as _vllm
            full_ver = getattr(_vllm, "__version__", "")
            if "+" in full_ver:
                info["vllm_version"] = full_ver
                suffix = full_ver.split("+", 1)[1]
                if suffix.startswith("git"):
                    info["vllm_commit"] = suffix[3:]
        except Exception:
            pass

        # 4. Commit hash — subprocess git fallback
        if "vllm_commit" not in info:
            try:
                import pathlib as _pl
                import subprocess as _sp
                import shutil as _sh
                import vllm as _vllm
                pkg_dir = str(_pl.Path(_vllm.__file__).parent.parent)
                if _sh.which("git"):
                    r = _sp.run(
                        ["git", "-C", pkg_dir, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if r.returncode == 0:
                        info["vllm_commit"] = r.stdout.strip()
            except Exception:
                pass

        return info

    async def run(self) -> None:
        from dataclasses import asdict
        self.events.emit("run_started", config=asdict(self.config),
                         engine=self.adapter.__class__.__name__ if self.adapter else "vllm")
        outcome = "completed"
        # Before the collector starts: a ladder that cannot prove its BI
        # server is engaged must stop the run, not degrade silently.
        self._check_confirm_ladder()
        # Measure elapsed discovery time before minimization and replay.
        self._campaign_start = time.time()
        (self.findings_dir / "run_meta.json").write_text(json.dumps({
            "campaign_start": self._campaign_start,
            "search_mode": "grief",
            "seed": self.config.seed,
            "confirm_ladder": self.config.confirm_ladder,
            "confirm_feedback": self._confirm_feedback_on(),
        }, indent=2))
        if self._collector is not None:
            await self._collector.start()
        try:
            await self._run_loop()
        except asyncio.CancelledError:
            outcome = "stopped"
            raise
        except BaseException:
            outcome = "failed"
            raise
        finally:
            # Dumped before the collector stop so a failing stop() cannot
            # cost the ledger.
            self._dump_signature_ledger()
            self._dump_confirmation_summary()
            if self._collector is not None:
                # final=True: this is the last thing the process does, so the
                # collector may skip any teardown the exit would do anyway.
                await self._collector.stop(final=True)
            if hasattr(self.monitor, "aclose"):
                await self.monitor.aclose()
            self.events.emit("run_finished", status=outcome, signatures=len(self._sig_ledger))

    async def _calibration_phase(self) -> None:
        """Empirically mine prompt families where the loaded adapters diverge.

        Sends n_candidates synthetic prompt families to each adapter in isolation,
        scores each family by logprob divergence minus instability penalty across
        n_repeats runs, and seeds the corpus with the top_k highest-scoring
        families. Only runs when config.path_calibration is True and the adapter
        is a VllmLoRAAdapter.

        The conceptual unit is a prompt family configuration (prompt_len,
        prefix_len, family_id, score) — not a raw trace. Traces are constructed
        from the top families for corpus seeding, but callers should think in
        terms of families.
        """
        if not self.config.path_calibration:
            return
        # Lazy import to avoid circular dependency
        from ..adapters.vllm_lora.adapter import VllmLoRAAdapter
        if not isinstance(self.adapter, VllmLoRAAdapter):
            return

        print(
            f"[calibration] probing {self.config.calibration_n_candidates} prompt families "
            f"({self.config.calibration_n_repeats} repeat(s) each, "
            f"keeping top {self.config.calibration_top_k}) ..."
        )
        families = await self.adapter.calibrate(
            executor=self.executor,
            base_url=self.config.base_url,
            n_candidates=self.config.calibration_n_candidates,
            top_k=self.config.calibration_top_k,
            n_repeats=self.config.calibration_n_repeats,
        )
        seeded = 0
        for fam in families:
            trace = fam["trace"]
            self.corpus.add(trace, findings=[], save=True, min_score=1.0)
            seeded += 1
        if families:
            print(
                f"[calibration] seeded {seeded} discriminating prompt families "
                f"(top score={families[0]['score']:.3f})"
            )
        else:
            print("[calibration] no discriminating families found — all candidates scored 0")

    async def _run_loop(self) -> None:
        loaded = self.corpus.load_seeds()
        print(f"[runner] loaded {loaded} seeds")

        if self.config.path_calibration:
            await self._calibration_phase()

        self._version_info = await self._discover_server_version()
        if self.config.campaign_id:
            self._campaign_id = self.config.campaign_id
        else:
            self._campaign_id = hashlib.md5(
                f"{self.config.base_url}:{self.config.seed}:{self.config.server_flags}".encode()
            ).hexdigest()[:8]

        baseline_ttft_ms = await self._measure_baseline_ttft()
        if baseline_ttft_ms is not None:
            per_tok = baseline_ttft_ms / _BASELINE_PROMPT_LEN
            _probe_ev = Event(action=ActionType.SEND, offset_ms=0,
                              request_id="baseline_probe",
                              prompt_len=_BASELINE_PROMPT_LEN, max_tokens=16, stream=True)
            if self.adapter is not None:
                _probe_ev = self.adapter.baseline_probe_event_overrides(_probe_ev)
            _adapter_tag = (f"  adapter={_probe_ev.adapter_name}"
                            if _probe_ev.adapter_name is not None else "")
            print(f"[runner] baseline TTFT: {baseline_ttft_ms:.0f} ms @ "
                  f"{_BASELINE_PROMPT_LEN} tokens ({per_tok:.2f} ms/token)  "
                  f"multiplier: {self.oracle.latency_multiplier}x{_adapter_tag}")
            self.oracle.baseline_ttft_ms    = baseline_ttft_ms
            self.oracle.baseline_prompt_len = _BASELINE_PROMPT_LEN
            self._cold_start_baseline       = baseline_ttft_ms
            self._baseline_window           = [baseline_ttft_ms]
        else:
            print(f"[runner] baseline measurement failed — using fixed threshold "
                  f"{self.oracle.latency_ttft_threshold_ms:.0f} ms")

        # Resolve once: same callable reaches both main + splice select() calls,
        # so known-bug-shaped traces cannot still dominate as splice parents.
        sm = _score_multiplier_for_adapter(self.adapter)

        for iteration in range(self.config.max_iterations):
            trace = self.corpus.select(self.rng, score_multiplier=sm)
            if trace is None:
                print("[runner] corpus empty — add seed traces to seeds/")
                break

            other_trace = self.corpus.select(self.rng, score_multiplier=sm)
            other_pressure = 0.0
            if other_trace is not None:
                for _entry in self.corpus._entries:
                    if _entry.trace is other_trace:
                        other_pressure = _entry.pressure_score
                        break

            trace, mutation_ops = self._mutate_for_mode(
                trace, other_trace, other_pressure)

            self.events.emit("trace_started", iteration=iteration, trace=trace.to_dict(),
                             mutations=mutation_ops, corpus=self.corpus.size())
            if self._collector is not None:
                await self._collector.begin_probe()
            await self.monitor.start()
            try:
                result = await self.executor.run(trace, self.config.base_url)
            finally:
                session = await self.monitor.stop()
            self._record_recent_trace(trace, iteration)
            # Collect probe summary — end_probe() reads the window opened above.
            kv_summary = None
            kv_fp      = None
            if self._collector is not None:
                kv_summary = await self._collector.end_probe()
                if kv_summary is not None:
                    kv_fp = getattr(kv_summary, "fingerprint", None) or None

            # Transition coverage: the edge is (previous iteration's
            # fingerprint -> this one).  prev_fp is captured before
            # _last_fingerprint is overwritten; _last_fingerprint itself keeps
            # being assigned here (not after corpus.add) because the structural
            # queue and _handle_finding both read it as "the current trace's
            # fingerprint" further down this iteration.
            prev_fp = self._last_fingerprint
            fp      = build_fingerprint(
                session, result, version=self.corpus.fingerprint_version)
            self._last_fingerprint = fp

            if not await self._check_liveness(iteration):
                continue

            if self.config.verbose:
                self._emit_verbose_log(iteration, trace, mutation_ops, kv_summary)

            findings    = self.oracle.evaluate(result, session)
            annotations = self.oracle.annotate(result)
            self._record_measurement_gaps(iteration)

            # Adapter extra checks (behavioral only — no kv_summary parameter)
            if self.adapter is not None:
                extra = self.adapter.evaluate_extra(result, session, trace=trace)
                if extra:
                    findings = findings + [f for f in extra if not f.passed]
                findings = self.adapter.filter_findings(findings)

            # Enqueue isolation probe candidates — scan all oracle results including
            # passed=True bootstrap triggers from evaluate_extra, not just bug findings.
            if self.adapter is not None and hasattr(self.adapter, "adapter_isolation_probe"):
                import heapq as _heapq
                for _f in findings + extra:
                    _meta = getattr(_f, "probe_metadata", None)
                    if _meta and _f.score >= 3.0:
                        _sig = (
                            _meta.get("adapter_a", ""),
                            _meta.get("adapter_b", ""),
                            _meta.get("prompt_len", 0) // 64,
                            _meta.get("prefix_len", 0) // 64,
                        )
                        _count = self._iso_sig_seen.get(_sig, 0)
                        if _count <= self._ISO_REENQUEUE_CAP:
                            self._iso_sig_seen[_sig] = _count + 1
                            _heapq.heappush(
                                self._iso_queue,
                                (-_f.score, self._iso_iteration, _meta),
                            )

            error_sig = error_signature(result.requests)

            novel = self.corpus.add(trace, findings, fingerprint=fp,
                                    error_sig=error_sig, session=session,
                                    kv_fingerprint=kv_fp,
                                    prev_fingerprint=prev_fp)

            bugs = [f for f in findings if not f.passed]
            self.events.emit("iteration_finished", iteration=iteration,
                corpus=self.corpus.size(), novel=novel,
                fingerprint=fp, candidates=len(bugs),
                requests=[{"request_id": r.request_id,
                    "sent_ms": round((r.sent_at - result.started_at) * 1000),
                    "end_ms": round(((r.completed_at or r.cancelled_at or result.finished_at)
                                     - result.started_at) * 1000),
                    "tokens": r.tokens_received, "error": r.error,
                    "cancelled": r.cancelled_at is not None} for r in result.requests],
                measurement_gaps=self._iter_measurement_gaps,
                bugs=[{"class": b.bug_class, "detail": b.detail, "score": b.score}
                      for b in bugs])

            # Structural collection analysis — delegated entirely to the adapter.
            # adapter.analyze_collection() runs both the generic and adapter-specific
            # passes and returns CollectionAnomaly objects.
            kv_anomalies: list = []
            if kv_summary is not None and self.adapter is not None:
                kv_anomalies = self.adapter.analyze_collection(kv_summary)

            # Pairing: attach structural evidence to any existing KV_CORRUPTION finding.
            # Does not change the finding score — the behavioral oracle owns its score.
            # Structural metadata is attached for triage and forensic context only.
            if kv_anomalies and bugs:
                top_kv = max(kv_anomalies, key=lambda a: a.score)
                if top_kv.score >= 5.0:
                    for b in bugs:
                        if b.bug_class == BugClass.KV_CORRUPTION:
                            if b.probe_metadata is None:
                                b.probe_metadata = {}
                            b.probe_metadata["kv_structural"] = {
                                "anomaly_type": top_kv.anomaly_type,
                                "count":        top_kv.count,
                                "score":        top_kv.score,
                                "detail":       top_kv.detail,
                                "metadata":     top_kv.metadata,
                            }

            # Structural alone: enqueue for confirmation repro when no behavioral
            # finding fired. If any bug (including non-KV classes) is present, the
            # trace is already being saved as a finding; don't also enqueue it for
            # structural repro — doing so would double-process the same trace and
            # distort repro budget. The structural metadata is already attached above
            # for KV_CORRUPTION co-occurrences; for other bug classes, the structural
            # signal is recorded in the finding JSON via collection_forensics in _handle_finding.
            if kv_anomalies and not bugs:
                top_kv = max(kv_anomalies, key=lambda a: a.score)
                if top_kv.score >= 5.0:
                    _sig = self._structural_candidate_sig(
                        trace, top_kv.anomaly_type
                    )
                    _expiry = self._structural_cooldown.get(_sig)
                    if not _expiry or time.time() >= _expiry:
                        self._structural_queue.append(StructuralCandidate(
                            trace=trace,
                            iteration=iteration,
                            anomaly_types=[a.anomaly_type for a in kv_anomalies
                                           if a.score >= 5.0],
                            top_score=top_kv.score,
                            summary=kv_summary,
                            fingerprint=self._last_fingerprint,
                            timestamp=time.time(),
                        ))
                        print(f"[{iteration:05d}] structural anomaly enqueued  "
                              f"type={top_kv.anomaly_type}  "
                              f"score={top_kv.score:.1f}  "
                              f"queue_len={len(self._structural_queue)}")
            pre_delta = session.preemption_delta()
            if bugs:
                await self._handle_finding(trace, bugs, iteration, annotations=annotations)
            elif novel:
                pre_tag = f" preemptions={pre_delta}" if pre_delta else ""
                print(f"[{iteration:05d}] ok  corpus={self.corpus.size()}  [novel {fp}{pre_tag}]")
            else:
                print(f"[{iteration:05d}] ok  corpus={self.corpus.size()}")

            if self.config.probe_determinism_every > 0 and \
                    iteration % self.config.probe_determinism_every == 0:
                await self._determinism_probe(trace, result, iteration)
                await self._refresh_baseline(iteration)
                await self._maybe_trigger_soft_repro()
                await self._maybe_trigger_structural_repro()
                if self.adapter is not None:
                    _adapter_pairs = await self.adapter.run_confirmation_batch(
                        self.executor, self.config.base_url
                    )
                    for _stub_trace, _finding in _adapter_pairs:
                        await self._handle_finding(_stub_trace, [_finding], iteration)

            if self.config.confirm_ladder and self.config.confirm_every > 0 and \
                    iteration % self.config.confirm_every == 0:
                await self._drain_confirmations(iteration)

            # Independent modulus: the relational probe runs *beside*
            # _determinism_probe, not instead of it, so the two probes'
            # survivor rates stay comparable in an A/B.
            if self.config.prefix_equiv_probe_every > 0 and \
                    iteration % self.config.prefix_equiv_probe_every == 0:
                await self._prefix_equivalence_probe(trace, result, iteration)

            if self.config.probe_cache_hit_every > 0 and \
                    iteration % self.config.probe_cache_hit_every == 0:
                await self._cache_hit_probe(iteration)

            await asyncio.sleep(self.config.settle_s)
            await self._update_idle_windows(iteration, trace)

            # Isolation probe drain
            self._iso_iteration += 1
            if self._iso_cooldown_remaining > 0:
                self._iso_cooldown_remaining -= 1
            elif self._iso_queue and self._iso_iteration % self._ISO_DRAIN_N == 0:
                await self._iso_drain(self.config.base_url)

    _ALWAYS_MINIMIZE: frozenset = frozenset({
        BugClass.LIFECYCLE_ERROR,
        BugClass.SCHEDULER_HANG,
        BugClass.MEMORY_LEAK,
        BugClass.KV_BLOCK_LEAK,
        BugClass.FEATURE_FAILURE,
    })
    _NEVER_MINIMIZE: frozenset = frozenset({
        BugClass.KV_CORRUPTION,
        # Latency spikes must not be minimized: (a) every minimizer probe involves
        # a slow request, making it disproportionately expensive; (b) minimization
        # tends to strip the earlier requests that create the cache pressure causing
        # the spike, producing a trace that is harder to reproduce in isolation.
        # The confirmed original trace is already the most informative artifact.
        BugClass.LATENCY_SPIKE,
        # Token rate stalls are measured over elapsed wall time — minimizing strips
        # the concurrent load that causes the stall, making the reduced trace fail
        # to reproduce. Also very expensive given the long elapsed times involved.
        BugClass.TOKEN_RATE_STALL,
    })
    _CONFIRM_BAND_LOW:  float = 10.0
    _CONFIRM_BAND_HIGH: float = 20.0

    def _should_minimize(self, bugs: list[OracleResult]) -> bool:
        if not self.config.minimize_findings:
            return False
        bug_classes = {b.bug_class for b in bugs if b.bug_class}
        if not bug_classes:
            return False
        # ALWAYS_MINIMIZE takes priority: if any structural bug is present,
        # minimize regardless of what else is in the set.
        if bug_classes & self._ALWAYS_MINIMIZE:
            return True
        # NEVER_MINIMIZE applies only when no always-minimize bug is present.
        if bug_classes & self._NEVER_MINIMIZE:
            return False
        return True

    def _worst_latency_score(self, bugs: list[OracleResult]) -> float:
        return max(
            (b.score for b in bugs if b.bug_class == BugClass.LATENCY_SPIKE),
            default=0.0,
        )

    async def _check_liveness(self, iteration: int) -> bool:
        """Hit /health after each trace execution.

        Tracks consecutive failures.  After _LIVENESS_ABORT_THRESHOLD consecutive
        failures, raises RuntimeError so the campaign exits rather than running
        blind for the rest of the time budget against a permanently dead server.
        """
        healthy = False
        detail  = ""

        if self.adapter is not None:
            healthy = await self.adapter.check_liveness(self.config.base_url)
            detail  = "server /health check failed"
        else:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self.config.base_url}/health", timeout=3.0
                    )
                    if resp.status_code != 200:
                        raise RuntimeError(f"health={resp.status_code}")
                healthy = True
            except Exception as exc:
                detail = f"vLLM /health failed: {exc}"

        if not healthy:
            self._consecutive_liveness_failures += 1
            finding = OracleResult(
                passed=False,
                bug_class=BugClass.LIVENESS_FAILURE,
                detail=detail,
                score=10.0,
            )
            finding_path = await self._handle_finding(
                Trace(seed=0, events=[]), [finding], iteration
            )
            if finding_path is not None:
                self._write_recent_trace_context(finding_path)

            if self._consecutive_liveness_failures >= self._LIVENESS_ABORT_THRESHOLD:
                raise RuntimeError(
                    f"[runner] server has been unhealthy for "
                    f"{self._consecutive_liveness_failures} consecutive iterations "
                    f"— aborting campaign to avoid wasting time budget."
                )
            return False

        self._consecutive_liveness_failures = 0
        return True

    async def _confirm_finding(self,
                                trace: Trace,
                                original_classes: set,
                                n: int = 2,
                                required: int = 2) -> bool:
        """Re-run the trace n times; return True if the oracle fires on >= required runs."""
        fires = 0
        for _ in range(n):
            await self.monitor.start()
            try:
                result = await self.executor.run(trace, self.config.base_url)
            finally:
                session = await self.monitor.stop()
            findings     = self.oracle.evaluate(result, session)
            found_classes = {f.bug_class for f in findings if not f.passed}
            if original_classes & found_classes:
                fires += 1
        return fires >= required

    # ------------------------------------------------------------------
    # Confirmation ladder (fuzzer/confirm/ladder.py)
    # ------------------------------------------------------------------

    def _check_confirm_ladder(self) -> None:
        """Refuse to start a ladder run whose BI server is not provably BI.

        Guard 1 (engagement, read from the BI server's log) and guard 2 (the
        attention backend allowlist) are the ones ``scripts/bi_confirm.py``
        applies before any replay.  A failure raises: a ladder that quietly
        compared against a non-invariant server would label real batch
        variance as confirmed.
        """
        cfg = self.config
        if not cfg.confirm_ladder:
            return
        from ..confirm.bi_guard import (prove_engagement, resolve_backend,
                                        resolve_gpu_arch, rule_backend)
        if not cfg.confirm_bi_url:
            raise RuntimeError("--confirm-ladder needs --confirm-bi-url "
                               "(a server started with VLLM_BATCH_INVARIANT=1)")
        if cfg.confirm_bi_url.rstrip("/") == cfg.base_url.rstrip("/"):
            raise RuntimeError("--confirm-bi-url must name a second server, "
                               "not the fuzzing server --base-url")
        engagement = prove_engagement(cfg.confirm_bi_log or None)
        log_text = ""
        if cfg.confirm_bi_log:
            try:
                log_text = Path(cfg.confirm_bi_log).read_text(errors="replace")
            except OSError:
                log_text = ""
        backend, fa_version, backend_src = resolve_backend(
            {}, log_text=log_text, override=cfg.confirm_bi_backend or None)
        arch = resolve_gpu_arch({}, log_text=log_text,
                                override=cfg.confirm_gpu_arch or None)
        ruling = rule_backend(
            backend, arch=arch, fa_version=fa_version,
            resolved_from=backend_src,
            fa_version_source=("server log" if fa_version is not None
                               else "declared"))
        self._confirm_guard = {
            "bi_url": cfg.confirm_bi_url,
            "bi_log": cfg.confirm_bi_log,
            "engagement": engagement.to_dict(),
            "backend": ruling.to_dict(),
        }
        try:
            (self.findings_dir / "confirmation_guard.json").write_text(
                json.dumps(self._confirm_guard, indent=2, default=str))
        except OSError as exc:
            print(f"[confirm] guard record not written: {exc}")
        if not engagement.proven:
            raise RuntimeError(f"BI engagement not proven from "
                               f"{cfg.confirm_bi_log or '(no --confirm-bi-log)'}: "
                               f"{engagement.reason}")
        if not ruling.allowed:
            raise RuntimeError(f"BI backend not allowlisted ({ruling.status}): "
                               f"{ruling.reason}")
        print(f"[confirm] ladder on: BI server {cfg.confirm_bi_url} engaged "
              f"({engagement.signal}); backend {ruling.name} "
              f"FA{ruling.fa_version} [{ruling.status}]")

    def _enqueue_confirmation(self, *, report_trace: Trace, replay_trace: Trace,
                              rid: str, source: str, bugs: list,
                              iteration: int, priority: int = 0,
                              twin_rid: Optional[str] = None,
                              twin_needs_cache_hit: bool = False,
                              extra: Optional[dict] = None,
                              logprob_gap: Optional[float] = None,
                              logprob_gap_ulps: Optional[float] = None) -> None:
        """Queue one divergence for the ladder instead of filing it."""
        import copy as _copy
        from ..confirm.ladder import (Candidate, FILED_VERDICTS, PRIORITY_LOW,
                                      VERDICT_BATCH_VARIANCE)
        prio = "tie" if priority == PRIORITY_LOW else "nontie"
        sig = mechanism_signature(report_trace, bugs)
        hist = self._confirm_sig_hist.get(sig)
        k = self.config.confirm_sig_cooldown_after
        if hist is not None and k > 0 \
                and hist[VERDICT_BATCH_VARIANCE] >= k \
                and not any(hist[v] for v in FILED_VERDICTS):
            self._confirm_stats["cooled"] += 1
            self._confirm_stats[f"cooled|{prio}"] += 1
            return
        report_copy = _copy.deepcopy(report_trace)
        replay_copy = (report_copy if replay_trace is report_trace
                       else _copy.deepcopy(replay_trace))
        self._confirm_seq += 1
        cand = Candidate(
            report_trace=report_copy, replay_trace=replay_copy, rid=rid,
            source=source, bugs=list(bugs), iteration=iteration,
            priority=priority, score=sum(b.score for b in bugs),
            twin_rid=twin_rid, twin_needs_cache_hit=twin_needs_cache_hit,
            extra=dict(extra or {}), fingerprint=self._last_fingerprint,
            mech_sig=sig, seq=self._confirm_seq,
            logprob_gap=logprob_gap, logprob_gap_ulps=logprob_gap_ulps)
        if priority == PRIORITY_LOW and self.config.confirm_ties == "log":
            self._confirm_stats["tie_logged"] += 1
            self._log_confirmation(self._confirmation_record(
                cand, -1, {"verdict": "TIE_LOGGED",
                           "reason": "FP tie; --confirm-ties log"}))
            return
        self._confirm_queue.append(cand)
        self._confirm_stats["enqueued"] += 1
        self._confirm_stats[f"enqueued|{prio}"] += 1
        print(f"[{iteration:05d}] divergence queued for confirmation "
              f"request={rid}  source={source}  priority={prio}  "
              f"queue={len(self._confirm_queue)}")
        cap = self.config.confirm_queue_max
        if cap > 0 and len(self._confirm_queue) > cap:
            self._confirm_queue.sort(key=lambda c: c.sort_key())
            dropped = self._confirm_queue.pop()
            self._confirm_stats["dropped"] += 1
            self._log_confirmation(self._confirmation_record(
                dropped, iteration, {"verdict": "DROPPED",
                                     "reason": f"queue over {cap}"}))

    async def _drain_confirmations(self, iteration: int) -> None:
        """Run the ladder on the highest-priority candidates, within budget.

        Called between iterations, so nothing else is in flight on the stock
        server while its solo arms run.
        """
        if not self._confirm_queue:
            return
        self._confirm_queue.sort(key=lambda c: c.sort_key())
        start = time.time()
        done = 0
        while (self._confirm_queue
               and done < self.config.confirm_max_per_drain
               and time.time() - start < self.config.confirm_budget_s):
            cand = self._confirm_queue.pop(0)
            await self._confirm_one(cand, iteration)
            done += 1

    async def _confirm_one(self, cand, iteration: int) -> None:
        from ..confirm import ladder as L
        t0 = time.time()
        outcomes: dict = {}
        try:
            arm_traces = L.build_arms(cand, self._confirm_rng)
        except KeyError as exc:
            verdict = L.LadderVerdict(L.VERDICT_UNINFORMATIVE,
                                      f"cannot build arms: {exc}")
        else:
            async def run_arms(names, url):
                out = {}
                for name in names:
                    try:
                        res = await self.executor.run(arm_traces[name], url)
                    except Exception as exc:                  # arm failure, not a crash
                        out[name] = L.arm_outcome(name, None, cand.rid,
                                                  cand.twin_rid, exc=exc)
                    else:
                        out[name] = L.arm_outcome(name, res, cand.rid,
                                                  cand.twin_rid)
                return out
            # Two servers, two independent sequences: the stock solo pair
            # and the three BI arms run concurrently.
            stock, bi = await asyncio.gather(
                run_arms(("S1", "S2"), self.config.base_url),
                run_arms(("B1", "B2", "BS"), self.config.confirm_bi_url))
            outcomes = {**stock, **bi}
            verdict = L.judge(outcomes, twin=cand.twin_rid is not None,
                              twin_needs_cache_hit=cand.twin_needs_cache_hit)
        latency = time.time() - t0
        prio = "tie" if cand.priority == L.PRIORITY_LOW else "nontie"
        self._confirm_stats[verdict.verdict] += 1
        self._confirm_stats[f"{verdict.verdict}|{prio}"] += 1
        self._confirm_sig_hist.setdefault(cand.mech_sig, Counter())[verdict.verdict] += 1
        feedback = self._confirm_feedback(cand, verdict)
        record = self._confirmation_record(
            cand, iteration, dict(verdict.to_log(), filed=verdict.filed),
            arms={n: o.to_log() for n, o in outcomes.items()},
            latency_s=latency, feedback=feedback)
        self._log_confirmation(record)
        print(f"[{iteration:05d}] confirmation {verdict.verdict}  "
              f"request={cand.rid}  source={cand.source}  priority={prio}  "
              f"({latency:.1f}s)  {verdict.reason}")
        if verdict.filed:
            extra = dict(cand.extra)
            extra["confirmation"] = {
                "verdict": verdict.verdict,
                "reason": verdict.reason,
                "request_id": cand.rid,
                "twin_request_id": cand.twin_rid,
                "source": cand.source,
                "deciding_pair": record["deciding_pair"],
                "diverge_char_idx": verdict.diverge_char_idx,
                "comparisons": verdict.comparisons,
                "arms": record["arms"],
                "priority": prio,
                "confirmed_at_iteration": iteration,
                "latency_s": latency,
                "bi_url": self.config.confirm_bi_url,
            }
            await self._handle_finding(cand.report_trace, cand.bugs,
                                       cand.iteration, extra=extra)

    def _log_latency_candidate(self, trace, bugs_detected, iteration,
                               detected_at, outcome) -> None:
        """Record every latency-oracle firing, filed or not.

        A one-shot stall (e.g. a kernel JIT-compiled mid-serving on first use)
        cannot be re-confirmed by a replay, so the confirmation band drops it
        without a trace.  ``latency_candidates.jsonl`` keeps each firing in the
        finding-file shape (``bugs_detected``, ``trace_original``,
        ``campaign_elapsed_s``) with its ``outcome``: filed,
        discarded_low_score or not_reproduced.
        """
        rec = {
            "iteration": iteration,
            "detected_at": detected_at,
            "campaign_elapsed_s": (detected_at - self._campaign_start
                                   if self._campaign_start else None),
            "search_mode": "grief",
            "outcome": outcome,
            "bugs_detected": bugs_detected,
            "trace_original": trace.to_dict(),
        }
        try:
            with open(self.findings_dir / "latency_candidates.jsonl", "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError as exc:
            print(f"[runner] latency candidate not logged: {exc}")

    def _log_report(self, trace, bugs_detected, iteration, detected_at) -> None:
        """Append one oracle report to findings/reports.jsonl (finding shape)."""
        rec = {
            "iteration": iteration,
            "detected_at": detected_at,
            "campaign_elapsed_s": (detected_at - self._campaign_start
                                   if self._campaign_start else None),
            "search_mode": "grief",
            "bugs_detected": bugs_detected,
            "trace_original": trace.to_dict(),
        }
        try:
            with open(self.findings_dir / "reports.jsonl", "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError as exc:
            print(f"[runner] report not logged: {exc}")

    def _mutate_for_mode(self, trace, other_trace, other_pressure):
        """Mutate using serving-state feedback and adapter operator weights."""
        return mutate(
            trace, self.rng, n=self.config.mutations_per_seed,
            other=other_trace, other_pressure=other_pressure,
            mutations=self._all_mutations, weights=self._mutation_weights,
        )

    def _confirm_feedback_on(self) -> bool:
        return self.config.confirm_feedback

    def _confirm_feedback(self, cand, verdict) -> str:
        """Feed the verdict back into corpus selection; returns what was done."""
        from ..confirm.ladder import VERDICT_BATCH_VARIANCE
        if not self._confirm_feedback_on():
            return "off"
        if cand.fingerprint is None:
            return "no_fingerprint"
        delta = self.config.confirm_feedback_delta
        if verdict.filed:
            if self.corpus.boost(cand.fingerprint, delta):
                return "boost"
            self.corpus.add(cand.report_trace, findings=cand.bugs, save=True,
                            fingerprint=cand.fingerprint)
            return "add"
        if verdict.verdict == VERDICT_BATCH_VARIANCE:
            if self.corpus.adjust(cand.fingerprint, -delta, floor=0.0):
                return "down"
            return "down_no_entry"
        return "none"

    def _confirmation_record(self, cand, iteration: int, verdict: dict,
                             arms: Optional[dict] = None,
                             latency_s: Optional[float] = None,
                             feedback: Optional[str] = None) -> dict:
        from ..confirm.ladder import PRIORITY_LOW
        rec = {
            "t": time.time(),
            "seq": cand.seq,
            "iteration": cand.iteration,
            "drained_at_iteration": iteration,
            "source": cand.source,
            "request_id": cand.rid,
            "twin_request_id": cand.twin_rid,
            "priority": "tie" if cand.priority == PRIORITY_LOW else "nontie",
            "bug_classes": [b.bug_class.value for b in cand.bugs if b.bug_class],
            "score": cand.score,
            "mech_sig": cand.mech_sig,
            "logprob_gap": cand.logprob_gap,
            "logprob_gap_ulps": cand.logprob_gap_ulps,
            "deciding_pair": None,
        }
        rec.update(verdict)
        if arms is not None:
            rec["arms"] = arms
        if latency_s is not None:
            rec["latency_s"] = latency_s
        if feedback is not None:
            rec["feedback"] = feedback
        return rec

    def _log_confirmation(self, record: dict) -> None:
        """Append one line to ``confirmation_log.jsonl`` -- the funnel's raw data."""
        self.events.emit("confirmation", **record)
        try:
            with open(self.findings_dir / "confirmation_log.jsonl", "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:                         # never fail a run over this
            print(f"[confirm] log line not written: {exc}")

    def _dump_confirmation_summary(self) -> None:
        """Write ``confirmation_summary.json``; log what was never drained."""
        if not self.config.confirm_ladder:
            return
        try:
            for cand in self._confirm_queue:
                self._confirm_stats["unprocessed"] += 1
                self._log_confirmation(self._confirmation_record(
                    cand, -1, {"verdict": "UNPROCESSED",
                               "reason": "still queued when the run ended"}))
            self._confirm_queue = []
            summary = {
                "generated_at": time.time(),
                "guard": self._confirm_guard,
                "counts": dict(self._confirm_stats),
                "config": {k: getattr(self.config, k) for k in (
                    "confirm_bi_url", "confirm_every", "confirm_max_per_drain",
                    "confirm_budget_s", "confirm_queue_max",
                    "confirm_sig_cooldown_after", "confirm_feedback", "confirm_ties",
                    "confirm_feedback_delta")},
            }
            (self.findings_dir / "confirmation_summary.json").write_text(
                json.dumps(summary, indent=2, default=str))
        except Exception as exc:                       # never fail a run over this
            print(f"[confirm] summary not written: {exc}")

    def _record_measurement_gaps(self, iteration: int) -> None:
        """Account for oracles that declined to judge this iteration.

        A gap is not a pass.  ``kv_block_leak`` spent this fuzzer's entire
        history reporting an unmeasured quantity as a measurement; gating the
        oracle without making the ungated case *visible* would just move the
        silence one step downstream — the campaign would show zero
        ``kv_block_leak`` findings and no indication of whether that means
        "no leaks" or "never once checked".

        Each distinct reason prints once, on first sighting, then only counts:
        under a timeout-heavy workload the drain gap can hit every iteration,
        and a per-iteration line would be log spam that is itself ignored.
        """
        gaps = list(getattr(self.oracle, "last_measurement_gaps", ()) or ())
        self._iter_measurement_gaps = []
        for g in gaps:
            reason = g.unmeasured or "unspecified"
            self._measurement_gaps[reason] += 1
            self._iter_measurement_gaps.append(reason)
            # Bucket the print by check name, not by the full reason: the
            # running/waiting counts vary per iteration and would otherwise
            # make every occurrence a "first" sighting.
            key = reason.split("(")[0].strip()
            if key not in self._gap_reasons_seen:
                self._gap_reasons_seen.add(key)
                print(f"[{iteration:05d}] UNMEASURED {reason} "
                      f"— this check cannot fire until the condition clears")

    async def _handle_finding(self,
                               trace: Trace,
                               bugs: list[OracleResult],
                               iteration: int,
                               extra: Optional[dict] = None,
                               annotations: Optional[list] = None) -> Optional[Path]:
        # Discovery time is recorded when the oracle fires, before any
        # confirmation / minimisation time is spent on it.
        detected_at = time.time()
        bugs_detected = [{"class": b.bug_class, "detail": b.detail, "score": b.score,
                          "probe_metadata": b.probe_metadata} for b in bugs]
        classes = [b.bug_class.value for b in bugs if b.bug_class]
        score   = sum(b.score for b in bugs)
        kv_boost_active = any(
            b.probe_metadata and b.probe_metadata.get("kv_cross_lora_boost")
            for b in bugs
        )
        kv_tag = "  [KV-confirmed]" if kv_boost_active else ""
        print(f"[{iteration:05d}] BUG {classes}  score={score:.1f}{kv_tag}")
        if self.config.log_reports:
            self._log_report(trace, bugs_detected, iteration, detected_at)

        original_classes = {b.bug_class for b in bugs if b.bug_class}
        latency_only = original_classes == {BugClass.LATENCY_SPIKE}

        if latency_only:
            worst = self._worst_latency_score(bugs)
            if worst < self._CONFIRM_BAND_LOW:
                print(f"[{iteration:05d}] discarded (score {worst:.1f} < {self._CONFIRM_BAND_LOW})")
                self._log_latency_candidate(trace, bugs_detected, iteration,
                                            detected_at, "discarded_low_score")
                return None
            if worst < self._CONFIRM_BAND_HIGH:
                confirmed = await self._confirm_finding(trace, original_classes)
                if not confirmed:
                    # Don't file a finding, but keep the trace in corpus at a
                    # reduced weight so mutations of it get another chance.
                    # Score floor = worst/4 (e.g. 14.9 → 3.7).
                    # Synthetic pressure: capped, compressed latency-ratio map.
                    # Ignores first 1× baseline, grows at 0.35/unit, capped at
                    # 5.0 so absurd ratios don't dominate splice selection.
                    weak_score = worst / 4.0
                    synth_pressure = min(5.0, max(0.5, (worst - 1.0) * 0.35))
                    self.corpus.add(
                        trace, findings=[], save=True,
                        fingerprint=self._last_fingerprint,
                        min_score=weak_score,
                        pressure_score=synth_pressure,
                    )
                    print(f"[{iteration:05d}] unconfirmed latency (score {worst:.1f}) "
                          f"→ corpus seed (weak_score={weak_score:.1f} "
                          f"pressure={synth_pressure:.2f})")
                    self._log_latency_candidate(trace, bugs_detected, iteration,
                                                detected_at, "not_reproduced")
                    return None
                print(f"[{iteration:05d}] confirmed (2/2 runs triggered)")
        if BugClass.LATENCY_SPIKE in original_classes:
            self._log_latency_candidate(trace, bugs_detected, iteration,
                                        detected_at, "filed")

        # Mechanism signature — deliberately computed from the *pre*-minimization
        # trace and bugs, so a successful minimization can never move a
        # re-discovery of the same mechanism into a different bucket than the
        # one it landed in last time.
        sig = mechanism_signature(trace, bugs)
        rec = self._sig_ledger.setdefault(sig, {
            "count":      0,
            "first_iter": iteration,
            "last_iter":  iteration,
            "max_score":  0.0,
            "exemplars":  0,
            "classes":    [],
        })
        rec["count"]    += 1
        rec["last_iter"] = iteration
        rec["max_score"] = max(rec["max_score"], score)
        for cls in classes:
            if cls not in rec["classes"]:
                rec["classes"].append(cls)

        # Never silently drop: past the K-th exemplar the JSON is suppressed
        # but the occurrence is still counted in the ledger.
        if rec["exemplars"] >= self.config.exemplars_per_signature:
            print(f"[{iteration:05d}] duplicate mechanism {sig} (n={rec['count']})")
            return None

        # Keep the trace the bugs were actually observed on.  ``bugs`` is never
        # recomputed below (``reverify_minimized`` is off by default and, even
        # on, only *replaces* them when the minimized trace re-triggers), so
        # after minimization the report would otherwise pair post-minimization
        # events with pre-minimization findings — 16 of the last campaign's 101
        # degenerate_mask findings, and all 3 schema_violation ones, named a
        # request id absent from their own stored trace.
        original_trace = trace
        minimized_validation = {"status": "not_minimized"}

        if self._should_minimize(bugs):
            trace = await self._minimize(trace, bugs)
            minimized_validation = {"status": "not_rechecked"}
            if self.config.reverify_minimized:
                try:
                    await self.monitor.start()
                    try:
                        min_result = await self.executor.run(trace, self.config.base_url)
                    finally:
                        min_session = await self.monitor.stop()
                    min_bugs = self.oracle.evaluate(min_result, min_session)
                    if self.adapter is not None:
                        min_bugs += self.adapter.evaluate_extra(min_result, min_session, trace=trace)
                        min_bugs = self.adapter.filter_findings(min_bugs)
                    min_bugs = [b for b in min_bugs if not b.passed]
                    matches = bool(min_bugs) and mechanism_signature(trace, min_bugs) == sig
                    minimized_validation = {"status": "signature_reproduced" if matches else "not_reproduced",
                        "observed_classes": [b.bug_class for b in min_bugs],
                        "scope": "single replay; mechanism signature, not root-cause proof"}
                except Exception as exc:
                    minimized_validation = {"status": "error", "reason": str(exc)}
            self.events.emit("minimization_finished", original_events=len(original_trace.events),
                             reduced_events=len(trace.events), validation=minimized_validation)

        # lifecycle_error is demoted to telemetry: the oracle fires reliably
        # during live runs but does not reproduce in controlled conditions.
        # Only suppress when it is the sole bug class; mixed findings still file.
        if all(b.bug_class == BugClass.LIFECYCLE_ERROR for b in bugs):
            rids = [b.detail for b in bugs]
            print(f"[{iteration:05d}] TELEMETRY lifecycle_error (not filed): {rids}")
            return None

        import os, socket
        report = {
            "minimized_validation": minimized_validation,
            "iteration":          iteration,
            "detected_at":        detected_at,
            "campaign_elapsed_s": (detected_at - self._campaign_start
                                   if self._campaign_start else None),
            "search_mode":        "grief",
            "bugs_detected":      bugs_detected,
            "seed":               trace.seed,
            "trace":              trace.to_dict(),
            "bugs":               [{"class": b.bug_class, "detail": b.detail,
                                    "score": b.score,
                                    "probe_metadata": b.probe_metadata}
                                   for b in bugs],
            "mech_sig":           sig,
            "mech_sig_version":   SIG_VERSION,
            "sig_occurrence":     rec["count"],
            "campaign_id":        self._campaign_id,
            "server_flags":       self.config.server_flags,
            "baseline_ttft_ms":   self._cold_start_baseline,
            "baseline_prompt_len": _BASELINE_PROMPT_LEN,
            # Campaign-mode metadata (empty when not running under a campaign).
            # Auto-detected: hostname, slurm_job_id. Explicit: rest.
            "campaign_name":      self.config.campaign_name,
            "config_id":          self.config.config_id,
            "config_file":        self.config.config_file,
            "mode":               self.config.mode,
            "model":              self.config.model_name,
            "engine": "sglang" if "sglang" in type(self.adapter).__module__ else "vllm",
            "adapter": type(self.adapter).__name__,
            "gpu_type":           self.config.gpu_type,
            "slot_id":            self.config.slot_id,
            "rotation_index":     self.config.rotation_index,
            "slurm_job_id":       os.environ.get("SLURM_JOB_ID", ""),
            "hostname":           socket.gethostname(),
            # Version keys come from the adapter's discover_version() so they
            # reflect the actual backend (e.g. "sglang_version" for SGLang,
            # "vllm_version"/"vllm_commit" for vLLM).  Spreading the dict
            # avoids writing misleading "vllm_version: unknown" for non-vLLM runs.
            **self._version_info,
        }
        # Store BOTH traces rather than recomputing the bugs.
        #
        # ``trace`` keeps its meaning — the *minimized* trace, the smallest
        # reproducer.  Offline reproduction and analysis tools
        # (analyze_results.py, finding_sig and others) all read that key and
        # all want the reduced form; changing it would silently degrade every
        # one of them.  ``trace_original`` is added beside it, and only when
        # minimization actually removed something, so a finding that was never
        # minimized does not grow by a single byte.
        #
        # Why not recompute the bugs against the minimized trace instead?  That
        # needs a re-run (``reverify_minimized``, off by default, is exactly
        # that machinery) — it costs a server round-trip per finding, it is not
        # reliable for the nondeterministic classes that motivate this fuzzer,
        # and it would *discard* the evidence the oracle actually saw: the
        # logprob rows, output excerpts and restrictor sets on
        # ``probe_metadata`` came from the original workload and cannot be
        # regenerated from a trace that no longer contains the request.  The
        # mechanism signature is deliberately pre-minimization for the same
        # reason (see ``sig`` above), so keeping the pre-minimization trace also
        # makes the stored ``mech_sig`` re-derivable from the finding alone.
        #
        # Why unconditionally on shrink, rather than only when a bug names a
        # dropped request?  Measured over the four 735* campaign slots (238
        # findings, 2.5 MB): storing on every shrink touches 86 findings and
        # costs +1.1 MB (+43%); storing only on a detected orphan touches 12 and
        # costs +0.2 MB (+9%).  The cheaper rule was rejected because detecting
        # an orphan means substring-matching free-text ``detail``, so a bug class
        # that spells its request id differently loses its context permanently —
        # findings are the durable artifact and the corpus entry they came from
        # is mutated by later iterations.  1 MB per campaign is not worth a
        # silent, unrecoverable failure mode.
        #
        # ``trace_original`` / ``minimized`` / ``bugs_request_ids_not_in_trace``
        # are new optional keys.  Every existing reader goes through
        # ``.get``/``["trace"]``/``["bugs"]``, so archived findings need no
        # migration — they simply lack the keys, exactly like a finding that was
        # never minimized (verified: the offline dedup validator round-trips
        # 310/310 across earlier findings and the campaign, unchanged).
        if original_trace is not trace:
            original_dict = original_trace.to_dict()
            report["trace_original"] = original_dict
            report["minimized"] = True
            # Cheap triage aid: which request ids the bugs talk about that the
            # minimized trace no longer contains.
            min_rids = {e.get("request_id")
                        for e in (report["trace"].get("events") or [])}
            orphans = sorted(
                {rid for rid in (e.get("request_id")
                                 for e in (original_dict.get("events") or []))
                 if rid is not None and rid not in min_rids
                 and any(rid in (b.get("detail") or "")
                         or (b.get("probe_metadata") or {}).get("request_id") == rid
                         for b in report["bugs"])}
            )
            if orphans:
                report["bugs_request_ids_not_in_trace"] = orphans
        else:
            report["minimized"] = False

        # Which checks could NOT be evaluated for the session this finding came
        # from.  A reader of a single finding JSON otherwise has no way to tell
        # a genuinely clean session from one where half the oracles were blind.
        if self._iter_measurement_gaps:
            report["measurement_gaps"] = list(self._iter_measurement_gaps)
        if extra:
            report.update(extra)
        if annotations:
            failed_annotations = [a for a in annotations if not a.passed]
            if failed_annotations:
                report["output_annotations"] = [
                    {"class": a.bug_class.value, "detail": a.detail, "score": a.score}
                    for a in failed_annotations
                ]
        if self._collector is not None:
            report["collection_forensics"] = self._collector.forensics_snapshot(
                window_s=self.config.kv_events_window_s
            )
        # Embed KV probe summary if the adapter attached one to any finding
        for b in bugs:
            if b.probe_metadata and "kv_summary" in b.probe_metadata:
                report["kv_probe_summary"] = b.probe_metadata["kv_summary"]
                break  # all findings share the same probe window; first is enough
        # Counted here, not at the ledger check, so the paths that bail out
        # in between (lifecycle-only telemetry) do not burn an exemplar slot.
        rec["exemplars"] += 1
        self._finding_seq += 1
        path = (self.findings_dir /
                f"finding_{iteration:05d}_{trace.seed}_{self._finding_seq:04d}.json")
        if path.exists():
            print(f"[runner] WARNING: finding path already exists, overwriting: {path}")
        path.write_text(json.dumps(report, indent=2, default=str))
        self.events.emit("finding_saved", path=str(path.resolve()), finding=report)
        print(f"[runner] saved finding → {path}")
        return path

    def _dump_signature_ledger(self) -> None:
        """Write ``${findings_dir}/signature_ledger.json`` and print one summary line.

        The ledger is per slot: it is not reloaded at startup, so a genuine
        regression after a server restart still files its exemplars.  Merging
        the per-slot ledgers is the offline triage tool's job — which is why the
        dump carries the campaign/config identity and the normalized server
        flags of the slot that produced it.
        """
        try:
            ledger = {
                "sig_version":      SIG_VERSION,
                "generated_at":     time.time(),
                "campaign_id":      self._campaign_id,
                "campaign_name":    self.config.campaign_name,
                "config_id":        self.config.config_id,
                "slot_id":          self.config.slot_id,
                "rotation_index":   self.config.rotation_index,
                "model":            self.config.model_name,
                "server_flags":     normalize_server_flags(self.config.server_flags),
                "exemplars_per_signature": self.config.exemplars_per_signature,
                "n_mechanisms":     len(self._sig_ledger),
                "n_occurrences":    sum(r["count"] for r in self._sig_ledger.values()),
                "n_findings_written": sum(r["exemplars"] for r in self._sig_ledger.values()),
                # Checks that declined to judge, and how often.  Read this
                # beside the finding counts: a bug class with zero findings and
                # a large gap count was not exonerated, it was never measured.
                "measurement_gaps": dict(self._measurement_gaps),
                "signatures":       self._sig_ledger,
            }
            path = self.findings_dir / "signature_ledger.json"
            path.write_text(json.dumps(ledger, indent=2, default=str))
        except Exception as exc:                       # never fail a run over this
            print(f"[runner] signature ledger not written: {exc}")
            return

        top = max(self._sig_ledger.items(), key=lambda kv: kv[1]["count"],
                  default=None)
        top_str = f"{top[0]} \u00d7{top[1]['count']}" if top else "-"
        print(f"[runner] {ledger['n_findings_written']} findings written, "
              f"{ledger['n_mechanisms']} mechanisms, top: {top_str}  "
              f"({ledger['n_occurrences']} occurrences) → {path}")

        if self._measurement_gaps:
            total_gaps = sum(self._measurement_gaps.values())
            print(f"[runner] {total_gaps} measurement gaps "
                  f"({len(self._measurement_gaps)} distinct) — checks that "
                  f"could not be evaluated, NOT passes:")
            for reason, n in self._measurement_gaps.most_common(10):
                print(f"[runner]   {n:6d}  {reason}")

        # Accounting check: the ledger's exemplar count is what the runner
        # *believes* it filed; the glob is what actually survived.  A mismatch
        # means findings were lost (filename collision, write error, external
        # deletion) and used to go unnoticed until the analysis stage.
        try:
            on_disk = len(list(self.findings_dir.glob("finding_*.json")))
            if on_disk != ledger["n_findings_written"]:
                print(f"[runner] WARNING: ledger says "
                      f"{ledger['n_findings_written']} findings written but "
                      f"{on_disk} finding_*.json files are on disk in "
                      f"{self.findings_dir}")
        except Exception as exc:
            print(f"[runner] finding-count cross-check skipped: {exc}")

    def _record_recent_trace(self, trace: Trace, iteration: int) -> None:
        self._recent_traces.append(RecentTraceRecord(
            iteration=iteration,
            timestamp=time.time(),
            trace=trace,
        ))

    def _emit_verbose_log(self,
                          iteration: int,
                          trace: Trace,
                          mutation_ops: "list[str]",
                          kv_summary) -> None:
        """Emit one structured per-iteration log line for offline analysis.

        Format (single line, parser-friendly key=value fields):
            [NNNNN] verbose  mut=op1,op2  events=N/M  adapters=K  dur=Tms  \
                shape=D/S  kv=stored:S/removed:R/xlora:X/fast:F/conflict:C

        - mut: comma-joined mutation operator names from this iteration
        - events: total events / SEND-only count
        - adapters: distinct adapter names referenced by SEND events
        - dur: trace.duration_ms() (offset span of the trace template)
        - shape: distinct prompt_len count / stdev of prompt_len across SENDs
                 — proxies for "shape diversity" (one of the 4 SGLang LoRA
                 scheduler-crash trigger conditions: chunked-prefill boundary
                 stress from mixed prompt sizes within a batch)
        - kv: KVProbeSummary fields, omitted when collector is disabled
        """
        send_events = [e for e in trace.events if e.action == ActionType.SEND]
        send_count = len(send_events)
        adapter_set = {e.adapter_name for e in send_events if e.adapter_name}
        prompt_lens = [e.prompt_len for e in send_events if e.prompt_len]
        distinct_lens = len(set(prompt_lens))
        prompt_stdev  = (int(statistics.stdev(prompt_lens))
                         if len(prompt_lens) >= 2 else 0)
        mut_str = ",".join(mutation_ops) if mutation_ops else "-"
        parts = [
            f"[{iteration:05d}] verbose",
            f"mut={mut_str}",
            f"events={len(trace.events)}/{send_count}",
            f"adapters={len(adapter_set)}",
            f"dur={int(trace.duration_ms())}ms",
            f"shape={distinct_lens}/{prompt_stdev}",
        ]
        if kv_summary is not None and getattr(kv_summary, "fingerprint", ""):
            parts.append(
                "kv=stored:{s}/removed:{r}/xlora:{x}/fast:{f}/conflict:{c}".format(
                    s=kv_summary.stored_count,
                    r=kv_summary.removed_count,
                    x=kv_summary.cross_lora_reused_blocks,
                    f=kv_summary.reuse_too_fast_count,
                    c=kv_summary.hash_content_conflict_count,
                )
            )
        print("  ".join(parts))

    def _write_recent_trace_context(self, finding_path: Path) -> None:
        context_dir = finding_path.with_name(f"{finding_path.stem}_context")
        context_dir.mkdir(parents=True, exist_ok=True)

        records = list(self._recent_traces)
        manifest = {
            "finding_file": finding_path.name,
            "trace_count": len(records),
            "buffer_size": self._RECENT_TRACE_BUFFER_SIZE,
            "most_recent_iteration": records[-1].iteration if records else None,
            "traces": [
                {
                    "index": idx,
                    "file": f"trace_{idx:02d}.json",
                    "iteration": record.iteration,
                    "timestamp": record.timestamp,
                    "is_most_recent": idx == len(records) - 1,
                }
                for idx, record in enumerate(records)
            ],
        }
        (context_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str)
        )

        for idx, record in enumerate(records):
            payload = {
                "iteration": record.iteration,
                "timestamp": record.timestamp,
                "is_most_recent": idx == len(records) - 1,
                "trace": record.trace.to_dict(),
            }
            (context_dir / f"trace_{idx:02d}.json").write_text(
                json.dumps(payload, indent=2, default=str)
            )

    async def _refresh_baseline(self, iteration: int) -> None:
        if self._cold_start_baseline is None:
            return
        _ev = Event(action=ActionType.SEND, offset_ms=0,
                    request_id="baseline_refresh",
                    prompt_len=_BASELINE_PROMPT_LEN, max_tokens=1, stream=True)
        if self.adapter is not None:
            _ev = self.adapter.baseline_probe_event_overrides(_ev)
        probe = Trace(seed=0, events=[_ev])
        try:
            result = await self.executor.run(probe, self.config.base_url)
            for req in result.requests:
                ms = req.ttft_ms()
                if ms is not None:
                    self._baseline_window.append(ms)
                    if len(self._baseline_window) > 20:
                        self._baseline_window.pop(0)
                    fresh = statistics.median(self._baseline_window)
                    cap   = 2.0 * self._cold_start_baseline
                    self.oracle.baseline_ttft_ms = min(fresh, cap)
                    break
        except Exception:
            pass

    async def _measure_baseline_ttft(self) -> Optional[float]:
        _ev = Event(action=ActionType.SEND, offset_ms=0,
                    request_id="baseline_probe",
                    prompt_len=_BASELINE_PROMPT_LEN, max_tokens=16, stream=True)
        if self.adapter is not None:
            _ev = self.adapter.baseline_probe_event_overrides(_ev)
        probe = Trace(seed=0, events=[_ev])
        ttfts = []
        for _ in range(3):
            try:
                result = await self.executor.run(probe, self.config.base_url)
                for req in result.requests:
                    ms = req.ttft_ms()
                    if ms is not None:
                        ttfts.append(ms)
            except Exception:
                pass
        return statistics.median(ttfts) if ttfts else None

    # ------------------------------------------------------------------
    # FP tie-break banding for the determinism probe
    # ------------------------------------------------------------------
    #
    # The old rule was ``gap < _FP_GAP_THRESHOLD`` with the threshold a fixed
    # 1e-3 *in nats*.  It could never fire.  The top1-top2 logprob gap is
    # ``logit_1 - logit_2`` exactly (``log_softmax`` subtracts the same
    # ``logsumexp`` from both), and on the 0.27.1 path those logits reach the
    # sampler in the model's own dtype:
    #
    #   * ``--dtype auto`` resolves to the HF config's ``torch_dtype``
    #     (``vllm/config/model.py:2130,2174``); Qwen2.5-0.5B-Instruct declares
    #     ``"torch_dtype": "bfloat16"`` and qwen2 is not in
    #     ``_FLOAT16_NOT_SUPPORTED_MODELS``, so it stays bf16;
    #   * the LM head keeps that dtype (``config/model.py:2236-2242``) and
    #     ``logits_processor.py:105-108`` takes the no-cast branch;
    #   * ``gpu_model_runner.py:4484-4589`` hands the bf16 tensor straight to
    #     the sampler, which calls
    #     ``logits.log_softmax(dim=-1, dtype=torch.float32)``
    #     (``v1/sample/sampler.py:305-306``) — fp32 *arithmetic* over inputs
    #     that were already rounded to bf16.  ``logits.to(torch.float32)`` at
    #     ``sampler.py:95-96`` happens **after** the logprobs are taken.
    #   * ``VLLM_BATCH_INVARIANT=1`` (which the campaign sets) routes the head
    #     GEMM through ``batch_invariant.py:144-146``, whose output is still
    #     ``a.dtype`` — it makes the values reproducible, not more precise.
    #
    # So every achievable gap is an integer multiple of one bf16 ULP at the
    # logit magnitude.  Measured in the 0.27.1 image, Qwen2.5-0.5B's top logits
    # land in [8, 32), i.e. a quantum of 0.0625-0.25 — three orders of magnitude
    # *above* the old 1e-3 threshold, which is why all 45 tie findings in the
    # last campaign carried ``logprob_gap == 0.125`` exactly (one at
    # 0.12500011920928955: the same value plus one fp32 ULP of the ~1.0-magnitude
    # logprobs, 2**-23, from the per-row logsumexp subtraction).
    #
    # The threshold is therefore expressed in ULPs at a reference logit
    # magnitude, not in nats.

    #: Significand bits (explicit mantissa + the implicit leading 1) per dtype.
    #: bfloat16 is 1+8+7 → 8; float16 is 1+5+10 → 11; float32 → 24.
    _DTYPE_SIGNIFICAND_BITS: dict = {
        "bfloat16": 8, "bf16": 8,
        "float16": 11, "fp16": 11, "half": 11,
        "float32": 24, "fp32": 24, "float": 24,
    }

    #: Dtype assumed when ``server_flags`` carries no ``--dtype`` or says
    #: ``auto``.  ``auto`` means "the checkpoint's ``torch_dtype``", which is
    #: bfloat16 for every model this harness has been pointed at.
    _FP_DEFAULT_LOGITS_DTYPE: str = "bfloat16"

    #: Reference |logit| the ULP is evaluated at.  Measured for
    #: Qwen2.5-0.5B-Instruct in the 0.27.1 image: top logits sit in [8, 32), and
    #: the observed gap grid (multiples of 0.0625, dominated by 0.125) puts the
    #: top-2 pair in [16, 32) — ULP = 16 * 2**-7 = 0.125.  The margin here is
    #: asymmetric, which is why this is measured and not guessed: dropping to
    #: [8,16) leaves the suppress/keep split identical, but raising it to
    #: [32,64) makes one ULP 0.25 and the 2-ULP band reach exactly the
    #: narrowest genuine divergence in the data (0.5).
    _FP_LOGIT_MAGNITUDE: float = 16.0

    #: How many ULPs of divergence count as benign.  2 sits between the
    #: single-ULP ties that batch composition can swing (0.125) and the
    #: narrowest genuine divergence observed (0.5 = 4 ULPs), with the whole
    #: 2-4 ULP band empty across 63 measured findings.
    _FP_TIE_MAX_ULPS: float = 2.0

    @classmethod
    def _dtype_ulp(cls, dtype_name: str, magnitude: float) -> float:
        """One ULP of *dtype_name* at |x| = *magnitude*.

        For a binary float with ``p`` significand bits, values in
        ``[2**e, 2**(e+1))`` are spaced ``2**(e - (p-1))`` apart.  Unknown
        dtypes fall back to float32 (the finest of the three), which makes the
        suppression conservative rather than blanket.
        """
        bits = cls._DTYPE_SIGNIFICAND_BITS.get(
            str(dtype_name).strip().lower(),
            cls._DTYPE_SIGNIFICAND_BITS["float32"])
        m = abs(float(magnitude))
        if not math.isfinite(m) or m <= 0.0:
            m = 1.0
        return 2.0 ** (math.floor(math.log2(m)) - (bits - 1))

    def _logits_dtype(self) -> str:
        """The dtype the served model's logits are in, from ``--dtype``.

        Parsed out of ``RunnerConfig.server_flags`` (``--dtype X`` or
        ``--dtype=X``); ``auto`` and absence both mean the checkpoint's
        ``torch_dtype``, for which ``_FP_DEFAULT_LOGITS_DTYPE`` stands in.
        """
        flags = self.config.server_flags or ""
        m = re.search(r"--dtype[=\s]+([A-Za-z0-9_]+)", flags)
        if m:
            name = m.group(1).strip().lower()
            if name not in ("auto", ""):
                return name
        return self._FP_DEFAULT_LOGITS_DTYPE

    #: How many top-k logprob entries ``_prefix_equivalence_probe`` requests,
    #: on BOTH arms.
    #:
    #: The probe needs one number: the gap, on one arm, between the token that
    #: arm emitted and the token the *other* arm emitted, at the position where
    #: they first disagree.  That is only computable when the other arm's token
    #: is present in this arm's top-k, so k has to be big enough to contain it.
    #:
    #: For the class this is meant to adjudicate — a bf16 near-tie at the top of
    #: the logits row, where one ULP of batch/prefill noise swings the argmax —
    #: the two tokens are by definition ranked 1 and 2, so k=2 would suffice.
    #: k=5 buys margin for the case where a logit mask (bad_words, logit_bias,
    #: allowed_token_ids) has pushed the competitor a few ranks down, and it
    #: matches ``_determinism_probe``'s value so ``logprob_gap`` means the same
    #: thing on findings from both probes and the ULP band is calibrated once.
    #: Higher costs response bytes linearly per generated token on every probe
    #: request, with nothing to gain: a competitor outside the top 5 is not a
    #: near-tie under any reading.  vLLM caps the request at ``max_logprobs``,
    #: default 20 (``vllm/config/model.py:242``, 0.27.1), so 5 is safely legal
    #: and never turns into a 400 that would empty one arm's output.
    _PFX_PROBE_LOGPROBS: int = 5

    @staticmethod
    def _first_token_divergence(tokens_a: list, tokens_b: list) -> Optional[int]:
        """First index at which two token streams differ.

        None when either stream is missing (no logprobs were returned).  When
        one stream is a strict prefix of the other the answer is the length of
        the shorter one — the position where one arm stopped and the other did
        not is a divergence.
        """
        if not tokens_a or not tokens_b:
            return None
        for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
            if a != b:
                return i
        if len(tokens_a) != len(tokens_b):
            return min(len(tokens_a), len(tokens_b))
        return None

    @staticmethod
    def _cross_arm_gap(own_tops: Optional[dict],
                       other_token: Optional[str]) -> tuple:
        """Logprob distance from this arm's top-1 to the other arm's token.

        Returns ``(gap, basis)``:

        * ``("cross_arm")`` — the other arm's token is in this arm's top-k, so
          the gap is EXACT: it is precisely how far this arm's argmax would have
          had to move to produce the other arm's token.  Only an exact gap is
          allowed to suppress a finding.
        * ``("topk_bound")`` — the other arm's token is outside this arm's
          top-k.  Then ``top1 - min(top-k)`` is a strict LOWER bound on the true
          cross-arm gap.  A lower bound may not suppress: ``bound <= threshold``
          does not imply ``true_gap <= threshold``, and suppressing on it would
          silently discard exactly the findings where the competitor was ranked
          far down, i.e. the interesting ones.  Recorded, never acted on.
        * ``(None, None)`` — no usable top-k dict at that position.
        """
        if not own_tops or len(own_tops) < 2:
            return None, None
        top1 = max(own_tops.values())
        if other_token is not None and other_token in own_tops:
            return top1 - own_tops[other_token], "cross_arm"
        return top1 - min(own_tops.values()), "topk_bound"

    def _pfx_pair_divergence(self, ra, rb, name_a: str, name_b: str) -> Optional[dict]:
        """Locate and measure the divergence between two arms of the probe.

        Returns None when the two outputs are identical.  Otherwise a dict with
        the character index, the token index, and the cross-arm logprob gap at
        that position — the same shape for every pair, so A-vs-B and B-vs-C are
        measured by one piece of code and are directly comparable.
        """
        text_a, text_b = ra.output_text, rb.output_text
        if text_a == text_b:
            return None

        char_idx = next(
            (i for i, (x, y) in enumerate(zip(text_a, text_b)) if x != y),
            min(len(text_a), len(text_b)),
        )
        # Both arms requested logprobs, so both carry a token list and the
        # divergence index is read off directly by comparing the two token
        # streams — no char-offset-to-token remapping and no dependence on
        # detokenisation being a plain concatenation.  Fall back to the remap
        # only if one arm returned no token data (older server, or a build that
        # dropped logprobs).
        toks_a = ra.output_tokens or []
        toks_b = rb.output_tokens or []
        token_idx = self._first_token_divergence(toks_a, toks_b)
        if token_idx is None and toks_b:
            offset = 0
            token_idx = len(toks_b)
            for ti, tok in enumerate(toks_b):
                tok_len = len(tok) if tok else 0
                if offset + tok_len > char_idx or (tok_len == 0 and offset >= char_idx):
                    token_idx = ti
                    break
                offset += tok_len

        # How far apart were the two arms' tokens, in each arm's own logprob
        # landscape, at the position where they first disagree?  Measured on
        # both arms; the SMALLER gap wins, because a near-tie on either arm is
        # enough to explain the flip with one ULP of noise.
        tops_a = ra.output_top_logprobs or []
        tops_b = rb.output_top_logprobs or []
        gap = gap_basis = gap_arm = None
        if token_idx is not None:
            tok_a = toks_a[token_idx] if token_idx < len(toks_a) else None
            tok_b = toks_b[token_idx] if token_idx < len(toks_b) else None
            cands = []
            if token_idx < len(tops_a):
                g, basis = self._cross_arm_gap(tops_a[token_idx], tok_b)
                if g is not None:
                    cands.append((g, basis, name_a))
            if token_idx < len(tops_b):
                g, basis = self._cross_arm_gap(tops_b[token_idx], tok_a)
                if g is not None:
                    cands.append((g, basis, name_b))
            if cands:
                gap, gap_basis, gap_arm = min(cands, key=lambda c: c[0])

        ulp = self._dtype_ulp(self._logits_dtype(), self._FP_LOGIT_MAGNITUDE)
        return {
            "char_idx": char_idx,
            "token_idx": token_idx,
            "gap": gap,
            "gap_basis": gap_basis,
            "gap_arm": gap_arm,
            "gap_ulps": (gap / ulp if gap is not None and ulp > 0 else None),
            # Only an EXACT cross-arm gap may license suppression.  A
            # ``topk_bound`` gap is a lower bound, so a small value does not
            # imply the true gap is small — see ``_cross_arm_gap``.
            "fp_tie": (gap is not None and gap_basis == "cross_arm"
                       and gap <= self._fp_tie_threshold()),
        }

    def _fp_tie_threshold(self) -> float:
        """Largest top1-top2 logprob gap still explainable by FP noise."""
        ulp = self._dtype_ulp(self._logits_dtype(), self._FP_LOGIT_MAGNITUDE)
        # +1 fp32 ULP of slack at logprob magnitude ~1: the fp32 logsumexp
        # subtraction perturbs the exact bf16 difference by that much, which is
        # how 0.125 came back as 0.12500011920928955 once in 45 samples.
        return self._FP_TIE_MAX_ULPS * ulp + 2.0 ** -23

    async def _determinism_probe(self,
                                  trace: Trace,
                                  first_result: TraceResult,
                                  iteration: int) -> None:
        """Re-run the trace with logprobs=5 and compare outputs.

        For each diverged request, check the top-1 vs top-2 logprob gap at
        the concurrent diverge position.  A gap within ``_FP_TIE_MAX_ULPS``
        ULPs of the logits dtype (see ``_fp_tie_threshold``) means the
        divergence is explainable by batch-composition FP noise; a wider gap
        means a real bug (KV corruption, spec decode state leak, etc.).

        No separate solo run needed — the logprob landscape at the diverge
        position is a model+prompt property visible in any execution context.
        """
        import copy as _copy

        probe_trace = _copy.deepcopy(trace)
        for ev in probe_trace.events:
            if ev.action == ActionType.SEND:
                ev.logprobs = max(ev.logprobs or 0, 5)
        try:
            second_result = await self.executor.run(probe_trace, self.config.base_url)
        except Exception as exc:
            print(f"[{iteration:05d}] determinism probe error: {exc}")
            return

        first_by_id  = {r.request_id: r for r in first_result.requests}
        second_by_id = {r.request_id: r for r in second_result.requests}

        for rid, r2 in second_by_id.items():
            r1 = first_by_id.get(rid)
            if r1 is None:
                continue
            if r1.cancelled_at is not None or r2.cancelled_at is not None:
                continue
            if getattr(r1, 'temperature', 0.0) > 0 or getattr(r2, 'temperature', 0.0) > 0:
                continue
            if getattr(r1, 'n_completions_requested', None) not in (None, 1):
                continue
            if r1.output_text is None or r2.output_text is None:
                continue
            finding = Oracle.check_determinism(r2.output_text, r1.output_text)
            if finding.passed:
                continue

            sim = (finding.probe_metadata or {}).get("similarity", 0.0)

            # Use result_2's top_logprobs to check tie at the diverge position.
            # result_1 may not have token-level data (no logprobs in first run),
            # so find the character-level diverge position and map it to a token
            # index using result_2's token list.
            top_logprobs2 = r2.output_top_logprobs or []
            tokens2 = r2.output_tokens or []
            text1 = r1.output_text or ""
            text2 = r2.output_text or ""

            token_div = None
            gap = None
            n_ulps = None
            is_tie = False
            ulp = self._dtype_ulp(self._logits_dtype(), self._FP_LOGIT_MAGNITUDE)
            if tokens2 and top_logprobs2:
                # Find first differing character position
                char_div = next(
                    (i for i, (a, b) in enumerate(zip(text1, text2)) if a != b),
                    min(len(text1), len(text2)),
                )
                # Map character position to token index in result_2
                char_offset = 0
                token_div = len(tokens2)  # fallback: past end
                for ti, tok in enumerate(tokens2):
                    tok_len = len(tok) if tok else 0
                    if char_offset + tok_len > char_div or (tok_len == 0 and char_offset >= char_div):
                        token_div = ti
                        break
                    char_offset += tok_len

                if token_div < len(top_logprobs2):
                    top_dict = top_logprobs2[token_div]
                    # Non-finite logprobs (the int4 KV NaN bug returns None/NaN
                    # here) are not orderable and are not a tie to suppress: the
                    # feature-failure oracle already owns them, so file rather
                    # than skip.
                    finite_lps = ([v for v in top_dict.values()
                                   if isinstance(v, (int, float)) and v == v]
                                  if top_dict else [])
                    if top_dict and len(top_dict) >= 2 and len(finite_lps) >= 2:
                        sorted_lps = sorted(finite_lps, reverse=True)
                        gap = sorted_lps[0] - sorted_lps[1]
                        n_ulps = gap / ulp if ulp > 0 else None
                        if gap <= self._fp_tie_threshold():
                            if not self.config.confirm_ladder:
                                print(f"[{iteration:05d}] FP tie-break suppressed "
                                      f"request={rid}  similarity={sim:.2f}  "
                                      f"gap={gap:.6f} ({n_ulps:.2f} ULP of "
                                      f"{self._logits_dtype()} @ |logit|="
                                      f"{self._FP_LOGIT_MAGNITUDE:g})  "
                                      f"pos={token_div}")
                                continue
                            # With the ladder on, the logit screen only sets
                            # priority: a tie is kept for confirmation, after
                            # every non-tie candidate.
                            is_tie = True

            # Keep the diverge shape on the finding: it is the only thing that
            # separates two kv_corruption findings once `similarity` saturates
            # at 1.00 (it is a word-set Jaccard, blind to order).
            finding.probe_metadata = dict(finding.probe_metadata or {})
            finding.probe_metadata["diverge_token_idx"] = token_div
            finding.probe_metadata["logprob_gap"] = gap
            # Gap in ULPs of the logits dtype — the unit the tie-break is
            # actually banded in.  ``logprob_gap`` stays as-is so
            # ``finding_sig.oracle_subtype``'s gtie/gwide split and every
            # archived finding keep their meaning.
            finding.probe_metadata["logprob_gap_ulps"] = n_ulps
            finding.probe_metadata["logits_dtype"] = self._logits_dtype()
            finding.probe_metadata["diverge_bucket"] = (
                log2_bucket(token_div) if token_div is not None else None
            )

            if self.config.confirm_ladder:
                from ..confirm.ladder import PRIORITY_HIGH, PRIORITY_LOW
                self._enqueue_confirmation(
                    report_trace=trace, replay_trace=trace, rid=rid,
                    source="determinism", bugs=[finding], iteration=iteration,
                    priority=PRIORITY_LOW if is_tie else PRIORITY_HIGH,
                    extra={"request_id": rid},
                    logprob_gap=gap, logprob_gap_ulps=n_ulps)
                continue

            print(f"[{iteration:05d}] KV corruption (hard) "
                  f"request={rid}  similarity={sim:.2f}")
            # One iteration can file several findings (one per diverging
            # request); record which request this one is about so the JSONs
            # stay distinguishable after the fact.
            await self._handle_finding(trace, [finding], iteration,
                                       extra={"request_id": rid})

    # ------------------------------------------------------------------
    # Two-path relational oracle (prefix-cache equivalence)
    # ------------------------------------------------------------------

    #: Suffix appended to a SEND's request_id to name its salted twin (arm B).
    _PFX_TWIN_SUFFIX: str = "~pfx"

    #: Suffix for the SECOND salted twin (arm C), the negative control.
    #: B and C carry two *different* salts, so neither can hit the unsalted
    #: prefix-cache blocks and neither can hit the other's: both are pure
    #: recomputes of a byte-identical prompt and must agree exactly.
    _PFX_CTRL_SUFFIX: str = "~pfx2"

    @staticmethod
    def _is_greedy_event(ev: Event) -> bool:
        """Greedy-decoding guard, mirroring ``Trace.is_deterministic`` per event.

        Same predicate ``_determinism_probe`` applies to its RequestResults, but
        applied to the Event because the twin is synthesized before any request
        exists.  n>1 is excluded as well: ``output_text`` is \x1f-joined for
        n>1 and the join order is not part of the contract being tested.
        """
        if ev.action != ActionType.SEND:
            return False
        if ev.temperature not in (None, 0.0):
            return False
        if ev.top_k not in (None, 1):
            return False
        if ev.top_p not in (None, 1.0):
            return False
        if ev.n_completions not in (None, 1):
            return False
        return True

    def _make_cache_salt(self, request_id: str) -> str:
        """A fresh cache_salt for one twin, drawn from the seeded probe RNG.

        Deterministic given ``RunnerConfig.seed`` and the order in which twins
        are created, so a rerun of the same campaign replays the same salts.
        """
        return "grief-%s-%08x" % (request_id[:8], self._pfx_rng.getrandbits(32))

    async def _prefix_equivalence_probe(self,
                                        trace: Trace,
                                        first_result: TraceResult,
                                        iteration: int) -> None:
        """Three-arm relational oracle: same prompt, three prefix-cache paths.

        ``_determinism_probe`` re-runs the whole trace and must tie-break on the
        top1-top2 logprob gap, because the second run's batch composition differs
        from the first's: batch nondeterminism is confounded with corruption.
        This probe removes *that* confound.  For every greedy SEND it adds two
        *twin* requests to the SAME trace — byte-identical prompt (pinned via
        ``prompt_family_id``), same offset, same sampling params — differing from
        the original and from each other in exactly one field, ``cache_salt``:

            A = plain   (no salt)   reads the prefix cache
            B = salt1               cannot hit A's blocks → recompute
            C = salt2               cannot hit A's or B's blocks → recompute

        Two arms would not be enough.  ``salt_verified`` establishes that B
        bypassed the cache; it does NOT establish that the cache is the *only*
        difference between A and B.  ``cache_salt`` goes into the request body
        and changes block hashes, so an A≠B divergence has two readings: a real
        cache-read-vs-recompute violation, or an artifact the instrument itself
        creates.  C separates them, and being a pure recompute of the same
        prompt it is also the negative control that a separate "prefix caching
        globally off" campaign slot would have provided:

          * ``B != C``  → two recomputes of a byte-identical prompt disagreed.
            The engine is nondeterministic independently of the prefix cache, or
            the probe is at fault.  Either way the A/B comparison is invalid, so
            this is filed as ``BugClass.PROBE_SELF_CHECK`` — never as
            ``kv_corruption``, which would assert something about the cache that
            the data does not support.
          * ``B == C`` and ``A != B`` → the two recomputes agree with each other
            and disagree with the cached read.  The prefix cache is implicated.
            This is the finding the probe exists to produce.
          * all three equal → clean.

        None of that removes the floating-point confound, and an earlier
        revision of this docstring claiming the probe "needs no logprob
        tie-break" was wrong.  Reading a prefix from cache and recomputing it
        are different numerical paths — different prefill chunking, different
        attention batch shapes — so the arms' logits differ in the low bits by
        construction.  Greedy ``argmax`` is not a stable function of logits that
        differ in the low bits: at a near-tie, a benign one-ULP difference and a
        genuine cache corruption produce the same symptom, a token flip followed
        by a similarity collapse.  Text comparison alone therefore detects
        *that* something differed and can never say *whether it matters* — which
        is why all three arms request ``_PFX_PROBE_LOGPROBS`` top-k entries and
        every divergence carries its cross-arm gap, banded with the same ULP
        threshold ``_determinism_probe`` uses.

        False-positive control:
          * greedy only (``_is_greedy_event``) — benign sampling divergence is
            excluded by construction;
          * any arm erroring / being cancelled → skip (a server that rejects
            ``cache_salt`` shows up here as a per-request 4xx, not a finding);
          * VACUITY: the salted arms must actually have bypassed the cache, and
            the plain arm must actually have hit it.  When the arms report
            ``cache_read_tokens`` this is checked and a failure is dropped
            outright, because a comparison between arms that took the same path
            proves nothing about the cache path.  See the gate below for what
            happens when the server does not report the counts at all.
          * SELF-CHECK: ``B != C`` invalidates the A/B comparison — reported,
            but never as a cache finding.
          * FP TIE: when the A-vs-B cross-arm logprob gap at the divergence
            position is exactly known and lies within ``_fp_tie_threshold``, the
            flip is explainable by one ULP of numerical noise and is not filed.
        False *negative* accepted: a common-mode failure that corrupts all three
        paths identically is invisible to this probe (differential-testing's
        standing limitation) — ``_determinism_probe`` still covers that case,
        which is why both probes run.

        Cost: one extra request per probe firing on top of the two the probe
        already sent.  See the module notes on ``prefix_equiv_probe_every``.
        """
        import copy as _copy

        probe_trace = _copy.deepcopy(trace)
        # original_rid -> (twin_b_rid, salt_b, twin_c_rid, salt_c)
        twins: dict[str, tuple[str, str, str, str]] = {}
        new_events: list[Event] = []
        for ev in probe_trace.events:
            if not self._is_greedy_event(ev):
                continue
            # Pin the prompt identity to the ORIGINAL request_id so all three
            # events render byte-identical prompts (adapters seed the prompt
            # from prompt_family_id or, failing that, request_id) — and so the
            # unsalted arm's prompt is unchanged from the run being probed.
            family = ev.prompt_family_id or ev.request_id
            ev.prompt_family_id = family
            # SYMMETRY.  The salt is the only variable under test, so every
            # other knob must be set on ``ev`` BEFORE the twins are copied off
            # it — the copies then carry the identical values by construction
            # and no later edit can desynchronise the arms.  This is deliberate
            # contrast with ``_determinism_probe``, which raises ``logprobs`` on
            # its second run only and so varies two things at once.
            #   logprobs      — needed on ALL arms: the adjudication is a
            #                   cross-arm gap, which requires each arm's top-k
            #                   at the divergence position.
            #   include_usage — needed on ALL arms: without it a streamed
            #                   response carries no usage block, so
            #                   ``cache_read_tokens`` is None and the vacuity
            #                   control below cannot run.  No-op when the event
            #                   is non-streaming.
            ev.logprobs = max(ev.logprobs or 0, self._PFX_PROBE_LOGPROBS)
            ev.include_usage = True

            # Both salts come from the same seeded ``_pfx_rng`` in a fixed
            # order (B then C, events in trace order), so the whole three-arm
            # trace replays identically from ``RunnerConfig.seed``.
            arms = []
            for suffix, origin in ((self._PFX_TWIN_SUFFIX, "pfx_twin"),
                                   (self._PFX_CTRL_SUFFIX, "pfx_ctrl")):
                twin = _copy.deepcopy(ev)
                twin.request_id = ev.request_id + suffix
                twin.prompt_family_id = family
                twin.cache_salt = self._make_cache_salt(ev.request_id)
                twin.origin = origin
                new_events.append(twin)
                arms.append((twin.request_id, twin.cache_salt))
            # Two salts drawn from the same stream can only collide on a 2**-32
            # coincidence; if they ever do, B and C are the same namespace and
            # the negative control is meaningless, so skip the whole request.
            if arms[0][1] == arms[1][1]:
                new_events = new_events[:-2]
                continue
            twins[ev.request_id] = (arms[0][0], arms[0][1],
                                    arms[1][0], arms[1][1])

        if not twins:
            return
        probe_trace.events.extend(new_events)

        try:
            probe_result = await self.executor.run(probe_trace, self.config.base_url)
        except Exception as exc:
            print(f"[{iteration:05d}] prefix-equiv probe error: {exc}")
            return

        by_id = {r.request_id: r for r in probe_result.requests}
        orig_by_id = {r.request_id: r for r in first_result.requests}

        for rid, (rid_b, salt_b, rid_c, salt_c) in twins.items():
            arms = {"plain": by_id.get(rid),
                    "salt1": by_id.get(rid_b),
                    "salt2": by_id.get(rid_c)}
            if any(r is None for r in arms.values()):
                continue
            if any(r.cancelled_at is not None for r in arms.values()):
                continue
            if any(r.error is not None for r in arms.values()):
                # Includes the "this build does not accept cache_salt" case.
                continue
            if any(r.output_text is None for r in arms.values()):
                continue
            if any(getattr(r, "temperature", 0.0) > 0 for r in arms.values()):
                continue
            plain, salted, ctrl = arms["plain"], arms["salt1"], arms["salt2"]

            # ---- Vacuity control -------------------------------------------
            # The probe's whole claim is "these arms took different cache
            # paths".  If they did not, a divergence between them says nothing
            # about the cache and the finding is vacuous.  Two ways to fail:
            #   * the PLAIN arm read nothing from the prefix cache — then no
            #     arm exercised the cache read path at all and any divergence is
            #     ordinary run-to-run nondeterminism, which is
            #     ``_determinism_probe``'s job, not this one;
            #   * a SALTED arm read as much as the plain arm — the salt failed
            #     to namespace anything (server ignored the field, prefix
            #     caching off, prompt shorter than one block) and the arms are
            #     literally the same path.
            # Both salted arms are checked: C is only a negative control if it
            # too recomputed.
            # Checkable only when the server reports the counts, which on vLLM
            # means it was launched with --enable-prompt-tokens-details AND (for
            # streamed requests) the arm asked for usage — see
            # ``Event.include_usage``, set on all three arms above.
            crp = plain.cache_read_tokens
            crs = salted.cache_read_tokens
            crc = ctrl.cache_read_tokens
            salt_verified: Optional[bool] = None
            if crp is not None and crs is not None and crc is not None:
                salt_verified = bool(crp > 0 and crs < crp and crc < crp)
                if not salt_verified:
                    # Vacuous: never filed.  A finding here would be an
                    # assertion about a cache path that no arm took.
                    why = ("plain arm never hit the cache" if crp <= 0
                           else "a salt did not bypass")
                    print(f"[{iteration:05d}] prefix-equiv vacuous "
                          f"request={rid}  cache_read plain={crp} "
                          f"salt1={crs} salt2={crc} ({why})")
                    continue
            elif not self._pfx_vacuity_warned:
                # salt_verified is None: the server never reported the counts,
                # so the control could not run.  Chosen behaviour is to FILE
                # ANYWAY, tagged, rather than suppress.  Suppressing on None
                # would turn one missing server flag into a probe that silently
                # produces nothing for a whole campaign with no signal that
                # anything is wrong — the same silent-failure class this gate
                # exists to close.  "Unverified" is not "verified false"; the
                # finding is downgraded and labelled instead, and every slot
                # script that runs this probe now passes the flag, so in
                # practice None means an older server.
                self._pfx_vacuity_warned = True
                print(f"[{iteration:05d}] prefix-equiv: server reports no "
                      f"prompt_tokens_details.cached_tokens — vacuity control "
                      f"is OFF for this campaign; findings will carry "
                      f"salt_verified=null and a reduced score.  Launch vLLM "
                      f"with --enable-prompt-tokens-details to enable it.")

            # ---- Three-way comparison --------------------------------------
            # B vs C first: two salted arms are both pure recomputes of a
            # byte-identical prompt, so they must agree.  If they do not, the
            # A/B comparison is not interpretable and nothing below it may be
            # filed as a cache finding.
            bc = self._pfx_pair_divergence(salted, ctrl, "salt1", "salt2")
            ab = self._pfx_pair_divergence(plain, salted, "plain", "salt1")

            if bc is None and ab is None:
                continue                                  # all three agree

            common = {
                "probe": "prefix_equiv",
                "salt_verified": salt_verified,
                # "verified" / "failed" is never reached here (a failed control
                # `continue`s above); "unavailable" is the flag an analyst
                # filters on to find campaigns run blind.
                "vacuity_control": ("verified" if salt_verified
                                    else "unavailable"),
                "cache_salt": salt_b,
                "cache_salt_ctrl": salt_c,
                "cache_read_plain": crp,
                "cache_read_salted": crs,
                "cache_read_ctrl": crc,
                "logits_dtype": self._logits_dtype(),
                "logprobs_requested": self._PFX_PROBE_LOGPROBS,
                # The B-vs-C pair, always reported: a 1-ULP B/C gap and a 20-ULP
                # B/C gap say very different things about how far the probe's
                # own premise is from holding.
                "bc_diverged": bc is not None,
                "bc_logprob_gap": (bc or {}).get("gap"),
                "bc_logprob_gap_basis": (bc or {}).get("gap_basis"),
                "bc_logprob_gap_ulps": (bc or {}).get("gap_ulps"),
                "bc_diverge_token_idx": (bc or {}).get("token_idx"),
                "bc_diverge_char_idx": (bc or {}).get("char_idx"),
                "bc_similarity": (
                    None if bc is None else
                    Oracle._text_similarity(ctrl.output_text, salted.output_text)
                ),
            }

            if bc is not None:
                # ---- Instrument self-check failure -------------------------
                # Two recomputes of the same prompt disagreed.  This is NOT a
                # cache finding: it says the engine is nondeterministic
                # independently of the prefix cache, or that the probe itself
                # perturbs the result.  Filed under its own bug class so it can
                # never merge with kv_corruption in the ledger, and so a
                # campaign where the premise fails is visible instead of
                # producing a stream of confident-looking cache findings.
                sim_bc = common["bc_similarity"]
                finding = OracleResult(
                    passed=False,
                    bug_class=BugClass.PROBE_SELF_CHECK,
                    detail="Prefix-equivalence self-check failed: two "
                           "differently-salted recomputes of a byte-identical "
                           "greedy prompt disagreed, so the cached-vs-recompute "
                           f"comparison is invalid (similarity={sim_bc:.2f})",
                    # Below any real finding: this reports on the instrument,
                    # not on the server under test.
                    score=4.0,
                    probe_metadata=dict(
                        common,
                        control_verdict="b_ne_c",
                        similarity=sim_bc,
                        # The primary shape fields describe the B/C divergence,
                        # since that is what this finding is about.
                        diverge_char_idx=bc["char_idx"],
                        diverge_token_idx=bc["token_idx"],
                        diverge_bucket=log2_bucket(bc["char_idx"]),
                        logprob_gap=bc["gap"],
                        logprob_gap_basis=bc["gap_basis"],
                        logprob_gap_ulps=bc["gap_ulps"],
                        logprob_gap_arm=bc["gap_arm"],
                        # What A/B would have said, kept for triage but not
                        # acted on — it is exactly the claim this finding
                        # invalidates.
                        ab_diverged=ab is not None,
                        ab_logprob_gap=(ab or {}).get("gap"),
                        ab_logprob_gap_basis=(ab or {}).get("gap_basis"),
                        ab_diverge_token_idx=(ab or {}).get("token_idx"),
                    ),
                )
                print(f"[{iteration:05d}] prefix-equiv SELF-CHECK FAILED "
                      f"request={rid}  salt1 != salt2  similarity={sim_bc:.2f}  "
                      f"gap={'n/a' if bc['gap'] is None else format(bc['gap'], '.6f')}"
                      f"[{bc['gap_basis'] or 'none'}]  "
                      f"(A/B divergence {'present' if ab else 'absent'}, not filed)")
                pe_extra = {"request_id": rid, "probe": "prefix_equiv",
                            "control_verdict": "b_ne_c"}
                if self.config.confirm_ladder:
                    # B and C are both recomputes; the ladder asks whether
                    # they still disagree under BI.
                    self._enqueue_confirmation(
                        report_trace=trace, replay_trace=probe_trace,
                        rid=rid_b, twin_rid=rid_c, twin_needs_cache_hit=False,
                        source="prefix_equiv", bugs=[finding],
                        iteration=iteration, extra=pe_extra,
                        logprob_gap=bc["gap"], logprob_gap_ulps=bc["gap_ulps"])
                else:
                    await self._handle_finding(trace, [finding], iteration,
                                               extra=pe_extra)
                continue

            # ---- B == C, A != B: the prefix cache is implicated -------------
            if ab["fp_tie"] and not self.config.confirm_ladder:
                print(f"[{iteration:05d}] prefix-equiv FP tie-break suppressed "
                      f"request={rid}  gap={ab['gap']:.6f} "
                      f"({ab['gap_ulps']:.2f} ULP of {self._logits_dtype()} @ "
                      f"|logit|={self._FP_LOGIT_MAGNITUDE:g})  "
                      f"pos={ab['token_idx']}")
                continue

            sim = Oracle._text_similarity(salted.output_text, plain.output_text)

            # Which side moved?  Comparing both against the unsalted output of
            # the run that triggered the probe says whether the cached path or
            # the recomputed path is the odd one out.
            baseline = orig_by_id.get(rid)
            base_text = baseline.output_text if baseline is not None else None
            if base_text is not None and plain.output_text == base_text:
                diverged_path = "salted"
            elif base_text is not None and salted.output_text == base_text:
                diverged_path = "cached"
            else:
                diverged_path = "both"

            finding = OracleResult(
                passed=False,
                bug_class=BugClass.KV_CORRUPTION,
                detail="Prefix-cache path divergence: identical greedy prompt "
                       "produced different output when routed around the prefix "
                       f"cache via cache_salt (similarity={sim:.2f}); two "
                       "independently salted recomputes agreed with each other",
                # Full score only when the vacuity control actually ran and
                # passed.  Unverified (server does not report the counts) is
                # still filed — see the gate above — but must not outrank a
                # finding whose control confirmed the arms took different cache
                # paths.
                score=12.0 if salt_verified else 8.0,
                probe_metadata=dict(
                    common,
                    # B == C and A != B.  Recorded explicitly so triage reads
                    # the three-way outcome instead of re-deriving it.
                    control_verdict="cache_implicated",
                    similarity=sim,
                    diverged_path=diverged_path,
                    diverge_char_idx=ab["char_idx"],
                    diverge_token_idx=ab["token_idx"],
                    diverge_bucket=log2_bucket(ab["char_idx"]),
                    # Distance from one arm's top-1 to the OTHER arm's token at
                    # the divergence position — how far the argmax had to move.
                    # "cross_arm" = exact; "topk_bound" = lower bound only (the
                    # other arm's token fell outside the top-k, so the true gap
                    # is at least this); None = no logprob data came back.
                    # Anything that survived to here is either wider than the FP
                    # band or was never exactly measurable.
                    logprob_gap=ab["gap"],
                    logprob_gap_basis=ab["gap_basis"],
                    logprob_gap_ulps=ab["gap_ulps"],
                    logprob_gap_arm=ab["gap_arm"],
                ),
            )
            print(f"[{iteration:05d}] KV corruption (prefix-equiv) "
                  f"request={rid}  similarity={sim:.2f}  path={diverged_path}  "
                  f"gap={'n/a' if ab['gap'] is None else format(ab['gap'], '.6f')}"
                  f"[{ab['gap_basis'] or 'none'}]  salt_verified={salt_verified}"
                  f"  control=b_eq_c")
            pe_extra = {"request_id": rid, "probe": "prefix_equiv",
                        "control_verdict": "cache_implicated"}
            if self.config.confirm_ladder:
                # A read its prefix from the cache, B recomputed it; the
                # relation is only tested if a BI replay reads it again.
                from ..confirm.ladder import PRIORITY_HIGH, PRIORITY_LOW
                self._enqueue_confirmation(
                    report_trace=trace, replay_trace=probe_trace,
                    rid=rid, twin_rid=rid_b, twin_needs_cache_hit=True,
                    source="prefix_equiv", bugs=[finding],
                    iteration=iteration, extra=pe_extra,
                    priority=PRIORITY_LOW if ab["fp_tie"] else PRIORITY_HIGH,
                    logprob_gap=ab["gap"], logprob_gap_ulps=ab["gap_ulps"])
                continue
            await self._handle_finding(trace, [finding], iteration,
                                       extra=pe_extra)

    # ------------------------------------------------------------------
    # Soft-suspicion batch reproduction
    # ------------------------------------------------------------------

    def _soft_candidate_sig(self, candidate: SoftCandidate) -> str:
        """Structural signature for dedup and cooldown keying.

        Hashes the event shapes of the trace (action, adapter, prompt, max_tokens,
        prefix_len) so candidates from the same trace family collapse to one sig.
        The trigger_request_id is intentionally NOT included — two candidates from
        the same trace shape but different trigger requests belong to the same family.
        """
        events = tuple(
            (
                e.action.value if hasattr(e.action, "value") else str(e.action),
                e.adapter_name,
                e.prompt_len,
                e.max_tokens,
                e.prefix_len,
                getattr(e, "prompt_family_id", None),
            )
            for e in candidate.trace.events
        )
        key = repr(("soft", events))
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _pick_baseline_event(self, trace: Trace, trigger_rid: str) -> Optional[Event]:
        """Pick the best isolated SEND event to use as a baseline probe.

        Scoring (higher = better candidate):
          +2.0  prefix_len > 0  (cache-relevant, targets the actual phenomenon)
          -cost/1000  lower replay cost preferred
          +1.0  origin in (None, "seed", "mutated")  — non-synthetic
          -0.5  origin in ("burst", "thrash", "storm", "retry") — synthetic penalty
          +1.5  request_id == trigger_rid  — the triggering request is preferred
        """
        sends = [e for e in trace.events if e.action == ActionType.SEND]
        if not sends:
            return None

        def _score(ev: Event) -> float:
            s = 0.0
            if (ev.prefix_len or 0) > 0:
                s += 2.0
            cost = (ev.prompt_len or 64) + (ev.max_tokens or 32)
            s -= cost / 1000.0
            origin = getattr(ev, "origin", None)
            if origin in (None, "seed", "mutated"):
                s += 1.0
            elif origin in ("burst", "thrash", "storm", "retry"):
                s -= 0.5
            if ev.request_id == trigger_rid:
                s += 1.5
            return s

        return max(sends, key=_score)

    async def _maybe_trigger_soft_repro(self) -> None:
        """Check whether the soft queue should trigger a batch reproduction run."""
        if not self._soft_queue:
            return
        elapsed_since_last = time.time() - self._last_soft_repro_at
        should_trigger = (
            len(self._soft_queue) >= self.config.soft_repro_batch_size
            or elapsed_since_last > self.config.soft_repro_interval_min * 60
        )
        if should_trigger:
            await self._run_soft_repro_batch()

    # ------------------------------------------------------------------
    # Structural KV anomaly batch reproduction
    # ------------------------------------------------------------------

    def _structural_candidate_sig(self, trace, anomaly_type: str) -> str:
        """Dedup / cooldown key: structural anomaly type + trace event shapes."""
        events = tuple(
            (
                e.action.value if hasattr(e.action, "value") else str(e.action),
                e.adapter_name,
                e.prompt_len,
                e.max_tokens,
                e.prefix_len,
                getattr(e, "prompt_family_id", None),
            )
            for e in trace.events
        )
        key = repr(("structural", anomaly_type, events))
        return hashlib.md5(key.encode()).hexdigest()[:16]

    async def _maybe_trigger_structural_repro(self) -> None:
        """Check whether the structural queue should trigger a confirmation run."""
        if not self._structural_queue:
            return
        elapsed = time.time() - self._last_structural_repro_at
        should_trigger = (
            len(self._structural_queue) >= self.config.structural_repro_batch_size
            or elapsed > self.config.structural_repro_interval_min * 60
        )
        if should_trigger:
            await self._run_structural_repro_batch()

    async def _run_structural_repro_batch(self) -> None:
        """Confirm or suppress pending structural KV anomaly candidates.

        For each candidate, re-runs the trace k times and checks whether the
        same anomaly family recurs.  Confirmed (≥ ceil(k*2/3) runs reproduce
        the anomaly) → file as KV_CORRUPTION. Otherwise → cooldown.
        """
        import math as _math

        candidates = list(self._structural_queue)
        self._structural_queue.clear()
        self._last_structural_repro_at = time.time()

        if not candidates:
            return

        # --- Dedup: keep earliest iteration per sig ----------------------
        seen_sigs: dict = {}
        for c in sorted(candidates, key=lambda c: c.iteration):
            sig = self._structural_candidate_sig(c.trace, c.anomaly_types[0])
            if sig not in seen_sigs:
                seen_sigs[sig] = c
        deduped = sorted(seen_sigs.values(), key=lambda c: -c.top_score)

        print(f"[runner] structural repro batch start: {len(deduped)} candidates "
              f"(from {len(candidates)} enqueued)")

        budget_deadline = time.time() + self.config.structural_repro_budget_s
        k               = self.config.structural_repro_k
        confirm_needed  = _math.ceil(k * 2 / 3)

        for c in deduped:
            sig = self._structural_candidate_sig(c.trace, c.anomaly_types[0])

            if time.time() > budget_deadline:
                print(f"[runner] structural repro budget exhausted — "
                      f"re-queuing {len(deduped) - deduped.index(c)} remaining")
                self._structural_queue.extend(deduped[deduped.index(c):])
                break

            confirmed_count = 0
            best_summary    = c.summary   # snapshot from trigger run
            _snapshots: list = []         # per-run block content snapshots

            for run_i in range(k):
                try:
                    if self._collector is not None:
                        await self._collector.begin_probe()
                    _rt_start = time.time()
                    _res      = await self.executor.run(c.trace, self.config.base_url)
                    _rt_end   = time.time()
                    if self._collector is not None:
                        _run_summary = await self._collector.end_probe()
                        _block_snap  = self._collector.get_block_snapshot(_rt_start, _rt_end)
                        if _block_snap is not None:
                            _snapshots.append(_block_snap)
                        if _run_summary is not None and self.adapter is not None:
                            _run_anomalies = self.adapter.analyze_collection(_run_summary)
                            _run_types = {a.anomaly_type for a in _run_anomalies
                                          if a.score >= 5.0}
                            if _run_types & set(c.anomaly_types):
                                confirmed_count += 1
                                best_summary = _run_summary
                except Exception as exc:
                    print(f"[runner] structural repro run error: {exc}")

            # Cross-run content comparison: same block_hash with different token_ids
            # across runs is a direct corruption signal independent of timing.
            # Compare all consecutive pairs so middle-run divergences are not missed.
            _cross_run_conflicts: list = []
            if len(_snapshots) >= 2:
                from ..kv_events import compare_block_snapshots
                seen_hashes: set = set()
                for _i in range(len(_snapshots) - 1):
                    for c_item in compare_block_snapshots(_snapshots[_i], _snapshots[_i + 1]):
                        h = c_item["hash"]
                        if h not in seen_hashes:
                            seen_hashes.add(h)
                            _cross_run_conflicts.append(c_item)
                if _cross_run_conflicts:
                    confirmed_count = max(confirmed_count, confirm_needed)

            if confirmed_count >= confirm_needed:
                # Build a confirmed finding
                score = (
                    9.0 if _cross_run_conflicts
                    else 8.0 if ("cross_lora_reuse" in c.anomaly_types
                                 or "hash_content_conflict" in c.anomaly_types)
                    else 6.0
                )
                detail = (
                    f"structural KV anomaly confirmed: {', '.join(c.anomaly_types)} "
                    f"recurred in {confirmed_count}/{k} reruns"
                )
                if hasattr(best_summary, "conflicting_hashes") and best_summary.conflicting_hashes:
                    detail += f", conflicting_hashes={best_summary.conflicting_hashes[:5]}"
                if hasattr(best_summary, "reuse_too_fast_min_gap_ms"):
                    gap = best_summary.reuse_too_fast_min_gap_ms
                    if gap < float('inf'):
                        detail += f", min_gap_ms={gap:.2f}"
                if _cross_run_conflicts:
                    detail += f", cross_run_token_conflicts={len(_cross_run_conflicts)}"

                finding = OracleResult(
                    passed=False,
                    bug_class=BugClass.KV_CORRUPTION,
                    score=score,
                    detail=detail,
                    probe_metadata={
                        "source":                    "structural_repro",
                        "anomaly_types":             c.anomaly_types,
                        "confirmed":                 f"{confirmed_count}/{k}",
                        "kv_summary":                best_summary.to_dict(),
                        "cross_run_conflicts_sample": _cross_run_conflicts[:3],
                    },
                )
                stub = Trace(seed=c.iteration, events=[])
                print(f"[runner] structural repro CONFIRMED  "
                      f"types={c.anomaly_types}  score={score:.1f}  "
                      f"confirmed={confirmed_count}/{k}")
                await self._handle_finding(stub, [finding], c.iteration, extra={
                    "probe": "structural_kv",
                    "anomaly_types": c.anomaly_types,
                })
                if c.fingerprint is not None:
                    self.corpus.boost(c.fingerprint)
            else:
                self._structural_cooldown[sig] = (
                    time.time() + self.config.structural_repro_cooldown_s
                )
                print(f"[runner] structural repro rejected  "
                      f"types={c.anomaly_types}  "
                      f"confirmed={confirmed_count}/{k}  sig={sig[:8]}")

        print("[runner] structural repro batch done")

    async def _run_soft_repro_batch(self) -> None:
        """Validate or suppress all pending soft-suspicion candidates in one batch."""
        import copy as _copy

        candidates = list(self._soft_queue)
        self._soft_queue.clear()
        self._last_soft_repro_at = time.time()

        if not candidates:
            return

        # --- Step 1: Family dedup ----------------------------------------
        # Keep one per structural signature (lowest iteration = first observed).
        seen_sigs: dict[str, SoftCandidate] = {}
        for c in sorted(candidates, key=lambda c: c.iteration):
            sig = self._soft_candidate_sig(c)
            if sig not in seen_sigs:
                seen_sigs[sig] = c
        deduped = list(seen_sigs.values())

        # --- Step 2: Priority sort ----------------------------------------
        # Prefer candidates whose trigger request has prefix_len > 0 (more
        # likely to be a genuine cache bug), then by lower total trace cost.
        def _priority(c: SoftCandidate) -> tuple:
            ev_map = {e.request_id: e for e in c.trace.events
                      if e.action == ActionType.SEND}
            trig = ev_map.get(c.trigger_request_id)
            has_prefix = (trig is not None and (trig.prefix_len or 0) > 0)
            cost = sum(
                (e.prompt_len or 64) + (e.max_tokens or 32)
                for e in c.trace.events if e.action == ActionType.SEND
            )
            return (0 if has_prefix else 1, cost)

        deduped.sort(key=_priority)
        deduped = deduped[:self.config.soft_repro_max_candidates]

        print(f"[runner] soft repro batch start: {len(deduped)} candidates "
              f"(from {len(candidates)} enqueued)")

        budget_deadline = time.time() + self.config.soft_repro_budget_s

        for c in deduped:
            sig = self._soft_candidate_sig(c)

            # --- Budget check --------------------------------------------
            if time.time() > budget_deadline:
                print(f"[runner] soft repro budget exhausted, "
                      f"re-queuing {len(deduped) - deduped.index(c)} remaining")
                self._soft_queue.extend(deduped[deduped.index(c):])
                break

            # --- Step 5b: Run full trace k times -------------------------
            target_by_rid: dict[str, list[str]] = {}
            for _ in range(self.config.soft_repro_k):
                try:
                    res = await self.executor.run(c.trace, self.config.base_url)
                    for req in res.requests:
                        if req.output_text is not None and req.cancelled_at is None:
                            target_by_rid.setdefault(req.request_id, []).append(
                                req.output_text
                            )
                except Exception as exc:
                    print(f"[runner] soft repro target run error: {exc}")

            target_outputs = target_by_rid.get(c.trigger_request_id, [])

            # --- Step 5c-d: Build solo baseline trace --------------------
            baseline_ev = self._pick_baseline_event(c.trace, c.trigger_request_id)
            if baseline_ev is None:
                print(f"[runner] soft repro: no SEND event, skipping iter={c.iteration}")
                continue

            solo_ev = _copy.deepcopy(baseline_ev)
            solo_ev.offset_ms = 0
            solo_trace = Trace(seed=c.trace.seed, events=[solo_ev])

            # --- Step 5e: Run solo trace k times -------------------------
            baseline_outputs: list[str] = []
            for _ in range(self.config.soft_repro_k):
                try:
                    res = await self.executor.run(solo_trace, self.config.base_url)
                    for req in res.requests:
                        if req.output_text is not None and req.cancelled_at is None:
                            baseline_outputs.append(req.output_text)
                except Exception as exc:
                    print(f"[runner] soft repro baseline run error: {exc}")

            # --- Step 5f: Stability oracle -------------------------------
            stability = Oracle.check_stability(target_outputs, baseline_outputs)

            if not stability.passed:
                # Confirmed: save finding and boost corpus entry
                print(f"[runner] soft repro CONFIRMED  "
                      f"request={c.trigger_request_id}  score={stability.score:.1f}")
                await self._handle_finding(c.trace, [stability], c.iteration)
                if c.fingerprint is not None:
                    boosted = self.corpus.boost(c.fingerprint)
                    if not boosted:
                        print(f"[runner] soft repro: corpus.boost() found no entry "
                              f"for fp={c.fingerprint}")
            else:
                # Rejected: apply cooldown so this family is suppressed for a while
                self._soft_cooldown[sig] = (
                    time.time() + self.config.soft_repro_cooldown_s
                )
                print(f"[runner] soft repro rejected (false positive, cooldown "
                      f"{self.config.soft_repro_cooldown_s/60:.0f}min)  "
                      f"request={c.trigger_request_id}  sig={sig[:8]}")

        print(f"[runner] soft repro batch done")

    async def _cache_hit_probe(self, iteration: int) -> None:
        probe = Trace(seed=0, events=[
            Event(action=ActionType.SEND, offset_ms=0,
                  request_id="cache_probe",
                  prompt_len=256, max_tokens=1, stream=True)
        ])
        try:
            if self._cache_probe_warm_ttft is None:
                await self.executor.run(probe, self.config.base_url)
                warm_result = await self.executor.run(probe, self.config.base_url)
                for req in warm_result.requests:
                    ttft = req.ttft_ms()
                    if ttft is not None:
                        self._cache_probe_warm_ttft = ttft
                        print(f"[{iteration:05d}] cache probe warm baseline: "
                              f"{ttft:.0f} ms")
                        break
            else:
                probe_result = await self.executor.run(probe, self.config.base_url)
                for req in probe_result.requests:
                    ttft = req.ttft_ms()
                    if ttft is None:
                        continue
                    finding = Oracle.check_cache_hit_proxy(
                        self._cache_probe_warm_ttft, ttft, multiplier=4.0
                    )
                    if not finding.passed:
                        print(f"[{iteration:05d}] CACHE EVICTION BUG "
                              f"probe_ttft={ttft:.0f} ms")
                        await self._handle_finding(probe, [finding], iteration)
                    break
        except Exception as exc:
            print(f"[{iteration:05d}] cache hit probe error: {exc}")

    async def _update_idle_windows(self, iteration: int, trace: Trace) -> None:
        """Take a post-settle idle snapshot and check for cross-iteration leaks."""
        try:
            snap = await self.monitor.peek()
        except Exception:
            return

        # --- Memory leak ---
        if snap.kv_cache_usage_pct is not None:
            if (snap.requests_running or 0) == 0 and (snap.requests_waiting or 0) == 0:
                self._mem_idle_window.append(snap.kv_cache_usage_pct)
                if len(self._mem_idle_window) > _IDLE_WINDOW_SIZE * 4:
                    self._mem_idle_window.pop(0)
                if len(self._mem_idle_window) >= _IDLE_WINDOW_SIZE:
                    baseline = min(self._mem_idle_window[:_IDLE_WINDOW_SIZE])
                    drift = snap.kv_cache_usage_pct - baseline
                    if drift > _MEM_LEAK_DRIFT_PCT:
                        finding = OracleResult(
                            passed=False,
                            bug_class=BugClass.MEMORY_LEAK,
                            detail=(f"KV cache usage drifted {drift:.1f}% above idle baseline "
                                    f"({baseline:.1f}% → {snap.kv_cache_usage_pct:.1f}%) "
                                    f"over {len(self._mem_idle_window)} iterations"),
                            score=min(drift / 5.0, 10.0),
                        )
                        await self._handle_finding(trace, [finding], iteration, extra={
                            "cross_iteration": True,
                            "leak_window_size": len(self._mem_idle_window),
                        })
                        self._mem_idle_window.clear()

        # --- KV block leak ---
        if snap.kv_cache_usage_pct is not None:
            if (snap.requests_running or 0) == 0 and (snap.requests_waiting or 0) == 0:
                if iteration % _KV_FLUSH_EVERY == 0:
                    post_flush_pct = await self._flush_kv_cache()
                    snap_kv_pct = post_flush_pct if post_flush_pct is not None else snap.kv_cache_usage_pct
                else:
                    snap_kv_pct = snap.kv_cache_usage_pct
                self._kv_idle_window.append(snap_kv_pct)
                if len(self._kv_idle_window) > _IDLE_WINDOW_SIZE * 4:
                    self._kv_idle_window.pop(0)
                if len(self._kv_idle_window) >= _IDLE_WINDOW_SIZE:
                    baseline = min(self._kv_idle_window[:_IDLE_WINDOW_SIZE])
                    drift = snap_kv_pct - baseline
                    if drift > _KV_LEAK_DRIFT_PCT:
                        finding = OracleResult(
                            passed=False,
                            bug_class=BugClass.KV_BLOCK_LEAK,
                            detail=(f"KV cache idle usage drifted {drift:.1f}% above baseline "
                                    f"({baseline:.1f}% → {snap_kv_pct:.1f}%) "
                                    f"over {len(self._kv_idle_window)} iterations"),
                            score=drift / 10.0,
                        )
                        await self._handle_finding(trace, [finding], iteration, extra={
                            "cross_iteration": True,
                            "leak_window_size": len(self._kv_idle_window),
                        })
                        self._kv_idle_window.clear()

    async def _resolve_model_name(self) -> str:
        """Return config.model_name if set, otherwise discover from /v1/models."""
        if self.adapter is not None:
            name = await self.adapter.resolve_model_name(
                self.config.base_url, self.config.model_name
            )
            if name:
                self.config.model_name = name
            return name

        if self.config.model_name:
            return self.config.model_name
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.config.base_url}/v1/models", timeout=5.0
                )
                data = resp.json()
                name = data["data"][0]["id"]
                self.config.model_name = name
                return name
        except Exception:
            return ""

    async def _flush_kv_cache(self) -> Optional[float]:
        """Send a large random prompt to pressure-evict all LRU-reclaimable KV blocks."""
        if self.adapter is not None:
            result = await self.adapter.flush_state(self.config.base_url)
            try:
                snap = await self.monitor.peek()
                return snap.kv_cache_usage_pct
            except Exception:
                return result

        import httpx
        import uuid
        model_name = await self._resolve_model_name()
        flush_tokens = " ".join(f"flush{uuid.uuid4().hex[:6]}_{i}" for i in range(2048))
        payload: dict = {
            "prompt":      flush_tokens,
            "max_tokens":  1,
            "temperature": 0.0,
            "stream":      False,
        }
        if model_name:
            payload["model"] = model_name
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.config.base_url}/v1/completions",
                    json=payload,
                    timeout=60.0,
                )
        except Exception:
            pass
        await asyncio.sleep(1.0)
        try:
            snap = await self.monitor.peek()
            return snap.kv_cache_usage_pct
        except Exception:
            return None

    async def _iso_drain(self, base_url: str) -> None:
        """Drain highest-priority isolation probe from the promotion queue."""
        import heapq

        now = self._iso_iteration

        # Expire stale entries older than T iterations
        fresh = [
            (neg_s, enq_at, meta)
            for neg_s, enq_at, meta in self._iso_queue
            if now - enq_at <= self._ISO_SIG_EXPIRY_T
        ]
        if len(fresh) != len(self._iso_queue):
            self._iso_queue = fresh
            heapq.heapify(self._iso_queue)

        if not self._iso_queue:
            return

        neg_score, enqueued_at, meta = heapq.heappop(self._iso_queue)

        sig = (
            meta.get("adapter_a", ""),
            meta.get("adapter_b", ""),
            meta.get("prompt_len", 0) // 64,
            meta.get("prefix_len", 0) // 64,
        )

        # Skip if this sig was drained recently (within cooldown)
        last = self._iso_sig_last.get(sig, -(self._ISO_COOLDOWN_M * 2))
        if now - last < self._ISO_COOLDOWN_M:
            return

        self._iso_sig_last[sig] = now
        self._iso_cooldown_remaining = self._ISO_COOLDOWN_M

        if self.adapter is None or not hasattr(self.adapter, "adapter_isolation_probe"):
            return

        try:
            findings = await self.adapter.adapter_isolation_probe(
                executor=self.executor,
                base_url=base_url,
                prompt_len=meta.get("prompt_len", 64),
                prefix_len=meta.get("prefix_len", 0),
            )
        except Exception as exc:
            print(f"[iso_probe] error: {exc}")
            return

        for f in findings:
            if not f.passed:
                stub = Trace(seed=self._iso_iteration, events=[])
                await self._handle_finding(stub, [f], self._iso_iteration, extra={
                    "probe": "adapter_isolation",
                    "probe_metadata": f.probe_metadata,
                })

    async def _minimize(self, trace: Trace,
                        original_bugs: list[OracleResult]) -> Trace:
        self.events.emit("minimization_started", events=len(trace.events))
        original_classes = {b.bug_class for b in original_bugs if b.bug_class}

        async def run_fn(t):
            await self.monitor.start()
            result  = await self.executor.run(t, self.config.base_url)
            session = await self.monitor.stop()
            return result, session

        def oracle_fn(run_output) -> bool:
            result, session = run_output
            findings = self.oracle.evaluate(result, session)
            found_classes = {f.bug_class for f in findings if not f.passed}
            return bool(original_classes & found_classes)

        minimized = await minimize_with_oracle(
            trace, run_fn, oracle_fn,
            min_length=2,
        )
        original_len  = len(trace.events)
        minimized_len = len(minimized.events)
        print(f"[runner] minimized {original_len} → {minimized_len} events")
        return minimized
