"""Tests for runner.py — config, minimizer gating, and confirmation pass."""

import json

import pytest

from fuzzer.oracles import BugClass, OracleResult
from fuzzer.runner import Runner, RunnerConfig
from fuzzer.trace import ActionType, Event, Trace


def make_bug(bug_class: BugClass, score: float) -> OracleResult:
    return OracleResult(passed=False, bug_class=bug_class, score=score)


def make_runner(minimize_findings: bool = True) -> Runner:
    config = RunnerConfig(minimize_findings=minimize_findings)
    runner = Runner.__new__(Runner)
    runner.config = config
    return runner


class TestWorstLatencyScore:
    def test_returns_max_latency_score(self):
        runner = make_runner()
        bugs = [
            make_bug(BugClass.LATENCY_SPIKE, 3.0),
            make_bug(BugClass.LATENCY_SPIKE, 7.0),
        ]
        assert runner._worst_latency_score(bugs) == 7.0

    def test_returns_zero_when_no_latency_bugs(self):
        runner = make_runner()
        bugs = [make_bug(BugClass.MEMORY_LEAK, 5.0)]
        assert runner._worst_latency_score(bugs) == 0.0

    def test_returns_zero_for_empty_list(self):
        runner = make_runner()
        assert runner._worst_latency_score([]) == 0.0


class TestShouldMinimize:
    def test_minimize_disabled_globally(self):
        runner = make_runner(minimize_findings=False)
        bugs = [make_bug(BugClass.MEMORY_LEAK, score=5.0)]
        assert runner._should_minimize(bugs) is False

    def test_lifecycle_error_minimized(self):
        runner = make_runner()
        assert runner._should_minimize([make_bug(BugClass.LIFECYCLE_ERROR, 3.0)]) is True

    def test_scheduler_hang_minimized(self):
        runner = make_runner()
        assert runner._should_minimize([make_bug(BugClass.SCHEDULER_HANG, 8.0)]) is True

    def test_memory_leak_minimized(self):
        runner = make_runner()
        assert runner._should_minimize([make_bug(BugClass.MEMORY_LEAK, 2.0)]) is True

    def test_kv_block_leak_minimized(self):
        runner = make_runner()
        assert runner._should_minimize([make_bug(BugClass.KV_BLOCK_LEAK, 1.0)]) is True

    def test_latency_spike_not_minimized(self):
        # Latency spikes are never minimized: every minimizer probe is slow by
        # definition, and stripping requests removes the cache pressure context
        # needed to reproduce the spike.
        runner = make_runner()
        assert runner._should_minimize([make_bug(BugClass.LATENCY_SPIKE, 3.0)]) is False

    def test_mixed_bugs_minimized(self):
        runner = make_runner()
        bugs = [
            make_bug(BugClass.LATENCY_SPIKE, 6.0),
            make_bug(BugClass.MEMORY_LEAK, 2.0),
        ]
        assert runner._should_minimize(bugs) is True

    def test_empty_bug_list_returns_false(self):
        runner = make_runner()
        assert runner._should_minimize([]) is False


# ---------------------------------------------------------------------------
# Score-multiplier resolution (Phase 1 of adapter/strategy split)
# ---------------------------------------------------------------------------

class TestCorpusMultiplierWiring:
    """The runner resolves the corpus-bias multiplier once per campaign and
    must hand the SAME callable (identity, not equality) to both
    corpus.select() calls — otherwise known-bug-shaped traces still dominate
    as splice parents. Legacy adapters with no Strategy resolve to None.
    """

    def test_none_adapter_resolves_to_none(self):
        from fuzzer.core.runner import _score_multiplier_for_adapter
        assert _score_multiplier_for_adapter(None) is None

    def test_legacy_adapter_without_strategy_resolves_to_none(self):
        """Legacy SglangBaseAdapter / VllmBaseAdapter / etc. don't have a
        `strategy` attribute, so the helper returns None and corpus
        selection runs the byte-exact pre-refactor path."""
        from types import SimpleNamespace
        from fuzzer.core.runner import _score_multiplier_for_adapter
        legacy_adapter = SimpleNamespace()
        assert not hasattr(legacy_adapter, "strategy")
        assert _score_multiplier_for_adapter(legacy_adapter) is None

    def test_adapter_with_strategy_returns_score_multiplier_by_identity(self):
        """Identity assertion (`is`), not equality — catches accidental
        wrapping or copying of the callback in the resolution path."""
        from types import SimpleNamespace
        from fuzzer.core.runner import _score_multiplier_for_adapter
        sentinel = lambda e: 0.7
        new_shape_adapter = SimpleNamespace(
            strategy=SimpleNamespace(score_multiplier=sentinel)
        )
        resolved = _score_multiplier_for_adapter(new_shape_adapter)
        assert resolved is sentinel

    def test_adapter_with_strategy_but_no_multiplier_resolves_to_none(self):
        """A Strategy may legitimately omit score_multiplier (or set it to
        None). The helper should tolerate that and degrade to default
        corpus behavior rather than raise."""
        from types import SimpleNamespace
        from fuzzer.core.runner import _score_multiplier_for_adapter
        adapter = SimpleNamespace(strategy=SimpleNamespace())
        assert _score_multiplier_for_adapter(adapter) is None

    def test_resolution_is_pure(self):
        """Same input → same output. No hidden state mutation."""
        from types import SimpleNamespace
        from fuzzer.core.runner import _score_multiplier_for_adapter
        sentinel = lambda e: 1.0
        adapter = SimpleNamespace(strategy=SimpleNamespace(score_multiplier=sentinel))
        assert _score_multiplier_for_adapter(adapter) is sentinel
        assert _score_multiplier_for_adapter(adapter) is sentinel


# ---------------------------------------------------------------------------
# Baseline probe event overrides
# ---------------------------------------------------------------------------

class TestBaselineProbeOverridesWiring:
    """Verify the runner calls baseline_probe_event_overrides when an adapter is set."""

    def _make_runner_with_lora(self):
        import tempfile, os
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor
        from fuzzer.adapters.vllm_lora.adapter import VllmLoRAAdapter
        tmp = tempfile.mkdtemp()
        seeds_dir  = os.path.join(tmp, "seeds")
        corpus_dir = os.path.join(tmp, "corpus")
        os.makedirs(seeds_dir)
        config  = RunnerConfig(seeds_dir=seeds_dir, corpus_dir=corpus_dir)
        adapter = VllmLoRAAdapter({"lora_a": "/tmp/a", "lora_b": "/tmp/b"})
        return Runner(config, adapter=adapter,
                      executor=PythonExecutor(adapter=adapter),
                      monitor=MockMonitor())

    def _make_runner_no_adapter(self):
        import tempfile, os
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor
        tmp = tempfile.mkdtemp()
        seeds_dir  = os.path.join(tmp, "seeds")
        corpus_dir = os.path.join(tmp, "corpus")
        os.makedirs(seeds_dir)
        config = RunnerConfig(seeds_dir=seeds_dir, corpus_dir=corpus_dir)
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def test_base_adapter_runner_baseline_event_has_no_adapter_name(self):
        """VllmBaseAdapter (default) baseline_probe_event_overrides is a no-op."""
        from fuzzer.trace import Event, ActionType
        from fuzzer.core.runner import _BASELINE_PROMPT_LEN
        runner = self._make_runner_no_adapter()
        ev = Event(action=ActionType.SEND, offset_ms=0,
                   request_id="baseline_probe",
                   prompt_len=_BASELINE_PROMPT_LEN, max_tokens=16, stream=True)
        result = runner.adapter.baseline_probe_event_overrides(ev)
        assert result.adapter_name is None


# ---------------------------------------------------------------------------
# KV summary passed to evaluate_extra; adapter layer merged into kv_anomalies
# ---------------------------------------------------------------------------

import math as _math
from fuzzer.kv_events import KVProbeSummary


def _make_kv_summary(cross_lora=0):
    return KVProbeSummary(
        start_t=0.0, end_t=1.0,
        stored_count=5, removed_count=2, cleared_count=0,
        distinct_blocks=4, per_lora_counts={},
        cross_lora_reused_blocks=cross_lora,
        conflicting_hashes=["0xabc"] * cross_lora,
        reuse_too_fast_count=0,
        reuse_too_fast_min_gap_ms=_math.inf,
        fingerprint="abcdef1234567890",
        raw_event_sample=[],
    )


class TestEvaluateExtraNewSignature:
    """evaluate_extra no longer receives kv_summary — it reads from the collector."""

    def test_evaluate_extra_accepts_trace_kwarg(self):
        """New signature is evaluate_extra(result, session, trace=None)."""
        from fuzzer.core.adapter import TargetAdapter

        class _TrackingAdapter(TargetAdapter):
            received_trace = None
            def build_payload(self, e): return ("/v1/completions", {})
            async def check_liveness(self, u): return True
            async def flush_state(self, u): return None
            async def resolve_model_name(self, u, m): return ""
            async def discover_version(self, u): return {}
            def evaluate_extra(self, result, session, trace=None):
                _TrackingAdapter.received_trace = trace
                return []

        sentinel = object()
        adapter = _TrackingAdapter()
        adapter.evaluate_extra(None, None, trace=sentinel)
        assert _TrackingAdapter.received_trace is sentinel

    def test_evaluate_extra_no_kv_summary_param(self):
        """evaluate_extra must NOT accept kv_summary as a parameter."""
        import inspect
        from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
        from fuzzer.adapters.sglang.legacy import SglangBaseAdapter
        for cls in (VllmBaseAdapter, SglangBaseAdapter):
            sig = inspect.signature(cls.evaluate_extra)
            assert "kv_summary" not in sig.parameters, (
                f"{cls.__name__}.evaluate_extra still has kv_summary parameter"
            )


class TestAdapterAnalyzeCollection:
    """analyze_collection() is the new collector→anomaly bridge."""

    def test_analyze_collection_via_vllm_base(self):
        """VllmBaseAdapter.analyze_collection wraps kv anomalies as CollectionAnomaly."""
        from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
        from fuzzer.core.collector import CollectionAnomaly

        class _MockLoRAAdapter(VllmBaseAdapter):
            def analyze_kv_summary(self, summary):
                if summary.cross_lora_reused_blocks > 0:
                    return [{"anomaly_type": "cross_lora_reuse",
                             "count": summary.cross_lora_reused_blocks,
                             "score": 6.0, "detail": "test", "metadata": {}}]
                return []

        adapter = _MockLoRAAdapter()
        summary = _make_kv_summary(cross_lora=2)
        anomalies = adapter.analyze_collection(summary)
        assert all(isinstance(a, CollectionAnomaly) for a in anomalies)
        assert any(a.anomaly_type == "cross_lora_reuse" for a in anomalies)
        assert any(a.score == 6.0 for a in anomalies)

    def test_generic_never_produces_cross_lora(self):
        from fuzzer.kv_events import analyze_kv_anomalies
        result = analyze_kv_anomalies(_make_kv_summary(cross_lora=5))
        assert "cross_lora_reuse" not in [r["anomaly_type"] for r in result]

    def test_default_analyze_kv_summary_returns_empty(self):
        from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter

        adapter = VllmBaseAdapter()
        assert adapter.analyze_kv_summary(_make_kv_summary(cross_lora=3)) == []

    def test_analyze_collection_returns_empty_on_none_summary(self):
        from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
        adapter = VllmBaseAdapter()
        assert adapter.analyze_collection(None) == []


class TestRecentTraceContext:

    def _make_full_runner(self, tmp_path):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor

        seeds_dir = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        findings_dir = tmp_path / "findings"
        seeds_dir.mkdir()
        corpus_dir.mkdir()
        findings_dir.mkdir()

        config = RunnerConfig(
            seeds_dir=str(seeds_dir),
            corpus_dir=str(corpus_dir),
            findings_dir=str(findings_dir),
        )
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def _trace(self, seed: int, prompt_len: int = 16) -> Trace:
        return Trace(seed=seed, events=[
            Event(
                action=ActionType.SEND,
                offset_ms=0,
                request_id=f"r{seed}",
                prompt_len=prompt_len,
                max_tokens=8,
                stream=True,
            )
        ])

    def test_recent_trace_buffer_rolls_over(self, tmp_path):
        runner = self._make_full_runner(tmp_path)

        for iteration in range(12):
            runner._record_recent_trace(self._trace(seed=iteration), iteration)

        assert len(runner._recent_traces) == 10
        assert [r.iteration for r in runner._recent_traces] == list(range(2, 12))
        assert runner._recent_traces[0].trace.seed == 2
        assert runner._recent_traces[-1].trace.seed == 11

    @pytest.mark.asyncio
    async def test_liveness_failure_writes_recent_trace_context(self, tmp_path):
        runner = self._make_full_runner(tmp_path)

        async def _dead_server(_base_url):
            return False

        runner.adapter.check_liveness = _dead_server

        for iteration in range(3):
            runner._record_recent_trace(self._trace(seed=100 + iteration), iteration)

        healthy = await runner._check_liveness(iteration=7)
        assert healthy is False

        # Filenames carry a per-run sequence number so two findings filed in
        # the same iteration for the same trace cannot overwrite each other.
        findings = sorted((tmp_path / "findings").glob("finding_*.json"))
        assert len(findings) == 1
        finding_path = findings[0]
        assert finding_path.name.startswith("finding_00007_0_")
        context_dir = finding_path.with_name(finding_path.stem + "_context")
        manifest_path = context_dir / "manifest.json"

        assert context_dir.is_dir()
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["finding_file"] == finding_path.name
        assert manifest["trace_count"] == 3
        assert manifest["most_recent_iteration"] == 2

        trace_files = [context_dir / f"trace_{idx:02d}.json" for idx in range(3)]
        assert all(path.exists() for path in trace_files)

        first = json.loads(trace_files[0].read_text())
        last = json.loads(trace_files[-1].read_text())
        assert first["iteration"] == 0
        assert first["trace"]["seed"] == 100
        assert first["is_most_recent"] is False
        assert last["iteration"] == 2
        assert last["trace"]["seed"] == 102
        assert last["is_most_recent"] is True


# ---------------------------------------------------------------------------
# Verbose per-iteration logging
# ---------------------------------------------------------------------------

def _make_verbose_trace() -> Trace:
    """Helper trace with mixed prompt_lens so the shape= field exercises
    both the distinct-count and stdev calculation paths."""
    return Trace(seed=42, events=[
        Event(action=ActionType.SEND, offset_ms=0,    request_id="r0",
              prompt_len=128, max_tokens=32, adapter_name="lora_a"),
        Event(action=ActionType.SEND, offset_ms=100,  request_id="r1",
              prompt_len=128, max_tokens=32, adapter_name="lora_b"),
        Event(action=ActionType.WAIT, offset_ms=500,  request_id=""),
        Event(action=ActionType.SEND, offset_ms=600,  request_id="r2",
              prompt_len=256, max_tokens=64, adapter_name="lora_a"),
    ])


class _FakeKvSummary:
    """Stand-in for kv_events.KVProbeSummary — only the fields read by
    _emit_verbose_log are populated. Avoids importing the real summary
    class (which pulls in zmq) for a unit test."""
    def __init__(self, fingerprint: str = "abc123",
                 stored: int = 100, removed: int = 30,
                 xlora: int = 2, fast: int = 1, conflict: int = 0):
        self.fingerprint                 = fingerprint
        self.stored_count                = stored
        self.removed_count               = removed
        self.cross_lora_reused_blocks    = xlora
        self.reuse_too_fast_count        = fast
        self.hash_content_conflict_count = conflict


class TestEmitVerboseLog:
    def test_emits_core_fields(self, capsys):
        runner = make_runner()
        trace = _make_verbose_trace()

        runner._emit_verbose_log(
            iteration=42,
            trace=trace,
            mutation_ops=["jitter_timing", "add_burst"],
            kv_summary=None,
        )
        out = capsys.readouterr().out.strip()

        assert out.startswith("[00042] verbose")
        assert "mut=jitter_timing,add_burst" in out
        # 4 events total, 3 SEND
        assert "events=4/3" in out
        # 2 distinct adapters: lora_a, lora_b
        assert "adapters=2" in out
        # trace.duration_ms() — max offset (600)
        assert "dur=600ms" in out
        # 2 distinct prompt_lens (128, 256); stdev([128,128,256]) ≈ 73.9 → 73
        assert "shape=2/73" in out

    def test_omits_kv_when_summary_none(self, capsys):
        runner = make_runner()
        runner._emit_verbose_log(
            iteration=1,
            trace=_make_verbose_trace(),
            mutation_ops=["jitter_timing"],
            kv_summary=None,
        )
        out = capsys.readouterr().out
        assert "kv=" not in out

    def test_omits_kv_when_collector_disabled_placeholder(self, capsys):
        # Real placeholder summaries from disabled collectors have empty
        # fingerprint — must not appear as a real measurement.
        runner = make_runner()
        runner._emit_verbose_log(
            iteration=1,
            trace=_make_verbose_trace(),
            mutation_ops=["jitter_timing"],
            kv_summary=_FakeKvSummary(fingerprint=""),
        )
        out = capsys.readouterr().out
        assert "kv=" not in out

    def test_includes_kv_fields_when_populated(self, capsys):
        runner = make_runner()
        runner._emit_verbose_log(
            iteration=7,
            trace=_make_verbose_trace(),
            mutation_ops=["splice"],
            kv_summary=_FakeKvSummary(stored=142, removed=38,
                                       xlora=2, fast=1, conflict=0),
        )
        out = capsys.readouterr().out
        assert "kv=stored:142/removed:38/xlora:2/fast:1/conflict:0" in out

    def test_handles_empty_mutation_ops(self, capsys):
        runner = make_runner()
        runner._emit_verbose_log(
            iteration=0,
            trace=_make_verbose_trace(),
            mutation_ops=[],
            kv_summary=None,
        )
        out = capsys.readouterr().out
        assert "mut=-" in out

    def test_iteration_number_is_zero_padded(self, capsys):
        runner = make_runner()
        runner._emit_verbose_log(
            iteration=3,
            trace=_make_verbose_trace(),
            mutation_ops=["jitter_timing"],
            kv_summary=None,
        )
        out = capsys.readouterr().out
        assert out.startswith("[00003]")

    def test_shape_zero_when_homogeneous(self, capsys):
        # All SEND prompt_lens identical → shape=1/0 (1 distinct, stdev=0)
        runner = make_runner()
        trace = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r0",
                  prompt_len=128, max_tokens=32),
            Event(action=ActionType.SEND, offset_ms=10, request_id="r1",
                  prompt_len=128, max_tokens=32),
            Event(action=ActionType.SEND, offset_ms=20, request_id="r2",
                  prompt_len=128, max_tokens=32),
        ])
        runner._emit_verbose_log(
            iteration=1, trace=trace,
            mutation_ops=["jitter_timing"], kv_summary=None,
        )
        out = capsys.readouterr().out
        assert "shape=1/0" in out

    def test_shape_handles_single_send(self, capsys):
        # stdev requires >=2 samples → with 1 SEND, stdev path falls back to 0
        runner = make_runner()
        trace = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r0",
                  prompt_len=512, max_tokens=64),
        ])
        runner._emit_verbose_log(
            iteration=1, trace=trace,
            mutation_ops=["jitter_timing"], kv_summary=None,
        )
        out = capsys.readouterr().out
        assert "shape=1/0" in out

    def test_shape_handles_no_sends(self, capsys):
        # Only WAIT events, no SENDs → shape=0/0
        runner = make_runner()
        trace = Trace(seed=1, events=[
            Event(action=ActionType.WAIT, offset_ms=0, request_id=""),
            Event(action=ActionType.WAIT, offset_ms=100, request_id=""),
        ])
        runner._emit_verbose_log(
            iteration=1, trace=trace,
            mutation_ops=["jitter_timing"], kv_summary=None,
        )
        out = capsys.readouterr().out
        assert "shape=0/0" in out


# ---------------------------------------------------------------------------
# Mechanism signature ledger (dedup plan §2.4 step 5)
# ---------------------------------------------------------------------------

class TestSignatureLedger:
    """The ledger replaces the old 'sig seen → return None' drop.

    Every occurrence is counted; only the JSONs past the K-th exemplar of a
    signature are suppressed.
    """

    def _runner(self, tmp_path, **overrides):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor

        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(exist_ok=True)
        config = RunnerConfig(
            seeds_dir=str(tmp_path / "seeds"),
            corpus_dir=str(tmp_path / "corpus"),
            findings_dir=str(tmp_path / "findings"),
            **overrides,
        )
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def _trace(self, seed=1, cancel=False, adapter=None,
               logprobs=None) -> Trace:
        events = [Event(action=ActionType.SEND, offset_ms=0, request_id="r0",
                        prompt_len=128, max_tokens=8, prefix_len=0,
                        adapter_name=adapter, logprobs=logprobs, stream=True)]
        if cancel:
            events.append(Event(action=ActionType.CANCEL, offset_ms=50,
                                request_id="r0"))
        return Trace(seed=seed, events=events)

    def _kv_bug(self, detail="(possible KV cache corruption, similarity=0.10)",
                score=10.0, probe_metadata=None):
        # KV_CORRUPTION is in _NEVER_MINIMIZE, so these findings take the
        # straight-through path with no executor traffic.
        return OracleResult(passed=False, bug_class=BugClass.KV_CORRUPTION,
                            detail=detail, score=score,
                            probe_metadata=probe_metadata)

    def _findings(self, tmp_path):
        return sorted((tmp_path / "findings").glob("finding_*.json"))

    @pytest.mark.asyncio
    async def test_duplicate_mechanism_increments_counter(self, tmp_path):
        runner = self._runner(tmp_path)
        for iteration in range(3):
            await runner._handle_finding(self._trace(seed=iteration),
                                         [self._kv_bug()], iteration)

        assert len(runner._sig_ledger) == 1
        sig, rec = next(iter(runner._sig_ledger.items()))
        assert rec["count"] == 3
        assert rec["exemplars"] == 3
        assert rec["first_iter"] == 0
        assert rec["last_iter"] == 2
        assert rec["max_score"] == 10.0
        assert rec["classes"] == ["kv_corruption"]

    @pytest.mark.asyncio
    async def test_seed_and_offsets_do_not_split_the_signature(self, tmp_path):
        """The legacy key hashed prompt_len/max_tokens per event, so one
        scale_prompt_lengths mutation refiled the same mechanism."""
        runner = self._runner(tmp_path)
        t1 = self._trace(seed=1)
        t2 = self._trace(seed=2)
        t2.events[0].prompt_len = 4096
        t2.events[0].max_tokens = 512
        t2.events[0].offset_ms = 900

        await runner._handle_finding(t1, [self._kv_bug()], 1)
        await runner._handle_finding(t2, [self._kv_bug()], 2)

        assert len(runner._sig_ledger) == 1

    @pytest.mark.asyncio
    async def test_distinct_mechanisms_get_distinct_entries(self, tmp_path):
        runner = self._runner(tmp_path)
        await runner._handle_finding(self._trace(), [self._kv_bug()], 1)
        # logprobs moves the workload shape, not just the numerics
        await runner._handle_finding(self._trace(logprobs=5),
                                     [self._kv_bug()], 2)
        assert len(runner._sig_ledger) == 2

    @pytest.mark.asyncio
    async def test_cancel_does_not_split_kv_corruption(self, tmp_path):
        """sig v2: for an all-kv_corruption finding the shape drops
        has_cancel and any_prefix, so a cancelled sibling request no longer
        files the same mechanism twice."""
        runner = self._runner(tmp_path)
        await runner._handle_finding(self._trace(), [self._kv_bug()], 1)
        await runner._handle_finding(self._trace(cancel=True),
                                     [self._kv_bug()], 2)
        assert len(runner._sig_ledger) == 1

    @pytest.mark.asyncio
    async def test_at_most_k_exemplars_written(self, tmp_path):
        runner = self._runner(tmp_path, exemplars_per_signature=2)
        for iteration in range(5):
            path = await runner._handle_finding(self._trace(seed=iteration),
                                                [self._kv_bug()], iteration)
            assert (path is not None) == (iteration < 2)

        assert len(self._findings(tmp_path)) == 2
        rec = next(iter(runner._sig_ledger.values()))
        assert rec["count"] == 5          # every occurrence counted
        assert rec["exemplars"] == 2      # only K written

    @pytest.mark.asyncio
    async def test_default_k_is_five(self, tmp_path):
        assert RunnerConfig().exemplars_per_signature == 5
        runner = self._runner(tmp_path)
        for iteration in range(7):
            await runner._handle_finding(self._trace(seed=iteration),
                                         [self._kv_bug()], iteration)
        assert len(self._findings(tmp_path)) == 5

    @pytest.mark.asyncio
    async def test_mech_sig_in_written_finding(self, tmp_path):
        from fuzzer.core.finding_sig import (SIG_VERSION,
                                             mechanism_signature_from_dict)

        runner = self._runner(tmp_path)
        path = await runner._handle_finding(self._trace(), [self._kv_bug()], 4)
        report = json.loads(path.read_text())

        assert report["mech_sig"] in runner._sig_ledger
        assert report["mech_sig_version"] == SIG_VERSION
        assert report["sig_occurrence"] == 1
        # the archived JSON re-signs to the same key it was filed under
        assert mechanism_signature_from_dict(report) == report["mech_sig"]

    @pytest.mark.asyncio
    async def test_two_findings_in_one_iteration_get_distinct_files(self, tmp_path):
        """_determinism_probe files one finding per diverging request of the
        same trace in the same iteration; a name built from (iteration, seed)
        alone made the second write clobber the first."""
        runner = self._runner(tmp_path)
        trace = self._trace(seed=7)
        p1 = await runner._handle_finding(trace, [self._kv_bug()], 12,
                                          extra={"request_id": "r0"})
        p2 = await runner._handle_finding(trace, [self._kv_bug()], 12,
                                          extra={"request_id": "r1"})

        assert p1 is not None and p2 is not None
        assert p1 != p2
        assert len(self._findings(tmp_path)) == 2
        assert json.loads(p1.read_text())["request_id"] == "r0"
        assert json.loads(p2.read_text())["request_id"] == "r1"

    @pytest.mark.asyncio
    async def test_recent_trace_context_dirs_do_not_collide(self, tmp_path):
        runner = self._runner(tmp_path)
        trace = self._trace(seed=7)
        p1 = await runner._handle_finding(trace, [self._kv_bug()], 12)
        p2 = await runner._handle_finding(trace, [self._kv_bug()], 12)
        runner._write_recent_trace_context(p1)
        runner._write_recent_trace_context(p2)

        ctx = sorted((tmp_path / "findings").glob("finding_*_context"))
        assert len(ctx) == 2

    @pytest.mark.asyncio
    async def test_ledger_count_matches_files_on_disk(self, tmp_path):
        runner = self._runner(tmp_path)
        trace = self._trace(seed=7)
        for _ in range(4):
            await runner._handle_finding(trace, [self._kv_bug()], 3)
        # a different mechanism, still same iteration and seed
        await runner._handle_finding(self._trace(seed=7, logprobs=5),
                                     [self._kv_bug()], 3)

        runner._dump_signature_ledger()
        ledger = json.loads(
            (tmp_path / "findings" / "signature_ledger.json").read_text())
        assert ledger["n_findings_written"] == 5
        assert len(self._findings(tmp_path)) == 5

    @pytest.mark.asyncio
    async def test_ledger_warns_when_files_are_missing(self, tmp_path, capsys):
        runner = self._runner(tmp_path)
        path = await runner._handle_finding(self._trace(), [self._kv_bug()], 1)
        path.unlink()
        runner._dump_signature_ledger()
        assert "WARNING" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_sig_occurrence_tracks_the_ledger_count(self, tmp_path):
        runner = self._runner(tmp_path)
        paths = [await runner._handle_finding(self._trace(seed=i),
                                              [self._kv_bug()], i)
                 for i in range(3)]
        occurrences = [json.loads(p.read_text())["sig_occurrence"] for p in paths]
        assert occurrences == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_probe_metadata_is_written_and_survives_non_json_values(
            self, tmp_path):
        class _Opaque:
            def __str__(self):
                return "opaque-summary"

        runner = self._runner(tmp_path)
        bug = self._kv_bug(probe_metadata={"similarity": 0.1,
                                           "diverge_token_idx": 7,
                                           "blob": _Opaque()})
        path = await runner._handle_finding(self._trace(), [bug], 1)
        pm = json.loads(path.read_text())["bugs"][0]["probe_metadata"]

        assert pm["similarity"] == 0.1
        assert pm["diverge_token_idx"] == 7
        assert pm["blob"] == "opaque-summary"   # json.dumps(default=str)

    @pytest.mark.asyncio
    async def test_sig_computed_before_minimization(self, tmp_path):
        """Minimization may rewrite the trace; it must not move the bucket."""
        from fuzzer.core.finding_sig import (mechanism_signature,
                                             mechanism_signature_from_dict)

        runner = self._runner(tmp_path)
        original = self._trace(cancel=True, adapter="lora_a")
        minimized = self._trace(cancel=False, adapter=None)

        async def _fake_minimize(trace, bugs):
            return minimized

        async def _no_server(*_a, **_kw):
            raise RuntimeError("no server in unit tests")

        runner._minimize = _fake_minimize
        runner.executor.run = _no_server

        # MEMORY_LEAK is in _ALWAYS_MINIMIZE
        bugs = [OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK,
                             detail="GPU memory grew by 148.0 MB after scenario drained",
                             score=5.0)]
        path = await runner._handle_finding(original, bugs, 9)
        report = json.loads(path.read_text())

        pre  = mechanism_signature(original, bugs)
        post = mechanism_signature(minimized, bugs)
        assert pre != post                       # the rewrite would have moved it
        assert report["mech_sig"] == pre
        assert report["trace"]["events"] == minimized.to_dict()["events"]
        # ...so the re-signed archived JSON differs — by design; the ledger key
        # is the pre-minimization mechanism.
        assert mechanism_signature_from_dict(report) == post

    @pytest.mark.asyncio
    async def test_lifecycle_only_finding_does_not_burn_an_exemplar(self, tmp_path):
        runner = self._runner(tmp_path, minimize_findings=False)
        bug = OracleResult(passed=False, bug_class=BugClass.LIFECYCLE_ERROR,
                           detail="requests never completed: ['r0']", score=3.0)
        assert await runner._handle_finding(self._trace(), [bug], 1) is None

        rec = next(iter(runner._sig_ledger.values()))
        assert rec["count"] == 1        # counted as telemetry
        assert rec["exemplars"] == 0    # but nothing was written
        assert self._findings(tmp_path) == []

    @pytest.mark.asyncio
    async def test_ledger_json_shape(self, tmp_path):
        from fuzzer.core.finding_sig import SIG_VERSION

        runner = self._runner(tmp_path, exemplars_per_signature=1,
                              campaign_name="grief", config_id="cfg7",
                              server_flags="--port 8123 --enable-lora")
        for iteration in range(4):
            await runner._handle_finding(self._trace(seed=iteration),
                                         [self._kv_bug()], iteration)
        await runner._handle_finding(self._trace(logprobs=5),
                                     [self._kv_bug()], 9)

        runner._dump_signature_ledger()
        ledger = json.loads(
            (tmp_path / "findings" / "signature_ledger.json").read_text())

        assert ledger["sig_version"] == SIG_VERSION
        assert ledger["campaign_name"] == "grief"
        assert ledger["config_id"] == "cfg7"
        assert ledger["server_flags"] == "--port PORT --enable-lora"
        assert ledger["exemplars_per_signature"] == 1
        assert ledger["n_mechanisms"] == 2
        assert ledger["n_occurrences"] == 5
        assert ledger["n_findings_written"] == 2
        assert isinstance(ledger["generated_at"], float)

        assert set(ledger["signatures"]) == set(runner._sig_ledger)
        for sig, rec in ledger["signatures"].items():
            assert len(sig) == 16
            assert set(rec) == {"count", "first_iter", "last_iter",
                                "max_score", "exemplars", "classes"}

    @pytest.mark.asyncio
    async def test_ledger_dumped_at_end_of_run(self, tmp_path):
        runner = self._runner(tmp_path)

        async def _one_finding():
            await runner._handle_finding(self._trace(), [self._kv_bug()], 1)

        runner._run_loop = _one_finding
        await runner.run()

        ledger = json.loads(
            (tmp_path / "findings" / "signature_ledger.json").read_text())
        assert ledger["n_occurrences"] == 1

    @pytest.mark.asyncio
    async def test_ledger_dumped_even_when_the_loop_raises(self, tmp_path):
        runner = self._runner(tmp_path)

        async def _boom():
            await runner._handle_finding(self._trace(), [self._kv_bug()], 1)
            raise RuntimeError("liveness abort")

        runner._run_loop = _boom
        with pytest.raises(RuntimeError):
            await runner.run()

        assert (tmp_path / "findings" / "signature_ledger.json").exists()


class TestDeterminismProbeMetadata:
    """_determinism_probe throws away the diverge token index and the logprob
    gap it computes; step 5/7 keeps them on the finding."""

    def _runner(self, tmp_path):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor

        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(exist_ok=True)
        config = RunnerConfig(
            seeds_dir=str(tmp_path / "seeds"),
            corpus_dir=str(tmp_path / "corpus"),
            findings_dir=str(tmp_path / "findings"),
        )
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def _result(self, text, tokens=None, top_logprobs=None):
        from fuzzer.core.types import RequestResult, TraceResult

        req = RequestResult(request_id="r0", sent_at=0.0, completed_at=1.0,
                            output_text=text, output_tokens=tokens,
                            output_top_logprobs=top_logprobs)
        return TraceResult(trace_seed=1, started_at=0.0, finished_at=1.0,
                           requests=[req])

    async def _run_probe(self, tmp_path, second, top_logprobs):
        runner = self._runner(tmp_path)
        trace = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r0",
                  prompt_len=32, max_tokens=4, stream=True)])
        first = self._result("aa bb cc")

        async def _second_run(*_a, **_kw):
            return self._result(second, tokens=list(second), top_logprobs=top_logprobs)

        captured = []

        async def _capture(trace_, bugs, iteration, *a, **kw):
            captured.append(bugs[0])
            return None

        runner.executor.run = _second_run
        runner._handle_finding = _capture
        await runner._determinism_probe(trace, first, 3)
        return captured

    @pytest.mark.asyncio
    async def test_diverge_metadata_attached(self, tmp_path):
        # 8 chars, diverge at index 3 ('b' vs 'X'); one dict per character
        top = [{"a": -0.1, "b": -5.0}] * 8
        captured = await self._run_probe(tmp_path, "aa Xb cc", top)

        assert len(captured) == 1
        pm = captured[0].probe_metadata
        assert pm["diverge_token_idx"] == 3
        assert pm["logprob_gap"] == pytest.approx(4.9)
        assert pm["diverge_bucket"] == 1        # log2(3)
        assert "similarity" in pm               # the oracle's own field survives

    @pytest.mark.asyncio
    async def test_metadata_is_none_when_no_logprobs_available(self, tmp_path):
        captured = await self._run_probe(tmp_path, "aa Xb cc", None)

        assert len(captured) == 1
        pm = captured[0].probe_metadata
        assert pm["diverge_token_idx"] is None
        assert pm["logprob_gap"] is None
        assert pm["diverge_bucket"] is None


class TestPrefixEquivProbe:
    """_prefix_equivalence_probe: the three-arm relational oracle.

    The probe puts two differently-salted twins beside every greedy SEND:

        A = plain (no salt)  reads the prefix cache
        B = salt1            cannot hit A's blocks  -> recompute
        C = salt2            cannot hit A's or B's  -> recompute

    B and C are both pure recomputes of a byte-identical prompt and must agree;
    when they do not, the A/B comparison the probe is built on is invalid and
    nothing may be filed as a cache bug.  Everything here is driven by a
    scripted executor -- no server, no GPU.
    """

    def _runner(self, tmp_path, seed=42):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor

        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        config = RunnerConfig(
            seed=seed,
            seeds_dir=str(tmp_path / "seeds"),
            corpus_dir=str(tmp_path / "corpus"),
            findings_dir=str(tmp_path / "findings"),
        )
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def _trace(self, **ev_kwargs):
        kw = dict(action=ActionType.SEND, offset_ms=0, request_id="r0",
                  prompt_len=32, max_tokens=8, stream=True)
        kw.update(ev_kwargs)
        return Trace(seed=1, events=[Event(**kw)])

    def _req(self, rid, text, **kw):
        from fuzzer.core.types import RequestResult
        return RequestResult(request_id=rid, sent_at=0.0, completed_at=1.0,
                             output_text=text, **kw)

    def _result(self, *requests):
        from fuzzer.core.types import TraceResult
        return TraceResult(trace_seed=1, started_at=0.0, finished_at=1.0,
                           requests=list(requests))

    # -- scripted arms ------------------------------------------------------

    def _arms(self, plain, salt1, salt2=None, *, plain_kw=None,
              salt1_kw=None, salt2_kw=None):
        """Three scripted RequestResults, one per arm.

        ``salt2`` defaults to ``salt1``: B == C, which is the "control passed,
        the cache is implicated" shape most tests want.  Pass it explicitly to
        exercise the self-check failure.
        """
        if salt2 is None:
            salt2 = salt1
        s1kw = dict(salt1_kw or {})
        s2kw = dict(salt2_kw if salt2_kw is not None else s1kw)
        return lambda t: [
            self._req("r0", plain, **(plain_kw or {})),
            self._req("r0~pfx", salt1, **s1kw),
            self._req("r0~pfx2", salt2, **s2kw),
        ]

    async def _run(self, tmp_path, trace, probe_requests, first_text="aa bb cc",
                   seed=42):
        """Run the probe against a scripted executor.

        *probe_requests* is a callable taking the probe Trace and returning the
        RequestResults the probe run should produce.
        """
        runner = self._runner(tmp_path, seed=seed)
        first = self._result(self._req("r0", first_text))
        seen_traces = []

        async def _probe_run(probe_trace, *_a, **_kw):
            seen_traces.append(probe_trace)
            return self._result(*probe_requests(probe_trace))

        captured = []

        async def _capture(trace_, bugs, iteration, *a, **kw):
            captured.append((bugs[0], kw.get("extra")))
            return None

        runner.executor.run = _probe_run
        runner._handle_finding = _capture
        await runner._prefix_equivalence_probe(trace, first, 5)
        return captured, seen_traces

    # -- trace construction -------------------------------------------------

    @pytest.mark.asyncio
    async def test_two_twins_are_added_with_distinct_salts(self, tmp_path):
        trace = self._trace()
        _, seen = await self._run(tmp_path, trace, self._arms("x", "x"))

        assert len(seen) == 1
        events = {e.request_id: e for e in seen[0].events}
        assert set(events) == {"r0", "r0~pfx", "r0~pfx2"}
        # the twins carry salts, the plain arm does not
        assert events["r0"].cache_salt is None
        assert events["r0~pfx"].cache_salt
        assert events["r0~pfx2"].cache_salt
        # DISTINCT salts: two arms sharing a salt would share block hashes and
        # the negative control would be comparing an arm against itself.
        assert events["r0~pfx"].cache_salt != events["r0~pfx2"].cache_salt
        # all three render byte-identical prompts, pinned to the ORIGINAL
        # request_id so the unsalted arm's prompt is unchanged from the probed run
        assert all(e.prompt_family_id == "r0" for e in events.values())
        # the caller's trace is untouched
        assert trace.events[0].prompt_family_id is None
        assert len(trace.events) == 1

    @pytest.mark.asyncio
    async def test_the_two_twins_are_labelled_distinctly(self, tmp_path):
        _, seen = await self._run(tmp_path, self._trace(), self._arms("x", "x"))
        events = {e.request_id: e for e in seen[0].events}
        assert events["r0~pfx"].origin == "pfx_twin"
        assert events["r0~pfx2"].origin == "pfx_ctrl"

    @pytest.mark.asyncio
    async def test_existing_prompt_family_id_is_preserved(self, tmp_path):
        trace = self._trace(prompt_family_id="fam7")
        _, seen = await self._run(tmp_path, trace, self._arms("x", "x"))
        events = {e.request_id: e for e in seen[0].events}
        assert all(e.prompt_family_id == "fam7" for e in events.values())

    # -- verdicts -----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_all_three_identical_no_finding(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(), self._arms("aa bb cc", "aa bb cc"))
        assert captured == []

    @pytest.mark.asyncio
    async def test_divergent_greedy_files_kv_corruption(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(), self._arms("aa bb cc", "aa XX cc"))

        assert len(captured) == 1
        finding, extra = captured[0]
        assert finding.passed is False
        assert finding.bug_class is BugClass.KV_CORRUPTION
        pm = finding.probe_metadata
        assert pm["probe"] == "prefix_equiv"
        assert pm["control_verdict"] == "cache_implicated"
        assert pm["bc_diverged"] is False
        assert pm["diverge_char_idx"] == 3
        assert pm["diverge_bucket"] == 1            # log2(3)
        assert pm["logprob_gap"] is None            # no logprobs in this script
        assert pm["cache_salt"] and pm["cache_salt_ctrl"]
        assert pm["cache_salt"] != pm["cache_salt_ctrl"]
        # the plain side matched the probed run, so the salted path is the mover
        assert pm["diverged_path"] == "salted"
        assert extra == {"request_id": "r0", "probe": "prefix_equiv",
                         "control_verdict": "cache_implicated"}

    @pytest.mark.asyncio
    async def test_diverged_path_cached_when_salted_matches_baseline(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(), self._arms("aa QQ cc", "aa bb cc"))
        assert captured[0][0].probe_metadata["diverged_path"] == "cached"

    @pytest.mark.asyncio
    async def test_finding_signs_as_pfx_divergent(self, tmp_path):
        from fuzzer.core.finding_sig import oracle_subtype

        captured, _ = await self._run(
            tmp_path, self._trace(), self._arms("aa bb cc", "aa XX cc"))
        finding = captured[0][0]
        sub = oracle_subtype(finding.bug_class, finding.detail,
                             finding.probe_metadata)
        assert sub == "pfx_divergent"

    @pytest.mark.asyncio
    async def test_mech_sig_differs_from_determinism_probe(self, tmp_path):
        """The two probes' findings must not collapse into one ledger entry."""
        from fuzzer.core.finding_sig import mechanism_signature

        trace = self._trace()
        captured, _ = await self._run(
            tmp_path, trace, self._arms("aa bb cc", "aa XX cc"))
        pfx_bug = captured[0][0]
        det_bug = OracleResult(
            passed=False, bug_class=BugClass.KV_CORRUPTION,
            detail="Non-deterministic output for identical prompt/temperature=0 "
                   "(possible KV cache corruption, similarity=0.50)",
            score=10.0,
            probe_metadata={"similarity": 0.5, "logprob_gap": 4.9},
        )
        assert mechanism_signature(trace, [pfx_bug]) != \
               mechanism_signature(trace, [det_bug])

    # -- the negative control (B vs C) --------------------------------------
    #
    # Two differently-salted arms are both pure recomputes of a byte-identical
    # prompt.  If they disagree, the engine is nondeterministic independently of
    # the prefix cache (or the probe perturbs its own measurement), and the
    # A-vs-B comparison proves nothing about the cache.

    @pytest.mark.asyncio
    async def test_salted_arms_disagreeing_is_not_a_cache_finding(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._arms("aa bb cc", "aa XX cc", "aa YY cc"))
        assert len(captured) == 1
        finding, extra = captured[0]
        assert finding.bug_class is BugClass.PROBE_SELF_CHECK
        assert finding.bug_class is not BugClass.KV_CORRUPTION
        pm = finding.probe_metadata
        assert pm["control_verdict"] == "b_ne_c"
        assert pm["bc_diverged"] is True
        # what A/B would have claimed is recorded but explicitly not acted on
        assert pm["ab_diverged"] is True
        assert extra["control_verdict"] == "b_ne_c"

    @pytest.mark.asyncio
    async def test_self_check_failure_scores_below_a_real_finding(self, tmp_path):
        """It reports on the instrument, not on the server under test."""
        bad, _ = await self._run(
            tmp_path / "a", self._trace(),
            self._arms("aa bb cc", "aa XX cc", "aa YY cc"))
        good, _ = await self._run(
            tmp_path / "b", self._trace(),
            self._arms("aa bb cc", "aa XX cc",
                       plain_kw={"cache_read_tokens": 100},
                       salt1_kw={"cache_read_tokens": 0}))
        assert bad[0][0].score < good[0][0].score

    @pytest.mark.asyncio
    async def test_self_check_fires_even_when_a_equals_b(self, tmp_path):
        """C alone disagreeing still invalidates the instrument."""
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._arms("aa bb cc", "aa bb cc", "aa YY cc"))
        assert len(captured) == 1
        pm = captured[0][0].probe_metadata
        assert pm["control_verdict"] == "b_ne_c"
        assert pm["ab_diverged"] is False

    @pytest.mark.asyncio
    async def test_self_check_does_not_sign_as_pfx_divergent(self, tmp_path):
        """It must be visibly distinct from the class it invalidates."""
        from fuzzer.core.finding_sig import mechanism_signature

        trace = self._trace()
        bad, _ = await self._run(
            tmp_path / "a", trace,
            self._arms("aa bb cc", "aa XX cc", "aa YY cc"))
        good, _ = await self._run(
            tmp_path / "b", trace, self._arms("aa bb cc", "aa XX cc"))
        assert mechanism_signature(trace, [bad[0][0]]) != \
               mechanism_signature(trace, [good[0][0]])

    @pytest.mark.asyncio
    async def test_bc_gap_is_recorded_on_both_verdicts(self, tmp_path):
        """"B/C gap at 1 ULP" and "B/C gap at 20 ULP" mean different things, so
        the pair is measured and reported whatever the verdict."""
        clean, _ = await self._run(
            tmp_path / "a", self._trace(), self._arms("aa bb cc", "aa XX cc"))
        assert clean[0][0].probe_metadata["bc_logprob_gap"] is None
        assert clean[0][0].probe_metadata["bc_similarity"] is None

        broken, _ = await self._run(
            tmp_path / "b", self._trace(),
            self._arms("aa bb cc", "aa XX cc", "aa YY cc",
                       salt1_kw={"output_tokens": ["aa", " XX", " cc"],
                                 "output_top_logprobs":
                                     [{}, {" XX": -0.1, " YY": -1.1}, {}]},
                       salt2_kw={"output_tokens": ["aa", " YY", " cc"],
                                 "output_top_logprobs":
                                     [{}, {" YY": -0.2, " XX": -1.4}, {}]}))
        pm = broken[0][0].probe_metadata
        assert pm["bc_logprob_gap"] == pytest.approx(1.0)
        assert pm["bc_logprob_gap_basis"] == "cross_arm"
        assert pm["bc_logprob_gap_ulps"] == pytest.approx(8.0)
        assert pm["bc_diverge_token_idx"] == 1
        assert pm["bc_similarity"] is not None

    # -- FP guards ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_sampling_event_is_skipped(self, tmp_path):
        captured, seen = await self._run(
            tmp_path, self._trace(temperature=0.8),
            self._arms("aa bb cc", "aa XX cc"))
        # no greedy SEND → no twins, no executor call at all
        assert captured == []
        assert seen == []

    @pytest.mark.asyncio
    async def test_n_gt_1_is_skipped(self, tmp_path):
        captured, seen = await self._run(
            tmp_path, self._trace(n_completions=2),
            self._arms("aa bb cc", "aa XX cc"))
        assert captured == [] and seen == []

    # -- arm symmetry -------------------------------------------------------
    #
    # The probe's entire claim is that the arms differ in ONE thing: the
    # cache_salt.  Anything else that ends up set on one arm and not another is
    # a silent confound -- the finding would then be "these requests disagreed",
    # not "the cache path disagreed with recompute", and nothing downstream can
    # tell the difference.  _determinism_probe has exactly this defect by
    # construction (it raises logprobs on run 2 only); this probe must not,
    # which is why the knobs are set on the source event BEFORE the twins are
    # deep-copied off it.

    _ARM_ASYMMETRIC_FIELDS = {"request_id", "cache_salt", "origin"}

    async def _arm_events(self, tmp_path, trace=None):
        _, seen = await self._run(tmp_path, trace or self._trace(),
                                  self._arms("x", "x"))
        return {e.request_id: e for e in seen[0].events}

    @pytest.mark.asyncio
    async def test_all_three_arms_identical_except_the_salt(self, tmp_path):
        import dataclasses
        import itertools

        events = await self._arm_events(tmp_path)
        assert len(events) == 3
        dicts = {rid: dataclasses.asdict(e) for rid, e in events.items()}
        for a, b in itertools.combinations(sorted(dicts), 2):
            differing = {k for k in dicts[a] if dicts[a][k] != dicts[b][k]}
            assert differing == self._ARM_ASYMMETRIC_FIELDS, (
                f"arms {a} and {b} differ in "
                f"{differing ^ self._ARM_ASYMMETRIC_FIELDS} beyond the "
                f"variable under test")

    @pytest.mark.asyncio
    async def test_logprobs_are_set_on_every_arm_not_just_one(self, tmp_path):
        """The specific asymmetry _determinism_probe has; must not recur here."""
        events = await self._arm_events(tmp_path)
        want = Runner._PFX_PROBE_LOGPROBS
        assert [e.logprobs for e in events.values()] == [want] * 3

    @pytest.mark.asyncio
    async def test_usage_reporting_is_asked_for_on_every_arm(self, tmp_path):
        """Without include_usage a STREAMED arm reports no usage at all, so
        cache_read_tokens is None and the vacuity control cannot run."""
        events = await self._arm_events(tmp_path)
        assert [e.include_usage for e in events.values()] == [True] * 3

    @pytest.mark.asyncio
    async def test_a_higher_logprobs_on_the_event_is_not_lowered(self, tmp_path):
        events = await self._arm_events(tmp_path, self._trace(logprobs=9))
        assert [e.logprobs for e in events.values()] == [9] * 3

    # -- vacuity control ----------------------------------------------------
    #
    # If a salted arm did not in fact bypass the cache, that arm took the same
    # path as the plain one and a divergence says nothing about the cache.  Such
    # a finding is vacuous and must never be filed.

    def _cache_reads(self, plain, salt1, salt2, texts=("aa bb cc", "aa XX cc")):
        return self._arms(texts[0], texts[1],
                          plain_kw={"cache_read_tokens": plain},
                          salt1_kw={"cache_read_tokens": salt1},
                          salt2_kw={"cache_read_tokens": salt2})

    @pytest.mark.asyncio
    async def test_salt_that_did_not_bypass_cache_is_skipped(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(), self._cache_reads(100, 100, 0))
        assert captured == []

    @pytest.mark.asyncio
    async def test_control_arm_that_did_not_bypass_is_also_vacuous(self, tmp_path):
        """C is only a negative control if IT recomputed too."""
        captured, _ = await self._run(
            tmp_path, self._trace(), self._cache_reads(100, 0, 100))
        assert captured == []

    @pytest.mark.asyncio
    async def test_vacuity_gate_beats_even_a_total_divergence(self, tmp_path):
        """A failed control is not a weak finding, it is no finding: the
        similarity of the texts is irrelevant once the comparison is vacuous."""
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._cache_reads(100, 100, 100, texts=("aa bb cc", "zz yy ww")))
        assert captured == []

    @pytest.mark.asyncio
    async def test_vacuity_gate_beats_a_self_check_failure(self, tmp_path):
        """Nothing downstream of the gate runs, self-check included."""
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._arms("aa bb cc", "aa XX cc", "aa YY cc",
                       plain_kw={"cache_read_tokens": 100},
                       salt1_kw={"cache_read_tokens": 100},
                       salt2_kw={"cache_read_tokens": 100}))
        assert captured == []

    @pytest.mark.asyncio
    async def test_plain_arm_that_never_hit_the_cache_is_skipped(self, tmp_path):
        """No arm read from the cache, so no arm exercised the path under test;
        a divergence here is ordinary nondeterminism (_determinism_probe's
        subject), not a cache bug."""
        captured, _ = await self._run(
            tmp_path, self._trace(), self._cache_reads(0, 0, 0))
        assert captured == []

    @pytest.mark.asyncio
    async def test_salt_that_did_bypass_cache_still_fires(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(), self._cache_reads(100, 0, 0))
        assert len(captured) == 1
        pm = captured[0][0].probe_metadata
        assert pm["salt_verified"] is True
        assert pm["vacuity_control"] == "verified"
        assert pm["cache_read_plain"] == 100
        assert pm["cache_read_salted"] == 0
        assert pm["cache_read_ctrl"] == 0
        assert captured[0][0].score == 12.0

    @pytest.mark.asyncio
    async def test_unreported_cache_read_tokens_leaves_salt_unverified(self, tmp_path):
        """salt_verified is None means "could not check", not "checked and
        failed".  Chosen behaviour: still file, but tagged and downgraded --
        suppressing here would turn one missing server flag into a probe that
        produces nothing all campaign with no signal that anything is wrong."""
        captured, _ = await self._run(
            tmp_path, self._trace(), self._arms("aa bb cc", "aa XX cc"))
        assert len(captured) == 1
        pm = captured[0][0].probe_metadata
        assert pm["salt_verified"] is None
        assert pm["vacuity_control"] == "unavailable"
        assert captured[0][0].score == 8.0

    @pytest.mark.asyncio
    async def test_partially_reported_cache_reads_are_unverified_not_failed(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._arms("aa bb cc", "aa XX cc",
                       plain_kw={"cache_read_tokens": 100}))
        assert len(captured) == 1
        assert captured[0][0].probe_metadata["salt_verified"] is None

    @pytest.mark.asyncio
    async def test_error_on_any_arm_is_skipped(self, tmp_path):
        """A build that rejects cache_salt 4xx's the twins — never a finding."""
        for kw in ({"salt1_kw": {"error": "400 Bad Request"}},
                   {"salt2_kw": {"error": "400 Bad Request"}}):
            captured, _ = await self._run(
                tmp_path, self._trace(),
                self._arms("aa bb cc", None, **kw))
            assert captured == []

    @pytest.mark.asyncio
    async def test_cancelled_request_is_skipped(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._arms("aa bb cc", "aa XX cc",
                       salt2_kw={"cancelled_at": 0.5}))
        assert captured == []

    @pytest.mark.asyncio
    async def test_executor_error_is_swallowed(self, tmp_path):
        runner = self._runner(tmp_path)

        async def _boom(*_a, **_kw):
            raise RuntimeError("server gone")

        runner.executor.run = _boom
        first = self._result(self._req("r0", "aa bb cc"))
        await runner._prefix_equivalence_probe(self._trace(), first, 5)

    # -- FP adjudication ----------------------------------------------------
    #
    # Greedy argmax is not a stable function of logits that differ in the low
    # bits, and reading a prefix from cache vs recomputing it IS a low-bit
    # difference.  The gap between one arm's top-1 and the OTHER arm's token at
    # the divergence position says whether the flip needed a real bug or just
    # one ULP.  Defaults here: bf16 at |logit|=16 -> 1 ULP = 0.125, band = 2 ULP.

    def _lp_arms(self, tops_plain=None, tops_salted=None,
                 toks_plain=("aa", " bb", " cc"),
                 toks_salted=("aa", " XX", " cc")):
        """A/B differ; C is scripted byte-identical to B so the control passes."""
        return self._arms(
            "aa bb cc", "aa XX cc",
            plain_kw={"cache_read_tokens": 100,
                      "output_tokens": list(toks_plain),
                      "output_top_logprobs": tops_plain},
            salt1_kw={"cache_read_tokens": 0,
                      "output_tokens": list(toks_salted),
                      "output_top_logprobs": tops_salted},
            salt2_kw={"cache_read_tokens": 0,
                      "output_tokens": list(toks_salted),
                      "output_top_logprobs": tops_salted})

    @pytest.mark.asyncio
    async def test_one_ulp_cross_arm_gap_is_suppressed(self, tmp_path):
        """0.125 = exactly one bf16 ULP at |logit|=16: a benign flip."""
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._lp_arms(
                tops_plain=[{}, {" bb": -0.100, " XX": -0.225}, {}],
                tops_salted=[{}, {" XX": -0.100, " bb": -0.900}, {}]))
        assert captured == []

    @pytest.mark.asyncio
    async def test_wide_cross_arm_gap_still_files_and_carries_the_gap(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._lp_arms(
                tops_plain=[{}, {" bb": -0.1, " XX": -1.1}, {}],
                tops_salted=[{}, {" XX": -0.2, " bb": -1.4}, {}]))
        assert len(captured) == 1
        pm = captured[0][0].probe_metadata
        assert pm["logprob_gap"] == pytest.approx(1.0)
        assert pm["logprob_gap_basis"] == "cross_arm"
        assert pm["logprob_gap_arm"] == "plain"      # the narrower of the two
        assert pm["logprob_gap_ulps"] == pytest.approx(8.0)
        assert pm["logits_dtype"] == "bfloat16"
        assert pm["logprobs_requested"] == Runner._PFX_PROBE_LOGPROBS

    @pytest.mark.asyncio
    async def test_a_near_tie_on_either_arm_suppresses(self, tmp_path):
        """One arm being a coin flip is enough to explain the divergence, even
        if the other arm's landscape looks decisive."""
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._lp_arms(
                tops_plain=[{}, {" bb": -0.1, " XX": -5.0}, {}],
                tops_salted=[{}, {" XX": -0.1, " bb": -0.2}, {}]))
        assert captured == []

    @pytest.mark.asyncio
    async def test_a_bound_gap_never_suppresses(self, tmp_path):
        """When the other arm's token is outside this arm's top-k, top1-topk is
        only a LOWER bound on the true cross-arm gap.  bound <= threshold does
        not imply true_gap <= threshold, so suppressing on it would silently
        discard the findings where the competitor ranked far down -- the
        interesting ones."""
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._lp_arms(
                tops_plain=[{}, {" bb": -0.10, "zz": -0.15}, {}],
                tops_salted=None))
        assert len(captured) == 1
        pm = captured[0][0].probe_metadata
        assert pm["logprob_gap_basis"] == "topk_bound"
        assert pm["logprob_gap"] == pytest.approx(0.05)

    @pytest.mark.asyncio
    async def test_diverge_token_idx_comes_from_the_token_streams(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(), self._lp_arms())
        pm = captured[0][0].probe_metadata
        assert pm["diverge_token_idx"] == 1
        assert pm["diverge_char_idx"] == 3      # char index is kept too

    @pytest.mark.asyncio
    async def test_diverge_token_idx_when_one_arm_stops_early(self, tmp_path):
        """One stream being a strict prefix of the other diverges at the point
        where one arm stopped and the other did not."""
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._lp_arms(toks_plain=("aa", " bb"),
                          toks_salted=("aa", " bb", " cc")))
        assert captured[0][0].probe_metadata["diverge_token_idx"] == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_char_remap_when_one_arm_has_no_tokens(self, tmp_path):
        captured, _ = await self._run(
            tmp_path, self._trace(),
            self._arms("aa bb cc", "aa XX cc",
                       plain_kw={"cache_read_tokens": 100},
                       salt1_kw={"cache_read_tokens": 0,
                                 "output_tokens": ["aa", " XX", " cc"]},
                       salt2_kw={"cache_read_tokens": 0,
                                 "output_tokens": ["aa", " XX", " cc"]}))
        pm = captured[0][0].probe_metadata
        assert pm["diverge_token_idx"] == 1
        assert pm["logprob_gap"] is None

    # -- reproducibility ----------------------------------------------------

    def test_cache_salt_is_seed_deterministic(self, tmp_path):
        a = self._runner(tmp_path / "a", seed=7)
        b = self._runner(tmp_path / "b", seed=7)
        c = self._runner(tmp_path / "c", seed=8)
        salts_a = [a._make_cache_salt("r0") for _ in range(3)]
        salts_b = [b._make_cache_salt("r0") for _ in range(3)]
        salts_c = [c._make_cache_salt("r0") for _ in range(3)]
        assert salts_a == salts_b
        assert salts_a != salts_c
        assert len(set(salts_a)) == 3          # fresh salt per arm

    @pytest.mark.asyncio
    async def test_both_arms_salts_replay_from_the_seed(self, tmp_path):
        """The whole three-arm trace must be reconstructible from the seed."""
        async def _salts(where, seed):
            _, seen = await self._run(where, self._trace(),
                                      self._arms("x", "x"), seed=seed)
            return {e.request_id: e.cache_salt for e in seen[0].events}

        assert await _salts(tmp_path / "a", 7) == await _salts(tmp_path / "b", 7)
        assert await _salts(tmp_path / "c", 7) != await _salts(tmp_path / "d", 8)

    def test_probe_rng_does_not_disturb_the_mutation_rng(self, tmp_path):
        a = self._runner(tmp_path / "a", seed=11)
        b = self._runner(tmp_path / "b", seed=11)
        for _ in range(5):
            b._make_cache_salt("r0")
        assert [a.rng.random() for _ in range(3)] == \
               [b.rng.random() for _ in range(3)]

    def test_probe_is_off_by_default(self):
        assert RunnerConfig().prefix_equiv_probe_every == 0
        assert RunnerConfig(prefix_equiv_probe_every=25).prefix_equiv_probe_every == 25


# ---------------------------------------------------------------------------
# Workstream 1: fingerprint version plumbing + transition threading
# ---------------------------------------------------------------------------

class TestFingerprintVersionWiring:
    """`--fingerprint-version` reaches the Corpus through GRIEF_FINGERPRINT_VERSION.

    Same stopgap route as `--selection-policy`: the backend CLI publishes the
    flag as an env var in build_adapter(), the Corpus resolves it in __init__,
    and the Runner reads the resolved value back off the Corpus.
    """

    def _runner(self, tmp_path):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor
        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        config = RunnerConfig(seeds_dir=str(tmp_path / "seeds"),
                              corpus_dir=str(tmp_path / "corpus"),
                              findings_dir=str(tmp_path / "findings"))
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def test_runner_corpus_defaults_to_v1(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GRIEF_FINGERPRINT_VERSION", raising=False)
        assert self._runner(tmp_path).corpus.fingerprint_version == 1

    def test_env_var_reaches_the_runners_corpus(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_FINGERPRINT_VERSION", "2")
        assert self._runner(tmp_path).corpus.fingerprint_version == 2

    def test_cli_flag_publishes_the_env_var(self, monkeypatch):
        import argparse
        from fuzzer.backends.vllm import cli as vllm_cli
        monkeypatch.delenv("GRIEF_FINGERPRINT_VERSION", raising=False)
        p = argparse.ArgumentParser()
        vllm_cli.register_args(p)
        args = p.parse_args(["--fingerprint-version", "2"])
        vllm_cli.build_adapter(args)
        import os
        assert os.environ["GRIEF_FINGERPRINT_VERSION"] == "2"

    def test_cli_default_publishes_nothing(self, monkeypatch):
        import argparse, os
        from fuzzer.backends.vllm import cli as vllm_cli
        monkeypatch.delenv("GRIEF_FINGERPRINT_VERSION", raising=False)
        p = argparse.ArgumentParser()
        vllm_cli.register_args(p)
        vllm_cli.build_adapter(p.parse_args([]))
        assert "GRIEF_FINGERPRINT_VERSION" not in os.environ

    def test_sglang_backend_publishes_it_too(self, monkeypatch):
        import argparse, os
        from fuzzer.backends.vllm import cli as vllm_cli
        from fuzzer.backends.sglang import cli as sglang_cli
        monkeypatch.delenv("GRIEF_FINGERPRINT_VERSION", raising=False)
        p = argparse.ArgumentParser()
        vllm_cli.register_args(p)
        sglang_cli.register_args(p)
        sglang_cli.build_adapter(p.parse_args(["--fingerprint-version", "2"]))
        assert os.environ["GRIEF_FINGERPRINT_VERSION"] == "2"


class TestTransitionThreading:
    """The run loop hands corpus.add() the *previous* iteration's fingerprint."""

    def _runner(self, tmp_path, **overrides):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor
        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        config = RunnerConfig(seeds_dir=str(tmp_path / "seeds"),
                              corpus_dir=str(tmp_path / "corpus"),
                              findings_dir=str(tmp_path / "findings"),
                              **overrides)
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def test_last_fingerprint_starts_as_none(self, tmp_path):
        assert self._runner(tmp_path)._last_fingerprint is None

    def test_corpus_add_accepts_prev_fingerprint(self, tmp_path):
        """A None predecessor is a valid key, not an error — it is the entry edge."""
        runner = self._runner(tmp_path)
        from fuzzer.monitor import SchedulerFingerprint
        fp = SchedulerFingerprint(False, False, 1, 0, 0, 0, 0, 0)
        runner.corpus.add(Trace(seed=1, events=[]), [], fingerprint=fp,
                          prev_fingerprint=None, save=False)
        assert runner.corpus._seen_transitions[(None, fp)] == 1


class TestReverifyMinimized:
    """The minimize re-verification block that never ran.

    It built ``MonitorSession([], [])`` — two positional arguments to a
    one-field dataclass — and the resulting TypeError was swallowed by a bare
    ``except Exception: pass``, so the block has never executed in any campaign.
    The call is fixed; the block is opt-in because running it changes which
    findings are filed.
    """

    def _runner(self, tmp_path, **overrides):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor
        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(parents=True, exist_ok=True)
        config = RunnerConfig(seeds_dir=str(tmp_path / "seeds"),
                              corpus_dir=str(tmp_path / "corpus"),
                              findings_dir=str(tmp_path / "findings"),
                              **overrides)
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    def _trace(self, seed=1):
        return Trace(seed=seed, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r0",
                  prompt_len=128, max_tokens=8, stream=True)])

    def test_monitor_session_takes_one_field(self):
        """The original call could only ever raise."""
        import dataclasses
        from fuzzer.monitor import MonitorSession
        assert [f.name for f in dataclasses.fields(MonitorSession)] == ["snapshots"]
        with pytest.raises(TypeError):
            MonitorSession([], [])

    def test_default_is_off(self):
        assert RunnerConfig().reverify_minimized is False

    @pytest.mark.asyncio
    async def test_off_does_not_re_run_the_minimized_trace(self, tmp_path):
        runner = self._runner(tmp_path)
        minimized = self._trace(seed=2)
        runs = []

        async def _fake_minimize(trace, bugs):
            return minimized

        async def _count(*a, **kw):
            runs.append(a)
            raise RuntimeError("should not be reached")

        runner._minimize = _fake_minimize
        runner.executor.run = _count
        bugs = [OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK,
                             detail="GPU memory grew by 148.0 MB", score=5.0)]
        path = await runner._handle_finding(self._trace(), bugs, 3)
        assert runs == []                    # no extra execution
        assert path is not None
        report = json.loads(path.read_text())
        assert [b["class"] for b in report["bugs"]] == ["memory_leak"]

    @pytest.mark.asyncio
    async def test_reverification_preserves_original_evidence(self, tmp_path):
        runner = self._runner(tmp_path, reverify_minimized=True)
        minimized = self._trace(seed=2)
        runs = []

        async def _fake_minimize(trace, bugs):
            return minimized

        async def _fake_run(trace, *a, **kw):
            from fuzzer.core.types import TraceResult
            runs.append(trace)
            return TraceResult(trace_seed=trace.seed, started_at=0.0,
                               finished_at=1.0, requests=[])

        replacement = [OracleResult(passed=False, bug_class=BugClass.KV_BLOCK_LEAK,
                                    detail="kv usage stayed high", score=6.0)]
        runner._minimize = _fake_minimize
        runner.executor.run = _fake_run
        runner.oracle.evaluate = lambda *a, **kw: list(replacement)

        bugs = [OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK,
                             detail="GPU memory grew by 148.0 MB", score=5.0)]
        path = await runner._handle_finding(self._trace(), bugs, 3)
        assert len(runs) == 1                # exactly one extra execution
        report = json.loads(path.read_text())
        assert [b["class"] for b in report["bugs"]] == ["memory_leak"]
        assert report["minimized_validation"]["status"] == "not_reproduced"
        assert report["minimized_validation"]["observed_classes"] == ["kv_block_leak"]

    @pytest.mark.asyncio
    async def test_on_keeps_the_original_bugs_when_the_re_run_is_clean(self, tmp_path):
        runner = self._runner(tmp_path, reverify_minimized=True)

        async def _fake_minimize(trace, bugs):
            return self._trace(seed=2)

        async def _fake_run(trace, *a, **kw):
            from fuzzer.core.types import TraceResult
            return TraceResult(trace_seed=trace.seed, started_at=0.0,
                               finished_at=1.0, requests=[])

        runner._minimize = _fake_minimize
        runner.executor.run = _fake_run
        runner.oracle.evaluate = lambda *a, **kw: []

        bugs = [OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK,
                             detail="GPU memory grew by 148.0 MB", score=5.0)]
        report = json.loads(
            (await runner._handle_finding(self._trace(), bugs, 3)).read_text())
        assert [b["class"] for b in report["bugs"]] == ["memory_leak"]

    @pytest.mark.asyncio
    async def test_on_reports_instead_of_silently_swallowing(self, tmp_path, capsys):
        runner = self._runner(tmp_path, reverify_minimized=True)

        async def _fake_minimize(trace, bugs):
            return self._trace(seed=2)

        async def _boom(*a, **kw):
            raise RuntimeError("no server")

        runner._minimize = _fake_minimize
        runner.executor.run = _boom
        bugs = [OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK,
                             detail="GPU memory grew by 148.0 MB", score=5.0)]
        path = await runner._handle_finding(self._trace(), bugs, 3)
        assert path is not None              # the finding is still filed
        assert json.loads(path.read_text())["minimized_validation"] == {"status": "error", "reason": "no server"}

    @pytest.mark.asyncio
    async def test_on_passes_a_live_monitor_session(self, tmp_path):
        """The original code passed an empty session, which would have made
        memory_leak / scheduler_hang / kv_block_leak — 3 of the 5 classes in
        _ALWAYS_MINIMIZE — structurally unable to re-verify themselves."""
        runner = self._runner(tmp_path, reverify_minimized=True)
        seen_sessions = []

        async def _fake_minimize(trace, bugs):
            return self._trace(seed=2)

        async def _fake_run(trace, *a, **kw):
            from fuzzer.core.types import TraceResult
            return TraceResult(trace_seed=trace.seed, started_at=0.0,
                               finished_at=1.0, requests=[])

        def _evaluate(result, session):
            seen_sessions.append(session)
            return []

        runner._minimize = _fake_minimize
        runner.executor.run = _fake_run
        runner.oracle.evaluate = _evaluate
        bugs = [OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK,
                             detail="GPU memory grew by 148.0 MB", score=5.0)]
        await runner._handle_finding(self._trace(), bugs, 3)
        from fuzzer.monitor import MonitorSession
        assert len(seen_sessions) == 1
        assert isinstance(seen_sessions[0], MonitorSession)


# ---------------------------------------------------------------------------
# FP tie-break banding (Fix 2)
# ---------------------------------------------------------------------------

class TestFpTieUlpBanding:
    """Regression: the FP tie-break threshold must be in ULPs, not nats.

    The old rule was `gap < _FP_GAP_THRESHOLD` with the threshold a fixed
    1e-3.  The top1-top2 logprob gap is `logit_1 - logit_2` exactly
    (`log_softmax` subtracts the same logsumexp from both), and those logits
    reach the sampler in the model's own dtype — verified in
    the vLLM 0.27.1 container: `--dtype auto` resolves to the
    checkpoint's `torch_dtype` (`config/model.py:2130,2174`), the LM head keeps
    it (`config/model.py:2236-2242`, `logits_processor.py:105-108` takes the
    no-cast branch), and `v1/sample/sampler.py:305-306` calls
    `logits.log_softmax(dim=-1, dtype=torch.float32)` on the *bf16* tensor
    (the `logits.to(torch.float32)` at `sampler.py:95-96` runs afterwards).
    `VLLM_BATCH_INVARIANT=1` keeps `a.dtype` (`batch_invariant.py:144-146`).

    Qwen2.5-0.5B-Instruct declares `"torch_dtype": "bfloat16"`, so the smallest
    achievable gap is one bf16 ULP — 0.125 at |logit| in [16,32).  The old
    threshold sat three orders of magnitude below that and could never fire:
    all 45 tie findings in the last campaign carried `logprob_gap == 0.125`.
    """

    def _runner(self, **overrides):
        config = RunnerConfig(**overrides)
        runner = Runner.__new__(Runner)
        runner.config = config
        return runner

    # -- the ULP arithmetic ------------------------------------------------

    @pytest.mark.parametrize("dtype,magnitude,expected", [
        # significand bits: bf16 8, fp16 11, fp32 24 → ulp = 2**(e-(p-1))
        ("bfloat16",  1.0, 2.0 ** -7),
        ("bfloat16",  8.0, 0.0625),
        ("bfloat16", 16.0, 0.125),
        ("bfloat16", 32.0, 0.25),
        ("bfloat16", 64.0, 0.5),
        ("float16",  16.0, 0.015625),
        ("float16",  32.0, 0.03125),
        ("float32",  16.0, 2.0 ** -19),
        ("float32",  32.0, 2.0 ** -18),
    ])
    def test_ulp_table(self, dtype, magnitude, expected):
        assert Runner._dtype_ulp(dtype, magnitude) == expected

    def test_unknown_dtype_falls_back_to_float32(self):
        """Conservative: the finest grid suppresses the least."""
        assert (Runner._dtype_ulp("mxfp4", 16.0)
                == Runner._dtype_ulp("float32", 16.0))

    @pytest.mark.parametrize("magnitude", [0.0, -1.0, float("nan"), float("inf")])
    def test_degenerate_magnitudes_do_not_raise(self, magnitude):
        assert Runner._dtype_ulp("bfloat16", magnitude) > 0.0

    # -- dtype discovery ---------------------------------------------------

    @pytest.mark.parametrize("flags,expected", [
        ("", "bfloat16"),
        ("--enable-prefix-caching", "bfloat16"),
        ("--dtype auto", "bfloat16"),
        ("--dtype float16", "float16"),
        ("--dtype=float16", "float16"),
        ("--enable-prefix-caching --dtype float32 --max-model-len 4096", "float32"),
    ])
    def test_logits_dtype_from_server_flags(self, flags, expected):
        assert self._runner(server_flags=flags)._logits_dtype() == expected

    # -- the threshold itself ---------------------------------------------

    def test_threshold_is_above_one_bf16_ulp_and_below_the_narrowest_real_gap(self):
        thr = self._runner()._fp_tie_threshold()
        assert 0.125 < thr < 0.5

    def test_old_fixed_threshold_could_not_fire(self):
        """The defect: 1e-3 sits three orders of magnitude below the grid."""
        assert 0.001 < Runner._dtype_ulp("bfloat16", 16.0) / 100

    def test_fp16_and_fp32_paths_give_a_much_tighter_threshold(self):
        """A model actually served in fp16/fp32 must not inherit bf16's band."""
        bf16 = self._runner(server_flags="--dtype bfloat16")._fp_tie_threshold()
        fp16 = self._runner(server_flags="--dtype float16")._fp_tie_threshold()
        fp32 = self._runner(server_flags="--dtype float32")._fp_tie_threshold()
        assert fp32 < fp16 < bf16
        assert fp16 < 0.125          # a 1-bf16-ULP gap is real news in fp16

    # -- the campaign data -------------------------------------------------

    #: Every distinct `logprob_gap` recorded by the determinism probe across
    #: four earlier tuning campaigns' findings,
    #: with its multiplicity.  45 sit at exactly one bf16 ULP (one of them
    #: carrying a single fp32 ULP of logsumexp noise, 2**-23); the other 18
    #: span 4.0-23.5 ULPs.
    _CAMPAIGN_GAPS_TIE = {0.125: 44, 0.12500011920928955: 1}
    _CAMPAIGN_GAPS_WIDE = {
        0.5: 2, 0.5000001192092896: 1, 0.6249998807907104: 1, 0.625: 2,
        0.75: 1, 0.875: 1, 0.9374998807907104: 1, 1.0000000596046448: 1,
        1.125: 1, 1.5625001192092896: 1, 1.6250000596046448: 1,
        1.7500000596046448: 1, 1.8750000298023224: 1, 1.8750001192092896: 1,
        2.2499999403953552: 1, 2.9374999403953552: 1,
    }

    def test_suppresses_all_45_single_ulp_findings(self):
        thr = self._runner()._fp_tie_threshold()
        assert sum(self._CAMPAIGN_GAPS_TIE.values()) == 45
        for gap in self._CAMPAIGN_GAPS_TIE:
            assert gap <= thr, gap

    def test_keeps_all_18_wide_findings(self):
        """Not a blanket suppression — 4 ULPs and up still file."""
        thr = self._runner()._fp_tie_threshold()
        assert sum(self._CAMPAIGN_GAPS_WIDE.values()) == 18
        for gap in self._CAMPAIGN_GAPS_WIDE:
            assert gap > thr, gap

    def test_the_2_to_4_ulp_band_is_empty_in_the_data(self):
        """Why _FP_TIE_MAX_ULPS = 2 is defensible: nothing observed between
        1 ULP and 4 ULPs, so the cut has a 2x margin on both sides."""
        ulp = Runner._dtype_ulp("bfloat16", Runner._FP_LOGIT_MAGNITUDE)
        observed = sorted({**self._CAMPAIGN_GAPS_TIE, **self._CAMPAIGN_GAPS_WIDE})
        in_band = [g for g in observed if 1.5 * ulp < g < 3.5 * ulp]
        assert in_band == []

    def test_reference_magnitude_has_headroom_downwards_only(self):
        """The margin on `_FP_LOGIT_MAGNITUDE` is asymmetric, which is why it is
        pinned by measurement rather than guessed.

        At |logit| in [8,16) the split is unchanged, so under-estimating by a
        binade is free.  At [32,64) one ULP is 0.25 and the 2-ULP band reaches
        0.5 — exactly the narrowest genuine divergence in the data — so
        over-estimating starts eating real findings.  The measurement (top
        logits in [8,32), observed gap grid a multiple of 0.0625 and dominated
        by 0.125) puts the top-2 pair in [16,32).
        """
        for magnitude in (8.0, 16.0):
            thr = 2.0 * Runner._dtype_ulp("bfloat16", magnitude) + 2.0 ** -23
            assert max(self._CAMPAIGN_GAPS_TIE) <= thr
            assert min(self._CAMPAIGN_GAPS_WIDE) > thr

        too_high = 2.0 * Runner._dtype_ulp("bfloat16", 32.0) + 2.0 ** -23
        assert min(self._CAMPAIGN_GAPS_WIDE) <= too_high


# ---------------------------------------------------------------------------
# Minimized trace vs. un-minimized bugs (Fix 3)
# ---------------------------------------------------------------------------

class TestFindingKeepsTheTraceItsBugsWereObservedOn:
    """Regression: `_handle_finding` wrote post-minimization events beside
    pre-minimization bugs.

    16 of the last campaign's 101 `degenerate_mask` findings — and all 3
    `schema_violation` ones, e.g.
    finding_00590_471613229_0056.json of one earlier campaign, whose bugs all name request `p7` while
    its stored trace holds only `q2_b_s` and `q4_storm_s` — were un-triageable
    without going back to the corpus.
    """

    def _runner(self, tmp_path, **overrides):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor

        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(exist_ok=True)
        config = RunnerConfig(
            seeds_dir=str(tmp_path / "seeds"),
            corpus_dir=str(tmp_path / "corpus"),
            findings_dir=str(tmp_path / "findings"),
            **overrides,
        )
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    @staticmethod
    def _send(rid):
        return Event(action=ActionType.SEND, offset_ms=0, request_id=rid,
                     prompt_len=64, max_tokens=8, stream=True)

    async def _report(self, tmp_path, *, minimize):
        original = Trace(seed=7, events=[self._send("p7"), self._send("q2_b_s"),
                                         self._send("q4_storm_s")])
        minimized = Trace(seed=7, events=[self._send("q2_b_s"),
                                          self._send("q4_storm_s")])

        runner = self._runner(tmp_path)

        async def _fake_minimize(trace, bugs):
            return minimized if minimize else trace

        async def _no_server(*_a, **_kw):
            raise RuntimeError("no server in unit tests")

        runner._minimize = _fake_minimize
        runner.executor.run = _no_server

        # MEMORY_LEAK is in _ALWAYS_MINIMIZE.
        bugs = [OracleResult(
            passed=False, bug_class=BugClass.MEMORY_LEAK,
            detail="structured output violated its json_schema constraint on "
                   "request p7 (finish_reason=stop, failure=json_parse)",
            score=5.0, probe_metadata={"request_id": "p7"})]
        path = await runner._handle_finding(original, bugs, 9)
        return json.loads(path.read_text()), original, minimized

    @pytest.mark.asyncio
    async def test_original_trace_is_kept_beside_the_minimized_one(self, tmp_path):
        report, original, minimized = await self._report(tmp_path, minimize=True)
        # `trace` still means "smallest reproducer" — offline reproduction
        # tools depend on that.
        assert report["trace"]["events"] == minimized.to_dict()["events"]
        assert report["trace_original"]["events"] == original.to_dict()["events"]
        assert report["minimized"] is True

    @pytest.mark.asyncio
    async def test_every_bug_request_id_resolves_against_one_of_the_traces(self, tmp_path):
        report, _, _ = await self._report(tmp_path, minimize=True)
        rids = {e.get("request_id")
                for e in report["trace_original"]["events"]}
        for bug in report["bugs"]:
            assert bug["probe_metadata"]["request_id"] in rids

    @pytest.mark.asyncio
    async def test_orphaned_request_ids_are_named(self, tmp_path):
        report, _, _ = await self._report(tmp_path, minimize=True)
        assert report["bugs_request_ids_not_in_trace"] == ["p7"]

    @pytest.mark.asyncio
    async def test_unminimized_findings_do_not_grow(self, tmp_path):
        """No `trace_original` when minimization changed nothing — findings
        that were never minimized keep their old size exactly."""
        report, _, _ = await self._report(tmp_path, minimize=False)
        assert "trace_original" not in report
        assert "bugs_request_ids_not_in_trace" not in report
        assert report["minimized"] is False

    @pytest.mark.asyncio
    async def test_archived_findings_without_the_key_still_parse(self, tmp_path):
        """Backward compatibility: `trace_original` is purely additive, so
        the offline dedup validator's `bugs`/`trace` contract is untouched."""
        from fuzzer.core.finding_sig import mechanism_signature_from_dict
        report, _, minimized = await self._report(tmp_path, minimize=True)
        legacy = {k: v for k, v in report.items()
                  if k not in ("trace_original", "minimized",
                               "bugs_request_ids_not_in_trace")}
        assert "bugs" in legacy and "trace" in legacy
        assert (mechanism_signature_from_dict(legacy)
                == mechanism_signature_from_dict(report))


# ---------------------------------------------------------------------------
# Measurement gaps
#
# An oracle that declines to judge must leave a trace.  The kv_block_leak
# false-positive class was an unmeasured quantity reported as a measurement;
# gating the oracle without surfacing the ungated case would only move the
# silence downstream — a campaign would show zero kv_block_leak findings with
# no way to tell "no leaks" from "never once checked".
# ---------------------------------------------------------------------------

class TestMeasurementGaps:

    def _runner(self, tmp_path):
        from fuzzer.monitor import MockMonitor
        from fuzzer.executor import PythonExecutor

        for name in ("seeds", "corpus", "findings"):
            (tmp_path / name).mkdir(exist_ok=True)
        config = RunnerConfig(
            seeds_dir=str(tmp_path / "seeds"),
            corpus_dir=str(tmp_path / "corpus"),
            findings_dir=str(tmp_path / "findings"),
        )
        return Runner(config, executor=PythonExecutor(), monitor=MockMonitor())

    @staticmethod
    def _busy_session():
        import time as _t
        from fuzzer.monitor import MetricSnapshot, MonitorSession
        s = MonitorSession()
        s.snapshots = [
            MetricSnapshot(timestamp=_t.time(), gpu_memory_used_mb=4000.0,
                           kv_cache_usage_pct=0.0,
                           requests_running=0, requests_waiting=0),
            MetricSnapshot(timestamp=_t.time(), gpu_memory_used_mb=4000.0,
                           kv_cache_usage_pct=89.5,
                           requests_running=3, requests_waiting=7),
        ]
        return s

    @staticmethod
    def _trace():
        return Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r0",
                  prompt_len=128, max_tokens=8, prefix_len=0, stream=True),
        ])

    def _evaluate_busy(self, runner):
        from fuzzer.executor import TraceResult
        import time as _t
        result = TraceResult(trace_seed=1, started_at=_t.time() - 1.0,
                             finished_at=_t.time(), requests=[])
        runner.oracle.evaluate(result, self._busy_session())
        runner._record_measurement_gaps(iteration=7)

    def test_gap_is_counted_and_announced_once(self, tmp_path, capsys):
        runner = self._runner(tmp_path)
        self._evaluate_busy(runner)
        first = capsys.readouterr().out
        assert "UNMEASURED" in first
        assert "kv_block_leak" in first

        self._evaluate_busy(runner)
        second = capsys.readouterr().out
        assert "UNMEASURED" not in second      # printed once, then counted

        total = sum(runner._measurement_gaps.values())
        assert total == 2
        assert any("engine not drained" in reason
                   for reason in runner._measurement_gaps)

    @pytest.mark.asyncio
    async def test_gap_is_recorded_in_the_finding_json(self, tmp_path):
        runner = self._runner(tmp_path)
        self._evaluate_busy(runner)
        bug = OracleResult(passed=False, bug_class=BugClass.KV_CORRUPTION,
                           detail="(possible KV cache corruption)", score=10.0)
        path = await runner._handle_finding(self._trace(), [bug], 7)
        report = json.loads(path.read_text())
        assert any("kv_block_leak: engine not drained" in g
                   for g in report["measurement_gaps"])

    @pytest.mark.asyncio
    async def test_clean_session_leaves_no_gap_key(self, tmp_path):
        """Purely additive: a fully measured session's finding is unchanged."""
        runner = self._runner(tmp_path)
        runner._iter_measurement_gaps = []
        bug = OracleResult(passed=False, bug_class=BugClass.KV_CORRUPTION,
                           detail="(possible KV cache corruption)", score=10.0)
        path = await runner._handle_finding(self._trace(), [bug], 3)
        assert "measurement_gaps" not in json.loads(path.read_text())

    def test_ledger_carries_the_gap_counts(self, tmp_path):
        runner = self._runner(tmp_path)
        self._evaluate_busy(runner)
        runner._dump_signature_ledger()
        ledger = json.loads(
            (tmp_path / "findings" / "signature_ledger.json").read_text())
        assert sum(ledger["measurement_gaps"].values()) == 1
