"""Tests for VllmLoRAAdapter — no GPU or server required."""

from __future__ import annotations

import pytest
from fuzzer.adapters.vllm_lora.adapter import VllmLoRAAdapter
from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
from fuzzer.core.adapter import TargetAdapter
from fuzzer.core.types import BugClass
from fuzzer.trace import Event, ActionType


ADAPTERS = {"lora_a": "/tmp/lora_a", "lora_b": "/tmp/lora_b"}


def make_send(adapter_name=None, prompt_len=64, prefix_len=0):
    return Event(
        action=ActionType.SEND,
        offset_ms=0,
        request_id="r1",
        prompt_len=prompt_len,
        max_tokens=32,
        prefix_len=prefix_len,
        adapter_name=adapter_name,
    )


# ---------------------------------------------------------------------------
# Inheritance and contract
# ---------------------------------------------------------------------------

def test_inherits_vllm_base():
    a = VllmLoRAAdapter(ADAPTERS)
    assert isinstance(a, VllmBaseAdapter)
    assert isinstance(a, TargetAdapter)


def test_requires_at_least_two_adapters():
    with pytest.raises(ValueError, match="at least 2"):
        VllmLoRAAdapter({"lora_a": "/tmp/lora_a"})


def test_adapter_names_sorted():
    a = VllmLoRAAdapter({"lora_b": "/b", "lora_a": "/a"})
    assert a.adapter_names == ["lora_a", "lora_b"]


# ---------------------------------------------------------------------------
# server_flags_extra
# ---------------------------------------------------------------------------

def test_server_flags_extra_contains_enable_lora():
    a = VllmLoRAAdapter(ADAPTERS)
    flags = a.server_flags_extra()
    assert "--enable-lora" in flags


def test_server_flags_extra_contains_all_modules():
    a = VllmLoRAAdapter(ADAPTERS)
    flags = a.server_flags_extra()
    assert "lora_a=/tmp/lora_a" in flags
    assert "lora_b=/tmp/lora_b" in flags


def test_server_flags_extra_empty_for_base():
    a = VllmBaseAdapter()
    assert a.server_flags_extra() == ""


# ---------------------------------------------------------------------------
# build_payload — adapter_name injection
# ---------------------------------------------------------------------------

def test_build_payload_injects_adapter_name():
    a = VllmLoRAAdapter(ADAPTERS)
    event = make_send(adapter_name="lora_a")
    endpoint, payload = a.build_payload(event)
    assert payload["model"] == "lora_a"


def test_build_payload_no_adapter_name_uses_base_model():
    a = VllmLoRAAdapter(ADAPTERS)
    event = make_send(adapter_name=None)
    endpoint, payload = a.build_payload(event)
    # Should not override model with None — base adapter handles model name
    assert payload.get("model") != "lora_a"
    assert payload.get("model") != "lora_b"


def test_build_payload_returns_completions_endpoint():
    a = VllmLoRAAdapter(ADAPTERS)
    event = make_send(adapter_name="lora_b")
    endpoint, payload = a.build_payload(event)
    assert "/completions" in endpoint


# ---------------------------------------------------------------------------
# extra_mutations
# ---------------------------------------------------------------------------

def test_extra_mutations_includes_lora_mutations():
    from fuzzer.adapters.vllm_lora.mutators import LORA_MUTATIONS
    a = VllmLoRAAdapter(ADAPTERS)
    extras = a.extra_mutations()
    for mut in LORA_MUTATIONS:
        assert mut in extras


def test_extra_mutations_includes_base_mutations():
    base = VllmBaseAdapter()
    lora = VllmLoRAAdapter(ADAPTERS)
    base_extras = set(base.extra_mutations())
    lora_extras  = set(lora.extra_mutations())
    assert base_extras.issubset(lora_extras)


def test_lora_has_more_mutations_than_base():
    base = VllmBaseAdapter()
    lora = VllmLoRAAdapter(ADAPTERS)
    assert len(lora.extra_mutations()) > len(base.extra_mutations())


# ---------------------------------------------------------------------------
# Runner integration — server_flags_extra appended
# ---------------------------------------------------------------------------

def test_runner_appends_lora_flags():
    from fuzzer.core.runner import Runner, RunnerConfig
    from fuzzer.monitor import MockMonitor

    config = RunnerConfig(
        base_url="http://localhost:9999",
        model_name="test-model",
        server_flags="--gpu-memory-utilization 0.9",
    )
    adapter = VllmLoRAAdapter(ADAPTERS)
    runner = Runner(config=config, adapter=adapter, monitor=MockMonitor())

    assert "--enable-lora" in runner.config.server_flags
    assert "lora_a=/tmp/lora_a" in runner.config.server_flags
    assert "--gpu-memory-utilization 0.9" in runner.config.server_flags


def test_runner_base_adapter_no_extra_flags():
    from fuzzer.core.runner import Runner, RunnerConfig
    from fuzzer.monitor import MockMonitor

    config = RunnerConfig(
        base_url="http://localhost:9999",
        model_name="test-model",
        server_flags="--gpu-memory-utilization 0.9",
    )
    runner = Runner(config=config, monitor=MockMonitor())
    assert runner.config.server_flags == "--gpu-memory-utilization 0.9"


# ---------------------------------------------------------------------------
# BugClass
# ---------------------------------------------------------------------------

def test_adapter_isolation_failure_bug_class_exists():
    assert BugClass.ADAPTER_ISOLATION_FAILURE.value == "adapter_isolation_failure"


# ---------------------------------------------------------------------------
# Event adapter_name field
# ---------------------------------------------------------------------------

def test_event_adapter_name_defaults_none():
    ev = make_send()
    assert ev.adapter_name is None


def test_event_adapter_name_set():
    ev = make_send(adapter_name="lora_a")
    assert ev.adapter_name == "lora_a"


def test_trace_round_trips_adapter_name():
    from fuzzer.trace import Trace
    ev = make_send(adapter_name="lora_b")
    trace = Trace(seed=1, events=[ev])
    d = trace.to_dict()
    trace2 = Trace.from_dict(d)
    assert trace2.events[0].adapter_name == "lora_b"


def test_old_trace_without_adapter_name_loads_cleanly():
    """Backward compat: traces serialised before adapter_name existed load fine."""
    import json
    from fuzzer.trace import Trace
    old_json = json.dumps({
        "seed": 42,
        "events": [{
            "action": "send", "offset_ms": 0, "request_id": "r1",
            "prompt_len": 16, "max_tokens": 8, "stream": True,
        }],
        "tags": [],
    })
    trace = Trace.from_json(old_json)
    assert trace.events[0].adapter_name is None


# ---------------------------------------------------------------------------
# baseline_probe_event_overrides
# ---------------------------------------------------------------------------

class TestBaselineProbeEventOverrides:
    def test_stamps_first_adapter_alphabetically(self):
        a = VllmLoRAAdapter({"lora_b": "/b", "lora_a": "/a"})
        ev = make_send(adapter_name=None)
        result = a.baseline_probe_event_overrides(ev)
        assert result.adapter_name == "lora_a"

    def test_returns_new_event_not_same_object(self):
        a = VllmLoRAAdapter(ADAPTERS)
        ev = make_send(adapter_name=None)
        result = a.baseline_probe_event_overrides(ev)
        assert result is not ev

    def test_original_event_unchanged(self):
        a = VllmLoRAAdapter(ADAPTERS)
        ev = make_send(adapter_name=None)
        a.baseline_probe_event_overrides(ev)
        assert ev.adapter_name is None

    def test_other_fields_preserved(self):
        a = VllmLoRAAdapter(ADAPTERS)
        ev = make_send(prompt_len=256, prefix_len=128)
        result = a.baseline_probe_event_overrides(ev)
        assert result.prompt_len == 256
        assert result.prefix_len == 128
        assert result.max_tokens == 32

    def test_three_adapters_picks_first_alphabetically(self):
        a = VllmLoRAAdapter({"lora_c": "/c", "lora_a": "/a", "lora_b": "/b"})
        ev = make_send()
        result = a.baseline_probe_event_overrides(ev)
        assert result.adapter_name == "lora_a"

    def test_base_adapter_default_is_noop(self):
        """TargetAdapter default returns the same event unchanged."""
        from fuzzer.core.adapter import TargetAdapter
        from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
        base = VllmBaseAdapter()
        ev = make_send(adapter_name=None)
        result = base.baseline_probe_event_overrides(ev)
        assert result is ev
        assert result.adapter_name is None


# ---------------------------------------------------------------------------
# _perm_idx and _build_prompt (Change C: non-sequential suffix)
# ---------------------------------------------------------------------------

class TestPermIdx:
    def test_not_sequential(self):
        """_perm_idx must not reproduce 0, 1, 2, ... for any non-trivial length."""
        from fuzzer.adapters.vllm_base.adapter import _perm_idx
        rid_hash = 1234
        n = 8
        indices = [_perm_idx(rid_hash, i) for i in range(n)]
        assert indices != list(range(n)), (
            "_perm_idx produced the trivial counting sequence — "
            "the anti-pattern-completion property is broken"
        )

    def test_deterministic(self):
        """Same inputs always produce the same output."""
        from fuzzer.adapters.vllm_base.adapter import _perm_idx
        for rid_hash in (0, 42, 9999):
            for pos in range(10):
                assert _perm_idx(rid_hash, pos) == _perm_idx(rid_hash, pos)

    def test_different_rid_hash_produces_different_sequence(self):
        """Different rid_hash values produce different index sequences."""
        from fuzzer.adapters.vllm_base.adapter import _perm_idx
        seq_a = [_perm_idx(111, i) for i in range(8)]
        seq_b = [_perm_idx(222, i) for i in range(8)]
        assert seq_a != seq_b

    def test_output_in_range(self):
        """_perm_idx always returns a value in [0, 9999]."""
        from fuzzer.adapters.vllm_base.adapter import _perm_idx
        for rid_hash in (0, 1, 5000, 9999):
            for pos in range(20):
                idx = _perm_idx(rid_hash, pos)
                assert 0 <= idx <= 9999


class TestBuildPrompt:
    def _make_event(self, prompt_len=64, prefix_len=8,
                    family_id=None, request_id="r1"):
        return Event(
            action=ActionType.SEND,
            offset_ms=0,
            request_id=request_id,
            prompt_len=prompt_len,
            max_tokens=32,
            prefix_len=prefix_len,
            prompt_family_id=family_id,
        )

    def test_same_family_id_produces_identical_prompts(self):
        """Two events with the same prompt_family_id must produce byte-identical prompts."""
        from fuzzer.adapters.vllm_base.adapter import _build_prompt
        ev_a = self._make_event(family_id="fam_xyz", request_id="req_a")
        ev_b = self._make_event(family_id="fam_xyz", request_id="req_b")
        assert _build_prompt(ev_a) == _build_prompt(ev_b)

    def test_different_family_ids_produce_different_prompts(self):
        """Different family IDs must produce different unique suffixes."""
        from fuzzer.adapters.vllm_base.adapter import _build_prompt
        ev_a = self._make_event(family_id="fam_aaa", request_id="r1")
        ev_b = self._make_event(family_id="fam_bbb", request_id="r1")
        assert _build_prompt(ev_a) != _build_prompt(ev_b)

    def test_suffix_not_trivially_sequential(self):
        """Unique suffix indices must not be 0, 1, 2, ... (the counting pattern)."""
        from fuzzer.adapters.vllm_base.adapter import _build_prompt
        ev = self._make_event(prompt_len=16, prefix_len=0, family_id="test_seq")
        prompt = _build_prompt(ev)
        tokens = prompt.split()
        # All tokens are u{hash}_{idx}; extract the integer index from each
        indices = []
        for tok in tokens:
            if "_" in tok:
                try:
                    indices.append(int(tok.split("_")[-1]))
                except ValueError:
                    pass
        assert len(indices) >= 4, "need enough tokens to test sequentiality"
        assert indices != list(range(len(indices))), (
            "suffix indices are the trivial counting sequence — "
            "LoRA adapters will be ignored by pattern completion"
        )

    def test_deterministic_across_calls(self):
        """Same event produces the same prompt string on repeated calls."""
        from fuzzer.adapters.vllm_base.adapter import _build_prompt
        ev = self._make_event(family_id="repro_check")
        assert _build_prompt(ev) == _build_prompt(ev)

    def test_shared_prefix_unchanged(self):
        """The shared prefix part (pre0 pre1 ...) must still be correct."""
        from fuzzer.adapters.vllm_base.adapter import _build_prompt
        ev = self._make_event(prompt_len=8, prefix_len=4, family_id="pfx_test")
        tokens = _build_prompt(ev).split()
        assert tokens[:4] == ["pre0", "pre1", "pre2", "pre3"]


# ---------------------------------------------------------------------------
# calibrate() — unit tests with a mock executor (Change B)
# ---------------------------------------------------------------------------

class TestCalibrate:
    """Tests for VllmLoRAAdapter.calibrate() using a mock executor."""

    def _make_adapter(self):
        return VllmLoRAAdapter({"lora_a": "/tmp/a", "lora_b": "/tmp/b"})

    def _make_mock_executor(self, divergence_map=None):
        """Return an async mock executor.

        divergence_map: dict mapping (prompt_len, prefix_len, adapter_name) ->
            list[float] of logprobs.  Defaults to small but nonzero divergence.
        """
        from fuzzer.core.types import RequestResult
        import time

        class MockResult:
            def __init__(self, requests):
                self.requests = requests

        class MockExecutor:
            async def run(self, trace, base_url):
                reqs = []
                for ev in trace.events:
                    key = (ev.prompt_len, ev.prefix_len, ev.adapter_name)
                    if divergence_map and key in divergence_map:
                        lps = divergence_map[key]
                    else:
                        # default: slightly different per adapter
                        lps = [-1.0 + (0.5 if ev.adapter_name == "lora_a" else 0.0)]
                    reqs.append(RequestResult(
                        request_id=ev.request_id,
                        adapter_name=ev.adapter_name,
                        output_text="tok",
                        output_logprobs=lps,
                        completed_at=time.time(),
                        error=None,
                        prompt_len=ev.prompt_len,
                    ))
                return MockResult(reqs)

        return MockExecutor()

    def test_returns_list_of_dicts(self):
        import asyncio
        adapter = self._make_adapter()
        executor = self._make_mock_executor()
        result = asyncio.run(
            adapter.calibrate(executor, "http://localhost", n_candidates=4, top_k=2, n_repeats=1)
        )
        assert isinstance(result, list)
        for item in result:
            assert "prompt_len" in item
            assert "prefix_len" in item
            assert "family_id" in item
            assert "score" in item
            assert "trace" in item

    def test_returns_at_most_top_k(self):
        import asyncio
        adapter = self._make_adapter()
        executor = self._make_mock_executor()
        result = asyncio.run(
            adapter.calibrate(executor, "http://localhost", n_candidates=8, top_k=3, n_repeats=1)
        )
        assert len(result) <= 3

    def test_sorted_by_score_descending(self):
        import asyncio
        from fuzzer.core.types import RequestResult
        import time

        # Give different shapes very different logprob divergences
        adapter = self._make_adapter()

        class VariedExecutor:
            _call = 0
            async def run(self, trace, base_url):
                reqs = []
                for ev in trace.events:
                    # lora_a gets logprob -1.0, lora_b gets -(1 + prompt_len/100)
                    lp = -1.0 if ev.adapter_name == "lora_a" else -(1.0 + ev.prompt_len / 100.0)
                    reqs.append(RequestResult(
                        request_id=ev.request_id,
                        adapter_name=ev.adapter_name,
                        output_text="x",
                        output_logprobs=[lp],
                        completed_at=time.time(),
                        error=None,
                        prompt_len=ev.prompt_len,
                    ))
                return type("R", (), {"requests": reqs})()

        result = asyncio.run(
            adapter.calibrate(VariedExecutor(), "http://localhost",
                              n_candidates=8, top_k=5, n_repeats=1)
        )
        scores = [f["score"] for f in result]
        assert scores == sorted(scores, reverse=True)

    def test_events_have_logprobs_5(self):
        """Calibration seed traces must carry logprobs=5 for oracle tiebreaker."""
        import asyncio
        adapter = self._make_adapter()
        executor = self._make_mock_executor()
        result = asyncio.run(
            adapter.calibrate(executor, "http://localhost", n_candidates=4, top_k=2, n_repeats=1)
        )
        for fam in result:
            for ev in fam["trace"].events:
                assert ev.logprobs == 5

    def test_events_share_prompt_family_id(self):
        """Both SEND events in a calibration seed trace share the same family_id."""
        import asyncio
        adapter = self._make_adapter()
        executor = self._make_mock_executor()
        result = asyncio.run(
            adapter.calibrate(executor, "http://localhost", n_candidates=4, top_k=2, n_repeats=1)
        )
        for fam in result:
            trace = fam["trace"]
            family_ids = {ev.prompt_family_id for ev in trace.events
                          if ev.action == ActionType.SEND}
            assert len(family_ids) == 1
            assert None not in family_ids

    def test_filters_zero_score_families(self):
        """Families with score <= 0 must not be returned, even if top_k allows it.

        When one adapter returns no logprobs, _divergence() returns 0.0 for every
        pair, so all candidates score 0.  calibrate() must return an empty list
        rather than seeding the corpus with non-discriminating families.
        """
        import asyncio
        from fuzzer.core.types import RequestResult
        import time

        adapter = self._make_adapter()

        class PartialExecutor:
            async def run(self, trace, base_url):
                reqs = []
                for ev in trace.events:
                    # Only lora_a returns logprobs; lora_b returns None
                    lps = [-1.0] if ev.adapter_name == "lora_a" else None
                    reqs.append(RequestResult(
                        request_id=ev.request_id,
                        adapter_name=ev.adapter_name,
                        output_text="x",
                        output_logprobs=lps,
                        completed_at=time.time(),
                        error=None,
                        prompt_len=ev.prompt_len,
                    ))
                return type("R", (), {"requests": reqs})()

        result = asyncio.run(
            adapter.calibrate(PartialExecutor(), "http://localhost",
                              n_candidates=4, top_k=4, n_repeats=1)
        )
        # All divergences are 0 (no lora_b logprobs) → all filtered out
        assert result == [], (
            f"Expected empty list, got {len(result)} families with scores "
            f"{[f['score'] for f in result]}"
        )

    def test_distinct_candidates_per_shape(self):
        """n_candidates > grid_size must produce distinct family_ids, not duplicates."""
        import asyncio
        adapter = self._make_adapter()
        executor = self._make_mock_executor()
        # 5 prompt_lens * 4 prefix_fracs = 20 shapes; request 24 candidates
        result = asyncio.run(
            adapter.calibrate(executor, "http://localhost",
                              n_candidates=24, top_k=24, n_repeats=1)
        )
        family_ids = [f["family_id"] for f in result]
        # All returned families must have distinct family_ids
        assert len(family_ids) == len(set(family_ids)), (
            "calibrate() returned duplicate family_ids — "
            "n_candidates > grid_size is not generating variant families"
        )

    def test_best_pair_in_result(self):
        """Each returned family dict must contain the best_pair key."""
        import asyncio
        adapter = self._make_adapter()
        executor = self._make_mock_executor()
        result = asyncio.run(
            adapter.calibrate(executor, "http://localhost",
                              n_candidates=4, top_k=2, n_repeats=1)
        )
        for fam in result:
            assert "best_pair" in fam
            a, b = fam["best_pair"]
            assert a in adapter.adapter_names
            assert b in adapter.adapter_names
            assert a != b

    def test_seed_trace_uses_best_pair(self):
        """Seed trace adapter names must match the best_pair that drove the score."""
        import asyncio
        adapter = self._make_adapter()
        executor = self._make_mock_executor()
        result = asyncio.run(
            adapter.calibrate(executor, "http://localhost",
                              n_candidates=4, top_k=2, n_repeats=1)
        )
        for fam in result:
            trace_adapters = {ev.adapter_name for ev in fam["trace"].events
                              if ev.action == ActionType.SEND}
            assert trace_adapters == set(fam["best_pair"])


# ---------------------------------------------------------------------------
# path_calibration RunnerConfig field (Change B)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# analyze_kv_summary wired in VllmLoRAAdapter
# ---------------------------------------------------------------------------

import asyncio as _asyncio
import math as _math_la
from fuzzer.kv_events import KVProbeSummary
from fuzzer.adapters.vllm_lora.adapter import VllmLoRAAdapter


def _cross_lora_summary(n=2):
    return KVProbeSummary(
        start_t=0.0, end_t=1.0,
        stored_count=10, removed_count=3, cleared_count=0,
        distinct_blocks=8, per_lora_counts={},
        cross_lora_reused_blocks=n,
        conflicting_hashes=["0xabc"] * n,
        reuse_too_fast_count=0,
        reuse_too_fast_min_gap_ms=_math_la.inf,
        fingerprint="abcdef1234567890",
        raw_event_sample=[],
    )


class TestAnalyzeKvSummaryWiredInLoRAAdapter:

    def test_cross_lora_returns_score_6(self):
        adapter = VllmLoRAAdapter({"lora_a": "/tmp/a", "lora_b": "/tmp/b"})
        result = adapter.analyze_kv_summary(_cross_lora_summary(n=2))
        assert len(result) == 1
        assert result[0]["anomaly_type"] == "cross_lora_reuse"
        assert result[0]["score"] == 6.0

    def test_zero_blocks_returns_empty(self):
        adapter = VllmLoRAAdapter({"lora_a": "/tmp/a", "lora_b": "/tmp/b"})
        assert adapter.analyze_kv_summary(_cross_lora_summary(n=0)) == []

    def test_run_confirmation_batch_default_empty(self):
        adapter = VllmLoRAAdapter({"lora_a": "/tmp/a", "lora_b": "/tmp/b"})
        result = _asyncio.run(
            adapter.run_confirmation_batch(None, "http://localhost:8000")
        )
        assert result == []
