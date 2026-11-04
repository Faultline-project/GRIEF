"""Tests for SglangLoRAAdapter: server flags, payload, mutations, and oracle."""
import pytest

from fuzzer.adapters.sglang_lora.adapter import SglangLoRAAdapter
from fuzzer.adapters.sglang_lora.oracles import (
    check_cross_adapter_isolation,
    analyze_cross_lora_kv,
    _logprobs_identical,
    _sim,
)
from fuzzer.kv_events import KVProbeSummary
from fuzzer.adapters.sglang_lora.mutators import (
    switch_adapter,
    concurrent_adapter_burst,
    same_prefix_different_adapters,
    adapter_thrash,
    mixed_base_lora,
    inject_same_prompt_pair,
    same_adapter_high_concurrency,
    preemption_stress_burst,
    LORA_MUTATIONS,
)
from fuzzer.core.types import TraceResult, RequestResult, BugClass
from fuzzer.trace import Trace, Event, ActionType
import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _adapter(names=("lora_a", "lora_b")):
    return SglangLoRAAdapter({n: f"/tmp/{n}" for n in names})


def _send(adapter_name="lora_a", prompt_len=64, prefix_len=0,
          family_id=None, logprobs=None, offset_ms=0) -> Event:
    return Event(
        action=ActionType.SEND,
        offset_ms=offset_ms,
        request_id=f"r_{adapter_name}_{offset_ms}",
        prompt_len=prompt_len,
        max_tokens=64,
        stream=True,
        prefix_len=prefix_len,
        adapter_name=adapter_name,
        temperature=0.0,
        prompt_family_id=family_id,
        logprobs=logprobs,
    )


def _req(request_id="r1", adapter_name="lora_a", output_text="hello",
         logprobs=None, family_id="fam1", prefix_len=0, prompt_len=64,
         error=None, completed_at=1.0):
    r = RequestResult(
        request_id=request_id,
        sent_at=0.0,
        completed_at=completed_at,
        cancelled_at=None,
        output_text=output_text,
        tokens_received=len(output_text.split()),
        error=error,
    )
    r.adapter_name      = adapter_name
    r.prompt_family_id  = family_id
    r.prefix_len        = prefix_len
    r.prompt_len        = prompt_len
    r.output_logprobs   = logprobs
    return r


def _result(*reqs):
    return TraceResult(trace_seed=0, started_at=0.0, finished_at=1.0,
                       requests=list(reqs))


def _trace(*events):
    return Trace(seed=0, events=list(events))


def _rng(seed=42):
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Init validation
# ---------------------------------------------------------------------------

class TestInit:
    def test_requires_two_adapters(self):
        with pytest.raises(ValueError):
            SglangLoRAAdapter({"only_one": "/tmp/a"})

    def test_accepts_two_adapters(self):
        a = _adapter()
        assert len(a.adapters) == 2

    def test_adapter_names_sorted(self):
        a = SglangLoRAAdapter({"b": "/tmp/b", "a": "/tmp/a"})
        assert a.adapter_names == ["a", "b"]


# ---------------------------------------------------------------------------
# Server flags
# ---------------------------------------------------------------------------

class TestServerFlags:
    def test_contains_lora_paths_flag(self):
        assert "--lora-paths" in _adapter().server_flags_extra()

    def test_includes_all_adapter_names(self):
        a = _adapter(("lora_a", "lora_b"))
        flags = a.server_flags_extra()
        assert "lora_a=" in flags
        assert "lora_b=" in flags

    def test_includes_paths(self):
        a = SglangLoRAAdapter({"x": "/data/x", "y": "/data/y"})
        flags = a.server_flags_extra()
        assert "/data/x" in flags
        assert "/data/y" in flags


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

class TestBuildPayload:
    def test_no_adapter_name_no_lora_path(self):
        a = _adapter()
        ev = Event(action=ActionType.SEND, offset_ms=0, request_id="r",
                   prompt_len=16, max_tokens=8, stream=True)
        _, payload = a.build_payload(ev)
        assert "lora_path" not in payload

    def test_adapter_name_injects_lora_path(self):
        a = _adapter()
        ev = _send(adapter_name="lora_a")
        _, payload = a.build_payload(ev)
        assert payload["lora_path"] == "lora_a"

    def test_unknown_adapter_name_no_lora_path(self):
        a = _adapter()
        ev = _send(adapter_name="unknown")
        _, payload = a.build_payload(ev)
        assert "lora_path" not in payload

    def test_model_field_not_injected(self):
        """SGLang uses lora_path, not model, for adapter selection."""
        a = _adapter()
        ev = _send(adapter_name="lora_a")
        _, payload = a.build_payload(ev)
        assert "model" not in payload


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

class TestMutations:
    def test_extra_mutations_includes_lora_mutations(self):
        a = _adapter()
        names = {m.__name__ for m in a.extra_mutations()}
        for m in LORA_MUTATIONS:
            assert m.__name__ in names

    def test_switch_adapter_reassigns(self):
        t = _trace(_send("lora_a"), _send("lora_b"))
        result = switch_adapter(t, _rng())
        names = {ev.adapter_name for ev in result.events
                 if ev.action == ActionType.SEND}
        assert names <= {"lora_a", "lora_b"}

    def test_switch_adapter_noop_one_adapter(self):
        t = _trace(_send("lora_a"), _send("lora_a"))
        assert switch_adapter(t, _rng()) is t

    def test_concurrent_burst_adds_events(self):
        t = _trace(_send("lora_a"), _send("lora_b"))
        result = concurrent_adapter_burst(t, _rng())
        assert len(result.events) > len(t.events)

    def test_same_prefix_aligns_prefix_len(self):
        t = _trace(
            _send("lora_a", prompt_len=64, prefix_len=10),
            _send("lora_b", prompt_len=64, prefix_len=20),
        )
        result = same_prefix_different_adapters(t, _rng())
        sends = [ev for ev in result.events if ev.action == ActionType.SEND]
        prefix_lens = {ev.prefix_len for ev in sends}
        assert len(prefix_lens) == 1

    def test_adapter_thrash_adds_events(self):
        t = _trace(_send("lora_a"), _send("lora_b"))
        result = adapter_thrash(t, _rng())
        assert len(result.events) > len(t.events)

    def test_mixed_base_lora_demotes_some(self):
        t = _trace(*[_send("lora_a") for _ in range(6)])
        result = mixed_base_lora(t, _rng())
        none_count = sum(1 for ev in result.events
                         if ev.action == ActionType.SEND and ev.adapter_name is None)
        assert none_count >= 1

    def test_inject_same_prompt_pair_stamps_family(self):
        t = _trace(_send("lora_a"), _send("lora_b"))
        result = inject_same_prompt_pair(t, _rng())
        sends = [ev for ev in result.events if ev.action == ActionType.SEND]
        family_ids = {ev.prompt_family_id for ev in sends
                      if ev.prompt_family_id is not None}
        assert len(family_ids) >= 1

    def test_inject_same_prompt_pair_noop_single_adapter(self):
        t = _trace(_send("lora_a"), _send("lora_a"))
        result = inject_same_prompt_pair(t, _rng())
        # result should be original trace (no two distinct adapters)
        assert result is t


# ---------------------------------------------------------------------------
# Oracle: check_cross_adapter_isolation
# ---------------------------------------------------------------------------

class TestCrossAdapterOracle:
    def test_identical_outputs_no_logprobs_score3(self):
        result = _result(
            _req("r1", "lora_a", "same", family_id="f1"),
            _req("r2", "lora_b", "same", family_id="f1"),
        )
        findings = check_cross_adapter_isolation(result)
        assert len(findings) == 1
        assert findings[0].score == 3.0
        assert findings[0].bug_class == BugClass.ADAPTER_ISOLATION_FAILURE

    def test_identical_outputs_identical_logprobs_score7(self):
        lps = [0.1, 0.2, 0.3]
        result = _result(
            _req("r1", "lora_a", "same", logprobs=lps, family_id="f1"),
            _req("r2", "lora_b", "same", logprobs=lps, family_id="f1"),
        )
        findings = check_cross_adapter_isolation(result)
        assert len(findings) == 1
        assert findings[0].score == 7.0


# ---------------------------------------------------------------------------
# evaluate_extra integration
# ---------------------------------------------------------------------------

class TestEvaluateExtra:
    def test_returns_cross_adapter_finding(self):
        a = _adapter()
        result = _result(
            _req("r1", "lora_a", "same", family_id="f1"),
            _req("r2", "lora_b", "same", family_id="f1"),
        )
        findings = a.evaluate_extra(result, None, trace=_trace(_send()))
        assert any(f.bug_class == BugClass.ADAPTER_ISOLATION_FAILURE for f in findings)

    def test_no_bug_finding_when_outputs_differ(self):
        a = _adapter()
        result = _result(
            _req("r1", "lora_a", "hello", family_id="f1"),
            _req("r2", "lora_b", "world", family_id="f1"),
        )
        findings = a.evaluate_extra(result, None, trace=_trace(_send()))
        # Bootstrap trigger (passed=True) is expected; no bug finding (passed=False)
        assert not any(
            f.bug_class == BugClass.ADAPTER_ISOLATION_FAILURE and not f.passed
            for f in findings
        )


# ---------------------------------------------------------------------------
# Baseline probe override
# ---------------------------------------------------------------------------

class TestBaselineProbeOverride:
    def test_stamps_first_adapter(self):
        a = SglangLoRAAdapter({"b_lora": "/tmp/b", "a_lora": "/tmp/a"})
        ev = Event(action=ActionType.SEND, offset_ms=0, request_id="bp",
                   prompt_len=16, max_tokens=8, stream=True)
        result = a.baseline_probe_event_overrides(ev)
        assert result.adapter_name == "a_lora"  # alphabetically first


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# analyze_cross_lora_kv — KV structural analysis
# ---------------------------------------------------------------------------

def _kv_summary(cross_lora=0, conflicting=None, fingerprint="abcdef1234567890"):
    return KVProbeSummary(
        start_t=0.0, end_t=1.0,
        stored_count=10, removed_count=0, cleared_count=0,
        distinct_blocks=10,
        cross_lora_reused_blocks=cross_lora,
        conflicting_hashes=conflicting if conflicting is not None else (["0xabc"] * cross_lora),
        per_lora_counts={},
        fingerprint=fingerprint,
    )


class TestAnalyzeCrossLoraKv:
    def test_returns_anomaly_when_blocks_reused(self):
        result = analyze_cross_lora_kv(_kv_summary(cross_lora=2))
        assert len(result) == 1
        r = result[0]
        assert r["anomaly_type"] == "cross_lora_reuse"
        assert r["score"] == pytest.approx(6.0)
        assert r["count"] == 2


class TestAnalyzeKvSummary:
    def test_delegates_to_cross_lora_kv(self):
        a = _adapter()
        result = a.analyze_kv_summary(_kv_summary(cross_lora=2))
        assert len(result) == 1
        assert result[0]["anomaly_type"] == "cross_lora_reuse"

    def test_empty_when_no_reuse(self):
        a = _adapter()
        assert a.analyze_kv_summary(_kv_summary(cross_lora=0)) == []


# ---------------------------------------------------------------------------
# New mutations: same_adapter_high_concurrency, preemption_stress_burst
# ---------------------------------------------------------------------------

class TestSameAdapterHighConcurrency:
    def test_adds_events(self):
        t = _trace(_send("lora_a"), _send("lora_b"))
        result = same_adapter_high_concurrency(t, _rng())
        assert len(result.events) > len(t.events)

    def test_all_added_events_use_same_adapter(self):
        t = _trace(_send("lora_a"), _send("lora_b"))
        result = same_adapter_high_concurrency(t, _rng())
        added = [ev for ev in result.events
                 if getattr(ev, "origin", None) == "same_adapter_high_concurrency"]
        assert len(added) >= 6
        assert len({ev.adapter_name for ev in added}) == 1

    def test_added_events_share_prefix(self):
        t = _trace(_send("lora_a", prompt_len=64, prefix_len=32),
                   _send("lora_b", prompt_len=64, prefix_len=16))
        result = same_adapter_high_concurrency(t, _rng())
        added = [ev for ev in result.events
                 if getattr(ev, "origin", None) == "same_adapter_high_concurrency"]
        assert len({ev.prefix_len for ev in added}) == 1

    def test_noop_no_sends(self):
        t = Trace(seed=0, events=[])
        assert same_adapter_high_concurrency(t, _rng()) is t

    def test_noop_no_adapters(self):
        ev = Event(action=ActionType.SEND, offset_ms=0, request_id="r",
                   prompt_len=16, max_tokens=8, stream=True)
        t = Trace(seed=0, events=[ev])
        assert same_adapter_high_concurrency(t, _rng()) is t


class TestPreemptionStressBurst:
    def test_adds_long_requests_to_concurrent_trace(self):
        t = _trace(
            _send("lora_a", offset_ms=0),
            _send("lora_b", offset_ms=10),
            _send("lora_a", offset_ms=20),
        )
        result = preemption_stress_burst(t, _rng())
        added = [ev for ev in result.events
                 if getattr(ev, "origin", None) == "preemption_stress"]
        assert len(added) >= 2
        assert all(ev.prompt_len >= 512 for ev in added)

    def test_noop_single_adapter(self):
        t = _trace(_send("lora_a", offset_ms=0), _send("lora_a", offset_ms=10))
        assert preemption_stress_burst(t, _rng()) is t

    def test_noop_only_one_send(self):
        t = _trace(_send("lora_a", offset_ms=0))
        assert preemption_stress_burst(t, _rng()) is t

    def test_noop_sparse_sequential_trace(self):
        # spread = 1000 ms > 500 ms gate → noop
        t = _trace(
            _send("lora_a", offset_ms=0),
            _send("lora_b", offset_ms=1000),
        )
        result = preemption_stress_burst(t, _rng())
        added = [ev for ev in result.events
                 if getattr(ev, "origin", None) == "preemption_stress"]
        assert len(added) == 0

    def test_applies_to_dense_concurrent_trace(self):
        # spread = 30 ms ≤ 500 ms → should apply
        t = _trace(
            _send("lora_a", offset_ms=0),
            _send("lora_b", offset_ms=10),
            _send("lora_a", offset_ms=30),
        )
        result = preemption_stress_burst(t, _rng())
        added = [ev for ev in result.events
                 if getattr(ev, "origin", None) == "preemption_stress"]
        assert len(added) >= 2


def test_sglang_lora_mutations_registry_is_exactly_this_list():
    """One exact-list assert replaces two `in LORA_MUTATIONS` tests, and also
    catches additions, removals and reordering."""
    assert [m.__name__ for m in LORA_MUTATIONS] == [
        "same_adapter_high_concurrency",
        "preemption_stress_burst",
        "switch_adapter",
        "concurrent_adapter_burst",
        "same_prefix_different_adapters",
        "adapter_thrash",
        "mixed_base_lora",
        "inject_same_prompt_pair",
    ]


# ---------------------------------------------------------------------------
# Probe scoring: _score_concurrent_finding attribution + pass_label offset
# ---------------------------------------------------------------------------

class TestScoreConcurrentFinding:
    def _score(self, conc_a, modal_a, conc_b, modal_base, logprobs, pass_label):
        a = _adapter()
        req = _req("iso_hp_a", "lora_a", conc_a, logprobs=logprobs)
        return a._score_concurrent_finding(
            conc_a, modal_a, conc_b, modal_base,
            req, "lora_a", "lora_b", 64, 0, pass_label,
        )

    def test_no_finding_when_output_unchanged(self):
        assert self._score("hello world", "hello world", "foo bar", None, None, "light") is None

    def test_toward_b_light_early_divergence(self):
        # Diverges at index 0 → div_pos=0.0 < 0.5 → base score 7; light pass → 7
        modal_a = "alpha beta gamma delta"
        conc_a  = "X beta gamma delta"     # diverges at token 0
        conc_b  = "X beta gamma delta"     # identical to conc_a → toward_B
        f = self._score(conc_a, modal_a, conc_b, None, None, "light")
        assert f is not None
        assert f.probe_metadata["attribution"] == "toward_B"
        assert f.score == pytest.approx(7.0)

    def test_toward_b_high_pressure_adds_offset(self):
        # Same early-divergence scenario; high_pressure offset +1 → score 8
        modal_a = "alpha beta gamma delta"
        conc_a  = "X beta gamma delta"
        conc_b  = "X beta gamma delta"
        f = self._score(conc_a, modal_a, conc_b, None, None, "high_pressure")
        assert f is not None
        assert f.probe_metadata["attribution"] == "toward_B"
        assert f.score == pytest.approx(8.0)

    def test_base_model_drift_light(self):
        modal_a    = "quick brown fox jumped"
        modal_base = "quick brown fox jumped"   # identical to modal_a means sim_base > sim_a only if conc diverges
        # conc_a matches base better than modal_a
        conc_a     = "quick brown fox jumped over the"
        # sim_to_b is low (b is unrelated)
        conc_b     = "completely different output text here"
        # modal_base should be very similar to conc_a; modal_a also similar
        # Need sim_to_base > sim_to_a: use modal_a that's quite different
        modal_a2   = "foo bar baz qux quux"
        modal_base2 = "quick brown fox jumped over the"  # very similar to conc_a
        f = self._score(conc_a, modal_a2, conc_b, modal_base2, None, "light")
        assert f is not None
        assert f.probe_metadata["attribution"] == "base_model_drift"
        assert f.score == pytest.approx(6.0)

    def test_base_model_drift_preemption_adds_offset(self):
        conc_a      = "quick brown fox jumped over the"
        modal_a     = "foo bar baz qux quux"
        conc_b      = "completely different output text here"
        modal_base  = "quick brown fox jumped over the"
        f = self._score(conc_a, modal_a, conc_b, modal_base, None, "preemption")
        assert f is not None
        assert f.probe_metadata["attribution"] == "base_model_drift"
        assert f.score == pytest.approx(7.0)

    def test_unknown_drift_light(self):
        # conc_a differs from modal_a but not toward b or base
        conc_a  = "completely novel output here"
        modal_a = "alpha beta gamma delta epsilon"
        conc_b  = "another very different string words"
        f = self._score(conc_a, modal_a, conc_b, None, None, "light")
        assert f is not None
        assert f.probe_metadata["attribution"] == "unknown_drift"
        assert f.score == pytest.approx(4.0)

    def test_unknown_drift_high_pressure_adds_offset(self):
        conc_a  = "completely novel output here"
        modal_a = "alpha beta gamma delta epsilon"
        conc_b  = "another very different string words"
        f = self._score(conc_a, modal_a, conc_b, None, None, "high_pressure")
        assert f is not None
        assert f.probe_metadata["attribution"] == "unknown_drift"
        assert f.score == pytest.approx(5.0)

    def test_pass_label_in_metadata(self):
        conc_a  = "completely novel output here"
        modal_a = "alpha beta gamma delta epsilon"
        conc_b  = "another very different string words"
        for label in ("light", "high_pressure", "preemption"):
            f = self._score(conc_a, modal_a, conc_b, None, None, label)
            assert f.probe_metadata["pass"] == label


# ---------------------------------------------------------------------------
# Bootstrap trigger: _probe_bootstrap_trigger
# ---------------------------------------------------------------------------

class TestProbeBootstrapTrigger:
    def test_emits_trigger_for_concurrent_adapters(self):
        a = _adapter()
        result = _result(
            _req("r1", "lora_a", "hello"),
            _req("r2", "lora_b", "world"),
        )
        triggers = a._probe_bootstrap_trigger(result)
        assert len(triggers) == 1
        t = triggers[0]
        assert t.passed is True
        assert t.score == pytest.approx(3.0)
        assert t.probe_metadata["adapter_a"] == "lora_a"
        assert t.probe_metadata["adapter_b"] == "lora_b"
        assert t.probe_metadata["source"] == "bootstrap_trigger"

    def test_no_trigger_single_adapter(self):
        a = _adapter()
        result = _result(
            _req("r1", "lora_a", "hello"),
            _req("r2", "lora_a", "world"),
        )
        assert a._probe_bootstrap_trigger(result) == []

    def test_no_trigger_errored_requests(self):
        a = _adapter()
        result = _result(
            _req("r1", "lora_a", "hello", error="timeout"),
            _req("r2", "lora_b", "world"),
        )
        # only lora_b completed successfully → single adapter → no trigger
        assert a._probe_bootstrap_trigger(result) == []

    def test_no_trigger_no_output_text(self):
        a = _adapter()
        r1 = _req("r1", "lora_a", "")
        r1.output_text = ""
        result = _result(r1, _req("r2", "lora_b", "world"))
        assert a._probe_bootstrap_trigger(result) == []

    def test_trigger_has_prompt_len_from_request(self):
        a = _adapter()
        result = _result(
            _req("r1", "lora_a", "hello", prompt_len=128),
            _req("r2", "lora_b", "world"),
        )
        triggers = a._probe_bootstrap_trigger(result)
        assert triggers[0].probe_metadata["prompt_len"] == 128

    def test_evaluate_extra_includes_trigger(self):
        a = _adapter()
        result = _result(
            _req("r1", "lora_a", "hello"),
            _req("r2", "lora_b", "world"),
        )
        all_results = a.evaluate_extra(result, None, trace=_trace(_send()))
        triggers = [f for f in all_results
                    if (f.probe_metadata or {}).get("source") == "bootstrap_trigger"]
        assert len(triggers) == 1
        assert triggers[0].passed is True


# ---------------------------------------------------------------------------
# A. filter_findings — oracle suppression
# ---------------------------------------------------------------------------

class TestFilterFindings:
    def test_suppresses_latency_spike(self):
        from fuzzer.core.types import OracleResult
        a = _adapter()
        findings = [
            OracleResult(passed=False, bug_class=BugClass.LATENCY_SPIKE, score=10.0,
                         detail="p99 TTFT exceeded"),
            OracleResult(passed=False, bug_class=BugClass.ADAPTER_ISOLATION_FAILURE,
                         score=7.0, detail="iso divergence"),
        ]
        out = a.filter_findings(findings)
        assert len(out) == 1
        assert out[0].bug_class == BugClass.ADAPTER_ISOLATION_FAILURE

    def test_suppresses_memory_leak(self):
        from fuzzer.core.types import OracleResult
        a = _adapter()
        findings = [
            OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK, score=3.0,
                         detail="GPU grew"),
        ]
        assert a.filter_findings(findings) == []

    def test_passes_through_other_classes(self):
        from fuzzer.core.types import OracleResult
        a = _adapter()
        findings = [
            OracleResult(passed=False, bug_class=BugClass.KV_CORRUPTION, score=8.0,
                         detail="kv mismatch"),
            OracleResult(passed=False, bug_class=BugClass.LIVENESS_FAILURE, score=10.0,
                         detail="server died"),
        ]
        out = a.filter_findings(findings)
        assert len(out) == 2

    def test_empty_list(self):
        a = _adapter()
        assert a.filter_findings([]) == []


# ---------------------------------------------------------------------------
# C3b. Round-robin pair sampling via _pair_rotation_idx
# ---------------------------------------------------------------------------

class TestPairRotation:
    def _six_adapter(self):
        names = ["lora_a", "lora_b", "lora_c", "lora_d", "lora_e", "lora_f"]
        return SglangLoRAAdapter({n: f"/tmp/{n}" for n in names})

    def test_two_adapter_probe_runs_single_pair(self):
        import itertools
        a = _adapter()  # 2 adapters → 1 pair
        all_pairs = list(itertools.combinations(sorted(a.adapters.keys()), 2))
        assert len(all_pairs) == 1
        # With max_pairs=3, only 1 pair available — should run that pair
        start = a._pair_rotation_idx % len(all_pairs)
        pairs = [all_pairs[(start + i) % len(all_pairs)]
                 for i in range(min(3, len(all_pairs)))]
        assert len(pairs) == 1


# ---------------------------------------------------------------------------
# C3a. Bootstrap trigger capped at 2 with many adapters
# ---------------------------------------------------------------------------

class TestBootstrapTriggerCap:
    def _six_adapter(self):
        names = ["lora_a", "lora_b", "lora_c", "lora_d", "lora_e", "lora_f"]
        return SglangLoRAAdapter({n: f"/tmp/{n}" for n in names})

    def test_six_adapters_emits_at_most_two_triggers(self):
        a = self._six_adapter()
        reqs = [_req(f"r{i}", name, "hello")
                for i, name in enumerate(
                    ["lora_a", "lora_b", "lora_c", "lora_d", "lora_e", "lora_f"]
                )]
        result = _result(*reqs)
        triggers = a._probe_bootstrap_trigger(result)
        assert len(triggers) == 2

    def test_two_adapters_emits_one_trigger(self):
        a = _adapter()
        result = _result(_req("r1", "lora_a", "hello"), _req("r2", "lora_b", "world"))
        triggers = a._probe_bootstrap_trigger(result)
        assert len(triggers) == 1

    def test_trigger_pairs_are_deterministic(self):
        # Same trace always produces same first 2 pairs (alphabetical order)
        a = self._six_adapter()
        reqs = [_req(f"r{i}", name, "hello")
                for i, name in enumerate(
                    ["lora_a", "lora_b", "lora_c", "lora_d", "lora_e", "lora_f"]
                )]
        result = _result(*reqs)
        t1 = a._probe_bootstrap_trigger(result)
        t2 = a._probe_bootstrap_trigger(result)
        pairs_1 = [(t.probe_metadata["adapter_a"], t.probe_metadata["adapter_b"])
                   for t in t1]
        pairs_2 = [(t.probe_metadata["adapter_a"], t.probe_metadata["adapter_b"])
                   for t in t2]
        assert pairs_1 == pairs_2
