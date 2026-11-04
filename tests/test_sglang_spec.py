"""Tests for SglangSpecAdapter: server flags, mutations, oracle, confirmation queue."""
import pytest
import random

from fuzzer.adapters.sglang_spec.adapter import SglangSpecAdapter, MAX_QUEUE_SIZE
from fuzzer.adapters.sglang_spec.mutators import (
    cancel_during_spec,
    spec_shared_prefix_race,
    SPEC_MUTATIONS,
)
from fuzzer.adapters.sglang_spec.oracles import compare_outputs, SpecCandidate
from fuzzer.core.types import TraceResult, RequestResult
from fuzzer.trace import Trace, Event, ActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send(offset_ms=0, max_tokens=64, temperature=None, stream=True,
          prefix_len=0, origin=None, family_id=None) -> Event:
    return Event(
        action=ActionType.SEND,
        offset_ms=offset_ms,
        request_id=f"r_{offset_ms}",
        prompt_len=64,
        max_tokens=max_tokens,
        stream=stream,
        prefix_len=prefix_len,
        temperature=temperature,
        origin=origin,
        prompt_family_id=family_id,
    )


def _cancel(offset_ms=100, origin=None) -> Event:
    return Event(action=ActionType.CANCEL, offset_ms=offset_ms, origin=origin)


def _trace(*events):
    return Trace(seed=0, events=list(events))


def _req(request_id="r1", output_text="hello", tokens_received=5,
         cancelled_at=None, completed_at=1.0, error=None):
    return RequestResult(
        request_id=request_id,
        sent_at=0.0,
        completed_at=completed_at,
        cancelled_at=cancelled_at,
        output_text=output_text,
        tokens_received=tokens_received,
        error=error,
    )


def _result(*reqs):
    return TraceResult(trace_seed=0, started_at=0.0, finished_at=1.0,
                       requests=list(reqs))


def _rng(seed=42):
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Server flags
# ---------------------------------------------------------------------------

class TestServerFlags:
    def test_contains_speculative_algorithm(self):
        assert "--speculative-algorithm" in SglangSpecAdapter().server_flags_extra()

    def test_default_algorithm_ngram(self):
        """Default is NGRAM, not NEXTN: NEXTN is an internal SGLang alias that
        maps to EAGLE and needs a draft checkpoint, so it is unusable as the
        no-extra-weights default."""
        flags = SglangSpecAdapter().server_flags_extra()
        assert "NGRAM" in flags
        assert "NEXTN" not in flags

    def test_custom_algorithm(self):
        a = SglangSpecAdapter(algorithm="EAGLE", draft_model_path="/tmp/eagle")
        assert "EAGLE" in a.server_flags_extra()

    def test_ngram_uses_num_draft_tokens_not_num_steps(self):
        """NGRAM's budget flag is --speculative-num-draft-tokens.  The absence
        assertion is the load-bearing half: SGLang 0.5.10.post1 rejects
        --speculative-num-steps for NGRAM, so emitting both would break the
        server, and only checking presence would not catch it."""
        flags = SglangSpecAdapter().server_flags_extra()
        assert "--speculative-num-draft-tokens" in flags
        assert "--speculative-num-steps" not in flags

    def test_eagle_uses_num_steps_not_num_draft_tokens(self):
        flags = SglangSpecAdapter(
            algorithm="EAGLE", draft_model_path="/tmp/eagle"
        ).server_flags_extra()
        assert "--speculative-num-steps" in flags
        assert "--speculative-num-draft-tokens" not in flags

    def test_custom_num_steps(self):
        flags = SglangSpecAdapter(num_steps=8).server_flags_extra()
        assert "8" in flags

    def test_no_speculative_config_json(self):
        """SGLang uses separate flags, not a JSON blob like vLLM."""
        assert "--speculative-config" not in SglangSpecAdapter().server_flags_extra()

    def test_eagle_requires_draft_model_path(self):
        with pytest.raises(ValueError, match="draft_model_path"):
            SglangSpecAdapter(algorithm="EAGLE")

    def test_eagle_emits_draft_model_path_flag(self):
        a = SglangSpecAdapter(algorithm="EAGLE", draft_model_path="/tmp/eagle")
        flags = a.server_flags_extra()
        assert "--speculative-draft-model-path" in flags
        assert "/tmp/eagle" in flags

    def test_ngram_no_draft_model_path_flag(self):
        """NGRAM needs no extra weights, so no draft checkpoint flag."""
        flags = SglangSpecAdapter().server_flags_extra()
        assert "--speculative-draft-model-path" not in flags


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

class TestMutations:
    def test_extra_mutations_includes_spec_mutations(self):
        a = SglangSpecAdapter()
        names = {m.__name__ for m in a.extra_mutations()}
        for m in SPEC_MUTATIONS:
            assert m.__name__ in names

    def test_cancel_during_spec_injects_cancel(self):
        t = _trace(_send(0, max_tokens=128, stream=True))
        result = cancel_during_spec(t, _rng())
        cancels = [ev for ev in result.events if ev.action == ActionType.CANCEL]
        assert len(cancels) == 1

    def test_cancel_during_spec_noop_no_eligible(self):
        t = _trace(_send(0, max_tokens=16, stream=True))  # max_tokens < 64
        assert cancel_during_spec(t, _rng()) is t

    def test_cancel_after_send(self):
        t = _trace(_send(0, max_tokens=128, stream=True))
        result = cancel_during_spec(t, _rng())
        cancel = next(ev for ev in result.events if ev.action == ActionType.CANCEL)
        assert cancel.offset_ms > 0

    def test_spec_shared_prefix_race_adds_events(self):
        t = _trace(_send(0))
        result = spec_shared_prefix_race(t, _rng())
        assert len(result.events) > len(t.events)

    def test_spec_shared_prefix_race_large_prefix(self):
        t = _trace(_send(0))
        result = spec_shared_prefix_race(t, _rng())
        new_sends = [ev for ev in result.events
                     if ev.action == ActionType.SEND and ev.origin == "mutated"]
        assert all(ev.prefix_len >= 256 for ev in new_sends)


# ---------------------------------------------------------------------------
# Mutation weights
# ---------------------------------------------------------------------------

class TestMutationWeights:
    def test_cancel_during_spec_upweighted(self):
        a = SglangSpecAdapter()
        w = a.mutation_weights()
        assert w.get("cancel_during_spec", 1.0) > 1.0

    def test_prefix_race_enabled_for_sglang(self):
        """Unlike vLLM, SGLang enables spec_shared_prefix_race (RadixAttention)."""
        a = SglangSpecAdapter()
        w = a.mutation_weights()
        assert w.get("spec_shared_prefix_race", 0.0) > 0.0


# ---------------------------------------------------------------------------
# compare_outputs oracle
# ---------------------------------------------------------------------------

class TestCompareOutputs:
    def test_identical_no_divergence(self):
        spec    = _result(_req("r1", "hello"))
        nospec  = _result(_req("r1", "hello"))
        assert compare_outputs(spec, nospec) == []

    def test_different_outputs_divergence(self):
        spec   = _result(_req("r1", "hello world"))
        nospec = _result(_req("r1", "hello earth"))
        divs   = compare_outputs(spec, nospec)
        assert len(divs) == 1
        assert divs[0]["request_id"] == "r1"
        assert divs[0]["first_divergence_pos"] == 6  # 'w' vs 'e'

    def test_cancelled_skipped(self):
        spec   = _result(_req("r1", cancelled_at=0.5))
        nospec = _result(_req("r1", cancelled_at=0.5))
        assert compare_outputs(spec, nospec) == []

    def test_error_skipped(self):
        spec   = _result(_req("r1", error="timeout"))
        nospec = _result(_req("r1", error="timeout"))
        assert compare_outputs(spec, nospec) == []

    def test_missing_nospec_skipped(self):
        spec   = _result(_req("r1", "hello"))
        nospec = _result()
        assert compare_outputs(spec, nospec) == []


# ---------------------------------------------------------------------------
# Suspicion signal detection
# ---------------------------------------------------------------------------

class TestSuspicionSignal:
    def _adapter(self, confirm_url="http://nospec:8001"):
        return SglangSpecAdapter(confirm_url=confirm_url)

    def test_empty_completion_is_signal(self):
        a = self._adapter()
        result = _result(_req("r1", output_text="", tokens_received=0))
        assert a._has_empty_completion(result)

    def test_non_empty_completion_no_signal(self):
        a = self._adapter()
        result = _result(_req("r1", "hello"))
        assert not a._has_empty_completion(result)

    def test_cancel_mutation_detected(self):
        a = self._adapter()
        t = _trace(_cancel(origin="mutated"))
        assert a._has_cancel_mutation(t)

    def test_cancel_non_mutated_not_detected(self):
        a = self._adapter()
        t = _trace(_cancel(origin=None))
        assert not a._has_cancel_mutation(t)

    def test_prefix_race_mutation_detected(self):
        a = self._adapter()
        ev = _send(0, origin="mutated", family_id="fam1")
        t = _trace(ev)
        assert a._has_prefix_race_mutation(t)

    def test_none_result_no_signal(self):
        a = self._adapter()
        assert not a._has_empty_completion(None)

    def test_no_kv_summary_field(self):
        """SGLang spec adapter does not use reuse_too_fast — method takes (result, trace)."""
        import inspect
        sig = inspect.signature(SglangSpecAdapter._has_suspicion_signal)
        assert "kv_summary" not in sig.parameters


# ---------------------------------------------------------------------------
# evaluate_extra enqueue logic
# ---------------------------------------------------------------------------

class TestEnqueue:
    def _adapter_with_confirm(self, **kw):
        return SglangSpecAdapter(confirm_url="http://nospec:8001", **kw)

    def _confirmable_trace(self):
        return _trace(_send(0, temperature=0.0, max_tokens=128, stream=True,
                            origin="mutated"))

    def test_enqueues_with_empty_completion_signal(self):
        a = self._adapter_with_confirm(enqueue_cooldown_s=0.0)
        result = _result(_req("r1", output_text="", tokens_received=0))
        t = _trace(_send(0, temperature=0.0))
        a.evaluate_extra(result, None, trace=t)
        assert len(a._queue) == 1

    def test_no_enqueue_without_confirm_url(self):
        a = SglangSpecAdapter(confirm_url="")
        result = _result(_req("r1", output_text="", tokens_received=0))
        t = _trace(_send(0, temperature=0.0))
        a.evaluate_extra(result, None, trace=t)
        assert len(a._queue) == 0

    def test_no_enqueue_stochastic_trace(self):
        a = self._adapter_with_confirm()
        result = _result(_req("r1", output_text="", tokens_received=0))
        t = _trace(_send(0, temperature=0.7))
        a.evaluate_extra(result, None, trace=t)
        assert len(a._queue) == 0

    def test_cooldown_deduplicates(self):
        a = self._adapter_with_confirm(enqueue_cooldown_s=999.0)
        result = _result(_req("r1", output_text="", tokens_received=0))
        t = _trace(_send(0, temperature=0.0))
        a.evaluate_extra(result, None, trace=t)
        a.evaluate_extra(result, None, trace=t)
        assert len(a._queue) == 1

    def test_queue_cap_respected(self):
        a = self._adapter_with_confirm(enqueue_cooldown_s=0.0)
        for i in range(MAX_QUEUE_SIZE + 5):
            result = _result(_req(f"r{i}", output_text="", tokens_received=0))
            t = _trace(_send(i, temperature=0.0))
            a.evaluate_extra(result, None, trace=t)
        assert len(a._queue) <= MAX_QUEUE_SIZE


# ---------------------------------------------------------------------------
# run_confirmation_batch gating
# ---------------------------------------------------------------------------

class TestConfirmationGating:
    def test_no_confirm_url_returns_empty(self):
        import asyncio
        a = SglangSpecAdapter(confirm_url="")
        assert asyncio.run(a.run_confirmation_batch(None, "http://x")) == []

    def test_empty_queue_returns_empty(self):
        import asyncio
        a = SglangSpecAdapter(confirm_url="http://nospec:8001")
        assert asyncio.run(a.run_confirmation_batch(None, "http://x")) == []

    def test_interval_gate_blocks_small_queue(self):
        import asyncio, time
        a = SglangSpecAdapter(confirm_url="http://nospec:8001",
                              interval_min=999.0, batch_size=10)
        a._queue.append(SpecCandidate(
            trace=_trace(_send(0)), iteration=1, sig="abc", timestamp=time.time()
        ))
        a._last_confirm_at = time.time()
        assert asyncio.run(a.run_confirmation_batch(None, "http://x")) == []


# ---------------------------------------------------------------------------
# Candidate signature
# ---------------------------------------------------------------------------

class TestCandidateSig:
    def test_same_trace_same_sig(self):
        a = SglangSpecAdapter()
        t = _trace(_send(0, max_tokens=64), _cancel(100))
        assert a._candidate_sig(t) == a._candidate_sig(t)

    def test_different_cancel_count_different_sig(self):
        a = SglangSpecAdapter()
        t1 = _trace(_send(0), _cancel(100))
        t2 = _trace(_send(0))
        assert a._candidate_sig(t1) != a._candidate_sig(t2)
