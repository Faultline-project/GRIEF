"""Tests for VllmSpecAdapter."""
import asyncio
import json
import time
import pytest

from fuzzer.adapters.vllm_spec.adapter import VllmSpecAdapter, MAX_QUEUE_SIZE
from fuzzer.core.types import TraceResult, RequestResult
from fuzzer.trace import Trace, Event, ActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send(offset_ms=0, max_tokens=64, temperature=None, prefix_len=0,
          origin=None, prompt_family_id=None) -> Event:
    return Event(
        action           = ActionType.SEND,
        offset_ms        = offset_ms,
        prompt_len       = 64,
        max_tokens       = max_tokens,
        stream           = True,
        prefix_len       = prefix_len,
        temperature      = temperature,
        origin           = origin,
        prompt_family_id = prompt_family_id,
    )


def _cancel(offset_ms=100, origin=None) -> Event:
    return Event(action=ActionType.CANCEL, offset_ms=offset_ms, origin=origin)


def _trace(*events) -> Trace:
    return Trace(seed=0, events=list(events))


def _req(request_id="r1", output_text="hello", tokens_received=5,
         cancelled_at=None, completed_at=1.0, error=None):
    return RequestResult(
        request_id      = request_id,
        sent_at         = 0.0,
        completed_at    = completed_at,
        cancelled_at    = cancelled_at,
        output_text     = output_text,
        tokens_received = tokens_received,
        error           = error,
    )


def _result(*reqs):
    return TraceResult(trace_seed=0, started_at=0.0, finished_at=1.0, requests=list(reqs))


def _kv_summary(reuse_too_fast_count=0):
    """Minimal KVProbeSummary-like object for testing."""
    class _S:
        pass
    s = _S()
    s.reuse_too_fast_count = reuse_too_fast_count
    return s


def _inject_kv(adapter, reuse_too_fast_count=0):
    """Attach a mock _kv_collector to an adapter for evaluate_extra tests.

    With the collector refactor, evaluate_extra reads KV summary from
    self._kv_collector.last_summary (property) rather than receiving it as
    a parameter.
    """
    summary = _kv_summary(reuse_too_fast_count)

    class _MC:
        @property
        def last_summary(self):
            return summary

    adapter._kv_collector = _MC()


# ---------------------------------------------------------------------------
# Server flags
# ---------------------------------------------------------------------------

class TestServerFlagsNgram:
    def test_contains_speculative_config_flag(self):
        a = VllmSpecAdapter()
        assert "--speculative-config" in a.server_flags_extra()

    def test_omits_prefix_caching_and_chunked_prefill(self):
        """These two are injected by the vLLM job script (directly and via
        EXTRA_SERVER_FLAGS).  Emitting them here too double-recorded them
        in the stored server_flags string and produced duplicate-key warnings on
        repro, so server_flags_extra must leave them out."""
        flags = VllmSpecAdapter().server_flags_extra()
        assert "--enable-prefix-caching" not in flags
        assert "--enable-chunked-prefill" not in flags

    def test_config_json_parseable(self):
        a = VllmSpecAdapter()
        flags = a.server_flags_extra()
        json_part = flags.split("'")[1]
        cfg = json.loads(json_part)
        assert isinstance(cfg, dict)

    def test_ngram_method_field(self):
        assert VllmSpecAdapter()._build_spec_config()["method"] == "ngram"

    def test_ngram_num_speculative_tokens(self):
        assert VllmSpecAdapter(default_k=7)._build_spec_config()["num_speculative_tokens"] == 7

    def test_ngram_prompt_lookup_fields(self):
        cfg = VllmSpecAdapter()._build_spec_config()
        assert "prompt_lookup_max" in cfg
        assert "prompt_lookup_min" in cfg


class TestServerFlagsDraftModel:
    def test_model_key_present(self):
        cfg = VllmSpecAdapter(draft_model="facebook/opt-125m")._build_spec_config()
        assert cfg["model"] == "facebook/opt-125m"

    def test_no_method_key_for_real_model(self):
        cfg = VllmSpecAdapter(draft_model="facebook/opt-125m")._build_spec_config()
        assert "method" not in cfg

    def test_num_speculative_tokens_present(self):
        cfg = VllmSpecAdapter(draft_model="facebook/opt-125m", default_k=4)._build_spec_config()
        assert cfg["num_speculative_tokens"] == 4


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

class TestExtraMutations:
    def test_includes_spec_mutations(self):
        from fuzzer.adapters.vllm_spec.mutators import SPEC_MUTATIONS
        mutations = VllmSpecAdapter().extra_mutations()
        for m in SPEC_MUTATIONS:
            assert m in mutations

    def test_includes_base_mutations(self):
        from fuzzer.adapters.vllm_base.mutators import VLLM_MUTATIONS
        mutations = VllmSpecAdapter().extra_mutations()
        for m in VLLM_MUTATIONS:
            assert m in mutations


# ---------------------------------------------------------------------------
# _is_confirmable
# ---------------------------------------------------------------------------

class TestIsConfirmable:
    def test_true_when_all_temperature_none(self):
        t = _trace(_send(temperature=None), _send(temperature=None))
        assert VllmSpecAdapter()._is_confirmable(t) is True

    def test_true_when_all_temperature_zero(self):
        t = _trace(_send(temperature=0.0))
        assert VllmSpecAdapter()._is_confirmable(t) is True

    def test_false_when_any_nonzero_temperature(self):
        t = _trace(_send(temperature=0.0), _send(temperature=0.7))
        assert VllmSpecAdapter()._is_confirmable(t) is False

    def test_true_for_empty_trace(self):
        assert VllmSpecAdapter()._is_confirmable(_trace()) is True


# ---------------------------------------------------------------------------
# _has_reuse_too_fast
# ---------------------------------------------------------------------------

class TestHasReuseTooFast:
    def test_true_when_count_positive(self):
        assert VllmSpecAdapter()._has_reuse_too_fast(_kv_summary(1)) is True

    def test_false_when_count_zero(self):
        assert VllmSpecAdapter()._has_reuse_too_fast(_kv_summary(0)) is False

    def test_false_when_summary_none(self):
        assert VllmSpecAdapter()._has_reuse_too_fast(None) is False


# ---------------------------------------------------------------------------
# Suspicion signal helpers
# ---------------------------------------------------------------------------

class TestHasEmptyCompletion:
    def test_true_when_tokens_received_zero(self):
        r = _req(tokens_received=0, output_text="")
        assert VllmSpecAdapter()._has_empty_completion(_result(r)) is True

    def test_true_when_output_text_none(self):
        r = _req(tokens_received=0, output_text=None)
        assert VllmSpecAdapter()._has_empty_completion(_result(r)) is True

    def test_false_when_normal_completion(self):
        r = _req(tokens_received=5, output_text="hello")
        assert VllmSpecAdapter()._has_empty_completion(_result(r)) is False

    def test_false_when_cancelled(self):
        r = _req(tokens_received=0, output_text="", cancelled_at=0.5, completed_at=None)
        assert VllmSpecAdapter()._has_empty_completion(_result(r)) is False

    def test_false_when_not_completed(self):
        r = _req(tokens_received=0, output_text="", completed_at=None)
        assert VllmSpecAdapter()._has_empty_completion(_result(r)) is False

    def test_false_when_result_none(self):
        assert VllmSpecAdapter()._has_empty_completion(None) is False


class TestHasCancelMutation:
    def test_true_when_cancel_with_mutated_origin(self):
        t = _trace(_send(), _cancel(origin="mutated"))
        assert VllmSpecAdapter()._has_cancel_mutation(t) is True

    def test_false_when_cancel_without_origin(self):
        t = _trace(_send(), _cancel(origin=None))
        assert VllmSpecAdapter()._has_cancel_mutation(t) is False

    def test_false_when_no_cancel(self):
        t = _trace(_send())
        assert VllmSpecAdapter()._has_cancel_mutation(t) is False


class TestHasPrefixRaceMutation:
    def test_true_when_send_mutated_with_family_id(self):
        t = _trace(_send(origin="mutated", prompt_family_id="fam123"))
        assert VllmSpecAdapter()._has_prefix_race_mutation(t) is True

    def test_false_when_mutated_but_no_family_id(self):
        t = _trace(_send(origin="mutated", prompt_family_id=None))
        assert VllmSpecAdapter()._has_prefix_race_mutation(t) is False

    def test_false_when_family_id_but_not_mutated(self):
        t = _trace(_send(origin=None, prompt_family_id="fam123"))
        assert VllmSpecAdapter()._has_prefix_race_mutation(t) is False

    def test_false_when_no_send(self):
        t = _trace(_cancel())
        assert VllmSpecAdapter()._has_prefix_race_mutation(t) is False


# ---------------------------------------------------------------------------
# Candidate signature
# ---------------------------------------------------------------------------

class TestCandidateSig:
    def test_different_cancel_offset_gives_different_sig(self):
        t1 = _trace(_send(offset_ms=0), _cancel(offset_ms=100, origin="mutated"))
        t2 = _trace(_send(offset_ms=0), _cancel(offset_ms=200, origin="mutated"))
        a = VllmSpecAdapter()
        assert a._candidate_sig(t1) != a._candidate_sig(t2)

    def test_extra_cancel_gives_different_sig(self):
        t1 = _trace(_send(), _cancel(offset_ms=100))
        t2 = _trace(_send(), _cancel(offset_ms=100), _cancel(offset_ms=200))
        a = VllmSpecAdapter()
        assert a._candidate_sig(t1) != a._candidate_sig(t2)

    def test_stable_across_request_ids(self):
        """Same structure, different request_ids → same sig."""
        a = VllmSpecAdapter()
        ev1 = Event(action=ActionType.SEND, offset_ms=0, prompt_len=64,
                    max_tokens=64, stream=True)
        ev2 = Event(action=ActionType.SEND, offset_ms=0, prompt_len=64,
                    max_tokens=64, stream=True)
        # request_ids differ (uuid default) but structure is identical
        assert a._candidate_sig(Trace(seed=0, events=[ev1])) == \
               a._candidate_sig(Trace(seed=0, events=[ev2]))

    def test_different_prompt_family_id_gives_different_sig(self):
        t1 = _trace(_send(prompt_family_id="famA"))
        t2 = _trace(_send(prompt_family_id="famB"))
        a = VllmSpecAdapter()
        assert a._candidate_sig(t1) != a._candidate_sig(t2)

    # Fix 2: shape fields now included in sig
    def test_different_prefix_len_gives_different_sig(self):
        t1 = _trace(_send(offset_ms=0, prefix_len=64))
        t2 = _trace(_send(offset_ms=0, prefix_len=512))
        a = VllmSpecAdapter()
        assert a._candidate_sig(t1) != a._candidate_sig(t2)

    def test_different_prompt_len_gives_different_sig(self):
        t1 = _trace(Event(action=ActionType.SEND, offset_ms=0,
                          prompt_len=64, max_tokens=64, stream=True))
        t2 = _trace(Event(action=ActionType.SEND, offset_ms=0,
                          prompt_len=256, max_tokens=64, stream=True))
        a = VllmSpecAdapter()
        assert a._candidate_sig(t1) != a._candidate_sig(t2)

    def test_different_max_tokens_gives_different_sig(self):
        t1 = _trace(_send(offset_ms=0, max_tokens=64))
        t2 = _trace(_send(offset_ms=0, max_tokens=128))
        a = VllmSpecAdapter()
        assert a._candidate_sig(t1) != a._candidate_sig(t2)


# ---------------------------------------------------------------------------
# evaluate_extra enqueue logic
# ---------------------------------------------------------------------------

class TestEvaluateExtraEnqueue:
    def _adapter_with_confirm(self, enqueue_cooldown_s=0.0):
        return VllmSpecAdapter(
            confirm_url        = "http://nospec:8001",
            enqueue_cooldown_s = enqueue_cooldown_s,
        )

    def test_enqueues_on_reuse_too_fast(self):
        a = self._adapter_with_confirm()
        _inject_kv(a, reuse_too_fast_count=1)
        t = _trace(_send())
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 1

    def test_enqueues_on_cancel_mutation(self):
        a = self._adapter_with_confirm()
        t = _trace(_send(), _cancel(origin="mutated"))
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 1

    def test_enqueues_on_prefix_race_mutation(self):
        a = self._adapter_with_confirm()
        t = _trace(_send(origin="mutated", prompt_family_id="fam1"))
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 1

    def test_enqueues_on_empty_completion(self):
        a = self._adapter_with_confirm()
        t = _trace(_send())
        r = _req(tokens_received=0, output_text="")
        a.evaluate_extra(_result(r), None, trace=t)
        assert len(a._queue) == 1

    def test_does_not_enqueue_on_generic_mutated_origin_only(self):
        """SEND with origin=mutated but no family_id, no kv anomaly, normal output."""
        a = self._adapter_with_confirm()
        t = _trace(_send(origin="mutated", prompt_family_id=None))
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 0

    def test_does_not_enqueue_when_non_deterministic(self):
        a = self._adapter_with_confirm()
        _inject_kv(a, reuse_too_fast_count=1)
        t = _trace(_send(temperature=0.7))
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 0

    def test_does_not_enqueue_when_no_confirm_url(self):
        """The same-server determinism fallback no longer exists: without a
        no-spec comparison server there is nothing to confirm against, so
        enqueue is gated on confirm_url (adapter.py:204-217) and the queue
        stays empty rather than filling with unconfirmable candidates."""
        a = VllmSpecAdapter(confirm_url="")
        _inject_kv(a, reuse_too_fast_count=1)
        t = _trace(_send())
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 0

    def test_respects_enqueue_cooldown(self):
        a = self._adapter_with_confirm(enqueue_cooldown_s=999.0)
        _inject_kv(a, reuse_too_fast_count=1)
        t = _trace(_send())
        a.evaluate_extra(_result(_req()), None, trace=t)
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 1  # second call blocked by cooldown

    def test_respects_queue_cap(self):
        a = self._adapter_with_confirm(enqueue_cooldown_s=0.0)
        _inject_kv(a, reuse_too_fast_count=1)
        for i in range(MAX_QUEUE_SIZE + 5):
            # Each trace must have a distinct sig to bypass dedup
            t = _trace(Event(
                action=ActionType.SEND, offset_ms=i,
                prompt_len=64, max_tokens=64, stream=True,
            ))
            a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == MAX_QUEUE_SIZE


# ---------------------------------------------------------------------------
# run_confirmation_batch — gate conditions
# ---------------------------------------------------------------------------

class TestRunConfirmationBatchGates:
    def test_returns_empty_when_queue_empty(self):
        a = VllmSpecAdapter(confirm_url="http://nospec:8001")
        result = asyncio.run(a.run_confirmation_batch(None, "http://spec:8000"))
        assert result == []


# ---------------------------------------------------------------------------
# Fix 1: minimum valid pairs
# Fix 3: cooldown set on all exits
# ---------------------------------------------------------------------------

class TestConfirmationMinValidPairs:
    """_run_confirmation must not file a finding when fewer than 2 runs survive."""

    def _make_adapter(self):
        return VllmSpecAdapter(
            confirm_url         = "http://nospec:8001",
            confirm_k           = 3,
            confirm_needed_frac = 2 / 3,
            confirm_cooldown_s  = 600.0,
        )

    def _make_candidate(self, adapter):
        t = _trace(_send())
        from fuzzer.adapters.vllm_spec.oracles import SpecCandidate
        return SpecCandidate(
            trace=t, iteration=1,
            sig=adapter._candidate_sig(t),
            timestamp=time.time(),
        )

    def _run(self, adapter, executor_run_fn, liveness_fn):
        """Helper: patch adapter methods and run _run_confirmation."""
        async def mock_flush(url):
            return None

        adapter.check_liveness = liveness_fn
        adapter.flush_state    = mock_flush

        class _Executor:
            async def run(self, trace, url):
                return await executor_run_fn(trace, url)

        spec_url    = "http://spec:8000"
        confirm_url = adapter.confirm_url if adapter.confirm_url else spec_url
        return asyncio.run(adapter._run_confirmation(_Executor(), spec_url, confirm_url))

    def test_no_finding_when_only_one_valid_run(self):
        """If only 1 run survives liveness, skip regardless of divergence."""
        adapter = self._make_adapter()
        candidate = self._make_candidate(adapter)
        adapter._queue.append(candidate)
        adapter._last_confirm_at = 0.0

        call_count = [0]

        async def mock_liveness(url):
            call_count[0] += 1
            # Only the first pair (2 liveness calls) passes; rest fail.
            return call_count[0] <= 2

        async def mock_run(trace, url):
            return _result(_req(output_text="spec" if "spec:8000" in url else "nospec"))

        findings = self._run(adapter, mock_run, mock_liveness)
        assert findings == []

    def test_cooldown_set_when_insufficient_valid_runs(self):
        """Fix 3: cooldown is set even when run count is too low to confirm."""
        adapter = self._make_adapter()
        candidate = self._make_candidate(adapter)
        adapter._queue.append(candidate)
        adapter._last_confirm_at = 0.0

        async def mock_liveness(url):
            return False  # all liveness checks fail → 0 valid runs

        async def mock_run(trace, url):
            return _result(_req())

        self._run(adapter, mock_run, mock_liveness)
        assert candidate.sig in adapter._confirm_cooldowns

    def test_cooldown_set_on_non_confirming_candidate(self):
        """Fix 3: cooldown is set when diverged_count < needed (non-confirming)."""
        adapter = self._make_adapter()
        candidate = self._make_candidate(adapter)
        adapter._queue.append(candidate)
        adapter._last_confirm_at = 0.0

        async def mock_liveness(url):
            return True

        async def mock_run(trace, url):
            return _result(_req(output_text="identical"))

        findings = self._run(adapter, mock_run, mock_liveness)
        assert findings == []
        assert candidate.sig in adapter._confirm_cooldowns
