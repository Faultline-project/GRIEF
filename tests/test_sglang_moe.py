"""Tests for SglangMoEAdapter: mutations, oracle, confirmation queue."""
import pytest
import random

from fuzzer.adapters.sglang_moe.adapter import SglangMoEAdapter, MAX_QUEUE_SIZE
from fuzzer.adapters.sglang_moe.mutators import (
    concurrent_expert_stress,
    jitter_offsets,
    batch_size_step,
    MOE_MUTATIONS,
)
from fuzzer.adapters.sglang_moe.oracles import compare_batch_vs_solo, MoECandidate
from fuzzer.core.types import TraceResult, RequestResult
from fuzzer.trace import Trace, Event, ActionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send(offset_ms=0, family_id=None, prompt_len=64, max_tokens=64,
          temperature=0.0) -> Event:
    return Event(
        action=ActionType.SEND,
        offset_ms=offset_ms,
        request_id=f"r_{offset_ms}",
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        prompt_family_id=family_id,
    )


def _trace(*events):
    return Trace(seed=0, events=list(events))


def _req(request_id="r1", output_text="hello", error=None,
         cancelled_at=None, completed_at=1.0):
    return RequestResult(
        request_id=request_id,
        sent_at=0.0,
        completed_at=completed_at,
        cancelled_at=cancelled_at,
        output_text=output_text,
        tokens_received=len(output_text.split()),
        error=error,
    )


def _result(*reqs):
    return TraceResult(trace_seed=0, started_at=0.0, finished_at=1.0,
                       requests=list(reqs))


def _rng(seed=42):
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------

class TestAdapterInit:
    def test_moe_adapter_is_target_adapter(self):
        """Phase 4: the new-shape SglangMoEAdapter must still satisfy
        the TargetAdapter contract the runner consumes. If this fails,
        Runner(config, adapter=SglangMoEAdapter(), ...) breaks."""
        from fuzzer.core.adapter import TargetAdapter
        assert isinstance(SglangMoEAdapter(), TargetAdapter)

    def test_moe_capabilities_carry_moe_mutators_not_base(self):
        """The MoE Capabilities must list only the 3 MoE mutators —
        the legacy adapter intentionally dropped the RadixAttention
        mutators (they stamp prompt_family_id which produces false
        MoE confirmations). The new shape preserves that choice."""
        a = SglangMoEAdapter()
        names = {m.name for m in a.capabilities.mutators}
        assert names == {
            "concurrent_expert_stress",
            "jitter_offsets",
            "batch_size_step",
        }

    def test_moe_strategy_has_no_corpus_penalty(self):
        """Per the Phase 4 design decision: legacy MoE had no corpus
        bias, so the new shape's ``MoeExplorationStrategy`` returns
        1.0 from ``score_multiplier`` for all entries. The #24845
        attractor is a KV-corruption trigger, orthogonal to the
        MoE batch-isolation oracle."""
        from types import SimpleNamespace
        a = SglangMoEAdapter()
        assert a.strategy.fingerprint_penalties == {}
        # Even a trace shaped like the #24845 attractor gets 1.0:
        from fuzzer.trace import ActionType, Event, Trace
        attractor = Trace(seed=0, events=[
            Event(action=ActionType.SEND, offset_ms=0,  request_id="r1",
                  prompt_len=64, max_tokens=16, prompt_family_id="fam0"),
            Event(action=ActionType.SEND, offset_ms=10, request_id="r2",
                  prompt_len=64, max_tokens=16, prompt_family_id="fam0"),
            Event(action=ActionType.CANCEL, offset_ms=20, request_id="r1"),
        ])
        assert a.strategy.score_multiplier(SimpleNamespace(trace=attractor)) == 1.0


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

class TestMoEMutations:
    def test_extra_mutations_includes_moe_mutations(self):
        a = SglangMoEAdapter()
        names = {m.__name__ for m in a.extra_mutations()}
        for m in MOE_MUTATIONS:
            assert m.__name__ in names

    def test_concurrent_expert_stress_adds_events(self):
        t = _trace(_send(0, family_id="fam0"))
        result = concurrent_expert_stress(t, _rng())
        assert len(result.events) > len(t.events)

    def test_concurrent_expert_stress_noop_empty(self):
        t = Trace(seed=0, events=[])
        assert concurrent_expert_stress(t, _rng()) is t

    def test_jitter_offsets_changes_timing(self):
        t = _trace(_send(100), _send(200))
        result = jitter_offsets(t, _rng())
        offsets_before = [ev.offset_ms for ev in t.events]
        offsets_after  = [ev.offset_ms for ev in result.events]
        assert offsets_before != offsets_after

    def test_jitter_offsets_sorted(self):
        t = _trace(*[_send(i * 50) for i in range(5)])
        result = jitter_offsets(t, _rng())
        offsets = [ev.offset_ms for ev in result.events]
        assert offsets == sorted(offsets)

    def test_jitter_min_separation_enforced(self):
        """Consecutive SEND events must be at least 1ms apart after jitter."""
        t = _trace(*[_send(0) for _ in range(4)])
        result = jitter_offsets(t, _rng())
        sends = [ev for ev in result.events if ev.action == ActionType.SEND]
        for i in range(1, len(sends)):
            assert sends[i].offset_ms > sends[i - 1].offset_ms

    def test_batch_size_step_noop_single_send(self):
        t = _trace(_send(0, family_id="f"))
        assert batch_size_step(t, _rng()) is t

    def test_batch_size_step_changes_count(self):
        t = _trace(_send(0, family_id="f0"), _send(10, family_id="f1"),
                   _send(20, family_id="f2"))
        rng = _rng(1)
        result = batch_size_step(t, rng)
        assert len(result.events) != len(t.events)


# ---------------------------------------------------------------------------
# Mutation weights
# ---------------------------------------------------------------------------

class TestMutationWeights:
    def test_concurrent_expert_stress_upweighted(self):
        a = SglangMoEAdapter()
        w = a.mutation_weights()
        assert w.get("concurrent_expert_stress", 1.0) > 1.0

    def test_prefix_mutations_disabled(self):
        a = SglangMoEAdapter()
        w = a.mutation_weights()
        assert w.get("inject_max_prefix", 1.0) == 0.0
        assert w.get("chunk_boundary_align", 1.0) == 0.0


# ---------------------------------------------------------------------------
# compare_batch_vs_solo oracle
# ---------------------------------------------------------------------------

class TestCompareBatchVsSolo:
    def test_identical_outputs_no_divergence(self):
        batch  = _result(_req("r1", "hello"))
        solo   = {"r1": _result(_req("r1", "hello"))}
        assert compare_batch_vs_solo(batch, solo) == []

    def test_different_outputs_divergence(self):
        batch  = _result(_req("r1", "hello"))
        solo   = {"r1": _result(_req("r1", "world"))}
        divs   = compare_batch_vs_solo(batch, solo)
        assert len(divs) == 1
        assert divs[0]["request_id"] == "r1"
        assert divs[0]["batch_output"] == "hello"
        assert divs[0]["solo_output"]  == "world"

    def test_cancelled_skipped(self):
        batch = _result(_req("r1", "hello", cancelled_at=0.5))
        solo  = {"r1": _result(_req("r1", "world", cancelled_at=0.5))}
        assert compare_batch_vs_solo(batch, solo) == []

    def test_error_skipped(self):
        batch = _result(_req("r1", "hello", error="timeout"))
        solo  = {"r1": _result(_req("r1", "world", error="timeout"))}
        assert compare_batch_vs_solo(batch, solo) == []

    def test_missing_solo_skipped(self):
        batch = _result(_req("r1", "hello"))
        assert compare_batch_vs_solo(batch, {}) == []


# ---------------------------------------------------------------------------
# evaluate_extra enqueue logic
# ---------------------------------------------------------------------------

class TestEnqueue:
    def _confirmable_trace(self):
        return _trace(
            _send(0,  family_id="fam0", temperature=0.0),
            _send(50, family_id="fam1", temperature=0.0),
        )

    def test_enqueues_confirmable_trace(self):
        a = SglangMoEAdapter(enqueue_cooldown_s=0.0)
        a.evaluate_extra(_result(_req()), None, trace=self._confirmable_trace())
        assert len(a._queue) == 1

    def test_no_enqueue_when_stochastic(self):
        a = SglangMoEAdapter()
        t = _trace(
            _send(0,  family_id="f", temperature=0.7),
            _send(50, family_id="f", temperature=0.7),
        )
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 0

    def test_no_enqueue_single_send(self):
        a = SglangMoEAdapter()
        t = _trace(_send(0, family_id="f", temperature=0.0))
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 0

    def test_no_enqueue_no_family_id(self):
        a = SglangMoEAdapter()
        t = _trace(_send(0, temperature=0.0), _send(50, temperature=0.0))
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 0

    def test_cooldown_deduplicates(self):
        a = SglangMoEAdapter(enqueue_cooldown_s=999.0)
        t = self._confirmable_trace()
        a.evaluate_extra(_result(_req()), None, trace=t)
        a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) == 1

    def test_queue_cap_respected(self):
        a = SglangMoEAdapter(enqueue_cooldown_s=0.0)
        for i in range(MAX_QUEUE_SIZE + 5):
            t = _trace(
                _send(i,      family_id=f"f{i}",   temperature=0.0),
                _send(i + 1,  family_id=f"f{i+1}", temperature=0.0),
            )
            a.evaluate_extra(_result(_req()), None, trace=t)
        assert len(a._queue) <= MAX_QUEUE_SIZE

    def test_no_enqueue_without_trace(self):
        a = SglangMoEAdapter()
        a.evaluate_extra(_result(_req()), None, trace=None)
        assert len(a._queue) == 0


# ---------------------------------------------------------------------------
# run_confirmation_batch gating
# ---------------------------------------------------------------------------

class TestConfirmationGating:
    def test_empty_queue_returns_empty(self):
        import asyncio
        a = SglangMoEAdapter()
        assert asyncio.run(a.run_confirmation_batch(None, "http://x")) == []

    def test_interval_gate_blocks_when_queue_small(self):
        import asyncio, time
        a = SglangMoEAdapter(interval_min=999.0, batch_size=10)
        a._queue.append(MoECandidate(
            trace=_trace(_send(0, family_id="f")),
            iteration=1, sig="abc", timestamp=time.time(),
        ))
        a._last_confirm_at = time.time()
        result = asyncio.run(a.run_confirmation_batch(None, "http://x"))
        assert result == []
