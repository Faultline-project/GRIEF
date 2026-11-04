"""Tests for the confirmation ladder (fuzzer/confirm/) and its runner wiring.

Everything is driven by scripted executors: no server, no GPU.

  1. ``ladder.judge``: the verdict table.
  2. Arm construction: salts, knobs, solo extraction, prompt identity.
  3. ``bi_guard``: GPU-arch resolution for the RTX A6000s.
  4. ``Corpus.adjust``.
  5. Runner: enqueue instead of file, drain, filing, log, feedback,
     queue cap, signature cooldown, startup guard, ladder-off parity.
"""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from fuzzer.confirm import ladder as L
from fuzzer.confirm.bi_guard import infer_fa_version, resolve_gpu_arch
from fuzzer.core.corpus import Corpus, CorpusEntry
from fuzzer.core.runner import Runner, RunnerConfig
from fuzzer.core.types import BugClass, OracleResult, RequestResult, TraceResult
from fuzzer.monitor import SchedulerFingerprint
from fuzzer.trace import ActionType, Event, Trace


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _send(rid, offset_ms=0, **kw):
    base = dict(action=ActionType.SEND, offset_ms=offset_ms, request_id=rid,
                prompt_len=32, max_tokens=8, stream=True)
    base.update(kw)
    return Event(**base)


def _trace(*events, seed=1):
    return Trace(seed=seed, events=list(events))


def _arm(name, text="x", ok=True, twin_text=None, cache=None, reason="bad"):
    if not ok:
        return L.ArmOutcome(name, False, reason)
    return L.ArmOutcome(name, True, text=text, digest=text,
                        twin_text=twin_text, cache_read_tokens=cache)


def _arms(s1="x", s2="x", b1="x", b2="x", bs="x", **kw):
    return {"S1": _arm("S1", s1), "S2": _arm("S2", s2), "B1": _arm("B1", b1),
            "B2": _arm("B2", b2), "BS": _arm("BS", bs), **kw}


def _req(rid, text, **kw):
    return RequestResult(request_id=rid, sent_at=0.0, completed_at=1.0,
                         output_text=text, **kw)


def _result(*reqs):
    return TraceResult(trace_seed=1, started_at=0.0, finished_at=1.0,
                       requests=list(reqs))


def _fingerprint(**overrides):
    values = {f.name: 0 for f in dataclasses.fields(SchedulerFingerprint)}
    values.update(overrides)
    return SchedulerFingerprint(**values)


def _kv_bug(score=10.0):
    return OracleResult(passed=False, bug_class=BugClass.KV_CORRUPTION,
                        detail="Non-deterministic output for identical "
                               "prompt/temperature=0 (possible KV cache "
                               "corruption, similarity=0.50)",
                        score=score, probe_metadata={"similarity": 0.5})


# ---------------------------------------------------------------------------
# 1. judge
# ---------------------------------------------------------------------------

class TestJudge:
    def test_all_equal_is_batch_variance(self):
        v = L.judge(_arms())
        assert v.verdict == L.VERDICT_BATCH_VARIANCE
        assert not v.filed

    def test_solo_pair_differs(self):
        v = L.judge(_arms(s1="abc", s2="abX"))
        assert v.verdict == L.VERDICT_SOLO_NONDETERMINISTIC
        assert v.deciding_pair == ("S1", "S2")
        assert v.diverge_char_idx == 2
        assert v.filed

    def test_solo_nondeterminism_decides_even_if_bi_arms_failed(self):
        arms = _arms(s1="a", s2="b")
        arms["B1"] = _arm("B1", ok=False, reason="error: 500")
        assert L.judge(arms).verdict == L.VERDICT_SOLO_NONDETERMINISTIC

    def test_failed_arm_is_uninformative_not_a_pass(self):
        arms = _arms()
        arms["BS"] = _arm("BS", ok=False, reason="r0: cancelled")
        v = L.judge(arms)
        assert v.verdict == L.VERDICT_UNINFORMATIVE
        assert "BS: r0: cancelled" in v.reason
        assert not v.filed

    def test_bi_replays_disagree(self):
        v = L.judge(_arms(b1="aaa", b2="aab", bs="aaa"))
        assert v.verdict == L.VERDICT_BI_SCHEDULE_DEPENDENT
        assert v.deciding_pair == ("B1", "B2")

    def test_bi_batch_vs_bi_solo(self):
        v = L.judge(_arms(b1="batch", b2="batch", bs="solo!"))
        assert v.verdict == L.VERDICT_BI_BATCH_DEPENDENT
        assert v.deciding_pair == ("B1", "BS")
        assert v.comparisons == {"S1!=S2": False, "B1!=B2": False, "B1!=BS": True}

    def test_twin_relation_persists_in_both_replays(self):
        arms = _arms()
        arms["B1"] = _arm("B1", "x", twin_text="y", cache=16)
        arms["B2"] = _arm("B2", "x", twin_text="y", cache=16)
        v = L.judge(arms, twin=True, twin_needs_cache_hit=True)
        assert v.verdict == L.VERDICT_BI_RELATION_PERSISTS
        assert v.filed

    def test_twin_relation_in_one_replay_only_is_batch_variance(self):
        arms = _arms()
        arms["B1"] = _arm("B1", "x", twin_text="y")
        arms["B2"] = _arm("B2", "x", twin_text="x")
        assert L.judge(arms, twin=True).verdict == L.VERDICT_BATCH_VARIANCE

    @pytest.mark.parametrize("hits", [(0, 16), (None, 16), (16, 0)])
    def test_cache_relation_not_recreated_is_uninformative(self, hits):
        arms = _arms()
        arms["B1"] = _arm("B1", "x", twin_text="x", cache=hits[0])
        arms["B2"] = _arm("B2", "x", twin_text="x", cache=hits[1])
        v = L.judge(arms, twin=True, twin_needs_cache_hit=True)
        assert v.verdict == L.VERDICT_UNINFORMATIVE
        assert "not re-created" in v.reason

    def test_cache_relation_recreated_and_agreeing_is_batch_variance(self):
        arms = _arms()
        arms["B1"] = _arm("B1", "x", twin_text="x", cache=16)
        arms["B2"] = _arm("B2", "x", twin_text="x", cache=16)
        v = L.judge(arms, twin=True, twin_needs_cache_hit=True)
        assert v.verdict == L.VERDICT_BATCH_VARIANCE

    def test_filed_verdicts(self):
        assert L.FILED_VERDICTS == {
            L.VERDICT_SOLO_NONDETERMINISTIC, L.VERDICT_BI_SCHEDULE_DEPENDENT,
            L.VERDICT_BI_BATCH_DEPENDENT, L.VERDICT_BI_RELATION_PERSISTS}

    def test_first_diff(self):
        assert L.first_diff("abc", "abc") is None
        assert L.first_diff("abc", "abd") == 2
        assert L.first_diff("ab", "abc") == 2


class TestArmOutcome:
    @pytest.mark.parametrize("req,why", [
        (None, "missing"),
        (dict(error="boom"), "error"),
        (dict(cancelled_at=1.0), "cancelled"),
        (dict(text=None), "no output_text"),
    ])
    def test_unusable_request(self, req, why):
        reqs = []
        if req is not None:
            kw = dict(req)
            text = kw.pop("text", "out")
            reqs.append(_req("r0", text, **kw))
        o = L.arm_outcome("S1", _result(*reqs), "r0")
        assert not o.ok and why in o.reason

    def test_exception_is_recorded(self):
        o = L.arm_outcome("B1", None, "r0", exc=TimeoutError("slow"))
        assert not o.ok and "TimeoutError" in o.reason

    def test_twin_read_only_in_batched_arms(self):
        res = _result(_req("r0", "a", cache_read_tokens=16), _req("r0~pfx", "b"))
        b1 = L.arm_outcome("B1", res, "r0", "r0~pfx")
        assert b1.ok and b1.twin_text == "b" and b1.cache_read_tokens == 16
        s1 = L.arm_outcome("S1", _result(_req("r0", "a")), "r0", "r0~pfx")
        assert s1.ok and s1.twin_text is None

    def test_missing_twin_in_batched_arm_is_not_ok(self):
        o = L.arm_outcome("B2", _result(_req("r0", "a")), "r0", "r0~pfx")
        assert not o.ok and "twin" in o.reason


# ---------------------------------------------------------------------------
# 2. arm construction
# ---------------------------------------------------------------------------

class TestArms:
    def _batch(self):
        return _trace(_send("r0"), _send("r1", 10, cache_salt="y"),
                      _send("r2", 20), _send("r3", 30, cache_salt="y"),
                      Event(action=ActionType.CANCEL, offset_ms=40,
                            request_id="r2"))

    def test_salt_groups_preserved_within_an_arm(self):
        arm = L.salted_arm(self._batch(), random.Random(0), "B1")
        salt = {e.request_id: e.cache_salt for e in arm.events
                if e.action == ActionType.SEND}
        assert salt["r0"] == salt["r2"]           # both were unsalted
        assert salt["r1"] == salt["r3"]           # both were "y"
        assert salt["r0"] != salt["r1"]
        assert all(s.startswith("grief-cf-B1-") for s in salt.values())

    def test_arms_never_share_a_salt_and_original_is_untouched(self):
        t = self._batch()
        rng = random.Random(0)
        a = L.salted_arm(t, rng, "B1")
        b = L.salted_arm(t, rng, "B2")
        sa = {e.cache_salt for e in a.events if e.action == ActionType.SEND}
        sb = {e.cache_salt for e in b.events if e.action == ActionType.SEND}
        assert not sa & sb
        assert [e.cache_salt for e in t.events] == [None, "y", None, "y", None]

    def test_knobs_only_on_compared_requests(self):
        arm = L.salted_arm(self._batch(), random.Random(0), "B1", ("r0",))
        ev = {e.request_id: e for e in arm.events if e.action == ActionType.SEND}
        assert ev["r0"].logprobs == L.ARM_LOGPROBS and ev["r0"].include_usage
        assert ev["r1"].logprobs is None and ev["r1"].include_usage is None

    def test_solo_trace(self):
        t = self._batch()
        solo = L.solo_trace(t, "r2")
        assert [e.request_id for e in solo.events] == ["r2"]
        assert solo.events[0].offset_ms == 0
        assert solo.events[0].action == ActionType.SEND   # the CANCEL is dropped
        assert solo.events[0].prompt_family_id == "r2"
        assert t.events[2].offset_ms == 20                # original untouched
        with pytest.raises(KeyError):
            L.solo_trace(t, "nope")

    def test_solo_prompt_is_byte_identical(self):
        from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
        adapter = VllmBaseAdapter()
        t = _trace(_send("r0", prefix_len=16), _send("r1", 5, prefix_len=16))
        solo = L.salted_arm(L.solo_trace(t, "r1"), random.Random(0), "S1", ("r1",))
        _, orig = adapter.build_payload(t.events[1])
        _, mine = adapter.build_payload(solo.events[0])
        assert mine["prompt"] == orig["prompt"]

    def test_build_arms(self):
        cand = L.Candidate(report_trace=self._batch(), replay_trace=self._batch(),
                           rid="r0", source="determinism", bugs=[], iteration=1)
        arms = L.build_arms(cand, random.Random(0))
        assert list(arms) == list(L.ARM_NAMES)
        assert len(arms["S1"].events) == len(arms["BS"].events) == 1
        assert len(arms["B1"].events) == 5
        solo_salts = {arms[n].events[0].cache_salt for n in ("S1", "S2", "BS")}
        assert len(solo_salts) == 3

    def test_priority_order(self):
        mk = lambda p, s, q: L.Candidate(report_trace=_trace(), replay_trace=_trace(),
                                         rid="r", source="d", bugs=[], iteration=0,
                                         priority=p, score=s, seq=q)
        cands = [mk(L.PRIORITY_LOW, 99, 1), mk(L.PRIORITY_HIGH, 1, 3),
                 mk(L.PRIORITY_HIGH, 5, 4), mk(L.PRIORITY_HIGH, 5, 2)]
        order = [c.seq for c in sorted(cands, key=lambda c: c.sort_key())]
        assert order == [2, 4, 3, 1]


# ---------------------------------------------------------------------------
# 3. bi_guard GPU arch
# ---------------------------------------------------------------------------

class TestGpuArch:
    @pytest.mark.parametrize("log,label,fa", [
        ("Device: NVIDIA RTX A6000", "sm86", 2),
        ("Device: NVIDIA RTX A6000 Ada Generation", "sm89", 2),
        ("Device: NVIDIA GeForce RTX 4090", "sm89", 2),
        ("Device: NVIDIA H100 80GB HBM3", "sm90", 3),
    ])
    def test_log_resolution(self, log, label, fa):
        arch = resolve_gpu_arch({}, log_text=log)
        assert arch.label == label
        assert infer_fa_version(arch)[0] == fa


# ---------------------------------------------------------------------------
# 4. Corpus.adjust
# ---------------------------------------------------------------------------

class TestCorpusAdjust:
    def _corpus(self, tmp_path):
        (tmp_path / "seeds").mkdir()
        (tmp_path / "corpus").mkdir()
        return Corpus(seeds_dir=str(tmp_path / "seeds"),
                      corpus_dir=str(tmp_path / "corpus"))

    def test_adjust_down_clamps_at_floor(self, tmp_path):
        c = self._corpus(tmp_path)
        fp = _fingerprint(had_swap=1)
        e = CorpusEntry(trace=Trace(seed=3, events=[]), score=1.5, fingerprint=fp)
        c._entries.append(e)
        assert c.adjust(fp, -2.0, floor=0.0) is True
        assert e.score == 0.0

    def test_boost_unchanged(self, tmp_path):
        c = self._corpus(tmp_path)
        fp = _fingerprint(had_swap=1)
        e = CorpusEntry(trace=Trace(seed=4, events=[]), score=1.0, fingerprint=fp)
        c._entries.append(e)
        assert c.boost(fp) is True and e.score == 3.0
        assert c.adjust(None, 1.0) is False
        assert c.adjust(_fingerprint(had_swap=2), 1.0) is False


# ---------------------------------------------------------------------------
# 5. Runner wiring
# ---------------------------------------------------------------------------

STOCK = "http://stock:8000"
BI = "http://bi:8001"


def _runner(tmp_path, **overrides):
    from fuzzer.executor import PythonExecutor
    from fuzzer.monitor import MockMonitor
    for name in ("seeds", "corpus", "findings"):
        (tmp_path / name).mkdir(exist_ok=True)
    cfg = dict(base_url=STOCK, seeds_dir=str(tmp_path / "seeds"),
               corpus_dir=str(tmp_path / "corpus"),
               findings_dir=str(tmp_path / "findings"),
               confirm_ladder=True, confirm_bi_url=BI)
    cfg.update(overrides)
    return Runner(RunnerConfig(**cfg), executor=PythonExecutor(),
                  monitor=MockMonitor())


def _script(runner, outputs):
    """outputs(url, trace) -> {rid: text}; records every call."""
    calls = []

    async def run(trace, url):
        calls.append((url, trace))
        texts = outputs(url, trace)
        return _result(*[_req(rid, t) for rid, t in texts.items()])

    runner.executor.run = run
    return calls


def _read_log(tmp_path):
    p = tmp_path / "findings" / "confirmation_log.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _findings(tmp_path):
    return sorted((tmp_path / "findings").glob("finding_*.json"))


def _enqueue(runner, trace, rid="r0", priority=L.PRIORITY_HIGH, score=10.0):
    runner._enqueue_confirmation(
        report_trace=trace, replay_trace=trace, rid=rid, source="determinism",
        bugs=[_kv_bug(score)], iteration=7, priority=priority,
        extra={"request_id": rid})


class TestRunnerLadder:
    @pytest.mark.asyncio
    async def test_batch_variance_is_logged_not_filed_and_down_weighted(self, tmp_path):
        r = _runner(tmp_path)
        trace = _trace(_send("r0"), _send("r1", 5))
        r._last_fingerprint = _fingerprint(had_swap=1)
        _enqueue(r, trace)
        calls = _script(r, lambda url, t: {e.request_id: "same" for e in t.events})
        adjusted = []
        r.corpus.adjust = lambda fp, d, floor=None: adjusted.append((d, floor)) or True

        await r._drain_confirmations(10)

        assert _findings(tmp_path) == []
        [rec] = _read_log(tmp_path)
        assert rec["verdict"] == L.VERDICT_BATCH_VARIANCE
        assert rec["filed"] is False and rec["feedback"] == "down"
        assert set(rec["arms"]) == set(L.ARM_NAMES)
        assert adjusted == [(-2.0, 0.0)]
        # two servers: S arms on stock, B arms on BI
        urls = sorted(u for u, _ in calls)
        assert urls.count(STOCK) == 2 and urls.count(BI) == 3

    @pytest.mark.asyncio
    async def test_bi_batch_dependent_is_filed_with_confirmation(self, tmp_path):
        r = _runner(tmp_path)
        trace = _trace(_send("r0"), _send("r1", 5))
        _enqueue(r, trace)

        def outputs(url, t):
            batched = len(t.events) > 1
            return {e.request_id: ("in-batch" if (url == BI and batched) else "alone")
                    for e in t.events}
        _script(r, outputs)

        await r._drain_confirmations(10)

        [path] = _findings(tmp_path)
        rep = json.loads(path.read_text())
        assert rep["confirmation"]["verdict"] == L.VERDICT_BI_BATCH_DEPENDENT
        assert rep["confirmation"]["deciding_pair"] == ["B1", "BS"]
        assert rep["iteration"] == 7                      # discovery iteration
        assert rep["confirmation"]["confirmed_at_iteration"] == 10
        assert rep["request_id"] == "r0"
        [rec] = _read_log(tmp_path)
        assert rec["filed"] is True

    @pytest.mark.asyncio
    async def test_drain_respects_priority_and_max(self, tmp_path):
        r = _runner(tmp_path, confirm_max_per_drain=1)
        trace = _trace(_send("r0"), _send("r1", 5))
        _enqueue(r, trace, rid="r1", priority=L.PRIORITY_LOW, score=50)
        _enqueue(r, trace, rid="r0", priority=L.PRIORITY_HIGH, score=1)
        _script(r, lambda url, t: {e.request_id: "same" for e in t.events})

        await r._drain_confirmations(10)

        [rec] = _read_log(tmp_path)
        assert rec["request_id"] == "r0" and rec["priority"] == "nontie"
        assert len(r._confirm_queue) == 1

    def test_queue_cap_drops_lowest(self, tmp_path):
        r = _runner(tmp_path, confirm_queue_max=1)
        trace = _trace(_send("r0"), _send("r1", 5))
        _enqueue(r, trace, rid="r0", priority=L.PRIORITY_HIGH)
        _enqueue(r, trace, rid="r1", priority=L.PRIORITY_LOW)
        assert [c.rid for c in r._confirm_queue] == ["r0"]
        [rec] = _read_log(tmp_path)
        assert rec["verdict"] == "DROPPED" and rec["request_id"] == "r1"
        assert r._confirm_stats["dropped"] == 1

    @pytest.mark.asyncio
    async def test_signature_cooldown_after_repeated_batch_variance(self, tmp_path):
        r = _runner(tmp_path, confirm_sig_cooldown_after=2)
        trace = _trace(_send("r0"), _send("r1", 5))
        _script(r, lambda url, t: {e.request_id: "same" for e in t.events})
        for _ in range(2):
            _enqueue(r, trace)
            await r._drain_confirmations(10)
        _enqueue(r, trace)
        assert r._confirm_queue == []
        assert r._confirm_stats["cooled"] == 1

    @pytest.mark.asyncio
    async def test_feedback_off(self, tmp_path):
        r = _runner(tmp_path, confirm_feedback=False)
        r._last_fingerprint = _fingerprint(had_swap=1)
        _enqueue(r, _trace(_send("r0")))
        _script(r, lambda url, t: {e.request_id: "same" for e in t.events})
        r.corpus.adjust = lambda *a, **k: pytest.fail("feedback must be off")
        await r._drain_confirmations(10)
        assert _read_log(tmp_path)[0]["feedback"] == "off"

    def test_summary_logs_unprocessed(self, tmp_path):
        r = _runner(tmp_path)
        _enqueue(r, _trace(_send("r0")))
        r._dump_confirmation_summary()
        s = json.loads((tmp_path / "findings" / "confirmation_summary.json").read_text())
        assert s["counts"]["enqueued"] == 1 and s["counts"]["unprocessed"] == 1
        assert _read_log(tmp_path)[0]["verdict"] == "UNPROCESSED"


class TestDeterminismProbeRouting:
    """With the ladder on, the determinism probe enqueues instead of filing,
    and an FP tie becomes a low-priority candidate instead of being dropped."""

    async def _probe(self, tmp_path, second_text, top, **cfg):
        r = _runner(tmp_path, **cfg)
        trace = _trace(_send("r0"))
        first = _result(_req("r0", "aa bb cc"))

        async def second(*_a, **_k):
            return _result(_req("r0", second_text, output_tokens=list(second_text),
                                output_top_logprobs=top))
        r.executor.run = second
        filed = []

        async def capture(trace_, bugs, iteration, *a, **kw):
            filed.append(bugs[0])
        r._handle_finding = capture
        await r._determinism_probe(trace, first, 3)
        return r, filed

    @pytest.mark.asyncio
    async def test_wide_gap_is_enqueued_high(self, tmp_path):
        r, filed = await self._probe(tmp_path, "aa Xb cc", [{"a": -0.1, "b": -5.0}] * 8)
        assert filed == []
        [c] = r._confirm_queue
        assert c.priority == L.PRIORITY_HIGH and c.rid == "r0"
        assert c.logprob_gap == pytest.approx(4.9)

    @pytest.mark.asyncio
    async def test_tie_is_enqueued_low(self, tmp_path):
        r, filed = await self._probe(tmp_path, "aa Xb cc", [{"a": -0.1, "b": -0.2}] * 8)
        assert filed == []
        [c] = r._confirm_queue
        assert c.priority == L.PRIORITY_LOW

    @pytest.mark.asyncio
    async def test_ladder_off_keeps_old_behaviour(self, tmp_path):
        r, filed = await self._probe(tmp_path, "aa Xb cc", [{"a": -0.1, "b": -0.2}] * 8,
                                     confirm_ladder=False)
        assert filed == [] and r._confirm_queue == []          # tie still dropped
        r, filed = await self._probe(tmp_path, "aa Xb cc", [{"a": -0.1, "b": -5.0}] * 8,
                                     confirm_ladder=False)
        assert len(filed) == 1 and r._confirm_queue == []      # filed at once


class TestPrefixEquivRouting:
    async def _probe(self, tmp_path, plain, salt1, salt2):
        r = _runner(tmp_path)
        trace = _trace(_send("r0"))
        first = _result(_req("r0", "aa bb cc"))

        async def run(probe_trace, *_a, **_k):
            return _result(_req("r0", plain, cache_read_tokens=100),
                           _req("r0~pfx", salt1, cache_read_tokens=0),
                           _req("r0~pfx2", salt2, cache_read_tokens=0))
        r.executor.run = run
        r._handle_finding = lambda *a, **k: pytest.fail("must enqueue, not file")
        await r._prefix_equivalence_probe(trace, first, 5)
        return r

    @pytest.mark.asyncio
    async def test_cache_implicated_is_enqueued_with_twin(self, tmp_path):
        r = await self._probe(tmp_path, "cached", "recomputed", "recomputed")
        [c] = r._confirm_queue
        assert (c.rid, c.twin_rid, c.twin_needs_cache_hit) == ("r0", "r0~pfx", True)
        assert c.source == "prefix_equiv"
        assert {e.request_id for e in c.replay_trace.events} == {"r0", "r0~pfx", "r0~pfx2"}
        assert [e.request_id for e in c.report_trace.events] == ["r0"]

    @pytest.mark.asyncio
    async def test_self_check_is_enqueued_as_recompute_pair(self, tmp_path):
        r = await self._probe(tmp_path, "x", "y", "z")
        [c] = r._confirm_queue
        assert (c.rid, c.twin_rid, c.twin_needs_cache_hit) == ("r0~pfx", "r0~pfx2", False)
        assert c.bugs[0].bug_class == BugClass.PROBE_SELF_CHECK


class TestStartupGuardRealLog:
    def test_explicit_backend_line_from_0271(self, tmp_path):
        """The exact lines of the first A6000 smoke job's BI server log."""
        log = tmp_path / "bi.log"
        log.write_text(
            "(EngineCore pid=1) INFO 09-28 23:04:10 [cuda.py:422] Using "
            "AttentionBackendEnum.FLASH_ATTN backend.\n"
            "(EngineCore pid=1) INFO 09-28 23:04:10 [flash_attn.py:789] Using "
            "FlashAttention version 2\n"
            '(EngineCore pid=1)   File "/usr/local/lib/python3.12/dist-packages/'
            'vllm/model_executor/layers/batch_invariant.py", line 141, in '
            "matmul_persistent\n")
        r = _runner(tmp_path, confirm_bi_log=str(log))
        r._check_confirm_ladder()
        guard = r._confirm_guard["backend"]
        assert (guard["name"], guard["fa_version"], guard["status"]) == ("FLASH_ATTN", 2, "ALLOWED")
        assert guard["resolved_from"] == "server log"


class TestStartupGuard:
    def test_off_is_a_no_op(self, tmp_path):
        _runner(tmp_path, confirm_ladder=False, confirm_bi_url="")._check_confirm_ladder()

    def test_needs_a_second_server(self, tmp_path):
        with pytest.raises(RuntimeError, match="confirm-bi-url"):
            _runner(tmp_path, confirm_bi_url="")._check_confirm_ladder()
        with pytest.raises(RuntimeError, match="second server"):
            _runner(tmp_path, confirm_bi_url=STOCK + "/")._check_confirm_ladder()

    def test_unproven_engagement_refuses(self, tmp_path):
        log = tmp_path / "bi.log"
        log.write_text("INFO Using FLASH_ATTN attention backend out of potential backends\n")
        r = _runner(tmp_path, confirm_bi_log=str(log))
        with pytest.raises(RuntimeError, match="engagement not proven"):
            r._check_confirm_ladder()
        assert (tmp_path / "findings" / "confirmation_guard.json").exists()

    def test_engaged_fa2_passes(self, tmp_path):
        log = tmp_path / "bi.log"
        log.write_text(
            "INFO Using FLASH_ATTN attention backend out of potential backends: [...]\n"
            "INFO Using FlashAttention version 2\n"
            "WARNING ... vllm/model_executor/layers/batch_invariant.py:141 ...\n")
        r = _runner(tmp_path, confirm_bi_log=str(log))
        r._check_confirm_ladder()
        guard = json.loads((tmp_path / "findings" / "confirmation_guard.json").read_text())
        assert guard["engagement"]["proven"] is True
        assert guard["backend"]["status"] == "ALLOWED"


class TestFunnel:
    def test_counts_by_verdict_and_priority(self, tmp_path, capsys):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "confirm_funnel", Path(__file__).resolve().parents[1] / "scripts" / "confirm_funnel.py")
        funnel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(funnel)

        run = tmp_path / "run1" / "findings"
        run.mkdir(parents=True)
        recs = [("BATCH_VARIANCE", "nontie"), ("BATCH_VARIANCE", "tie"),
                ("BI_BATCH_DEPENDENT", "nontie"), ("UNPROCESSED", "tie")]
        (run / "confirmation_log.jsonl").write_text(
            "".join(json.dumps({"verdict": v, "priority": p}) + "\n" for v, p in recs))
        (run / "confirmation_summary.json").write_text(json.dumps({"counts": {"cooled": 3}}))

        assert funnel.main(["--json", str(tmp_path)]) == 0
        [row] = json.loads(capsys.readouterr().out)
        c = row["counts"]
        assert c["all"] == 4 and c["filed"] == 1 and c["cooled"] == 3
        assert c["BATCH_VARIANCE|tie"] == 1 and c["BATCH_VARIANCE|nontie"] == 1

        assert funnel.main([str(tmp_path)]) == 0
        table = capsys.readouterr().out
        assert "BATCH_VARIANCE" in table and "2 (1/1)" in table
