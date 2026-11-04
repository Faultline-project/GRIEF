"""Request-surface operators (prompt_embeds, malformed constraints, cache
reset), the reset oracle, the non-finite-serialisation check, and the
executor's wire handling for both new event features."""

from __future__ import annotations

import asyncio
import base64
import io
import random

import pytest

from fuzzer.adapters.vllm_base import mutators as M
from fuzzer.adapters.vllm_base.adapter import VllmBaseAdapter
from fuzzer.core.oracles import Oracle
from fuzzer.core.types import BugClass, RequestResult, TraceResult
from fuzzer.executor import PythonExecutor, _prompt_embeds_b64
from fuzzer.trace import ActionType, Event, Trace


def _send(rid, offset_ms=0, **kw):
    base = dict(action=ActionType.SEND, offset_ms=offset_ms, request_id=rid,
                prompt_len=64, max_tokens=8, stream=False, temperature=0.0,
                prompt_family_id=f"fam-{rid}")
    base.update(kw)
    return Event(**base)


def _trace(n=4):
    return Trace(seed=1, events=[_send(f"r{i}", 100 * i) for i in range(n)])


# ── operators ────────────────────────────────────────────────────────────────

def test_inject_prompt_embeds_marks_completions_only():
    t = Trace(seed=1, events=[_send("a"), _send("b", messages=[{"role": "user", "content_len": 8}],
                                                prompt_len=None)])
    for s in range(20):
        out = M.inject_prompt_embeds(t, random.Random(s))
        assert out.events[0].prompt_embeds is True
        assert out.events[1].prompt_embeds is None
    assert t.events[0].prompt_embeds is None                      # input untouched


def test_malformed_constraint_sets_exactly_one():
    for s in range(50):
        ev = M.malformed_constraint(_trace(3), random.Random(s)).events
        set_ = [(e.request_id, k) for e in ev for k in
                ("response_format", "guided_json", "guided_regex", "guided_grammar", "guided_choice")
                if getattr(e, k) is not None]
        assert len(set_) == 1


def test_inject_cache_reset_appends_reset_then_copy():
    t = _trace(3)
    out = M.inject_cache_reset(t, random.Random(0))
    reset, again = out.events[-2], out.events[-1]
    assert reset.action == ActionType.RESET
    assert again.action == ActionType.SEND and again.request_id.endswith("_after_reset")
    assert again.offset_ms >= reset.offset_ms > max(e.offset_ms for e in t.events)
    src = next(e for e in t.events if again.request_id.startswith(e.request_id))
    assert (again.prompt_len, again.prompt_family_id) == (src.prompt_len, src.prompt_family_id)
    assert again.include_usage is True and again.cache_salt is None
    rt = Trace.from_json(out.to_json())
    assert rt.events[-2].action == ActionType.RESET


def test_new_operators_registered():
    names = {fn.__name__ for fn in VllmBaseAdapter().extra_mutations()}
    assert {"inject_prompt_embeds", "malformed_constraint", "inject_cache_reset"} <= names


# ── oracles ──────────────────────────────────────────────────────────────────

def _req(rid, sent_at, cached=None, error=None):
    r = RequestResult(request_id=rid, sent_at=sent_at, stream=False, prompt_len=64)
    r.completed_at = sent_at + 0.1
    r.cache_read_tokens = cached
    r.error = error
    return r


def _res(reqs, resets):
    return TraceResult(trace_seed=1, started_at=0.0, finished_at=10.0,
                       requests=reqs, resets=resets)


RESET_OK = {"success": True, "in_flight": [], "status": 200, "done_at": 5.0}


@pytest.mark.parametrize("cached,resets,fires", [
    (64, [RESET_OK], True),
    (0, [RESET_OK], False),
    (None, [RESET_OK], False),
    (64, [dict(RESET_OK, success=False)], False),
    (64, [dict(RESET_OK, in_flight=["r0"])], False),
    (64, [], False),
])
def test_reset_oracle(cached, resets, fires):
    trace = Trace(seed=1, events=[_send("r0"), _send("r0_after_reset", 6000)])
    res = _res([_req("r0", 1.0, cached=0), _req("r0_after_reset", 5.1, cached=cached)], resets)
    out = Oracle()._check_reset_honored(res, trace)
    assert bool(out) is fires
    if fires:
        assert out[0].bug_class == BugClass.CACHE_RESET_IGNORED


def test_reset_oracle_only_first_request_counts():
    trace = Trace(seed=1, events=[_send("a"), _send("b"), _send("c")])
    res = _res([_req("a", 1.0), _req("b", 5.1, cached=0), _req("c", 5.2, cached=64)], [RESET_OK])
    assert Oracle()._check_reset_honored(res, trace) == []


def test_reset_oracle_skips_salted():
    trace = Trace(seed=1, events=[_send("a"), _send("b", cache_salt="s")])
    res = _res([_req("a", 1.0), _req("b", 5.1, cached=64)], [RESET_OK])
    assert Oracle()._check_reset_honored(res, trace) == []


def test_nonfinite_serialisation_error_is_a_feature_failure():
    bad = _req("x", 1.0, error="HTTP 400: Out of range float values are not JSON compliant: nan")
    ok400 = _req("y", 1.0, error="HTTP 400: max_tokens too large")
    out = Oracle()._check_feature_failure(_res([bad, ok400], []))
    fails = [f for f in out if not f.passed]
    assert len(fails) == 1 and "x" in fails[0].detail
    assert fails[0].bug_class == BugClass.FEATURE_FAILURE


# ── executor wire format ────────────────────────────────────────────────────

def test_prompt_embeds_tensor_is_deterministic():
    torch = pytest.importorskip("torch")
    a = _prompt_embeds_b64("fam", 7, 16)
    assert a == _prompt_embeds_b64("fam", 7, 16)
    assert a != _prompt_embeds_b64("fam2", 7, 16)
    t = torch.load(io.BytesIO(base64.b64decode(a)), weights_only=True)
    assert tuple(t.shape) == (7, 16) and t.dtype == torch.bfloat16


class _FakeResp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body

    def json(self):
        return self._body


class _FakeClient:
    def __init__(self, status=200, body=None):
        self.status, self.body, self.posts = status, body, []

    async def post(self, path, **kw):
        self.posts.append((path, kw))
        return _FakeResp(self.status, self.body)


@pytest.mark.parametrize("status,body,success", [
    (200, {"success": True}, True), (200, None, True),
    (200, {"success": False}, False), (404, {"detail": "Not Found"}, False),
])
def test_executor_reset_record(status, body, success):
    ex = PythonExecutor()
    cl = _FakeClient(status, body)
    ev = Event(action=ActionType.RESET, offset_ms=5, request_id="reset5")
    rec = asyncio.run(ex._reset_prefix_cache(cl, ev, {}))
    assert cl.posts[0][0] == "/reset_prefix_cache"
    assert rec["success"] is success and rec["status"] == status and rec["in_flight"] == []


def test_executor_sends_embeds_only_when_configured():
    pytest.importorskip("torch")
    ex = PythonExecutor()
    ev = _send("e", prompt_embeds=True)
    captured = {}

    class _C:
        async def post(self, endpoint, json=None, **kw):
            captured["payload"] = json
            raise RuntimeError("stop here")

    r = RequestResult(request_id="e", sent_at=0.0, stream=False, prompt_len=64)
    asyncio.run(ex._send(_C(), ev, r))
    assert "prompt" in captured["payload"] and "prompt_embeds" not in captured["payload"]
    ex.prompt_embeds_dim = 32
    asyncio.run(ex._send(_C(), ev, r))
    assert "prompt_embeds" in captured["payload"] and "prompt" not in captured["payload"]


def test_inject_structured_output_one_valid_constraint_each():
    for s in range(50):
        out = M.inject_structured_output(_trace(4), random.Random(s))
        for e in out.events:
            set_ = [k for k in ("response_format", "guided_json", "guided_regex",
                                "guided_grammar", "guided_choice") if getattr(e, k) is not None]
            assert len(set_) <= 1
        assert any(getattr(e, k) is not None for e in out.events
                   for k in ("response_format", "guided_json", "guided_regex", "guided_choice"))


def _choice_run(text, finish):
    ev = _send("c", guided_choice=["yes", "no"], max_tokens=32)
    r = _req("c", 1.0)
    r.output_text, r.finish_reason = text, finish
    return Oracle()._check_schema_violation(_res([r], []), Trace(seed=1, events=[ev]))


@pytest.mark.parametrize("text,finish,fires", [
    ("Yes\n\nOkay, let's see", "length", True),     # unconstrained ramble
    ("ye", "length", False),                         # a legal prefix
    ("yes", "length", False),
    ("", "length", False),
])
def test_choice_prefix_check_on_truncation(text, finish, fires):
    out = [f for f in _choice_run(text, finish) if not f.passed]
    assert bool(out) is fires
    if fires:
        assert out[0].probe_metadata["failure"] == "choice_prefix"
