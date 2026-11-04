"""Tests for PythonExecutor — no live server required."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

import time
from fuzzer.executor import PythonExecutor, RequestResult, _build_prompt, error_signature
from fuzzer.trace import Trace, Event, ActionType


def make_simple_trace() -> Trace:
    return Trace(seed=1, events=[
        Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
              prompt_len=16, max_tokens=8, stream=True)
    ])


class TestBuildPrompt:
    def _event(self, prompt_len: int, prefix_len: int = 0,
               rid: str = "r1") -> Event:
        return Event(action=ActionType.SEND, offset_ms=0,
                     request_id=rid, prompt_len=prompt_len,
                     max_tokens=8, prefix_len=prefix_len)

    def test_total_token_count(self):
        ev = self._event(prompt_len=10, prefix_len=0)
        tokens = _build_prompt(ev).split()
        assert len(tokens) == 10

    def test_unique_suffix_tokens(self):
        ev = self._event(prompt_len=8, prefix_len=4)
        tokens = _build_prompt(ev).split()
        # suffix tokens start with "u<hash>_"
        for t in tokens[4:]:
            assert t.startswith("u")

    def test_full_prefix_capped_to_total_minus_one(self):
        # prefix_len = prompt_len would leave no unique tokens — clamped internally
        ev = self._event(prompt_len=4, prefix_len=10)  # prefix_len > prompt_len
        tokens = _build_prompt(ev).split()
        assert len(tokens) == 4

    def test_different_request_ids_different_unique_tokens(self):
        ev1 = self._event(prompt_len=6, prefix_len=0, rid="r1")
        ev2 = self._event(prompt_len=6, prefix_len=0, rid="r2")
        assert _build_prompt(ev1) != _build_prompt(ev2)

    def test_same_prefix_len_produces_shared_prefix(self):
        ev1 = self._event(prompt_len=8, prefix_len=4, rid="r1")
        ev2 = self._event(prompt_len=8, prefix_len=4, rid="r2")
        tokens1 = _build_prompt(ev1).split()
        tokens2 = _build_prompt(ev2).split()
        assert tokens1[:4] == tokens2[:4]

    def test_zero_prompt_len_uses_fallback(self):
        ev = Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                   prompt_len=0, max_tokens=8)
        prompt = _build_prompt(ev)
        assert len(prompt.split()) == 16  # falls back to 16


class TestErrorSignature:
    def _req(self, error=None):
        return RequestResult(request_id="r1", sent_at=0.0, error=error)

    def test_no_errors_returns_none(self):
        assert error_signature([self._req(), self._req()]) is None

    def test_single_error_returns_hash(self):
        sig = error_signature([self._req("ConnectionError: refused")])
        assert sig is not None
        assert len(sig) == 12

    def test_same_error_same_hash(self):
        a = error_signature([self._req("ValueError: bad input")])
        b = error_signature([self._req("ValueError: bad input")])
        assert a == b

    def test_different_errors_different_hashes(self):
        a = error_signature([self._req("TimeoutError")])
        b = error_signature([self._req("ConnectionRefused")])
        assert a != b

    def test_address_normalization(self):
        a = error_signature([self._req("Error at 0xdeadbeef in module")])
        b = error_signature([self._req("Error at 0xcafebabe in module")])
        assert a == b

    def test_long_number_normalization(self):
        a = error_signature([self._req("pid=1234567 failed")])
        b = error_signature([self._req("pid=9999999 failed")])
        assert a == b

    def test_empty_list_returns_none(self):
        assert error_signature([]) is None


class TestPythonExecutorTimeout:
    @pytest.mark.asyncio
    async def test_timeout_cancels_hanging_tasks(self):
        """If gather would block forever, TimeoutError fires and run() returns."""
        async def hanging_send(*args, **kwargs):
            await asyncio.sleep(9999)

        ex = PythonExecutor(trace_timeout_s=0.05)   # 50 ms

        with patch.object(ex, '_send', side_effect=hanging_send):
            import httpx
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)

            with patch('httpx.AsyncClient', return_value=mock_client):
                trace = make_simple_trace()
                result = await ex.run(trace, "http://fake")
                assert result is not None




# ---------------------------------------------------------------------------
# Layer-12 optional body fields (cache_salt + structured output)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 1}}


class _CapturingClient:
    """Stands in for httpx.AsyncClient on the non-streaming path."""

    def __init__(self):
        self.calls = []

    async def post(self, endpoint, json=None):
        self.calls.append((endpoint, json))
        return _FakeResponse()


class TestOptionalBodyFields:
    async def _payload(self, **ev_kwargs) -> dict:
        kw = dict(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=16, max_tokens=8, stream=False)
        kw.update(ev_kwargs)
        event = Event(**kw)
        client = _CapturingClient()
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(client, event, result)
        assert len(client.calls) == 1
        return client.calls[0][1]

    @pytest.mark.asyncio
    async def test_absent_by_default(self):
        """Every new field is opt-in: a server build that rejects one never
        sees it unless a seed or a probe put it on the Event."""
        payload = await self._payload()
        for key in ("cache_salt", "response_format", "structured_outputs",
                    "guided_json", "guided_regex", "guided_grammar",
                    "guided_choice"):
            assert key not in payload

    @pytest.mark.asyncio
    async def test_cache_salt_on_the_wire(self):
        payload = await self._payload(cache_salt="grief-r1-0000beef")
        assert payload["cache_salt"] == "grief-r1-0000beef"

    @pytest.mark.asyncio
    async def test_response_format_on_the_wire(self):
        rf = {"type": "json_schema",
              "json_schema": {"name": "t", "schema": {"type": "object"}}}
        payload = await self._payload(response_format=rf)
        assert payload["response_format"] == rf

    @pytest.mark.parametrize("attr,wire,value", [
        ("guided_json",    "json",    {"type": "object"}),
        ("guided_regex",   "regex",   "[0-9]+"),
        ("guided_grammar", "grammar", 'root ::= "a"'),
        ("guided_choice",  "choice",  ["yes", "no"]),
    ])
    @pytest.mark.asyncio
    async def test_guided_fields_nest_under_structured_outputs(
            self, attr, wire, value):
        """``guided_*`` are NOT top-level request fields on either vLLM build we
        run (0.19.0 or 0.27.1); sent top-level they land in the request model's
        ignored-extras bucket and the mutation is a silent no-op.  The real wire
        location is ``structured_outputs.{json,regex,grammar,choice}`` --
        ``StructuredOutputsParams``, ``vllm/sampling_params.py:72-77``."""
        payload = await self._payload(**{attr: value})
        assert attr not in payload
        assert payload["structured_outputs"] == {wire: value}

    @pytest.mark.asyncio
    async def test_chat_endpoint_carries_them_too(self):
        payload = await self._payload(
            prompt_len=None, messages=[{"role": "user", "content_len": 8}],
            cache_salt="s", response_format={"type": "json_object"})
        assert payload["cache_salt"] == "s"
        assert payload["response_format"] == {"type": "json_object"}
        assert "messages" in payload


class TestMaskRestrictorBodyFields:
    """The four extra ``-inf`` writers ride the same opt-in path as the
    layer-12 fields: omitted unless the Event carries them, so a server build
    that rejects one only ever 4xxs the single request that opted in."""

    _FIELDS = {
        "allowed_token_ids": [1, 2, 3],
        "bad_words":         ["{", "}"],
        "logit_bias":        {"1": -100.0},
        "stop_token_ids":    [1],
    }

    async def _send(self, **ev_kwargs):
        kw = dict(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=16, max_tokens=8, stream=False)
        kw.update(ev_kwargs)
        event = Event(**kw)
        client = _CapturingClient()
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(client, event, result)
        return client.calls[0][1], result

    @pytest.mark.asyncio
    async def test_absent_by_default(self):
        payload, result = await self._send()
        for key in self._FIELDS:
            assert key not in payload
        assert result.mask_restrictors is None

    @pytest.mark.asyncio
    async def test_all_four_on_the_wire(self):
        payload, _ = await self._send(**self._FIELDS)
        for key, val in self._FIELDS.items():
            assert payload[key] == val

    @pytest.mark.asyncio
    async def test_chat_endpoint_carries_them_too(self):
        payload, _ = await self._send(
            prompt_len=None, messages=[{"role": "user", "content_len": 8}],
            **self._FIELDS)
        assert "messages" in payload
        for key, val in self._FIELDS.items():
            assert payload[key] == val

    @pytest.mark.asyncio
    async def test_restrictors_are_snapshotted_onto_the_result(self):
        """The oracle runs off ``TraceResult`` alone, so what the request
        engaged has to travel on the RequestResult."""
        _, result = await self._send(
            min_tokens=32, response_format={"type": "json_object"},
            **self._FIELDS)
        assert result.mask_restrictors == {
            "structured_output": "json_object",
            "min_tokens":        32,
            "allowed_token_ids": [1, 2, 3],
            "bad_words":         ["{", "}"],
            "logit_bias":        {"1": -100.0},
            "stop_token_ids":    [1],
        }

    @pytest.mark.asyncio
    async def test_falsy_restrictors_do_not_count(self):
        """``min_tokens=0`` never registers the processor
        (``MinTokensLogitsProcessor.add_request`` returns None), and an empty
        list is a field the server would reject or ignore — neither is part of
        a composition."""
        _, result = await self._send(min_tokens=0, allowed_token_ids=[],
                                     bad_words=[], stop_token_ids=[])
        assert result.mask_restrictors is None


class TestSchedulerAndAmplificationBodyFields:
    """``priority`` / ``prompt_logprobs`` / ``seed`` / ``truncate_prompt_tokens``
    on the wire.  Same opt-in contract as every optional field before them."""

    _FIELDS = {
        "priority":               4,
        "prompt_logprobs":        5,
        "seed":                   20260826,
        "truncate_prompt_tokens": 128,
    }

    async def _send(self, **ev_kwargs):
        kw = dict(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=16, max_tokens=8, stream=False)
        kw.update(ev_kwargs)
        event = Event(**kw)
        client = _CapturingClient()
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(client, event, result)
        return client.calls[0][1], result

    @pytest.mark.asyncio
    async def test_absent_by_default(self):
        payload, _ = await self._send()
        for key in self._FIELDS:
            assert key not in payload

    @pytest.mark.asyncio
    async def test_all_four_on_the_wire(self):
        payload, _ = await self._send(**self._FIELDS)
        for key, val in self._FIELDS.items():
            assert payload[key] == val

    @pytest.mark.asyncio
    async def test_chat_endpoint_carries_them_too(self):
        payload, _ = await self._send(
            prompt_len=None, messages=[{"role": "user", "content_len": 8}],
            **self._FIELDS)
        assert "messages" in payload
        for key, val in self._FIELDS.items():
            assert payload[key] == val

    @pytest.mark.asyncio
    async def test_priority_zero_reaches_the_wire(self):
        """0 is both the default and the HIGHEST priority.  The pass-through
        loop filters on ``is not None``, not on truthiness -- if it filtered on
        truthiness, the single most important priority value would be the one
        that could never be sent, and a trace's top-priority requests would go
        out indistinguishable from its unprioritised ones."""
        payload, _ = await self._send(priority=0)
        assert payload["priority"] == 0

    @pytest.mark.asyncio
    async def test_negative_priority_reaches_the_wire(self):
        """vLLM bounds priority only by int64 and lower means higher, so a
        negative priority is the legitimate way to outrank the default 0."""
        payload, _ = await self._send(priority=-5)
        assert payload["priority"] == -5

    @pytest.mark.asyncio
    async def test_n_comes_from_n_completions_and_is_written_once(self):
        """There is exactly one Event attribute behind the body's ``n``."""
        payload, _ = await self._send(n_completions=3)
        assert payload["n"] == 3

    @pytest.mark.asyncio
    async def test_best_of_is_never_sent(self):
        """vLLM's V1 engine removed ``best_of`` from both request models and
        from SamplingParams, and the request models are ``extra="allow"`` --
        so sending it would be accepted, silently dropped, and recorded in the
        corpus as a parameter that had an effect.  It is not carried at all."""
        payload, _ = await self._send(**self._FIELDS)
        assert "best_of" not in payload


class TestCachedTokenUsageParsing:
    """vLLM reports prefix-cache hits as usage.prompt_tokens_details.
    cached_tokens, not the Anthropic-style usage.cache_read_input_tokens the
    parser originally looked for; the prefix-equivalence probe's
    "did the salt bypass the cache" guard depends on this."""

    class _Resp:
        def __init__(self, usage):
            self._usage = usage

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"index": 0, "text": "ok",
                                 "finish_reason": "stop"}],
                    "usage": self._usage}

    class _Client:
        def __init__(self, usage):
            self._usage = usage

        async def post(self, endpoint, json=None):
            return TestCachedTokenUsageParsing._Resp(self._usage)

    async def _cache_read(self, usage):
        event = Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                      prompt_len=16, max_tokens=8, stream=False)
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(self._Client(usage), event, result)
        return result.cache_read_tokens

    @pytest.mark.asyncio
    async def test_prompt_tokens_details_is_read(self):
        assert await self._cache_read(
            {"completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 128}}
        ) == 128

    @pytest.mark.asyncio
    async def test_explicit_field_still_wins(self):
        assert await self._cache_read(
            {"completion_tokens": 1, "cache_read_input_tokens": 64,
             "prompt_tokens_details": {"cached_tokens": 128}}
        ) == 64

    @pytest.mark.asyncio
    async def test_absent_stays_none(self):
        assert await self._cache_read({"completion_tokens": 1}) is None
        assert await self._cache_read(
            {"completion_tokens": 1, "prompt_tokens_details": None}) is None


class TestStreamingUsageParsing:
    """Streamed responses carry NO usage block unless the request asks.

    ``stream_options: {"include_usage": true}`` is what asks, and the block then
    arrives on its own final SSE chunk whose ``choices`` list is EMPTY.  Both
    halves are silent-failure classes: without the request field the server
    sends nothing, and without reading the empty-choices chunk the parser drops
    what it did send.  Either way ``cache_read_tokens`` is None on every
    streamed request and ``Runner._prefix_equivalence_probe``'s vacuity control
    never runs -- which is exactly what happened for all 71 findings of the
    novelty_ab campaign.
    """

    class _Resp:
        def __init__(self, lines):
            self._lines = lines

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _Ctx:
        def __init__(self, outer, lines):
            self._outer, self._lines = outer, lines

        async def __aenter__(self):
            return TestStreamingUsageParsing._Resp(self._lines)

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, lines):
            self._lines, self.calls = lines, []

        def stream(self, method, endpoint, json=None):
            self.calls.append((endpoint, json))
            return TestStreamingUsageParsing._Ctx(self, self._lines)

    async def _run(self, chunks, **ev_kwargs):
        import json as _json
        lines = ["data: " + _json.dumps(c) for c in chunks] + ["data: [DONE]"]
        kw = dict(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=16, max_tokens=8, stream=True)
        kw.update(ev_kwargs)
        client = self._Client(lines)
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(client, Event(**kw), result)
        return result, client.calls[0][1]

    _BODY = [{"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
             {"choices": [],
              "usage": {"prompt_tokens": 300, "completion_tokens": 1,
                        "prompt_tokens_details": {"cached_tokens": 256}}}]

    @pytest.mark.asyncio
    async def test_usage_chunk_with_empty_choices_is_read(self):
        result, _ = await self._run(self._BODY, include_usage=True)
        assert result.cache_read_tokens == 256
        assert result.prompt_tokens_reported == 300
        # the empty-choices chunk must not damage the assembled text
        assert result.output_text == "ok"

    @pytest.mark.asyncio
    async def test_stream_options_is_requested_when_asked(self):
        _, payload = await self._run(self._BODY, include_usage=True)
        assert payload["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_stream_options_absent_by_default(self):
        """Opt-in like every other extension field: an ordinary trace must not
        change shape on the wire just because the probe exists."""
        _, payload = await self._run(self._BODY)
        assert "stream_options" not in payload

    @pytest.mark.asyncio
    async def test_stream_options_never_sent_on_a_non_streaming_request(self):
        """vLLM 400s ``stream_options`` when ``stream`` is false
        (``entrypoints/openai/completion/protocol.py:518-522`` and
        ``chat_completion/protocol.py:769-773``, 0.27.1).  A 400 would empty one
        arm of the probe and turn it into a spurious divergence."""
        client = _CapturingClient()
        event = Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                      prompt_len=16, max_tokens=8, stream=False,
                      include_usage=True)
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(client, event, result)
        assert "stream_options" not in client.calls[0][1]


# ---------------------------------------------------------------------------
# stop_reason capture (Fix 1)
# ---------------------------------------------------------------------------

class TestStopReasonCapture:
    """`stop_reason` is vLLM's per-choice sibling of `finish_reason`.

    Verified in the vLLM 0.27.1 container at
    `entrypoints/openai/completion/protocol.py:603` (non-streaming) and `:651`
    (streaming), `entrypoints/openai/chat_completion/protocol.py:110` and
    `:160` — declared `int | str | None` on all four choice classes and
    populated straight from `output.stop_reason` in the serving layer
    (`completion/serving.py:399,414,587`,
    `chat_completion/serving.py:724,1026`).  Same field in 0.19.0 at
    `completion/protocol.py:460,494` / `chat_completion/protocol.py:91,120`.

    Without it the schema-violation oracle cannot tell a grammar that reached
    an accepting state from one a stop string truncated.
    """

    class _Resp:
        def __init__(self, choices):
            self._choices = choices

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": self._choices, "usage": {"completion_tokens": 1}}

    class _Client:
        def __init__(self, choices):
            self._choices = choices

        async def post(self, endpoint, json=None):
            return TestStopReasonCapture._Resp(self._choices)

    async def _run(self, choices, stream=False):
        event = Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                      prompt_len=16, max_tokens=8, stream=stream)
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(self._Client(choices), event, result)
        return result

    async def test_stop_string_is_captured(self):
        r = await self._run([{"index": 0, "text": "{\n\n", "finish_reason": "stop",
                              "stop_reason": "\n\n"}])
        assert r.finish_reason == "stop"
        assert r.stop_reason == "\n\n"
        assert r.stop_reason_reported is True

    async def test_stop_token_id_is_captured_as_int(self):
        r = await self._run([{"index": 0, "text": "x", "finish_reason": "stop",
                              "stop_reason": 151643}])
        assert r.stop_reason == 151643
        assert r.stop_reason_reported is True

    async def test_eos_reports_null(self):
        """`v1/core/sched/utils.py:104-106` finishes on the primary EOS without
        ever assigning stop_reason; `:112-117` does the same for the length cap."""
        r = await self._run([{"index": 0, "text": "x", "finish_reason": "stop",
                              "stop_reason": None}])
        assert r.stop_reason is None
        assert r.stop_reason_reported is True

    async def test_absent_field_is_distinguished_from_null(self):
        """A backend that never emits the key must not read as 'EOS'."""
        r = await self._run([{"index": 0, "text": "x", "finish_reason": "stop"}])
        assert r.stop_reason is None
        assert r.stop_reason_reported is False

    async def test_non_null_from_any_choice_wins(self):
        """output_text is \\x1f-joined for n>1, so any truncated sequence taints it."""
        r = await self._run([
            {"index": 0, "text": "a", "finish_reason": "stop", "stop_reason": None},
            {"index": 1, "text": "b", "finish_reason": "stop", "stop_reason": "\n\n"},
        ])
        assert r.stop_reason == "\n\n"

    # -- streaming ---------------------------------------------------------

    class _StreamResp:
        def __init__(self, lines):
            self._lines = lines

        async def aiter_lines(self):
            for line in self._lines:
                yield line

    class _StreamCtx:
        def __init__(self, lines):
            self._lines = lines

        async def __aenter__(self):
            return TestStopReasonCapture._StreamResp(self._lines)

        async def __aexit__(self, *exc):
            return False

    class _StreamClient:
        def __init__(self, lines):
            self._lines = lines

        def stream(self, method, endpoint, json=None):
            return TestStopReasonCapture._StreamCtx(self._lines)

    async def _run_stream(self, chunks):
        import json as _json
        lines = ["data: " + _json.dumps(c) for c in chunks] + ["data: [DONE]"]
        event = Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                      prompt_len=16, max_tokens=8, stream=True)
        result = RequestResult(request_id="r1", sent_at=time.time())
        await PythonExecutor()._send(self._StreamClient(lines), event, result)
        return result

    async def test_streaming_takes_the_final_chunks_stop_reason(self):
        """`v1/engine/output_processor.py:422` emits
        `stop_reason if finished else None`, so every non-final SSE chunk
        carries the key with a null.  An early null must not mask the real
        value on the last chunk."""
        r = await self._run_stream([
            {"choices": [{"index": 0, "text": "{",
                          "finish_reason": None, "stop_reason": None}]},
            {"choices": [{"index": 0, "text": "\n\n",
                          "finish_reason": None, "stop_reason": None}]},
            {"choices": [{"index": 0, "text": "",
                          "finish_reason": "stop", "stop_reason": "\n\n"}]},
        ])
        assert r.finish_reason == "stop"
        assert r.stop_reason == "\n\n"
        assert r.stop_reason_reported is True

    async def test_streaming_all_null_reads_as_eos(self):
        r = await self._run_stream([
            {"choices": [{"index": 0, "text": "x",
                          "finish_reason": None, "stop_reason": None}]},
            {"choices": [{"index": 0, "text": "",
                          "finish_reason": "stop", "stop_reason": None}]},
        ])
        assert r.stop_reason is None
        assert r.stop_reason_reported is True

    async def test_streaming_absent_key_is_flagged(self):
        r = await self._run_stream([
            {"choices": [{"index": 0, "text": "x", "finish_reason": None}]},
            {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
        ])
        assert r.stop_reason_reported is False
