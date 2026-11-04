"""
Executor: takes a Trace and runs it against a live inference server.

Two implementations:
  - PythonExecutor  asyncio + httpx, good for development and moderate load
  - RustExecutor    delegates to the compiled Rust runner binary for high
                    concurrency (>100 simultaneous clients)

Both return a TraceResult with per-request outcomes.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import time
from typing import Optional

from typing import Optional as _Optional

from .trace import Trace, Event, ActionType
from .core.types import (
    RequestResult,
    TraceResult,
    error_signature,
    mask_restrictors_of,
)
from .core.executor import Executor
from .core.adapter import TargetAdapter
from .adapters.vllm_base.adapter import VllmBaseAdapter, _build_prompt, _build_chat_messages

# Default adapter used when none is explicitly provided — preserves backward compat.
_DEFAULT_ADAPTER = VllmBaseAdapter()


#: ``Event.guided_*`` name -> the key it takes inside the request body's
#: ``structured_outputs`` object.
#:
#: GROUND TRUTH (read out of the containers on 2026-08-26, not from memory):
#: ``guided_json`` / ``guided_regex`` / ``guided_grammar`` / ``guided_choice``
#: are NOT top-level request fields on either build we run.  Both
#: ``vllm/entrypoints/openai/completion/protocol.py`` and
#: ``.../chat_completion/protocol.py`` declare exactly one structured-output
#: field, ``structured_outputs: StructuredOutputsParams | None``
#: (0.27.1: completion ``:128``, chat ``:375``; 0.19.0: completion ``:103``,
#: chat ``:281``), and ``StructuredOutputsParams``
#: (``vllm/sampling_params.py:72``) names its constraints ``json`` ``:74``,
#: ``regex`` ``:75``, ``choice`` ``:76``, ``grammar`` ``:77``.  The dataclass is
#: byte-identical between 0.19.0 and 0.27.1, so one mapping serves both.  Sent
#: top-level (as GRIEF did until now) the keys land in the request model's
#: ignored-extras bucket and every guided_* mutation is a silent no-op.
#:
#: Only ONE constraint may be set per request: ``StructuredOutputsParams``'s
#: ``__post_init__`` rejects a count != 1, and ``check_structured_outputs_count``
#: (``completion/protocol.py:430``) rejects >1 of json/regex/choice with a 400.
#: An Event that sets two therefore now 4xxs that single request instead of
#: silently doing nothing — which is the correct, visible behaviour.
_GUIDED_TO_STRUCTURED_OUTPUTS = (
    ("guided_json",    "json"),
    ("guided_regex",   "regex"),
    ("guided_grammar", "grammar"),
    ("guided_choice",  "choice"),
)


def _structured_outputs_of(event: Event) -> Optional[dict]:
    """The ``structured_outputs`` request-body object for *event*, or None.

    None (the common case) means the key is omitted from the body entirely, so
    a build that does not know the field never sees it.
    """
    so = {}
    for _attr, _wire in _GUIDED_TO_STRUCTURED_OUTPUTS:
        _val = getattr(event, _attr, None)
        if _val is not None:
            so[_wire] = _val
    return so or None


#: Upper bound on a prompt_embeds tensor's rows, so one request stays a few MB.
PROMPT_EMBEDS_MAX_ROWS = 2048


@functools.lru_cache(maxsize=64)
def _prompt_embeds_b64(key: str, rows: int, dim: int) -> str:
    """Base64 ``torch.save`` of a (rows, dim) bf16 tensor, a pure function of
    (key, rows, dim): a replay of the same request sends the same bytes."""
    import base64
    import io
    import torch
    seed = int.from_bytes(hashlib.sha256(f"{key}|{rows}|{dim}".encode()).digest()[:8], "big")
    gen = torch.Generator().manual_seed(seed)
    t = (torch.randn(rows, dim, generator=gen) * 0.02).to(torch.bfloat16)
    buf = io.BytesIO()
    torch.save(t, buf)
    return base64.b64encode(buf.getvalue()).decode()


def _apply_usage_token_counts(result, usage: Optional[dict]) -> None:
    """Read the prompt/cache token counts out of one ``usage`` block.

    Shared by the streaming and non-streaming paths so both report
    ``cache_read_tokens`` identically.  In streaming the block arrives on a
    *separate final SSE chunk* whose ``choices`` list is empty, and only when
    the request asked for it via ``stream_options.include_usage`` (see
    ``Event.include_usage``); in non-streaming it is always on the body.

    ``cache_read_tokens`` has two spellings.  ``usage.cache_read_input_tokens``
    is the Anthropic-style name some servers use; vLLM's own is
    ``usage.prompt_tokens_details.cached_tokens``
    (``entrypoints/openai/engine/protocol.py:105-106``, vLLM 0.27.1), populated
    only when the server was launched with ``--enable-prompt-tokens-details``
    (``entrypoints/openai/cli_args.py:132``; gated at
    ``completion/serving.py:452,610`` and ``chat_completion/serving.py:88-95``).
    Without that flag the number is still tracked by the engine
    (``v1/engine/output_processor.py:644-645``) but never serialised, so the
    field stays None and ``_prefix_equivalence_probe`` cannot verify that its
    ``cache_salt`` bypassed the cache.
    """
    if not isinstance(usage, dict):
        return
    pt = usage.get("prompt_tokens")
    if pt is not None:
        result.prompt_tokens_reported = int(pt)
    crt = usage.get("cache_read_input_tokens")
    if crt is None:
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            crt = details.get("cached_tokens")
    if crt is not None:
        result.cache_read_tokens = int(crt)
    cct = usage.get("cache_creation_input_tokens")
    if cct is None:
        details = usage.get("prompt_tokens_details") or {}
        if isinstance(details, dict):
            # vLLM 0.27.1 sibling of cached_tokens on the same sub-object
            # (``engine/protocol.py:107``); only the chat endpoint fills it.
            cct = details.get("created_cache_tokens")
    if cct is not None:
        result.cache_creation_tokens = int(cct)


# ---------------------------------------------------------------------------
# Python executor (asyncio + httpx)
# ---------------------------------------------------------------------------

class PythonExecutor(Executor):
    """
    Development executor. Suitable for up to ~100 concurrent requests.
    Replace with RustExecutor for higher-concurrency stress tests.

    Pass a TargetAdapter instance to delegate payload construction to the
    adapter (e.g. VllmLoRAAdapter injects the ``model`` field for LoRA).
    When omitted, VllmBaseAdapter is used (backward-compatible default).
    """

    def __init__(self,
                 request_timeout_s: float = 60.0,
                 trace_timeout_s:   float = 90.0,
                 adapter: _Optional[TargetAdapter] = None):
        self.model_name = ""
        self.request_timeout_s = request_timeout_s
        self.trace_timeout_s   = trace_timeout_s
        self.adapter: TargetAdapter = adapter if adapter is not None else _DEFAULT_ADAPTER
        # Hidden size for Event.prompt_embeds; None = send those requests as text.
        self.prompt_embeds_dim: Optional[int] = None

    async def run(self, trace: Trace, base_url: str) -> TraceResult:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("pip install httpx")

        results: dict[str, RequestResult] = {}
        tasks:   dict[str, asyncio.Task]  = {}
        resets:  list[dict]               = []
        started_at = time.time()

        async with httpx.AsyncClient(base_url=base_url,
                                     timeout=self.request_timeout_s) as client:
            trace_start = time.time()
            try:
                for event in sorted(trace.events, key=lambda e: e.offset_ms):
                    # Sleep until this event's absolute offset from trace start.
                    # Previously used asyncio.sleep(offset) each iteration which
                    # accumulated delays — event at 400ms would fire at 600ms+.
                    delay = (trace_start + event.offset_ms / 1000) - time.time()
                    if delay > 0:
                        await asyncio.sleep(delay)

                    if event.action == ActionType.SEND:
                        n = event.n_completions or 1
                        # For chat requests, prompt_len comes from message
                        # content_len (event.prompt_len is often None).
                        if event.messages and not event.prompt_len:
                            _prompt_len = sum(
                                m.get("content_len", 0) for m in event.messages
                            )
                        else:
                            _prompt_len = event.prompt_len or 0
                        result = RequestResult(request_id=event.request_id,
                                               sent_at=time.time(),
                                               stream=event.stream,
                                               prompt_len=_prompt_len,
                                               max_tokens_requested=(event.max_tokens or 64) * n,
                                               prefix_len=event.prefix_len,
                                               adapter_name=event.adapter_name,
                                               prompt_family_id=getattr(event, "prompt_family_id", None))
                        results[event.request_id] = result
                        tasks[event.request_id] = asyncio.create_task(
                            self._send(client, event, result)
                        )

                    elif event.action in (ActionType.CANCEL, ActionType.DISCONNECT):
                        rid = event.request_id
                        if rid in tasks and not tasks[rid].done():
                            tasks[rid].cancel()
                        if rid in results:
                            results[rid].cancelled_at = time.time()

                    elif event.action == ActionType.WAIT:
                        await asyncio.sleep((event.duration_ms or 0) / 1000)

                    elif event.action == ActionType.RESET:
                        resets.append(await self._reset_prefix_cache(client, event, tasks))

                if tasks:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks.values(), return_exceptions=True),
                            timeout=self.trace_timeout_s,
                        )
                    except asyncio.TimeoutError:
                        for t in tasks.values():
                            if not t.done():
                                t.cancel()
                        await asyncio.gather(*tasks.values(), return_exceptions=True)
            finally:
                for task in tasks.values():
                    if not task.done():
                        task.cancel()
                if tasks:
                    await asyncio.gather(*tasks.values(), return_exceptions=True)

        return TraceResult(
            trace_seed=trace.seed,
            started_at=started_at,
            finished_at=time.time(),
            requests=list(results.values()),
            resets=resets,
        )

    async def _reset_prefix_cache(self, client, event: Event, tasks: dict) -> dict:
        """POST /reset_prefix_cache inline (the trace waits for the answer)."""
        rec = {"offset_ms": event.offset_ms, "sent_at": time.time(),
               "in_flight": sorted(rid for rid, t in tasks.items() if not t.done()),
               "status": None, "success": False}
        try:
            resp = await client.post("/reset_prefix_cache")
            rec["status"] = resp.status_code
            try:
                body = resp.json()
            except Exception:
                body = None
            # 0.27.1 answers {"success": true}; older builds an empty 200.
            rec["success"] = resp.status_code == 200 and not (
                isinstance(body, dict) and body.get("success") is False)
        except Exception as exc:                       # transport failure
            rec["error"] = str(exc)[:200]
        rec["done_at"] = time.time()
        return rec

    async def _send(self, client, event: Event, result: RequestResult) -> None:
        import json as _json

        result.temperature = event.temperature if event.temperature is not None else 0.0

        # Delegate endpoint + base payload construction to the adapter.
        # VllmBaseAdapter produces the same result as the old inline code;
        # VllmLoRAAdapter additionally injects the "model" field for LoRA.
        endpoint, payload = self.adapter.build_payload(event)
        if self.model_name:
            payload.setdefault("model", self.model_name)
        if (getattr(event, "prompt_embeds", None) and self.prompt_embeds_dim
                and not endpoint.endswith("/chat/completions")):
            rows = max(1, min(event.prompt_len or 1, PROMPT_EMBEDS_MAX_ROWS))
            key = getattr(event, "prompt_family_id", None) or event.request_id
            payload.pop("prompt", None)
            payload["prompt_embeds"] = _prompt_embeds_b64(key, rows, self.prompt_embeds_dim)
        # build_payload sets temperature from event; sync it to result.temperature
        # in case the adapter used a different default.
        payload["temperature"] = result.temperature

        for _key, _val in [
            ("top_p",              event.top_p),
            ("top_k",              event.top_k),
            ("min_tokens",         event.min_tokens),
            ("n",                  event.n_completions),
            ("repetition_penalty", event.repetition_penalty),
            ("stop",               event.stop),
        ]:
            if _val is not None:
                payload[_key] = _val
        # logprobs is endpoint-specific: /v1/completions takes an integer
        # (`logprobs: N`), but /v1/chat/completions requires `logprobs: true`
        # plus `top_logprobs: N` — sending an integer there is a 400 Bad Request,
        # which empties the response and makes the determinism probe (re-run with
        # logprobs=5) compare text-vs-empty and flag spurious kv_corruption.
        if event.logprobs is not None:
            if endpoint.endswith("/chat/completions"):
                payload["logprobs"] = True
                payload["top_logprobs"] = event.logprobs
            else:
                payload["logprobs"] = event.logprobs
        # Optional extra body fields (vLLM OpenAI-server extensions).  Every one
        # of these is omitted unless the Event explicitly carries it, so a server
        # build that rejects the field never sees it on an ordinary trace: only a
        # seed/probe that opted in can trip a 4xx, and that 4xx lands on the one
        # request (``result.error``), not on the whole trace.  The oracles that
        # consume these fields all skip requests with ``error`` set.
        #   cache_salt   — prefix-cache namespace (W3 relational probe)
        #   response_format / structured_outputs — structured-output constraints
        #       (W2 oracle).  The Event still carries the four ``guided_*``
        #       fields, but they go on the wire nested under ``structured_outputs``
        #       — see ``_structured_outputs_of``.
        #   allowed_token_ids / bad_words / logit_bias / stop_token_ids —
        #       the other independent writers of -inf into the logits row
        #       (mask-composition surface; Oracle._check_degenerate_mask)
        for _key, _val in [
            ("cache_salt",        getattr(event, "cache_salt", None)),
            ("response_format",   getattr(event, "response_format", None)),
            ("structured_outputs", _structured_outputs_of(event)),
            ("allowed_token_ids", getattr(event, "allowed_token_ids", None)),
            ("bad_words",         getattr(event, "bad_words", None)),
            ("logit_bias",        getattr(event, "logit_bias", None)),
            ("stop_token_ids",    getattr(event, "stop_token_ids", None)),
            # Scheduler + amplification parameters.  `priority` only *does*
            # anything when the server ran with --scheduling-policy priority;
            # everywhere else it is accepted and ignored, which is why it is
            # safe to carry unconditionally.  NOTE: lower value = higher
            # priority.  `n` is NOT listed here — it is already written above
            # from event.n_completions, and `best_of` is not carried at all
            # because vLLM's V1 engine dropped it.
            ("priority",          getattr(event, "priority", None)),
            ("prompt_logprobs",   getattr(event, "prompt_logprobs", None)),
            ("seed",              getattr(event, "seed", None)),
            ("truncate_prompt_tokens",
             getattr(event, "truncate_prompt_tokens", None)),
        ]:
            if _val is not None:
                payload[_key] = _val
        # Snapshot which -inf writers this request actually engaged, so the
        # degenerate-mask oracle can reason about the composition without being
        # handed the Trace (``Oracle.evaluate`` only ever sees a TraceResult).
        result.mask_restrictors = mask_restrictors_of(event) or None
        # A streamed response carries NO usage block unless the request asks
        # for one, so without this the prefix-equivalence probe's vacuity
        # control (``cache_read_tokens``) is None on every streamed request no
        # matter how the server was launched.  Only legal alongside stream=true:
        # vLLM 400s ``stream_options`` on a non-streaming request
        # (``entrypoints/openai/completion/protocol.py:518-522``,
        # ``chat_completion/protocol.py:769-773``, vLLM 0.27.1).
        if event.stream and getattr(event, "include_usage", None):
            payload["stream_options"] = {"include_usage": True}
        if event.stop:
            result.had_stop_sequences = True
        try:
            if event.stream:
                # Collect output per sequence index so that n_completions > 1
                # requests are comparable across runs regardless of SSE
                # interleaving order.  For n=1 the result is identical to the
                # old plain-string behaviour.
                output_by_index: dict[int, list[str]] = {}
                finish_reason_by_index: dict[int, str] = {}
                stop_reason_by_index: dict[int, object] = {}
                stop_reason_seen = False
                logprobs_by_index: dict[int, list[float]] = {}
                top_logprobs_by_index: dict[int, list[dict]] = {}
                tokens_by_index: dict[int, list[str]] = {}
                has_logprobs_in_stream = False
                async with client.stream("POST", endpoint,
                                         json=payload) as resp:
                    async for chunk in resp.aiter_lines():
                        if not chunk or not chunk.startswith("data: "):
                            continue
                        raw = chunk[len("data: "):]
                        if raw.strip() == "[DONE]":
                            continue
                        try:
                            payload_json = _json.loads(raw)
                            # The usage block rides on its own final chunk with
                            # an EMPTY choices list, so it must be read before
                            # the no-choices early-out below — otherwise
                            # cache_read_tokens is None on every streamed
                            # request and the prefix-equivalence probe's
                            # vacuity control silently never runs.
                            _apply_usage_token_counts(
                                result, payload_json.get("usage"))
                            choices = payload_json.get("choices", [])
                            if not choices:
                                continue
                            if result.first_token_at is None:
                                result.first_token_at = time.time()
                            for choice in choices:
                                idx = choice.get("index", 0)
                                # completions uses "text"; chat uses "delta"."content"
                                text = choice.get("text") or \
                                       choice.get("delta", {}).get("content", "")
                                if text:
                                    result.last_token_at = time.time()
                                    result.tokens_received += 1
                                    output_by_index.setdefault(idx, []).append(text)
                                lp = choice.get("logprobs")
                                if lp is not None:
                                    has_logprobs_in_stream = True
                                    # completions format: logprobs.{token_logprobs, top_logprobs, tokens}
                                    token_lps = lp.get("token_logprobs") or []
                                    for v in token_lps:
                                        if v is not None:
                                            logprobs_by_index.setdefault(idx, []).append(float(v))
                                    for td in (lp.get("top_logprobs") or []):
                                        if td is not None:
                                            top_logprobs_by_index.setdefault(idx, []).append(td)
                                    for tok in (lp.get("tokens") or []):
                                        if tok is not None:
                                            tokens_by_index.setdefault(idx, []).append(tok)
                                    # chat format: logprobs.content[].{token, logprob, top_logprobs}
                                    for entry in (lp.get("content") or []):
                                        if entry is None:
                                            continue
                                        tok = entry.get("token")
                                        if tok is not None:
                                            tokens_by_index.setdefault(idx, []).append(tok)
                                        tlp = entry.get("logprob")
                                        if tlp is not None:
                                            logprobs_by_index.setdefault(idx, []).append(float(tlp))
                                        tops = entry.get("top_logprobs") or []
                                        if tops:
                                            td = {t["token"]: t["logprob"] for t in tops
                                                  if "token" in t and "logprob" in t}
                                            if td:
                                                top_logprobs_by_index.setdefault(idx, []).append(td)
                                fr = choice.get("finish_reason")
                                if fr is not None:
                                    finish_reason_by_index[idx] = fr
                                # vLLM's stop_reason rides on the *final* chunk
                                # only: _new_completion_output passes
                                # ``stop_reason if finished else None``
                                # (v1/engine/output_processor.py:422), so every
                                # earlier chunk carries the key with a null.
                                # Record presence separately from value, and let
                                # a later non-null overwrite an earlier null.
                                if "stop_reason" in choice:
                                    stop_reason_seen = True
                                    sr = choice.get("stop_reason")
                                    if sr is not None:
                                        stop_reason_by_index[idx] = sr
                        except (KeyError, IndexError, ValueError):
                            pass
                # Use finish_reason from choice index 0 (primary sequence) as
                # the representative value; "error" from any choice takes priority.
                if finish_reason_by_index:
                    all_reasons = list(finish_reason_by_index.values())
                    result.finish_reason = (
                        "error" if "error" in all_reasons
                        else finish_reason_by_index.get(0, all_reasons[0])
                    )
                if stop_reason_seen:
                    result.stop_reason_reported = True
                    # Any sequence truncated by a stop string/token taints the
                    # whole \x1f-joined output_text, so a non-null from *any*
                    # index wins over index 0's null.
                    if stop_reason_by_index:
                        result.stop_reason = stop_reason_by_index.get(
                            0, next(iter(stop_reason_by_index.values())))
                elif finish_reason_by_index:
                    result.stop_reason_reported = False
                # Join sequences in stable index order using ASCII unit separator
                # (\x1f) so the string is directly comparable across runs.
                sequences = [
                    "".join(output_by_index.get(i, []))
                    for i in sorted(output_by_index)
                ]
                result.output_text = "\x1f".join(sequences)
                result.logprobs_requested = event.logprobs
                result.n_completions_requested = event.n_completions
                result.response_has_logprobs = has_logprobs_in_stream
                result.response_n_choices = len(output_by_index) if output_by_index else 1
                # Store logprobs for choice index 0 (used by LoRA isolation tiebreaker)
                if logprobs_by_index:
                    result.output_logprobs = logprobs_by_index.get(0)
                if top_logprobs_by_index:
                    result.output_top_logprobs = top_logprobs_by_index.get(0)
                if tokens_by_index:
                    result.output_tokens = tokens_by_index.get(0)
                # Replace SSE-chunk counter with word count of assembled output —
                # chunk count is not a reliable proxy for token count.
                # Split by \x1f first so multi-completion outputs are counted per
                # sequence; otherwise "word\x1fword\x1fword" counts as 1 word
                # (since \x1f is not whitespace) and abnormal_length fires falsely.
                if result.output_text:
                    _seqs = result.output_text.split("\x1f")
                    result.tokens_received = sum(len(s.split()) for s in _seqs)
                else:
                    result.tokens_received = 0
            else:
                resp = await client.post(endpoint, json=payload)
                resp.raise_for_status()
                result.first_token_at = time.time()
                result.last_token_at  = time.time()
                try:
                    body = resp.json()
                    choices_sorted = sorted(
                        body.get("choices", []), key=lambda c: c.get("index", 0)
                    )
                    # completions: choice["text"]; chat: choice["message"]["content"]
                    sequences = [
                        c.get("text") or c.get("message", {}).get("content", "")
                        for c in choices_sorted
                    ]
                    result.output_text = "\x1f".join(sequences)
                    result.logprobs_requested = event.logprobs
                    result.n_completions_requested = event.n_completions
                    result.response_has_logprobs = any(
                        c.get("logprobs") is not None for c in choices_sorted
                    )
                    result.response_n_choices = len(choices_sorted)
                    # Validate logprob values — NaN/inf is always a bug
                    # Also store logprobs for choice index 0 (LoRA isolation tiebreaker)
                    if result.response_has_logprobs:
                        import math as _math
                        for c in choices_sorted:
                            lp = c.get("logprobs") or {}
                            token_lps = lp.get("token_logprobs") or []
                            for token_lp in token_lps:
                                if token_lp is not None and not _math.isfinite(token_lp):
                                    result.logprob_has_invalid = True
                                    break
                        # Store from choice index 0
                        first_lp = (choices_sorted[0].get("logprobs") or {}) if choices_sorted else {}
                        # completions format
                        raw_lps = [v for v in (first_lp.get("token_logprobs") or []) if v is not None]
                        if raw_lps:
                            result.output_logprobs = [float(v) for v in raw_lps]
                        raw_tops = first_lp.get("top_logprobs") or []
                        if raw_tops:
                            result.output_top_logprobs = raw_tops
                        raw_tokens = first_lp.get("tokens") or []
                        if raw_tokens:
                            result.output_tokens = raw_tokens
                        # chat format fallback
                        if not raw_lps and not raw_tops:
                            content_entries = first_lp.get("content") or []
                            if content_entries:
                                result.output_logprobs = [
                                    float(e["logprob"]) for e in content_entries
                                    if e and "logprob" in e
                                ]
                                result.output_tokens = [
                                    e["token"] for e in content_entries
                                    if e and "token" in e
                                ]
                                result.output_top_logprobs = [
                                    {t["token"]: t["logprob"] for t in (e.get("top_logprobs") or [])
                                     if "token" in t and "logprob" in t}
                                    for e in content_entries if e
                                ]
                    # finish_reason: use "error" from any choice if present.
                    all_reasons = [c.get("finish_reason") for c in choices_sorted
                                   if c.get("finish_reason") is not None]
                    if all_reasons:
                        result.finish_reason = (
                            "error" if "error" in all_reasons else all_reasons[0]
                        )
                    # stop_reason: vLLM-only sibling of finish_reason.  See
                    # RequestResult.stop_reason for the semantics and the lines
                    # they were read at.  Presence is tracked separately so the
                    # oracle can tell "server says EOS" from "server never
                    # reports this field".
                    if any("stop_reason" in c for c in choices_sorted):
                        result.stop_reason_reported = True
                        all_stops = [c.get("stop_reason") for c in choices_sorted
                                     if c.get("stop_reason") is not None]
                        if all_stops:
                            result.stop_reason = all_stops[0]
                    elif choices_sorted:
                        result.stop_reason_reported = False
                    # usage fields — completion_tokens already used; also grab
                    # prompt_tokens and vLLM-specific cache token counts.
                    usage = body.get("usage") or {}
                    usage_tokens = usage.get("completion_tokens")
                    if usage_tokens is not None:
                        result.tokens_received = int(usage_tokens)
                    elif result.output_text:
                        _seqs = result.output_text.split("\x1f")
                        result.tokens_received = sum(len(s.split()) for s in _seqs)
                    else:
                        result.tokens_received = 0
                    _apply_usage_token_counts(result, usage)
                except (KeyError, IndexError, ValueError):
                    result.output_text = None

            result.completed_at = time.time()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            result.error        = str(exc)
            result.completed_at = time.time()


# ---------------------------------------------------------------------------
# Rust executor (subprocess via stdin/stdout JSON)
# ---------------------------------------------------------------------------

class RustExecutor(Executor):
    """
    High-concurrency executor. Build with: cd runner && cargo build --release
    """

    def __init__(self, binary_path: str = "./runner/target/release/runner"):
        self.binary_path = binary_path

    async def run(self, trace: Trace, base_url: str) -> TraceResult:
        # TODO: pass trace JSON to binary via stdin, parse TraceResult from stdout
        raise NotImplementedError
