"""
Core types shared across the fuzzer framework.

Moved from fuzzer/executor.py (RequestResult, TraceResult, error_signature)
and fuzzer/oracles.py (BugClass, OracleResult, constants).
"""

from __future__ import annotations

import hashlib as _hashlib
import re as _re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Result types (from executor.py)
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    request_id:      str
    sent_at:         float
    first_token_at:  Optional[float] = None
    last_token_at:   Optional[float] = None   # wall time of most recent SSE chunk with text
    completed_at:    Optional[float] = None
    cancelled_at:    Optional[float] = None
    tokens_received:      int             = 0
    error:                Optional[str]   = None
    stream:               bool            = True   # False for non-streaming requests
    prompt_len:           int             = 0      # tokens in prompt (for TTFT normalization)
    output_text:          Optional[str]   = None   # concatenated response text
    max_tokens_requested: Optional[int]   = None   # max_tokens from the SEND event
    temperature:          float           = 0.0    # propagated for determinism probe guard
    logprobs_requested:   Optional[int]   = None
    n_completions_requested: Optional[int] = None
    response_has_logprobs:   Optional[bool] = None
    response_n_choices:      Optional[int]  = None
    logprob_has_invalid:     Optional[bool] = None   # True if any NaN/inf logprob value
    had_stop_sequences:      bool           = False  # True if stop sequences were set on the request
    finish_reason:           Optional[str]  = None   # "stop" | "length" | "error" | None
    #: vLLM's per-choice ``stop_reason`` (a vLLM extension, not OpenAI spec).
    #: Verified in the 0.27.1 image at
    #: ``entrypoints/openai/completion/protocol.py:603`` (non-streaming) and
    #: ``:651`` (streaming), ``entrypoints/openai/chat_completion/protocol.py:110``
    #: and ``:160``; declared ``int | str | None`` on all four choice classes and
    #: populated straight from ``output.stop_reason`` in the serving layer
    #: (``completion/serving.py:399,414,587``,
    #: ``chat_completion/serving.py:724,1026``).  Same field, different lines, in
    #: 0.19.0 (``completion/protocol.py:460,494``,
    #: ``chat_completion/protocol.py:91,120``).
    #:
    #: Semantics read out of ``v1/core/sched/utils.py:94-130`` and
    #: ``v1/engine/output_processor.py:656-661``:
    #:   * ``None``  — the request ended without any stop-string / stop-token
    #:                 truncation: the primary EOS (``utils.py:104-106``) or the
    #:                 length cap (``:112-117``), neither of which assigns it;
    #:   * ``str``   — a **stop string** matched: the text was truncated at
    #:                 ``detokenizer.py:141`` and ``finish_reason`` was
    #:                 *overwritten* to ``"stop"`` at ``output_processor.py:660``;
    #:   * ``int``   — a **stop token id** fired (``utils.py:110``), checked
    #:                 *before* the length cap.
    #: Caveat carried by ``Oracle._terminated_voluntarily``: the id set is
    #: silently augmented from the model's ``generation_config.json``
    #: (``sampling_params.py:661-674``), so on Qwen2.5-0.5B-Instruct
    #: (``eos_token_id: [151645, 151643]``) a *voluntary* ``<|endoftext|>``
    #: also reports an ``int``.
    stop_reason:             Optional[object] = None
    #: True/False when the response choice did/did not carry a ``stop_reason``
    #: key at all; None when no choice was parsed.  Distinguishes "the server
    #: said EOS" from "the server never reports this field", which a bare
    #: ``stop_reason is None`` cannot.
    stop_reason_reported:    Optional[bool] = None
    prompt_tokens_reported:  Optional[int]  = None   # usage.prompt_tokens from response
    cache_read_tokens:       Optional[int]  = None   # usage.cache_read_input_tokens, else
                                                     # usage.prompt_tokens_details.cached_tokens
                                                     # (vLLM: needs --enable-prompt-tokens-details)
    cache_creation_tokens:   Optional[int]  = None   # usage.cache_creation_input_tokens (vLLM extension)
    prefix_len:              Optional[int]  = None   # prefix_len from the SEND event (for oracle grouping)
    adapter_name:            Optional[str]  = None   # LoRA adapter name from the SEND event; None = base model
    output_logprobs:         Optional[list] = None   # per-token logprob floats for choice index 0; None if not requested
    output_top_logprobs:     Optional[list] = None   # per-token top-k dicts [{token: logprob, ...}, ...]; None if not requested
    output_tokens:           Optional[list] = None   # per-token text list for choice index 0; None if not available
    prompt_family_id:        Optional[str]  = None   # shared prompt identity tag (same value → byte-identical prompt)
    #: Which independent ``-inf`` writers this SEND engaged, as
    #: ``{restrictor_name: value}`` — see :func:`mask_restrictors_of`.  None or
    #: {} means the request set none of them.  Populated by the executor at send
    #: time so ``Oracle._check_degenerate_mask`` can reason about the mask
    #: composition without being handed the ``Trace`` (``Oracle.evaluate`` is
    #: only ever given a ``TraceResult``).
    mask_restrictors:        Optional[dict] = None

    def ttft_ms(self) -> Optional[float]:
        if self.first_token_at:
            return (self.first_token_at - self.sent_at) * 1000
        return None

    def total_ms(self) -> Optional[float]:
        end = self.completed_at or self.cancelled_at
        return (end - self.sent_at) * 1000 if end else None

    def generation_elapsed_s(self) -> Optional[float]:
        """Wall time from first token to completion (streaming generation phase).
        Returns None if the request hasn't completed or never received a token."""
        if self.first_token_at is not None and self.completed_at is not None:
            return self.completed_at - self.first_token_at
        return None

    def timed_out(self, timeout_s: float, observed_at: float) -> bool:
        """True iff the request had neither completed nor cancelled by *observed_at*
        AND was sent more than *timeout_s* seconds before that observation.

        Callers must pass an explicit observation timestamp (typically
        ``TraceResult.finished_at``). Anchoring to wall-clock ``time.time()``
        in post-hoc verification would mark earlier runs as "more hung" simply
        because verification ran later.
        """
        if self.completed_at is not None or self.cancelled_at is not None:
            return False
        return (observed_at - self.sent_at) > timeout_s

    def post_cancel_delta_ms(self) -> Optional[float]:
        """ms between cancelled_at and last_token_at; positive means tokens arrived after cancel."""
        if self.cancelled_at is not None and self.last_token_at is not None:
            return (self.last_token_at - self.cancelled_at) * 1000
        return None


@dataclass
class TraceResult:
    trace_seed:  int
    started_at:  float
    finished_at: float
    requests:    list[RequestResult] = field(default_factory=list)
    # One dict per RESET event: offset_ms, sent_at, done_at, status, success,
    # in_flight (SEND requests not yet finished when the reset was issued).
    resets:      list = field(default_factory=list)

    def hung_requests(self, timeout_s: float = 30.0,
                      observed_at: Optional[float] = None) -> list[RequestResult]:
        """Requests that hadn't completed or cancelled by *observed_at* and whose
        sent_at is more than *timeout_s* seconds before that observation.

        *observed_at* defaults to ``self.finished_at``, so post-hoc analysis is
        anchored to when the trace stopped collecting (not when verification ran).
        """
        anchor = observed_at if observed_at is not None else self.finished_at
        return [r for r in self.requests if r.timed_out(timeout_s, anchor)]

    def post_cancel_tokens(self, threshold_ms: float = 5.0) -> list[RequestResult]:
        """Requests where last_token_at exceeds cancelled_at by more than threshold_ms.

        Uses last_token_at (not first_token_at) so mid-stream cancellations are
        detected correctly — a request can receive tokens normally and still have
        late tokens arrive after cancel.  The threshold filters TCP/SSE in-flight
        noise (typically < 4ms); anything beyond it implies the server was still
        actively generating after the client disconnected.
        """
        t = threshold_ms / 1000.0
        return [
            r for r in self.requests
            if r.cancelled_at is not None
            and r.last_token_at is not None
            and r.last_token_at > r.cancelled_at + t
        ]


# ---------------------------------------------------------------------------
# Error signature hashing (from executor.py)
# ---------------------------------------------------------------------------

def error_signature(requests: list) -> Optional[str]:
    """
    Returns a 12-char hex hash of the distinct error classes seen in this
    trace, or None if there were no errors.

    Variable parts (memory addresses, long numbers, request IDs) are stripped
    before hashing so the same exception type with different runtime values
    hashes identically across runs.
    """
    errors = [r.error for r in requests if r.error is not None]
    if not errors:
        return None
    normalized = []
    for e in errors:
        s = _re.sub(r'0x[0-9a-fA-F]+', 'ADDR', e)
        s = _re.sub(r'\b\d{6,}\b', 'NUM', s)
        s = _re.sub(r'request_id=[^\s,]+', 'RID', s)
        normalized.append(s[:120])
    key = "|".join(sorted(set(normalized)))
    return _hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Logit-mask restrictors
# ---------------------------------------------------------------------------
#
# vLLM writes ``-inf`` into the logits row from several *independent* places and
# composes them in sequence with no emptiness check anywhere on the path.  The
# pair that produced the archived ``min_tokens`` + grammar finding is one cell of
# this table; the mask-composition surface fuzzes the rest of it.
#
# ``structured_output`` is the odd one out in a way the oracle depends on: the
# grammar bitmask is applied in the model runner, *before* the sampler snapshots
# the logprobs it reports, whereas every other entry here is a logits processor
# that runs *after* that snapshot.  So a composition that includes the grammar
# is directly observable over HTTP (the sampled token carries the ``-inf``
# sentinel while a grammar-legal alternative is still finite), and one that does
# not is invisible in the logprobs — see ``Oracle._check_degenerate_mask``.
#
# ``logit_bias`` is listed because callers reach for it as a ban, but vLLM
# applies it *additively*; a large negative bias lowers a logit, it does not
# make it ``-inf``, so on its own it cannot empty a row.

#: Restrictor names, in the order they are reported.
MASK_RESTRICTORS = (
    "structured_output",
    "min_tokens",
    "allowed_token_ids",
    "bad_words",
    "logit_bias",
    "stop_token_ids",
)

#: Restrictors applied *before* the sampler's reported-logprob snapshot.  Only a
#: composition containing one of these is visible to the sentinel predicate.
PRE_LOGPROB_RESTRICTORS = frozenset({"structured_output"})

#: vLLM renders ``-inf`` logprobs as this value for JSON transport.
#:
#: Read in the 0.27.1 image: for *output* logprobs it is a **clamp, not a
#: sentinel map** — ``max(step_token.logprob, -9999.0)`` at
#: ``entrypoints/openai/chat_completion/serving.py:1215`` (sampled token) and
#: ``:1158`` (top-k), and ``entrypoints/openai/completion/serving.py:703`` /
#: ``:721`` for the completions endpoint.  Only the *prompt*-logprob path maps
#: ``-inf`` exactly (``entrypoints/generate/base/serving.py:305-317``).
#:
#: Two ways the value can therefore appear without any ``-inf`` behind it, both
#: of which ``Oracle._check_degenerate_mask`` guards against explicitly:
#:   1. a genuine finite logprob at or below -9999 clamps onto it.  That needs a
#:      logit spread of 1e4 nats inside one row — far outside anything a model
#:      produces, and unreachable through ``logit_bias``, whose values are hard
#:      clamped to +/-100 on every HTTP path (``sampling_params.py:399``,
#:      ``:411``) *and* which is additive rather than a mask
#:      (``v1/sample/logits_processor/builtin.py:161``).
#:   2. ``ChatCompletionLogProb.logprob`` **defaults** to -9999.0
#:      (``chat_completion/protocol.py:83``) and
#:      ``chat_completion/serving.py:1194-1207`` builds the content entry with
#:      no ``logprob`` kwarg when the sampled token is missing from that step's
#:      top-k.  That emits -9999.0 with no ``-inf`` anywhere.  The oracle rules
#:      it out by requiring the sampled token to be *present* in the position's
#:      top-k, itself carrying the value.
NEG_INF_LOGPROB_SENTINEL = -9999.0


def _structured_output_kind(event) -> Optional[str]:
    """Which structured-output constraint a SEND carries, or None."""
    if getattr(event, "guided_json", None):
        return "json_schema"
    if getattr(event, "guided_regex", None):
        return "regex"
    if getattr(event, "guided_choice", None):
        return "choice"
    if getattr(event, "guided_grammar", None):
        return "grammar"
    rf = getattr(event, "response_format", None)
    if isinstance(rf, dict) and rf:
        t = rf.get("type")
        if t in ("json_schema", "json_object"):
            return t
    return None


def mask_restrictors_of(event) -> dict:
    """``{restrictor_name: value}`` for every ``-inf`` writer a SEND engages.

    Order follows :data:`MASK_RESTRICTORS`.  ``min_tokens`` only counts when it
    is truthy: ``min_tokens=0`` never registers the processor at all
    (``MinTokensLogitsProcessor.add_request`` returns ``None``), so it is not a
    writer.  Empty lists/dicts are likewise dropped — a field the server would
    reject or ignore is not part of the composition.
    """
    out: dict = {}
    kind = _structured_output_kind(event)
    if kind:
        out["structured_output"] = kind
    min_tokens = getattr(event, "min_tokens", None)
    if min_tokens:
        out["min_tokens"] = min_tokens
    for name in ("allowed_token_ids", "bad_words", "logit_bias",
                 "stop_token_ids"):
        val = getattr(event, name, None)
        if val:
            out[name] = val
    return out


# ---------------------------------------------------------------------------
# Bug classification (from oracles.py)
# ---------------------------------------------------------------------------

class BugClass(str, Enum):
    MEMORY_LEAK      = "memory_leak"
    LIFECYCLE_ERROR  = "lifecycle_error"   # tokens after cancel, stuck req
    SCHEDULER_HANG   = "scheduler_hang"    # waiting queue never drains
    LIVENESS_FAILURE = "liveness_failure"  # server stopped responding
    LATENCY_SPIKE    = "latency_spike"     # p99 TTFT exceeds threshold
    CORRECTNESS      = "correctness"       # differential output mismatch
    KV_CORRUPTION    = "kv_corruption"     # non-deterministic output after fuzzing
    CACHE_EVICTION   = "cache_eviction"    # unexpected KV block eviction (TTFT regression)
    KV_BLOCK_LEAK    = "kv_block_leak"     # KV cache usage doesn't return to baseline
    INCONSISTENCY    = "inconsistency"
    ABNORMAL_LENGTH  = "abnormal_length"
    REPETITION       = "repetition"
    CHAR_CORRUPTION  = "char_corruption"
    FEATURE_FAILURE  = "feature_failure"
    TOKEN_RATE_STALL          = "token_rate_stall"           # mid-stream stall after first token
    ADAPTER_ISOLATION_FAILURE = "adapter_isolation_failure"  # LoRA cross-adapter KV leakage
    # Structured-output constraint violated (a *specified* oracle: the engine
    # guarantees the grammar, so a finished output that does not satisfy it is
    # a defect with no statistical threshold involved).
    SCHEMA_VIOLATION = "schema_violation"
    CACHE_RESET_IGNORED = "cache_reset_ignored"  # prefix-cache reset reported success, cache still served
    # Two or more independent writers of ``-inf`` composed into a logits row
    # with no legal token left.  Also a *specified* oracle: whatever the pair,
    # "the composed mask is never empty" is an invariant vLLM has no code to
    # enforce, and each predicate below decides it exactly.
    DEGENERATE_MASK  = "degenerate_mask"
    # A probe's own negative control failed, so the probe's finding is not
    # interpretable.  This reports on the INSTRUMENT, not on the server under
    # test, and must never merge with the class it invalidates.  Raised by
    # ``Runner._prefix_equivalence_probe`` when its two independently salted
    # arms — both pure recomputes of a byte-identical greedy prompt, neither
    # able to hit the other's prefix-cache blocks — disagree with each other.
    # When that happens the engine is nondeterministic independently of the
    # prefix cache (or the probe perturbs its own measurement), and the
    # cached-vs-recompute comparison the probe is built on says nothing.
    PROBE_SELF_CHECK = "probe_self_check"


# ---------------------------------------------------------------------------
# Fatal-error classification
# ---------------------------------------------------------------------------
#
# ``RequestResult.error`` is ``str(exc)`` (``fuzzer/executor.py:396``), so the
# strings we classify are whatever the client raised.  The two shapes that
# carry an HTTP status are httpx's ``raise_for_status()`` rendering \u2014
#     "Client error '400 Bad Request' for url 'http://.../v1/completions'"
#     "Server error '503 Service Unavailable' for url 'http://.../v1/completions'"
# \u2014 and a bare leading status, e.g. "500 Internal Server Error" or the OpenAI
# client's "Error code: 500 - {...}".  Everything else never got a reply.
#
# This used to be a tuple ``("500", "502", "503", "504", ...)`` tested with a
# bare ``p in error`` substring match, which fired on any 5xx *digit run*
# anywhere in the text.  Verified false positives, all three of which are among
# the commonest errors under fuzzing load:
#     "400 Bad Request: max_tokens 8500 exceeds context length"   -> "500"
#     "429 Too Many Requests (retry after 5040ms)"                -> "504"
#     "Request timed out after 500 s"                             -> "500"
# Each was filed as a LIVENESS_FAILURE finding.  Match on the actual status
# instead.

_RE_HTTPX_STATUS = _re.compile(r"\b(?:client|server) error\s+'([1-5]\d\d)\b",
                               _re.IGNORECASE)
#: A status at the very start of the message ("503 Service Unavailable").
#: Anchored deliberately: an unanchored 3-digit search matches "500" inside
#: "Request timed out after 500 s".
_RE_LEADING_STATUS = _re.compile(r"^\s*(?:HTTP/[\d.]+\s+)?([1-5]\d\d)\b")
#: OpenAI/vLLM client rendering: "Error code: 500 - {'object': 'error', ...}".
_RE_ERROR_CODE = _re.compile(r"\berror code:\s*([1-5]\d\d)\b", _re.IGNORECASE)

#: Transport failures \u2014 no HTTP status ever arrived, so the server is not
#: answering at all.  Matched case-insensitively against the whole message.
#: ``timed out`` is deliberately absent: timeouts belong to the latency oracle.
_FATAL_CONN_MARKERS = (
    "connection refused", "connectionrefusederror",
    "connection reset", "connectionreseterror",
    "connecterror",
)


def http_status_from_error(error: str) -> Optional[int]:
    """Return the HTTP status an error string reports, or None if it has none.

    Only the renderings that actually encode a status are consulted (see the
    module comment above) \u2014 a loose digit search cannot tell a status apart
    from a token budget or a retry-after value.
    """
    for pattern in (_RE_HTTPX_STATUS, _RE_LEADING_STATUS, _RE_ERROR_CODE):
        m = pattern.search(error)
        if m:
            return int(m.group(1))
    return None


def is_fatal_error(error: Optional[str]) -> bool:
    """True when *error* means the server failed, not that the request was bad.

    Fatal: any 5xx response, or a transport failure (refused / reset / connect
    error) where no status ever came back.

    Not fatal: every 4xx \u2014 400 (context length, bad params) and 429 (rate
    limit) are the expected outcome of fuzzing an inference server and say
    nothing about its health \u2014 and timeouts, which the latency oracle owns.
    """
    if not error:
        return False
    status = http_status_from_error(error)
    if status is not None:
        # A status came back, so the server is alive and answering.  Only its
        # own 5xx failures count; a 4xx is a verdict on our request.
        return 500 <= status <= 599
    low = error.lower()
    return any(marker in low for marker in _FATAL_CONN_MARKERS)


#: Retained for backwards compatibility with importers of the old constant
#: (``fuzzer/oracles.py`` re-exports it).  No longer used for matching \u2014 see
#: :func:`is_fatal_error`.
_FATAL_PATTERNS = ("500", "502", "503", "504",
                   "Connection refused", "Connection reset",
                   "ConnectionRefusedError", "ConnectError")

_CORRUPT_RE = _re.compile(r'\ufffd|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

MIN_LATENCY_SAMPLE = 5       # minimum streaming requests for a meaningful p99
_LATENCY_SPIKE_MIN_MS = 500.0  # absolute TTFT floor — ignore ratios below this


@dataclass
class OracleResult:
    passed:    bool
    bug_class: Optional[BugClass] = None
    detail:    str                = ""
    score:     float              = 0.0   # higher = more interesting for corpus
    probe_metadata: Optional[dict] = None  # structured data for isolation probe scheduler
    #: Set when the oracle *declined to judge* because a precondition of its
    #: measurement was not met (e.g. the engine had not drained, so an
    #: end-of-trace metric describes a busy server).  Such a result still has
    #: ``passed=True`` — it files nothing — but it is NOT a clean bill of
    #: health, and the runner records it as a measurement gap instead of
    #: letting it look like a pass.  Format: ``"<check>: <reason>"``.
    unmeasured: Optional[str] = None
