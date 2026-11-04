"""
Trace: the core fuzzing primitive.

A Trace is a timed sequence of client-side operations replayed against a
running inference server. It is the unit of mutation, corpus storage, and
reproduction.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    SEND       = "send"        # POST /v1/completions or /v1/chat/completions
    CANCEL     = "cancel"      # explicit abort via request ID
    DISCONNECT = "disconnect"  # close TCP socket without cancelling
    WAIT       = "wait"        # idle delay (used as a timing primitive)
    RESET      = "reset"       # POST /reset_prefix_cache (needs VLLM_SERVER_DEV_MODE=1)


@dataclass
class Event:
    action: ActionType
    offset_ms: int                  # ms from scenario start when this fires

    # Identity — populated at generation time, preserved across mutations
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # SEND parameters — completions path
    prompt_len: Optional[int]  = None   # synthetic prompt length in tokens
    max_tokens: Optional[int]  = None   # max output tokens
    stream:     bool           = True
    prefix_len: Optional[int]  = None   # shared prefix length (0 = no sharing)

    # SEND parameters — chat path (mutually exclusive with prompt_len)
    # Each entry: {"role": "system"|"user"|"assistant", "content_len": int}
    # system messages share a common prefix (cached); user messages are unique per request_id
    messages:   Optional[list] = None

    # Sampling parameters — all default None (omit from payload = use server default)
    temperature:        Optional[float] = None
    top_p:              Optional[float] = None
    top_k:              Optional[int]   = None
    min_tokens:         Optional[int]   = None
    n_completions:      Optional[int]   = None   # vLLM "n" — multiple outputs per request
    logprobs:           Optional[int]   = None
    repetition_penalty: Optional[float] = None
    stop:               Optional[list]  = None   # stop sequences (list of strings)

    # WAIT / DISCONNECT parameter
    duration_ms: Optional[int] = None   # used by WAIT; ms to hold disconnect open

    # Unicode stress testing
    unicode_stress: Optional[str] = None

    # LoRA adapter name — set by vllm_lora adapter mutations.
    # None means use the base model. Logic lives entirely in adapters/vllm_lora/.
    adapter_name: Optional[str] = None

    # Provenance tag — set by synthetic-event generators at mutation time.
    # None = unknown (seed event, loaded from disk, or original trace event).
    # Values: "seed", "mutated", "burst", "thrash", "storm"
    origin: Optional[str] = None

    # Prompt family ID — when set, requests with the same prompt_family_id
    # and prefix_len receive byte-identical prompts regardless of request_id.
    # Used by the LoRA isolation oracle and probe to make valid output comparisons.
    # None = default behaviour (unique suffix per request_id).
    prompt_family_id: Optional[str] = None

    # Prompt style tag — set by seed files or mutations to request a non-default
    # prompt generation strategy.  "rejection_heavy" alternates per-position hash
    # families to defeat n-gram draft speculation.  None = default behaviour.
    prompt_style: Optional[str] = None

    # Prefix-cache namespace salt (vLLM extra body field ``cache_salt``).
    # When set, the server hashes it into every prefix-cache block key for this
    # request, so an otherwise byte-identical prompt cannot hit blocks written
    # by an unsalted (or differently salted) twin.  Used by
    # ``Runner._prefix_equivalence_probe`` to run the same greedy prompt once
    # through the prefix cache and once around it inside a single batch.
    # None = default behaviour (field omitted from the request body).
    cache_salt: Optional[str] = None

    # Structured-output constraints (vLLM OpenAI-server body fields).  All
    # default None = omitted from the request body.  When any of them is set the
    # engine is contractually obliged to emit output matching the constraint, so
    # ``Oracle._check_schema_violation`` can decide pass/fail exactly, with no
    # statistical threshold — see an internal analysis note.
    #   response_format: {"type": "json_schema", "json_schema": {"schema": {...}}}
    #                    or {"type": "json_object"} (parseability only)
    response_format: Optional[dict] = None
    guided_json:     Optional[dict] = None   # JSON Schema, vLLM extension
    guided_regex:    Optional[str]  = None   # output must fullmatch this regex
    guided_grammar:  Optional[str]  = None   # EBNF/Lark; not exactly checked
    guided_choice:   Optional[list] = None   # output must be one of these

    # Logit-mask restrictors (vLLM OpenAI-server body fields).  Each one is an
    # *independent writer of ``-inf`` into the logits row*, and vLLM composes
    # them in sequence with no emptiness check anywhere on the compose path (see
    # an earlier finding).  Carried
    # here as plain pass-through so ``compose_mask_restrictors`` can set two or
    # more on one SEND and ``Oracle._check_degenerate_mask`` can see which ones
    # were in play.  All default None = the field is omitted from the request
    # body entirely, so a server build that rejects one never sees it.
    #   allowed_token_ids: allowlist — ONLY these ids may be sampled
    #   bad_words:         list of strings, tokenized server-side and banned
    #   logit_bias:        {"<token id>": float} additive bias on those ids
    #   stop_token_ids:    extra stop ids; feeds ``all_stop_token_ids``, which is
    #                      exactly the set ``min_tokens`` bans below the floor
    allowed_token_ids: Optional[list] = None
    bad_words:         Optional[list] = None
    logit_bias:        Optional[dict] = None
    stop_token_ids:    Optional[list] = None

    # First-class vLLM sampling parameters the Event could not carry until now.
    # Same plain pass-through contract as the fields above: None = the key is
    # omitted from the request body entirely, so a build that rejects one never
    # sees it on a trace that did not ask for it.
    #
    #   priority   Scheduling priority.  LOWER VALUE = HIGHER PRIORITY (vLLM's
    #              convention, ``SamplingParams``/``Request.priority``; the
    #              scheduler's waiting queue is ordered by ``(priority,
    #              arrival_time)``).  Only binds when the server was started
    #              with ``--scheduling-policy priority``; under the default
    #              ``fcfs`` the field is accepted and ignored.  Its absence is
    #              why the ``--enable-priority-scheduling`` campaign produced
    #              401 corpus entries and zero findings: with no request
    #              carrying a priority, the priority queue degenerates to
    #              arrival order and is byte-identical to FCFS.
    #   prompt_logprobs
    #              Number of logprobs to return FOR THE PROMPT.  Unlike
    #              ``logprobs`` (output only) this is O(prompt_len * k) work and
    #              O(prompt_len * k) response bytes, so it is the amplification
    #              dimension rather than an observability one.
    #   seed       Per-request sampling seed.  Makes a temperature > 0 request
    #              reproducible, which is what lets an oracle compare two runs
    #              of the same non-greedy request at all.
    #   truncate_prompt_tokens
    #              Server-side left-truncation of the prompt to the last N
    #              tokens, applied BEFORE prefix-cache hashing — so the same
    #              wire prompt with two different truncations is two different
    #              cache keys.
    #
    # ``n`` is deliberately NOT added here: ``n_completions`` above already
    # serialises to the request body's ``n`` (executor.py), and a second field
    # writing the same key would let two Event attributes disagree about one
    # wire value.  ``best_of`` is likewise absent — see the note in
    # ``fuzzer/executor.py``; vLLM's V1 engine removed it.
    priority:               Optional[int] = None
    prompt_logprobs:        Optional[int] = None
    seed:                   Optional[int] = None
    truncate_prompt_tokens: Optional[int] = None

    # Ask the server to report a ``usage`` block on a STREAMING request
    # (``stream_options: {"include_usage": true}``).  Without it a streamed
    # response carries no usage at all, so
    # ``RequestResult.cache_read_tokens`` — the only signal that says whether a
    # ``cache_salt`` actually bypassed the prefix cache — is unconditionally
    # None and ``_prefix_equivalence_probe``'s vacuity control cannot run.
    # vLLM rejects ``stream_options`` on a non-streaming request with a 400
    # (``entrypoints/openai/completion/protocol.py:518-522`` and
    # ``chat_completion/protocol.py:769-773`` in 0.27.1), so the executor only
    # puts it on the wire when ``stream`` is also set; on a non-streaming
    # request usage is returned unconditionally and this flag is a no-op.
    # None/False = the key is omitted entirely, so a build that rejects it
    # never sees it on a trace that did not ask for it.
    include_usage:          Optional[bool] = None

    # Send the prompt as ``prompt_embeds`` (a base64 ``torch.save`` tensor of
    # shape (prompt_len, hidden)) instead of text, on /v1/completions only.
    # The tensor is a deterministic function of the request's prompt identity,
    # so a replay sends the same bytes.  Needs --enable-prompt-embeds on the
    # server and the executor's ``prompt_embeds_dim``; without the latter the
    # request goes out as ordinary text.
    prompt_embeds:          Optional[bool] = None

    def validate(self) -> None:
        if self.action == ActionType.SEND:
            assert self.prompt_len is not None and self.prompt_len > 0
            assert self.max_tokens is not None and self.max_tokens > 0
        if self.action == ActionType.WAIT:
            assert self.duration_ms is not None and self.duration_ms >= 0


@dataclass
class Trace:
    seed:   int
    events: list[Event] = field(default_factory=list)

    # Optional tags for corpus bookkeeping
    tags: list[str] = field(default_factory=list)

    # --- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        d = asdict(self)
        # Convert enums to strings so JSON stays readable
        for ev in d["events"]:
            ev["action"] = ev["action"]
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> Trace:
        events = [
            Event(
                action=ActionType(ev["action"]),
                offset_ms=ev["offset_ms"],
                request_id=ev.get("request_id", str(uuid.uuid4())[:8]),
                prompt_len=ev.get("prompt_len"),
                max_tokens=ev.get("max_tokens"),
                stream=ev.get("stream", True),
                prefix_len=ev.get("prefix_len"),
                temperature=ev.get("temperature"),
                top_p=ev.get("top_p"),
                top_k=ev.get("top_k"),
                min_tokens=ev.get("min_tokens"),
                n_completions=ev.get("n_completions"),
                logprobs=ev.get("logprobs"),
                repetition_penalty=ev.get("repetition_penalty"),
                stop=ev.get("stop"),
                duration_ms=ev.get("duration_ms"),
                messages=ev.get("messages"),
                unicode_stress=ev.get("unicode_stress"),
                adapter_name=ev.get("adapter_name"),
                origin=ev.get("origin"),
                prompt_family_id=ev.get("prompt_family_id"),
                prompt_style=ev.get("prompt_style"),
                cache_salt=ev.get("cache_salt"),
                response_format=ev.get("response_format"),
                guided_json=ev.get("guided_json"),
                guided_regex=ev.get("guided_regex"),
                guided_grammar=ev.get("guided_grammar"),
                guided_choice=ev.get("guided_choice"),
                allowed_token_ids=ev.get("allowed_token_ids"),
                bad_words=ev.get("bad_words"),
                logit_bias=ev.get("logit_bias"),
                stop_token_ids=ev.get("stop_token_ids"),
                priority=ev.get("priority"),
                prompt_logprobs=ev.get("prompt_logprobs"),
                seed=ev.get("seed"),
                truncate_prompt_tokens=ev.get("truncate_prompt_tokens"),
                include_usage=ev.get("include_usage"),
                prompt_embeds=ev.get("prompt_embeds"),
            )
            for ev in d["events"]
        ]
        return cls(seed=d["seed"], events=events, tags=d.get("tags", []))

    @classmethod
    def from_json(cls, s: str) -> Trace:
        return cls.from_dict(json.loads(s))

    # --- Helpers -------------------------------------------------------------

    def send_events(self) -> list[Event]:
        return [e for e in self.events if e.action == ActionType.SEND]

    def is_deterministic(self) -> bool:
        """Returns True if all SEND events use greedy decoding parameters."""
        for ev in self.events:
            if ev.action != ActionType.SEND:
                continue
            if ev.temperature not in (None, 0.0):
                return False
            if ev.top_k not in (None, 1):
                return False
            if ev.top_p not in (None, 1.0):
                return False
        return True

    def duration_ms(self) -> int:
        """Wall-clock length of the scenario."""
        return max((e.offset_ms for e in self.events), default=0)

    def validate(self) -> None:
        for ev in self.events:
            ev.validate()
