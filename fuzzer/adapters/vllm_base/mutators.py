"""
vLLM-specific mutation strategies.

These mutations exercise vLLM-specific features and code paths
that are not generic to other serving backends.
"""

from __future__ import annotations

import copy
import hashlib
import random

from ...trace import Trace, Event, ActionType


_CUDA_GRAPH_SMALL = [16, 32, 64, 128]
_CUDA_GRAPH_LARGE = [1024, 2048, 4096, 8192]  # large values force eager mode fallback


def cuda_graph_thrash(trace: Trace, rng: random.Random) -> Trace:
    """Alternate prompt_len between small/large buckets to stress
    CUDA graph shape transitions and eager fallback."""
    t = copy.deepcopy(trace)
    sends = sorted(
        [ev for ev in t.events if ev.action == ActionType.SEND],
        key=lambda e: e.offset_ms,
    )
    small = rng.choice(_CUDA_GRAPH_SMALL)
    large = rng.choice(_CUDA_GRAPH_LARGE)
    for i, ev in enumerate(sends):
        ev.prompt_len = small if i % 2 == 0 else large
    return t


def mutate_sampling_params(trace: Trace, rng: random.Random) -> Trace:
    """Randomize sampling parameters to exercise non-greedy decoding paths.

    temperature > 0  → stochastic sampling (different CUDA kernel path)
    n_completions > 1 → sequence group branching (multiplies KV memory usage)
    logprobs > 0      → activates logprobs computation and serialization path

    Every request this mutator leaves non-greedy also gets an explicit
    ``seed``.  On vLLM 0.27.1 the V2 sampler picks its FlashInfer branch **per
    batch**, not per request (``v1/worker/gpu/sample/sampler.py:240-252``): the
    branch is taken unless the batch contains a greedy request *or* a request
    with an explicit seed, and ``flashinfer_sample`` then draws from the default
    CUDA generator instead of a per-request one.  So an unseeded non-greedy
    request's token stream depends on whether some *other* tenant in the same
    forward batch happened to be greedy -- which our determinism / stability
    oracles would file as corruption.  A per-request seed pins the request to
    the Gumbel path, whose noise is ``f(seed, position, vocab_index)`` and
    therefore invariant to batch shape, chunking and prefix-cache hits.
    (Analysed in an internal analysis note.)  An
    Event that already carries a seed keeps it, so a seed/probe that pinned a
    specific value stays reproducible.
    """
    t     = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    targets = rng.sample(sends, min(2, len(sends)))
    for ev in targets:
        # temperature: 50% greedy, 30% low, 20% high
        choice = rng.random()
        if choice < 0.50:
            ev.temperature = 0.0
        elif choice < 0.80:
            ev.temperature = round(rng.uniform(0.01, 0.5), 2)
        else:
            ev.temperature = round(rng.uniform(0.5, 2.0), 2)

        if rng.random() < 0.4:
            ev.top_p = round(rng.uniform(0.8, 1.0), 2)
        if rng.random() < 0.3:
            ev.top_k = rng.choice([10, 20, 50])

        # n_completions > 1 stresses sequence group management
        n = rng.choices([1, 2, 4, 8], weights=[70, 15, 10, 5])[0]
        ev.n_completions = None if n == 1 else n

        if rng.random() < 0.3:
            ev.logprobs = rng.choice([1, 5, 20])
        if rng.random() < 0.3:
            ev.repetition_penalty = round(rng.uniform(1.0, 1.5), 2)

        # Non-greedy => must carry an explicit seed (see the docstring).  Drawn
        # from this mutator's own rng so the same (trace, rng seed) pair always
        # produces the same wire value; never overwritten if one is already set.
        if ev.temperature and ev.temperature > 0.0 and ev.seed is None:
            ev.seed = rng.randrange(1, 2 ** 31)
    return t


def inject_chat_turn(trace: Trace, rng: random.Random) -> Trace:
    """Convert a random SEND event to a chat completions request.

    System message = shared prefix (same content_len → same cached KV block).
    User message   = unique per request_id → different KV suffix per request.
    This exercises the /v1/chat/completions prefix-caching path.
    """
    t     = copy.deepcopy(trace)
    sends = [ev for ev in t.events
             if ev.action == ActionType.SEND and ev.messages is None
             and ev.prompt_len is not None]
    if not sends:
        return t
    target     = rng.choice(sends)
    system_len = min(target.prefix_len or rng.choice([64, 128, 256]),
                     target.prompt_len - 1)
    system_len = max(1, system_len)
    user_len   = max(1, target.prompt_len - system_len)
    target.messages   = [
        {"role": "system", "content_len": system_len},
        {"role": "user",   "content_len": user_len},
    ]
    target.prompt_len = None   # consumed by messages; not used for chat path
    target.prefix_len = None
    return t


def retry_storm(trace: Trace, rng: random.Random) -> Trace:
    """Pick a SEND, inject cancel(s), then re-send with a storm request_id."""
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    target = rng.choice(sends)
    first_cancel_offset = target.offset_ms + rng.randint(50, 300)
    cancel1 = Event(
        action=ActionType.CANCEL,
        offset_ms=first_cancel_offset,
        request_id=target.request_id,
    )
    t.events.append(cancel1)
    # 50% chance of a second cancel
    if rng.random() < 0.5:
        cancel2 = Event(
            action=ActionType.CANCEL,
            offset_ms=first_cancel_offset + rng.randint(10, 100),
            request_id=target.request_id,
        )
        t.events.append(cancel2)
    storm_offset = first_cancel_offset + rng.randint(20, 150)
    prefix_len = max(0, min(
        (target.prefix_len or 0) + rng.randint(-16, 16),
        (target.prompt_len or 1) - 1
    ))
    storm_send            = copy.deepcopy(target)
    storm_send.offset_ms  = storm_offset
    storm_send.request_id = target.request_id + "_storm"
    storm_send.prefix_len = prefix_len
    storm_send.origin     = "storm"
    t.events.append(storm_send)
    return t


def inject_mixed_endpoints(trace: Trace, rng: random.Random) -> Trace:
    """Ensure trace has both chat and completions SEND events."""
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    chat_sends = [ev for ev in sends if ev.messages is not None]
    comp_sends = [ev for ev in sends if ev.messages is None]

    if chat_sends and comp_sends:
        # Already mixed — return as-is
        return t

    if not chat_sends:
        # Convert one completions SEND to chat
        target = rng.choice(comp_sends)
        prompt_len = target.prompt_len or 64
        system_len = 32
        user_len = max(1, prompt_len - system_len)
        target.messages = [
            {"role": "system", "content_len": system_len},
            {"role": "user", "content_len": user_len},
        ]
        target.prompt_len = None
        target.prefix_len = None
    else:
        # Convert one chat SEND to completions
        target = rng.choice(chat_sends)
        total_len = sum(msg.get("content_len", 16) for msg in target.messages)
        target.prompt_len = max(1, total_len)
        target.messages = None
        target.prefix_len = None
    return t


_UNICODE_STRESS_TYPES = ["cjk", "emoji", "combining", "rtl_mixed"]


def inject_unicode_stress(trace: Trace, rng: random.Random) -> Trace:
    """Inject unicode stress tokens into 1-3 SEND events."""
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    count = min(rng.randint(1, 3), len(sends))
    targets = rng.sample(sends, count)
    stress_type = rng.choice(_UNICODE_STRESS_TYPES)
    for ev in targets:
        ev.unicode_stress = stress_type
    return t


def inject_multi_system_message(trace: Trace, rng: random.Random) -> Trace:
    """Add an extra system message at the beginning of a chat event's messages."""
    t = copy.deepcopy(trace)
    chat_sends = [ev for ev in t.events
                  if ev.action == ActionType.SEND and ev.messages is not None]
    if not chat_sends:
        return inject_chat_turn(t, rng)
    target = rng.choice(chat_sends)
    extra_system = {"role": "system", "content_len": rng.choice([16, 32, 64])}
    target.messages = [extra_system] + list(target.messages)
    return t


def inject_rtl_system_message(trace: Trace, rng: random.Random) -> Trace:
    """Set unicode_stress='rtl_mixed' on chat SEND events."""
    t = copy.deepcopy(trace)
    chat_sends = [ev for ev in t.events
                  if ev.action == ActionType.SEND and ev.messages is not None]
    if not chat_sends:
        return t
    for ev in chat_sends:
        ev.unicode_stress = "rtl_mixed"
    return t


def inject_max_prefix(trace: Trace, rng: random.Random) -> Trace:
    """Set prefix_len = prompt_len - 1 on all completions SEND events."""
    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND and ev.prompt_len:
            ev.prefix_len = ev.prompt_len - 1
    return t


_STOP_SEQUENCES = [
    "The", "I ", "In ", "\n\n", "Note", "Here", "This", "However",
    "For", "A ", "1.", "2.", "First", "Second", "Therefore",
]


def inject_stop_sequences(trace: Trace, rng: random.Random) -> Trace:
    """Add random stop sequences to SEND events."""
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    targets = rng.sample(sends, min(rng.randint(1, 3), len(sends)))
    for ev in targets:
        count = rng.randint(1, 3)
        ev.stop = rng.sample(_STOP_SEQUENCES, min(count, len(_STOP_SEQUENCES)))
    return t


def memory_pressure(trace: Trace, rng: random.Random) -> Trace:
    """Scale all prompts to near-limit lengths (2000–8000 tokens).

    Targets the KV allocator under high utilisation: swap events, OOM-adjacent
    preemptions, and block-reuse races that only appear when the cache is nearly
    full. Registered once in VLLM_MUTATIONS (same weight as others); campaigns
    wanting to avoid pressure-regime saturation can pass a filtered mutations=
    list to mutate().
    """
    t      = copy.deepcopy(trace)
    target = rng.choice([2000, 3000, 4000, 6000, 8000])
    for ev in t.events:
        if ev.action == ActionType.SEND:
            if ev.prompt_len:
                ev.prompt_len = target
            elif ev.messages:
                per_msg = max(1, target // len(ev.messages))
                for msg in ev.messages:
                    msg["content_len"] = per_msg
    return t


def chunk_boundary_align(trace: Trace, rng: random.Random) -> Trace:
    """Snap prompt lengths to chunk-boundary neighbourhoods (k×512 − 1/0/+1).

    Tests fencepost conditions in chunked-prefill: the scheduler fills exactly
    one chunk, splits across a boundary, or has one token left over. All three
    offsets are equally probable so the mutation covers the full neighbourhood.
    """
    CHUNK = 512
    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND and ev.prompt_len:
            k             = max(1, round(ev.prompt_len / CHUNK))
            base          = k * CHUNK
            ev.prompt_len = max(1, base + rng.choice([-1, 0, 1]))
    return t


def kv_block_boundary(trace: Trace, rng: random.Random) -> Trace:
    """Snap prompt lengths to KV block-size boundaries (k×16 − 1/0/+1).

    vLLM allocates KV cache in blocks of 16 tokens.  Off-by-one errors in
    block_table management (stale tail zeroing, partial-block reuse) are most
    likely to surface at exact block boundaries.
    """
    BLOCK = 16
    t = copy.deepcopy(trace)
    for ev in t.events:
        if ev.action == ActionType.SEND and ev.prompt_len:
            k             = max(1, round(ev.prompt_len / BLOCK))
            base          = k * BLOCK
            ev.prompt_len = max(1, base + rng.choice([-1, 0, 1]))
    return t


def min_tokens_conflict(trace: Trace, rng: random.Random) -> Trace:
    """Set min_tokens AND an early stop sequence on a SEND event.

    When min_tokens forces the model to keep generating but a stop sequence
    matches, vLLM must resolve the conflict.  Edge cases in this interaction
    have produced incorrect finish_reason and truncated outputs.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    target = rng.choice(sends)
    target.min_tokens = rng.choice([8, 16, 32, 64])
    target.stop = rng.sample(
        ["The", "I ", "\n", ".", ",", "is", "and", "to"],
        min(rng.randint(1, 3), 8),
    )
    return t


# ---------------------------------------------------------------------------
# Mask composition
# ---------------------------------------------------------------------------
#
# vLLM has at least six independent writers of ``-inf`` into the logits row and
# composes them in sequence with no emptiness check anywhere on the path (the
# archived ``min_tokens`` + grammar finding is one cell of that N x N table).
# ``compose_mask_restrictors`` sets two or more of them on one SEND so their
# intersection is *plausibly* — in one recipe, *provably* — empty.
#
# Choosing token ids without a tokenizer.  The mutator has no vocabulary: the
# adapter builds prompts out of synthetic word tokens and never loads a
# tokenizer, so it cannot know which id is ``{`` or which is EOS.  Three
# tokenizer-free sources are used instead:
#
#   * **Low ids.**  Every vocabulary has them, so they can never be
#     out-of-range (which is a 400, not a finding).  In byte-level BPE
#     vocabularies — Qwen2, Llama, Mistral — ids 1..8 are single printable
#     punctuation bytes, none of which a JSON grammar admits at the root, where
#     only ``{`` and whitespace are legal.  Id 0 is deliberately excluded: it is
#     the token an all-``-inf`` ``argmax`` returns, so allowing it would let the
#     degenerate row look legal.
#   * **Ids derived from the prompt.**  ``_prompt_derived_ids`` hashes the
#     request's own identity into a small in-range id, so different requests in
#     one trace restrict different parts of the row instead of all colliding on
#     the same window.
#   * **The stop token, named by us.**  The adapter does not know the model's
#     EOS id — but ``stop_token_ids`` *adds* to the set ``min_tokens`` bans
#     (``all_stop_token_ids``), so a request can nominate its own stop id.
#     ``stop_token_ids=[k]`` + ``allowed_token_ids=[k]`` + ``min_tokens>0`` is
#     then an empty mask by construction on every model and every tokenizer,
#     with no grammar and no vocabulary knowledge involved.

#: A two-field object the base model satisfies unconstrained, so a failure is
#: mask composition and not a capability gap.  Same shape as
#: ``seeds/guided_smoke/guided_json_flat.json``.
_MASK_COMPOSE_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["name", "count"],
    "additionalProperties": False,
}

#: Ids safe to name on any vocabulary (see the note above).  0 is excluded.
_SAFE_LOW_IDS = (1, 2, 3, 4, 5, 6, 7, 8)

#: Strings a JSON grammar needs at or near the root of an object.  Banning them
#: as ``bad_words`` leaves the grammar with (at most) whitespace.  These are
#: single characters on purpose: ``bad_words`` bans a *sequence*, and only the
#: single-token case bans unconditionally — a multi-token word has
#: ``prefix_length > 0`` and fires only once the preceding context matches
#: (``v1/sample/ops/bad_words.py:14-27``).  A one-token word takes the
#: ``actual_prefix == expected_prefix == []`` branch at every step.
_JSON_BAD_WORDS = ("{", "\"", "}", ":")

#: ``min_tokens`` floors.  Every one exceeds the ~13-19 tokens the flat schema
#: costs on a small instruct model, which is the condition under which the
#: grammar mask collapses to stop-tokens-only while the floor is still binding.
_MASK_MIN_TOKENS = (32, 64, 96)

#: Recipe names, in registry order.  Recorded on the event only through the
#: fields themselves — the oracle re-derives the composition from the wire
#: fields, so nothing here has to stay in sync with it.
MASK_COMPOSE_RECIPES = (
    "grammar_x_min_tokens",
    "grammar_x_allowlist",
    "grammar_x_bad_words",
    "allowlist_x_min_tokens",
    "grammar_x_allowlist_x_min_tokens",
)


def _prompt_derived_ids(event, rng: random.Random, count: int) -> list:
    """*count* distinct small token ids derived from this request's identity.

    Stable for a given request_id so a replay restricts the same part of the
    row, and bounded well below any real vocabulary so it can never 400.
    """
    seed_str = (getattr(event, "prompt_family_id", None)
                or getattr(event, "request_id", None) or "")
    if seed_str:
        # md5, not hash(): PYTHONHASHSEED randomises str hashing per process,
        # which would make the same trace restrict a different window on every
        # replay.
        digest = hashlib.md5(seed_str.encode()).hexdigest()[:8]
        base = (int(digest, 16) % 64) + 1
    else:
        base = rng.randint(1, 64)
    return [base + i for i in range(count)]


# NOTE: out-of-range stop_token_ids / allowed_token_ids are deliberately NOT
# generated here. An id past the vocabulary reaches logits.index_put_ once
# min_tokens >= 1 and kills the engine unrecoverably, which would end the slot
# for every other request. It is also already upstream (vLLM PRs #44968,
# #42614, #49080 -- all open, unmerged) and is not a GRIEF finding.
# VERIFIED 2026-08-26 -- do not re-open this line: #42614 is not confined to the
# V1 index_put_ crash, it already carries the bounds mask for the V2 Triton
# kernel too (the ``tl.store`` of ``stop_token_ids`` in
# ``vllm/v1/worker/gpu/sample/logit_bias.py``).  So the V2 silent
# cross-request-corruption variant is reported upstream as well, not only the V1
# CUDA-assert variant.
def compose_mask_restrictors(trace: Trace, rng: random.Random) -> Trace:
    """Set two or more independent ``-inf`` writers on one SEND so that their
    intersection is plausibly empty.

    Four recipes, one drawn per call:

    ``grammar_x_min_tokens``
        The archived cell, kept as the surface's control: a JSON-schema
        constraint whose value the model finishes before a ``min_tokens`` floor
        the grammar can then only satisfy with a stop token — which is exactly
        what ``min_tokens`` bans.
    ``grammar_x_allowlist``
        An ``allowed_token_ids`` window of low punctuation ids, disjoint from
        the ``{``-or-whitespace opening the JSON grammar admits.
    ``grammar_x_bad_words``
        The same grammar with the structural characters it cannot do without
        (``{``, ``"``, ``}``, ``:``) banned as single-token ``bad_words``,
        which is the one ``bad_words`` shape that bans unconditionally.
    ``allowlist_x_min_tokens``
        The tokenizer-free, provably-empty one: the request nominates its own
        stop id through ``stop_token_ids`` and then allows *only* that id, while
        ``min_tokens`` bans the whole stop set.  No grammar, no vocabulary
        knowledge, empty on every model.
    ``grammar_x_allowlist_x_min_tokens``
        All three at once — strictly harder to satisfy than any pair.

    Two properties every recipe guarantees, both required by the oracle:

    * ``logprobs`` is set, because the sentinel predicate reads the sampled
      token's own logprob and the top-k alternatives beside it;
    * ``temperature = 0``, because the defect is ``argmax`` over an all-``-inf``
      row.  Above zero the row goes through ``softmax`` instead and produces
      NaN, which is a different path with a different signature.

    ``prompt_len`` / ``prefix_len`` are never touched, so the trace-validity
    invariant (``0 <= prefix_len < prompt_len``) is preserved by construction.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    ev = rng.choice(sends)
    recipe = rng.choice(MASK_COMPOSE_RECIPES)

    if recipe.startswith("grammar_"):
        ev.response_format = {
            "type": "json_schema",
            "json_schema": {"name": "mask_compose",
                            "schema": copy.deepcopy(_MASK_COMPOSE_SCHEMA)},
        }

    if recipe in ("grammar_x_min_tokens", "grammar_x_allowlist_x_min_tokens"):
        ev.min_tokens = rng.choice(_MASK_MIN_TOKENS)

    if recipe in ("grammar_x_allowlist", "grammar_x_allowlist_x_min_tokens"):
        if rng.random() < 0.5:
            width = rng.randint(1, 4)
            start = rng.randint(0, len(_SAFE_LOW_IDS) - width)
            ev.allowed_token_ids = list(_SAFE_LOW_IDS[start:start + width])
        else:
            ev.allowed_token_ids = _prompt_derived_ids(ev, rng,
                                                       rng.randint(1, 4))

    if recipe == "grammar_x_bad_words":
        # Single-character bad words ban unconditionally, and these are exactly
        # the characters a JSON grammar has no substitute for.
        ev.bad_words = list(rng.sample(_JSON_BAD_WORDS,
                                       rng.randint(1, len(_JSON_BAD_WORDS))))

    if recipe == "allowlist_x_min_tokens":
        # Provably empty: the only allowed id is one we have just added to the
        # stop set, and min_tokens bans the whole stop set below the floor.
        k = rng.choice(_SAFE_LOW_IDS)
        ev.allowed_token_ids = [k]
        ev.stop_token_ids    = [k]
        ev.min_tokens        = rng.choice(_MASK_MIN_TOKENS)

    # Optional decorations — a third writer on the same row, plus one control.
    if (ev.response_format is not None and not ev.bad_words
            and rng.random() < 0.25):
        ev.bad_words = list(rng.sample(_JSON_BAD_WORDS,
                                       rng.randint(1, len(_JSON_BAD_WORDS))))
    # logit_bias is the *negative control* of this surface: it is applied
    # additively (``builtin.py:161``) with no -inf special case anywhere in the
    # tree, and every HTTP path hard-clamps its values to +/-100
    # (``sampling_params.py:399``, ``:411``), so -100 lowers a logit and leaves
    # the token reachable.  A composition that only adds logit_bias must NOT
    # produce a degenerate row; if it ever does, that is itself the finding.
    # Kept at low probability because ``_validate_spec_decode``
    # (``sampling_params.py:887-899``) rejects logit_bias outright under
    # speculative decoding, which is a 400 rather than a result.
    if ev.allowed_token_ids and rng.random() < 0.15:
        ev.logit_bias = {str(i): -100.0 for i in ev.allowed_token_ids}

    # The oracle needs the top-k alternatives beside the sampled token.
    if not ev.logprobs or ev.logprobs < 5:
        ev.logprobs = 5
    ev.temperature = 0.0
    # A floor is only binding if the request is allowed to reach it.
    if ev.min_tokens:
        ev.max_tokens = max(ev.max_tokens or 0, ev.min_tokens + 32)
    return t


# ---------------------------------------------------------------------------
# Scheduler priority
# ---------------------------------------------------------------------------
#
# vLLM's waiting queue is ordered by ``(priority, arrival_time)`` and LOWER
# priority values are served FIRST.  Under the default ``fcfs`` policy the
# field is accepted and ignored; under ``--scheduling-policy priority`` it is
# the only thing that distinguishes the two policies.  Because ``Event`` could
# not carry a priority until now, every request in the
# ``--enable-priority-scheduling`` campaign went out at the default 0, the
# ``(priority, arrival_time)`` key degenerated to ``arrival_time``, and the
# priority queue executed a byte-identical schedule to FCFS.  401 corpus
# entries, zero findings, and nothing in them could ever have been a finding.
#
# The mutation that makes the knob live is not "set a priority" but "set
# priorities that CONTRADICT arrival order".  If priority ascends with arrival
# the correct schedule and the FCFS schedule coincide and the workload is once
# again blind to the bug.  Inverting them makes a correct scheduler visibly
# reorder and a broken one visibly not, which is a difference an oracle can
# see from completion order alone.

#: How tightly the SEND events are packed so they are co-resident in the
#: waiting queue.  Priority only decides anything among requests that are
#: waiting at the same time; requests that arrive after the queue has drained
#: are served in arrival order whatever their priority.
_PRIORITY_ARRIVAL_WINDOW_MS = 60

#: ``n`` values for the amplification cell.  ``n`` is the dimension that
#: produced finding V9, and it is placed on the *highest*-priority requests on
#: purpose: the scheduler is supposed to admit those first, so mis-ordering
#: them is maximally visible in the completion times.
_PRIORITY_N = (2, 3, 4)

#: ``prompt_logprobs`` values.  Capped well under ``max_logprobs`` (default 20,
#: config/model.py:242) so the request is never rejected for the cap.  vLLM
#: rejects ``prompt_logprobs`` on a streaming request outright
#: (chat_completion/protocol.py:805-810, completion/protocol.py:486+), so any
#: event that gets one is also forced to ``stream=False``.
_PRIORITY_PROMPT_LOGPROBS = (1, 2, 5)

#: ``n > 1`` is rejected outright under greedy sampling
#: (``_verify_greedy_sampling``, sampling_params.py:640-644), so the ``n`` cell
#: has to leave greedy decoding.  ``seed`` is set alongside it to keep the
#: request reproducible anyway -- that is the whole reason ``seed`` is now an
#: Event field.
_PRIORITY_TEMPERATURE = 0.7


def invert_priority_order(trace: Trace, rng: random.Random) -> Trace:
    """Assign scheduling priorities *inversely* to arrival order.

    The first request to arrive is given the numerically largest priority (the
    WORST), the last to arrive is given 0 (the BEST).  A server running
    ``--scheduling-policy priority`` must therefore serve the trace close to
    backwards; a server running ``fcfs`` -- or a priority queue with an
    ordering bug -- serves it forwards.  The two schedules differ in completion
    order for every pair of requests, which is the strongest signal this
    workload can carry.

    Three things are set beyond ``priority`` itself:

    * **Arrival packing.**  Priority only arbitrates between requests that are
      waiting *simultaneously*.  The SEND offsets are compressed into
      ``_PRIORITY_ARRIVAL_WINDOW_MS`` so the queue actually has a choice to
      make.  Only SEND offsets move and they only ever move EARLIER, so any
      CANCEL or DISCONNECT still lands after the request it refers to; out of
      caution the packing is skipped altogether on traces that contain one,
      since their timing is load-bearing in a way this mutator has no view of.
    * **Amplification on the winners.**  ``n`` (via ``n_completions``) or
      ``prompt_logprobs`` is set on the highest-priority third of the requests.
      Both multiply the work a single admitted request represents, so a
      scheduler that admits them in the wrong order is wrong by a much larger
      margin in wall-clock terms than the request count suggests.
    * **A seed.**  ``n > 1`` forces ``temperature > 0``; ``seed`` keeps the
      result reproducible so a replay of the same trace is comparable.

    Only one amplification cell is used per call, and the ``prompt_logprobs``
    cell leaves ``temperature`` alone, so roughly half of the traces this
    mutator produces are still greedy and still eligible for the determinism
    probe.

    ``prompt_len`` and ``prefix_len`` are never touched, so the trace-validity
    invariant (``0 <= prefix_len < prompt_len``) holds by construction.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    # One request cannot be scheduled out of order with respect to itself.
    if len(sends) < 2:
        return t

    arrival = sorted(sends, key=lambda e: e.offset_ms)

    has_lifecycle_events = any(
        ev.action in (ActionType.CANCEL, ActionType.DISCONNECT)
        for ev in t.events
    )
    if not has_lifecycle_events:
        t0 = arrival[0].offset_ms
        span = arrival[-1].offset_ms - t0
        if span > _PRIORITY_ARRIVAL_WINDOW_MS:
            scale = _PRIORITY_ARRIVAL_WINDOW_MS / span
            for ev in arrival:
                ev.offset_ms = t0 + int(round((ev.offset_ms - t0) * scale))

    # Inverted: earliest arrival gets the worst priority, latest gets the best.
    # Lower value = higher priority (vLLM's convention).
    n_sends = len(arrival)
    for i, ev in enumerate(arrival):
        ev.priority = n_sends - 1 - i

    # The best-priority third: priority 0, 1, ... — i.e. the tail of `arrival`.
    n_amplified = max(1, n_sends // 3)
    winners = arrival[-n_amplified:]

    if rng.random() < 0.5:
        # `n` cell.  Cannot be combined with greedy decoding.
        for ev in winners:
            ev.n_completions = rng.choice(_PRIORITY_N)
            ev.temperature = _PRIORITY_TEMPERATURE
            ev.seed = rng.randrange(1, 2 ** 31)
    else:
        # `prompt_logprobs` cell.  Streaming is a hard 400 with it, and it is
        # O(prompt_len * k) on both compute and response size, so the request
        # it lands on is genuinely more expensive than its neighbours.
        for ev in winners:
            ev.prompt_logprobs = rng.choice(_PRIORITY_PROMPT_LOGPROBS)
            ev.stream = False

    return t


# Registry of all vLLM-specific mutations

# ---------------------------------------------------------------------------
# Request-surface operators (ported for the FSE build from an earlier
# branch's API surfaces, commits deb4f649c / 66b12d5af): send a prompt as
# embeddings, send an invalid structured-output specification, and reset the
# prefix cache mid-trace.  Each is generic -- it picks its targets at random and
# encodes no particular bug's shape.
# ---------------------------------------------------------------------------

def inject_prompt_embeds(trace: Trace, rng: random.Random) -> Trace:
    """Send 1-3 completions requests as ``prompt_embeds`` instead of text.

    Chat requests are skipped (the field exists only on /v1/completions).  The
    executor turns the flag into a deterministic tensor of the request's
    prompt length; without ``--prompt-embeds-dim`` it is a no-op.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events
             if ev.action == ActionType.SEND and not ev.messages and ev.prompt_len]
    if not sends:
        return t
    for ev in rng.sample(sends, min(len(sends), rng.randint(1, 3))):
        ev.prompt_embeds = True
    return t


#: Invalid (or pathological) structured-output specifications, one per kind of
#: mistake a client can make: a non-type, a bad nested type, a dangling or
#: self-referential $ref, an empty enum, a non-regex pattern, an unbalanced
#: regex, an unsatisfiable repeat, an unclosed or undefined grammar rule.
_MALFORMED_JSON = [
    {"type": "not-a-type"},
    {"type": "object", "properties": {"a": {"type": "strng"}}, "required": ["a"]},
    {"type": "object", "properties": {"a": {"$ref": "#/definitions/missing"}}},
    {"type": "string", "enum": []},
    {"type": "string", "pattern": "(["},
    {"type": "object", "properties": {"next": {"$ref": "#"}}, "required": ["next"]},
]
_MALFORMED_REGEX = ["([a-z]+", "a{3,1}", "(?P<x>a)(?P<x>b)"]
_MALFORMED_GRAMMAR = ['root ::= "a" | ', "root ::= undefined_rule", 'root ::= ("a"']


def malformed_constraint(trace: Trace, rng: random.Random) -> Trace:
    """Give one request an invalid structured-output specification.

    A server must reject it with a 4xx naming the problem (or serve it); the
    error oracle flags any 5xx.  The event's other constraints are cleared so
    exactly one is on the wire.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    ev = rng.choice(sends)
    ev.response_format = ev.guided_json = ev.guided_regex = None
    ev.guided_grammar = ev.guided_choice = None
    kind = rng.choice(["response_format", "json", "regex", "grammar"])
    if kind == "response_format":
        ev.response_format = {"type": "json_schema", "json_schema": {
            "name": "out", "schema": copy.deepcopy(rng.choice(_MALFORMED_JSON))}}
    elif kind == "json":
        ev.guided_json = copy.deepcopy(rng.choice(_MALFORMED_JSON))
    elif kind == "regex":
        ev.guided_regex = rng.choice(_MALFORMED_REGEX)
    else:
        ev.guided_grammar = rng.choice(_MALFORMED_GRAMMAR)
    return t


def inject_cache_reset(trace: Trace, rng: random.Random) -> Trace:
    """Reset the prefix cache after the trace, then resend one of its prompts.

    Appends a RESET event after the last event plus a gap, and a copy of a
    random SEND (same prompt, fresh id, usage requested) shortly after it.  The
    reset oracle then knows what the cache must no longer serve.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND and ev.prompt_len]
    if not sends:
        return t
    src = rng.choice(sends)
    at = max(e.offset_ms for e in t.events) + rng.choice([2000, 4000, 8000])
    t.events.append(Event(action=ActionType.RESET, offset_ms=at,
                          request_id=f"reset{at}", origin="mutated"))
    again = copy.deepcopy(src)
    again.request_id = f"{src.request_id}_after_reset"
    again.offset_ms = at + rng.choice([0, 100, 500])
    again.include_usage = True
    again.cache_salt = None
    again.max_tokens = min(src.max_tokens or 8, 16)
    again.origin = "mutated"
    t.events.append(again)
    return t


#: Valid constraints a small base model satisfies unconstrained (the guard-4
#: design of ``seeds/guided_smoke/``), one per structured-output kind.
_VALID_JSON = [
    {"type": "object", "properties": {"name": {"type": "string"},
                                      "count": {"type": "integer"}},
     "required": ["name", "count"], "additionalProperties": False},
    {"type": "object", "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
     "required": ["answer"]},
]
_VALID_REGEX = [r"\d{4}-\d{2}-\d{2}", r"(yes|no)", r"[A-Z][a-z]{2,8}"]
_VALID_CHOICE = [["yes", "no"], ["red", "green", "blue"], ["positive", "negative", "neutral"]]


def inject_structured_output(trace: Trace, rng: random.Random) -> Trace:
    """Give 1-2 requests a valid structured-output constraint.

    One kind per request (json schema via ``guided_json`` or
    ``response_format``, regex, or choice); the schema oracle checks the output
    against it.  Other constraints on the event are cleared.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events if ev.action == ActionType.SEND]
    if not sends:
        return t
    for ev in rng.sample(sends, min(len(sends), rng.randint(1, 2))):
        ev.response_format = ev.guided_json = ev.guided_regex = None
        ev.guided_grammar = ev.guided_choice = None
        kind = rng.choice(["response_format", "json", "regex", "choice"])
        if kind == "response_format":
            ev.response_format = {"type": "json_schema", "json_schema": {
                "name": "out", "schema": copy.deepcopy(rng.choice(_VALID_JSON))}}
        elif kind == "json":
            ev.guided_json = copy.deepcopy(rng.choice(_VALID_JSON))
        elif kind == "regex":
            ev.guided_regex = rng.choice(_VALID_REGEX)
        else:
            ev.guided_choice = list(rng.choice(_VALID_CHOICE))
    return t

VLLM_MUTATIONS = [
    cuda_graph_thrash,
    mutate_sampling_params,
    inject_chat_turn,
    inject_unicode_stress,
    inject_multi_system_message,
    inject_rtl_system_message,
    retry_storm,
    inject_mixed_endpoints,
    inject_stop_sequences,
    inject_max_prefix,
    memory_pressure,
    chunk_boundary_align,
    kv_block_boundary,
    min_tokens_conflict,
    compose_mask_restrictors,
    invert_priority_order,
    inject_prompt_embeds,
    malformed_constraint,
    inject_cache_reset,
    inject_structured_output,
]
