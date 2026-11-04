"""Engine request id -> caller base id.  ONE implementation, shared.

Why this module exists
----------------------
vLLM's OpenAI layer does not hand the engine the caller's ``X-Request-Id``
verbatim: it wraps it.  Every consumer that has to align two executions of the
same trace -- the batch-replay script/echo (``fuzzer/batch_replay/script.py``),
the R3 descriptor-contract checker (``fuzzer/contracts.py``), the slice
certificates (``fuzzer/slice_cert.py``), the prefix-cache evictor
(``fuzzer/prefix_evict.py``) -- must undo that wrapping the same way, or the
same request looks like two different requests to two different tools.

Until 2026-08-30 there were two copies of the regex (``script.py`` and
``contracts.py``), both anchored on the ``/v1/completions`` shape
``<api>-<base>-<prompt_idx>[-<8hex>]``.  vLLM 0.27.1's
``/v1/chat/completions`` emits ``chatcmpl-<base>-<8hex>`` -- no prompt index,
because a chat request carries exactly one prompt -- so both copies FAILED to
match a chat id and returned the whole engine id unchanged.  A chat request
therefore looked like a fresh, unique id in every execution, and every
cross-run comparison silently skipped it.  The defect was found by
``campaign_logs/battle1_recon_2026-08-30`` against sixteen live tap files and
is fixed here, in one place, for all callers.

Pure stdlib.  Imports no vLLM (it is used inside the EngineCore subprocess) and
nothing else from ``fuzzer``.

OBSERVED SHAPES (vLLM 0.27.1, counted over
``campaign_logs/r3_capture_2026-08-25/*.jsonl``)
------------------------------------------------------------------------
  ``cmpl-<base>-<prompt_idx>-<8hex>``  /v1/completions          (~120k records)
  ``chatcmpl-<base>-<8hex>``           /v1/chat/completions     (~17k records)
  ``<api>-<base>-<prompt_idx>``        no per-instance tail     (older builds)
  ``_warmup_3_``                       engine-internal, unwrapped (~8k records)

The ``<8hex>`` tail is a per-request-instance random suffix: it differs between
two executions of the same trace, which is exactly why it has to be stripped.
The ``<prompt_idx>`` is the index of the prompt inside a batched
``/v1/completions`` request (always ``0`` for the single-prompt traces this
repo sends, but part of the identity, not of the base).

DISAMBIGUATION RULE (the base may itself contain ``-``, digits, and hex)
-----------------------------------------------------------------------
1. Shapes are tried MOST SPECIFIC FIRST: ``-<idx>-<8hex>``, then ``-<8hex>``,
   then ``-<idx>``.  Order is load bearing.  ``cmpl-zf-r7-0-12345678`` has an
   all-digit tail that is also valid hex; trying the two-field shape first
   keeps the base ``zf-r7`` instead of swallowing ``-0`` into it.
2. Within a shape the base is NON-GREEDY and the suffix is anchored at the end
   of the string, so the base is the SHORTEST prefix for which the rest of the
   id is *exactly* the wrapper suffix.  A base that itself ends in something
   tail-shaped is therefore NOT over-stripped, because the regex has to consume
   the whole remainder:

       cmpl-r1-deadbeef-0-ab12cd34   -> r1-deadbeef   (not r1)
       chatcmpl-r1-ab12cd34-de67ff90 -> r1-ab12cd34   (not r1)
       cmpl-ff5c1947-0-a7a97413      -> ff5c1947      (base IS 8 hex)
       cmpl-zf-r7-0-b7018ac5         -> zf-r7         (base contains '-')

3. An id matching no shape is returned unchanged.  It is still stable within
   one engine process (``_warmup_3_``), so it can safely be used as its own
   key; it simply carries no caller identity.

RESIDUAL AMBIGUITY, on purpose
------------------------------
The shape table is prefix-agnostic: ``chatcmpl``/``cmpl``/``resp`` all go
through the same three shapes.  Consequently a CHAT request whose caller id
ends in ``-<digits>`` (``X-Request-Id: r05-2`` -> ``chatcmpl-r05-2-a09bffdd``)
is read by shape 1 as base ``r05`` + prompt index 2, not as base ``r05-2``.
Making the table prefix-aware (chat has no prompt index in 0.27.1) would
resolve that one case, and is deliberately NOT done: the prefix-agnostic table
is the one validated against the live captures and the one that keeps working
if a future vLLM does emit a chat prompt index.  The practical rule for
callers is: do not end a request id with ``-<digits>``.  The fuzzer's own
generated ids (``r01_retry_s``, ``fill_0``, ``thrash_5_fork2``, ``c04_b_s_b_s``)
never do.
"""

from __future__ import annotations

import re

__all__ = [
    "ENGINE_RID_SHAPES",
    "REPLAY_MARKER",
    "base_request_id",
    "normalized_request_key",
    "replay_marker_of",
    "split_engine_request_id",
    "split_replay_marker",
    "with_replay_marker",
]

#: The OpenAI-layer wrappers this repo has observed, MOST SPECIFIC FIRST.
#: Every pattern exposes ``base``; ``idx`` is present only where the shape
#: carries a prompt index.
ENGINE_RID_SHAPES: tuple[re.Pattern[str], ...] = (
    # /v1/completions with a per-instance tail: cmpl-r01-0-8d6ac621
    re.compile(r"^(?:chatcmpl|cmpl|resp)-(?P<base>.+?)-(?P<idx>\d+)-[0-9a-f]{8}$"),
    # /v1/chat/completions: chatcmpl-c01_b_s_s-a09bffdd  (NO prompt index)
    re.compile(r"^(?:chatcmpl|cmpl|resp)-(?P<base>.+?)-[0-9a-f]{8}$"),
    # no per-instance tail: cmpl-r01-0
    re.compile(r"^(?:chatcmpl|cmpl|resp)-(?P<base>.+?)-(?P<idx>\d+)$"),
)


#: THE REPLAY MARKER (2026-09-04).  Two replays of one trace send the same
#: caller ids, so the engine ids they produce differ only in the random
#: ``<8hex>`` tail, which every consumer here strips.  Without a marker the
#: replays would therefore be indistinguishable in a descriptor-tap stream.  A
#: replay appends ``~r<k>`` to the caller id it sends
#: (``X-Request-Id: kvp-p7~r2``).  Everything in this repo strips the marker
#: when it recovers the caller's base (so units, scripts, echoes and
#: certificates are unchanged by it), and :func:`replay_marker_of` recovers
#: ``k``.  ``~`` is chosen because it is legal in a request id, never appears
#: in a fuzzer-generated id, and is not ``-<digits>``, which the shape table
#: would read as a prompt index.
REPLAY_MARKER: re.Pattern[str] = re.compile(r"~r(?P<k>\d+)$")

#: The endpoint prefixes vLLM's OpenAI layer glues onto a caller id (a
#: response ``id`` is ``<prefix><caller id>`` with no tail at all).
_API_PREFIXES: tuple[str, ...] = ("chatcmpl-", "cmpl-", "resp-")


def with_replay_marker(base: str, k: int) -> str:
    """``("kvp-p7", 2)`` -> ``"kvp-p7~r2"``: the caller id replay ``k`` sends."""
    return f"{base}~r{int(k)}"


def split_replay_marker(s: str) -> tuple[str, int | None]:
    """``"kvp-p7~r2"`` -> ``("kvp-p7", 2)``; an unmarked id -> ``(s, None)``."""
    m = REPLAY_MARKER.search(s)
    if not m:
        return s, None
    return s[:m.start()], int(m.group("k"))


def replay_marker_of(rid: str) -> int | None:
    """The replay marker carried by a caller id, a response id or an engine id.

    ``"kvp-p7~r2"`` / ``"cmpl-kvp-p7~r2"`` / ``"cmpl-kvp-p7~r2-0-ab12cd34"`` ->
    ``2``; anything without a marker -> ``None``.
    """
    for pat in ENGINE_RID_SHAPES:
        m = pat.match(rid)
        if m:
            return split_replay_marker(m.group("base"))[1]
    for p in _API_PREFIXES:
        if rid.startswith(p):
            return split_replay_marker(rid[len(p):])[1]
    return split_replay_marker(rid)[1]


def split_engine_request_id(rid: str) -> tuple[str, str | None] | None:
    """``rid`` -> ``(base, prompt_idx or None)``, or ``None`` if unwrapped.

    ``None`` for the prompt index means the shape does not carry one, which in
    0.27.1 means the request came through ``/v1/chat/completions`` -- a chat
    request has exactly one prompt, so the caller that needs an index may read
    the missing one as ``0`` (see :func:`normalized_request_key`).
    """
    for pat in ENGINE_RID_SHAPES:
        m = pat.match(rid)
        if m:
            return split_replay_marker(m.group("base"))[0], m.groupdict().get("idx")
    return None


def base_request_id(rid: str) -> str:
    """Engine request id -> caller base id (the ``X-Request-Id`` value).

    ``cmpl-zf-r7-0-b7018ac5`` -> ``zf-r7``;
    ``chatcmpl-c01_b_s_s-a09bffdd`` -> ``c01_b_s_s``;
    ``cmpl-kvp-p7~r2-0-ab12cd34`` -> ``kvp-p7`` (the replay marker is stripped).
    Ids that match no wrapper shape are returned unchanged -- marker included,
    because such an id carries no caller identity to recover.
    """
    split = split_engine_request_id(rid)
    return split[0] if split else rid


def normalized_request_key(rid: str) -> str:
    """Engine request id -> ``"<base>#<prompt_idx>"`` cross-run alignment key.

    The ``'#'`` is a marker, not decoration: a report can count how many keys
    were recovered by normalization (i.e. are the caller's own trace requests)
    versus how many aligned only because the engine happened to reuse an
    internal id such as ``_warmup_3_``.  Ids that match no wrapper shape are
    returned unchanged and therefore carry no ``'#'``.

    A chat id has no prompt index; ``0`` is used, which is exact rather than a
    placeholder -- a chat request carries exactly one prompt.
    """
    split = split_engine_request_id(rid)
    if not split:
        return rid
    base, idx = split
    return f"{base}#{idx if idx is not None else 0}"
