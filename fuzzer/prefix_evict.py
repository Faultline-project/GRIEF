"""Targeted prefix-cache eviction: deletion lists, and the acceptance check.

Uses ``fuzzer.engine_ids.base_request_id`` and nothing else from ``fuzzer``,
and imports **no vLLM**: everything here is arithmetic over a descriptor-tap slice
and over raw ``/v1/completions`` response bodies, so it is usable from an
offline driver and from the unit tests alike.

THE PROBLEM
-----------
A trace observed once (the DISCOVERY pass) is not replayable, because running it
changes the prefix cache: the second run's requests hit blocks that the first
run's requests created.  ``POST /reset_prefix_cache`` is too blunt -- it also
destroys the *ancestral* state the discovery pass was observed against, so the
replay's admission profile (and therefore its chunked-prefill boundaries, and
therefore its arithmetic) is a different experiment.

TARGETED eviction restores the environment instead of clearing it:

  * for each request in the trace, everything the request itself computed is
    made unreachable;
  * everything the request only *read* -- the ancestral prefix it hit on -- is
    left in place.

The split point is recorded by the tap and needs no guessing.  At the step that
first admits a request, ``num_computed_tokens`` is exactly the number of prompt
tokens the prefix cache served (``KVCacheManager.get_computed_blocks`` returns
it, ``Scheduler`` sets it before the composition record is emitted).  Prefix hits
are whole blocks, so::

    keep_prefix_blocks = ancestral_hit_len // block_size

is exact, not a floor approximation -- ``ancestral_hit_len`` is already a
multiple of ``block_size``.  ``keep_prefix_blocks`` is nevertheless computed
with ``//`` and the non-alignment is reported (``hit_len_block_aligned``)
rather than asserted away, because a group with a coarser block size or a
future fine-grained partial hit would break the alignment silently.

WHAT THE ENGINE IS TOLD
-----------------------
Only ``token_ids`` and ``keep_prefix_blocks``.  The block-hash chain is computed
**engine-side** from those token ids with the engine's own hasher, so the client
never has to reproduce ``prefix_caching_hash_algo``, ``NONE_HASH`` seeding, the
chaining rule, or the extra-key rules.  See
the vLLM cache-evict patch notes (not included).

TOKEN SEQUENCE, NOT PROMPT
--------------------------
Blocks are cached for generated tokens too (``BlockPool.cache_full_blocks`` is
called as the request appends tokens), so the deletion list must cover
``prompt_token_ids + token_ids``.  A plan built from the prompt alone would
leave the decode-side blocks reachable and the next pass would hit them.

ONE EVICTION, FOUR STEPS
------------------------
Deriving the plan is only the first of them, and the other three live here too
so that one module answers "was the environment restored?":

  1. ``eviction_plan``        -- what to delete.
  2. ``wait_for_drain``       -- BEFORE deleting: the engine must be idle.
     Eviction skips every block a running request holds (``ref_cnt > 0``) and
     reports it as ``skipped_refcount`` rather than refusing, so an eviction
     issued while requests are in flight silently under-restores.  The gate is
     enforced, not advisory: poll the endpoint's own
     ``num_unfinished_requests`` under a bounded wait and raise ``DrainTimeout``
     naming the count.
  3. ``accept_eviction``      -- AFTER deleting: one authoritative accept/reject
     over the response's checkable bookkeeping, with a structured reason.
  4. ``restore_prefix_cache`` -- what to do when (3) rejects: nothing (the
     default, unchanged behaviour), or exactly one reset-and-rebuild retry.

Everything here is PRE-FLIGHT: it catches a bad restoration before the pass
runs.  It does not prove the replay then took the same admission path; that
evidence comes from the pass's own descriptor-tap slice (the composition-pinned
replay scheduler that checked it step by step on an earlier branch is not
part of this branch).
"""

from __future__ import annotations

import json
import re

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from fuzzer.engine_ids import base_request_id

__all__ = [
    "ACCEPTANCE_FAILURES",
    "ACCEPTANCE_WARNINGS",
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_DRAIN_POLL_S",
    "DEFAULT_DRAIN_TIMEOUT_S",
    "DRAIN_PROBE_LABEL",
    "FALLBACK_MODES",
    "FALLBACK_OFF",
    "FALLBACK_RESET_REBUILD",
    "DrainTimeout",
    "RestorationError",
    "RestorationFailed",
    "accept_eviction",
    "admission_hits",
    "check_hit_restoration",
    "drain_count",
    "drain_probe_body",
    "eviction_plan",
    "full_token_ids",
    "restore_prefix_cache",
    "summarize_evict_response",
    "wait_for_drain",
]

#: vLLM's ``CacheConfig.block_size`` default for a dense attention model.  The
#: probe reads the real value out of the eviction endpoint's response and
#: re-derives the plan if it disagrees; this is only the offline default.
DEFAULT_BLOCK_SIZE = 16


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def full_token_ids(choice: Mapping[str, Any]) -> list[int]:
    """``prompt_token_ids + token_ids`` from one ``/v1/completions`` choice.

    Both fields require ``return_token_ids: true`` on the request body (the
    zero-floor campaign body already sets it).  Raises rather than guessing if
    either is missing: a plan built from a truncated sequence would silently
    under-evict.
    """
    try:
        prompt = list(choice["prompt_token_ids"])
        out = list(choice["token_ids"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "choice is missing prompt_token_ids/token_ids; the recording must "
            "have been made with return_token_ids=true"
        ) from exc
    if not prompt:
        raise ValueError("empty prompt_token_ids")
    return prompt + out


def admission_hits(recs: Iterable[Mapping[str, Any]]) -> dict[str, dict]:
    """Per base request id, what the scheduler saw when it FIRST admitted it.

    ``hit_len`` is ``num_computed_tokens`` at that moment == the prefix-cache
    hit length in tokens (0 == full miss).  Same quantity as
    ``campaign_logs/prefix_state_2026-08-29/analyze_matched.py:admission_profile``,
    keyed by base request id instead of by prompt slot so it does not need the
    fixture.

    Only ``kind == "composition"`` records are read; the iteration order of
    ``recs`` is the tap order and is load-bearing (first occurrence wins).
    """
    out: dict[str, dict] = {}
    for rec in recs:
        if rec.get("kind") != "composition":
            continue
        for q in rec.get("requests") or ():
            base = base_request_id(q["request_id"])
            if base in out:
                continue
            out[base] = {
                "base": base,
                "engine_request_id": q["request_id"],
                "hit_len": int(q["num_computed_tokens"]),
                "num_prompt_tokens": int(q["num_prompt_tokens"]),
                "new": bool(q.get("new")),
                "tap_step_index": rec.get("step_index"),
            }
    for e in out.values():
        e["miss_len"] = e["num_prompt_tokens"] - e["hit_len"]
        e["full_miss"] = e["hit_len"] == 0
    return out


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

#: ``SpeculativeConfig.use_eagle()`` in vLLM 0.27.1: the methods whose drafter
#: reads the target's hidden state for the last cached token, for which the
#: full-attention manager prunes the last matching block of a prefix hit.
EAGLE_LIKE_METHODS = frozenset({"eagle", "eagle3", "mtp", "dflash", "dspark"})


def speculative_hit_shift_blocks(serve_args: Sequence[str]) -> int:
    """How many blocks the admission hit under-reports the cached prefix by.

    Reads ``--speculative-config <json>`` / ``--speculative-config=<json>`` from
    the extra serve args; 1 for an EAGLE-like method, else 0 (no speculation,
    ngram, draft models, unparseable config).
    """
    args = list(serve_args or ())
    values: list[str] = []
    for i, a in enumerate(args):
        if a == "--speculative-config":
            # JSON in the next token, or the cluster's ``key:value`` tokens
            # (``method:eagle3 model:... num_speculative_tokens:3``) up to
            # the next option.
            for b in args[i + 1:]:
                if b.startswith("--"):
                    break
                values.append(b)
        elif a.startswith("--speculative-config="):
            values.append(a.split("=", 1)[1])
    method = ""
    for v in values:
        try:
            cfg = json.loads(v)
        except (TypeError, ValueError):
            cfg = None
        if isinstance(cfg, dict):
            method = str(cfg.get("method") or "").lower()
            break
        m = re.match(r"^\s*method\s*[:=]\s*([A-Za-z0-9_]+)\s*$", v)
        if m:
            method = m.group(1).lower()
            break
    return 1 if method in EAGLE_LIKE_METHODS else 0


def eviction_plan(
    token_ids_by_base: Mapping[str, Sequence[int]],
    hits_by_base: Mapping[str, Mapping[str, Any]],
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    keep_override: Mapping[str, int] | None = None,
    order: Sequence[str] | None = None,
    hit_shift_blocks: int = 0,
) -> dict:
    """Build the ``POST /evict_prefix_cache_entries`` body plus its provenance.

    Args:
        token_ids_by_base: base request id -> full token sequence
            (``full_token_ids``).
        hits_by_base: base request id -> entry from ``admission_hits``.
        block_size: the engine's cache block size.
        keep_override: base -> forced ``keep_prefix_blocks``.  The G5 negative
            control sets a base that had ancestral hits to 0; nothing else
            should use it.
        order: explicit base ordering (defaults to ``token_ids_by_base`` order).
        hit_shift_blocks: blocks the engine's admission hit UNDER-reports the
            cached prefix by.  Under the speculative methods that need the
            target's hidden state for the last cached token (EAGLE, EAGLE3,
            MTP, DFlash, DSpark -- ``SpeculativeConfig.use_eagle``) vLLM's
            full-attention manager prunes the last matching block, so a
            request whose prompt shares nine cached blocks with its ancestor
            is admitted with ``num_computed_tokens`` = 8 blocks; a plan that
            kept 8 would evict the ancestor's ninth block and the next
            replicate would be admitted at 7 (the 2026-09-14 EAGLE3 / DFlash
            / DSpark hunts: every replicate after a restore one block
            shorter).  ``speculative_hit_shift_blocks`` derives it from the
            serve args.  Applied only when the hit is non-zero: a full miss
            says nothing about a pruned block.

    Returns:
        ``{"body": {...}, "entries": [...], "block_size": int}``.  ``entries``
        carries, per base, everything needed to explain the plan after the fact:
        the recorded hit length, whether it was block aligned, the keep count,
        where the keep count came from, and how many blocks the plan expects to
        make unreachable.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    bases = list(order) if order is not None else list(token_ids_by_base)
    keep_override = dict(keep_override or {})

    entries: list[dict] = []
    body_reqs: list[dict] = []
    for base in bases:
        toks = list(token_ids_by_base[base])
        hit = hits_by_base.get(base)
        if hit is None:
            raise ValueError(
                f"no admission record for {base!r}; the tap slice and the raw "
                "responses disagree about which requests ran"
            )
        hit_len = int(hit["hit_len"])
        derived_keep = hit_len // block_size
        if base in keep_override:
            keep = int(keep_override[base])
            source = "override"
        elif hit_shift_blocks and hit_len > 0:
            keep = derived_keep + int(hit_shift_blocks)
            source = "discovery_admission+pruned_block"
        else:
            keep = derived_keep
            source = "discovery_admission"
        n_hashes = len(toks) // block_size
        entries.append({
            "base": base,
            "n_tokens": len(toks),
            "num_prompt_tokens": int(hit["num_prompt_tokens"]),
            "ancestral_hit_len": hit_len,
            "hit_len_block_aligned": hit_len % block_size == 0,
            "keep_prefix_blocks": keep,
            "keep_source": source,
            "keep_derived_from_discovery": derived_keep,
            "hit_shift_blocks": int(hit_shift_blocks) if hit_len > 0 else 0,
            "n_full_blocks": n_hashes,
            "n_blocks_targeted": max(0, n_hashes - keep),
        })
        body_reqs.append({
            "label": base,
            "token_ids": toks,
            "keep_prefix_blocks": keep,
        })

    return {
        "block_size": block_size,
        "body": {"requests": body_reqs},
        "entries": entries,
    }


# --------------------------------------------------------------------------
# acceptance
# --------------------------------------------------------------------------

def check_hit_restoration(
    discovery_hits: Mapping[str, Mapping[str, Any]],
    replay_hits: Mapping[str, Mapping[str, Any]],
    *,
    bases: Sequence[str] | None = None,
) -> dict:
    """Did targeted eviction restore the discovery pass's admission profile?

    The contract is EQUALITY, per request, on the prefix-cache hit length --
    not "greater than zero" and not "zero".  A restored environment reproduces
    the discovery hit exactly: ancestral entries still serve their hit, and
    everything the trace computed for itself misses again.

    Returns a report with ``ok`` and a per-request expected/actual table.  It
    never raises on a mismatch -- reporting the mismatch loudly is the point
    (gate G5 asserts that a deliberately wrong plan lands here with ok=False).
    """
    names = list(bases) if bases is not None else sorted(
        set(discovery_hits) | set(replay_hits))
    rows = []
    ok = True
    for base in names:
        d = discovery_hits.get(base)
        a = replay_hits.get(base)
        exp = None if d is None else int(d["hit_len"])
        act = None if a is None else int(a["hit_len"])
        match = d is not None and a is not None and exp == act
        # a prompt-length change means the two passes did not send the same
        # request, which invalidates the comparison rather than failing it
        exp_p = None if d is None else int(d["num_prompt_tokens"])
        act_p = None if a is None else int(a["num_prompt_tokens"])
        same_prompt = exp_p == act_p
        ok &= bool(match and same_prompt)
        rows.append({
            "base": base,
            "expected_hit_len": exp,
            "actual_hit_len": act,
            "delta": None if (exp is None or act is None) else act - exp,
            "expected_num_prompt_tokens": exp_p,
            "actual_num_prompt_tokens": act_p,
            "same_prompt_length": same_prompt,
            "match": bool(match),
            "missing_in_discovery": d is None,
            "missing_in_replay": a is None,
        })
    return {
        "ok": bool(ok),
        "n_requests": len(rows),
        "n_matched": sum(1 for r in rows if r["match"]),
        "n_mismatched": sum(1 for r in rows if not r["match"]),
        "requests": rows,
    }


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------

def summarize_evict_response(resp: Mapping[str, Any]) -> dict:
    """Fold the endpoint's per-entry bookkeeping into checkable identities.

    Three counters must be kept apart, because a prefix-cache hash key does NOT
    map one-to-one onto a block.  ``BlockHashToBlockMap`` deliberately does not
    de-duplicate identical content (its NOTE #1), so successive requests that
    recompute the same tail land several distinct blocks under one key, and
    ``pop`` keeps the key alive while any block remains under it:

      ``evicted``             blocks made unreachable
      ``hash_keys_removed``   block -> key links removed  (>= evicted)
      ``table_keys_dropped``  keys that became EMPTY      (<= hash_keys_removed)

    Only the third can equal the table's shrink.  The identities returned:

      ``cache_entries_before - cache_entries_after == table_keys_dropped``
          the negative half of the no-over-reach claim: the table lost exactly
          the keys the endpoint says it dropped, and nothing else.
      ``kept_present == kept * num_kv_cache_groups``
          the positive half: every PRESERVED ancestral key is still resolvable.
          The engine counts ``kept_present`` once per (block, KV cache group)
          but reports ``kept`` as a block count, so the two are only directly
          comparable at one group -- which is every deployment this has run on,
          and is what the field measurement in the patch README shows
          (``num_kv_cache_groups: 1``, ``264 == 264``).  The group factor is
          applied rather than assumed so a multi-group engine reports a real
          verdict instead of a spurious failure.
      ``skipped_refcount == 0``
          nothing was left behind because a request still held it.

    A ``dry_run`` response satisfies the shrink identity trivially (0 == 0).
    """
    entries = list(resp.get("entries") or ())
    groups = int(resp.get("num_kv_cache_groups") or 1)

    def _s(field):
        return sum(e.get(field, 0) for e in entries)

    tot = {
        "hashes_computed": _s("hashes_computed"),
        "kept": _s("kept"),
        "kept_present": _s("kept_present"),
        "considered": _s("considered"),
        "found": _s("found"),
        "not_found": _s("not_found"),
        "blocks_found": _s("blocks_found"),
        "evicted": _s("evicted"),
        "skipped_refcount": _s("skipped_refcount"),
        "hash_keys_removed": _s("hash_keys_removed"),
        "table_keys_dropped": _s("table_keys_dropped"),
    }
    before = resp.get("cache_entries_before")
    after = resp.get("cache_entries_after")
    shrink = None if (before is None or after is None) else before - after
    free_before = resp.get("free_blocks_before")
    free_after = resp.get("free_blocks_after")
    return {
        "totals": tot,
        "num_kv_cache_groups": groups,
        "cache_entries_before": before,
        "cache_entries_after": after,
        "cache_entries_shrink": shrink,
        "free_blocks_before": free_before,
        "free_blocks_after": free_after,
        "blocks_per_found_key": (
            round(tot["blocks_found"] / tot["found"], 3) if tot["found"] else None),
        "kept_present_expected": tot["kept"] * groups,
        "shrink_equals_keys_dropped": shrink == tot["table_keys_dropped"],
        "all_kept_still_present": tot["kept_present"] == tot["kept"] * groups,
        "no_refcount_skips": tot["skipped_refcount"] == 0,
        "considered_accounts_for_all":
            tot["considered"] == tot["found"] + tot["not_found"],
        # eviction unlinks a block, it does not free it (patch README S6): the
        # block stays where it is in ``free_block_queue`` holding unreachable KV.
        "free_blocks_unchanged": (None if (free_before is None or free_after is None)
                                  else free_before == free_after),
    }


# --------------------------------------------------------------------------
# the drain gate (BEFORE the eviction)
# --------------------------------------------------------------------------

#: Bounded wait for the engine to go idle before a real eviction.  Seconds.
#: A pass that has just finished needs only the tail of its last step plus the
#: output-processor hop, so this is generous rather than tuned.
DEFAULT_DRAIN_TIMEOUT_S = 5.0

#: Gap between drain probes.  Each probe is one UTILITY RPC drained between two
#: engine steps, so polling faster than a step buys nothing.
DEFAULT_DRAIN_POLL_S = 0.25

#: Label carried by the probe body, so a probe is identifiable in the engine log
#: and can never be confused with a real plan entry.
DRAIN_PROBE_LABEL = "__drain_probe__"


class RestorationError(RuntimeError):
    """Base of the loud failures raised by the restoration path.

    Carries ``record``, the same structure the caller would have written to the
    run manifest, so an aborting driver can still record WHY it aborted.
    """

    def __init__(self, message: str, record: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.record: dict = dict(record or {})


class DrainTimeout(RestorationError):
    """The engine still had unfinished requests when the wait ran out."""


class RestorationFailed(RestorationError):
    """Acceptance rejected the eviction, and the fallback did not save it."""


def drain_probe_body() -> dict:
    """The cheapest legal ``/evict_prefix_cache_entries`` body, as a dry run.

    The endpoint refuses an empty ``requests`` list and an empty ``token_ids``
    (``cache_evict.py:_parse``), so the floor is one entry of one token.  One
    token is under a block, so the hash chain is empty: ``hashes_computed == 0``,
    nothing is considered, nothing is looked up, and with ``dry_run: true``
    nothing could be mutated even if it were.  What comes back that we want is
    ``num_unfinished_requests`` -- the engine's own
    ``scheduler.get_num_unfinished_requests()``, read on the EngineCore thread
    between two steps.

    Why not ``/metrics``: the Prometheus gauges are published by the FRONTEND
    from aggregated stats snapshots, so they lag the scheduler by up to a
    logging interval and are the wrong side of the process boundary.  Why not
    the real plan as a dry run: it hashes every token of every entry on the
    engine thread, which is the expensive thing to repeat under a poll.
    """
    return {
        "requests": [{"label": DRAIN_PROBE_LABEL, "token_ids": [0],
                      "keep_prefix_blocks": 0}],
        "dry_run": True,
    }


def drain_count(resp: Mapping[str, Any]) -> int:
    """``num_unfinished_requests`` out of an endpoint response, or raise.

    Raises rather than defaulting to 0: an engine whose response lacks the field
    is not one this gate can enforce anything against, and treating "absent" as
    "drained" is exactly the silent pass the gate exists to remove.
    """
    if not isinstance(resp, Mapping):
        raise ValueError(
            f"drain probe returned {type(resp).__name__}, not a JSON object; "
            "the eviction endpoint is not answering (dev mode off, or "
            "cache_evict_027 not applied?)")
    if "num_unfinished_requests" not in resp:
        raise ValueError(
            "drain probe response carries no num_unfinished_requests: "
            f"keys={sorted(resp)}; the drain gate cannot be enforced against "
            "this engine (is cache_evict_027 applied?)")
    return int(resp["num_unfinished_requests"])


def wait_for_drain(
    probe: Callable[[], Mapping[str, Any]],
    *,
    timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_DRAIN_POLL_S,
    sleep: Callable[[float], Any] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    what: str = "targeted eviction",
) -> dict:
    """Block until the engine reports zero unfinished requests, or raise.

    WHY THIS IS ENFORCED, NOT ADVISORY.  The endpoint deliberately does not
    refuse a busy engine: it skips every block held with ``ref_cnt > 0`` and
    counts it in ``skipped_refcount`` (patch README S6).  That is safe for the
    ENGINE and wrong for the EXPERIMENT -- the blocks a running request holds
    are exactly the ones the trace computed and the plan meant to delete, so an
    eviction issued mid-flight under-restores, the next pass hits blocks the
    recording missed on, and the only symptom is a hit-length mismatch a hundred
    steps later.

    ``probe`` is injected rather than a URL so this stays HTTP-free and
    unit-testable; the driver passes a closure that POSTs ``drain_probe_body()``.
    ``sleep``/``monotonic`` are injected for the same reason.

    Args:
        timeout_s: total budget.  ``0`` means "probe exactly once and require
            it to be drained"; the caller passes ``None``-shaped opt-outs by not
            calling this at all.
        poll_interval_s: gap between probes, clamped so a poll never overruns
            the budget.

    Returns:
        ``{"drained": True, "num_unfinished_requests": 0, "n_probes": ...,
        "waited_s": ..., "timeout_s": ..., "samples": [...]}``.

    Raises:
        DrainTimeout: naming the last count, the wait, and every sample.
        ValueError: the probe answered something this gate cannot read
            (``drain_count``).
    """
    if timeout_s < 0:
        raise ValueError(
            f"drain timeout must be >= 0, got {timeout_s}; disable the gate by "
            "not calling wait_for_drain, not by passing a negative budget")
    t0 = monotonic()
    samples: list[dict] = []
    while True:
        n = drain_count(probe())
        waited = monotonic() - t0
        samples.append({"t_s": round(waited, 4), "num_unfinished_requests": n})
        if n == 0:
            return {
                "drained": True,
                "num_unfinished_requests": 0,
                "n_probes": len(samples),
                "waited_s": round(waited, 4),
                "timeout_s": timeout_s,
                "samples": samples,
            }
        if waited >= timeout_s:
            rec = {
                "drained": False,
                "num_unfinished_requests": n,
                "n_probes": len(samples),
                "waited_s": round(waited, 4),
                "timeout_s": timeout_s,
                "samples": samples,
            }
            raise DrainTimeout(
                f"engine not drained before {what}: num_unfinished_requests={n} "
                f"after {waited:.2f}s and {len(samples)} probe(s) "
                f"(timeout {timeout_s}s).  Evicting now would skip every block "
                f"those requests hold (ref_cnt > 0), report them as "
                f"skipped_refcount, and silently under-restore the cache.",
                rec)
        sleep(min(poll_interval_s, max(0.0, timeout_s - waited)))


# --------------------------------------------------------------------------
# the acceptance check (AFTER the eviction)
# --------------------------------------------------------------------------

#: Every way ``accept_eviction`` can REJECT.  Exhaustive on purpose: a caller
#: switching on the reason must be able to enumerate what it will see, and a new
#: class must be added here (and to the tests) rather than appearing in the wild.
ACCEPTANCE_FAILURES = (
    "HTTP_STATUS",                  # transport/route said no
    "RESPONSE_NOT_OK",              # body is not the endpoint's success shape
    "NO_ENTRIES",                   # no per-entry bookkeeping => nothing checkable
    "DRY_RUN_FLAG_MISMATCH",        # engine ran a different mode than asked
    "DRY_RUN_MUTATED",              # a dry run changed the table
    "BLOCK_SIZE_DISAGREES",         # plan derived at a different block size
    "PLAN_ECHO_MISMATCH",           # entries do not echo the plan that was sent
    "ENGINE_NOT_DRAINED",           # requests in flight during the eviction
    "REFCOUNT_SKIPS",               # blocks left behind because they were held
    "KEPT_NOT_PRESENT",             # a PRESERVED ancestral entry is gone
    "SHRINK_IDENTITY_VIOLATED",     # table lost keys the endpoint did not report
    "CONSIDERED_IDENTITY_VIOLATED",  # considered != found + not_found
    "FREE_BLOCKS_CHANGED",          # eviction must not free blocks
)

#: Recorded, never rejected.  Each is a real observation that does NOT by itself
#: mean the environment is unrestored.
ACCEPTANCE_WARNINGS = (
    "DRY_RUN",                # nothing was mutated, so nothing was restored
    "DRAIN_COUNT_ABSENT",     # engine did not report num_unfinished_requests
    "CHAIN_NOT_FOUND",        # an entry matched no cached block at all
    "HIT_NOT_BLOCK_ALIGNED",  # keep count was floored (plan-side)
    "HASH_COUNT_DIFFERS",     # engine hashed a different number of blocks
)


def accept_eviction(
    resp: Mapping[str, Any],
    *,
    http_status: int = 200,
    dry_run: bool = False,
    plan: Mapping[str, Any] | None = None,
    block_size: int | None = None,
    require_drained: bool = True,
) -> dict:
    """The single authoritative accept/reject over one eviction response.

    Consolidates what used to be spread over ``summarize_evict_response`` (the
    arithmetic) and the driver's inline ``rec["ok"] = ...`` (the policy), so
    there is exactly one definition of "the environment was restored" and it
    comes with a machine-readable reason instead of a bare False.

    The three the caller asked about, in the endpoint's own vocabulary:

      ``skipped_refcount > 0``      blocks the plan targeted were left cached
          because a request still held them.  The eviction is INCOMPLETE: the
          next pass will hit exactly those blocks.
      ``kept_present != kept``      an ancestral entry the plan promised to
          PRESERVE is no longer resolvable (scaled by
          ``num_kv_cache_groups``, see ``summarize_evict_response``).  The
          eviction OVER-REACHED, or something else evicted underneath it.
      identity violated             ``cache_entries_before - after !=
          table_keys_dropped``, ``considered != found + not_found``, or
          ``free_blocks_before != free_blocks_after``.  The bookkeeping does not
          describe the mutation that happened, so no claim about the cache can
          be made from it at all.

    Args:
        resp: the parsed response body.
        http_status: the POST's status code.
        dry_run: what the caller asked for.  A dry run can still be ACCEPTED --
            its identities are checkable -- but it is flagged as restoring
            nothing (``restores_state`` False), so a caller cannot mistake a
            rehearsal for a restoration.
        plan: the ``eviction_plan`` result that produced the body.  When given,
            the response entries must echo it (count, labels, keep counts).
        block_size: expected engine block size, if no plan is passed.
        require_drained: reject a response whose ``num_unfinished_requests`` is
            non-zero.  This is the POST-HOC half of the drain gate: the gate
            polls before the call, this catches an arrival between the two.

    Returns:
        ``{"ok", "reason", "reasons", "failures", "warnings", "checks",
        "bookkeeping", "dry_run", "restores_state", ...}``.  Never raises on a
        rejection -- reporting it is the point.
    """
    failures: list[dict] = []
    warnings: list[dict] = []

    def _fail(code: str, message: str, **detail):
        assert code in ACCEPTANCE_FAILURES, code   # noqa: S101 - developer error
        failures.append({"code": code, "message": message, "detail": detail})

    def _warn(code: str, message: str, **detail):
        assert code in ACCEPTANCE_WARNINGS, code   # noqa: S101
        warnings.append({"code": code, "message": message, "detail": detail})

    if http_status != 200:
        body = resp if isinstance(resp, Mapping) else {"raw": repr(resp)[:500]}
        _fail("HTTP_STATUS",
              f"POST /evict_prefix_cache_entries returned HTTP {http_status}",
              http_status=http_status, body=dict(body))
    if not isinstance(resp, Mapping):
        _fail("RESPONSE_NOT_OK",
              f"response is {type(resp).__name__}, not a JSON object")
        resp = {}
    elif resp.get("ok") is not True:
        _fail("RESPONSE_NOT_OK",
              f"response ok={resp.get('ok')!r}; the endpoint did not report "
              "success", body={k: v for k, v in resp.items() if k != "entries"})

    entries = list(resp.get("entries") or ())
    book = summarize_evict_response(resp)
    tot = book["totals"]
    if not entries:
        _fail("NO_ENTRIES",
              "response carries no per-entry bookkeeping, so no identity is "
              "checkable and no claim about the cache can be made")

    engine_dry = bool(resp.get("dry_run", False))
    if entries and engine_dry != bool(dry_run):
        _fail("DRY_RUN_FLAG_MISMATCH",
              f"asked dry_run={bool(dry_run)}, engine ran dry_run={engine_dry}",
              requested=bool(dry_run), engine=engine_dry)
    if dry_run:
        _warn("DRY_RUN",
              "dry run: the plan was computed and nothing was mutated, so the "
              "cache is exactly as unrestored as before the call")
        if book["cache_entries_shrink"]:
            _fail("DRY_RUN_MUTATED",
                  f"dry run changed the table by "
                  f"{book['cache_entries_shrink']} entr(ies)",
                  shrink=book["cache_entries_shrink"])

    want_bs = block_size if block_size is not None else (
        None if plan is None else plan.get("block_size"))
    engine_bs = resp.get("block_size")
    if want_bs is not None and engine_bs is not None and engine_bs != want_bs:
        _fail("BLOCK_SIZE_DISAGREES",
              f"plan derived at block_size={want_bs}, engine reports "
              f"{engine_bs}; every keep_prefix_blocks in the plan means a "
              "different number of tokens than intended",
              plan=want_bs, engine=engine_bs)

    if plan is not None and entries:
        pents = list(plan.get("entries") or ())
        echo_diffs: list[dict] = []
        if len(pents) != len(entries):
            echo_diffs.append({"field": "n_entries", "plan": len(pents),
                               "engine": len(entries)})
        for pe, ee in zip(pents, entries):
            if ee.get("label") != pe.get("base"):
                echo_diffs.append({"field": "label", "index": ee.get("index"),
                                   "plan": pe.get("base"),
                                   "engine": ee.get("label")})
            if ee.get("keep_prefix_blocks") != pe.get("keep_prefix_blocks"):
                echo_diffs.append({"field": "keep_prefix_blocks",
                                   "label": pe.get("base"),
                                   "plan": pe.get("keep_prefix_blocks"),
                                   "engine": ee.get("keep_prefix_blocks")})
            if (pe.get("n_full_blocks") is not None
                    and ee.get("hashes_computed") is not None
                    and pe["n_full_blocks"] != ee["hashes_computed"]):
                _warn("HASH_COUNT_DIFFERS",
                      f"{pe.get('base')}: plan expected "
                      f"{pe['n_full_blocks']} full block(s), engine hashed "
                      f"{ee['hashes_computed']} (hash_block_size != block_size?)",
                      label=pe.get("base"), plan=pe["n_full_blocks"],
                      engine=ee["hashes_computed"])
            if pe.get("hit_len_block_aligned") is False:
                _warn("HIT_NOT_BLOCK_ALIGNED",
                      f"{pe.get('base')}: recorded hit length "
                      f"{pe.get('ancestral_hit_len')} is not a whole number of "
                      "blocks, so keep_prefix_blocks was floored",
                      label=pe.get("base"),
                      ancestral_hit_len=pe.get("ancestral_hit_len"))
        if echo_diffs:
            _fail("PLAN_ECHO_MISMATCH",
                  f"{len(echo_diffs)} response entr(ies) do not echo the plan "
                  "that was sent; the engine acted on something else",
                  diffs=echo_diffs)

    n_unfinished = resp.get("num_unfinished_requests")
    if n_unfinished is None:
        if entries:
            _warn("DRAIN_COUNT_ABSENT",
                  "response carries no num_unfinished_requests; the post-hoc "
                  "half of the drain gate could not be evaluated")
    elif require_drained and int(n_unfinished) > 0:
        _fail("ENGINE_NOT_DRAINED",
              f"{int(n_unfinished)} request(s) were still unfinished during the "
              "eviction; blocks they hold are skipped, not evicted",
              num_unfinished_requests=int(n_unfinished))

    if entries and tot["skipped_refcount"] > 0:
        held = [{"label": e.get("label"), "skipped_refcount": e["skipped_refcount"]}
                for e in entries if e.get("skipped_refcount")]
        _fail("REFCOUNT_SKIPS",
              f"{tot['skipped_refcount']} block(s) the plan targeted were left "
              "cached because a request still held them (ref_cnt > 0); the next "
              "pass will hit exactly those blocks",
              total=tot["skipped_refcount"], entries=held)

    if entries and not book["all_kept_still_present"]:
        _fail("KEPT_NOT_PRESENT",
              f"kept_present={tot['kept_present']} but "
              f"{tot['kept']} kept block(s) x {book['num_kv_cache_groups']} "
              f"group(s) = {book['kept_present_expected']} were promised: an "
              "ancestral entry the plan preserves is no longer resolvable",
              kept=tot["kept"], kept_present=tot["kept_present"],
              expected=book["kept_present_expected"],
              num_kv_cache_groups=book["num_kv_cache_groups"])

    if entries and not dry_run and not book["shrink_equals_keys_dropped"]:
        _fail("SHRINK_IDENTITY_VIOLATED",
              f"table shrank by {book['cache_entries_shrink']} but the endpoint "
              f"reports {tot['table_keys_dropped']} key(s) dropped; the "
              "bookkeeping does not describe the mutation that happened",
              shrink=book["cache_entries_shrink"],
              table_keys_dropped=tot["table_keys_dropped"])

    if entries and not book["considered_accounts_for_all"]:
        _fail("CONSIDERED_IDENTITY_VIOLATED",
              f"considered={tot['considered']} != found={tot['found']} + "
              f"not_found={tot['not_found']}",
              considered=tot["considered"], found=tot["found"],
              not_found=tot["not_found"])

    if entries and book["free_blocks_unchanged"] is False:
        _fail("FREE_BLOCKS_CHANGED",
              f"free blocks moved {book['free_blocks_before']} -> "
              f"{book['free_blocks_after']}; eviction unlinks a block, it does "
              "not free it, so something other than this call touched the pool",
              before=book["free_blocks_before"], after=book["free_blocks_after"])

    for e in entries:
        if e.get("considered") and not e.get("found"):
            _warn("CHAIN_NOT_FOUND",
                  f"{e.get('label')}: {e['considered']} hash(es) considered and "
                  "none resolved to a cached block -- the chain was already "
                  "absent (LRU pressure), or is not reproducible from token ids "
                  "(LoRA / multi-modal / cache_salt, patch README S5)",
                  label=e.get("label"), considered=e["considered"])

    codes = [f["code"] for f in failures]
    return {
        "ok": not failures,
        "reason": codes[0] if codes else None,
        "reasons": codes,
        "failures": failures,
        "warnings": warnings,
        "warning_codes": [w["code"] for w in warnings],
        "http_status": http_status,
        "dry_run": bool(dry_run),
        "restores_state": not dry_run,
        "num_unfinished_requests": n_unfinished,
        "checks": {
            "http_ok": http_status == 200,
            "response_ok": resp.get("ok") is True,
            "has_entries": bool(entries),
            "block_size_agrees": not any(c == "BLOCK_SIZE_DISAGREES" for c in codes),
            "plan_echoed": not any(c == "PLAN_ECHO_MISMATCH" for c in codes),
            "engine_drained": not any(c == "ENGINE_NOT_DRAINED" for c in codes),
            "no_refcount_skips": book["no_refcount_skips"],
            "all_kept_still_present": book["all_kept_still_present"],
            "shrink_equals_keys_dropped": book["shrink_equals_keys_dropped"],
            "considered_accounts_for_all": book["considered_accounts_for_all"],
            "free_blocks_unchanged": book["free_blocks_unchanged"],
        },
        "bookkeeping": book if entries else None,
    }


# --------------------------------------------------------------------------
# the fallback (WHEN the acceptance rejects)
# --------------------------------------------------------------------------

#: Do nothing on a rejected restoration.  The DEFAULT: the rejection is
#: recorded, the pass runs anyway, and the gated replay's DESYNC echo is left as
#: the arbiter -- exactly the behaviour that existed before the fallback.
FALLBACK_OFF = "off"

#: On a rejected restoration: ``POST /reset_prefix_cache``, replay the prep
#: traffic, then re-derive and re-verify the eviction ONCE.  Trades the surgical
#: guarantee for a reproducible cold start plus a rebuild, and says so in the
#: manifest.
FALLBACK_RESET_REBUILD = "reset-rebuild"

FALLBACK_MODES = (FALLBACK_OFF, FALLBACK_RESET_REBUILD)


def _attempt_verdict(rec: Mapping[str, Any], attempt: int) -> dict:
    """Pull the acceptance out of one attempt record, or say why it can't."""
    if not isinstance(rec, Mapping) or "acceptance" not in rec:
        raise ValueError(
            f"eviction attempt {attempt} returned no 'acceptance' key "
            f"({sorted(rec) if isinstance(rec, Mapping) else type(rec).__name__}); "
            "restore_prefix_cache decides on accept_eviction's verdict and "
            "will not guess one")
    acc = rec["acceptance"]
    if "ok" not in acc:
        raise ValueError(f"eviction attempt {attempt} acceptance has no 'ok'")
    return acc


def restore_prefix_cache(
    evict: Callable[[int], Mapping[str, Any]],
    *,
    fallback: str = FALLBACK_OFF,
    reset: Callable[[], Mapping[str, Any]] | None = None,
    rebuild: Callable[[], Mapping[str, Any]] | None = None,
    log: Callable[[str], Any] | None = None,
) -> dict:
    """Restore the cache surgically; optionally, once, the blunt way instead.

    The state machine, in full::

        evict(0) -> accepted                       -> restored_surgically
        evict(0) -> rejected, fallback off         -> ok=False, run continues
        evict(0) -> rejected -> reset -> rebuild
                 -> evict(1) -> accepted           -> ok, used_fallback
                 -> evict(1) -> rejected           -> raise RestorationFailed

    ONE retry, never two: a second rejection after a full cache reset means the
    plan itself does not describe the traffic, and looping would only burn GPU
    hours producing the same reason.

    The retry is NOT a re-POST of the same plan.  ``evict`` is called again from
    scratch, so the driver re-derives the plan against the rebuilt cache; only
    then is it re-verified.  The drain gate lives inside ``evict``, so it is
    re-run too, and a ``DrainTimeout`` from either attempt propagates untouched
    (a reset cannot help an engine that will not go idle).

    THE FALLBACK NEVER MASKS THE FAILURE.  Both attempt records are returned,
    the trigger reason is recorded verbatim, and ``restored_surgically`` is the
    field that distinguishes a run that was restored from one that was rebuilt.

    Args:
        evict: ``attempt_index -> record``.  The record must carry
            ``["acceptance"]["ok"]`` (an ``accept_eviction`` result).
        fallback: one of ``FALLBACK_MODES``.
        reset: called on a rejection, before ``rebuild``.  POSTs
            ``/reset_prefix_cache`` (dev-mode gated, like the evict endpoint).
        rebuild: called after ``reset``.  Replays the prep traffic that seeded
            the ancestral state, so the re-derived plan has something to keep.
        log: optional line sink for the loud running commentary.

    Raises:
        ValueError: unknown mode, or reset-rebuild without both callables.
        RestorationFailed: the retry was rejected too.
    """
    if fallback not in FALLBACK_MODES:
        raise ValueError(
            f"unknown fallback mode {fallback!r}; expected one of "
            f"{list(FALLBACK_MODES)}")
    if fallback == FALLBACK_RESET_REBUILD and (reset is None or rebuild is None):
        raise ValueError(
            "fallback=reset-rebuild needs both a reset and a rebuild callable; "
            "resetting the cache without replaying the prep traffic would leave "
            "the ancestral blocks the plan preserves permanently absent")

    def _say(msg: str) -> None:
        if log is not None:
            log(msg)

    attempts: list[Mapping[str, Any]] = []
    rec0 = evict(0)
    attempts.append(rec0)
    acc0 = _attempt_verdict(rec0, 0)
    out: dict = {
        "fallback_mode": fallback,
        "attempts": attempts,
        "n_attempts": 1,
        "first_failure": acc0.get("reason"),
        "final_failure": None if acc0["ok"] else acc0.get("reason"),
        "first_failure_detail": None if acc0["ok"] else acc0.get("failures"),
        "fallback": None,
        "ok": bool(acc0["ok"]),
        "used_fallback": False,
        "restored_surgically": bool(acc0["ok"]),
    }
    if acc0["ok"]:
        _say("[restore] surgical restoration ACCEPTED"
             + (f" (warnings: {acc0['warning_codes']})"
                if acc0.get("warning_codes") else ""))
        return out

    reasons = ", ".join(acc0.get("reasons") or ["?"])
    _say(f"[restore] surgical restoration REJECTED: {reasons}")
    for f in acc0.get("failures") or ():
        _say(f"[restore]   {f['code']}: {f['message']}")
    if fallback == FALLBACK_OFF:
        _say("[restore] fallback=off: leaving the cache as it is; the gated "
             "pass's prefix_cache_hit echo remains the arbiter")
        return out

    _say(f"[restore] fallback={fallback}: POST /reset_prefix_cache, replay the "
         "prep traffic, re-derive and re-verify ONCE")
    reset_rec = dict(reset())
    _say(f"[restore] reset: {reset_rec}")
    rebuild_rec = dict(rebuild())
    _say(f"[restore] rebuild: {rebuild_rec.get('summary', rebuild_rec)}")

    rec1 = evict(1)
    attempts.append(rec1)
    acc1 = _attempt_verdict(rec1, 1)
    out.update({
        "n_attempts": 2,
        "ok": bool(acc1["ok"]),
        "used_fallback": True,
        "restored_surgically": False,
        "final_failure": None if acc1["ok"] else acc1.get("reason"),
        "fallback": {
            "triggered": True,
            "mode": fallback,
            "trigger_reason": acc0.get("reason"),
            "trigger_reasons": acc0.get("reasons"),
            "trigger_failures": acc0.get("failures"),
            "reset": reset_rec,
            "rebuild": rebuild_rec,
            "retry_ok": bool(acc1["ok"]),
            "retry_reason": None if acc1["ok"] else acc1.get("reason"),
            "retry_failures": None if acc1["ok"] else acc1.get("failures"),
        },
    })
    if acc1["ok"]:
        _say("[restore] retry after reset+rebuild ACCEPTED -- this run was "
             "REBUILT, not restored surgically")
        return out

    raise RestorationFailed(
        f"prefix-cache restoration failed twice: first attempt rejected "
        f"({reasons}), and after POST /reset_prefix_cache + a prep-traffic "
        f"rebuild the re-derived plan was rejected again "
        f"({', '.join(acc1.get('reasons') or ['?'])}).  A second reset would "
        "not change the answer: the plan does not describe the traffic that is "
        "actually running.",
        out)
