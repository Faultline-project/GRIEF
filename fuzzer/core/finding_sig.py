"""
Finding signatures: mechanism-level dedup keys for findings.

The current in-run dedup key (``_finding_sig`` in ``core/runner.py``) hashes the
exact ``(action, adapter_name, prompt_len, max_tokens, prefix_len)`` tuple of
every event, so a single ``scale_prompt_lengths`` mutation makes a
re-discovery of the same bug look new.  This module implements the replacement:

  * ``oracle_subtype``  — a small ordinal per (bug_class, detail) that separates
    genuinely different mechanisms hiding inside one ``BugClass``.
  * ``workload_shape``  — five booleans/counts describing the *request features*
    of a trace, with every raw numeric (offsets, prompt_len, max_tokens,
    prefix_len, event count, event order) deliberately dropped.
  * ``mechanism_signature`` — the in-run key, over live ``Trace`` /
    ``OracleResult`` objects.
  * ``mechanism_signature_from_dict`` / ``triage_signature`` — the same key (and
    a strictly finer offline variant) computed from a finding JSON dict, so
    archived findings re-sign without a server.
  * ``legacy_signature`` — a verbatim re-implementation of the old closure,
    kept for before/after comparison in the offline dedup validator.

Everything here is pure: no imports from the runner, no I/O, no global state.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

SIG_VERSION = 2
#: v1 → v2 (2026-08-23), both changes scoped to ``kv_corruption``:
#:   * the workload shape drops ``has_cancel`` and ``any_prefix`` when every
#:     bug in the finding is ``kv_corruption`` (A1);
#:   * the sub-type bands the determinism probe's ``logprob_gap`` instead of
#:     the word-set similarity whenever a numeric gap is available (B1).
#: Findings written under v1 carry ``mech_sig_version: 1``; the offline triage tool
#: re-signs those from the dict rather than trusting the stored key.
#:
#: Still 2 after the layer-12 additions (``pfx_divergent`` for the two-path
#: prefix-cache probe, ``sv_*`` for ``schema_violation``): both are *additive*
#: sub-types reached only by ``probe_metadata`` keys / a ``BugClass`` that no
#: archived finding carries, so every pre-existing finding re-signs to exactly
#: the same key it had.  A version bump is warranted only when an existing
#: finding's signature would change.
#:
#: Still 2 after the mask-composition additions (``dm_*`` for
#: ``degenerate_mask``) for the same reason: ``degenerate_mask`` is a new
#: ``BugClass`` that no archived finding carries, so no existing signature
#: moves.  Note that the sub-type branch and the ``BugClass`` landed in the
#: *same* change on purpose — a class with no branch here falls through to
#: ``sub = cls`` and every predicate collapses into one ledger entry.

#: How many exemplar findings a single signature keeps on disk before the
#: runner stops writing JSONs for it (the ledger still counts every hit).
#: Decided at 5; lives here so the runner and the offline tools cannot drift.
DEFAULT_EXEMPLARS_PER_SIGNATURE = 5

#: prompt-length buckets used by the coarse triage shape (same edges as the
#: offline triage tool, so the two agree).
_PROMPT_LEN_EDGES = (64, 256, 1024)

#: similarity bands for the kv_corruption determinism probe (fallback: used
#: only when the finding carries no numeric ``logprob_gap``)
_SIM_HI = 0.9
_SIM_MID = 0.5

#: logprob-gap band edge for the kv_corruption determinism probe.
#: ``_determinism_probe`` already discards anything below
#: ``Runner._FP_GAP_THRESHOLD`` (1e-3) as FP tie-break noise, so every gap that
#: reaches here is a real divergence; what the band separates is *how* decided
#: the model was at the diverge position.  0.5 nat sits between the two regimes
#: seen in practice: the batch-invariance flips cluster at gaps of 0.125/0.25
#: (a near-tie the batch composition can swing) while a genuinely corrupted KV
#: read lands on a token the model was confident about.  Two bands only —
#: finer bands would re-create the over-splitting v1's similarity bands caused.
_GAP_TIE_MAX = 0.5

#: classes whose mechanism is fully described by the class name itself
_SINGLE_SUBTYPE_CLASSES = frozenset({
    "kv_block_leak", "cache_eviction", "scheduler_hang", "liveness_failure",
    "inconsistency", "correctness", "abnormal_length", "repetition",
    "char_corruption", "feature_failure", "adapter_isolation_failure",
})

#: schema_violation failure mode -> sub-type.  Four distinct FSM failures.
_SCHEMA_FAILURE_SUBTYPES = {
    "json_parse":  "sv_json_parse",
    "json_schema": "sv_json_schema",
    "regex":       "sv_regex",
    "choice":      "sv_choice",
}

#: degenerate_mask predicate -> sub-type.  Three genuinely different pieces of
#: evidence for the same missing invariant ("no composed mask may be empty"),
#: and they must not collapse into one ledger entry: the sentinel predicate is
#: the silent pre-guard symptom read out of the logprobs, the grammar guard is
#: the post-0.27 HTTP 500, and the allowlist escape is the processor-only
#: composition that never reaches the reported logprobs at all.
_DEGENERATE_MASK_SUBTYPES = {
    "sentinel":         "dm_sentinel",
    "grammar_guard":    "dm_grammar_guard",
    "allowlist_escape": "dm_allowlist_escape",
}

_RE_SIMILARITY = re.compile(r"similarity\s*=\s*([0-9]*\.?[0-9]+)")
_RE_SCHEMA_FAILURE = re.compile(r"failure\s*=\s*([a-z_]+)")
_RE_SCHEMA_KIND = re.compile(r"violated its\s+([a-z_]+)\s+constraint")
_RE_CONCURRENCY = re.compile(r"concurrency\s*=\s*([0-9]+)")
_RE_MEM_MB = re.compile(r"grew by\s+(-?[0-9]*\.?[0-9]+)\s*MB")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _log2_bucket(value: Any) -> int:
    """floor(log2(value)) clamped at 0; non-numeric / missing → 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if not (v == v) or v in (float("inf"), float("-inf")):  # NaN / inf
        return 0
    v = int(v)
    if v <= 1:
        return 0
    return v.bit_length() - 1


#: public alias — the runner buckets the determinism-probe diverge token index
#: with exactly the same rule the subtypes use.
log2_bucket = _log2_bucket


def _md5(obj: Any, width: int) -> str:
    return hashlib.md5(repr(obj).encode()).hexdigest()[:width]


def _class_name(bug_class: Any) -> str:
    """Normalize a bug class to its wire string.

    Accepts a ``BugClass`` enum, ``"kv_corruption"``, or the ``str(enum)``
    form ``"BugClass.KV_CORRUPTION"`` that older finding JSONs carry (the
    report writer used ``default=str``).
    """
    if bug_class is None:
        return ""
    name = getattr(bug_class, "value", None)
    if isinstance(name, str):
        return name
    s = str(bug_class)
    if s.startswith("BugClass."):
        return s.split(".", 1)[1].lower()
    return s


def _prompt_len_bucket(n: Optional[int]) -> str:
    if n is None or n <= _PROMPT_LEN_EDGES[0]:
        return "<=64"
    if n <= _PROMPT_LEN_EDGES[1]:
        return "<=256"
    if n <= _PROMPT_LEN_EDGES[2]:
        return "<=1024"
    return ">1024"


def _stack_hash(probe_metadata: Optional[dict]) -> Optional[str]:
    """Instrumentation stack hash, wherever the caller stashed it.

    ``VllmInstrumentor`` returns ``hashes["stack"]``; the runner attaches it to
    ``probe_metadata`` either flat (``stack_hash``) or as the raw ``hashes``
    dict.  Both are accepted so neither side has to normalize first.
    """
    if not probe_metadata:
        return None
    h = probe_metadata.get("stack_hash")
    if not h:
        hashes = probe_metadata.get("hashes")
        if isinstance(hashes, dict):
            h = hashes.get("stack")
    if isinstance(h, str) and h:
        return h
    return None


# ---------------------------------------------------------------------------
# oracle subtype
# ---------------------------------------------------------------------------

def oracle_subtype(bug_class: Any, detail: str,
                   probe_metadata: Optional[dict] = None) -> str:
    """Map one oracle result to a coarse mechanism sub-type.

    Parsing is best-effort: a detail string missing its numbers degrades to a
    ``*na`` sub-type rather than raising, because the archive contains details
    written by several oracle generations.
    """
    cls = _class_name(bug_class)
    detail = detail or ""
    pm = probe_metadata or {}

    if cls == "kv_corruption":
        sub = _kv_corruption_subtype(detail, pm)
    elif cls == "latency_spike":
        m = _RE_CONCURRENCY.search(detail)
        sub = "c%d" % _log2_bucket(m.group(1)) if m else "cna"
    elif cls == "token_rate_stall":
        sub = "stall"
    elif cls == "schema_violation":
        sub = _schema_violation_subtype(detail, pm)
    elif cls == "degenerate_mask":
        sub = _degenerate_mask_subtype(detail, pm)
    elif cls == "lifecycle_error":
        if "after cancel" in detail:
            sub = "post_cancel"
        elif "never completed" in detail:
            sub = "never_completed"
        else:
            sub = "lifecycle_error"
    elif cls == "memory_leak":
        m = _RE_MEM_MB.search(detail)
        sub = "mb%d" % _log2_bucket(m.group(1)) if m else "mbna"
    elif cls in _SINGLE_SUBTYPE_CLASSES:
        sub = cls
    elif cls:
        sub = cls          # unknown class → the class name
    else:
        sub = "unknown"

    kv_structural = pm.get("kv_structural")
    if isinstance(kv_structural, dict):
        anomaly = kv_structural.get("anomaly_type")
        if anomaly:
            sub += "+%s" % anomaly
    elif isinstance(kv_structural, str) and kv_structural:
        sub += "+%s" % kv_structural

    stack = _stack_hash(pm)
    if stack:
        sub += "+stk%s" % stack[:6]

    return sub


def _kv_corruption_subtype(detail: str, pm: dict) -> str:
    """kv_corruption covers four oracles; separate them.

    ``check_determinism`` → logprob-gap band (similarity band as fallback),
    ``_prefix_equivalence_probe`` → ``pfx_divergent``,
    ``check_stability`` → ``stability``, the retired trailing-probe oracle
    (still present in the archive) → ``probe_missing``.

    v2: when ``probe_metadata`` carries a numeric ``logprob_gap`` the band is
    computed from it and the similarity is ignored.  The similarity is a
    word-set Jaccard over the two outputs — it measures how far the divergence
    propagated, not what caused it, so a single mechanism firing on prompts of
    different lengths spread over all three ``sim_*`` bands (v1 split one V7
    into 13–22 "mechanisms" per 30-minute slot).  The gap is a property of the
    model at the diverge position and is stable across those prompts.
    Archived findings have no ``probe_metadata``, so they keep the v1
    similarity bands and their cluster counts are unaffected by this change.
    """
    if pm.get("probe") == "prefix_equiv":
        # Two-path relational probe (Runner._prefix_equivalence_probe): the two
        # outputs came from ONE batch, one through the prefix cache and one
        # routed around it by cache_salt.  The mechanism is "the cache read path
        # disagrees with recompute", which is categorically different from the
        # determinism probe's gtie/gwide and must not merge with it in the
        # ledger — so the subtype is fixed here and the gap band below is not
        # consulted.  ``probe_metadata`` DOES now carry a ``logprob_gap`` (the
        # probe requests top-k logprobs on both arms and adjudicates the FP
        # near-ties at source, dropping them before they reach a finding), but
        # banding on it here would split one mechanism across two subtypes for
        # no gain: everything that survives the probe's own ULP gate is already
        # on the wide side of it.
        return "pfx_divergent"
    if detail.startswith("Validated output instability"):
        return "stability"
    if "trailing probe missing" in detail:
        return "probe_missing"

    gap = pm.get("logprob_gap")
    if isinstance(gap, bool):
        gap = None
    if isinstance(gap, (int, float)) and gap == gap:        # numeric, not NaN
        return "gtie" if float(gap) < _GAP_TIE_MAX else "gwide"

    sim = pm.get("similarity")
    if sim is None:
        m = _RE_SIMILARITY.search(detail)
        if m:
            sim = m.group(1)
    if sim is None:
        return "sim_na"
    try:
        sim = float(sim)
    except (TypeError, ValueError):
        return "sim_na"
    if sim >= _SIM_HI:
        return "sim_hi"
    if sim >= _SIM_MID:
        return "sim_mid"
    return "sim_lo"


def _schema_violation_subtype(detail: str, pm: dict) -> str:
    """``schema_violation`` covers four genuinely different FSM failures.

    They must not collapse into one ledger entry: "the grammar never closed"
    (``sv_json_parse``), "the document parsed but broke the schema"
    (``sv_json_schema``), "the regex FSM let a bad token through"
    (``sv_regex``) and "the choice FSM left its alphabet" (``sv_choice``) are
    four separate defects in the structured-output engine.

    The failure mode is read from ``probe_metadata["failure"]`` (what the
    oracle actually decided); the detail string is only a fallback for
    finding JSONs that predate the metadata.
    """
    failure = pm.get("failure")
    if not isinstance(failure, str) or not failure:
        m = _RE_SCHEMA_FAILURE.search(detail or "")
        failure = m.group(1) if m else ""
    failure = failure.strip().lower()
    if failure in _SCHEMA_FAILURE_SUBTYPES:
        return _SCHEMA_FAILURE_SUBTYPES[failure]

    # No failure mode recorded — fall back to the constraint kind, which at
    # least keeps json / regex / choice apart.
    kind = pm.get("constraint")
    if not isinstance(kind, str):
        m = _RE_SCHEMA_KIND.search(detail or "")
        kind = m.group(1) if m else ""
    kind = (kind or "").strip().lower()
    if kind in ("json_schema", "json_object", "json"):
        return "sv_json"
    if kind in ("regex", "choice"):
        return "sv_%s" % kind
    return "sv_na"


def _degenerate_mask_subtype(detail: str, pm: dict) -> str:
    """``degenerate_mask`` covers three predicates over one missing invariant.

    The predicate is read from ``probe_metadata["predicate"]`` (what the oracle
    actually decided); the detail string is only a fallback, for finding JSONs
    written by a future oracle generation that drops the metadata.

    The composition itself is deliberately *not* part of the sub-type.  Which
    pair of restrictors emptied the row is a property of the workload, not of
    the mechanism — one defect ("nothing reconciles two -inf writers") wearing
    N different hats is exactly the monoculture that made the archived
    ``min_tokens`` finding look like 122 separate bugs.
    """
    predicate = pm.get("predicate")
    if not isinstance(predicate, str) or not predicate:
        d = detail or ""
        if "HTTP 500" in d or "grammar guard" in d:
            predicate = "grammar_guard"
        elif "allowed_token_ids escaped" in d:
            predicate = "allowlist_escape"
        elif "-inf sentinel" in d:
            predicate = "sentinel"
        else:
            predicate = ""
    predicate = predicate.strip().lower()
    return _DEGENERATE_MASK_SUBTYPES.get(predicate, "dm_na")


# ---------------------------------------------------------------------------
# workload shape
# ---------------------------------------------------------------------------

def _event_field(event: Any, name: str) -> Any:
    """Read a field from either an ``Event`` object or an event dict."""
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _events_of(trace: Any) -> list:
    if isinstance(trace, dict):
        events = trace.get("events")
        if events is None:
            events = trace.get("trace", {}).get("events", [])
        return events or []
    return list(getattr(trace, "events", []) or [])


def all_kv_corruption(bugs: Any) -> bool:
    """True when *bugs* is non-empty and every entry is ``kv_corruption``.

    Accepts anything ``mechanism_pairs`` accepts (live ``OracleResult`` objects
    or ``_BugView``s over a finding dict), so the live and the from-dict paths
    reach the same verdict.
    """
    classes = [_class_name(getattr(b, "bug_class", None)) for b in (bugs or [])]
    classes = [c for c in classes if c]
    return bool(classes) and all(c == "kv_corruption" for c in classes)


def workload_shape(trace: Any, bugs: Any = None) -> tuple:
    """(has_cancel, n_distinct_adapters, any_prefix_gt0, any_logprobs, any_n_gt1)

    Accepts a live ``Trace``, a trace dict (``trace.to_dict()``), or a whole
    finding dict.  Deliberately invariant to offsets, event order, event count,
    prompt_len, max_tokens and prefix_len *magnitude* — those are the numerics
    that make the legacy key refile the same mechanism after one mutation.

    v2: when *bugs* is given and every bug is ``kv_corruption``, the coarse
    3-tuple ``(n_distinct_adapters, any_logprobs, any_n_gt1)`` is returned
    instead.  ``has_cancel`` and ``any_prefix`` are workload decorations that
    do not change the corruption mechanism — a cancelled sibling request or a
    shared prefix elsewhere in the trace does not make a different bug — yet
    they multiplied one V7 into four shape-variants per similarity band.  The
    two shapes are different lengths, so a coarse key can never collide with a
    full one.  Passing no *bugs* keeps the v1 5-tuple (used by the offline
    sensitivity sweep in the offline dedup validator).
    """
    events = _events_of(trace)
    has_cancel = False
    adapters = set()
    any_prefix = False
    any_logprobs = False
    any_n_gt1 = False

    for ev in events:
        action = _event_field(ev, "action")
        action = getattr(action, "value", action)
        if action == "cancel":
            has_cancel = True

        adapter = _event_field(ev, "adapter_name")
        if adapter:
            adapters.add(adapter)

        prefix = _event_field(ev, "prefix_len")
        if isinstance(prefix, (int, float)) and prefix > 0:
            any_prefix = True

        logprobs = _event_field(ev, "logprobs")
        if logprobs is not None:
            any_logprobs = True

        n = _event_field(ev, "n_completions")
        if isinstance(n, (int, float)) and n > 1:
            any_n_gt1 = True

    if bugs is not None and all_kv_corruption(bugs):
        return (len(adapters), any_logprobs, any_n_gt1)
    return (has_cancel, len(adapters), any_prefix, any_logprobs, any_n_gt1)


# ---------------------------------------------------------------------------
# mechanism signature — live objects
# ---------------------------------------------------------------------------

def mechanism_pairs(bugs: list) -> tuple:
    """Sorted ``(class, subtype)`` pairs for a list of ``OracleResult``."""
    pairs = []
    for b in bugs:
        bug_class = getattr(b, "bug_class", None)
        if not bug_class:
            continue
        pairs.append((
            _class_name(bug_class),
            oracle_subtype(bug_class, getattr(b, "detail", "") or "",
                           getattr(b, "probe_metadata", None)),
        ))
    return tuple(sorted(set(pairs)))


def mechanism_signature(trace: Any, bugs: list) -> str:
    """In-run dedup key: mechanism (class+subtype set) × workload shape.

    The shape is bug-aware (see ``workload_shape``), which is why *bugs* is
    threaded through: ``mechanism_signature_from_dict`` calls straight into
    here, so the live key and the re-signed key cannot drift apart.
    """
    return _md5((mechanism_pairs(bugs), workload_shape(trace, bugs)), 16)


# ---------------------------------------------------------------------------
# mechanism / triage signature — finding dicts
# ---------------------------------------------------------------------------

class _BugView:
    """Duck-typed stand-in for ``OracleResult`` over a ``bugs[]`` dict entry."""

    __slots__ = ("bug_class", "detail", "score", "probe_metadata")

    def __init__(self, entry: dict, instrumentation_stack: Optional[str] = None):
        self.bug_class = entry.get("class")
        self.detail = entry.get("detail") or ""
        self.score = entry.get("score", 0.0)
        pm = entry.get("probe_metadata")
        pm = dict(pm) if isinstance(pm, dict) else None
        if instrumentation_stack and _stack_hash(pm) is None:
            pm = dict(pm or {})
            pm["stack_hash"] = instrumentation_stack
        self.probe_metadata = pm


def _instrumentation_stack(finding: dict) -> Optional[str]:
    """``instrumentation.hashes["stack"]`` if any run recorded one.

    The block is a list of per-run records under ``runs`` in the archived
    schema; in practice every archived run has ``hashes == {}``.
    """
    inst = finding.get("instrumentation")
    if not isinstance(inst, dict):
        return None
    runs = inst.get("runs")
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                continue
            h = _stack_hash(run)
            if h:
                return h
    return _stack_hash(inst)


def bug_views(finding: dict) -> list:
    """``bugs[]`` entries as OracleResult-shaped objects."""
    stack = _instrumentation_stack(finding)
    return [_BugView(b, stack) for b in finding.get("bugs", []) or []
            if isinstance(b, dict)]


def mechanism_signature_from_dict(finding: dict) -> str:
    """``mechanism_signature`` computed from a finding JSON dict.

    Equal, by construction, to the value the runner would have computed from
    the live objects — this is what lets the offline validation predict online
    behaviour.
    """
    trace = finding.get("trace") or {}
    return mechanism_signature(trace, bug_views(finding))


def normalize_server_flags(flags: Optional[str]) -> str:
    """Strip per-run volatility (port, ZMQ endpoint) from a server flag string.

    Two slots of the same config differ only in the port they were handed; that
    is not a configuration difference and must not split a cluster.
    """
    if not flags:
        return ""
    s = str(flags)
    s = re.sub(r"(--port\s+)\d+", r"\1PORT", s)
    s = re.sub(r"tcp://[^\s\"']+", "tcp://EP", s)
    s = re.sub(r"(--kv-events-config\s+)\S+", r"\1CFG", s)
    return " ".join(s.split())


def coarse_shape(finding: dict) -> tuple:
    """(log2(n_events), log2(max prompt_len), min(3, n distinct prompt buckets))."""
    events = _events_of(finding.get("trace") or {})
    sends = [e for e in events
             if getattr(_event_field(e, "action"), "value",
                        _event_field(e, "action")) == "send"]
    prompt_lens = [_event_field(e, "prompt_len") for e in sends]
    prompt_lens = [p for p in prompt_lens if isinstance(p, (int, float))]
    max_prompt = max(prompt_lens) if prompt_lens else 0
    buckets = {_prompt_len_bucket(p) for p in prompt_lens}
    return (_log2_bucket(len(events)), _log2_bucket(max_prompt),
            min(3, len(buckets)))


def triage_signature(finding: dict) -> str:
    """Offline cluster key: the in-run key plus config identity and shape.

    Strictly finer than ``mechanism_signature_from_dict`` — triage can always
    split an over-merged in-run bucket without re-running anything.
    """
    return _md5((
        mechanism_signature_from_dict(finding),
        _md5(normalize_server_flags(finding.get("server_flags")), 8),
        finding.get("model") or "",
        coarse_shape(finding),
    ), 16)


# ---------------------------------------------------------------------------
# legacy key (for before/after comparison only)
# ---------------------------------------------------------------------------

def _legacy_action(action: Any) -> str:
    """``e.action.value if hasattr(e.action, "value") else str(e.action)``."""
    return action.value if hasattr(action, "value") else str(action)


def legacy_signature(finding: dict) -> str:
    """Verbatim re-implementation of ``runner.py::_handle_finding._finding_sig``
    (runner.py:753-763) applied to a finding dict.

    Kept only so the offline dedup validator can measure the before/after
    cluster counts on the same corpus of findings.
    """
    bugs = finding.get("bugs", []) or []
    classes = tuple(sorted(_class_name(b.get("class")) for b in bugs
                           if b.get("class")))
    events = tuple(
        (_legacy_action(e.get("action")), e.get("adapter_name"),
         e.get("prompt_len"), e.get("max_tokens"), e.get("prefix_len"))
        for e in _events_of(finding.get("trace") or {})
    )
    return _md5((classes, events), 16)
