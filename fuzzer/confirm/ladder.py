"""The confirmation ladder for output-divergence candidates.

A divergence candidate is one request (``rid``) of one trace whose greedy
output differed between two executions on the stock fuzzing server.  On a
stock server most such differences are batch variance: the request's logits
depend on what else was in its batch.  The ladder asks three questions, each
of which isolates one variable:

    S1 vs S2   the request ALONE, twice, on the stock server.
               Different -> the request is nondeterministic even without a
               batch around it.                      SOLO_NONDETERMINISTIC
    B1 vs B2   the whole trace, twice, on a batch-invariant (BI) server.
               Different -> the output depends on scheduling/timing even
               though batch composition no longer moves the arithmetic.
                                                     BI_SCHEDULE_DEPENDENT
    B1 vs BS   the whole trace vs the request alone, both on the BI server.
               Different -> co-scheduled requests change this request's
               output under BI: cross-request interference.
                                                     BI_BATCH_DEPENDENT
    (twin)     prefix-equivalence candidates carry a twin request that took
               a different prefix-cache path in the SAME run.  If the
               rid/twin difference reappears in both BI replays, the cache
               path itself changes the output under BI.
                                                     BI_RELATION_PERSISTS
    otherwise  the divergence evaporates under the controls.
                                                     BATCH_VARIANCE

Every comparison is an exact comparison of the request's ``output_text``.  A
BI server is expected to be bitwise reproducible, so no tolerance is needed.
An arm that errored, was cancelled or returned no text makes the ladder
UNINFORMATIVE rather than silently passing.

Prefix-cache control needs no dev-mode endpoint: every arm gets fresh
``cache_salt`` values (:func:`salted_arm`).  Within one arm the original
trace's salt GROUPS are preserved (requests that shared a salt, or shared
"no salt", still share one), so intra-trace prefix sharing is unchanged while
no arm can hit blocks another arm created.

This module is pure: no network, no vLLM.  The runner owns execution.
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from fuzzer.trace import ActionType, Event, Trace

VERDICT_SOLO_NONDETERMINISTIC = "SOLO_NONDETERMINISTIC"
VERDICT_BI_SCHEDULE_DEPENDENT = "BI_SCHEDULE_DEPENDENT"
VERDICT_BI_BATCH_DEPENDENT = "BI_BATCH_DEPENDENT"
VERDICT_BI_RELATION_PERSISTS = "BI_RELATION_PERSISTS"
VERDICT_BATCH_VARIANCE = "BATCH_VARIANCE"
VERDICT_UNINFORMATIVE = "UNINFORMATIVE"

#: Verdicts that are filed as findings.  BATCH_VARIANCE and UNINFORMATIVE are
#: logged only.
FILED_VERDICTS = frozenset({
    VERDICT_SOLO_NONDETERMINISTIC,
    VERDICT_BI_SCHEDULE_DEPENDENT,
    VERDICT_BI_BATCH_DEPENDENT,
    VERDICT_BI_RELATION_PERSISTS,
})

#: Queue priority from the logit screen.  Lower sorts first.
PRIORITY_HIGH = 0     # divergence not explainable by an FP tie
PRIORITY_LOW = 1      # top1-top2 gap inside the FP tie band

#: Top-k logprobs requested for the candidate request in EVERY arm, so the log
#: can report the cross-arm gap at the first divergence.
ARM_LOGPROBS = 5

ARM_NAMES = ("S1", "S2", "B1", "B2", "BS")


@dataclass
class Candidate:
    """One divergence waiting for the ladder."""

    report_trace: Trace          # the trace a filed finding is reported on
    replay_trace: Trace          # the trace the batched arms replay
    rid: str                     # the request whose output is compared
    source: str                  # "determinism" | "prefix_equiv"
    bugs: list                   # the OracleResults the probe produced
    iteration: int
    priority: int = PRIORITY_HIGH
    score: float = 0.0
    twin_rid: Optional[str] = None
    #: True when the rid/twin relation is "cache read vs recompute": the
    #: relation is only tested if the BI replay actually read rid's prefix
    #: from the cache again.
    twin_needs_cache_hit: bool = False
    extra: dict = field(default_factory=dict)
    fingerprint: Any = None      # corpus fingerprint at enqueue time
    mech_sig: str = ""
    seq: int = 0
    logprob_gap: Optional[float] = None
    logprob_gap_ulps: Optional[float] = None

    def sort_key(self) -> tuple:
        return (self.priority, -self.score, self.seq)


# ---------------------------------------------------------------------------
# Arm construction
# ---------------------------------------------------------------------------

def find_send(trace: Trace, rid: str) -> Optional[Event]:
    for ev in trace.events:
        if ev.action == ActionType.SEND and ev.request_id == rid:
            return ev
    return None


def _apply_probe_knobs(ev: Event, logprobs: int = ARM_LOGPROBS) -> None:
    """The knobs every arm sets identically on the compared request(s)."""
    ev.logprobs = max(ev.logprobs or 0, logprobs)
    ev.include_usage = True


def solo_trace(trace: Trace, rid: str) -> Trace:
    """The SEND of ``rid`` alone, at offset 0.

    The request id is kept, and ``prompt_family_id`` is pinned to its current
    value, so the adapter renders the byte-identical prompt.  CANCEL /
    DISCONNECT events are not carried over: a candidate is only ever a request
    that completed on both original runs.
    """
    ev = find_send(trace, rid)
    if ev is None:
        raise KeyError(f"no SEND event for request {rid!r}")
    solo = copy.deepcopy(ev)
    solo.offset_ms = 0
    solo.prompt_family_id = solo.prompt_family_id or solo.request_id
    return Trace(seed=trace.seed, events=[solo], tags=list(trace.tags))


def salted_arm(trace: Trace, rng: random.Random, arm: str,
               knob_rids: tuple[str, ...] = ()) -> Trace:
    """A deep copy of ``trace`` with arm-private cache salts.

    Every distinct original ``cache_salt`` value (``None`` included) maps to a
    fresh salt for this arm, so requests that shared a prefix-cache namespace
    in the original still share one, and nothing in this arm can hit a block
    that another arm (or earlier fuzzing traffic) created.  The compared
    requests in ``knob_rids`` get the same probe knobs in every arm.
    """
    out = copy.deepcopy(trace)
    mapping: dict[Optional[str], str] = {}
    for ev in out.events:
        if ev.action != ActionType.SEND:
            continue
        if ev.cache_salt not in mapping:
            mapping[ev.cache_salt] = "grief-cf-%s-%08x-%d" % (
                arm, rng.getrandbits(32), len(mapping))
        ev.cache_salt = mapping[ev.cache_salt]
        if ev.request_id in knob_rids:
            _apply_probe_knobs(ev)
    return out


def build_arms(cand: Candidate, rng: random.Random) -> dict[str, Trace]:
    """The five arm traces, in :data:`ARM_NAMES` order."""
    knobs = (cand.rid,) + ((cand.twin_rid,) if cand.twin_rid else ())
    solo = solo_trace(cand.replay_trace, cand.rid)
    return {
        "S1": salted_arm(solo, rng, "S1", (cand.rid,)),
        "S2": salted_arm(solo, rng, "S2", (cand.rid,)),
        "B1": salted_arm(cand.replay_trace, rng, "B1", knobs),
        "B2": salted_arm(cand.replay_trace, rng, "B2", knobs),
        "BS": salted_arm(solo, rng, "BS", (cand.rid,)),
    }


# ---------------------------------------------------------------------------
# Outcomes and the verdict
# ---------------------------------------------------------------------------

def _digest(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _usable(req) -> Optional[str]:
    """None if the request result can be compared, else why not."""
    if req is None:
        return "missing"
    if getattr(req, "error", None):
        return f"error: {str(req.error)[:120]}"
    if getattr(req, "cancelled_at", None) is not None:
        return "cancelled"
    if getattr(req, "output_text", None) is None:
        return "no output_text"
    return None


@dataclass
class ArmOutcome:
    name: str
    ok: bool
    reason: str = ""
    text: Optional[str] = None
    digest: Optional[str] = None
    cache_read_tokens: Optional[int] = None
    twin_text: Optional[str] = None
    twin_digest: Optional[str] = None
    top_logprobs: Optional[list] = None

    def to_log(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "digest": self.digest,
            "len": None if self.text is None else len(self.text),
            "cache_read_tokens": self.cache_read_tokens,
            "twin_digest": self.twin_digest,
        }


def arm_outcome(name: str, result: Any, rid: str,
                twin_rid: Optional[str] = None,
                exc: Optional[BaseException] = None) -> ArmOutcome:
    """Extract the compared request(s) from one arm's TraceResult."""
    if exc is not None:
        return ArmOutcome(name, False, f"arm raised {type(exc).__name__}: {exc}"[:200])
    if result is None:
        return ArmOutcome(name, False, "no result")
    by_rid = {r.request_id: r for r in getattr(result, "requests", [])}
    req = by_rid.get(rid)
    why = _usable(req)
    if why:
        return ArmOutcome(name, False, f"{rid}: {why}")
    out = ArmOutcome(
        name, True,
        text=req.output_text, digest=_digest(req.output_text),
        cache_read_tokens=getattr(req, "cache_read_tokens", None),
        top_logprobs=getattr(req, "output_top_logprobs", None),
    )
    if twin_rid is not None and name in ("B1", "B2"):
        twin = by_rid.get(twin_rid)
        twin_why = _usable(twin)
        if twin_why:
            return ArmOutcome(name, False, f"twin {twin_rid}: {twin_why}")
        out.twin_text = twin.output_text
        out.twin_digest = _digest(twin.output_text)
    return out


def first_diff(a: str, b: str) -> Optional[int]:
    """Index of the first differing character; None when equal."""
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


@dataclass
class LadderVerdict:
    verdict: str
    reason: str
    deciding_pair: Optional[tuple[str, str]] = None
    diverge_char_idx: Optional[int] = None
    comparisons: dict = field(default_factory=dict)

    @property
    def filed(self) -> bool:
        return self.verdict in FILED_VERDICTS

    def to_log(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "deciding_pair": list(self.deciding_pair) if self.deciding_pair else None,
            "diverge_char_idx": self.diverge_char_idx,
            "comparisons": self.comparisons,
        }


def judge(arms: dict[str, ArmOutcome], twin: bool = False,
          twin_needs_cache_hit: bool = False) -> LadderVerdict:
    """Apply the ladder to the five arm outcomes.

    ``twin_needs_cache_hit``: for a cache-path relation, a BI replay in which
    the request read nothing from the prefix cache did not re-create the
    condition under test, so the relation rung is UNINFORMATIVE rather than
    BATCH_VARIANCE (unknown counts are treated the same way).
    """
    s1, s2, b1, b2, bs = (arms[n] for n in ARM_NAMES)
    cmp: dict[str, Optional[bool]] = {}

    def differ(a: ArmOutcome, b: ArmOutcome) -> bool:
        return a.text != b.text

    # Rung 1 can decide on its own: a request that is nondeterministic alone
    # is reported whatever the BI arms did.
    if s1.ok and s2.ok:
        cmp["S1!=S2"] = differ(s1, s2)
        if cmp["S1!=S2"]:
            return LadderVerdict(
                VERDICT_SOLO_NONDETERMINISTIC,
                "the request alone produced two different outputs on the stock server",
                ("S1", "S2"), first_diff(s1.text, s2.text), cmp)

    bad = [a for a in (s1, s2, b1, b2, bs) if not a.ok]
    if bad:
        return LadderVerdict(
            VERDICT_UNINFORMATIVE,
            "; ".join(f"{a.name}: {a.reason}" for a in bad), None, None, cmp)

    cmp["B1!=B2"] = differ(b1, b2)
    if cmp["B1!=B2"]:
        return LadderVerdict(
            VERDICT_BI_SCHEDULE_DEPENDENT,
            "two BI replays of the same trace disagreed",
            ("B1", "B2"), first_diff(b1.text, b2.text), cmp)

    cmp["B1!=BS"] = differ(b1, bs)
    if cmp["B1!=BS"]:
        return LadderVerdict(
            VERDICT_BI_BATCH_DEPENDENT,
            "under BI the request's output in the batch differs from its output alone",
            ("B1", "BS"), first_diff(b1.text, bs.text), cmp)

    if twin:
        rel1 = b1.text != b1.twin_text
        rel2 = b2.text != b2.twin_text
        cmp["B1:rid!=twin"] = rel1
        cmp["B2:rid!=twin"] = rel2
        if rel1 and rel2:
            return LadderVerdict(
                VERDICT_BI_RELATION_PERSISTS,
                "the request and its twin disagreed in both BI replays",
                ("B1.rid", "B1.twin"), first_diff(b1.text, b1.twin_text), cmp)
        if twin_needs_cache_hit:
            hits = [b1.cache_read_tokens, b2.cache_read_tokens]
            if any(h is None or h <= 0 for h in hits):
                return LadderVerdict(
                    VERDICT_UNINFORMATIVE,
                    "cache-path relation not re-created: the request's BI "
                    f"replays read {hits} prefix tokens from the cache",
                    None, None, cmp)

    return LadderVerdict(
        VERDICT_BATCH_VARIANCE,
        "every controlled comparison agreed; the divergence evaporated",
        None, None, cmp)
