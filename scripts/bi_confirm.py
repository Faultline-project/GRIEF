#!/usr/bin/env python3
"""
scripts/bi_confirm.py — R2 batch-invariance confirmation runner.

Implements state **R2** of the attribution machine
(section "R2 — BI as three-state corroborating evidence"): replay a finding N times
against a batch-invariant reference server and emit *corroborating evidence*,
never a classification.

Three outcomes, exactly as the canonical machine defines them:

  CORROBORATES          the divergence survives the BI intervention set — the
                        strongest single piece of evidence that the finding is
                        not batch numerics.  The *sole* route to this verdict is
                        internal non-determinism across the N replays: some
                        comparable request produced >= 2 distinct outputs over
                        >= 2 completed runs.
  DOES_NOT_CORROBORATE  the divergence evaporates.  This LOWERS confidence and
                        flags the finding for re-examination.  It does **not**
                        unclassify a signature-positive finding: evaporation
                        proves sensitivity to the intervention set, not
                        benignness (see ``EVAPORATION_CAVEAT``).
  UNINFORMATIVE         a guard failed — no evidence was produced at all.

Two guards run *before* any replay; either failure short-circuits to
UNINFORMATIVE with no requests sent:

  1. Engagement proof — batch-invariant mode must be shown to have actually
     initialized in the server, read from the server log (``--vllm-log-file``).
     See ``BI_ENGAGEMENT_SIGNALS`` for the exact lines matched and why.
  2. Backend allowlist — the campaign backend must be BI-sound.  See
     ``BI_BACKEND_ALLOWLIST`` / ``BI_BACKEND_EXCLUSIONS``.

The replay outputs are also compared against the reference output recorded on
the finding, and the result is written to the sidecar as
``per_request.reference_mismatch``.  That comparison is an **observation, never
evidence**: the reference was recorded without the BI intervention set, which
legitimately changes outputs, so a mismatch is the expected outcome and says
nothing about batch numerics.  It cannot produce CORROBORATES — see
``REFERENCE_MISMATCH_CAVEAT``.

Usage:
    python3 scripts/bi_confirm.py findings/finding_XXXXX_YYYYY.json \\
        --base-url http://localhost:8000 \\
        --vllm-log-file logs/vllm_bi_server.log \\
        [--runs 10] \\
        [--backend FLASH_ATTN] [--fa-version 2] [--gpu-arch sm90] \\
        [--output-dir DIR]   # default: <repo>/findings/reproduction/bi,
                             # anchored on the finding's own repo root

Exit codes: 0 = CORROBORATES, 1 = DOES_NOT_CORROBORATE, 2 = UNINFORMATIVE,
3 = usage/IO error.  The finding JSON is never modified; the verdict is written
as a sidecar under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import dataclasses
import datetime
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from fuzzer.trace import Trace, Event, ActionType
from fuzzer.executor import PythonExecutor, TraceResult, RequestResult
# Guard 1 / Guard 2 and the backend/arch resolvers live in
# fuzzer/confirm/bi_guard.py so the fuzzing loop can use them too.
from fuzzer.confirm.bi_guard import (  # noqa: E402,F401 -- moved, re-exported
    ATTRIBUTION_DOC,
    ATTRIBUTION_SECTION,
    _SAMPLE_CHARS,
    LogSignal,
    BI_ENGAGEMENT_SIGNALS,
    BI_NON_PROOF_SIGNALS,
    EngagementProof,
    prove_engagement,
    BI_BACKEND_ALLOWLIST,
    BI_BACKEND_EXCLUSIONS,
    BI_BACKEND_KNOWN_OTHER,
    _BACKEND_ALIASES,
    _HOPPER_TOKENS,
    _NON_HOPPER_TOKENS,
    GpuArch,
    UNKNOWN_ARCH,
    normalize_gpu_arch,
    normalize_backend,
    infer_fa_version,
    BackendRuling,
    rule_backend,
    _FLAG_BACKEND_PATTERNS,
    _FLAG_FA_VERSION_PATTERNS,
    _LOG_BACKEND_PATTERNS,
    _LOG_FA_VERSION_PATTERNS,
    _first_match,
    _walk_strings,
    _search_backend_key,
    resolve_backend,
    resolve_gpu_arch,
)

# ---------------------------------------------------------------------------
# Canonical semantics — every constant below is data, with a doc pointer.
# ---------------------------------------------------------------------------

# 2: per_request.diverges_from_reference became per_request.reference_mismatch,
#    an observation that no longer feeds the verdict.
SCHEMA_VERSION = 2
SIDECAR_SUFFIX = "_bi_r2.json"
DEFAULT_OUTPUT_DIR = "findings/reproduction/bi"

VERDICT_CORROBORATES = "CORROBORATES"
VERDICT_DOES_NOT_CORROBORATE = "DOES_NOT_CORROBORATE"
VERDICT_UNINFORMATIVE = "UNINFORMATIVE"

EXIT_CODES = {
    VERDICT_CORROBORATES: 0,
    VERDICT_DOES_NOT_CORROBORATE: 1,
    VERDICT_UNINFORMATIVE: 2,
}

# Standing caveat attached to every DOES_NOT_CORROBORATE verdict.  R2 attaches
# evidence; it never promotes and never unclassifies.
EVAPORATION_CAVEAT = (
    "Evaporation under BI proves sensitivity to the intervention set, not "
    "benignness: BI=1 replaces aten ops, restricts the attention admission "
    "list, forces num_splits=1, disables cascade attention and changes "
    "comm/compile settings. This result lowers confidence and flags the "
    "finding for re-examination; it never auto-unclassifies a "
    f"signature-positive finding ({ATTRIBUTION_DOC}, {ATTRIBUTION_SECTION})."
)

# Standing caveat attached whenever a recorded reference output was compared.
# The finding's reference was recorded on a run without the intervention set, so
# a mismatch is the *expected* result and carries no information either way.
REFERENCE_MISMATCH_CAVEAT = (
    "reference_mismatch compares the BI replay outputs against the output "
    "recorded on the finding, which was recorded without the BI intervention "
    "set. BI=1 replaces aten ops, restricts the attention admission list, "
    "forces num_splits=1, disables cascade attention and changes comm/compile "
    "settings, so a mismatch against that reference is expected and "
    "uninformative. It is recorded as an observation only and never drives "
    "CORROBORATES; the sole corroboration route is internal non-determinism "
    f"across the BI replays ({ATTRIBUTION_DOC}, {ATTRIBUTION_SECTION})."
)

# One replay has no statistical power (april_10 reproduced 23/200 under BI
# probing) — the canonical default is ten.
DEFAULT_RUNS = 10

# Fewer than this many *successful* replays cannot support an
# internal-uniqueness comparison.
MIN_RUNS_FOR_COMPARISON = 2

# Top-k logprobs requested on every comparable request.  Logprob divergence is
# recorded as auxiliary evidence; the verdict is driven by exact output text.
DEFAULT_LOGPROBS = 1



# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


@dataclass
class NormalizedTrace:
    """A trace rewritten for a determinism probe, plus what it can compare."""

    trace: Trace
    comparable_request_ids: tuple[str, ...]
    excluded: dict[str, str] = field(default_factory=dict)


def normalize_for_bi(trace: Trace, logprobs: int = DEFAULT_LOGPROBS) -> NormalizedTrace:
    """Force greedy decoding + logprobs on every comparable SEND event.

    Requests that cannot be made deterministic are left untouched and recorded
    in ``excluded``: ``n_completions > 1`` requires ``temperature > 0`` (Event
    invariant), so those requests are never compared.
    """
    events: list[Event] = []
    comparable: list[str] = []
    excluded: dict[str, str] = {}
    cancelled: set[str] = set()

    for event in trace.events:
        if event.action is not ActionType.SEND:
            if event.action in (ActionType.CANCEL, ActionType.DISCONNECT):
                cancelled.add(event.request_id)
            events.append(copy.deepcopy(event))
            continue

        if event.n_completions is not None and event.n_completions > 1:
            excluded[event.request_id] = (
                "n_completions > 1 requires temperature > 0 — not comparable"
            )
            events.append(copy.deepcopy(event))
            continue

        new_logprobs = event.logprobs if (event.logprobs or 0) >= 1 else logprobs
        events.append(
            dataclasses.replace(
                copy.deepcopy(event), temperature=0.0, logprobs=new_logprobs
            )
        )
        comparable.append(event.request_id)

    for request_id in cancelled:
        if request_id in comparable:
            comparable.remove(request_id)
            excluded[request_id] = "cancelled/disconnected during the trace"

    return NormalizedTrace(
        trace=Trace(seed=trace.seed, events=events, tags=list(trace.tags)),
        comparable_request_ids=tuple(comparable),
        excluded=excluded,
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceOutput:
    """An output recorded on the finding, compared as an observation only.

    Recorded without the BI intervention set, so it can never corroborate —
    see ``REFERENCE_MISMATCH_CAVEAT``.
    """

    text: str
    truncated: bool
    source: str

    def matches(self, observed: str) -> bool:
        return observed.startswith(self.text) if self.truncated else observed == self.text


def load_reference_outputs(finding: dict) -> dict[str, ReferenceOutput]:
    """Collect per-request reference outputs recorded on the finding.

    Recorded samples written by the standalone reproducer are truncated at 200
    characters, so those are compared by prefix and flagged ``truncated``.
    """
    references: dict[str, ReferenceOutput] = {}

    for key in ("bi_reference_outputs", "reference_outputs"):
        raw = finding.get(key)
        if isinstance(raw, dict):
            for rid, text in raw.items():
                if isinstance(text, str):
                    references[str(rid)] = ReferenceOutput(
                        text=text, truncated=False, source=f"finding.{key}"
                    )
    if references:
        return references

    evidence = ((finding.get("reproduction") or {}).get("evidence") or {})
    for bug_class, block in evidence.items():
        if not isinstance(block, dict):
            continue
        keyed = block.get("reference_outputs")
        if isinstance(keyed, dict):
            for rid, text in keyed.items():
                if isinstance(text, str):
                    references[str(rid)] = ReferenceOutput(
                        text=text,
                        truncated=False,
                        source=f"reproduction.evidence.{bug_class}.reference_outputs",
                    )
            continue
        diverged = block.get("diverged_requests")
        sample = block.get("sample_A")
        if isinstance(diverged, list) and diverged and isinstance(sample, str):
            references[str(diverged[0])] = ReferenceOutput(
                text=sample,
                truncated=True,
                source=f"reproduction.evidence.{bug_class}.sample_A (truncated)",
            )

    return references


@dataclass
class RequestComparison:
    """Per-request evidence collected across the N replays."""

    request_id: str
    runs_observed: int
    unique_outputs: int
    samples: list[str] = field(default_factory=list)
    unique_logprob_signatures: Optional[int] = None
    reference_source: Optional[str] = None
    # Observation only — recorded, never scored (REFERENCE_MISMATCH_CAVEAT).
    reference_mismatch: Optional[bool] = None

    @property
    def reference_compared(self) -> bool:
        return self.reference_mismatch is not None

    @property
    def evidenced(self) -> bool:
        """Was this request seen often enough to compare it with itself?"""
        return self.runs_observed >= MIN_RUNS_FOR_COMPARISON

    def to_dict(self) -> dict:
        return {
            "runs_observed": self.runs_observed,
            "unique_outputs": self.unique_outputs,
            "unique_logprob_signatures": self.unique_logprob_signatures,
            "samples": self.samples,
            "reference_source": self.reference_source,
            "reference_mismatch": self.reference_mismatch,
        }


def _comparable_requests(result: TraceResult) -> Iterable[RequestResult]:
    for request in result.requests:
        if request.cancelled_at is not None:
            continue
        if request.error is not None:
            continue
        if request.temperature is not None and request.temperature > 0:
            continue
        if request.output_text is None:
            continue
        yield request


def compare_runs(
    results: Sequence[TraceResult],
    comparable_request_ids: Sequence[str] = (),
    references: Optional[dict[str, ReferenceOutput]] = None,
) -> dict[str, RequestComparison]:
    """Exact-output comparison per request across the replays.

    ``comparable_request_ids`` restricts the comparison when the caller knows
    which requests were normalized; an empty tuple means "every deterministic
    request the replays produced".

    ``references`` adds the recorded-reference observation to each comparison.
    It is reported, never scored (``REFERENCE_MISMATCH_CAVEAT``).
    """
    references = references or {}
    allowed = set(comparable_request_ids)

    outputs: dict[str, list[str]] = {}
    logprob_sigs: dict[str, list[str]] = {}
    for result in results:
        for request in _comparable_requests(result):
            if allowed and request.request_id not in allowed:
                continue
            outputs.setdefault(request.request_id, []).append(request.output_text or "")
            if request.output_logprobs is not None:
                logprob_sigs.setdefault(request.request_id, []).append(
                    json.dumps([round(float(v), 6) for v in request.output_logprobs])
                )

    comparisons: dict[str, RequestComparison] = {}
    for request_id, texts in outputs.items():
        distinct = list(dict.fromkeys(texts))
        reference = references.get(request_id)
        mismatch: Optional[bool] = None
        if reference is not None:
            mismatch = any(not reference.matches(text) for text in texts)
        sigs = logprob_sigs.get(request_id)
        comparisons[request_id] = RequestComparison(
            request_id=request_id,
            runs_observed=len(texts),
            unique_outputs=len(distinct),
            samples=[text[:_SAMPLE_CHARS] for text in distinct[:2]],
            unique_logprob_signatures=len(set(sigs)) if sigs else None,
            reference_source=reference.source if reference else None,
            reference_mismatch=mismatch,
        )
    return comparisons


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


@dataclass
class Decision:
    verdict: str
    reason: str
    caveat: Optional[str] = None
    diverged_requests: list[str] = field(default_factory=list)
    replay_reason: str = ""


def decide(
    engagement: EngagementProof,
    ruling: BackendRuling,
    comparisons: Optional[dict[str, RequestComparison]] = None,
    completed_runs: int = 0,
) -> Decision:
    """Apply the R2 verdict rule.

    Guard failures take precedence over every replay result: an unproven
    engagement or a non-allowlisted backend means the replays carry no
    information about batch invariance, however they came out.

    CORROBORATES rests **solely** on internal non-determinism under BI: some
    comparable request produced >= 2 distinct outputs across >= 2 completed
    replays.  Comparison against the finding's recorded reference is
    deliberately *not* a route to this verdict — that reference was recorded
    without the BI intervention set, which legitimately changes outputs, so a
    mismatch is expected and uninformative (``REFERENCE_MISMATCH_CAVEAT``).
    It is carried into the sidecar as ``per_request.reference_mismatch`` and
    noted in the reason string, and that is all it does.

    A request is *evidenced* only if it was observed in at least
    ``MIN_RUNS_FOR_COMPARISON`` replays: one observation cannot be compared
    against anything.
    """
    if not engagement.proven:
        return Decision(
            verdict=VERDICT_UNINFORMATIVE,
            reason=f"engagement guard failed: {engagement.reason}",
        )

    if not ruling.allowed:
        return Decision(
            verdict=VERDICT_UNINFORMATIVE,
            reason=(
                f"backend guard failed ({ruling.status}): "
                f"{ruling.name or 'unresolved'} — {ruling.reason}"
            ),
        )

    comparisons = comparisons or {}

    if completed_runs < MIN_RUNS_FOR_COMPARISON:
        return Decision(
            verdict=VERDICT_UNINFORMATIVE,
            reason=(
                f"only {completed_runs} replay(s) completed; "
                f"{MIN_RUNS_FOR_COMPARISON} are needed to compare outputs"
            ),
        )

    evidenced = [c for c in comparisons.values() if c.evidenced]
    if not evidenced:
        return Decision(
            verdict=VERDICT_UNINFORMATIVE,
            reason=(
                "no comparable deterministic request produced output in at "
                "least two replays — the replay yielded no evidence either way"
            ),
        )

    internal = sorted(c.request_id for c in evidenced if c.unique_outputs >= 2)
    if internal:
        return Decision(
            verdict=VERDICT_CORROBORATES,
            reason=(
                f"{len(internal)} request(s) produced >= 2 distinct outputs "
                f"across {completed_runs} BI replays — the divergence survives "
                "the intervention set"
            ),
            diverged_requests=internal,
        )

    reason = (
        f"all {len(evidenced)} comparable request(s) produced identical "
        f"output across {completed_runs} BI replays"
    )
    mismatched = [c for c in evidenced if c.reference_mismatch]
    if mismatched:
        reason += (
            f"; {len(mismatched)} of them differ from the finding's recorded "
            "reference output, which is recorded as an observation only and is "
            "not evidence (the reference predates the intervention set)"
        )

    return Decision(
        verdict=VERDICT_DOES_NOT_CORROBORATE,
        reason=reason,
        caveat=EVAPORATION_CAVEAT,
    )


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------


def build_sidecar(
    *,
    finding_path: Path,
    base_url: str,
    requested_runs: int,
    completed_runs: int,
    engagement: EngagementProof,
    ruling: BackendRuling,
    decision: Decision,
    comparisons: Optional[dict[str, RequestComparison]] = None,
    excluded: Optional[dict[str, str]] = None,
    server_version: str = "",
    wall_time_s: float = 0.0,
) -> dict:
    """Assemble the R2 sidecar payload. Pure — writes nothing."""
    comparisons = comparisons or {}
    payload: dict = {
        "schema_version": SCHEMA_VERSION,
        "tool": "scripts/bi_confirm.py",
        "state": "R2",
        "semantics": f"{ATTRIBUTION_DOC} — {ATTRIBUTION_SECTION}",
        "finding_path": str(finding_path),
        "verdict": decision.verdict,
        "reason": decision.reason,
        "runs": {"requested": requested_runs, "completed": completed_runs},
        "guard": {
            "engagement": engagement.to_dict(),
            "backend": ruling.to_dict(),
        },
        "per_request": {
            rid: comparison.to_dict() for rid, comparison in sorted(comparisons.items())
        },
        "diverged_requests": decision.diverged_requests,
        "excluded_requests": excluded or {},
        "base_url": base_url,
        "server_version": server_version,
        "wall_time_s": round(wall_time_s, 1),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
    }
    if any(c.reference_compared for c in comparisons.values()):
        payload["reference_caveat"] = REFERENCE_MISMATCH_CAVEAT
    if decision.caveat:
        payload["caveat"] = decision.caveat
    return payload


def resolve_output_dir(finding_path: Path, explicit: Optional[str] = None) -> Path:
    """Anchor the default sidecar directory on the repo the finding lives in.

    ``--output-dir`` is honoured verbatim when given.  Otherwise the repo root
    is derived by walking up from the finding to the directory that contains
    ``findings/``, so running the tool from outside the repo writes sidecars
    beside their siblings instead of scattering them under the caller's cwd.
    """
    if explicit:
        return Path(explicit)

    for parent in finding_path.resolve().parents:
        if (parent / "findings").is_dir():
            return parent / DEFAULT_OUTPUT_DIR

    print(
        f"WARNING: no 'findings/' directory above {finding_path} — writing the "
        f"sidecar under the current directory ({Path.cwd() / DEFAULT_OUTPUT_DIR}). "
        "Pass --output-dir to place it explicitly.",
        file=sys.stderr,
    )
    return Path.cwd() / DEFAULT_OUTPUT_DIR


def write_sidecar(payload: dict, output_dir: Path, finding_path: Path) -> Path:
    """Write the sidecar next to its siblings. The finding JSON is never touched."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / (finding_path.stem + SIDECAR_SUFFIX)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, dest)
    return dest


def format_report(payload: dict) -> str:
    """Human-readable console summary of a sidecar payload."""
    guard = payload["guard"]
    lines = [
        "=" * 68,
        f"  R2 BI CONFIRMATION : {payload['verdict']}",
        "=" * 68,
        f"  finding   : {payload['finding_path']}",
        f"  runs      : {payload['runs']['completed']}/{payload['runs']['requested']}",
        f"  engagement: {'PROVEN' if guard['engagement']['proven'] else 'NOT PROVEN'}"
        f" — {guard['engagement']['reason']}",
        f"  backend   : {guard['backend']['name'] or 'unresolved'}"
        f" [{guard['backend']['status']}] via {guard['backend']['resolved_from']}",
        f"  reason    : {payload['reason']}",
    ]
    per_request = payload.get("per_request") or {}
    if per_request:
        lines.append("  per-request uniqueness:")
        for rid, block in list(per_request.items())[:10]:
            lines.append(
                f"    {rid}: {block['unique_outputs']} unique / "
                f"{block['runs_observed']} runs"
                + (
                    f", reference_mismatch={block['reference_mismatch']}"
                    " (observation only)"
                    if block["reference_source"]
                    else ""
                )
            )
        if len(per_request) > 10:
            lines.append(f"    ... ({len(per_request)} requests total)")
    if payload.get("caveat"):
        lines.append(f"  caveat    : {payload['caveat']}")
    lines.append("=" * 68)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server probe
# ---------------------------------------------------------------------------


async def _get_json(url: str, timeout_s: float = 5.0) -> Optional[dict]:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return None
            data = response.json()
            return data if isinstance(data, dict) else None
    except Exception:
        return None


async def probe_server(base_url: str) -> tuple[str, Optional[dict]]:
    """Return ``(server_version, server_info)``; both best effort."""
    root = base_url.rstrip("/")
    version_doc = await _get_json(f"{root}/version")
    info = await _get_json(f"{root}/server_info")
    version = str((version_doc or {}).get("version", ""))
    return version, info


def _load_context_manifest(finding_path: Path) -> Optional[dict]:
    manifest = finding_path.with_name(f"{finding_path.stem}_context") / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------


async def _async_main(args) -> int:
    finding_path = Path(args.finding_path)
    if not finding_path.exists():
        print(f"ERROR: finding not found: {finding_path}", file=sys.stderr)
        return 3

    try:
        finding = json.loads(finding_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read finding: {exc}", file=sys.stderr)
        return 3

    if "trace" not in finding:
        print("ERROR: finding JSON has no 'trace' field", file=sys.stderr)
        return 3

    wall_start = time.time()

    # ---- Guard 1: engagement ------------------------------------------------
    engagement = prove_engagement(args.vllm_log_file)

    log_text = ""
    if args.vllm_log_file and Path(args.vllm_log_file).exists():
        try:
            log_text = Path(args.vllm_log_file).read_text(errors="replace")
        except OSError:
            log_text = ""

    # ---- Guard 2: backend allowlist ----------------------------------------
    server_version, server_info = await probe_server(args.base_url)
    context_manifest = _load_context_manifest(finding_path)
    backend_raw, fa_version, backend_source = resolve_backend(
        finding,
        context_manifest=context_manifest,
        log_text=log_text,
        server_info=server_info,
        override=args.backend,
    )
    # Provenance for the FA version, so the sidecar never claims "declared"
    # for a value that was actually read out of the server log.
    if args.fa_version is not None:
        fa_version = args.fa_version
        fa_source = "declared (--fa-version)"
    elif fa_version is not None:
        fa_source = "server log (flash_attn.py:789) or campaign flags"
    else:
        fa_source = "declared"
    arch = resolve_gpu_arch(finding, log_text=log_text, override=args.gpu_arch)
    ruling = rule_backend(
        backend_raw,
        arch=arch,
        fa_version=fa_version,
        resolved_from=backend_source,
        fa_version_source=fa_source,
    )

    output_dir = resolve_output_dir(finding_path, args.output_dir)

    if not engagement.proven or not ruling.allowed:
        decision = decide(engagement, ruling)
        payload = build_sidecar(
            finding_path=finding_path,
            base_url=args.base_url,
            requested_runs=args.runs,
            completed_runs=0,
            engagement=engagement,
            ruling=ruling,
            decision=decision,
            server_version=server_version,
            wall_time_s=time.time() - wall_start,
        )
        dest = write_sidecar(payload, output_dir, finding_path)
        print(format_report(payload))
        print(f"  sidecar   : {dest}")
        return EXIT_CODES[decision.verdict]

    # ---- Replay -------------------------------------------------------------
    normalized = normalize_for_bi(Trace.from_dict(finding["trace"]), args.logprobs)
    references = load_reference_outputs(finding)
    executor = PythonExecutor(
        request_timeout_s=args.request_timeout_s,
        trace_timeout_s=args.request_timeout_s + 30,
    )

    print(
        f"R2 BI replay: {finding_path.name} x {args.runs} against {args.base_url}"
    )
    print(f"  backend   : {ruling.name} FA{ruling.fa_version} [{ruling.status}]")
    print(f"  engagement: {engagement.signal} ({engagement.source})")
    print(f"  comparable: {len(normalized.comparable_request_ids)} request(s)")

    results: list[TraceResult] = []
    for i in range(args.runs):
        print(f"  run {i + 1}/{args.runs} ...", end=" ", flush=True)
        try:
            result = await executor.run(normalized.trace, args.base_url)
            results.append(result)
            errors = sum(1 for r in result.requests if r.error is not None)
            print(f"done ({len(result.requests)} requests, {errors} errors)")
        except Exception as exc:
            print(f"FAILED: {exc}")
        await asyncio.sleep(args.settle_s)

    comparisons = compare_runs(
        results, normalized.comparable_request_ids, references
    )
    decision = decide(engagement, ruling, comparisons, completed_runs=len(results))

    payload = build_sidecar(
        finding_path=finding_path,
        base_url=args.base_url,
        requested_runs=args.runs,
        completed_runs=len(results),
        engagement=engagement,
        ruling=ruling,
        decision=decision,
        comparisons=comparisons,
        excluded=normalized.excluded,
        server_version=server_version,
        wall_time_s=time.time() - wall_start,
    )
    dest = write_sidecar(payload, output_dir, finding_path)
    print()
    print(format_report(payload))
    print(f"  sidecar   : {dest}")
    return EXIT_CODES[decision.verdict]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "R2 batch-invariance confirmation runner — three-state "
            "corroborating evidence for one finding."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("finding_path", help="Path to finding_*.json")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of a BI-enabled vLLM server (VLLM_BATCH_INVARIANT=1)",
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="Replays of the trace"
    )
    parser.add_argument(
        "--vllm-log-file",
        default=None,
        help="Server log file — the only accepted proof that BI initialized",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Attention backend override (else read from the finding, then probed)",
    )
    parser.add_argument(
        "--fa-version",
        type=int,
        default=None,
        choices=[2, 3, 4],
        help="FlashAttention version override (else inferred from the GPU arch)",
    )
    parser.add_argument(
        "--gpu-arch",
        default=None,
        help="GPU arch override, e.g. sm90 / H100 (else best effort)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            f"Directory for the verdict sidecar; when unset it is anchored to "
            f"<repo>/{DEFAULT_OUTPUT_DIR}, with <repo> derived from the "
            f"finding path"
        ),
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        default=DEFAULT_LOGPROBS,
        help="Top-k logprobs requested per comparable request",
    )
    parser.add_argument(
        "--request-timeout-s", type=float, default=30.0, help="Per-request timeout"
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=1.0,
        help="Sleep between replays so the server queue drains",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
