"""Replay saved workloads through the same executor and contract oracles."""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from .core.finding_sig import mechanism_signature
from .core.oracles import Oracle
from .executor import PythonExecutor
from .monitor import EmptyMonitor, NvmlMonitor
from .preflight import check_server
from .trace import Trace


# These require controlled comparisons, a baseline, or a separate collector.
# A successful HTTP replay alone cannot establish them.
RELATIONAL_CLASSES = {
    "kv_corruption", "adapter_isolation_failure", "inconsistency",
    "latency_spike", "cache_eviction", "probe_self_check",
}


def load_finding(path, original=False):
    report = json.loads(Path(path).read_text())
    data = report.get("trace_original", report.get("trace")) if original else report.get("trace")
    if data is None and "events" in report:
        data = report
    if not isinstance(data, dict):
        raise ValueError("expected a finding with a trace, or a trace JSON")
    trace = Trace.from_dict(data)
    trace.validate()
    if not trace.events:
        raise ValueError("finding contains an empty probe trace; rerun its confirmation configuration")
    return report, trace


def judge_replay(report, trace, observations):
    original = report.get("bugs_detected") or report.get("bugs") or []
    expected = {b["class"] for b in original if b.get("class")}
    observed = {b.bug_class.value for b in observations if b.bug_class and not b.passed}
    actual_sig = mechanism_signature(trace, observations) if observations else None
    if not expected:
        verdict = "executed"
    elif expected & RELATIONAL_CLASSES:
        verdict = "inconclusive"
    elif report.get("mech_sig") and actual_sig == report["mech_sig"]:
        verdict = "signature_reproduced"
    elif expected & observed:
        verdict = "same_class_observed"
    else:
        verdict = "not_reproduced"
    return {"verdict": verdict, "expected_classes": sorted(expected),
            "observed_classes": sorted(observed), "observed_signature": actual_sig}


async def replay(path, args, *, runs=3, original=False):
    from .__main__ import _BACKENDS
    report, trace = load_finding(path, original)
    if report.get("demo"):
        raise ValueError("this is a simulated teaching fixture, not a live-server finding")
    if report.get("engine") == "sglang" and not args.sglang:
        raise ValueError("this finding used SGLang; pass --sglang and the original adapter flags")
    info = await check_server(args.base_url, args.model_name)
    adapter = next(a for backend in _BACKENDS if (a := backend.build_adapter(args)) is not None)
    executor = PythonExecutor(adapter=adapter, request_timeout_s=args.request_timeout_s)
    executor.model_name = info["model"]
    executor.prompt_embeds_dim = args.prompt_embeds_dim or None
    monitor = (EmptyMonitor() if args.mock_monitor else NvmlMonitor(
        metrics_url=args.base_url.rstrip("/") + "/metrics", local_gpu=args.local_gpu_metrics))
    oracle = Oracle()
    oracle.memory_leak_tolerance_mb = args.memory_leak_tolerance_mb
    oracle.latency_ttft_threshold_ms = args.latency_threshold_ms
    oracle.request_timeout_s = args.request_timeout_s
    outcomes = []
    for _ in range(runs):
        await monitor.start()
        try:
            result = await executor.run(trace, args.base_url)
        finally:
            session = await monitor.stop()
        findings = oracle.evaluate(result, session)
        findings += adapter.evaluate_extra(result, session, trace=trace)
        findings = [f for f in adapter.filter_findings(findings) if not f.passed]
        outcome = judge_replay(report, trace, findings)
        errors = [r.error for r in result.requests if r.error]
        outcome.update(evidence=[asdict(f) for f in findings], request_errors=errors)
        # A connection failure is not evidence that a recorded bug disappeared.
        if errors and len(errors) == len(result.requests):
            outcome["verdict"] = "inconclusive"
        outcomes.append(outcome)
        await asyncio.sleep(0.1)
    if hasattr(monitor, "aclose"):
        await monitor.aclose()
    return {"schema_version": 1, "finding": str(Path(path).resolve()),
            "base_url": args.base_url, "trace_source": "original" if original else "stored",
            "runs": outcomes,
            "scope": "Contract replay. Relational findings require the controlled confirmation pipeline; "
                     "a matching signature does not prove a common root cause."}
