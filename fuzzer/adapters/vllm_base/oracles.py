"""
vLLM-specific oracle checks as standalone functions.

These were originally methods of the Oracle class but are vLLM-specific
enough to belong in the adapter layer. They are also still kept in the
Oracle class for backward compatibility.
"""

from __future__ import annotations

from ...core.types import BugClass, OracleResult


def check_kv_block_leak(result, session) -> list[OracleResult]:
    """Check for KV cache block leaks after a trace completes.

    Drain-gated, exactly like ``Oracle._check_kv_block_leak`` — see that
    method for why the gate is load-bearing.  This copy is currently unused by
    the runner but is kept in sync deliberately: an ungated copy sitting beside
    the gated one is how the defect comes back.
    """
    baseline = session.kv_cache_baseline_pct()
    final    = session.kv_cache_final_pct()
    if baseline is None or final is None:
        return [OracleResult(passed=True, detail="no kv cache data",
                             unmeasured="kv_block_leak: no kv cache data")]

    drained = session.kv_cache_final_drained()
    if drained is None:
        return [OracleResult(
            passed=True,
            detail="drain state unknown — no queue telemetry in the snapshot "
                   "that supplied the final KV reading",
            unmeasured="kv_block_leak: no queue telemetry",
        )]
    if not drained:
        q = session.final_queue_state() or (0, 0)
        return [OracleResult(
            passed=True,
            detail=f"engine not drained (running={q[0]} waiting={q[1]}) — "
                   f"KV retention not measurable; final reading {final:.1f}% "
                   f"is in-flight occupancy, not a leak",
            unmeasured=f"kv_block_leak: engine not drained "
                       f"(running={q[0]} waiting={q[1]})",
        )]

    delta = final - baseline
    kv_block_leak_pct = 5.0  # default threshold
    if delta > kv_block_leak_pct:
        return [OracleResult(
            passed=False,
            bug_class=BugClass.KV_BLOCK_LEAK,
            detail=f"KV cache usage grew {delta:.1f}% after scenario drained "
                   f"(baseline {baseline:.1f}% → final {final:.1f}%, "
                   f"engine idle: running=0 waiting=0)",
            score=delta / 10.0,
        )]
    return [OracleResult(passed=True)]


def check_determinism(actual: str, expected: str) -> OracleResult:
    """Determinism oracle: with temperature=0, identical prompts must produce identical output."""
    if actual != expected:
        return OracleResult(
            passed=False,
            bug_class=BugClass.KV_CORRUPTION,
            detail="Non-deterministic output for identical prompt/temperature=0 "
                   "(possible KV cache corruption)",
            score=10.0,
        )
    return OracleResult(passed=True)


def check_cache_hit_proxy(warm_ttft_ms: float,
                           probe_ttft_ms: float,
                           multiplier: float = 4.0) -> OracleResult:
    """Cache-hit proxy oracle: after warming the prefix cache, a re-probe should be fast."""
    threshold = max(warm_ttft_ms * multiplier, 500.0)
    if probe_ttft_ms > threshold:
        return OracleResult(
            passed=False,
            bug_class=BugClass.CACHE_EVICTION,
            detail=f"Cache probe TTFT {probe_ttft_ms:.0f} ms vs warm baseline "
                   f"{warm_ttft_ms:.0f} ms — possible incorrect block eviction",
            score=probe_ttft_ms / max(warm_ttft_ms, 1.0),
        )
    return OracleResult(passed=True)
