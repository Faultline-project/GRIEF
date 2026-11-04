"""
SGLang LoRA-specific oracles.

Primary oracle — solo-vs-concurrent invariant (adapter_isolation_probe on adapter class):
  For each adapter pair (A, B), run A solo k times → modal_solo_A.
  Then run A under concurrent load and compare to modal_solo_A.
  Attribution and scoring:
    toward_B       (sim_to_B − sim_to_A > 0.25)  → score 7 / 8 / 9
    base_model_drift (sim_to_base > sim_to_A)     → score 6
    unknown_drift                                  → score 4
  The base_model_drift path requires Pass 0 (adapter_name=None solo run) to
  produce modal_solo_base; without it sim_to_base defaults to 0.0.

Secondary oracle — cross-adapter fast path (check_cross_adapter_isolation):
  Fires when adapter A and B produce identical output AND identical logprobs
  for the same prompt in the same run.  This is a strict equality check and
  rarely fires in practice (SGLang correctly isolates KV via extra_key), but
  is kept as a secondary signal for genuine greedy collisions.
  Score 7 (identical output + logprobs) / 3 (no logprobs).

KV structural signal (analyze_cross_lora_kv):
  Score 6 when cross_lora_reused_blocks > 0 in a probe window.
  +1 KV boost applied by the probe coordinator to findings with score >= 7.
"""

from __future__ import annotations

import math

from ...core.types import BugClass, OracleResult, TraceResult

_LOGPROB_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return text.split()


def _jaccard_score(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _prefix_score(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    n = min(len(ta), len(tb))
    common = sum(1 for i in range(n) if ta[i] == tb[i])
    denom = max(len(ta), len(tb))
    return common / denom if denom else 1.0


def _sim(a: str, b: str) -> float:
    return 0.5 * _prefix_score(a, b) + 0.5 * _jaccard_score(a, b)


def _first_divergence_index(a_tokens: list[str], b_tokens: list[str]) -> int:
    for i, (x, y) in enumerate(zip(a_tokens, b_tokens)):
        if x != y:
            return i
    return min(len(a_tokens), len(b_tokens))


# ---------------------------------------------------------------------------
# Logprob identity check
# ---------------------------------------------------------------------------

def _logprobs_identical(a: list[float], b: list[float]) -> bool:
    if len(a) != len(b):
        return False
    return all(
        abs(x - y) <= _LOGPROB_TOLERANCE
        for x, y in zip(a, b)
        if math.isfinite(x) and math.isfinite(y)
    )


# ---------------------------------------------------------------------------
# Fast-path oracle
# ---------------------------------------------------------------------------

def check_cross_adapter_isolation(result: TraceResult) -> list[OracleResult]:
    """
    For each (prompt_family_id, prefix_len, prompt_len) group sent with multiple
    different adapters in the same run, check that outputs differ.

    Only considers completed, non-errored requests with output_text and
    adapter_name set, and with an explicit prompt_family_id.
    """
    findings: list[OracleResult] = []

    groups: dict[tuple, dict[str, list[tuple]]] = {}
    for req in result.requests:
        if req.completed_at is None:
            continue
        if req.error is not None:
            continue
        if not req.output_text:
            continue
        if req.adapter_name is None:
            continue
        if not getattr(req, "prompt_family_id", None):
            continue

        key = (req.prompt_family_id, req.prefix_len, req.prompt_len)
        groups.setdefault(key, {}).setdefault(req.adapter_name, []).append(
            (req.output_text, req.output_logprobs)
        )

    for (prompt_family_id, prefix_len, prompt_len), adapter_outputs in groups.items():
        adapters = list(adapter_outputs.keys())
        if len(adapters) < 2:
            continue

        for i in range(len(adapters)):
            for j in range(i + 1, len(adapters)):
                a_name, b_name = adapters[i], adapters[j]
                for a_out, a_lps in adapter_outputs[a_name]:
                    for b_out, b_lps in adapter_outputs[b_name]:
                        if a_out != b_out:
                            continue

                        if a_lps and b_lps:
                            if not _logprobs_identical(a_lps, b_lps):
                                continue
                            score   = 7.0
                            verdict = (
                                "identical output AND identical logprobs "
                                "(high-confidence candidate — queued for probe confirmation)"
                            )
                        else:
                            score   = 3.0
                            verdict = (
                                "identical output, logprobs unavailable "
                                "(candidate for structured probe)"
                            )

                        findings.append(OracleResult(
                            passed=False,
                            bug_class=BugClass.ADAPTER_ISOLATION_FAILURE,
                            detail=(
                                f"adapter '{a_name}' and '{b_name}' produced "
                                f"{verdict} for prompt_family={prompt_family_id} "
                                f"prompt_len={prompt_len} prefix_len={prefix_len}: "
                                f"{repr(a_out[:120])}"
                            ),
                            score=score,
                            probe_metadata={
                                "adapter_a":        a_name,
                                "adapter_b":        b_name,
                                "prompt_family_id": prompt_family_id,
                                "prompt_len":       prompt_len,
                                "prefix_len":       prefix_len,
                            },
                        ))

    return findings


# ---------------------------------------------------------------------------
# KV structural analysis — cross-adapter block reuse
# ---------------------------------------------------------------------------

_CROSS_LORA_SCORE = 6.0


def analyze_cross_lora_kv(summary) -> list:
    """Score cross-LoRA block reuse from a KVProbeSummary.

    Returns one anomaly dict if cross_lora_reused_blocks > 0, else [].
    Uses the same dict schema as kv_events.analyze_kv_anomalies():
        anomaly_type, count, score, detail, metadata

    SGLang emits lora_id at index 5 of BlockStored events. kv_events.py
    parses it and computes cross_lora_reused_blocks (blocks seen under 2+
    distinct lora_ids within the probe window). Score 6.0 is above the
    runner's 5.0 structural-queue threshold, so confirmed cross-adapter
    block reuse reaches the structural confirmation queue even without a
    co-occurring behavioral finding.
    """
    if not summary.fingerprint:
        return []
    if summary.cross_lora_reused_blocks <= 0:
        return []
    return [{
        "anomaly_type": "cross_lora_reuse",
        "count":  summary.cross_lora_reused_blocks,
        "score":  _CROSS_LORA_SCORE,
        "detail": (
            f"{summary.cross_lora_reused_blocks} block(s) stored under "
            f"2+ distinct lora_ids in trace window "
            f"[{summary.start_t:.3f}, {summary.end_t:.3f}]"
        ),
        "metadata": {
            "cross_lora_reused_blocks": summary.cross_lora_reused_blocks,
            "conflicting_hashes":       summary.conflicting_hashes[:10],
        },
    }]
