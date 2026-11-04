"""
LoRA-specific oracle: cross-adapter KV isolation check.

Invariant: at temperature=0, two requests using different LoRA adapters
but the same prompt shape must produce different outputs. Random LoRA weights
with different seeds are mathematically guaranteed to produce different logit
distributions, so identical outputs AND identical logprobs = real KV contamination.

Fast-path tiebreaker logic (check_cross_adapter_isolation):
  - outputs differ                          → no bug (adapters are isolated)
  - outputs match, logprobs differ          → false positive (greedy collision;
                                              both adapters shifted logits but
                                              argmax stayed the same)
  - outputs match, logprobs match           → score 7 (high-confidence candidate;
                                              queued for probe confirmation —
                                              not proof alone, random adapters can
                                              still coincide on easy prompts)
  - outputs match, logprobs unavailable     → score 3 (observability gap;
                                              candidate for structured probe)

Structured probe (adapter_isolation_probe on the adapter class):
  Solo-vs-concurrent invariant:
    1. Run adapter A alone k times → compute modal(solo_A_outputs)
    2. Run A+B concurrently once
    3. Gate: concurrent_A != modal_solo_A
    4. Attribution: sim(concurrent_A, B) - sim(concurrent_A, modal_A) > 0.25
    5. Score ladder: 3 (no logprobs) / 7 (gate+attribution) / 8 (+position) / 10 (+logprob margin)

Score ladder summary:
  0   → gate not met (no finding)
  3   → gate met, no logprobs (observability gap, not evidence)
  7   → gate met + attribution confirmed
  8   → above + divergence_pos_ratio >= 0.5
  10  → above + logprob margin >= 1.0
"""

from __future__ import annotations

import math

from ...core.types import BugClass, OracleResult, TraceResult

# Two logprob sequences are "identical" if every value matches within this
# tolerance. IEEE 754 float32 values coming back as JSON should be bit-for-bit
# identical if the same computation was done; 1e-5 gives a tiny margin for
# JSON serialisation rounding while remaining far tighter than any real ΔW
# perturbation from a rank-8 LoRA adapter.
_LOGPROB_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Similarity utilities (used by both fast-path and structured probe)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split on whitespace — cheap proxy for token boundaries."""
    return text.split()


def _jaccard_score(a: str, b: str) -> float:
    """Token-level Jaccard similarity: |A∩B| / |A∪B|. Returns 1.0 if both empty."""
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta and not tb:
        return 1.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _prefix_score(a: str, b: str) -> float:
    """Fraction of tokens in common prefix relative to max length."""
    ta, tb = _tokenize(a), _tokenize(b)
    n = min(len(ta), len(tb))
    common = sum(1 for i in range(n) if ta[i] == tb[i])
    denom = max(len(ta), len(tb))
    return common / denom if denom else 1.0


def _sim(a: str, b: str) -> float:
    """Combined similarity = 0.5 * prefix_score + 0.5 * jaccard_score."""
    return 0.5 * _prefix_score(a, b) + 0.5 * _jaccard_score(a, b)


def _first_divergence_index(a_tokens: list[str], b_tokens: list[str]) -> int:
    """Index of first position where a and b differ. Returns min length if all match."""
    for i, (x, y) in enumerate(zip(a_tokens, b_tokens)):
        if x != y:
            return i
    return min(len(a_tokens), len(b_tokens))


# ---------------------------------------------------------------------------
# Logprob identity check (fast-path tiebreaker)
# ---------------------------------------------------------------------------

def _logprobs_identical(a: list[float], b: list[float]) -> bool:
    """Return True if two logprob sequences are effectively identical."""
    if len(a) != len(b):
        return False
    return all(
        abs(x - y) <= _LOGPROB_TOLERANCE
        for x, y in zip(a, b)
        if math.isfinite(x) and math.isfinite(y)
    )


# ---------------------------------------------------------------------------
# Fast-path oracle: single-trace candidate generator
# ---------------------------------------------------------------------------

def check_cross_adapter_isolation(result: TraceResult) -> list[OracleResult]:
    """
    For each (prompt_len, prefix_len) pair sent with multiple different adapters
    in the same run, check that outputs differ.

    Uses logprobs as a tiebreaker to filter greedy-decoding false positives:
      - identical output + identical logprobs → ADAPTER_ISOLATION_FAILURE (score 10)
      - identical output + different logprobs → greedy collision, skip (score 0)
      - identical output + no logprobs        → candidate for structured probe (score 3)

    Emits probe_metadata on every finding so the probe scheduler knows what to follow up.

    Only considers:
    - Completed requests (completed_at is not None)
    - Non-errored requests (error is None)
    - Requests with output_text (not None or empty)
    - Requests that have an adapter_name set (LoRA requests only)
    """
    findings: list[OracleResult] = []

    # Collect completed LoRA requests grouped by (prompt_family_id, prefix_len).
    # Only requests with an explicit prompt_family_id are considered — these are
    # the only ones guaranteed to carry byte-identical prompts across request_ids.
    # Grouping by shape alone (prompt_len, prefix_len) is insufficient because
    # _build_prompt generates different unique suffixes per request_id.
    # key → {adapter_name: [(output_text, output_logprobs|None), ...]
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
            continue   # skip requests without explicit prompt identity

        key = (req.prompt_family_id, req.prefix_len, req.prompt_len)
        groups.setdefault(key, {}).setdefault(req.adapter_name, []).append(
            (req.output_text, req.output_logprobs)
        )

    # For each prompt family, check pairs of different adapters
    for (prompt_family_id, prefix_len, prompt_len), adapter_outputs in groups.items():
        adapters = list(adapter_outputs.keys())
        if len(adapters) < 2:
            continue

        for i in range(len(adapters)):
            for j in range(i + 1, len(adapters)):
                a_name, b_name = adapters[i], adapters[j]
                a_pairs = adapter_outputs[a_name]
                b_pairs = adapter_outputs[b_name]

                for a_out, a_lps in a_pairs:
                    for b_out, b_lps in b_pairs:
                        if a_out != b_out:
                            continue  # outputs differ — adapters are isolated

                        # Outputs are identical — apply logprob tiebreaker
                        if a_lps and b_lps:
                            if not _logprobs_identical(a_lps, b_lps):
                                # Logprobs differ → greedy collision, not a bug
                                continue
                            # Logprobs also identical → high-confidence candidate.
                            # Score 7: strong fast-path evidence but not proof on its
                            # own — requires structured probe or KV structural support
                            # to reach score 10. Random adapters can still coincide
                            # on top-1 logprobs for easy prompts.
                            score = 7.0
                            verdict = (
                                f"identical output AND identical logprobs "
                                f"(high-confidence candidate — queued for probe confirmation)"
                            )
                        else:
                            # No logprobs available — observability gap, not evidence
                            # Score 3 = candidate for structured isolation probe
                            score = 3.0
                            verdict = (
                                f"identical output, logprobs unavailable "
                                f"(candidate for structured probe)"
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
                                "adapter_a": a_name,
                                "adapter_b": b_name,
                                "prompt_family_id": prompt_family_id,
                                "prompt_len": prompt_len,
                                "prefix_len": prefix_len,
                            },
                        ))

    return findings


# ---------------------------------------------------------------------------
# KV structural analysis — LoRA-specific
# ---------------------------------------------------------------------------

_CROSS_LORA_SCORE = 6.0


def analyze_cross_lora_kv(summary) -> list:
    """Score cross-LoRA block reuse from a KVProbeSummary.

    Returns a list with one anomaly dict if cross_lora_reused_blocks > 0,
    empty list otherwise. Uses the same dict schema as
    kv_events.analyze_kv_anomalies():
        anomaly_type, count, score, detail, metadata

    Called by VllmLoRAAdapter.analyze_kv_summary(). The runner merges this
    with the output of analyze_kv_anomalies() (generic layer):
        kv_anomalies = analyze_kv_anomalies(summary)
        if adapter is not None:
            kv_anomalies += adapter.analyze_kv_summary(summary)
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
