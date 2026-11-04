"""Tests for LoRA cross-adapter isolation oracle."""

from __future__ import annotations

import math
import pytest
from fuzzer.adapters.vllm_lora.oracles import (
    check_cross_adapter_isolation,
    analyze_cross_lora_kv,
    _sim,
    _first_divergence_index,
    _jaccard_score,
    _prefix_score,
)
from fuzzer.core.types import BugClass, TraceResult, RequestResult
from fuzzer.kv_events import KVProbeSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_req(
    request_id: str,
    prompt_len: int,
    prefix_len: int,
    output_text: str,
    adapter_name: str | None,
    error: str | None = None,
    completed: bool = True,
    output_logprobs: list[float] | None = None,
    prompt_family_id: str | None = None,
) -> RequestResult:
    return RequestResult(
        request_id=request_id,
        sent_at=0.0,
        completed_at=1.0 if completed else None,
        prompt_len=prompt_len,
        prefix_len=prefix_len,
        adapter_name=adapter_name,
        output_text=output_text if not error else None,
        error=error,
        output_logprobs=output_logprobs,
        prompt_family_id=prompt_family_id,
    )


def make_result(requests: list[RequestResult]) -> TraceResult:
    return TraceResult(trace_seed=1, started_at=0.0, finished_at=1.0, requests=requests)


# Shared family id for tests that expect findings
_FAM = "test_family_0"
_FAM2 = "test_family_1"


# ---------------------------------------------------------------------------
# No bug when outputs differ
# ---------------------------------------------------------------------------

def test_different_outputs_no_finding():
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="hello",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="world",
                 adapter_name="lora_b", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_single_adapter_no_finding():
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="hello",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="hello",
                 adapter_name="lora_a", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_no_lora_requests_no_finding():
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="hello",
                 adapter_name=None, prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="hello",
                 adapter_name=None, prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_no_prompt_family_id_no_finding():
    """Requests without prompt_family_id must be skipped — prompt identity unknown."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", prompt_family_id=None),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", prompt_family_id=None),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


# ---------------------------------------------------------------------------
# Bug detected when outputs are identical across adapters
# ---------------------------------------------------------------------------

def test_identical_output_same_shape_is_finding():
    """The positive case, asserted once instead of three times over the same
    two-request fixture (was ..._is_finding / ..._detail_mentions_adapters /
    ..._score_positive)."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same output",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same output",
                 adapter_name="lora_b", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert len(findings) == 1
    assert findings[0].passed is False
    assert findings[0].bug_class == BugClass.ADAPTER_ISOLATION_FAILURE
    assert findings[0].score > 0.0
    assert "lora_a" in findings[0].detail or "lora_b" in findings[0].detail


def test_finding_has_probe_metadata():
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", output_logprobs=None, prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", output_logprobs=None, prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings[0].probe_metadata is not None
    meta = findings[0].probe_metadata
    assert meta["adapter_a"] == "lora_a"
    assert meta["adapter_b"] == "lora_b"
    assert meta["prompt_family_id"] == _FAM
    assert meta["prefix_len"] == 16
    assert meta["prompt_len"] == 32


# ---------------------------------------------------------------------------
# Filtering: incomplete / errored / no output requests skipped
# ---------------------------------------------------------------------------

def test_incomplete_request_skipped():
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="abc",
                 adapter_name="lora_a", completed=False, prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="abc",
                 adapter_name="lora_b", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_errored_request_skipped():
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="abc",
                 adapter_name="lora_a", error="timeout", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="abc",
                 adapter_name="lora_b", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_empty_output_skipped():
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="abc",
                 adapter_name="lora_b", prompt_family_id=_FAM),
    ])
    # r1 has empty output; only r2 remains — no pair to compare
    findings = check_cross_adapter_isolation(result)
    assert findings == []


# ---------------------------------------------------------------------------
# Different prompt families don't cross-compare
# ---------------------------------------------------------------------------

def test_different_prompt_families_not_compared():
    """Same output text but different prompt_family_id — not a bug."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", prompt_family_id="family_A"),
        make_req("r2", prompt_len=64, prefix_len=32, output_text="same",
                 adapter_name="lora_b", prompt_family_id="family_B"),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_different_prefix_len_not_compared():
    """Same family but different prefix_len — requests are in different oracle groups."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=24, output_text="same",
                 adapter_name="lora_b", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_different_prompt_len_not_compared():
    """Same family and prefix_len but different prompt_len — different prompts, no finding."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=64, prefix_len=16, output_text="same",
                 adapter_name="lora_b", prompt_family_id=_FAM),
    ])
    # Different prompt_len → different suffix lengths → different prompts → different groups
    findings = check_cross_adapter_isolation(result)
    assert findings == []


# ---------------------------------------------------------------------------
# Multiple findings from multiple shape groups
# ---------------------------------------------------------------------------

def test_one_shape_contaminated_other_clean():
    result = make_result([
        # contaminated family
        make_req("r1", prompt_len=32, prefix_len=16, output_text="bad",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="bad",
                 adapter_name="lora_b", prompt_family_id=_FAM),
        # clean family
        make_req("r3", prompt_len=64, prefix_len=32, output_text="good-a",
                 adapter_name="lora_a", prompt_family_id=_FAM2),
        make_req("r4", prompt_len=64, prefix_len=32, output_text="good-b",
                 adapter_name="lora_b", prompt_family_id=_FAM2),
    ])
    findings = check_cross_adapter_isolation(result)
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# Three adapters — all pairs checked
# ---------------------------------------------------------------------------

def test_three_adapters_all_contaminated_generates_three_findings():
    """With adapters A, B, C all producing the same output: pairs (A,B), (A,C), (B,C)."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", prompt_family_id=_FAM),
        make_req("r3", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_c", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert len(findings) == 3


def test_three_adapters_one_different():
    """A and B contaminated (same output), C is different — only (A,B) fires."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", prompt_family_id=_FAM),
        make_req("r3", prompt_len=32, prefix_len=16, output_text="different",
                 adapter_name="lora_c", prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_empty_result_no_findings():
    result = make_result([])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


# ---------------------------------------------------------------------------
# Logprob tiebreaker
# ---------------------------------------------------------------------------

def test_same_output_same_logprobs_is_high_confidence_candidate():
    """Identical outputs AND identical logprobs = high-confidence candidate (score 7).

    Score 10 is reserved for findings confirmed by the structured probe or
    KV structural evidence. The fast path alone cannot guarantee contamination
    for random adapters even when logprobs match.
    """
    lps = [-0.1, -0.2, -0.3]
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", output_logprobs=lps, prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", output_logprobs=lps, prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert len(findings) == 1
    assert findings[0].score == 7.0
    assert "high-confidence" in findings[0].detail


def test_same_output_no_logprobs_is_probe_candidate():
    """Identical outputs with no logprobs = observability gap; candidate for structured probe."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", output_logprobs=None, prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", output_logprobs=None, prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert len(findings) == 1
    assert findings[0].score == 3.0
    assert "candidate" in findings[0].detail


def test_logprob_tiebreaker_within_tolerance():
    """Logprobs differing by < 1e-5 are treated as identical → score 7 candidate."""
    lps_a = [-0.1, -0.2, -0.3]
    lps_b = [-0.1, -0.2, -0.3 - 1e-6]  # within tolerance
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", output_logprobs=lps_a, prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", output_logprobs=lps_b, prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert len(findings) == 1
    assert findings[0].score == 7.0


def test_logprob_tiebreaker_outside_tolerance():
    """Logprobs differing by > 1e-5 are treated as different (greedy collision)."""
    lps_a = [-0.1, -0.2, -0.3]
    lps_b = [-0.1, -0.2, -0.3 - 1e-4]  # outside tolerance
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", output_logprobs=lps_a, prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", output_logprobs=lps_b, prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


def test_different_logprob_lengths_treated_as_different():
    """Mismatched logprob list lengths → treated as different → false positive filtered."""
    result = make_result([
        make_req("r1", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_a", output_logprobs=[-0.1, -0.2],
                 prompt_family_id=_FAM),
        make_req("r2", prompt_len=32, prefix_len=16, output_text="same",
                 adapter_name="lora_b", output_logprobs=[-0.1, -0.2, -0.3],
                 prompt_family_id=_FAM),
    ])
    findings = check_cross_adapter_isolation(result)
    assert findings == []


# ---------------------------------------------------------------------------
# analyze_cross_lora_kv — KV structural analysis
# ---------------------------------------------------------------------------

def _kv_summary(cross_lora=0, conflicting=None, fingerprint="abcdef1234567890"):
    return KVProbeSummary(
        start_t=0.0, end_t=1.0,
        stored_count=10, removed_count=5, cleared_count=0,
        distinct_blocks=8, per_lora_counts={},
        cross_lora_reused_blocks=cross_lora,
        conflicting_hashes=conflicting if conflicting is not None else (["0xabc"] * cross_lora),
        reuse_too_fast_count=0,
        reuse_too_fast_min_gap_ms=math.inf,
        fingerprint=fingerprint,
        raw_event_sample=[],
    )


class TestCrossLoraKvAnalysis:

    def test_cross_lora_kv_score(self):
        result = analyze_cross_lora_kv(_kv_summary(cross_lora=2))
        assert len(result) == 1
        r = result[0]
        assert r["anomaly_type"] == "cross_lora_reuse"
        assert r["score"] == 6.0
        assert r["count"] == 2

    def test_cross_lora_kv_zero_blocks(self):
        assert analyze_cross_lora_kv(_kv_summary(cross_lora=0)) == []

    def test_disabled_collector_returns_empty(self):
        assert analyze_cross_lora_kv(_kv_summary(cross_lora=2, fingerprint="")) == []


# ---------------------------------------------------------------------------
# Similarity helpers — _sim / _jaccard_score / _prefix_score /
# _first_divergence_index
#
# Relocated from tests/test_isolation_scheduler.py, which was deleted: its other
# 14 tests drove a hand-copied private fork of fuzzer/core/runner.py:526-540 and
# :1929-1942 (the fork had already drifted — it omits the cooldown veto at
# runner.py:1951-1954), or tested collections.Counter / dataclass defaults.
# These 11 are the only coverage these four helpers have anywhere in the suite,
# and they belong next to the module that defines them.
#
# NOTE: the real isolation-probe scheduler (runner.py:526-540, :643-644,
# :1923-1975) still has NO test. That gap is now visible instead of hidden
# behind a file named after it.
# ---------------------------------------------------------------------------

def test_sim_identical():
    assert _sim("hello world foo", "hello world foo") == 1.0


def test_sim_disjoint():
    score = _sim("aaa bbb ccc", "xxx yyy zzz")
    assert score < 0.05


def test_sim_partial_overlap():
    score = _sim("hello world", "hello foo")
    assert 0.0 < score < 1.0


def test_first_divergence_identical():
    a = ["a", "b", "c"]
    assert _first_divergence_index(a, a) == 3


def test_first_divergence_at_zero():
    assert _first_divergence_index(["x"], ["y"]) == 0


def test_first_divergence_mid():
    a = ["a", "b", "c", "d"]
    b = ["a", "b", "x", "d"]
    assert _first_divergence_index(a, b) == 2


def test_first_divergence_different_lengths():
    a = ["a", "b"]
    b = ["a", "b", "c"]
    # All of a matches b[:2], so divergence = min(len(a), len(b)) = 2
    assert _first_divergence_index(a, b) == 2


def test_jaccard_both_empty():
    assert _jaccard_score("", "") == 1.0


def test_jaccard_one_empty():
    assert _jaccard_score("hello", "") == 0.0


def test_prefix_score_full_match():
    assert _prefix_score("a b c", "a b c") == 1.0


def test_prefix_score_no_match():
    score = _prefix_score("x y z", "a b c")
    assert score == 0.0
