"""
Tests for fuzzer/core/finding_sig.py — the mechanism-level finding signature.

Two groups:

  1. ``oracle_subtype`` parsing, one case per detail format the oracles emit.
  2. ``workload_shape`` invariance: offsets, event order, prompt_len,
     max_tokens and prefix_len magnitude must not move the shape.
"""

import json
from pathlib import Path

import pytest

from fuzzer.core import finding_sig as fs
from fuzzer.core.finding_sig import (
    SIG_VERSION,
    legacy_signature,
    mechanism_signature,
    mechanism_signature_from_dict,
    oracle_subtype,
    triage_signature,
    workload_shape,
)
from fuzzer.core.types import BugClass, OracleResult
from fuzzer.trace import ActionType, Event, Trace

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _send(offset_ms=0, prompt_len=128, max_tokens=32, prefix_len=0, **kw):
    return Event(action=ActionType.SEND, offset_ms=offset_ms,
                 prompt_len=prompt_len, max_tokens=max_tokens,
                 prefix_len=prefix_len, **kw)


def _bug(bug_class, detail="", score=1.0, probe_metadata=None):
    return OracleResult(passed=False, bug_class=bug_class, detail=detail,
                        score=score, probe_metadata=probe_metadata)


# ---------------------------------------------------------------------------
# 1. oracle_subtype — one case per observed detail format
# ---------------------------------------------------------------------------

class TestOracleSubtype:
    """Detail strings are copied verbatim from the archived finding JSONs."""

    @pytest.mark.parametrize("detail,expected", [
        # observed 209x — check_determinism, oracles.py:452-458
        ("Non-deterministic output for identical prompt/temperature=0 "
         "(possible KV cache corruption, similarity=1.00)", "sim_hi"),
        ("Non-deterministic output for identical prompt/temperature=0 "
         "(possible KV cache corruption, similarity=0.50)", "sim_mid"),
        ("Non-deterministic output for identical prompt/temperature=0 "
         "(possible KV cache corruption, similarity=0.08)", "sim_lo"),
        # observed 22x — older probe, no similarity in the string
        ("Non-deterministic output for identical prompt/temperature=0 "
         "(possible KV cache corruption)", "sim_na"),
        # observed 2x — check_stability, oracles.py:495
        ("Validated output instability: target_rate=1.00 baseline_rate=0.00 "
         "delta=1.00", "stability"),
        # observed 6x — retired trailing-probe oracle, march_24(fp)
        ("trailing probe missing from output of request r3: expected "
         "'PROBE7F2' in response (output='7')", "probe_missing"),
    ])
    def test_kv_corruption(self, detail, expected):
        assert oracle_subtype(BugClass.KV_CORRUPTION, detail) == expected

    def test_kv_corruption_prefers_probe_metadata_over_detail(self):
        detail = ("Non-deterministic output for identical prompt/temperature=0 "
                  "(possible KV cache corruption)")
        assert oracle_subtype(BugClass.KV_CORRUPTION, detail,
                              {"similarity": 0.95}) == "sim_hi"

    # -- two-path prefix-cache probe (Runner._prefix_equivalence_probe) -----

    _PFX_DETAIL = ("Prefix-cache path divergence: identical greedy prompt "
                   "produced different output when routed around the prefix "
                   "cache via cache_salt (similarity=0.60)")

    def test_prefix_equiv_probe_is_its_own_subtype(self):
        assert oracle_subtype(
            BugClass.KV_CORRUPTION, self._PFX_DETAIL,
            {"probe": "prefix_equiv", "similarity": 0.6,
             "logprob_gap": None}) == "pfx_divergent"

    def test_prefix_equiv_wins_over_gap_and_similarity_bands(self):
        """The probe key is checked first: a gap left over from another probe
        (or the similarity fallback) must not re-band a relational finding."""
        for pm in ({"probe": "prefix_equiv", "logprob_gap": 4.9},
                   {"probe": "prefix_equiv", "similarity": 0.95},
                   {"probe": "prefix_equiv"}):
            assert oracle_subtype(BugClass.KV_CORRUPTION,
                                  self._PFX_DETAIL, pm) == "pfx_divergent"

    def test_prefix_equiv_does_not_collapse_into_determinism_subtypes(self):
        pfx = oracle_subtype(BugClass.KV_CORRUPTION, self._PFX_DETAIL,
                             {"probe": "prefix_equiv"})
        det_tie = oracle_subtype(BugClass.KV_CORRUPTION, "x", {"logprob_gap": 0.1})
        det_wide = oracle_subtype(BugClass.KV_CORRUPTION, "x", {"logprob_gap": 4.9})
        assert pfx not in (det_tie, det_wide)
        assert (det_tie, det_wide) == ("gtie", "gwide")

    def test_other_probes_are_unaffected(self):
        """Only probe == "prefix_equiv" takes the branch."""
        assert oracle_subtype(BugClass.KV_CORRUPTION, "x",
                              {"probe": "determinism", "logprob_gap": 4.9}) == "gwide"

    @pytest.mark.parametrize("detail,expected", [
        # observed 24x — with baseline, oracles.py:206-209
        ("p99 TTFT 34640 ms exceeds expected 1401 ms (24.7x) at concurrency=8",
         "c3"),
        ("p99 TTFT 9738 ms exceeds expected 500 ms (19.5x) at concurrency=2",
         "c1"),
        ("p99 TTFT 9738 ms exceeds expected 500 ms (19.5x) at concurrency=1",
         "c0"),
        # observed 51x — no baseline, fixed threshold, oracles.py:220
        ("p99 TTFT 3021 ms exceeds threshold 500 ms", "cna"),
    ])
    def test_latency_spike(self, detail, expected):
        assert oracle_subtype(BugClass.LATENCY_SPIKE, detail) == expected

    def test_latency_ratio_is_not_in_the_subtype(self):
        """§2.3: the ratio band is deliberately excluded from the key."""
        a = oracle_subtype(BugClass.LATENCY_SPIKE,
                           "p99 TTFT 9738 ms exceeds expected 500 ms (19.5x) "
                           "at concurrency=8")
        b = oracle_subtype(BugClass.LATENCY_SPIKE,
                           "p99 TTFT 98000 ms exceeds expected 500 ms (196.0x) "
                           "at concurrency=8")
        assert a == b == "c3"

    @pytest.mark.parametrize("detail", [
        "token rate 0.200 tok/s on request r5_b (floor=0.5 tok/s, elapsed=15.0s, tokens=3)",
        "token rate 0.067 tok/s on request fill_3 (floor=0.5 tok/s, elapsed=30.0s, tokens=2)",
        "token rate 0.133 tok/s on request reuse_1_s (floor=0.5 tok/s, elapsed=15.0s, tokens=2)",
    ])
    def test_token_rate_stall(self, detail):
        """One mechanism; the request id must not fragment the key."""
        assert oracle_subtype(BugClass.TOKEN_RATE_STALL, detail) == "stall"

    @pytest.mark.parametrize("detail,expected", [
        ("tokens received after cancel on requests: ['r1', 'r2']", "post_cancel"),
        ("requests never completed: ['fill_0', 'fill_1']", "never_completed"),
    ])
    def test_lifecycle_error_splits(self, detail, expected):
        """Two genuinely different mechanisms share one BugClass."""
        assert oracle_subtype(BugClass.LIFECYCLE_ERROR, detail) == expected

    @pytest.mark.parametrize("detail,expected", [
        ("GPU memory grew by 148.0 MB after scenario drained", "mb7"),
        ("GPU memory grew by 512.0 MB after scenario drained", "mb9"),
    ])
    def test_memory_leak(self, detail, expected):
        assert oracle_subtype(BugClass.MEMORY_LEAK, detail) == expected

    @pytest.mark.parametrize("bug_class,detail", [
        (BugClass.SCHEDULER_HANG,
         "waiting queue non-empty after running queue drained"),
        (BugClass.LIVENESS_FAILURE, "server /health check failed"),
        (BugClass.CACHE_EVICTION,
         "Cache probe TTFT 2100 ms vs warm baseline 120 ms — possible "
         "incorrect block eviction"),
        (BugClass.INCONSISTENCY,
         "non-determinism on spec server: output divergence confirmed 3/5 runs"),
        (BugClass.KV_BLOCK_LEAK,
         "KV cache usage grew 12.0% after scenario drained (5.0% → 17.0%)"),
    ])
    def test_single_subtype_classes(self, bug_class, detail):
        assert oracle_subtype(bug_class, detail) == bug_class.value

    def test_unknown_class_falls_back_to_class_name(self):
        assert oracle_subtype("brand_new_class", "whatever") == "brand_new_class"
        assert oracle_subtype(None, "whatever") == "unknown"

    @pytest.mark.parametrize("bug_class,detail", [
        (BugClass.LATENCY_SPIKE, "p99 TTFT exceeds expected at concurrency="),
        (BugClass.MEMORY_LEAK, "GPU memory grew by  MB after scenario drained"),
        (BugClass.KV_CORRUPTION, "possible KV cache corruption, similarity="),
        (BugClass.KV_CORRUPTION, ""),
        (BugClass.LATENCY_SPIKE, ""),
    ])
    def test_missing_numbers_do_not_raise(self, bug_class, detail):
        """Parsing must degrade, not explode, on truncated details."""
        assert isinstance(oracle_subtype(bug_class, detail), str)

    def test_string_class_forms_accepted(self):
        """Archived JSONs carry both 'kv_corruption' and 'BugClass.KV_CORRUPTION'."""
        detail = "Non-deterministic output (possible KV cache corruption, similarity=0.10)"
        assert (oracle_subtype("kv_corruption", detail)
                == oracle_subtype("BugClass.KV_CORRUPTION", detail)
                == oracle_subtype(BugClass.KV_CORRUPTION, detail)
                == "sim_lo")

    def test_kv_structural_suffix(self):
        pm = {"kv_structural": {"anomaly_type": "reuse_too_fast"}}
        assert oracle_subtype(BugClass.KV_CORRUPTION,
                              "corruption, similarity=0.10", pm) == \
            "sim_lo+reuse_too_fast"

    def test_stack_hash_suffix(self):
        pm = {"stack_hash": "abcdef1234567890"}
        assert oracle_subtype(BugClass.SCHEDULER_HANG, "queue", pm) == \
            "scheduler_hang+stkabcdef"
        pm2 = {"hashes": {"stack": "0123456789"}}
        assert oracle_subtype(BugClass.SCHEDULER_HANG, "queue", pm2) == \
            "scheduler_hang+stk012345"

    def test_both_suffixes(self):
        pm = {"kv_structural": {"anomaly_type": "orphan_child_block"},
              "stack_hash": "ffeedd0011"}
        assert oracle_subtype(BugClass.KV_CORRUPTION, "similarity=0.99", pm) == \
            "sim_hi+orphan_child_block+stkffeedd"


# ---------------------------------------------------------------------------
# 2. workload_shape
# ---------------------------------------------------------------------------

class TestWorkloadShape:

    def test_fields(self):
        tr = Trace(seed=1, events=[
            _send(prefix_len=64, logprobs=20, n_completions=4,
                  adapter_name="lora_a"),
            _send(adapter_name="lora_b"),
            Event(action=ActionType.CANCEL, offset_ms=10),
        ])
        assert workload_shape(tr) == (True, 2, True, True, True)

    def test_negative_fields(self):
        tr = Trace(seed=1, events=[_send(), _send(prefix_len=0)])
        assert workload_shape(tr) == (False, 0, False, False, False)

    def test_n_equal_one_is_not_n_gt1(self):
        tr = Trace(seed=1, events=[_send(n_completions=1)])
        assert workload_shape(tr)[4] is False

    def test_invariant_to_offsets(self):
        a = Trace(seed=1, events=[_send(offset_ms=0), _send(offset_ms=40)])
        b = Trace(seed=1, events=[_send(offset_ms=900), _send(offset_ms=1500)])
        assert workload_shape(a) == workload_shape(b)

    def test_invariant_to_event_order(self):
        e1, e2 = _send(prompt_len=64), _send(prompt_len=2048, prefix_len=128)
        a = Trace(seed=1, events=[e1, e2])
        b = Trace(seed=1, events=[e2, e1])
        assert workload_shape(a) == workload_shape(b)

    def test_invariant_to_prompt_len_and_max_tokens(self):
        a = Trace(seed=1, events=[_send(prompt_len=128, max_tokens=32)])
        b = Trace(seed=1, events=[_send(prompt_len=24576, max_tokens=512)])
        assert workload_shape(a) == workload_shape(b)

    def test_invariant_to_prefix_len_magnitude(self):
        a = Trace(seed=1, events=[_send(prefix_len=1)])
        b = Trace(seed=1, events=[_send(prefix_len=4096)])
        assert workload_shape(a) == workload_shape(b)

    def test_invariant_to_request_ids_and_seed(self):
        a = Trace(seed=1, events=[_send(request_id="aaaa")])
        b = Trace(seed=999, events=[_send(request_id="zzzz")])
        assert workload_shape(a) == workload_shape(b)

    def test_dict_and_object_agree(self):
        tr = Trace(seed=7, events=[
            _send(prefix_len=64, logprobs=3),
            Event(action=ActionType.CANCEL, offset_ms=5),
        ])
        assert workload_shape(tr) == workload_shape(tr.to_dict())

    def test_accepts_whole_finding_dict(self):
        tr = Trace(seed=7, events=[_send(logprobs=20)])
        assert workload_shape({"trace": tr.to_dict()}) == workload_shape(tr)

    def test_empty_trace(self):
        assert workload_shape(Trace(seed=0, events=[])) == \
            (False, 0, False, False, False)


# ---------------------------------------------------------------------------
# 3. mechanism_signature over live objects
# ---------------------------------------------------------------------------

class TestMechanismSignature:

    def test_mutated_numerics_keep_one_signature(self):
        """The exact failure mode of the legacy key (runner.py:753-763)."""
        bugs = [_bug(BugClass.LATENCY_SPIKE,
                     "p99 TTFT 9738 ms exceeds expected 500 ms (19.5x) "
                     "at concurrency=8")]
        a = Trace(seed=1, events=[_send(prompt_len=2048, max_tokens=64)])
        b = Trace(seed=2, events=[_send(prompt_len=3072, max_tokens=128)])
        assert mechanism_signature(a, bugs) == mechanism_signature(b, bugs)

    def test_different_subtype_splits(self):
        tr = Trace(seed=1, events=[_send()])
        post = [_bug(BugClass.LIFECYCLE_ERROR,
                     "tokens received after cancel on requests: ['r1']")]
        never = [_bug(BugClass.LIFECYCLE_ERROR,
                      "requests never completed: ['r1']")]
        assert mechanism_signature(tr, post) != mechanism_signature(tr, never)

    def test_logprobs_bit_splits(self):
        """§2.2: the any_logprobs bit is what separates V8 from V9."""
        bugs = [_bug(BugClass.LATENCY_SPIKE,
                     "p99 TTFT 9738 ms exceeds expected 500 ms (19.5x) "
                     "at concurrency=8")]
        plain = Trace(seed=1, events=[_send()])
        amped = Trace(seed=1, events=[_send(logprobs=20, n_completions=8)])
        assert mechanism_signature(plain, bugs) != mechanism_signature(amped, bugs)

    def test_bug_order_does_not_matter(self):
        tr = Trace(seed=1, events=[_send()])
        b1 = _bug(BugClass.LATENCY_SPIKE, "at concurrency=4")
        b2 = _bug(BugClass.TOKEN_RATE_STALL, "token rate 0.1 tok/s on request r1")
        assert mechanism_signature(tr, [b1, b2]) == mechanism_signature(tr, [b2, b1])

    def test_passed_results_without_class_are_ignored(self):
        tr = Trace(seed=1, events=[_send()])
        bug = _bug(BugClass.SCHEDULER_HANG, "queue")
        ok = OracleResult(passed=True, detail="no gpu data")
        assert mechanism_signature(tr, [bug, ok]) == mechanism_signature(tr, [bug])

    def test_signature_is_16_hex_chars(self):
        sig = mechanism_signature(Trace(seed=1, events=[_send()]),
                                  [_bug(BugClass.SCHEDULER_HANG, "q")])
        assert len(sig) == 16
        int(sig, 16)

    def test_sig_version(self):
        assert SIG_VERSION == 2


# ---------------------------------------------------------------------------
# 3b. sig v2 — the coarser in-run key for kv_corruption (A1 + B1)
# ---------------------------------------------------------------------------

def _kv(detail="(possible KV cache corruption, similarity=0.10)", **pm):
    return _bug(BugClass.KV_CORRUPTION, detail, score=10.0,
                probe_metadata=(pm or None))


class TestKvCorruptionCoarseShape:
    """A1: for an all-kv_corruption finding the shape drops has_cancel and
    any_prefix — a cancelled sibling request or a shared prefix elsewhere in
    the trace does not make it a different corruption mechanism."""

    def test_cancel_does_not_split(self):
        plain = Trace(seed=1, events=[_send()])
        cancelled = Trace(seed=1, events=[
            _send(), Event(action=ActionType.CANCEL, offset_ms=10)])
        assert (mechanism_signature(plain, [_kv()]) ==
                mechanism_signature(cancelled, [_kv()]))

    def test_prefix_does_not_split(self):
        a = Trace(seed=1, events=[_send(prefix_len=0)])
        b = Trace(seed=1, events=[_send(prefix_len=512)])
        assert (mechanism_signature(a, [_kv()]) ==
                mechanism_signature(b, [_kv()]))

    @pytest.mark.parametrize("kw", [
        {"adapter_name": "lora_a"},
        {"logprobs": 5},
        {"n_completions": 4},
    ])
    def test_kept_shape_bits_still_split(self, kw):
        plain = Trace(seed=1, events=[_send()])
        other = Trace(seed=1, events=[_send(**kw)])
        assert (mechanism_signature(plain, [_kv()]) !=
                mechanism_signature(other, [_kv()]))

    def test_cancel_still_splits_a_mixed_finding(self):
        """Only findings that are entirely kv_corruption get the coarse key."""
        bugs = [_kv(), _bug(BugClass.LATENCY_SPIKE, "at concurrency=8")]
        plain = Trace(seed=1, events=[_send()])
        cancelled = Trace(seed=1, events=[
            _send(), Event(action=ActionType.CANCEL, offset_ms=10)])
        assert (mechanism_signature(plain, bugs) !=
                mechanism_signature(cancelled, bugs))

    def test_cancel_still_splits_a_non_kv_finding(self):
        bugs = [_bug(BugClass.LATENCY_SPIKE, "at concurrency=8")]
        plain = Trace(seed=1, events=[_send()])
        cancelled = Trace(seed=1, events=[
            _send(), Event(action=ActionType.CANCEL, offset_ms=10)])
        assert (mechanism_signature(plain, bugs) !=
                mechanism_signature(cancelled, bugs))

    def test_shapes_have_different_arity_so_they_cannot_collide(self):
        tr = Trace(seed=1, events=[_send()])
        assert len(workload_shape(tr, [_kv()])) == 3
        assert len(workload_shape(tr, None)) == 5
        assert len(workload_shape(tr,
                                  [_bug(BugClass.LATENCY_SPIKE, "c=8")])) == 5

    def test_all_kv_corruption_helper(self):
        assert fs.all_kv_corruption([_kv()]) is True
        assert fs.all_kv_corruption([]) is False
        assert fs.all_kv_corruption([_kv(), _bug(BugClass.LATENCY_SPIKE)]) is False
        # a passed result carries no class and must not veto the coarse key
        assert fs.all_kv_corruption(
            [_kv(), OracleResult(passed=True, detail="ok")]) is True

    def test_live_and_from_dict_agree_on_the_coarse_key(self):
        """A1 must land identically on both paths or the archive re-signs
        differently from the runner."""
        trace = Trace(seed=3, events=[
            _send(prefix_len=128, logprobs=5),
            Event(action=ActionType.CANCEL, offset_ms=10),
        ])
        bug = _kv(logprob_gap=0.125)
        finding = {
            "trace": trace.to_dict(),
            "bugs": [{"class": "kv_corruption", "detail": bug.detail,
                      "score": 10.0, "probe_metadata": bug.probe_metadata}],
        }
        assert (mechanism_signature_from_dict(finding) ==
                mechanism_signature(trace, [bug]))


class TestKvCorruptionGapBand:
    """B1: band the determinism probe's logprob_gap, not the word-set
    similarity, whenever a numeric gap is available."""

    def test_narrow_gap_is_gtie(self):
        assert oracle_subtype(BugClass.KV_CORRUPTION, "similarity=0.10",
                              {"logprob_gap": 0.125}) == "gtie"

    def test_wide_gap_is_gwide(self):
        assert oracle_subtype(BugClass.KV_CORRUPTION, "similarity=0.10",
                              {"logprob_gap": 3.5}) == "gwide"

    def test_band_edge_is_inclusive_upward(self):
        assert oracle_subtype(BugClass.KV_CORRUPTION, "s",
                              {"logprob_gap": fs._GAP_TIE_MAX}) == "gwide"

    def test_gap_overrides_similarity(self):
        """The whole point: similarity spread one V7 over three bands."""
        lo = oracle_subtype(BugClass.KV_CORRUPTION,
                            "(possible KV cache corruption, similarity=0.10)",
                            {"similarity": 0.10, "logprob_gap": 0.25})
        hi = oracle_subtype(BugClass.KV_CORRUPTION,
                            "(possible KV cache corruption, similarity=0.98)",
                            {"similarity": 0.98, "logprob_gap": 0.25})
        assert lo == hi == "gtie"

    def test_gap_bands_split_two_gaps(self):
        tr = Trace(seed=1, events=[_send()])
        assert (mechanism_signature(tr, [_kv(logprob_gap=0.125)]) !=
                mechanism_signature(tr, [_kv(logprob_gap=4.0)]))

    @pytest.mark.parametrize("pm", [
        {},
        {"logprob_gap": None},
        {"logprob_gap": "0.25"},          # non-numeric → not a gap
        {"logprob_gap": float("nan")},
    ])
    def test_falls_back_to_the_similarity_band(self, pm):
        pm = dict(pm, similarity=0.10)
        assert oracle_subtype(BugClass.KV_CORRUPTION,
                              "similarity=0.10", pm) == "sim_lo"

    def test_fallback_bands_are_unchanged_from_v1(self):
        for sim, expected in ((0.98, "sim_hi"), (0.70, "sim_mid"),
                              (0.10, "sim_lo")):
            assert oracle_subtype(BugClass.KV_CORRUPTION,
                                  "(possible KV cache corruption, "
                                  "similarity=%.2f)" % sim) == expected

    def test_stability_and_probe_missing_ignore_the_gap(self):
        assert oracle_subtype(BugClass.KV_CORRUPTION,
                              "Validated output instability across runs",
                              {"logprob_gap": 0.125}) == "stability"
        assert oracle_subtype(BugClass.KV_CORRUPTION,
                              "trailing probe missing",
                              {"logprob_gap": 0.125}) == "probe_missing"

class TestTriageSignature:

    def _finding(self, **kw):
        tr = Trace(seed=1, events=[_send(prompt_len=2048)])
        f = {"trace": tr.to_dict(),
             "bugs": [{"class": "latency_spike",
                       "detail": "p99 TTFT 9738 ms exceeds expected 500 ms "
                                 "(19.5x) at concurrency=8",
                       "score": 19.5}]}
        f.update(kw)
        return f

    def test_server_flags_split_the_triage_key(self):
        a = self._finding(server_flags="--model X --enable-prefix-caching")
        b = self._finding(server_flags="--model X")
        assert mechanism_signature_from_dict(a) == mechanism_signature_from_dict(b)
        assert triage_signature(a) != triage_signature(b)

    def test_port_and_endpoint_do_not_split(self):
        a = self._finding(server_flags="--model-path X --port 30089 "
                                       "--kv-events-config {\"a\":1}")
        b = self._finding(server_flags="--model-path X --port 31771 "
                                       "--kv-events-config {\"a\":2}")
        assert triage_signature(a) == triage_signature(b)

    def test_model_splits_the_triage_key(self):
        a = self._finding(model="Qwen/Qwen2.5-0.5B-Instruct")
        b = self._finding(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
        assert triage_signature(a) != triage_signature(b)

    def test_coarse_shape_splits_the_triage_key(self):
        a = self._finding()
        big = Trace(seed=1, events=[_send(prompt_len=2048) for _ in range(16)])
        b = self._finding(trace=big.to_dict())
        assert triage_signature(a) != triage_signature(b)

    def test_missing_optional_fields(self):
        f = {"trace": {"events": []}, "bugs": []}
        assert len(triage_signature(f)) == 16


# ---------------------------------------------------------------------------
# 5. legacy_signature reproduces the runner closure
# ---------------------------------------------------------------------------

class TestLegacySignature:

    def test_matches_runner_closure(self):
        """Recompute runner.py:753-763 inline and compare."""
        import hashlib

        tr = Trace(seed=1, events=[
            _send(prompt_len=512, max_tokens=64, prefix_len=32,
                  adapter_name="lora_a"),
            Event(action=ActionType.CANCEL, offset_ms=17),
        ])
        bugs = [_bug(BugClass.LATENCY_SPIKE, "at concurrency=4"),
                _bug(BugClass.TOKEN_RATE_STALL, "token rate 0.1 tok/s")]

        classes = tuple(sorted(b.bug_class.value for b in bugs if b.bug_class))
        events = tuple(
            (e.action.value if hasattr(e.action, "value") else str(e.action),
             e.adapter_name, e.prompt_len, e.max_tokens,
             e.prefix_len if hasattr(e, "prefix_len") else None)
            for e in tr.events
        )
        expected = hashlib.md5(repr((classes, events)).encode()).hexdigest()[:16]

        finding = {"trace": tr.to_dict(),
                   "bugs": [{"class": b.bug_class.value, "detail": b.detail,
                             "score": b.score} for b in bugs]}
        assert legacy_signature(finding) == expected

    def test_one_prompt_len_mutation_changes_the_legacy_key(self):
        """Why the legacy key refiles: this is the whole problem."""
        bugs = [{"class": "latency_spike", "detail": "at concurrency=8",
                 "score": 1.0}]
        a = {"trace": Trace(seed=1, events=[_send(prompt_len=2048)]).to_dict(),
             "bugs": bugs}
        b = {"trace": Trace(seed=1, events=[_send(prompt_len=2049)]).to_dict(),
             "bugs": bugs}
        assert legacy_signature(a) != legacy_signature(b)
        assert mechanism_signature_from_dict(a) == mechanism_signature_from_dict(b)


def test_log2_bucket_edges():
    assert fs._log2_bucket(0) == 0
    assert fs._log2_bucket(1) == 0
    assert fs._log2_bucket(2) == 1
    assert fs._log2_bucket(8) == 3
    assert fs._log2_bucket("148.0") == 7
    assert fs._log2_bucket(None) == 0
    assert fs._log2_bucket("nope") == 0
    assert fs._log2_bucket(float("nan")) == 0


def test_normalize_server_flags_none():
    assert fs.normalize_server_flags(None) == ""
    assert fs.normalize_server_flags("") == ""


# ---------------------------------------------------------------------------
# degenerate_mask sub-types (mask-composition surface)
# ---------------------------------------------------------------------------

class TestDegenerateMaskSubtype:
    """A BugClass with no branch in ``oracle_subtype`` falls through to
    ``sub = cls``, and every predicate collapses into one ledger entry — which
    is exactly what happened the last time a class landed without its
    sub-type.  These tests are the guard for that."""

    _SENTINEL_DETAIL = (
        "empty composed logits mask on request r1: the sampled token at output "
        "position 19 carries the -inf sentinel (-9999.0) while a rival "
        "candidate is finite at 0.0000 "
        "(restrictors=['min_tokens', 'structured_output'])")
    _GUARD_DETAIL = (
        "grammar guard terminated request r1 with HTTP 500 after composing 2 "
        "mask restrictors (['min_tokens', 'structured_output']) — the post-0.27 "
        "symptom of an empty composed mask")
    _ESCAPE_DETAIL = (
        "allowed_token_ids escaped on request r1: the allowlist declares 1 "
        "id(s), every one of them is in the stop set that min_tokens=16 bans, "
        "yet 3 distinct token strings were emitted in the first 3 output "
        "positions")

    @pytest.mark.parametrize("predicate,expected", [
        ("sentinel",         "dm_sentinel"),
        ("grammar_guard",    "dm_grammar_guard"),
        ("allowlist_escape", "dm_allowlist_escape"),
    ])
    def test_predicate_drives_the_subtype(self, predicate, expected):
        assert oracle_subtype(BugClass.DEGENERATE_MASK, "",
                              {"predicate": predicate}) == expected

    @pytest.mark.parametrize("detail,expected", [
        (_SENTINEL_DETAIL, "dm_sentinel"),
        (_GUARD_DETAIL,    "dm_grammar_guard"),
        (_ESCAPE_DETAIL,   "dm_allowlist_escape"),
    ])
    def test_detail_string_is_the_fallback(self, detail, expected):
        assert oracle_subtype(BugClass.DEGENERATE_MASK, detail) == expected

    def test_unknown_predicate_degrades_rather_than_raising(self):
        assert oracle_subtype(BugClass.DEGENERATE_MASK, "",
                              {"predicate": "future"}) == "dm_na"
        assert oracle_subtype(BugClass.DEGENERATE_MASK, "") == "dm_na"

    def test_it_does_not_fall_through_to_the_class_name(self):
        for pm in ({"predicate": "sentinel"}, {"predicate": "grammar_guard"},
                   {"predicate": "allowlist_escape"}):
            assert oracle_subtype(BugClass.DEGENERATE_MASK, "", pm) \
                != "degenerate_mask"

    def test_the_three_predicates_do_not_collapse_into_one_signature(self):
        from fuzzer.core.finding_sig import mechanism_signature
        from fuzzer.trace import ActionType, Event, Trace
        from fuzzer.core.types import OracleResult
        trace = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=96)])
        sigs = {
            mechanism_signature(trace, [OracleResult(
                passed=False, bug_class=BugClass.DEGENERATE_MASK,
                detail="", probe_metadata={"predicate": p})])
            for p in ("sentinel", "grammar_guard", "allowlist_escape")
        }
        assert len(sigs) == 3

    def test_the_composition_is_not_part_of_the_subtype(self):
        """Which pair of restrictors emptied the row is workload, not
        mechanism — one defect must not refile as N."""
        a = oracle_subtype(BugClass.DEGENERATE_MASK, "", {
            "predicate": "sentinel",
            "restrictors": ["min_tokens", "structured_output"]})
        b = oracle_subtype(BugClass.DEGENERATE_MASK, "", {
            "predicate": "sentinel",
            "restrictors": ["allowed_token_ids", "structured_output"]})
        assert a == b == "dm_sentinel"
