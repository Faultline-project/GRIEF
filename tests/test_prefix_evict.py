"""Unit tests for ``fuzzer.prefix_evict`` (pure python, no vLLM, no server).

The live end-to-end evidence for the same code is
``campaign_logs/kv_probe_2026-08-29/verdicts.json`` (gates G1-G5); these tests
pin the arithmetic and the failure modes so a refactor cannot quietly change
what "restored" means.
"""

import json

import pytest

from fuzzer.prefix_evict import (
    ACCEPTANCE_FAILURES,
    ACCEPTANCE_WARNINGS,
    DRAIN_PROBE_LABEL,
    FALLBACK_OFF,
    FALLBACK_RESET_REBUILD,
    DrainTimeout,
    RestorationFailed,
    accept_eviction,
    admission_hits,
    check_hit_restoration,
    drain_count,
    drain_probe_body,
    eviction_plan,
    full_token_ids,
    restore_prefix_cache,
    summarize_evict_response,
    wait_for_drain,
    speculative_hit_shift_blocks,
)

BS = 16


def _comp(step_index, reqs):
    return {"kind": "composition", "step_index": step_index, "requests": reqs}


def _req(rid, computed, prompt, sched=1, new=False):
    return {"request_id": rid, "num_scheduled_tokens": sched,
            "num_computed_tokens": computed, "num_prompt_tokens": prompt,
            "new": new}


# --------------------------------------------------------------------------
# full_token_ids
# --------------------------------------------------------------------------

def test_full_token_ids_is_prompt_plus_generated():
    choice = {"prompt_token_ids": [1, 2, 3], "token_ids": [9, 8]}
    assert full_token_ids(choice) == [1, 2, 3, 9, 8]


def test_full_token_ids_refuses_a_recording_without_token_ids():
    # a plan built from the prompt alone would leave decode blocks reachable
    with pytest.raises(ValueError):
        full_token_ids({"prompt_token_ids": [1, 2, 3]})


def test_full_token_ids_refuses_an_empty_prompt():
    with pytest.raises(ValueError):
        full_token_ids({"prompt_token_ids": [], "token_ids": [1]})


# --------------------------------------------------------------------------
# admission_hits
# --------------------------------------------------------------------------

def test_admission_hit_is_the_first_occurrence_not_the_last():
    recs = [
        _comp(0, [_req("cmpl-kvp-a-0-deadbeef", 2112, 2176, 64, True)]),
        _comp(1, [_req("cmpl-kvp-a-0-deadbeef", 2176, 2176, 1)]),
    ]
    got = admission_hits(recs)
    assert set(got) == {"kvp-a"}
    assert got["kvp-a"]["hit_len"] == 2112
    assert got["kvp-a"]["miss_len"] == 64
    assert got["kvp-a"]["full_miss"] is False


def test_admission_hit_zero_is_a_full_miss():
    recs = [_comp(0, [_req("cmpl-kvp-n-0-deadbeef", 0, 265, 265, True)])]
    assert admission_hits(recs)["kvp-n"]["full_miss"] is True


def test_admission_hits_ignores_non_composition_records():
    recs = [{"kind": "step", "groups": []},
            _comp(0, [_req("cmpl-kvp-a-0-deadbeef", 16, 32, 16, True)])]
    assert list(admission_hits(recs)) == ["kvp-a"]


# --------------------------------------------------------------------------
# eviction_plan
# --------------------------------------------------------------------------

def _plan(hit, n_tokens, **kw):
    hits = {"b": {"base": "b", "hit_len": hit, "num_prompt_tokens": n_tokens - 64}}
    return eviction_plan({"b": list(range(n_tokens))}, hits, block_size=BS, **kw)


def test_keep_prefix_blocks_is_the_hit_length_in_blocks():
    p = _plan(2112, 2240)
    e = p["entries"][0]
    assert e["keep_prefix_blocks"] == 2112 // BS == 132
    assert e["keep_source"] == "discovery_admission"
    assert e["hit_len_block_aligned"] is True
    assert e["n_full_blocks"] == 140
    assert e["n_blocks_targeted"] == 8
    assert p["body"]["requests"][0]["keep_prefix_blocks"] == 132


def test_a_full_miss_keeps_nothing():
    e = _plan(0, 329)["entries"][0]
    assert e["keep_prefix_blocks"] == 0
    assert e["n_blocks_targeted"] == e["n_full_blocks"] == 329 // BS


def test_a_non_block_aligned_hit_is_reported_not_silently_floored():
    e = _plan(2110, 2240)["entries"][0]
    assert e["hit_len_block_aligned"] is False
    assert e["keep_prefix_blocks"] == 131          # floor, and flagged


def test_keep_override_is_recorded_alongside_the_value_it_replaced():
    e = _plan(2112, 2240, keep_override={"b": 0})["entries"][0]
    assert e["keep_prefix_blocks"] == 0
    assert e["keep_source"] == "override"
    assert e["keep_derived_from_discovery"] == 132


def test_the_body_carries_the_full_token_sequence():
    body = _plan(0, 96)["body"]
    assert body["requests"][0]["token_ids"] == list(range(96))
    assert body["requests"][0]["label"] == "b"


def test_an_eagle_pruned_block_is_kept_when_the_hit_is_non_zero():
    # EAGLE3 hunt 2026-09-14: the ancestor and the extension share 144 tokens
    # (9 blocks); vLLM prunes the last matching block under EAGLE-like
    # speculation and admits the request at 128.  Keeping 8 evicts the
    # ancestor's ninth block and the next replicate is admitted at 112.
    e = _plan(128, 778, hit_shift_blocks=1)["entries"][0]
    assert e["keep_prefix_blocks"] == 9
    assert e["keep_source"] == "discovery_admission+pruned_block"
    assert e["keep_derived_from_discovery"] == 8
    assert e["hit_shift_blocks"] == 1


def test_an_eagle_shift_does_not_invent_a_block_on_a_full_miss():
    e = _plan(0, 80, hit_shift_blocks=1)["entries"][0]
    assert e["keep_prefix_blocks"] == 0
    assert e["keep_source"] == "discovery_admission"
    assert e["hit_shift_blocks"] == 0


def test_an_override_still_wins_over_the_eagle_shift():
    e = _plan(128, 778, hit_shift_blocks=1, keep_override={"b": 0})["entries"][0]
    assert e["keep_prefix_blocks"] == 0 and e["keep_source"] == "override"


@pytest.mark.parametrize("args, want", [
    ([], 0),
    (["--enforce-eager"], 0),
    (["--speculative-config", '{"method":"eagle3","model":"AngelSlim/Qwen3-8B_eagle3","num_speculative_tokens":3}'], 1),
    (['--speculative-config={"method":"dspark","model":"deepseek-ai/dspark_qwen3_8b_block7"}'], 1),
    (["--speculative-config", '{"method":"dflash","model":"z-lab/Qwen3-8B-DFlash-b16"}'], 1),
    (["--speculative-config", '{"method":"mtp","num_speculative_tokens":1}'], 1),
    (["--speculative-config", '{"method":"ngram","num_speculative_tokens":3}'], 0),
    (["--speculative-config", '{"method":"draft_model","model":"x"}'], 0),
    (["--speculative-config", "not json"], 0),
    (["--speculative-config", "method:eagle3", "model:AngelSlim/Qwen3-8B_eagle3", "num_speculative_tokens:3"], 1),
    (["--speculative-config", "method:ngram_gpu", "num_speculative_tokens:5", "--max-num-seqs", "64"], 0),
    (["--attention-backend", "FLASH_ATTN", "--speculative-config", "method:dflash", "model:z-lab/Qwen3-8B-DFlash-b16"], 1),
    (["--speculative-config"], 0),
])
def test_the_shift_is_read_from_the_serve_args(args, want):
    assert speculative_hit_shift_blocks(args) == want


def test_plan_refuses_a_request_with_no_admission_record():
    with pytest.raises(ValueError, match="no admission record"):
        eviction_plan({"b": [1] * 32}, {}, block_size=BS)


def test_plan_refuses_a_nonpositive_block_size():
    with pytest.raises(ValueError, match="block_size"):
        eviction_plan({"b": [1] * 32}, {"b": {"hit_len": 0,
                                              "num_prompt_tokens": 32}},
                      block_size=0)


def test_plan_order_is_explicit_and_honoured():
    hits = {n: {"hit_len": 0, "num_prompt_tokens": 32} for n in ("x", "y")}
    toks = {"x": [1] * 32, "y": [2] * 32}
    p = eviction_plan(toks, hits, block_size=BS, order=["y", "x"])
    assert [e["base"] for e in p["entries"]] == ["y", "x"]


# --------------------------------------------------------------------------
# check_hit_restoration
# --------------------------------------------------------------------------

def _h(**kw):
    return {b: {"hit_len": v, "num_prompt_tokens": 2176}
            for b, v in kw.items()}


def test_restoration_is_equality_not_merely_nonzero():
    d = _h(a=2112, n=0)
    assert check_hit_restoration(d, _h(a=2112, n=0))["ok"] is True
    # a LARGER hit is still a failure: it means the trace hit its own last run
    r = check_hit_restoration(d, _h(a=2160, n=0))
    assert r["ok"] is False
    assert r["n_mismatched"] == 1
    assert [x["delta"] for x in r["requests"] if x["base"] == "a"] == [48]


def test_restoration_reports_a_zeroed_ancestral_hit_loudly():
    r = check_hit_restoration(_h(a=2112), _h(a=0))
    assert r["ok"] is False
    row = r["requests"][0]
    assert (row["expected_hit_len"], row["actual_hit_len"]) == (2112, 0)
    assert row["match"] is False


def test_restoration_flags_a_missing_request_rather_than_passing_it():
    r = check_hit_restoration(_h(a=2112, b=0), _h(a=2112))
    assert r["ok"] is False
    missing = [x for x in r["requests"] if x["base"] == "b"][0]
    assert missing["missing_in_replay"] is True


def test_restoration_flags_a_changed_prompt_length():
    d = {"a": {"hit_len": 16, "num_prompt_tokens": 100}}
    a = {"a": {"hit_len": 16, "num_prompt_tokens": 101}}
    r = check_hit_restoration(d, a)
    assert r["ok"] is False
    assert r["requests"][0]["same_prompt_length"] is False


# --------------------------------------------------------------------------
# summarize_evict_response
# --------------------------------------------------------------------------

def _resp(before, after, **entry):
    base = {"hashes_computed": 0, "kept": 0, "kept_present": 0, "considered": 0,
            "found": 0, "not_found": 0, "blocks_found": 0, "evicted": 0,
            "skipped_refcount": 0, "hash_keys_removed": 0,
            "table_keys_dropped": 0}
    base.update(entry)
    return {"cache_entries_before": before, "cache_entries_after": after,
            "entries": [base]}


def test_the_shrink_identity_uses_dropped_KEYS_not_removed_LINKS():
    # one key holding two blocks: 2 links removed, 1 key leaves the table.
    r = summarize_evict_response(_resp(
        800, 799, found=1, blocks_found=2, evicted=2, considered=1,
        hash_keys_removed=2, table_keys_dropped=1))
    assert r["shrink_equals_keys_dropped"] is True
    assert r["cache_entries_shrink"] == 1
    assert r["totals"]["hash_keys_removed"] == 2
    assert r["blocks_per_found_key"] == 2.0


def test_a_table_that_shrank_by_more_than_reported_fails_the_identity():
    r = summarize_evict_response(_resp(800, 700, table_keys_dropped=1,
                                       hash_keys_removed=1, evicted=1))
    assert r["shrink_equals_keys_dropped"] is False


def test_a_missing_preserved_entry_fails_the_positive_half():
    r = summarize_evict_response(_resp(10, 10, kept=132, kept_present=131))
    assert r["all_kept_still_present"] is False


def test_a_refcount_skip_is_surfaced():
    r = summarize_evict_response(_resp(10, 10, skipped_refcount=3))
    assert r["no_refcount_skips"] is False


def test_considered_must_account_for_found_plus_not_found():
    ok = summarize_evict_response(_resp(10, 10, considered=8, found=7,
                                        not_found=1))
    assert ok["considered_accounts_for_all"] is True
    bad = summarize_evict_response(_resp(10, 10, considered=8, found=7,
                                         not_found=0))
    assert bad["considered_accounts_for_all"] is False


def test_kept_present_is_counted_per_kv_cache_group():
    # the engine counts kept_present once per (block, group) but reports kept
    # as a block count, so the identity only reads directly at one group
    one = summarize_evict_response(_resp(10, 10, kept=4, kept_present=4))
    assert one["all_kept_still_present"] is True
    two = _resp(10, 10, kept=4, kept_present=8)
    two["num_kv_cache_groups"] = 2
    r = summarize_evict_response(two)
    assert r["kept_present_expected"] == 8
    assert r["all_kept_still_present"] is True


def test_eviction_must_not_free_blocks():
    r = _resp(10, 10)
    r["free_blocks_before"], r["free_blocks_after"] = 10794, 10794
    assert summarize_evict_response(r)["free_blocks_unchanged"] is True
    r["free_blocks_after"] = 10810
    assert summarize_evict_response(r)["free_blocks_unchanged"] is False


# --------------------------------------------------------------------------
# the drain gate (before the eviction)
# --------------------------------------------------------------------------

class _Clock:
    """Injected time, so a bounded wait is tested without waiting."""

    def __init__(self):
        self.t = 0.0
        self.slept = []

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def _probe_returning(*counts):
    """A probe callable that answers each count in turn, then repeats the last."""
    seq = list(counts)
    calls = []

    def probe():
        n = seq[min(len(calls), len(seq) - 1)]
        calls.append(n)
        return {"ok": True, "dry_run": True, "num_unfinished_requests": n,
                "entries": []}

    probe.calls = calls
    return probe


def test_the_drain_probe_is_one_token_and_a_dry_run():
    # the endpoint refuses an empty requests list and empty token_ids, so this
    # is the floor; one token is under a block, so nothing is hashed or looked up
    body = drain_probe_body()
    assert body["dry_run"] is True
    assert len(body["requests"]) == 1
    only = body["requests"][0]
    assert only["token_ids"] == [0]
    assert only["keep_prefix_blocks"] == 0
    assert only["label"] == DRAIN_PROBE_LABEL


def test_drain_count_reads_the_engines_own_counter():
    assert drain_count({"num_unfinished_requests": 3}) == 3


def test_drain_count_refuses_a_response_without_the_counter():
    # treating "absent" as "drained" is the silent pass the gate exists to remove
    with pytest.raises(ValueError, match="no num_unfinished_requests"):
        drain_count({"ok": True})


def test_drain_count_refuses_a_non_object_response():
    with pytest.raises(ValueError, match="not a JSON object"):
        drain_count("<html>404</html>")


def test_drain_gate_accepts_an_idle_engine_without_sleeping():
    clk = _Clock()
    probe = _probe_returning(0)
    r = wait_for_drain(probe, timeout_s=5.0, poll_interval_s=0.25,
                       sleep=clk.sleep, monotonic=clk.monotonic)
    assert r["drained"] is True
    assert r["n_probes"] == 1
    assert clk.slept == []


def test_drain_gate_polls_until_the_engine_goes_quiet():
    clk = _Clock()
    probe = _probe_returning(2, 1, 0)
    r = wait_for_drain(probe, timeout_s=5.0, poll_interval_s=0.25,
                       sleep=clk.sleep, monotonic=clk.monotonic)
    assert r["drained"] is True
    assert r["n_probes"] == 3
    assert clk.slept == [0.25, 0.25]
    assert [s["num_unfinished_requests"] for s in r["samples"]] == [2, 1, 0]


def test_drain_gate_times_out_loudly_naming_the_count():
    clk = _Clock()
    probe = _probe_returning(3)
    with pytest.raises(DrainTimeout) as exc:
        wait_for_drain(probe, timeout_s=1.0, poll_interval_s=0.25,
                       sleep=clk.sleep, monotonic=clk.monotonic,
                       what="targeted eviction 00_pass_01")
    assert "num_unfinished_requests=3" in str(exc.value)
    assert "targeted eviction 00_pass_01" in str(exc.value)
    assert "skipped_refcount" in str(exc.value)     # says WHAT would go wrong
    rec = exc.value.record
    assert rec["drained"] is False
    assert rec["num_unfinished_requests"] == 3
    assert rec["n_probes"] == 5                     # t=0, .25, .5, .75, 1.0
    assert rec["waited_s"] == 1.0


def test_drain_gate_never_overruns_its_budget():
    clk = _Clock()
    with pytest.raises(DrainTimeout):
        wait_for_drain(_probe_returning(1), timeout_s=0.3, poll_interval_s=0.25,
                       sleep=clk.sleep, monotonic=clk.monotonic)
    assert clk.slept == pytest.approx([0.25, 0.05])  # the last poll is clamped
    assert clk.t == pytest.approx(0.3)


def test_a_zero_budget_probes_exactly_once_and_still_requires_zero():
    clk = _Clock()
    probe = _probe_returning(1)
    with pytest.raises(DrainTimeout):
        wait_for_drain(probe, timeout_s=0.0, sleep=clk.sleep,
                       monotonic=clk.monotonic)
    assert len(probe.calls) == 1
    assert clk.slept == []
    assert wait_for_drain(_probe_returning(0), timeout_s=0.0, sleep=clk.sleep,
                          monotonic=clk.monotonic)["drained"] is True


def test_the_gate_is_opted_out_of_by_not_calling_it_not_by_a_negative_budget():
    with pytest.raises(ValueError, match="must be >= 0"):
        wait_for_drain(_probe_returning(0), timeout_s=-1.0)


def test_a_probe_that_cannot_be_read_raises_rather_than_passing():
    with pytest.raises(ValueError, match="no num_unfinished_requests"):
        wait_for_drain(lambda: {"ok": True}, timeout_s=1.0)


# --------------------------------------------------------------------------
# accept_eviction -- the one authoritative verdict
# --------------------------------------------------------------------------

def _entry(**kw):
    """One response entry, defaulting to the patch README's real numbers."""
    e = {"index": 0, "label": "kvp-p0_a_ext", "n_token_ids": 2240,
         "hashes_computed": 140, "keep_prefix_blocks": 132, "kept": 132,
         "kept_present": 132, "considered": 8, "found": 7, "not_found": 1,
         "blocks_found": 16, "evicted": 16, "skipped_refcount": 0,
         "hash_keys_removed": 16, "table_keys_dropped": 7}
    e.update(kw)
    return e


def _good(**kw):
    """A response the endpoint really produced (evict_main.json), one entry."""
    r = {"ok": True, "dry_run": False, "block_size": 16, "hash_block_size": 16,
         "num_kv_cache_groups": 1, "num_unfinished_requests": 0,
         "cache_entries_before": 800, "cache_entries_after": 793,
         "free_blocks_before": 10794, "free_blocks_after": 10794,
         "entries": [_entry()]}
    r.update(kw)
    return r


def _matching_plan(**kw):
    e = {"base": "kvp-p0_a_ext", "n_tokens": 2240, "num_prompt_tokens": 2176,
         "ancestral_hit_len": 2112, "hit_len_block_aligned": True,
         "keep_prefix_blocks": 132, "keep_source": "discovery_admission",
         "keep_derived_from_discovery": 132, "n_full_blocks": 140,
         "n_blocks_targeted": 8}
    e.update(kw)
    return {"block_size": 16, "body": {"requests": []}, "entries": [e]}


def test_a_clean_eviction_is_accepted_with_no_reason():
    r = accept_eviction(_good(), plan=_matching_plan())
    assert r["ok"] is True
    assert r["reason"] is None and r["reasons"] == []
    assert r["restores_state"] is True
    assert r["warning_codes"] == []
    assert r["bookkeeping"]["totals"]["evicted"] == 16


def test_every_reason_it_can_return_is_declared():
    # a caller switching on the reason must be able to enumerate the classes
    bad = accept_eviction({"ok": False}, http_status=500, dry_run=False)
    assert set(bad["reasons"]) <= set(ACCEPTANCE_FAILURES)
    assert bad["reason"] == "HTTP_STATUS"


def test_a_refcount_skip_is_a_restoration_FAILURE_not_a_note():
    # the skipped blocks are exactly the ones the plan targeted: the next pass
    # will hit them, so the environment is NOT restored
    r = accept_eviction(_good(entries=[_entry(skipped_refcount=3, evicted=13)]))
    assert r["ok"] is False
    assert "REFCOUNT_SKIPS" in r["reasons"]
    f = [x for x in r["failures"] if x["code"] == "REFCOUNT_SKIPS"][0]
    assert f["detail"]["total"] == 3
    assert f["detail"]["entries"][0]["label"] == "kvp-p0_a_ext"


def test_a_missing_preserved_entry_is_a_restoration_FAILURE():
    r = accept_eviction(_good(entries=[_entry(kept_present=131)]))
    assert r["ok"] is False
    assert r["reasons"] == ["KEPT_NOT_PRESENT"]
    d = r["failures"][0]["detail"]
    assert (d["kept"], d["kept_present"], d["expected"]) == (132, 131, 132)


def test_kept_present_is_scaled_by_the_number_of_cache_groups():
    two = _good(num_kv_cache_groups=2,
                entries=[_entry(kept_present=264, considered=16, found=14,
                                not_found=2)])
    assert accept_eviction(two)["ok"] is True
    one_short = _good(num_kv_cache_groups=2,
                      entries=[_entry(kept_present=263, considered=16, found=14,
                                      not_found=2)])
    assert accept_eviction(one_short)["reasons"] == ["KEPT_NOT_PRESENT"]


def test_a_violated_shrink_identity_rejects():
    # the table lost keys the endpoint did not report: the bookkeeping does not
    # describe the mutation, so no claim can be made from it
    r = accept_eviction(_good(cache_entries_after=700))
    assert r["ok"] is False
    assert r["reasons"] == ["SHRINK_IDENTITY_VIOLATED"]
    assert r["failures"][0]["detail"] == {"shrink": 100, "table_keys_dropped": 7}


def test_a_violated_considered_identity_rejects():
    r = accept_eviction(_good(entries=[_entry(considered=8, found=7,
                                              not_found=0)]))
    assert r["reasons"] == ["CONSIDERED_IDENTITY_VIOLATED"]


def test_a_free_block_that_moved_rejects():
    # eviction unlinks a block, it never frees one (patch README S6)
    r = accept_eviction(_good(free_blocks_after=10810))
    assert r["reasons"] == ["FREE_BLOCKS_CHANGED"]


def test_an_undrained_engine_rejects_after_the_fact_too():
    # the post-hoc half of the drain gate: a request that arrived between the
    # last probe and the call
    r = accept_eviction(_good(num_unfinished_requests=2))
    assert r["reasons"] == ["ENGINE_NOT_DRAINED"]
    assert r["failures"][0]["detail"]["num_unfinished_requests"] == 2
    assert accept_eviction(_good(num_unfinished_requests=2),
                           require_drained=False)["ok"] is True


def test_an_http_error_rejects_before_any_arithmetic():
    r = accept_eviction({"detail": "dev mode off"}, http_status=404)
    assert r["ok"] is False
    assert r["reason"] == "HTTP_STATUS"
    assert "RESPONSE_NOT_OK" in r["reasons"] and "NO_ENTRIES" in r["reasons"]


def test_a_response_with_no_entries_is_rejected_not_vacuously_accepted():
    r = accept_eviction({"ok": True, "entries": []})
    assert r["ok"] is False
    assert "NO_ENTRIES" in r["reasons"]
    assert r["bookkeeping"] is None


def test_a_dry_run_is_acceptable_but_restores_nothing():
    dry = _good(dry_run=True, cache_entries_after=800,
                entries=[_entry(hash_keys_removed=0, table_keys_dropped=0)])
    r = accept_eviction(dry, dry_run=True)
    assert r["ok"] is True
    assert r["restores_state"] is False           # cannot be mistaken for one
    assert "DRY_RUN" in r["warning_codes"]


def test_a_dry_run_that_mutated_the_table_rejects():
    r = accept_eviction(_good(dry_run=True), dry_run=True)
    assert r["ok"] is False
    assert "DRY_RUN_MUTATED" in r["reasons"]


def test_a_mode_the_engine_did_not_run_rejects():
    r = accept_eviction(_good(dry_run=True), dry_run=False)
    assert "DRY_RUN_FLAG_MISMATCH" in r["reasons"]


def test_a_block_size_disagreement_rejects():
    # every keep_prefix_blocks in the plan would mean a different token count
    r = accept_eviction(_good(block_size=32), plan=_matching_plan())
    assert "BLOCK_SIZE_DISAGREES" in r["reasons"]


def test_a_response_that_does_not_echo_the_plan_rejects():
    r = accept_eviction(_good(), plan=_matching_plan(keep_prefix_blocks=64))
    assert r["reasons"] == ["PLAN_ECHO_MISMATCH"]
    diffs = r["failures"][0]["detail"]["diffs"]
    assert diffs[0]["field"] == "keep_prefix_blocks"
    assert (diffs[0]["plan"], diffs[0]["engine"]) == (64, 132)


def test_a_relabelled_entry_rejects():
    r = accept_eviction(_good(), plan=_matching_plan(base="somebody-else"))
    assert r["reasons"] == ["PLAN_ECHO_MISMATCH"]
    assert r["failures"][0]["detail"]["diffs"][0]["field"] == "label"


def test_an_entry_that_matched_nothing_is_a_note_not_a_rejection():
    # legitimate under LRU pressure, and the signature of the LoRA/MM scope
    # limit: absent blocks cannot be hit, so the state is not wrong
    r = accept_eviction(_good(cache_entries_after=800,
                              entries=[_entry(found=0, not_found=8,
                                              blocks_found=0, evicted=0,
                                              hash_keys_removed=0,
                                              table_keys_dropped=0)]))
    assert r["ok"] is True
    assert "CHAIN_NOT_FOUND" in r["warning_codes"]
    assert set(r["warning_codes"]) <= set(ACCEPTANCE_WARNINGS)


def test_a_floored_keep_count_is_carried_through_as_a_note():
    r = accept_eviction(_good(), plan=_matching_plan(hit_len_block_aligned=False,
                                                     ancestral_hit_len=2110))
    assert r["ok"] is True
    assert "HIT_NOT_BLOCK_ALIGNED" in r["warning_codes"]


def test_several_failures_are_all_reported_not_just_the_first():
    r = accept_eviction(_good(num_unfinished_requests=1, cache_entries_after=700,
                              entries=[_entry(skipped_refcount=2)]))
    assert set(r["reasons"]) == {"ENGINE_NOT_DRAINED", "REFCOUNT_SKIPS",
                                 "SHRINK_IDENTITY_VIOLATED"}
    assert r["reason"] == "ENGINE_NOT_DRAINED"       # first, for the one-liner
    assert r["checks"]["no_refcount_skips"] is False


# --------------------------------------------------------------------------
# restore_prefix_cache -- the fallback state machine
# --------------------------------------------------------------------------

def _attempt(ok, reason=None, tag="00_pass_01"):
    """What the driver's run_eviction returns, reduced to what the machine reads."""
    return {"tag": tag, "ok": ok,
            "acceptance": {"ok": ok, "reason": reason,
                           "reasons": [] if ok else [reason],
                           "failures": ([] if ok else
                                        [{"code": reason, "message": f"{reason} happened",
                                          "detail": {}}]),
                           "warning_codes": []}}


class _Machine:
    """Records the ORDER of the calls the state machine makes."""

    def __init__(self, *verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    def evict(self, attempt):
        self.calls.append(f"evict{attempt}")
        return self.verdicts[attempt]

    def reset(self):
        self.calls.append("reset")
        return {"endpoint": "/reset_prefix_cache", "http_status": 200}

    def rebuild(self):
        self.calls.append("rebuild")
        return {"round": "rb1", "n_passes": 1, "summary": ["rb1_ancestor"]}


def test_a_surgical_restoration_never_touches_the_fallback():
    m = _Machine(_attempt(True))
    r = restore_prefix_cache(m.evict, fallback=FALLBACK_RESET_REBUILD,
                             reset=m.reset, rebuild=m.rebuild)
    assert m.calls == ["evict0"]
    assert (r["ok"], r["restored_surgically"], r["used_fallback"]) == (
        True, True, False)
    assert r["fallback"] is None and r["n_attempts"] == 1


def test_with_the_fallback_off_a_rejection_is_recorded_and_the_run_continues():
    # the pre-fallback behaviour, unchanged: the gated pass's DESYNC echo is
    # still the arbiter, so this must not raise and must not reset anything
    m = _Machine(_attempt(False, "REFCOUNT_SKIPS"))
    r = restore_prefix_cache(m.evict, fallback=FALLBACK_OFF)
    assert m.calls == ["evict0"]
    assert r["ok"] is False
    assert r["restored_surgically"] is False and r["used_fallback"] is False
    assert r["first_failure"] == r["final_failure"] == "REFCOUNT_SKIPS"


def test_the_fallback_runs_reset_then_rebuild_then_exactly_one_retry():
    m = _Machine(_attempt(False, "KEPT_NOT_PRESENT"), _attempt(True))
    r = restore_prefix_cache(m.evict, fallback=FALLBACK_RESET_REBUILD,
                             reset=m.reset, rebuild=m.rebuild)
    assert m.calls == ["evict0", "reset", "rebuild", "evict1"]
    assert r["ok"] is True
    assert r["used_fallback"] is True
    assert r["restored_surgically"] is False      # rebuilt, not restored
    assert r["n_attempts"] == 2


def test_the_fallback_does_not_mask_the_failure_it_recovered_from():
    m = _Machine(_attempt(False, "KEPT_NOT_PRESENT"), _attempt(True))
    r = restore_prefix_cache(m.evict, fallback=FALLBACK_RESET_REBUILD,
                             reset=m.reset, rebuild=m.rebuild)
    fb = r["fallback"]
    assert fb["triggered"] is True
    assert fb["trigger_reason"] == "KEPT_NOT_PRESENT"
    assert fb["trigger_failures"][0]["message"] == "KEPT_NOT_PRESENT happened"
    assert fb["retry_ok"] is True and fb["retry_reason"] is None
    assert fb["reset"]["http_status"] == 200
    assert fb["rebuild"]["summary"] == ["rb1_ancestor"]
    assert r["first_failure"] == "KEPT_NOT_PRESENT"   # kept even though ok
    assert len(r["attempts"]) == 2


def test_a_second_failure_aborts_loudly_and_carries_both_reasons():
    m = _Machine(_attempt(False, "KEPT_NOT_PRESENT"),
                 _attempt(False, "REFCOUNT_SKIPS"))
    with pytest.raises(RestorationFailed) as exc:
        restore_prefix_cache(m.evict, fallback=FALLBACK_RESET_REBUILD,
                             reset=m.reset, rebuild=m.rebuild)
    assert m.calls == ["evict0", "reset", "rebuild", "evict1"]   # never twice
    msg = str(exc.value)
    assert "KEPT_NOT_PRESENT" in msg and "REFCOUNT_SKIPS" in msg
    rec = exc.value.record
    assert rec["ok"] is False and rec["used_fallback"] is True
    assert rec["first_failure"] == "KEPT_NOT_PRESENT"
    assert rec["final_failure"] == "REFCOUNT_SKIPS"
    assert rec["fallback"]["retry_ok"] is False


def test_a_drain_timeout_is_not_something_a_reset_can_fix():
    # it propagates out of the machine untouched: resetting the cache would not
    # make a busy engine idle, and evicting anyway is the thing being prevented
    def evict(attempt):
        raise DrainTimeout("engine not drained: num_unfinished_requests=4")

    m = _Machine()
    with pytest.raises(DrainTimeout):
        restore_prefix_cache(evict, fallback=FALLBACK_RESET_REBUILD,
                             reset=m.reset, rebuild=m.rebuild)
    assert m.calls == []


def test_the_fallback_refuses_to_reset_without_a_rebuild():
    # a reset with nothing replayed afterwards leaves the ancestral blocks the
    # plan preserves permanently absent
    with pytest.raises(ValueError, match="reset and a rebuild"):
        restore_prefix_cache(lambda a: _attempt(True),
                             fallback=FALLBACK_RESET_REBUILD,
                             reset=lambda: {})


def test_an_unknown_fallback_mode_is_refused():
    with pytest.raises(ValueError, match="unknown fallback mode"):
        restore_prefix_cache(lambda a: _attempt(True), fallback="reset")


def test_the_machine_will_not_guess_a_verdict_it_was_not_given():
    with pytest.raises(ValueError, match="no 'acceptance' key"):
        restore_prefix_cache(lambda a: {"ok": True})


def test_the_machine_narrates_every_transition():
    log = []
    m = _Machine(_attempt(False, "REFCOUNT_SKIPS"), _attempt(True))
    restore_prefix_cache(m.evict, fallback=FALLBACK_RESET_REBUILD,
                         reset=m.reset, rebuild=m.rebuild, log=log.append)
    text = "\n".join(log)
    assert "REJECTED" in text and "REFCOUNT_SKIPS" in text
    assert "reset_prefix_cache" in text
    assert "REBUILT, not restored surgically" in text


# --------------------------------------------------------------------------
# what the run manifest has to be able to say afterwards
# --------------------------------------------------------------------------

def _manifest_row(rest):
    """The driver's result.json row, built from the machine's record."""
    return {"ok": rest["ok"], "used_fallback": rest["used_fallback"],
            "restored_surgically": rest["restored_surgically"],
            "first_failure": rest["first_failure"],
            "final_failure": rest["final_failure"],
            "n_attempts": rest["n_attempts"], "fallback": rest["fallback"]}


def test_a_rebuilt_run_is_distinguishable_from_a_surgical_one():
    surgical = _manifest_row(restore_prefix_cache(
        _Machine(_attempt(True)).evict, fallback=FALLBACK_RESET_REBUILD,
        reset=lambda: {}, rebuild=lambda: {}))
    m = _Machine(_attempt(False, "REFCOUNT_SKIPS"), _attempt(True))
    rebuilt = _manifest_row(restore_prefix_cache(
        m.evict, fallback=FALLBACK_RESET_REBUILD, reset=m.reset,
        rebuild=m.rebuild))
    # both PASSED, and they must not read alike
    assert surgical["ok"] is rebuilt["ok"] is True
    assert surgical["restored_surgically"] != rebuilt["restored_surgically"]
    assert surgical["used_fallback"] != rebuilt["used_fallback"]
    assert surgical["first_failure"] is None
    assert rebuilt["first_failure"] == "REFCOUNT_SKIPS"


def test_the_manifest_row_survives_json():
    m = _Machine(_attempt(False, "KEPT_NOT_PRESENT"), _attempt(True))
    row = _manifest_row(restore_prefix_cache(
        m.evict, fallback=FALLBACK_RESET_REBUILD, reset=m.reset,
        rebuild=m.rebuild))
    assert json.loads(json.dumps(row))["fallback"]["trigger_reason"] == (
        "KEPT_NOT_PRESENT")
