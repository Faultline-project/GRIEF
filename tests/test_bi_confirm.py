"""
Unit tests for scripts/bi_confirm.py — the R2 BI confirmation runner.

No live server, no GPU: guards read fixture log files, replays are synthetic
TraceResult objects, and every verdict path is exercised through the pure
decision functions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fuzzer.executor import TraceResult, RequestResult
from fuzzer.trace import Trace, Event, ActionType

# The script lives in scripts/, not in a package, so import via sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from bi_confirm import (  # noqa: E402
    BI_ENGAGEMENT_SIGNALS,
    DEFAULT_OUTPUT_DIR,
    EVAPORATION_CAVEAT,
    REFERENCE_MISMATCH_CAVEAT,
    UNKNOWN_ARCH,
    VERDICT_CORROBORATES,
    VERDICT_DOES_NOT_CORROBORATE,
    VERDICT_UNINFORMATIVE,
    BackendRuling,
    EngagementProof,
    ReferenceOutput,
    build_sidecar,
    compare_runs,
    decide,
    load_reference_outputs,
    normalize_for_bi,
    normalize_gpu_arch,
    prove_engagement,
    resolve_backend,
    resolve_gpu_arch,
    resolve_output_dir,
    rule_backend,
    write_sidecar,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

CASCADE_LINE = (
    "WARNING 08-19 11:02:31 [vllm.py:1163] Disabling cascade attention when "
    "VLLM_BATCH_INVARIANT is enabled."
)
BACKEND_LINE = "INFO 08-19 11:02:30 [cuda.py:274] Using FLASH_ATTN backend."


def _log(tmp_path: Path, *lines: str, name: str = "vllm.log") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _request(
    request_id: str,
    output_text: str | None,
    *,
    temperature: float = 0.0,
    cancelled: bool = False,
    error: str | None = None,
    logprobs: list | None = None,
) -> RequestResult:
    now = time.time()
    return RequestResult(
        request_id=request_id,
        sent_at=now - 1.0,
        completed_at=now,
        cancelled_at=now if cancelled else None,
        error=error,
        output_text=output_text,
        temperature=temperature,
        output_logprobs=logprobs,
    )


def _result(*requests: RequestResult) -> TraceResult:
    return TraceResult(
        trace_seed=1,
        started_at=time.time() - 1.0,
        finished_at=time.time(),
        requests=list(requests),
    )


def _proven() -> EngagementProof:
    return EngagementProof(
        proven=True,
        reason="matched cascade_attention_disabled in vllm.log",
        signal="cascade_attention_disabled",
        evidence_line=CASCADE_LINE,
        source=BI_ENGAGEMENT_SIGNALS[0].source,
    )


def _allowed() -> BackendRuling:
    return rule_backend("FLASH_ATTN", arch=normalize_gpu_arch("H100"), fa_version=2)


# ---------------------------------------------------------------------------
# Guard 1 — engagement proof
# ---------------------------------------------------------------------------

def test_engagement_proven_by_cascade_line(tmp_path):
    proof = prove_engagement(str(_log(tmp_path, BACKEND_LINE, CASCADE_LINE)))
    assert proof.proven is True
    assert proof.signal == "cascade_attention_disabled"
    assert "Disabling cascade attention" in proof.evidence_line
    assert "config/vllm.py" in proof.source


def test_engagement_proven_by_non_decode_invariant_warning(tmp_path):
    line = (
        "WARNING [batch_invariant.py:1023] You are using a non-decode-invariant "
        "form of batch invariance. This will not be invariant between prefill "
        "and decode."
    )
    proof = prove_engagement(str(_log(tmp_path, line)))
    assert proof.proven is True
    assert proof.signal == "non_decode_invariant_warning"


def test_engagement_not_proven_by_launcher_env_echo(tmp_path):
    """A launcher echo proves the export, not that the engine initialized BI."""
    path = _log(
        tmp_path,
        "+ export VLLM_BATCH_INVARIANT=1",
        BACKEND_LINE,
        "INFO Starting vLLM API server on http://0.0.0.0:8000",
    )
    proof = prove_engagement(str(path))
    assert proof.proven is False
    assert proof.weak_hint is not None
    assert "launcher_env_echo" in proof.weak_hint


def test_engagement_miss_on_ordinary_log(tmp_path):
    proof = prove_engagement(str(_log(tmp_path, BACKEND_LINE)))
    assert proof.proven is False
    assert proof.weak_hint is None
    assert "no batch-invariant engagement line" in proof.reason


def test_engagement_requires_a_log_file():
    proof = prove_engagement(None)
    assert proof.proven is False
    assert "never assumed" in proof.reason


def test_engagement_missing_log_file_is_not_proof(tmp_path):
    proof = prove_engagement(str(tmp_path / "absent.log"))
    assert proof.proven is False
    assert "not found" in proof.reason


# ---------------------------------------------------------------------------
# Guard 2 — backend allowlist
# ---------------------------------------------------------------------------

def test_flash_attn_fa2_is_the_allowlisted_backend():
    ruling = rule_backend("FLASH_ATTN", arch=normalize_gpu_arch("sm90"), fa_version=2)
    assert ruling.allowed is True
    assert ruling.status == "ALLOWED"
    assert ruling.fa_version == 2


def test_flash_attn_on_hopper_defaults_to_fa3_and_is_excluded():
    """vLLM picks FA3 on sm90, so an unpinned FLASH_ATTN on H100 is FA3."""
    ruling = rule_backend("FLASH_ATTN", arch=normalize_gpu_arch("NVIDIA H100 80GB"))
    assert ruling.allowed is False
    assert ruling.status == "EXCLUDED"
    assert ruling.fa_version == 3
    assert "june_27" in ruling.reason


def test_fa3_backend_name_on_hopper_is_excluded():
    ruling = rule_backend("FLASH_ATTN_VLLM_V1", arch=normalize_gpu_arch("H100"),
                          fa_version=3)
    assert ruling.allowed is False
    assert ruling.status == "EXCLUDED"


def test_flashinfer_is_excluded_outright():
    ruling = rule_backend("FLASHINFER", arch=normalize_gpu_arch("H100"))
    assert ruling.allowed is False
    assert ruling.status == "EXCLUDED"
    assert "FlashInfer" in ruling.reason or "FLASHINFER" in ruling.reason


def test_flash_attn_on_ampere_infers_fa2_and_is_allowed():
    ruling = rule_backend("FLASH_ATTN", arch=normalize_gpu_arch("A100-SXM4-80GB"))
    assert ruling.allowed is True
    assert ruling.fa_version == 2
    assert "inferred" in ruling.fa_version_source


def test_flash_attn_with_unresolved_arch_is_not_assumed_sound():
    ruling = rule_backend("FLASH_ATTN", arch=UNKNOWN_ARCH)
    assert ruling.allowed is False
    assert ruling.status == "UNKNOWN"
    assert ruling.fa_version is None


def test_triton_attn_is_known_but_not_allowlisted():
    ruling = rule_backend("TRITON_ATTN", arch=normalize_gpu_arch("H100"))
    assert ruling.allowed is False
    assert ruling.status == "NOT_ALLOWLISTED"


def test_unknown_backend_is_uninformative_not_assumed():
    ruling = rule_backend(None)
    assert ruling.allowed is False
    assert ruling.status == "UNKNOWN"
    assert ruling.name is None


def test_fa4_is_not_allowlisted_without_an_observed_downgrade():
    ruling = rule_backend("FLASH_ATTN", arch=normalize_gpu_arch("B200"))
    assert ruling.allowed is False
    assert ruling.fa_version == 4


# ---------------------------------------------------------------------------
# Backend / arch resolution
# ---------------------------------------------------------------------------

def test_resolve_backend_from_server_flags():
    finding = {
        "server_flags": "--model foo --attention-backend FLASH_ATTN "
                        "--enable-prefix-caching",
        "server_env": "VLLM_BATCH_INVARIANT=1 VLLM_FLASH_ATTN_VERSION=2",
    }
    backend, fa_version, source = resolve_backend(finding)
    assert backend == "FLASH_ATTN"
    assert fa_version == 2
    assert source == "finding.server_flags"


def test_resolve_backend_from_capture_manifest():
    finding = {"capture_manifest": {"engine": {"attention_backend": "FLASHINFER"}}}
    backend, _, source = resolve_backend(finding)
    assert backend == "FLASHINFER"
    assert "capture_manifest" in source


def test_resolve_backend_falls_back_to_server_log():
    backend, _, source = resolve_backend({}, log_text=BACKEND_LINE)
    assert backend == "FLASH_ATTN"
    assert source == "server log"


def test_resolve_backend_override_wins_but_keeps_flag_fa_version():
    finding = {"server_flags": "--attention-backend FLASHINFER "
                               "VLLM_FLASH_ATTN_VERSION=2"}
    backend, fa_version, source = resolve_backend(finding, override="FLASH_ATTN")
    assert (backend, fa_version, source) == ("FLASH_ATTN", 2, "override")


def test_resolve_backend_unresolved():
    backend, fa_version, source = resolve_backend({})
    assert backend is None and fa_version is None and source == "unresolved"


def test_resolve_gpu_arch_from_finding():
    arch = resolve_gpu_arch({"gpu_type": "h100"})
    assert arch.is_hopper is True
    assert arch.label == "sm90"
    assert arch.source == "finding.gpu_type"


def test_normalize_gpu_arch_unknown_stays_unknown():
    arch = normalize_gpu_arch("some-future-gpu")
    assert arch.is_hopper is None
    assert arch.label is None


# ---------------------------------------------------------------------------
# Trace normalization
# ---------------------------------------------------------------------------

def test_normalize_forces_greedy_and_logprobs():
    trace = Trace(
        seed=7,
        events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=32, max_tokens=16, temperature=0.9),
            Event(action=ActionType.WAIT, offset_ms=10, request_id="w",
                  duration_ms=5),
        ],
    )
    normalized = normalize_for_bi(trace, logprobs=1)
    send = normalized.trace.events[0]
    assert send.temperature == 0.0
    assert send.logprobs == 1
    assert normalized.comparable_request_ids == ("r1",)
    # original trace untouched
    assert trace.events[0].temperature == 0.9


def test_normalize_excludes_multi_completion_and_cancelled_requests():
    trace = Trace(
        seed=7,
        events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="multi",
                  prompt_len=32, max_tokens=16, temperature=0.8, n_completions=4),
            Event(action=ActionType.SEND, offset_ms=1, request_id="killed",
                  prompt_len=32, max_tokens=16),
            Event(action=ActionType.CANCEL, offset_ms=5, request_id="killed"),
        ],
    )
    normalized = normalize_for_bi(trace)
    assert normalized.comparable_request_ids == ()
    assert "n_completions" in normalized.excluded["multi"]
    assert "cancelled" in normalized.excluded["killed"]
    # the n>1 event keeps temperature > 0 so the Event invariant still holds
    normalized.trace.events[0].validate()


# ---------------------------------------------------------------------------
# Comparison + verdict
# ---------------------------------------------------------------------------

def test_corroborates_on_internal_divergence():
    runs = [
        _result(_request("r1", "alpha"), _request("r2", "same")),
        _result(_request("r1", "beta"), _request("r2", "same")),
        _result(_request("r1", "alpha"), _request("r2", "same")),
    ]
    comparisons = compare_runs(runs, ("r1", "r2"))
    assert comparisons["r1"].unique_outputs == 2
    assert comparisons["r2"].unique_outputs == 1

    decision = decide(_proven(), _allowed(), comparisons, completed_runs=3)
    assert decision.verdict == VERDICT_CORROBORATES
    assert decision.diverged_requests == ["r1"]
    assert decision.caveat is None


def test_does_not_corroborate_carries_the_standing_caveat():
    runs = [_result(_request("r1", "stable")) for _ in range(10)]
    comparisons = compare_runs(runs, ("r1",))
    decision = decide(_proven(), _allowed(), comparisons, completed_runs=10)
    assert decision.verdict == VERDICT_DOES_NOT_CORROBORATE
    assert decision.caveat == EVAPORATION_CAVEAT
    assert "not benignness" in decision.caveat
    assert "never auto-unclassifies" in decision.caveat


def test_reference_mismatch_alone_never_corroborates(tmp_path):
    """The reference predates the BI intervention set, so a mismatch is expected.

    Stable outputs across the replays are evaporation, however far they sit
    from what the finding recorded.
    """
    runs = [_result(_request("r1", "gamma-output")) for _ in range(4)]
    references = {
        "r1": ReferenceOutput(text="alpha-out", truncated=True, source="sample_A")
    }
    comparisons = compare_runs(runs, ("r1",), references)
    assert comparisons["r1"].unique_outputs == 1
    assert comparisons["r1"].reference_mismatch is True

    decision = decide(_proven(), _allowed(), comparisons, completed_runs=4)
    assert decision.verdict == VERDICT_DOES_NOT_CORROBORATE
    assert decision.caveat == EVAPORATION_CAVEAT
    assert decision.diverged_requests == []
    assert "observation only" in decision.reason

    # the mismatch is still recorded, with the caveat that disarms it
    payload = build_sidecar(
        finding_path=tmp_path / "finding_00010_1.json",
        base_url="http://localhost:8000",
        requested_runs=4,
        completed_runs=4,
        engagement=_proven(),
        ruling=_allowed(),
        decision=decision,
        comparisons=comparisons,
    )
    assert payload["per_request"]["r1"]["reference_mismatch"] is True
    assert payload["per_request"]["r1"]["reference_source"] == "sample_A"
    assert payload["reference_caveat"] == REFERENCE_MISMATCH_CAVEAT
    assert "never drives CORROBORATES" in payload["reference_caveat"]


def test_reference_caveat_is_absent_when_nothing_was_compared(tmp_path):
    runs = [_result(_request("r1", "stable")) for _ in range(2)]
    comparisons = compare_runs(runs, ("r1",))
    decision = decide(_proven(), _allowed(), comparisons, completed_runs=2)
    payload = build_sidecar(
        finding_path=tmp_path / "finding_00011_1.json",
        base_url="http://localhost:8000",
        requested_runs=2,
        completed_runs=2,
        engagement=_proven(),
        ruling=_allowed(),
        decision=decision,
        comparisons=comparisons,
    )
    assert payload["per_request"]["r1"]["reference_mismatch"] is None
    assert "reference_caveat" not in payload


def test_reference_prefix_match_is_not_a_mismatch():
    runs = [_result(_request("r1", "alpha-output-continues"))]
    references = {
        "r1": ReferenceOutput(text="alpha-out", truncated=True, source="sample_A")
    }
    comparisons = compare_runs(runs, ("r1",), references)
    assert comparisons["r1"].reference_mismatch is False

    decision = decide(_proven(), _allowed(), comparisons, completed_runs=1)
    # one completed run cannot support an internal comparison
    assert decision.verdict == VERDICT_UNINFORMATIVE


def test_single_run_request_is_never_evidenced_even_with_a_reference():
    """A request seen once cannot be compared with itself, reference or not."""
    runs = [
        _result(_request("r1", "alpha")),
        _result(_request("r2", "beta")),
    ]
    references = {
        "r1": ReferenceOutput(text="recorded", truncated=False, source="finding.x"),
        "r2": ReferenceOutput(text="recorded", truncated=False, source="finding.x"),
    }
    comparisons = compare_runs(runs, ("r1", "r2"), references)
    assert comparisons["r1"].runs_observed == 1
    assert comparisons["r1"].reference_mismatch is True
    assert comparisons["r1"].evidenced is False

    decision = decide(_proven(), _allowed(), comparisons, completed_runs=2)
    assert decision.verdict == VERDICT_UNINFORMATIVE
    assert "no comparable deterministic request" in decision.reason


def test_non_deterministic_and_failed_requests_are_not_compared():
    runs = [
        _result(
            _request("hot", "a", temperature=0.7),
            _request("killed", "a", cancelled=True),
            _request("broken", None, error="timeout"),
        ),
        _result(
            _request("hot", "b", temperature=0.7),
            _request("killed", "c", cancelled=True),
            _request("broken", None, error="timeout"),
        ),
    ]
    comparisons = compare_runs(runs)
    assert comparisons == {}

    decision = decide(_proven(), _allowed(), comparisons, completed_runs=2)
    assert decision.verdict == VERDICT_UNINFORMATIVE
    assert "no comparable deterministic request" in decision.reason


def test_logprob_uniqueness_is_recorded_as_auxiliary_evidence():
    runs = [
        _result(_request("r1", "same", logprobs=[-0.1, -0.2])),
        _result(_request("r1", "same", logprobs=[-0.1, -0.9])),
    ]
    comparisons = compare_runs(runs, ("r1",))
    assert comparisons["r1"].unique_outputs == 1
    assert comparisons["r1"].unique_logprob_signatures == 2
    # auxiliary only — the text was stable, so the verdict is evaporation
    decision = decide(_proven(), _allowed(), comparisons, completed_runs=2)
    assert decision.verdict == VERDICT_DOES_NOT_CORROBORATE


def test_insufficient_completed_runs_is_uninformative():
    runs = [_result(_request("r1", "only"))]
    comparisons = compare_runs(runs, ("r1",))
    decision = decide(_proven(), _allowed(), comparisons, completed_runs=1)
    assert decision.verdict == VERDICT_UNINFORMATIVE
    assert "replay" in decision.reason


# ---------------------------------------------------------------------------
# UNINFORMATIVE precedence
# ---------------------------------------------------------------------------

def _divergent_comparisons():
    runs = [_result(_request("r1", "alpha")), _result(_request("r1", "beta"))]
    return compare_runs(runs, ("r1",))


def test_unproven_engagement_beats_a_divergent_replay():
    engagement = EngagementProof(proven=False, reason="no --vllm-log-file given")
    decision = decide(engagement, _allowed(), _divergent_comparisons(), completed_runs=10)
    assert decision.verdict == VERDICT_UNINFORMATIVE
    assert "engagement guard failed" in decision.reason
    assert decision.diverged_requests == []


def test_excluded_backend_beats_a_divergent_replay():
    ruling = rule_backend("FLASH_ATTN", arch=normalize_gpu_arch("H100"))  # FA3/sm90
    decision = decide(_proven(), ruling, _divergent_comparisons(), completed_runs=10)
    assert decision.verdict == VERDICT_UNINFORMATIVE
    assert "backend guard failed (EXCLUDED)" in decision.reason


def test_engagement_guard_is_evaluated_before_the_backend_guard():
    engagement = EngagementProof(proven=False, reason="log missing")
    ruling = rule_backend("FLASHINFER", arch=normalize_gpu_arch("H100"))
    decision = decide(engagement, ruling)
    assert decision.verdict == VERDICT_UNINFORMATIVE
    assert "engagement guard failed" in decision.reason


# ---------------------------------------------------------------------------
# Reference loading + sidecar
# ---------------------------------------------------------------------------

def test_load_reference_outputs_from_truncated_reproduction_samples():
    finding = {
        "reproduction": {
            "evidence": {
                "kv_corruption": {
                    "diverged_requests": ["r3", "r4"],
                    "sample_A": "u1277_4441 u1277_4445",
                }
            }
        }
    }
    references = load_reference_outputs(finding)
    assert set(references) == {"r3"}
    assert references["r3"].truncated is True
    assert references["r3"].matches("u1277_4441 u1277_4445 u1277_4449")


def test_load_reference_outputs_prefers_explicit_keyed_map():
    finding = {
        "reference_outputs": {"r1": "exact"},
        "reproduction": {
            "evidence": {"kv_corruption": {"diverged_requests": ["r3"],
                                           "sample_A": "trunc"}}
        },
    }
    references = load_reference_outputs(finding)
    assert set(references) == {"r1"}
    assert references["r1"].truncated is False
    assert references["r1"].matches("exact") and not references["r1"].matches("exactly")


def test_sidecar_payload_shape_and_write(tmp_path):
    finding_path = tmp_path / "finding_00007_123.json"
    original = {"trace": {"seed": 1, "events": []}, "bugs": []}
    finding_path.write_text(json.dumps(original))
    before = finding_path.read_bytes()

    comparisons = _divergent_comparisons()
    engagement = _proven()
    ruling = _allowed()
    decision = decide(engagement, ruling, comparisons, completed_runs=10)

    payload = build_sidecar(
        finding_path=finding_path,
        base_url="http://localhost:8000",
        requested_runs=10,
        completed_runs=10,
        engagement=engagement,
        ruling=ruling,
        decision=decision,
        comparisons=comparisons,
        excluded={"multi": "n_completions > 1"},
        server_version="0.19.0",
        wall_time_s=12.34,
    )

    assert payload["verdict"] == VERDICT_CORROBORATES
    assert payload["runs"] == {"requested": 10, "completed": 10}
    assert payload["guard"]["engagement"]["proven"] is True
    assert payload["guard"]["engagement"]["evidence_line"] == CASCADE_LINE
    assert payload["guard"]["backend"]["allowed"] is True
    assert payload["guard"]["backend"]["name"] == "FLASH_ATTN"
    assert payload["per_request"]["r1"]["unique_outputs"] == 2
    assert payload["server_version"] == "0.19.0"
    assert payload["timestamp"].endswith("Z")
    assert "attribution_state_machine.md" in payload["semantics"]

    dest = write_sidecar(payload, tmp_path / "bi", finding_path)
    assert dest.name == "finding_00007_123_bi_r2.json"
    assert json.loads(dest.read_text())["verdict"] == VERDICT_CORROBORATES
    # the raw finding is never modified
    assert finding_path.read_bytes() == before


def test_uninformative_sidecar_records_zero_runs_and_no_caveat(tmp_path):
    finding_path = tmp_path / "finding_00008_1.json"
    engagement = EngagementProof(proven=False, reason="no --vllm-log-file given")
    ruling = rule_backend(None)
    decision = decide(engagement, ruling)

    payload = build_sidecar(
        finding_path=finding_path,
        base_url="http://localhost:8000",
        requested_runs=10,
        completed_runs=0,
        engagement=engagement,
        ruling=ruling,
        decision=decision,
    )
    assert payload["verdict"] == VERDICT_UNINFORMATIVE
    assert payload["runs"]["completed"] == 0
    assert payload["per_request"] == {}
    assert "caveat" not in payload


def test_evaporation_sidecar_carries_the_caveat(tmp_path):
    runs = [_result(_request("r1", "stable")) for _ in range(10)]
    comparisons = compare_runs(runs, ("r1",))
    decision = decide(_proven(), _allowed(), comparisons, completed_runs=10)
    payload = build_sidecar(
        finding_path=tmp_path / "finding_00009_1.json",
        base_url="http://localhost:8000",
        requested_runs=10,
        completed_runs=10,
        engagement=_proven(),
        ruling=_allowed(),
        decision=decision,
        comparisons=comparisons,
    )
    assert payload["verdict"] == VERDICT_DOES_NOT_CORROBORATE
    assert payload["caveat"] == EVAPORATION_CAVEAT


# ---------------------------------------------------------------------------
# Output directory anchoring
# ---------------------------------------------------------------------------

def test_output_dir_is_anchored_on_the_repo_the_finding_lives_in(tmp_path):
    """A nested finding still resolves to <repo>/findings/reproduction/bi."""
    finding_path = tmp_path / "repo" / "findings" / "KV_CORRUPTION" / "f_1.json"
    finding_path.parent.mkdir(parents=True)
    finding_path.write_text("{}")

    resolved = resolve_output_dir(finding_path)
    assert resolved == (tmp_path / "repo").resolve() / DEFAULT_OUTPUT_DIR


def test_output_dir_anchoring_walks_past_deeply_nested_findings(tmp_path):
    finding_path = (
        tmp_path / "repo" / "findings" / "reproduction" / "confirmed" / "f_2.json"
    )
    finding_path.parent.mkdir(parents=True)
    finding_path.write_text("{}")

    assert resolve_output_dir(finding_path) == (
        tmp_path / "repo"
    ).resolve() / DEFAULT_OUTPUT_DIR


def test_explicit_output_dir_is_honoured_verbatim(tmp_path):
    finding_path = tmp_path / "repo" / "findings" / "f_3.json"
    finding_path.parent.mkdir(parents=True)
    finding_path.write_text("{}")

    assert resolve_output_dir(finding_path, "/tmp/elsewhere") == Path("/tmp/elsewhere")


def test_output_dir_falls_back_to_cwd_with_a_warning(tmp_path, capsys):
    finding_path = tmp_path / "loose" / "f_4.json"
    finding_path.parent.mkdir(parents=True)
    finding_path.write_text("{}")

    resolved = resolve_output_dir(finding_path)
    assert resolved == Path.cwd() / DEFAULT_OUTPUT_DIR
    assert "WARNING" in capsys.readouterr().err
