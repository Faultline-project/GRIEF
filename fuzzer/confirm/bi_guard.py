"""BI guards: prove batch-invariant mode is engaged, and rule on the backend.

Moved verbatim from ``scripts/bi_confirm.py`` (Guard 1 and Guard 2 plus the
backend / GPU-arch resolvers) so that the fuzzing loop's confirmation ladder
(``fuzzer.confirm.ladder`` and ``fuzzer.core.runner``) can use them without
importing from ``scripts/``.  ``scripts/bi_confirm.py`` re-imports every name
defined here, so its CLI and ``tests/test_bi_confirm.py`` are unchanged.

Pure: no vLLM import, no network.  ``prove_engagement`` reads a server log file;
``resolve_gpu_arch`` falls back to a local ``torch.cuda`` query only when no
other source names the architecture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

ATTRIBUTION_DOC = "docs/attribution_state_machine.md"
ATTRIBUTION_SECTION = "R2 — BI as three-state corroborating evidence"

_SAMPLE_CHARS = 240


# ---------------------------------------------------------------------------
# Guard 1 — engagement proof
#
# vLLM 0.19 has NO unconditional "batch invariance enabled" log line: the only
# logger call inside batch_invariant.py fires exclusively on the *non*-decode-
# invariant path.  Engagement is therefore proven from BI-conditioned log lines
# emitted by call sites that are reached only when envs.VLLM_BATCH_INVARIANT is
# truthy.  Each pattern below names the source line it mirrors.
#
# On 0.19 the primary signal was the cascade-attention line: emitted during
# VllmConfig validation in the serving process, its text names the environment
# variable itself (so it cannot be produced by any other configuration).
#
# !! VERSION REGRESSION (measured on vLLM 0.27.1, 2026-08-23) !!
# The cascade-attention signal is DEAD on 0.27.1 and cannot fire on any config.
# `ModelConfig.disable_cascade_attn` now defaults to **True**
# (vllm/config/model.py:266), so the guard at vllm/config/vllm.py:1560-1568
#     if self.model_config and envs.VLLM_BATCH_INVARIANT \
#            and not self.model_config.disable_cascade_attn:
# is never entered — BI has no cascade attention left to disable.  On 0.19 the
# default was False, so the line fired.  Measured: 0 occurrences of "cascade"
# in a genuine VLLM_BATCH_INVARIANT=1 0.27.1 server log.
#
# The other three 0.19 signals are all conditional on configs outside the R2
# reference config (FA4 / FLASHINFER / TRITON_MLA / mamba), so on the reference
# config (FLASH_ATTN @ FA2 + bf16) *no 0.27.1 log line names VLLM_BATCH_INVARIANT
# at all*.  Engagement on 0.27.1 is therefore proven from BI-conditioned
# observables instead — see the 0.27.1 entries in BI_ENGAGEMENT_SIGNALS.
#
# Rejected alternatives, recorded so the choice is auditable:
#   * an importable check (`vllm.envs.VLLM_BATCH_INVARIANT` /
#     `vllm_is_batch_invariant`, which does not exist in local 0.19) reads the
#     env of *this* client process, which says nothing about the server;
#   * the launcher's own `export VLLM_BATCH_INVARIANT=1` echo proves the
#     variable was exported to the launcher, not that the engine initialized BI
#     (it is recorded as a weak hint and never as proof — see
#     ``BI_NON_PROOF_SIGNALS``);
#   * there is no server endpoint exposing engine env/config in 0.19 or 0.27.1.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogSignal:
    """A log line pattern with the upstream source it mirrors."""

    name: str
    pattern: str
    source: str
    rationale: str


BI_ENGAGEMENT_SIGNALS: tuple[LogSignal, ...] = (
    LogSignal(
        name="cascade_attention_disabled",
        pattern=r"Disabling cascade attention when VLLM_BATCH_INVARIANT is enabled",
        source="vllm/config/vllm.py:1157-1166 (vLLM 0.19)",
        rationale=(
            "Emitted only under `envs.VLLM_BATCH_INVARIANT and not "
            "disable_cascade_attn`; the message names the env var, so it "
            "cannot be produced by any other configuration. Primary signal."
        ),
    ),
    LogSignal(
        name="non_decode_invariant_warning",
        pattern=r"You are using a non-decode-invariant form of batch invariance",
        source="vllm/model_executor/layers/batch_invariant.py:1018-1023 (vLLM 0.19)",
        rationale=(
            "Emitted from override_envs_for_invariance(), which is reached "
            "only from init_batch_invariance() under `if "
            "envs.VLLM_BATCH_INVARIANT`. Proves engagement inside the worker, "
            "but fires only for backends outside {FLASH_ATTN, TRITON_ATTN} — "
            "so seeing it also means the backend is not decode-invariant."
        ),
    ),
    LogSignal(
        name="prefix_caching_disabled_for_bi",
        pattern=r"Disabling prefix caching for .*with batch invariance",
        source=(
            "vllm/model_executor/layers/attention/attention.py:328-333 and "
            "mla_attention.py:380-385 (vLLM 0.19)"
        ),
        rationale=(
            "BI-conditioned; fires only for FLASHINFER / TRITON_MLA with "
            "prefix caching on — both outside the R2 allowlist."
        ),
    ),
    LogSignal(
        name="fa4_downgraded_for_bi",
        pattern=r"Cannot use FA version 4 with batch invariance",
        source="vllm/v1/attention/backends/fa_utils.py:117-122 (vLLM 0.19)",
        rationale=(
            "Emitted under `envs.VLLM_BATCH_INVARIANT and fa_version == 4`; "
            "proves engagement and records the FA4 -> FA2 downgrade. Still "
            "present on 0.27.1 at fa_utils.py:235-240 with identical text, but "
            "only reachable when the selected FA version is 4."
        ),
    ),
    # -- vLLM 0.27.1 signals -------------------------------------------------
    # Added 2026-08-23 after the four 0.19 signals were measured to produce
    # UNINFORMATIVE against a genuine VLLM_BATCH_INVARIANT=1 0.27.1 server
    # (FLASH_ATTN @ FA2, bf16, sm_89).  None of the 0.19 patterns can fire on
    # that config; see the VERSION REGRESSION note above.
    LogSignal(
        name="bi_matmul_in_compiled_graph_027",
        pattern=r"vllm/model_executor/layers/batch_invariant\.py",
        source=(
            "vllm/model_executor/layers/batch_invariant.py:141 "
            "(matmul_persistent), installed by enable_batch_invariant_mode() "
            "called from init_batch_invariance() at batch_invariant.py:984-987 "
            "under `if envs.VLLM_BATCH_INVARIANT` (vLLM 0.27.1)"
        ),
        rationale=(
            "STRONGEST 0.27.1 signal. torch.compile/Dynamo traces the BI matmul "
            "override and prints its source path while compiling the model, so "
            "the module path appears in the server log only when the BI kernel "
            "overrides were actually installed in the engine process. Measured "
            "2026-08-23: 2 occurrences under BI=1, 0 occurrences on a stock "
            "server of the same version/model. Depends on torch emitting the "
            "lru_cache trace warning, so it is necessary-but-not-guaranteed; "
            "the config-echo signal below is the independent cross-check."
        ),
    ),
    LogSignal(
        name="custom_all_reduce_disabled_for_bi_027",
        pattern=r"disable_custom_all_reduce=True",
        source="vllm/config/parallel.py:1003-1004 (vLLM 0.27.1)",
        rationale=(
            "ParallelConfig._verify_args sets disable_custom_all_reduce=True "
            "under `if envs.VLLM_BATCH_INVARIANT`, and the flag is echoed in "
            "the engine config INFO line (core.py:121). Measured 2026-08-23: "
            "True under BI=1 vs False on the stock server, same box/model. "
            "CAVEAT: parallel.py:1025-1026 also sets it when the platform has "
            "no custom allreduce, so on multi-GPU/unsupported platforms this "
            "is not BI-exclusive — it is corroborating, not conclusive, and is "
            "why it is listed after the matmul signal."
        ),
    ),
)

BI_NON_PROOF_SIGNALS: tuple[LogSignal, ...] = (
    LogSignal(
        name="launcher_env_echo",
        pattern=r"VLLM_BATCH_INVARIANT\s*[=:]\s*1",
        source="launcher echo, e.g. a reproduction launcher",
        rationale=(
            "Proves the launcher exported the variable, not that the engine "
            "initialized batch-invariant mode. Recorded as a hint only."
        ),
    ),
)


@dataclass
class EngagementProof:
    """Outcome of guard 1."""

    proven: bool
    reason: str
    signal: Optional[str] = None
    evidence_line: Optional[str] = None
    source: Optional[str] = None
    weak_hint: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "proven": self.proven,
            "signal": self.signal,
            "evidence_line": self.evidence_line,
            "source": self.source,
            "reason": self.reason,
            "weak_hint": self.weak_hint,
        }


def prove_engagement(log_file: Optional[str]) -> EngagementProof:
    """Prove batch-invariant mode initialized in the server, from its log.

    Absence of proof is never treated as engagement: a missing log file, an
    unreadable log, or a log with no BI-conditioned line all return
    ``proven=False``, which the caller turns into UNINFORMATIVE.
    """
    if not log_file:
        return EngagementProof(
            proven=False,
            reason=(
                "no --vllm-log-file given; batch-invariant engagement cannot "
                "be proven and is never assumed"
            ),
        )

    path = Path(log_file)
    if not path.exists():
        return EngagementProof(
            proven=False,
            reason=f"server log not found: {path}",
        )

    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return EngagementProof(
            proven=False,
            reason=f"server log unreadable: {exc}",
        )

    lines = text.splitlines()
    for signal in BI_ENGAGEMENT_SIGNALS:
        rx = re.compile(signal.pattern)
        for line in lines:
            if rx.search(line):
                return EngagementProof(
                    proven=True,
                    reason=f"matched {signal.name} in {path.name}",
                    signal=signal.name,
                    evidence_line=line.strip()[:_SAMPLE_CHARS],
                    source=signal.source,
                )

    weak_hint: Optional[str] = None
    for signal in BI_NON_PROOF_SIGNALS:
        rx = re.compile(signal.pattern)
        for line in lines:
            if rx.search(line):
                weak_hint = (
                    f"{signal.name}: {line.strip()[:_SAMPLE_CHARS]} "
                    f"({signal.rationale})"
                )
                break
        if weak_hint:
            break

    return EngagementProof(
        proven=False,
        reason=(
            f"no batch-invariant engagement line in {path.name}; searched "
            + ", ".join(s.name for s in BI_ENGAGEMENT_SIGNALS)
        ),
        weak_hint=weak_hint,
    )


# ---------------------------------------------------------------------------
# Guard 2 — backend allowlist
#
# Canonical rule (R2): "the campaign backend
# must be in the BI allowlist — FA3-on-Hopper is not a sound BI backend and
# FlashInfer is excluded — otherwise the outcome is UNINFORMATIVE."
# ---------------------------------------------------------------------------

BI_BACKEND_ALLOWLIST: dict[str, str] = {
    "FLASH_ATTN@FA2": (
        "FLASH_ATTN running FA2 is the only backend R2 admits: it is the "
        "decode-invariant backend upstream calls 'best supported' "
        "(batch_invariant.py override_envs_for_invariance) and the reference "
        "BI server config for this project uses it."
    ),
}

BI_BACKEND_EXCLUSIONS: dict[str, str] = {
    "FLASHINFER": (
        "excluded outright — upstream removed FLASHINFER from the BI "
        "admission list over non-invariant CTA sizes (FlashInfer issue "
        "#2424), and prefix caching is force-disabled under BI for it."
    ),
    "FLASH_ATTN@FA3/sm90": (
        "FA3 on Hopper (sm90) is not a sound BI backend: the june_27 "
        "investigation showed FA3 remains batch-shape sensitive under BI=1, "
        "so a BI replay on it cannot separate batch numerics from a real "
        "defect. Note vLLM DEFAULTS to FA3 on sm90 "
        "(fa_utils.get_flash_attn_version: capability 9 -> FA3), so "
        "'FLASH_ATTN on H100' means FA3 unless the version is pinned to 2."
    ),
}

# Backends that exist but are outside the R2 allowlist.  Not "excluded" (no
# positive finding that they are unsound) and not admitted — both route to
# UNINFORMATIVE, with distinct reasons.
BI_BACKEND_KNOWN_OTHER: tuple[str, ...] = (
    "TRITON_ATTN",
    "TRITON_MLA",
    "FLASH_ATTN_MLA",
    "FLASHMLA",
    "FLEX_ATTENTION",
    "FLASHINFER_MLA",
    "ROCM_AITER_FA",
    "XFORMERS",
)

_BACKEND_ALIASES: dict[str, tuple[str, Optional[int]]] = {
    # alias -> (family, implied FA version)
    "FLASH_ATTN": ("FLASH_ATTN", None),
    "FLASHATTENTION": ("FLASH_ATTN", None),
    "FLASH_ATTN_VLLM_V1": ("FLASH_ATTN", None),
    "FA": ("FLASH_ATTN", None),
    "FA2": ("FLASH_ATTN", 2),
    "FLASH_ATTN_2": ("FLASH_ATTN", 2),
    "FLASHATTENTION2": ("FLASH_ATTN", 2),
    "FA3": ("FLASH_ATTN", 3),
    "FLASH_ATTN_3": ("FLASH_ATTN", 3),
    "FLASHATTENTION3": ("FLASH_ATTN", 3),
    "FA4": ("FLASH_ATTN", 4),
    "FLASH_ATTN_4": ("FLASH_ATTN", 4),
    "FLASHINFER": ("FLASHINFER", None),
    "FLASHINFER_VLLM_V1": ("FLASHINFER", None),
}

_HOPPER_TOKENS = ("SM90", "SM_90", "9.0", "HOPPER", "H100", "H200", "H800", "GH200")
_NON_HOPPER_TOKENS: tuple[tuple[str, str], ...] = (
    ("SM100", "sm100"), ("SM_100", "sm100"), ("B200", "sm100"),
    ("B100", "sm100"), ("GB200", "sm100"), ("BLACKWELL", "sm100"),
    ("SM89", "sm89"), ("SM_89", "sm89"), ("L40", "sm89"), ("L4", "sm89"),
    ("ADA", "sm89"),
    # GeForce Ada (compute capability 8.9). Added 2026-08-23: without these the
    # RTX 4090 campaign box resolved to UNKNOWN_ARCH, infer_fa_version returned
    # None, and every R2 run returned UNINFORMATIVE on an otherwise BI-sound
    # FLASH_ATTN@FA2 config. Verified against
    # vllm/v1/attention/backends/fa_utils.py:163-171 (0.27.1): capability
    # major 9 -> FA3, major 10 -> FA4, else FA2 — so sm_89 selects FA2, which
    # the server log confirms ("Using FlashAttention version 2").
    ("RTX_4090", "sm89"), ("RTX_4080", "sm89"),
    ("RTX_4070", "sm89"), ("RTX_4060", "sm89"),
    ("SM80", "sm80"), ("SM_80", "sm80"), ("A100", "sm80"),
    ("A800", "sm80"), ("AMPERE", "sm80"),
    # Workstation Ampere (compute capability 8.6).  Added 2026-09-28 for the
    # FSE campaigns on RTX A6000s: without it "NVIDIA RTX A6000"
    # resolved to UNKNOWN_ARCH.  Listed after "ADA" above, so the RTX A6000
    # Ada (sm89) still resolves to sm89.  0.27.1 fa_utils: major 8 -> FA2.
    ("SM86", "sm86"), ("SM_86", "sm86"), ("A6000", "sm86"),
)


@dataclass(frozen=True)
class GpuArch:
    """Resolved GPU architecture, with provenance."""

    raw: Optional[str]
    label: Optional[str]          # "sm90", "sm80", ... or None when unresolved
    is_hopper: Optional[bool]     # None = unknown, never guessed
    source: str = "unresolved"

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "label": self.label,
            "is_hopper": self.is_hopper,
            "source": self.source,
        }


UNKNOWN_ARCH = GpuArch(raw=None, label=None, is_hopper=None)


def normalize_gpu_arch(raw: Optional[str], source: str = "override") -> GpuArch:
    """Map a free-form GPU string ('H100', 'sm90', 'NVIDIA H100 80GB') to an arch."""
    if not raw:
        return UNKNOWN_ARCH
    token = raw.upper().replace("-", "_").replace(" ", "_")
    for hopper in _HOPPER_TOKENS:
        if hopper in token:
            return GpuArch(raw=raw, label="sm90", is_hopper=True, source=source)
    for needle, label in _NON_HOPPER_TOKENS:
        if needle in token:
            return GpuArch(raw=raw, label=label, is_hopper=False, source=source)
    return GpuArch(raw=raw, label=None, is_hopper=None, source=source)


def normalize_backend(raw: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Return ``(family, implied_fa_version)`` for a backend string."""
    if not raw:
        return None, None
    token = raw.strip().upper().replace("-", "_")
    if token in ("", "UNKNOWN", "NONE", "AUTO"):
        return None, None
    if token in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[token]
    return token, None


def infer_fa_version(arch: GpuArch) -> tuple[Optional[int], str]:
    """Infer the FlashAttention version vLLM would select on *arch*.

    Mirrors ``vllm/v1/attention/backends/fa_utils.get_flash_attn_version``:
    capability 9 -> FA3, capability 10 -> FA4, otherwise FA2.
    """
    if arch.is_hopper is True:
        return 3, "inferred from sm90 default (fa_utils: capability 9 -> FA3)"
    if arch.label == "sm100":
        return 4, "inferred from sm100 default (fa_utils: capability 10 -> FA4)"
    if arch.label in ("sm80", "sm86", "sm89"):
        return 2, f"inferred from {arch.label} default (fa_utils: FA2)"
    return None, "gpu arch unresolved — FA version not inferable"


@dataclass
class BackendRuling:
    """Outcome of guard 2."""

    name: Optional[str]
    allowed: bool
    reason: str
    status: str                       # ALLOWED | EXCLUDED | NOT_ALLOWLISTED | UNKNOWN
    raw: Optional[str] = None
    fa_version: Optional[int] = None
    fa_version_source: str = "unset"
    gpu_arch: GpuArch = UNKNOWN_ARCH
    resolved_from: str = "unresolved"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "raw": self.raw,
            "fa_version": self.fa_version,
            "fa_version_source": self.fa_version_source,
            "gpu_arch": self.gpu_arch.to_dict(),
            "resolved_from": self.resolved_from,
            "allowlist_doc": f"{ATTRIBUTION_DOC} — {ATTRIBUTION_SECTION}",
        }


def rule_backend(
    raw_backend: Optional[str],
    arch: GpuArch = UNKNOWN_ARCH,
    fa_version: Optional[int] = None,
    resolved_from: str = "unresolved",
    fa_version_source: str = "declared",
) -> BackendRuling:
    """Apply the R2 BI backend allowlist.

    ALLOWED is exactly {FLASH_ATTN at FA2}.  FlashInfer is excluded outright,
    FA3-on-Hopper is excluded as unsound, every other backend — and every
    unresolved one — is UNINFORMATIVE rather than assumed sound.
    """
    family, implied_fa = normalize_backend(raw_backend)

    if family is None:
        return BackendRuling(
            name=None,
            allowed=False,
            status="UNKNOWN",
            reason=(
                "attention backend could not be resolved; R2 never assumes a "
                "BI-sound backend — pass --backend to declare it"
            ),
            raw=raw_backend,
            gpu_arch=arch,
            resolved_from=resolved_from,
        )

    if family == "FLASHINFER":
        return BackendRuling(
            name=family,
            allowed=False,
            status="EXCLUDED",
            reason=BI_BACKEND_EXCLUSIONS["FLASHINFER"],
            raw=raw_backend,
            gpu_arch=arch,
            resolved_from=resolved_from,
        )

    if family != "FLASH_ATTN":
        known = family in BI_BACKEND_KNOWN_OTHER
        return BackendRuling(
            name=family,
            allowed=False,
            status="NOT_ALLOWLISTED",
            reason=(
                f"{family} is {'a known backend but ' if known else ''}outside "
                "the R2 BI allowlist (ALLOWED = FLASH_ATTN at FA2 only)"
            ),
            raw=raw_backend,
            gpu_arch=arch,
            resolved_from=resolved_from,
        )

    # FLASH_ATTN family — the ruling turns on the FA version.
    if fa_version is not None:
        version, version_source = fa_version, fa_version_source
    elif implied_fa is not None:
        version, version_source = implied_fa, "implied by backend name"
    else:
        version, version_source = infer_fa_version(arch)

    if version == 2:
        return BackendRuling(
            name=family,
            allowed=True,
            status="ALLOWED",
            reason=BI_BACKEND_ALLOWLIST["FLASH_ATTN@FA2"],
            raw=raw_backend,
            fa_version=2,
            fa_version_source=version_source,
            gpu_arch=arch,
            resolved_from=resolved_from,
        )

    if version == 3:
        return BackendRuling(
            name=family,
            allowed=False,
            status="EXCLUDED",
            reason=BI_BACKEND_EXCLUSIONS["FLASH_ATTN@FA3/sm90"]
            + f" (FA version {version_source})",
            raw=raw_backend,
            fa_version=3,
            fa_version_source=version_source,
            gpu_arch=arch,
            resolved_from=resolved_from,
        )

    if version == 4:
        return BackendRuling(
            name=family,
            allowed=False,
            status="NOT_ALLOWLISTED",
            reason=(
                "FA4 is not BI-sound (batch-shape-dependent scheduling on "
                "SM100+); vLLM downgrades it to FA2 under BI "
                "(fa_utils.py:117-122), but that downgrade must be observed in "
                "the server log — declare --fa-version 2 once it is"
            ),
            raw=raw_backend,
            fa_version=4,
            fa_version_source=version_source,
            gpu_arch=arch,
            resolved_from=resolved_from,
        )

    return BackendRuling(
        name=family,
        allowed=False,
        status="UNKNOWN",
        reason=(
            "FLASH_ATTN with an unresolved FA version: on sm90 vLLM defaults "
            "to FA3, which is excluded, so FA2 cannot be assumed — pass "
            "--fa-version or --gpu-arch"
        ),
        raw=raw_backend,
        fa_version=None,
        fa_version_source=version_source,
        gpu_arch=arch,
        resolved_from=resolved_from,
    )


# ---------------------------------------------------------------------------
# Backend / arch resolution from the finding, its context manifest, the server
# log, and (last) a server probe.
# ---------------------------------------------------------------------------

_FLAG_BACKEND_PATTERNS = (
    r"--attention[-_]backend[=\s]+([A-Za-z0-9_]+)",
    r"VLLM_ATTENTION_BACKEND[=:]\s*([A-Za-z0-9_]+)",
)
_FLAG_FA_VERSION_PATTERNS = (
    r"VLLM_FLASH_ATTN_VERSION[=:]\s*(\d)",
    r"--flash[-_]attn[-_]version[=\s]+(\d)",
)
_LOG_BACKEND_PATTERNS = (
    r"Using ([A-Z0-9_]+) backend\.",
    r"KV cache layout for ([A-Z0-9_]+) backend",
    # vLLM 0.27.1 phrasing (added 2026-08-23). The two patterns above match
    # neither line emitted by 0.27.1, so backend resolution fell through to
    # UNKNOWN and every R2 run on a 0.27.1 campaign returned UNINFORMATIVE.
    # Source: vllm/platforms/cuda.py:481-485 —
    #   logger.info_once("Using %s attention backend out of potential "
    #                    "backends: %s.", selected_backend.name, ...)
    r"Using ([A-Z0-9_]+) attention backend",
    # vLLM 0.27.1 with an EXPLICIT --attention-backend (added 2026-09-28,
    # measured on the FSE A6000 smoke job 7685301): cuda.py:422 logs the enum
    # member, "Using AttentionBackendEnum.FLASH_ATTN backend.", which the
    # first pattern cannot match because of the dot.
    r"Using AttentionBackendEnum\.([A-Z0-9_]+) backend",
)
# FA version read from the server log. vLLM logs the *resolved* version, which
# is more authoritative than inferring it from the GPU arch, and it already
# reflects any BI-driven downgrade (fa_utils.py:235-240).
# Source: vllm/v1/attention/backends/flash_attn.py:789-792 —
#   logger.info_once("Using FlashAttention version %s", ...)   (vLLM 0.27.1)
_LOG_FA_VERSION_PATTERNS = (
    r"Using FlashAttention version (\d)",
)


def _first_match(patterns: Iterable[str], text: str) -> Optional[str]:
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def _walk_strings(obj: Any) -> Iterable[str]:
    """Yield every string value nested inside *obj* (dicts/lists/scalars)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _walk_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _walk_strings(value)


def _search_backend_key(obj: Any) -> Optional[str]:
    """Find an attention-backend value by key name anywhere inside *obj*."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and "attention_backend" in str(key).lower():
                return value
        for value in obj.values():
            found = _search_backend_key(value)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            found = _search_backend_key(value)
            if found:
                return found
    return None


def resolve_backend(
    finding: dict,
    context_manifest: Optional[dict] = None,
    log_text: str = "",
    server_info: Optional[dict] = None,
    override: Optional[str] = None,
) -> tuple[Optional[str], Optional[int], str]:
    """Resolve ``(backend, fa_version, source)`` for the campaign under test.

    Order: explicit override, the finding's own engine config / server flags,
    the finding's context manifest, the server log, then a server probe.
    Anything left unresolved stays ``None`` — the caller turns that into
    UNINFORMATIVE.
    """
    fa_version: Optional[int] = None

    flags = " ".join(
        s for s in (
            str(finding.get("server_flags") or ""),
            str(finding.get("server_env") or ""),
        ) if s
    )
    fa_raw = _first_match(_FLAG_FA_VERSION_PATTERNS, flags)
    if fa_raw:
        fa_version = int(fa_raw)

    # Fall back to the version the server itself reported. Declared flags win,
    # but a 0.27.1 campaign usually declares nothing (VLLM_ATTENTION_BACKEND was
    # removed in 0.27.x — the server logs "Unknown vLLM environment variable
    # detected: VLLM_ATTENTION_BACKEND" and ignores it), so without this the FA
    # version had to be guessed from the GPU arch.
    if fa_version is None and log_text:
        fa_log = _first_match(_LOG_FA_VERSION_PATTERNS, log_text)
        if fa_log:
            fa_version = int(fa_log)

    if override:
        return override, fa_version, "override"

    declared = finding.get("attention_backend") or _search_backend_key(
        finding.get("capture_manifest")
    )
    if declared:
        return declared, fa_version, "finding.capture_manifest/attention_backend"

    from_flags = _first_match(_FLAG_BACKEND_PATTERNS, flags)
    if from_flags:
        return from_flags, fa_version, "finding.server_flags"

    if context_manifest:
        declared = _search_backend_key(context_manifest)
        if declared:
            return declared, fa_version, "context manifest"
        for value in _walk_strings(context_manifest):
            from_manifest = _first_match(_FLAG_BACKEND_PATTERNS, value)
            if from_manifest:
                return from_manifest, fa_version, "context manifest"

    if log_text:
        from_log = _first_match(_LOG_BACKEND_PATTERNS, log_text)
        if from_log:
            return from_log, fa_version, "server log"

    if server_info:
        declared = _search_backend_key(server_info)
        if declared:
            return declared, fa_version, "server probe"

    return None, fa_version, "unresolved"


def resolve_gpu_arch(
    finding: dict,
    log_text: str = "",
    override: Optional[str] = None,
) -> GpuArch:
    """Resolve the GPU architecture, best effort, never guessing."""
    if override:
        return normalize_gpu_arch(override, source="override")

    declared = finding.get("gpu_type") or finding.get("gpu")
    if declared:
        arch = normalize_gpu_arch(str(declared), source="finding.gpu_type")
        if arch.is_hopper is not None:
            return arch

    if log_text:
        # NOTE (2026-08-23): `NVIDIA\s+[A-Z0-9]+` cannot match a GeForce name —
        # on "NVIDIA GeForce RTX 4090" the character class stops at the "G" of
        # "GeForce" (lowercase 'e' ends the match), yielding "NVIDIA G", which
        # normalizes to UNKNOWN.
        # NOTE (2026-09-28): putting the RTX alternative first in ONE regex did
        # not help: re.search returns the LEFTMOST match, and "NVIDIA" precedes
        # "RTX" in the device name, so the generic alternative always won
        # ("NVIDIA RTX A6000" -> "NVIDIA RTX" -> UNKNOWN).  The patterns are
        # therefore tried one at a time, specific models first.
        m = None
        for pat in (
            r"RTX\s+A\d{4}(?:\s+Ada)?",
            r"RTX\s+\d{4}",
            r"GH200|H100|H200|H800|A100|L40S?|GB200|B200",
            r"NVIDIA\s+[A-Z0-9]+",
        ):
            m = re.search(f"({pat})", log_text)
            if m:
                break
        if m:
            arch = normalize_gpu_arch(m.group(1), source="server log")
            if arch.is_hopper is not None:
                return arch

    try:  # last resort: the local device, only valid when client == server host
        import torch  # noqa: PLC0415 — optional, GPU-only dependency

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            arch = normalize_gpu_arch(
                name, source="local probe (assumes client and server share a host)"
            )
            if arch.is_hopper is not None:
                return arch
    except Exception:
        pass

    return UNKNOWN_ARCH
