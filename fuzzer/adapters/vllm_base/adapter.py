"""
VllmBaseAdapter: concrete TargetAdapter for vLLM serving.

Implements all abstract methods from TargetAdapter using vLLM-specific
logic extracted from fuzzer/runner.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Optional

from ...core.adapter import TargetAdapter
from ...core.collector import AdapterCollector, CollectionAnomaly
from ...core.types import OracleResult
from .mutators import VLLM_MUTATIONS


# ---------------------------------------------------------------------------
# Unicode stress pools
# ---------------------------------------------------------------------------

_UNICODE_POOLS: dict = {
    "cjk":       ["你好", "世界", "深度学习", "模型", "推理", "令牌"],
    "emoji":     ["🔥", "💡", "🤖", "✨", "🧠", "⚡", "🌍"],
    "combining": ["a\u0301", "e\u0308", "n\u0303", "o\u0302"],
    "rtl_mixed": ["\u05e9\u05dc\u05d5\u05dd", "hello", "\u0645\u0631\u062d\u0628\u0627", "world"],
}


def _interleave_unicode(tokens: list, stress_type: str) -> list:
    """Interleave unicode tokens from the pool every 3rd position."""
    pool = _UNICODE_POOLS.get(stress_type, [])
    if not pool:
        return tokens
    result = []
    pool_idx = 0
    for i, tok in enumerate(tokens):
        result.append(tok)
        if (i + 1) % 3 == 0:
            result.append(pool[pool_idx % len(pool)])
            pool_idx += 1
    return result


def _stable_hash(s: str) -> int:
    """MD5-based stable hash truncated to 4 decimal digits.

    Unlike Python's hash(), this is process- and run-stable, so the same
    prompt_family_id or request_id always maps to the same unique suffix
    across runs, workers, and Python versions.
    """
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 10_000


def _perm_idx(rid_hash: int, position: int) -> int:
    """Map (rid_hash, position) to a pseudo-random but deterministic index.

    Hashes the combined key so that consecutive positions produce scattered
    indices, breaking the trivial u{h}_0 u{h}_1 u{h}_2 ... counting sequence
    that language models can complete by arithmetic pattern matching regardless
    of which LoRA adapter is applied.

    Collisions (two positions mapping to the same index) are acceptable by
    design — token strings are prompt text, not unique identifiers, and the
    occasional repeated token does not compromise the prompt's unpredictability.

    Returns a value in [0, 9999] (same range as _stable_hash) so token strings
    remain in the form u{h}_{0000–9999}.
    """
    key = f"{rid_hash}:{position}"
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 10_000


def _build_prompt(event) -> str:
    """
    Build a prompt string that honours prefix_len for real KV prefix sharing.

    The first `prefix_len` tokens are deterministic shared tokens ("pre0 pre1 …")
    so requests with identical prefix_len share the same radix-tree blocks.
    The remaining tokens are unique per (prompt_family_id or request_id) to
    prevent accidental cache hits across logically different requests.

    When prompt_family_id is set, all requests sharing that ID and prefix_len
    receive byte-identical prompts — required for valid isolation oracle comparisons.

    The unique suffix uses hash-scrambled indices (via _perm_idx) rather than
    sequential 0, 1, 2, ... so the model cannot complete the suffix by arithmetic
    pattern matching. This ensures LoRA adapter weights have genuine influence on
    the output, making the isolation oracle sensitive to KV contamination.
    """
    total = event.prompt_len or 16
    plen  = min(event.prefix_len or 0, total)
    ulen  = total - plen
    shared   = [f"pre{i}" for i in range(plen)]
    seed_str = getattr(event, "prompt_family_id", None) or event.request_id
    rid_hash = _stable_hash(seed_str)
    if getattr(event, "prompt_style", None) == "rejection_heavy":
        # Each position gets its own hash-derived family so no two adjacent
        # tokens share the same u-prefix.  This defeats n-gram draft lookup
        # (which requires a repeated sequence) without reducing prompt length.
        unique = [
            f"u{_stable_hash(seed_str + ':pos' + str(i))}_{_perm_idx(rid_hash, i)}"
            for i in range(ulen)
        ]
    else:
        unique = [f"u{rid_hash}_{_perm_idx(rid_hash, i)}" for i in range(ulen)]
    tokens = shared + unique
    if event.unicode_stress is not None:
        tokens = _interleave_unicode(tokens, event.unicode_stress)
    return " ".join(tokens)


def _build_chat_messages(event) -> list:
    """
    Build chat messages for /v1/chat/completions with prefix-cache-aware content.

    System messages use a shared deterministic token sequence ("sys0 sys1 …")
    so all requests with the same system content_len share the same KV block.
    User messages use unique tokens per (prompt_family_id or request_id) to
    prevent accidental hits — same prompt_family_id guarantees identical content.
    This mirrors the real-world pattern of a fixed system prompt + varying user
    queries, which is the primary prefix-caching hot path in chat models.
    """
    seed_str = getattr(event, "prompt_family_id", None) or event.request_id
    rid_hash = _stable_hash(seed_str)
    result = []
    for i, msg in enumerate(event.messages or []):
        role = msg.get("role", "user")
        content_len = max(1, msg.get("content_len", 16))
        if role == "system":
            # Shared deterministic content — same across all requests
            tokens = [f"sys{j}" for j in range(content_len)]
            # For "rtl_mixed" stress type also apply to system messages
            if event.unicode_stress is not None:
                if event.unicode_stress == "rtl_mixed" or role != "system":
                    tokens = _interleave_unicode(tokens, event.unicode_stress)
        elif role == "user":
            # Unique per request_id — prevents accidental cache deduplication
            if getattr(event, "prompt_style", None) == "rejection_heavy":
                tokens = [
                    f"u{_stable_hash(seed_str + ':turn' + str(i) + ':pos' + str(j))}_{_perm_idx(rid_hash, j)}"
                    for j in range(content_len)
                ]
            else:
                tokens = [f"u{rid_hash}_t{i}_{j}" for j in range(content_len)]
            if event.unicode_stress is not None:
                tokens = _interleave_unicode(tokens, event.unicode_stress)
        else:
            # Assistant turns: deterministic but turn-indexed
            tokens = [f"ast{i}_{j}" for j in range(content_len)]
            if event.unicode_stress is not None:
                tokens = _interleave_unicode(tokens, event.unicode_stress)
        result.append({"role": role, "content": " ".join(tokens)})
    return result


class VllmBaseAdapter(TargetAdapter):
    """Adapter for vLLM OpenAI-compatible serving."""

    def __init__(self):
        self._cached_model_name: str = ""
        # Lazily built in evaluate_extra; the schema oracle is stateless w.r.t.
        # the tuned thresholds, so one instance is reused for the whole run.
        self._extra_oracle = None

    def build_payload(self, event) -> tuple[str, dict]:
        """Return (endpoint, payload_dict) for a SEND Event.

        Note: This method is not called by PythonExecutor._send() directly —
        the executor uses module-level _build_prompt/_build_chat_messages.
        This method is provided for future use by a generic executor.
        """
        if event.messages:
            endpoint = "/v1/chat/completions"
            payload = {
                "messages":    _build_chat_messages(event),
                "max_tokens":  event.max_tokens or 64,
                "stream":      event.stream,
                "temperature": event.temperature if event.temperature is not None else 0.0,
            }
        else:
            endpoint = "/v1/completions"
            payload = {
                "prompt":      _build_prompt(event),
                "max_tokens":  event.max_tokens or 64,
                "stream":      event.stream,
                "temperature": event.temperature if event.temperature is not None else 0.0,
            }
        return endpoint, payload

    async def check_liveness(self, base_url: str) -> bool:
        """Return True if the server is healthy."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{base_url}/health", timeout=3.0
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def flush_state(self, base_url: str) -> Optional[float]:
        """Evict reclaimable server state. Returns kv_cache_pct if available."""
        import httpx
        import uuid

        model_name = self._cached_model_name
        # Tokens must be unique across every call — no prefix sharing with cached content
        flush_tokens = " ".join(f"flush{uuid.uuid4().hex[:6]}_{i}" for i in range(2048))
        payload: dict = {
            "prompt":      flush_tokens,
            "max_tokens":  1,
            "temperature": 0.0,
            "stream":      False,
        }
        if model_name:
            payload["model"] = model_name
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{base_url}/v1/completions",
                    json=payload,
                    timeout=60.0,
                )
        except Exception:
            pass
        await asyncio.sleep(1.0)   # allow LRU eviction to complete
        return None  # kv_cache_pct obtained by caller via monitor.peek()

    async def resolve_model_name(self, base_url: str, config_model_name: str) -> str:
        """Return the model name to use in requests."""
        if config_model_name:
            self._cached_model_name = config_model_name
            return config_model_name
        if self._cached_model_name:
            return self._cached_model_name
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{base_url}/v1/models", timeout=5.0
                )
                data = resp.json()
                name = data["data"][0]["id"]
                self._cached_model_name = name
                return name
        except Exception:
            return ""

    async def discover_version(self, base_url: str) -> dict:
        """Return version metadata dict."""
        info: dict = {}

        # 1. HTTP /version endpoint
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{base_url}/version", timeout=5.0
                )
                if resp.status_code == 200:
                    data = resp.json()
                    info["vllm_version"] = data.get("version", "unknown")
        except Exception:
            pass

        # 2. importlib.metadata fallback
        if "vllm_version" not in info:
            try:
                from importlib.metadata import version as _pkg_version
                info["vllm_version"] = _pkg_version("vllm")
            except Exception:
                info["vllm_version"] = "unknown"

        # 3. Commit hash — try embedded suffix first (e.g. "0.6.2+git7f3a1bc")
        try:
            import vllm as _vllm
            full_ver = getattr(_vllm, "__version__", "")
            if "+" in full_ver:
                info["vllm_version"] = full_ver
                suffix = full_ver.split("+", 1)[1]
                if suffix.startswith("git"):
                    info["vllm_commit"] = suffix[3:]
        except Exception:
            pass

        # 4. Commit hash — subprocess git fallback
        if "vllm_commit" not in info:
            try:
                import pathlib as _pl
                import subprocess as _sp
                import shutil as _sh
                import vllm as _vllm
                pkg_dir = str(_pl.Path(_vllm.__file__).parent.parent)
                if _sh.which("git"):
                    r = _sp.run(
                        ["git", "-C", pkg_dir, "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if r.returncode == 0:
                        info["vllm_commit"] = r.stdout.strip()
            except Exception:
                pass

        return info

    def extra_mutations(self) -> list:
        """Return vLLM-specific mutation functions."""
        return list(VLLM_MUTATIONS)

    def evaluate_extra(self, result, session, trace=None) -> list:
        """vLLM-specific oracle checks.

        Everything the core ``Oracle.evaluate`` list can decide from a
        ``TraceResult`` alone already runs there.  This hook exists for the
        checks that additionally need the *request* side, and it is the only
        evaluation path the runner hands a ``Trace`` (``runner.py``:
        ``adapter.evaluate_extra(result, session, trace=trace)``).

        ``_check_schema_violation`` is exactly such a check: the declared
        structured-output constraint lives on the SEND ``Event``, not on the
        ``RequestResult``, so the oracle must correlate the two.  Results with
        ``passed=True`` are dropped by the caller.
        """
        if trace is None:
            return []
        oracle = getattr(self, "_extra_oracle", None)
        if oracle is None:
            from ...core.oracles import Oracle as _Oracle
            oracle = self._extra_oracle = _Oracle()
        found = list(oracle._check_schema_violation(result, trace))
        found += oracle._check_reset_honored(result, trace)
        return [f for f in found if not f.passed]

    # ------------------------------------------------------------------
    # Collector hooks
    # ------------------------------------------------------------------

    def make_collector(self, config):
        """Create a KV-event collector if kv_events_endpoint is configured."""
        if config.kv_events_endpoint:
            from ...core.collector import KvEventAdapterCollector
            self._kv_collector = KvEventAdapterCollector(
                endpoint=config.kv_events_endpoint,
            )
            return self._kv_collector
        return None

    def analyze_collection(self, summary) -> list:
        """Convert a KVProbeSummary into CollectionAnomaly objects.

        Runs both the generic analyze_kv_anomalies() pass and the adapter's
        own analyze_kv_summary() hook, then wraps results as CollectionAnomaly.
        """
        if summary is None:
            return []
        from ...kv_events import analyze_kv_anomalies
        raw = analyze_kv_anomalies(summary) + self.analyze_kv_summary(summary)
        return [
            CollectionAnomaly(
                anomaly_type=a["anomaly_type"],
                count=a.get("count", 0),
                score=a["score"],
                detail=a.get("detail", ""),
                metadata=a.get("metadata", {}),
            )
            for a in raw
        ]


# _VllmCollector removed — use KvEventAdapterCollector from fuzzer.core.collector
