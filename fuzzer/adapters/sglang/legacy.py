"""
SglangBaseAdapter: concrete TargetAdapter for SGLang serving.

Implements all abstract methods from TargetAdapter using SGLang-specific
logic. SGLang exposes the same OpenAI-compatible endpoints as vLLM
(/v1/completions, /v1/chat/completions, /health, /v1/models), so most
of the runtime behaviour is identical to VllmBaseAdapter.

Key differences from VllmBaseAdapter:
- discover_version() uses `import sglang` instead of `import vllm`
- extra_mutations() returns [] (no SGLang-specific mutations yet)
- evaluate_extra() returns [] (no SGLang-specific oracle checks yet)
"""

from __future__ import annotations

import dataclasses
import time
from typing import Optional

from ...core.adapter import TargetAdapter
from ...core.collector import AdapterCollector, CollectionAnomaly
from . import client as _client


# ---------------------------------------------------------------------------
# Backwards-compatibility aliases (Phase 2 of adapter/strategy split)
#
# The payload-construction helpers and HTTP primitives moved into
# `sglang/client.py`. Existing code (`test_sglang_collector.py`, etc.)
# imports the underscored names from this module — keep them as thin
# aliases so the move is invisible to callers.
# ---------------------------------------------------------------------------

_UNICODE_POOLS         = _client.UNICODE_POOLS
_interleave_unicode    = _client.interleave_unicode
_stable_hash           = _client.stable_hash
_perm_idx              = _client.perm_idx
_build_prompt          = _client.build_prompt
_build_chat_messages   = _client.build_chat_messages
_fetch_server_info_raw = _client.fetch_server_info_raw


# Helpers above are now sourced from sglang/client.py — see the
# backwards-compatibility aliases.


class SglangBaseAdapter(TargetAdapter):
    """Adapter for SGLang OpenAI-compatible serving."""

    def __init__(self):
        self._cached_model_name: str = ""

    def build_payload(self, event) -> tuple[str, dict]:
        """Return (endpoint, payload_dict) for a SEND Event."""
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
        return await _client.check_liveness(base_url)

    async def flush_state(self, base_url: str) -> Optional[float]:
        """Flush RadixAttention cache, preferring /flush_cache endpoint.

        Tries the native /flush_cache first. On failure, falls back to a
        large unique-prompt POST that evicts the LRU cache. The fallback
        threads ``self._cached_model_name`` so requests carry the right
        ``model`` field on servers that require it.
        """
        if await _client.flush_cache_endpoint(base_url):
            return None
        await _client.flush_via_large_prompt(base_url, self._cached_model_name)
        return None

    async def resolve_model_name(self, base_url: str, config_model_name: str) -> str:
        """Return the model name to use in requests.

        Caches the result on the instance so we only hit /v1/models once
        per campaign. Falls back to "" on discovery failure.
        """
        if config_model_name:
            self._cached_model_name = config_model_name
            return config_model_name
        if self._cached_model_name:
            return self._cached_model_name
        name = await _client.fetch_model_name(base_url)
        if name is None:
            return ""
        self._cached_model_name = name
        return name

    async def discover_version(self, base_url: str) -> dict:
        """Return version metadata dict for SGLang."""
        return await _client.discover_version_chain(base_url)

    def extra_mutations(self) -> list:
        """Return SGLang RadixAttention-specific mutations."""
        from .mutators import SGLANG_MUTATIONS
        return SGLANG_MUTATIONS

    def mutation_weights(self) -> dict:
        """Return name→weight map for SGLang-specific mutations.

        Runner looks up weights by fn.__name__ in this dict.
        """
        from .mutators import SGLANG_MUTATION_WEIGHTS
        return {fn.__name__: w for fn, w in SGLANG_MUTATION_WEIGHTS.items()}

    def evaluate_extra(self, result, session, trace=None) -> list:
        """No SGLang-specific oracle checks yet."""
        return []

    # ------------------------------------------------------------------
    # Collector hooks
    # ------------------------------------------------------------------

    def make_collector(self, config):
        """Return a KV-event collector if endpoint is configured, else _SglangCollector."""
        if config.kv_events_endpoint:
            from ...core.collector import KvEventAdapterCollector
            return KvEventAdapterCollector(endpoint=config.kv_events_endpoint)
        return _SglangCollector(base_url=config.base_url)

    def analyze_collection(self, summary) -> list:
        """Analyze a probe summary, dispatching by summary type.

        KV-event mode: use analyze_kv_anomalies (same as vLLM adapter).
        Fallback mode: use _analyze_sglang_summary (server_info heuristics).
        """
        if summary is None:
            return []
        from ...kv_events import KVProbeSummary, analyze_kv_anomalies
        if isinstance(summary, KVProbeSummary):
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
        return _analyze_sglang_summary(summary)


# ---------------------------------------------------------------------------
# Probe summary
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _SglangProbeSummary:
    """Raw pre/post snapshots from a single probe window."""
    pre_info:     Optional[dict]   # raw /server_info JSON at begin_probe
    post_info:    Optional[dict]   # raw /server_info JSON at end_probe
    pre_metrics:  Optional[dict]   # {metric_name: float} at begin_probe, or None
    post_metrics: Optional[dict]   # {metric_name: float} at end_probe, or None

    def to_dict(self) -> dict:
        return {
            "pre_info":     self.pre_info,
            "post_info":    self.post_info,
            "pre_metrics":  self.pre_metrics,
            "post_metrics": self.post_metrics,
        }


# ---------------------------------------------------------------------------
# Anomaly analysis (Stage 1)
# ---------------------------------------------------------------------------

# Fields whose names match these substrings suggest utilization counters.
_UTIL_KEYWORDS  = ("token", "mem", "usage", "cache")
# Fields that match these substrings are capacity constants, not counters.
_CAPACITY_WORDS = ("max", "limit")

# Thresholds: both must be exceeded to suppress noise from near-zero fields.
_MIN_RELATIVE_GROWTH = 0.05   # 5% relative increase
_MIN_ABSOLUTE_GROWTH = 16     # at least 16 tokens / bytes / units


def _analyze_sglang_summary(summary: _SglangProbeSummary) -> list:
    """Return at most one memory_growth CollectionAnomaly.

    Only fires when a utilization-named numeric field in server_info grew
    clearly and directionally across the probe window (relative > 5% AND
    absolute > 16). Single anomaly max so the finding JSON stays uncluttered.
    """
    pre  = summary.pre_info  or {}
    post = summary.post_info or {}
    if not pre or not post:
        return []

    best_field:  Optional[str]   = None
    best_delta:  float           = 0.0
    best_pct:    float           = 0.0

    for key in pre:
        if key not in post:
            continue
        # Only numeric fields
        try:
            v_pre  = float(pre[key])
            v_post = float(post[key])
        except (TypeError, ValueError):
            continue
        # Skip capacity constants
        key_lower = key.lower()
        if any(w in key_lower for w in _CAPACITY_WORDS):
            continue
        # Only utilization-flavoured names
        if not any(w in key_lower for w in _UTIL_KEYWORDS):
            continue
        delta = v_post - v_pre
        if delta <= 0:
            continue
        # Both thresholds must be met
        if v_pre > 0 and (delta / v_pre) < _MIN_RELATIVE_GROWTH:
            continue
        if delta < _MIN_ABSOLUTE_GROWTH:
            continue
        # Keep the single largest delta as representative evidence
        if delta > best_delta:
            best_delta = delta
            best_field = key
            best_pct   = (delta / v_pre * 100) if v_pre > 0 else float("inf")

    if best_field is None:
        return []

    return [CollectionAnomaly(
        anomaly_type = "memory_growth",
        count        = 1,
        score        = 1.5,
        detail       = (
            f"server_info[{best_field!r}] grew by {best_delta:.0f} "
            f"({best_pct:.1f}%) across probe window"
        ),
        metadata     = {
            "field":     best_field,
            "pre_value": pre[best_field],
            "post_value": post[best_field],
            "delta":     best_delta,
        },
    )]


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class _SglangCollector(AdapterCollector):
    """Lightweight probe-aligned collector for SGLang.

    Snapshots /server_info (and /metrics if available) at begin_probe /
    end_probe.  Results are annotations only — no structural repro triggers
    until calibration data is available (score 1.5 < 5.0 threshold).
    """

    def __init__(self, base_url: str) -> None:
        self._base_url        = base_url
        self._metrics_ok:  Optional[bool] = None   # None = not yet probed
        self._pre_info:    Optional[dict] = None
        self._pre_metrics: Optional[dict] = None
        self._last_summary: Optional[_SglangProbeSummary] = None

    # -- Campaign lifecycle ------------------------------------------------

    async def start(self) -> None:
        """Probe /metrics availability once at campaign start."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._base_url}/metrics", timeout=3.0
                )
                self._metrics_ok = resp.status_code == 200
        except Exception:
            self._metrics_ok = False
        status = "available" if self._metrics_ok else "not available"
        print(f"[sglang collector] /metrics: {status}")

    async def stop(self, final: bool = False) -> None:
        pass

    # -- Probe lifecycle ---------------------------------------------------

    async def begin_probe(self) -> None:
        self._pre_info    = await self._fetch_server_info()
        self._pre_metrics = await self._fetch_metrics() if self._metrics_ok else None

    async def end_probe(self) -> Optional[_SglangProbeSummary]:
        post_info    = await self._fetch_server_info()
        post_metrics = await self._fetch_metrics() if self._metrics_ok else None
        summary = _SglangProbeSummary(
            pre_info     = self._pre_info,
            post_info    = post_info,
            pre_metrics  = self._pre_metrics,
            post_metrics = post_metrics,
        )
        self._last_summary = summary
        return summary

    # -- Forensics ---------------------------------------------------------

    def forensics_snapshot(self, window_s: float = 30.0) -> Optional[dict]:
        """Return last probe summary as a plain dict for embedding in findings."""
        if self._last_summary is None:
            return None
        return self._last_summary.to_dict()

    @property
    def last_summary(self) -> Optional[_SglangProbeSummary]:
        return self._last_summary

    # -- Helpers -----------------------------------------------------------

    async def _fetch_server_info(self) -> Optional[dict]:
        """Delegate to the shared helper so collector and version-discovery
        always probe the same endpoint on any SGLang deployment."""
        return await _fetch_server_info_raw(self._base_url)

    async def _fetch_metrics(self) -> Optional[dict]:
        """Parse Prometheus text format into {metric_name: float}.

        Only captures the most recent sample for each metric name.
        Comment lines and empty lines are skipped.
        """
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._base_url}/metrics", timeout=3.0
                )
                if resp.status_code != 200:
                    return None
            return _parse_prometheus(resp.text)
        except Exception:
            return None


def _parse_prometheus(text: str) -> dict:
    """Minimal Prometheus text-format parser: metric_name → float.

    Handles simple gauge/counter lines of the form:
        metric_name{labels} value [timestamp]
        metric_name value [timestamp]
    Ignores # HELP / # TYPE comment lines and blank lines.
    When the same metric name appears multiple times (e.g. labelled series),
    the last value wins — sufficient for our coarse snapshot comparison.
    """
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip label block {…} if present
        if "{" in line:
            name_part = line[: line.index("{")]
            rest      = line[line.index("}") + 1 :].strip()
        else:
            parts     = line.split(None, 1)
            name_part = parts[0]
            rest      = parts[1] if len(parts) > 1 else ""
        # First token of rest is the value; second (optional) is timestamp
        value_str = rest.split()[0] if rest.split() else ""
        try:
            result[name_part.strip()] = float(value_str)
        except (ValueError, IndexError):
            pass
    return result


    # _SglangKvCollector removed — use KvEventAdapterCollector from fuzzer.core.collector
