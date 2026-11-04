"""TargetAdapter: contract between generic fuzzing loop and a concrete serving mode."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..trace import Event, Trace
    from .types import OracleResult, TraceResult
    from ..monitor import MonitorSession
    from .collector import AdapterCollector, CollectionAnomaly


class TargetAdapter(ABC):
    """Abstract base class for per-mode serving adapters."""

    @abstractmethod
    def build_payload(self, event) -> tuple[str, dict]:
        """Return (endpoint, payload_dict) for a SEND Event."""
        ...

    @abstractmethod
    async def check_liveness(self, base_url: str) -> bool:
        """Return True if the server is healthy."""
        ...

    @abstractmethod
    async def flush_state(self, base_url: str) -> Optional[float]:
        """Evict reclaimable server state. Returns kv_cache_pct if available."""
        ...

    @abstractmethod
    async def resolve_model_name(self, base_url: str, config_model_name: str) -> str:
        """Return the model name to use in requests."""
        ...

    @abstractmethod
    async def discover_version(self, base_url: str) -> dict:
        """Return version metadata dict."""
        ...

    def server_flags_extra(self) -> str:
        """Extra server flags required by this mode. Appended to config.server_flags."""
        return ""

    def extra_mutations(self) -> list:
        """Mode-specific mutation functions appended to MUTATIONS list. Default: []."""
        return []

    @abstractmethod
    def evaluate_extra(self, result, session, trace=None) -> list:
        """Mode-specific behavioral oracle checks. Return [] if none apply.

        trace is the Trace that produced result (or None if unavailable).
        Structural/collection data is accessed via the adapter's own collector
        (self._kv_collector, etc.) — it is NOT passed as a parameter here.
        """
        ...

    def make_collector(self, config) -> "Optional[AdapterCollector]":
        """Return an AdapterCollector for this adapter, or None if not applicable.

        Called once at Runner init.  The runner stores the returned collector
        and drives its lifecycle (start/stop, begin_probe/end_probe).
        Default: None (no collection).
        """
        return None

    def analyze_collection(self, summary) -> "list[CollectionAnomaly]":
        """Convert a probe summary into structural anomaly signals.

        Called by the runner after each probe when a collector is active.
        Returns CollectionAnomaly objects (distinct from OracleResult).
        Default: [] (no structural analysis).
        """
        return []

    def analyze_kv_summary(self, summary) -> list:
        """Adapter-specific KV anomaly analysis (legacy hook).

        Called internally by VllmBaseAdapter.analyze_collection().
        Subclasses that need custom KV scoring override this.
        Returns raw anomaly dicts matching the analyze_kv_anomalies() schema.
        Default: [] (no adapter-specific KV analysis).
        """
        return []

    async def run_confirmation_batch(self, executor, base_url: str) -> list:
        """Drain adapter-owned suspicion queue with confirmation probes.

        Called periodically by the runner alongside structural repro.
        Returns a list of (Trace, OracleResult) tuples so the runner can
        call _handle_finding() with the correct trace context.

        Default: [] (no adapter-owned queue).
        """
        return []

    def filter_findings(self, findings: list) -> list:
        """Post-filter oracle findings before they reach the finding queue.

        Called by the runner after both oracle.evaluate() and evaluate_extra()
        have run.  Default is a no-op.  Override to suppress bug classes that
        are not meaningful for a specific adapter (e.g. suppress LATENCY_SPIKE
        for a LoRA campaign where that signal is pure noise).
        """
        return findings

    def baseline_probe_event_overrides(self, event) -> "Event":
        """Return a (possibly augmented) copy of a baseline probe Event.

        Called by the runner before each TTFT baseline measurement so that
        adapters can inject mode-specific fields (e.g. adapter_name for LoRA).
        The default is a no-op — returns the event unchanged.
        """
        return event

    def determinism_probe_trace(self, baseline_prompt_len: int):
        """Return Trace for determinism probe, or None to disable."""
        return None

    def cache_hit_probe_trace(self):
        """Return Trace for cache-hit probe, or None to disable."""
        return None
