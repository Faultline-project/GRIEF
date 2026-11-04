"""
AdapterCollector: interface between the core runner and adapter-owned collection.

The runner knows nothing about KV blocks, ZMQ, or vLLM internals.
It only calls begin_probe/end_probe per trace and start/stop per campaign.
Adapters provide a concrete collector (or return None from make_collector).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CollectionAnomaly:
    """Structural suspicion payload returned by AdapterCollector analysis.

    Distinct from OracleResult — preserves the "suspicion vs confirmed bug"
    boundary.  Structural anomalies are enqueued for confirmation; only after
    confirmation does the runner file an OracleResult finding.

    Fields match the schema previously used in the kv_anomalies dict list so
    that downstream code can swap `a["score"]` for `a.score` without other
    changes.
    """
    anomaly_type: str
    count:        int
    score:        float
    detail:       str
    metadata:     dict = field(default_factory=dict)


class AdapterCollector(ABC):
    """Campaign-scoped + probe-scoped data collector owned by an adapter.

    Lifecycle
    ---------
    Campaign level (one per fuzzing run):
        await collector.start()   # called once in Runner.run() before the loop
        ...
        await collector.stop()    # called once in the finally block

    Probe level (one per fuzzing iteration / repro run):
        await collector.begin_probe()   # called just before executor.run()
        result = await executor.run(...)
        summary = await collector.end_probe()  # returns summary or None
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the background collection loop (campaign-scoped)."""
        ...

    @abstractmethod
    async def stop(self, final: bool = False) -> None:
        """Stop the background collection loop (campaign-scoped).

        ``final=True`` marks the last stop of the process, letting an
        implementation skip teardown that only matters if the process keeps
        running (e.g. reclaiming a ZMQ context).
        """
        ...

    @abstractmethod
    async def begin_probe(self) -> None:
        """Mark the start of a probe window (per-trace)."""
        ...

    @abstractmethod
    async def end_probe(self) -> Optional[object]:
        """Close the probe window and return a summary (or None if disabled).

        The returned object type is adapter-defined (e.g. KVProbeSummary for
        vLLM).  The runner passes it opaquely to adapter.analyze_collection().
        """
        ...

    def forensics_snapshot(self, window_s: float = 30.0) -> Optional[object]:
        """Return a forensics blob for the last `window_s` seconds.

        Called when a finding is being saved.  Default: None (no forensics).
        """
        return None

    def get_block_snapshot(self, start_t: float, end_t: float) -> Optional[object]:
        """Return a per-block content snapshot for the given time window.

        Used by structural repro to detect cross-run content conflicts.
        Default: None.
        """
        return None

    def summarize_window(self, start_t: float, end_t: float) -> Optional[object]:
        """Return a probe summary for an arbitrary [start_t, end_t] window.

        Used by adapter-internal probes (e.g. isolation probe) that have their
        own timing independent of the main loop's begin_probe/end_probe cycle.
        Default: None.
        """
        return None

    @property
    def last_summary(self) -> Optional[object]:
        """The summary produced by the most recent end_probe() call.

        Exposed as a property so feature adapters (e.g. vllm_spec) can read
        the KV state for the current trace without reaching into internals.
        Default: None.
        """
        return None


class KvEventAdapterCollector(AdapterCollector):
    """Shared AdapterCollector wrapping KVEventCollector.

    Used by both vLLM and SGLang adapters — identical lifecycle and probe
    semantics regardless of backend.
    """

    def __init__(self, endpoint: str) -> None:
        import time as _time
        self._time = _time
        from ..kv_events import KVEventCollector
        self._inner = KVEventCollector(endpoint=endpoint)
        self._last_summary = None
        self._probe_start = 0.0

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self, final: bool = False) -> None:
        await self._inner.stop(final=final)

    async def begin_probe(self) -> None:
        self._probe_start = self._time.time()

    async def end_probe(self):
        summary = self._inner.summarize_probe(self._probe_start, self._time.time())
        self._last_summary = summary
        return summary

    def forensics_snapshot(self, window_s: float = 30.0):
        return self._inner.snapshot_forensics(window_s)

    def get_block_snapshot(self, start_t: float, end_t: float):
        return self._inner.get_block_snapshot(start_t, end_t)

    def summarize_window(self, start_t: float, end_t: float):
        return self._inner.summarize_probe(start_t, end_t)

    @property
    def last_summary(self):
        return self._last_summary
