"""Public ``fuzzer.adapters.sglang.adapter`` module.

Phase 2 of the adapter / strategy split moved the legacy
``SglangBaseAdapter`` class to ``sglang/legacy.py``.  Phase 3 introduces
the new-shape ``SglangAdapter`` (``Capabilities`` + ``Strategy`` umbrella)
alongside it and exposes a ``make_adapter()`` factory.

Both shapes coexist:

* Out-of-tree callers (archived repros, ``grave/__main__.py``, ...) that
  still import ``SglangBaseAdapter`` from this module get the legacy
  class via re-export.
* New callers in this codebase use ``make_adapter()`` to obtain the
  new-shape instance.


"""

from __future__ import annotations

from typing import Optional

# ----- Backwards-compatibility re-exports (from Phase 2) -------------------
# Explicit list (not ``from .legacy import *``) so adding a new private
# symbol to legacy.py does not silently leak it through this public surface.
from .legacy import (
    SglangBaseAdapter,
    _SglangCollector,
    _SglangProbeSummary,
    _analyze_sglang_summary,
    _parse_prometheus,
    _MIN_RELATIVE_GROWTH,
    _MIN_ABSOLUTE_GROWTH,
    _UNICODE_POOLS,
    _interleave_unicode,
    _stable_hash,
    _perm_idx,
    _build_prompt,
    _build_chat_messages,
    _fetch_server_info_raw,
)

# ----- New-shape adapter (Phase 3) ----------------------------------------
from ..types import Adapter, Capabilities, Strategy
from . import client as _client
from .capabilities import SglangBaseCapabilities
from .strategies import DefaultExplorationStrategy


class SglangAdapter(Adapter):
    """New-shape SGLang base adapter.

    Implements ``TargetAdapter`` via the ``Adapter`` umbrella; composes a
    ``Capabilities`` + ``Strategy`` internally; delegates execution-path
    helpers (payload construction, health check, flush, version
    discovery) to the pure functions in ``sglang/client.py`` so the
    legacy class and this class share the same byte-level behavior.

    Per the v4 design review's determinism contract: ``build_payload()``
    and the other RuntimeHooks methods MUST NOT consult ``self.strategy``.
    Endpoint routing is determined entirely by the Event's content,
    matching the existing convention (``event.messages`` present → chat;
    absent → completions).
    """

    def __init__(
        self,
        capabilities: Optional[Capabilities] = None,
        strategy:     Optional[Strategy]     = None,
    ):
        super().__init__(
            capabilities = capabilities or SglangBaseCapabilities(),
            strategy     = strategy     or DefaultExplorationStrategy(),
        )
        self._cached_model_name: str = ""

    # --- RuntimeHooks: payload construction (deterministic from Event) ----

    def build_payload(self, event) -> tuple[str, dict]:
        """Build the (endpoint, payload) tuple for a SEND Event.

        Deterministic from the Event. Reads Capabilities (which is
        static) to discover the endpoint string; does NOT consult
        Strategy. This contract is what the ``_ExplodingStrategy`` test
        verifies.
        """
        endpoint_cap = self.capabilities.endpoint_for_event(event)
        common = {
            "max_tokens":  event.max_tokens or 64,
            "stream":      event.stream,
            "temperature": event.temperature if event.temperature is not None else 0.0,
        }
        if endpoint_cap.path == "/v1/chat/completions":
            payload = {"messages": _client.build_chat_messages(event), **common}
        else:
            payload = {"prompt":   _client.build_prompt(event),        **common}
        return endpoint_cap.path, payload

    # --- RuntimeHooks: server interaction ---------------------------------

    async def check_liveness(self, base_url: str) -> bool:
        return await _client.check_liveness(base_url)

    async def flush_state(self, base_url: str) -> Optional[float]:
        if await _client.flush_cache_endpoint(base_url):
            return None
        await _client.flush_via_large_prompt(base_url, self._cached_model_name)
        return None

    async def resolve_model_name(self, base_url: str, config_model_name: str) -> str:
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
        return await _client.discover_version_chain(base_url)

    # --- RuntimeHooks: mutation / oracle wiring ---------------------------

    def extra_mutations(self) -> list:
        """Expose Capabilities.mutators in the format the existing runner
        consumes (``list[Callable]``). Strategy.choose_mutator() is the
        capabilities-aware alternative; this method preserves the
        existing wiring."""
        return [m.fn for m in self.capabilities.mutators]

    def mutation_weights(self) -> dict:
        """Expose Strategy.mutator_weights in the format the existing
        runner consumes (``dict[str, float]`` keyed by ``fn.__name__``)."""
        return dict(self.strategy.mutator_weights)

    def evaluate_extra(self, result, session, trace=None) -> list:
        return []

    # --- RuntimeHooks: collector + structural analysis --------------------

    def make_collector(self, config):
        # Same logic as legacy; lifted as-is so the new shape mirrors
        # collector wiring exactly.
        if config.kv_events_endpoint:
            from ...core.collector import KvEventAdapterCollector
            return KvEventAdapterCollector(endpoint=config.kv_events_endpoint)
        return _SglangCollector(base_url=config.base_url)

    def analyze_collection(self, summary) -> list:
        # Delegate to the legacy implementation — same code path either way.
        return SglangBaseAdapter.analyze_collection(self, summary)


def make_adapter() -> Adapter:
    """Factory for the new-shape SGLang base adapter.

    Returns an instance that:

    * implements ``TargetAdapter`` (the runner sees it as today's adapter),
    * composes ``SglangBaseCapabilities`` (the 6 existing mutators tagged
      with surface_tags + known_bug_affinity for #24845) and
      ``DefaultExplorationStrategy`` (existing mutator weights + soft
      corpus penalty for the #24845 attractor).
    """
    return SglangAdapter()


__all__ = [
    # Legacy re-exports (Phase 2)
    "SglangBaseAdapter",
    "_SglangCollector",
    "_SglangProbeSummary",
    "_analyze_sglang_summary",
    "_parse_prometheus",
    "_MIN_RELATIVE_GROWTH",
    "_MIN_ABSOLUTE_GROWTH",
    "_UNICODE_POOLS",
    "_interleave_unicode",
    "_stable_hash",
    "_perm_idx",
    "_build_prompt",
    "_build_chat_messages",
    "_fetch_server_info_raw",
    # New shape (Phase 3)
    "SglangAdapter",
    "make_adapter",
]
