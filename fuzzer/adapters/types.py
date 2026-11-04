"""Adapter abstractions for the new-shape adapters.

Phase 3 of the adapter / strategy split refactor.

Three responsibilities, one umbrella:

* ``Capabilities`` — declarative inventory of what a serving surface
  supports (mutators, endpoints, flags, oracles). Per-surface, static
  for the lifetime of a campaign. Never mutated.
* ``Strategy`` — mutable policy over Capabilities. Holds mutator weights,
  endpoint distribution, fingerprint penalties, oracle activation.
  Provides ``choose_mutator()`` / ``choose_endpoint()`` for mutation-time
  decisions and ``score_multiplier()`` for corpus selection bias.
* ``Adapter(TargetAdapter)`` — abstract umbrella. Subclasses implement
  the existing ``TargetAdapter`` ABC methods (RuntimeHooks) using their
  surface-specific client helpers, and compose a ``Capabilities`` +
  ``Strategy`` pair internally.

The runner consumes ``TargetAdapter``, so the new shape is
interchangeable with legacy adapters at the runner boundary.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.adapter import TargetAdapter


# ---------------------------------------------------------------------------
# Capability primitives
# ---------------------------------------------------------------------------

@dataclass
class Capability:
    """Base class for every capability entry. Empty by design — concrete
    Capability kinds carry their own fields."""


@dataclass
class MutationCapability(Capability):
    """A mutation function plus metadata.

    Attributes:
        fn:                The mutator. Signature ``(Trace, random.Random) -> Trace``.
        name:              Stable identifier for weight lookup and logging.
        surface_tags:      Free-form tags describing what the mutation
                           stresses (e.g. ``{"radix_cache", "abort"}``).
                           Strategies use these to bias selection.
        known_bug_affinity:IDs of filed bugs this mutator tends to
                           trigger (e.g. ``["sglang_24845"]``).
                           **Annotation-only in Phase 3.** Strategies do
                           not consult this field today; the
                           ``DefaultExplorationStrategy`` applies
                           fingerprint penalties by invoking predicate
                           functions directly (see
                           ``fingerprint_sglang_24845`` in
                           ``sglang/strategies.py``). The field is
                           recorded here for future strategies,
                           telemetry, and documentation; a planned
                           later refinement may build a registry
                           mapping affinity ID → predicate so strategies
                           pick up new known bugs automatically.
    """
    fn:                 Callable
    name:               str
    surface_tags:       set        = field(default_factory=set)
    known_bug_affinity: list       = field(default_factory=list)


@dataclass
class EndpointCapability(Capability):
    """An HTTP endpoint the serving surface accepts.

    Phase 3 records only the path; payload schema is captured by the
    adapter's ``build_payload()`` routing (which reads only the Event,
    never the Strategy — see the determinism contract).
    """
    path: str


@dataclass
class FlagCapability(Capability):
    """A server flag plus its valid values.

    ``values`` is a tuple of strings for enum-style flags
    (e.g. ``("", "on")`` for boolean flags) or a 2-tuple of numbers for
    ranges. Used by meta-mutation that varies flag combinations.
    """
    name:   str
    values: tuple = ()


@dataclass
class OracleCapability(Capability):
    """An oracle the strategy can choose to run.

    Phase 3 treats the oracle signature as opaque; Phase 5+ may formalise
    a Protocol once we know what additional oracles structured-gen,
    multi-modal, etc. need.
    """
    fn:                Callable
    name:              str
    cost_estimate_sec: float = 0.0


# ---------------------------------------------------------------------------
# Capabilities registry
# ---------------------------------------------------------------------------

@dataclass
class Capabilities:
    """Per-surface registry. Bundles the four Capability kinds.

    Surface-specific subclasses (``SglangBaseCapabilities``,
    ``SglangMoeCapabilities``, ...) populate the lists in ``__init__``
    or via class-level attributes overridden by ``__post_init__``.
    """
    mutators:  list  = field(default_factory=list)
    endpoints: list  = field(default_factory=list)
    flags:     list  = field(default_factory=list)
    oracles:   list  = field(default_factory=list)

    def endpoint_for_event(self, event) -> EndpointCapability:
        """Deterministic endpoint routing — reads only the Event.

        Mirrors the existing ``SglangBaseAdapter.build_payload()``
        convention: ``event.messages`` present → ``/v1/chat/completions``;
        absent → ``/v1/completions``.

        Called from ``build_payload()`` during execution; MUST be a pure
        function of the Event. Endpoint *choice* (when a mutator
        constructs a new SEND Event) happens via
        ``Strategy.choose_endpoint()`` and the choice is encoded into the
        Event itself — this method just reads that encoding back out.
        """
        prefer_chat = bool(getattr(event, "messages", None))
        want_path = "/v1/chat/completions" if prefer_chat else "/v1/completions"
        for ep in self.endpoints:
            if ep.path == want_path:
                return ep
        # Fallback to /v1/completions if chat wasn't declared (unusual but safe)
        for ep in self.endpoints:
            if ep.path == "/v1/completions":
                return ep
        raise ValueError(
            f"Capabilities has no endpoint for event "
            f"(wanted {want_path!r}, registered: {[e.path for e in self.endpoints]})"
        )


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    """Mutable policy over Capabilities. Default implementation provides
    safe no-op behavior; surface-specific strategies override.

    Mutator/endpoint/oracle/flag-related decisions live here.
    Determinism contract: methods on Strategy are called from MUTATION
    code (when constructing/transforming Events), NEVER from EXECUTION
    code (``build_payload``, ``check_liveness``, ...). The
    ``score_multiplier`` hook IS called from execution-adjacent code
    (Corpus selection) but does not affect Event content or payload.
    """
    mutator_weights:       dict = field(default_factory=dict)
    endpoint_distribution: dict = field(default_factory=dict)
    flag_search_space:     dict = field(default_factory=dict)
    fingerprint_penalties: dict = field(default_factory=dict)
    oracle_activation:     dict = field(default_factory=dict)

    def choose_mutator(self, rng: random.Random) -> Optional[MutationCapability]:
        """Mutator selection from MUTATION code. Default returns None;
        the runner falls back to its existing mutator-selection path."""
        return None

    def choose_endpoint(self, rng: random.Random) -> Optional[EndpointCapability]:
        """Endpoint selection from MUTATION code only. The returned
        capability's identity must be encoded into the constructed
        Event so that downstream ``build_payload()`` routes
        deterministically without consulting Strategy."""
        return None

    def score_multiplier(self, corpus_entry) -> float:
        """Multiplicative bias applied by ``Corpus.select()``. Default 1.0
        (no bias). Surface-specific strategies override to down-weight
        traces matching known-bug fingerprints, etc."""
        return 1.0


# ---------------------------------------------------------------------------
# Adapter umbrella
# ---------------------------------------------------------------------------

class Adapter(TargetAdapter):
    """New-shape adapter umbrella. Abstract — concrete subclasses
    implement the ``TargetAdapter`` ABC methods using their surface's
    client helpers, and compose a ``Capabilities`` + ``Strategy`` pair
    via ``__init__``.

    The runner consumes ``TargetAdapter`` exactly as today; new-shape
    adapters are interchangeable with legacy ones at that boundary.
    """

    def __init__(self, capabilities: Capabilities, strategy: Strategy):
        self.capabilities: Capabilities = capabilities
        self.strategy:     Strategy     = strategy
