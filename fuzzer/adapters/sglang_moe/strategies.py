"""SGLang MoE ``Strategy`` implementations.

Phase 4 of the adapter / strategy split refactor.

``MoeExplorationStrategy`` carries the existing legacy MoE mutation
weights (currently held inline in the legacy adapter's
``mutation_weights()`` method) and intentionally applies **no** corpus
penalty.

The reasoning for not inheriting ``DefaultExplorationStrategy``'s
#24845 penalty:

* The legacy MoE adapter did not consume a ``score_multiplier`` (the
  runner did not pass one for any legacy adapter) — keeping the new
  shape penalty-free preserves byte-level corpus-selection equivalence
  with the legacy.

* The #24845 attractor is a *KV corruption* trigger
  (cancel/retry-cascade). The MoE oracle measures *batch isolation* —
  output(R in batch) == output(R alone) — a different surface. A
  penalty calibrated for one is not obviously useful for the other.

If a future campaign wants the #24845 penalty on MoE as well, swap in
``DefaultExplorationStrategy`` or subclass it; the structure is
designed for that.
"""

from __future__ import annotations

from ..types import Strategy


# Mirror the legacy MoE adapter's mutation_weights() return dict
# verbatim. Same keys, same values — equivalence tests rely on this.
_MOE_MUTATOR_WEIGHTS = {
    # Upweight — RadixAttention amplifies expert dispatch density effects
    "concurrent_expert_stress": 4.0,
    "jitter_offsets":           2.0,
    "batch_size_step":          2.0,
    "inject_cancel":            1.5,
    # Disable — prefix/cache mutations reshape the oracle surface
    "inject_max_prefix":    0.0,
    "chunk_boundary_align": 0.0,
    # Downweight — tangential
    "memory_pressure":        0.3,
    "inject_mixed_endpoints": 0.5,
    "retry_storm":            0.5,
}


class MoeExplorationStrategy(Strategy):
    """MoE-specific weights, no corpus penalty.

    Inherits ``Strategy`` directly (not ``DefaultExplorationStrategy``):
    the default returns ``1.0`` from ``score_multiplier``, which matches
    the legacy MoE behavior of no corpus bias.
    """

    def __init__(self):
        super().__init__(
            mutator_weights       = dict(_MOE_MUTATOR_WEIGHTS),
            endpoint_distribution = {},
            flag_search_space     = {},
            fingerprint_penalties = {},   # explicit: no penalties applied
            oracle_activation     = {},
        )
