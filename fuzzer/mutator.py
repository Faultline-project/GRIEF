"""
Mutator: derives new Traces from existing ones by applying targeted mutations.

Each mutation function takes a Trace + RNG and returns a new Trace.
The top-level mutate() picks one strategy at random.

This module is a thin shim: generic mutations live in fuzzer.core.mutator,
vLLM-specific mutations live in fuzzer.adapters.vllm_base.mutators.
"""

from __future__ import annotations

import random
from typing import Optional

from .trace import Trace

# Import all generic mutations and helpers from core
from .core.mutator import (
    MutationFn,
    MAX_TOKENS_CAP,
    MAX_BURST_EXTRA,
    jitter_timing,
    scale_prompt_lengths,
    scale_max_tokens,
    inject_cancel,
    inject_disconnect,
    inject_retry,
    change_prefix_overlap,
    add_burst,
    toggle_stream,
    remove_random_event,
    splice,
    directed_splice,
    MUTATIONS as _CORE_MUTATIONS,
    mutate as _core_mutate,
)

# Import vLLM-specific mutations from the adapter
from .adapters.vllm_base.mutators import (
    cuda_graph_thrash,
    mutate_sampling_params,
    inject_chat_turn,
    inject_unicode_stress,
    inject_multi_system_message,
    inject_rtl_system_message,
    retry_storm,
    inject_mixed_endpoints,
    inject_stop_sequences,
    inject_max_prefix,
    VLLM_MUTATIONS,
)


# ---------------------------------------------------------------------------
# Combined mutation registry
# ---------------------------------------------------------------------------

MUTATIONS: list[MutationFn] = list(_CORE_MUTATIONS) + list(VLLM_MUTATIONS)

__all__ = [
    # generic mutations
    "MutationFn", "MAX_TOKENS_CAP", "MAX_BURST_EXTRA",
    "jitter_timing", "scale_prompt_lengths", "scale_max_tokens",
    "inject_cancel", "inject_disconnect", "inject_retry",
    "change_prefix_overlap", "add_burst", "toggle_stream",
    "remove_random_event",
    "splice", "directed_splice",
    # vLLM-specific mutations
    "cuda_graph_thrash", "mutate_sampling_params", "inject_chat_turn",
    "inject_unicode_stress", "inject_multi_system_message",
    "inject_rtl_system_message", "retry_storm", "inject_mixed_endpoints",
    "inject_stop_sequences", "inject_max_prefix",
    "VLLM_MUTATIONS",
    # combined
    "MUTATIONS",
    "mutate",
]


def mutate(trace: Trace, rng: random.Random, n: int = 1,
           other: Optional[Trace] = None,
           other_pressure: float = 0.0,
           mutations: Optional[list] = None,
           weights: Optional[list] = None) -> "tuple[Trace, list[str]]":
    """Apply n randomly chosen mutation strategies sequentially.

    If `other` is provided, splice probability scales with other_pressure:
      - base 20%, up to 40% at pressure >= 10
    When other_pressure is high (other trace stressed the scheduler heavily),
    directed_splice is used instead of random splice to maximise the chance
    of building KV state then immediately evicting it.

    If `mutations` is provided, it overrides the default MUTATIONS list.

    Returns (mutated_trace, list_of_operator_names_applied).
    """
    _mutations = mutations if mutations is not None else MUTATIONS
    return _core_mutate(trace, rng, n=n, other=other,
                        other_pressure=other_pressure, mutations=_mutations,
                        weights=weights)
