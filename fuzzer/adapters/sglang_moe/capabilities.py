"""SGLang MoE ``Capabilities``: declarative inventory for the MoE
serving surface.

Phase 4 of the adapter / strategy split refactor.

Intentionally does NOT extend ``SglangBaseCapabilities`` for mutators.
The base RadixAttention mutators (``eviction_race``, ``lock_ref_stress``,
etc.) stamp ``prompt_family_id`` which triggers false MoE confirmations
— the legacy adapter dropped them via an ``extra_mutations`` override
that returned ``list(MOE_MUTATIONS)`` only. The new shape preserves that
choice by listing the 3 MoE-specific mutators directly here, with no
inheritance of the base mutator list.

Endpoints match the base adapter (OpenAI-compatible /v1/completions and
/v1/chat/completions).
"""

from __future__ import annotations

from ..types import (
    Capabilities,
    MutationCapability,
    EndpointCapability,
)
from .mutators import (
    concurrent_expert_stress,
    jitter_offsets,
    batch_size_step,
)


class SglangMoeCapabilities(Capabilities):
    """MoE-only mutators + the standard OpenAI-compatible endpoints."""

    def __init__(self):
        super().__init__(
            mutators=[
                MutationCapability(
                    fn=concurrent_expert_stress,
                    name="concurrent_expert_stress",
                    surface_tags={"moe", "expert_dispatch", "concurrent"},
                    # Annotation-only: not driving the strategy today.
                    known_bug_affinity=[],
                ),
                MutationCapability(
                    fn=jitter_offsets,
                    name="jitter_offsets",
                    surface_tags={"moe", "timing"},
                    known_bug_affinity=[],
                ),
                MutationCapability(
                    fn=batch_size_step,
                    name="batch_size_step",
                    surface_tags={"moe", "batch"},
                    known_bug_affinity=[],
                ),
            ],
            endpoints=[
                EndpointCapability(path="/v1/completions"),
                EndpointCapability(path="/v1/chat/completions"),
            ],
            flags=[],
            oracles=[],   # batch-isolation oracle is the suspicion-queue
                          # loop on the adapter itself (runtime hook),
                          # not an OracleCapability the strategy
                          # activates per request.
        )
