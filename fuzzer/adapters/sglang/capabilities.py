"""SGLang base ``Capabilities``: declarative inventory of what the
SGLang OpenAI-compatible serving surface supports.

Phase 3 of the adapter / strategy split refactor.

``surface_tags`` and ``known_bug_affinity`` on each ``MutationCapability``
record design-review evidence: which mutators stress what, and which
mutators are known attractors for filed bug families.

Important Phase-3 caveat: ``known_bug_affinity`` is **annotation-only**
today.  ``DefaultExplorationStrategy`` (``strategies.py``) applies
fingerprint penalties by invoking predicate functions directly
(``fingerprint_sglang_24845``), not by consulting the affinity list on
each ``MutationCapability``.  The annotation is here so future
strategies, telemetry, and documentation can pick it up without us
needing to thread it through retroactively.
"""

from __future__ import annotations

from ..types import (
    Capabilities,
    MutationCapability,
    EndpointCapability,
)
from .mutators import (
    prefix_boundary_align,
    prefix_fork_stress,
    eviction_race,
    lock_ref_stress,
    partial_prefix_mismatch,
    concurrent_same_prefix,
)


# Known-bug IDs used in ``known_bug_affinity`` annotations.
# These correspond to entries in the (informal) known-bugs registry the
# proposal sketches; for now they are just string tokens that strategies
# can pattern-match against.
SGLANG_24845 = "sglang_24845"   # cancel/retry-cascade KV corruption


class SglangBaseCapabilities(Capabilities):
    """The 6 existing RadixAttention-targeting mutators plus the two
    OpenAI-compatible endpoints SGLang serves. Surface-specific
    subclasses (MoE, Spec, LoRA) extend this with their own
    capabilities.
    """

    def __init__(self):
        super().__init__(
            mutators = [
                MutationCapability(
                    fn=prefix_boundary_align,
                    name="prefix_boundary_align",
                    surface_tags={"radix_cache", "boundary"},
                    known_bug_affinity=[],
                ),
                MutationCapability(
                    fn=prefix_fork_stress,
                    name="prefix_fork_stress",
                    surface_tags={"radix_cache", "concurrent"},
                    known_bug_affinity=[],
                ),
                MutationCapability(
                    fn=eviction_race,
                    name="eviction_race",
                    surface_tags={"radix_cache", "eviction"},
                    known_bug_affinity=[SGLANG_24845],
                ),
                MutationCapability(
                    fn=lock_ref_stress,
                    name="lock_ref_stress",
                    # lock_ref_stress is the cancel/retry-cascade
                    # mutator — exactly the trigger shape of #24845.
                    surface_tags={"abort", "retry", "ref_count"},
                    known_bug_affinity=[SGLANG_24845],
                ),
                MutationCapability(
                    fn=partial_prefix_mismatch,
                    name="partial_prefix_mismatch",
                    surface_tags={"radix_cache", "partial_match"},
                    known_bug_affinity=[],
                ),
                MutationCapability(
                    fn=concurrent_same_prefix,
                    name="concurrent_same_prefix",
                    surface_tags={"radix_cache", "concurrent"},
                    known_bug_affinity=[],
                ),
            ],
            endpoints = [
                EndpointCapability(path="/v1/completions"),
                EndpointCapability(path="/v1/chat/completions"),
            ],
            flags   = [],   # populated by surface-specific subclasses
            oracles = [],   # populated by surface-specific subclasses
        )
