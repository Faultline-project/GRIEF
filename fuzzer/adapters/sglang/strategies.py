"""SGLang ``Strategy`` implementations.

Phase 3 of the adapter / strategy split refactor.

``DefaultExplorationStrategy`` is the baseline policy: it preserves the
existing ``SGLANG_MUTATION_WEIGHTS`` for mutator selection and applies a
soft fingerprint penalty to corpus selection so the fuzzer steers away
from already-explored attractor regions (currently #24845, the
cancel/retry-cascade KV corruption).

This is a *search* bias, not a *triage* bias — findings still go through
the same oracle and confirmation paths. Only which traces breed next
changes.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from ..types import Strategy
from .mutators import SGLANG_MUTATION_WEIGHTS
from .capabilities import SGLANG_24845


# Tunable: penalty applied to corpus entries whose trace matches the
# fingerprint of a known-bug family. 0.5 = entries are 2× less likely to
# be selected relative to fingerprint-free entries; the floor in
# ``Corpus.select`` (0.01) still keeps every entry reachable.
_KNOWN_BUG_PENALTY = 0.5


def _has_shared_prefix_family(trace, min_members: int = 2) -> bool:
    """True if at least ``min_members`` SEND events share a ``prompt_family_id``."""
    from ...trace import ActionType   # late import to avoid cycle at module load
    families = Counter()
    for e in trace.events:
        if e.action == ActionType.SEND and e.prompt_family_id:
            families[e.prompt_family_id] += 1
    return any(c >= min_members for c in families.values())


def _has_abort_event(trace) -> bool:
    """True if the trace contains a CANCEL or DISCONNECT event."""
    from ...trace import ActionType
    return any(e.action in (ActionType.CANCEL, ActionType.DISCONNECT)
               for e in trace.events)


def fingerprint_sglang_24845(trace) -> bool:
    """Trigger shape of the cancel/retry-cascade KV corruption family.

    Necessary-but-not-sufficient: traces matching this fingerprint *can*
    trigger #24845; traces failing it do not. The fuzzer uses this to
    bias the corpus away from already-explored attractor shapes; it does
    NOT use it to triage or classify findings.
    """
    return _has_shared_prefix_family(trace, min_members=2) and _has_abort_event(trace)


# Default mutator weights — preserves the existing
# ``SGLANG_MUTATION_WEIGHTS`` byte-exactly at the time of cutover.
_DEFAULT_MUTATOR_WEIGHTS = {fn.__name__: w for fn, w in SGLANG_MUTATION_WEIGHTS.items()}

# Default endpoint distribution — completions is the existing default
# (legacy adapter routes by ``event.messages`` presence). Set to None to
# leave mutator-side endpoint choice unbiased.
_DEFAULT_ENDPOINT_DISTRIBUTION = {"/v1/completions": 1.0}


class DefaultExplorationStrategy(Strategy):
    """Baseline policy: existing mutator weights + #24845 fingerprint bias."""

    def __init__(self):
        super().__init__(
            mutator_weights       = dict(_DEFAULT_MUTATOR_WEIGHTS),
            endpoint_distribution = dict(_DEFAULT_ENDPOINT_DISTRIBUTION),
            flag_search_space     = {},
            fingerprint_penalties = {SGLANG_24845: _KNOWN_BUG_PENALTY},
            oracle_activation     = {},
        )

    def score_multiplier(self, corpus_entry) -> float:
        """Down-weight corpus entries whose trace matches the #24845
        fingerprint by ``fingerprint_penalties[SGLANG_24845]``.

        Called from ``Corpus.select()`` via the score_multiplier hook
        added in Phase 1. The corpus applies the post-multiplication
        floor (0.01) — even an all-penalised corpus stays reachable.

        Implementation note (Phase 3): this method invokes
        ``fingerprint_sglang_24845`` directly rather than consulting
        the ``known_bug_affinity`` field on each ``MutationCapability``.
        The affinity metadata is currently annotation-only. A future
        refinement may build a registry mapping affinity ID → predicate
        so additional known bugs are picked up automatically.
        """
        if corpus_entry is None:
            return 1.0
        trace = getattr(corpus_entry, "trace", None)
        if trace is None:
            return 1.0
        if fingerprint_sglang_24845(trace):
            return self.fingerprint_penalties.get(SGLANG_24845, 1.0)
        return 1.0
