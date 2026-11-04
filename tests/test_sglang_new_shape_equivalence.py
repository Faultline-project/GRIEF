"""Behavioral equivalence between the new-shape SGLang adapter and the
legacy ``SglangBaseAdapter``, plus the determinism contract enforced by
``_ExplodingStrategy``.

Phase 3 of the adapter / strategy split refactor.

Two concerns, both expressed as exact assertions (no smoke campaign, no
sampling):

1. ``TestSglangBaseEquivalence`` — byte-identical ``build_payload``
   outputs vs the legacy adapter for both chat and completion events;
   same mutator set, same weight dict.

2. ``TestBuildPayloadDeterminism`` — uses ``_ExplodingStrategy`` whose
   every method raises, then exercises ``build_payload``,
   ``check_liveness``, ``flush_state``, ``evaluate_extra``. If any of
   those starts consulting Strategy in the future, the test fails with
   a clear assertion message. This is the executable form of the v4
   design review's determinism contract.
"""

import asyncio
import pytest

from fuzzer.adapters.types import Capabilities, Strategy
from fuzzer.adapters.sglang.adapter import (
    make_adapter,
    SglangAdapter,
    SglangBaseAdapter,
)
from fuzzer.adapters.sglang.capabilities import SglangBaseCapabilities
from fuzzer.adapters.sglang.mutators import (
    SGLANG_MUTATIONS,
    SGLANG_MUTATION_WEIGHTS,
)
from fuzzer.trace import ActionType, Event


# ---------------------------------------------------------------------------
# Event helpers (kept tiny and deterministic so payload comparisons are exact)
# ---------------------------------------------------------------------------

def _completion_event(request_id="r1"):
    return Event(
        action=ActionType.SEND,
        offset_ms=0,
        request_id=request_id,
        prompt_len=64,
        max_tokens=16,
        stream=True,
        temperature=0.0,
        messages=None,
    )


def _chat_event(request_id="c1"):
    return Event(
        action=ActionType.SEND,
        offset_ms=0,
        request_id=request_id,
        prompt_len=0,
        max_tokens=16,
        stream=True,
        temperature=0.0,
        messages=[
            {"role": "system", "content_len": 8},
            {"role": "user",   "content_len": 8},
        ],
    )


# ---------------------------------------------------------------------------
# 1. Behavioral equivalence with the legacy adapter
# ---------------------------------------------------------------------------

class TestSglangBaseEquivalence:
    """The new-shape adapter must produce the same outputs as the legacy
    adapter for the same inputs. If anything diverges, the new shape is
    not a drop-in replacement and we need to fix it before continuing."""

    def test_build_payload_chat_event_byte_identical_to_legacy(self):
        ev = _chat_event()
        legacy = SglangBaseAdapter()
        new    = make_adapter()
        l_ep, l_payload = legacy.build_payload(ev)
        n_ep, n_payload = new.build_payload(ev)
        assert l_ep      == n_ep      == "/v1/chat/completions"
        assert l_payload == n_payload      # exact dict equality

    def test_build_payload_completion_event_byte_identical_to_legacy(self):
        ev = _completion_event()
        legacy = SglangBaseAdapter()
        new    = make_adapter()
        l_ep, l_payload = legacy.build_payload(ev)
        n_ep, n_payload = new.build_payload(ev)
        assert l_ep      == n_ep      == "/v1/completions"
        assert l_payload == n_payload

    def test_extra_mutations_set_matches_legacy(self):
        """Same mutator functions, identified by ``fn.__name__``."""
        legacy_names = {m.__name__ for m in SglangBaseAdapter().extra_mutations()}
        new_names    = {m.__name__ for m in make_adapter().extra_mutations()}
        assert legacy_names == new_names
        # Cross-check against the source-of-truth mutator list.
        assert legacy_names == {m.__name__ for m in SGLANG_MUTATIONS}

    def test_mutation_weights_match_source_of_truth(self):
        """The new-shape Strategy must carry the same weights the legacy
        adapter exposes through ``mutation_weights()``."""
        legacy_weights = SglangBaseAdapter().mutation_weights()
        new_weights    = make_adapter().mutation_weights()
        assert legacy_weights == new_weights
        # And both equal the source-of-truth dict.
        expected = {fn.__name__: w for fn, w in SGLANG_MUTATION_WEIGHTS.items()}
        assert new_weights == expected

    def test_check_liveness_delegates_to_client_module(self):
        """The new-shape adapter routes ``check_liveness`` through
        ``sglang/client.py``. We verify by patching the client function
        and observing the call — no live HTTP."""
        from fuzzer.adapters.sglang import client as _client

        calls = []

        async def fake_check_liveness(base_url):
            calls.append(base_url)
            return True

        original = _client.check_liveness
        _client.check_liveness = fake_check_liveness
        try:
            result = asyncio.run(make_adapter().check_liveness("http://test"))
        finally:
            _client.check_liveness = original

        assert result is True
        assert calls == ["http://test"]


# ---------------------------------------------------------------------------
# 2. Determinism contract via _ExplodingStrategy
# ---------------------------------------------------------------------------

class _ExplodingStrategy(Strategy):
    """Every Strategy hook raises. Used to prove that ``build_payload`` and
    the other execution-path methods consult ONLY the Event and the
    Capabilities — never Strategy.

    If a future refactor "innocently" reaches for ``self.strategy.X()``
    from an execution path, these tests fail loudly with the assertion
    message identifying the rule that was broken.
    """

    def choose_mutator(self, rng):
        raise AssertionError(
            "Strategy.choose_mutator must NOT be called from execution paths "
            "(build_payload / check_liveness / flush_state / evaluate_extra). "
            "Mutator selection happens in MUTATION code only."
        )

    def choose_endpoint(self, rng):
        raise AssertionError(
            "Strategy.choose_endpoint must NOT be called from execution paths. "
            "Endpoint must be encoded into the Event at mutation time and "
            "read back by Capabilities.endpoint_for_event() during execution."
        )

    def score_multiplier(self, corpus_entry):
        raise AssertionError(
            "Strategy.score_multiplier must NOT be called from build_payload "
            "or other execution-side adapter methods. It belongs to "
            "Corpus.select() only."
        )


def _adapter_with_exploding_strategy():
    """Build a new-shape adapter whose Strategy raises on any access.
    Capabilities is the real SGLang capabilities so routing works."""
    return SglangAdapter(
        capabilities=SglangBaseCapabilities(),
        strategy=_ExplodingStrategy(),
    )


class TestBuildPayloadDeterminism:
    """build_payload() reads only the Event + the (static) Capabilities."""

    def test_chat_event_routes_to_chat_without_consulting_strategy(self):
        a = _adapter_with_exploding_strategy()
        ep, payload = a.build_payload(_chat_event())   # must NOT raise
        assert ep == "/v1/chat/completions"
        assert "messages" in payload

    def test_completion_event_routes_to_completions_without_consulting_strategy(self):
        a = _adapter_with_exploding_strategy()
        ep, payload = a.build_payload(_completion_event())   # must NOT raise
        assert ep == "/v1/completions"
        assert "prompt" in payload

    def test_build_payload_is_pure_for_repeated_calls_on_same_event(self):
        """Same Event → same (endpoint, payload). No hidden state in the
        adapter changes the output."""
        a = _adapter_with_exploding_strategy()
        ev = _completion_event()
        ep1, payload1 = a.build_payload(ev)
        ep2, payload2 = a.build_payload(ev)
        assert ep1 == ep2
        assert payload1 == payload2


class TestOtherRuntimeHooksDeterminism:
    """Lock down the determinism contract for the *other* execution-path
    methods named in the v4 review. None of them should consult Strategy.

    For the HTTP-touching ones (check_liveness, flush_state) we patch
    the client helpers to no-ops so the methods complete without doing
    real I/O; if they had consulted Strategy first, _ExplodingStrategy
    would raise before the patched helper is reached.
    """

    def test_check_liveness_does_not_consult_strategy(self):
        from fuzzer.adapters.sglang import client as _client
        a = _adapter_with_exploding_strategy()

        async def fake_check_liveness(base_url):
            return True

        original = _client.check_liveness
        _client.check_liveness = fake_check_liveness
        try:
            ok = asyncio.run(a.check_liveness("http://x"))
        finally:
            _client.check_liveness = original
        assert ok is True

    def test_flush_state_does_not_consult_strategy(self):
        from fuzzer.adapters.sglang import client as _client
        a = _adapter_with_exploding_strategy()

        async def fake_flush_cache_endpoint(base_url):
            return True   # short-circuit — no fallback path needed

        original = _client.flush_cache_endpoint
        _client.flush_cache_endpoint = fake_flush_cache_endpoint
        try:
            result = asyncio.run(a.flush_state("http://x"))
        finally:
            _client.flush_cache_endpoint = original
        assert result is None

    def test_evaluate_extra_does_not_consult_strategy(self):
        a = _adapter_with_exploding_strategy()
        # The new-shape base returns [] (no extra oracle checks); the
        # invariant we care about is that it doesn't raise from
        # _ExplodingStrategy.
        assert a.evaluate_extra(result=None, session=None, trace=None) == []
