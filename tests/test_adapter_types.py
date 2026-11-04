"""Tests for ``fuzzer/adapters/types.py`` — Capability, Strategy, Adapter.

Phase 3 of the adapter / strategy split refactor.

Two concerns:

* ``TestAdapterTypes`` — the dataclasses instantiate with sensible
  defaults; ``Adapter`` is abstract (cannot be instantiated directly
  because ``TargetAdapter`` ABC methods are not implemented); concrete
  subclasses ARE ``TargetAdapter`` instances (the runner contract).
* ``TestCapabilitiesRegistry`` — ``endpoint_for_event()`` is a
  deterministic function of the Event, routing by ``event.messages``
  presence exactly as the legacy adapter does. Exact assertions, no
  sampling.
"""

import pytest

from fuzzer.adapters.types import (
    Capability,
    MutationCapability,
    EndpointCapability,
    FlagCapability,
    OracleCapability,
    Capabilities,
    Strategy,
    Adapter,
)
from fuzzer.core.adapter import TargetAdapter
from fuzzer.trace import ActionType, Event


# ---------------------------------------------------------------------------
# Capability + Strategy + Adapter dataclasses
# ---------------------------------------------------------------------------

class TestAdapterTypes:
    """ABC conformance: the runner consumes TargetAdapter, so the new-shape
    Adapter must satisfy that contract statically and at runtime.

    The nine dataclass-default / "assert what the test just passed in" /
    "base class returns None" tests that used to live here were removed: they
    cannot fail on any behaviour change to GRIEF.
    """

    def test_adapter_is_subclass_of_target_adapter(self):
        """The runner consumes TargetAdapter; new-shape Adapter must
        satisfy that contract at the type level."""
        assert issubclass(Adapter, TargetAdapter)

    def test_adapter_is_abstract_cannot_be_instantiated_directly(self):
        """Adapter inherits TargetAdapter's abstract methods without
        implementing them, so it must be subclassed per surface. This
        prevents misuse (e.g. instantiating a generic Adapter and
        finding RuntimeHooks return None at runtime)."""
        with pytest.raises(TypeError):
            Adapter(capabilities=Capabilities(), strategy=Strategy())

    def test_concrete_adapter_subclass_satisfies_target_adapter_isinstance(self):
        """The interesting runtime check: a concrete subclass is a
        TargetAdapter instance. This is what the runner's adapter slot
        relies on."""
        from fuzzer.adapters.sglang.adapter import make_adapter
        a = make_adapter()
        assert isinstance(a, TargetAdapter)
        assert isinstance(a, Adapter)


# ---------------------------------------------------------------------------
# Capabilities.endpoint_for_event — deterministic routing by Event content
# ---------------------------------------------------------------------------

def _send(messages=None, request_id="r"):
    return Event(
        action=ActionType.SEND,
        offset_ms=0,
        request_id=request_id,
        prompt_len=64,
        max_tokens=16,
        stream=True,
        temperature=0.0,
        messages=messages,
    )


class TestCapabilitiesRegistry:
    """``endpoint_for_event`` reads only the Event — never the Strategy.
    Routing matches the legacy adapter convention
    (``sglang/legacy.py:build_payload``)."""

    def _caps(self):
        return Capabilities(endpoints=[
            EndpointCapability(path="/v1/completions"),
            EndpointCapability(path="/v1/chat/completions"),
        ])

    def test_event_with_messages_routes_to_chat_completions(self):
        ep = self._caps().endpoint_for_event(
            _send(messages=[{"role": "user", "content_len": 8}])
        )
        assert ep.path == "/v1/chat/completions"

    def test_event_without_messages_routes_to_completions(self):
        ep = self._caps().endpoint_for_event(_send(messages=None))
        assert ep.path == "/v1/completions"

    def test_event_with_empty_messages_list_routes_to_completions(self):
        """Empty list is falsy → no chat content → completions endpoint.
        Matches ``bool(event.messages)`` test in
        ``SglangBaseAdapter.build_payload``."""
        ep = self._caps().endpoint_for_event(_send(messages=[]))
        assert ep.path == "/v1/completions"

    def test_missing_chat_endpoint_falls_back_to_completions(self):
        """A surface that only exposes /v1/completions still routes
        sensibly when a chat-style Event somehow appears."""
        caps = Capabilities(endpoints=[EndpointCapability(path="/v1/completions")])
        ep = caps.endpoint_for_event(_send(messages=[{"role": "user", "content_len": 8}]))
        assert ep.path == "/v1/completions"

    def test_no_completions_endpoint_raises(self):
        """A degenerate Capabilities with neither endpoint cannot route."""
        caps = Capabilities(endpoints=[])
        with pytest.raises(ValueError, match="no endpoint"):
            caps.endpoint_for_event(_send(messages=None))
