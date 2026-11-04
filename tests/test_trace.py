"""Tests for fuzzer/trace.py — no GPU or server required."""

import json
import pytest

from fuzzer.trace import ActionType, Event, Trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_send(offset_ms: int = 0, prompt_len: int = 128,
              max_tokens: int = 64, stream: bool = True,
              prefix_len: int = 0, request_id: str = "r1") -> Event:
    return Event(
        action=ActionType.SEND,
        offset_ms=offset_ms,
        request_id=request_id,
        prompt_len=prompt_len,
        max_tokens=max_tokens,
        stream=stream,
        prefix_len=prefix_len,
    )


def make_cancel(offset_ms: int, request_id: str = "r1") -> Event:
    return Event(action=ActionType.CANCEL, offset_ms=offset_ms,
                 request_id=request_id)


def make_wait(duration_ms: int, offset_ms: int = 0) -> Event:
    return Event(action=ActionType.WAIT, offset_ms=offset_ms,
                 duration_ms=duration_ms)


def make_trace(*events: Event, seed: int = 42) -> Trace:
    return Trace(seed=seed, events=list(events))


# ---------------------------------------------------------------------------
# ActionType
# ---------------------------------------------------------------------------

class TestActionType:
    def test_values(self):
        assert ActionType.SEND       == "send"
        assert ActionType.CANCEL     == "cancel"
        assert ActionType.DISCONNECT == "disconnect"
        assert ActionType.WAIT       == "wait"


# ---------------------------------------------------------------------------
# Event validation
# ---------------------------------------------------------------------------

class TestEventValidate:
    def test_valid_send(self):
        make_send().validate()  # should not raise

    def test_send_missing_prompt_len_raises(self):
        ev = Event(action=ActionType.SEND, offset_ms=0, prompt_len=None,
                   max_tokens=64)
        with pytest.raises(AssertionError):
            ev.validate()

    def test_send_zero_prompt_len_raises(self):
        ev = Event(action=ActionType.SEND, offset_ms=0, prompt_len=0,
                   max_tokens=64)
        with pytest.raises(AssertionError):
            ev.validate()

    def test_send_missing_max_tokens_raises(self):
        ev = Event(action=ActionType.SEND, offset_ms=0, prompt_len=64,
                   max_tokens=None)
        with pytest.raises(AssertionError):
            ev.validate()

    def test_valid_wait(self):
        make_wait(duration_ms=100).validate()  # should not raise

    def test_wait_missing_duration_raises(self):
        ev = Event(action=ActionType.WAIT, offset_ms=0, duration_ms=None)
        with pytest.raises(AssertionError):
            ev.validate()

    def test_wait_zero_duration_is_valid(self):
        Event(action=ActionType.WAIT, offset_ms=0, duration_ms=0).validate()

    def test_cancel_needs_no_extra_fields(self):
        make_cancel(offset_ms=100).validate()  # should not raise


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

class TestTraceHelpers:
    def test_send_events_filters_correctly(self):
        t = make_trace(
            make_send(0, request_id="r1"),
            make_cancel(100, "r1"),
            make_send(200, request_id="r2"),
            make_wait(50, 300),
        )
        sends = t.send_events()
        assert len(sends) == 2
        assert all(e.action == ActionType.SEND for e in sends)

    def test_send_events_empty_trace(self):
        assert make_trace().send_events() == []

    def test_duration_ms_returns_max_offset(self):
        t = make_trace(
            make_send(0),
            make_cancel(500, "r1"),
            make_send(250, request_id="r2"),
        )
        assert t.duration_ms() == 500

    def test_duration_ms_empty_trace(self):
        assert make_trace().duration_ms() == 0

    def test_validate_propagates_to_events(self):
        bad_event = Event(action=ActionType.SEND, offset_ms=0,
                          prompt_len=None, max_tokens=64)
        t = make_trace(bad_event)
        with pytest.raises(AssertionError):
            t.validate()

    def test_validate_clean_trace_passes(self):
        t = make_trace(make_send(0), make_cancel(200, "r1"))
        t.validate()


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerialization:
    def _full_trace(self) -> Trace:
        return Trace(
            seed=99,
            tags=["test"],
            events=[
                make_send(0,   request_id="r1", prompt_len=128,
                          max_tokens=64, stream=True,  prefix_len=32),
                make_send(100, request_id="r2", prompt_len=256,
                          max_tokens=128, stream=False, prefix_len=0),
                make_cancel(300, "r1"),
                make_wait(50, 400),
                Event(action=ActionType.DISCONNECT, offset_ms=500,
                      request_id="r2"),
            ],
        )

    def test_json_roundtrip_send_fields(self):
        t  = self._full_trace()
        rt = Trace.from_json(t.to_json())
        for orig, loaded in zip(t.events, rt.events):
            assert orig.offset_ms   == loaded.offset_ms
            assert orig.request_id  == loaded.request_id
            assert orig.prompt_len  == loaded.prompt_len
            assert orig.max_tokens  == loaded.max_tokens
            assert orig.stream      == loaded.stream
            assert orig.prefix_len  == loaded.prefix_len
            assert orig.duration_ms == loaded.duration_ms

    def test_to_json_is_valid_json(self):
        json.loads(self._full_trace().to_json())  # should not raise

    def test_from_dict_missing_request_id_gets_assigned(self):
        d = {
            "seed": 1,
            "events": [{"action": "send", "offset_ms": 0,
                        "prompt_len": 64, "max_tokens": 32}],
        }
        t = Trace.from_dict(d)
        assert t.events[0].request_id  # non-empty string

    def test_empty_trace_roundtrip(self):
        t = Trace(seed=0)
        rt = Trace.from_json(t.to_json())
        assert rt.seed == 0
        assert rt.events == []


# ---------------------------------------------------------------------------
# Unicode stress field on Event + is_deterministic()
# ---------------------------------------------------------------------------

class TestEventUnicodeStress:
    def test_unicode_stress_roundtrip_serialization(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0,
                  request_id="r1", prompt_len=64, max_tokens=32,
                  unicode_stress="cjk"),
        ])
        rt = Trace.from_json(t.to_json())
        assert rt.events[0].unicode_stress == "cjk"

    def test_is_deterministic_all_greedy(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0,
                  request_id="r1", prompt_len=64, max_tokens=32,
                  temperature=0.0, top_k=None, top_p=None),
        ])
        assert t.is_deterministic() is True

    def test_is_deterministic_nonzero_temp_false(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0,
                  request_id="r1", prompt_len=64, max_tokens=32,
                  temperature=0.5),
        ])
        assert t.is_deterministic() is False

    def test_is_deterministic_nonunit_top_k_false(self):
        t = Trace(seed=1, events=[
            Event(action=ActionType.SEND, offset_ms=0,
                  request_id="r1", prompt_len=64, max_tokens=32,
                  top_k=10),
        ])
        assert t.is_deterministic() is False




# ---------------------------------------------------------------------------
# Layer-12 Event fields: cache_salt (W3) + structured output (W2)
# ---------------------------------------------------------------------------

class TestLayer12EventFields:
    """``Trace.from_dict`` enumerates every field explicitly, so a new field is
    silently dropped on corpus reload unless it is added there too.  These
    round-trips are the regression guard for exactly that."""

    _NEW_FIELDS = {
        "cache_salt": "grief-r1-0000beef",
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "t",
                                            "schema": {"type": "object"}}},
        "guided_json": {"type": "object", "required": ["a"]},
        "guided_regex": r"[0-9]{3}",
        "guided_grammar": "root ::= \"a\"",
        "guided_choice": ["yes", "no"],
    }

    def _trace(self) -> Trace:
        return Trace(seed=7, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=32, **self._NEW_FIELDS)])

    def test_full_dict_roundtrip_is_stable(self):
        d = self._trace().to_dict()
        assert Trace.from_dict(d).to_dict() == d

    def test_absent_fields_load_as_none(self):
        t = Trace.from_dict({"seed": 1, "events": [
            {"action": "send", "offset_ms": 0, "prompt_len": 8,
             "max_tokens": 8}]})
        for name in self._NEW_FIELDS:
            assert getattr(t.events[0], name) is None


    def test_every_event_field_survives_both_enumerating_sites(self):
        """The drift guard, generalised.  ``Trace.from_dict`` lists the Event
        fields by hand, so a field added to the dataclass but not there is
        silently dropped on corpus reload.  Rather than naming the fields again
        here (a second list to drift), walk the dataclass."""
        import dataclasses
        structural = {"action": ActionType.SEND, "offset_ms": 11,
                      "request_id": "rid", "stream": False}
        typed = {
            "prompt_len": 64, "max_tokens": 32, "prefix_len": 8,
            "messages": [{"role": "user", "content_len": 4}],
            "temperature": 0.0, "top_p": 0.9, "top_k": 5, "min_tokens": 3,
            "n_completions": 2, "logprobs": 5, "repetition_penalty": 1.1,
            "stop": ["x"], "duration_ms": 12, "unicode_stress": "cjk",
            "adapter_name": "a", "origin": "seed", "prompt_family_id": "pf",
            "prompt_style": "rejection_heavy", "cache_salt": "salt",
            "response_format": {"type": "json_object"},
            "guided_json": {"type": "object"}, "guided_regex": "[0-9]",
            "guided_grammar": 'root ::= "a"', "guided_choice": ["y"],
            "allowed_token_ids": [1, 2], "bad_words": ["{"],
            "logit_bias": {"1": -100.0}, "stop_token_ids": [1],
            "priority": 3, "prompt_logprobs": 2, "seed": 1234,
            "truncate_prompt_tokens": 16,
            "include_usage": True,
            "prompt_embeds": True,
        }
        # a field added to Event but not listed above is not being checked
        assert {f.name for f in dataclasses.fields(Event)} == (
            set(structural) | set(typed))
        ev = Event(**structural, **typed)
        for name, value in typed.items():
            assert getattr(Trace.from_dict(Trace(seed=1, events=[ev]).to_dict())
                           .events[0], name) == value, f"Trace.from_dict: {name}"


class TestMaskRestrictorEventFields:
    """The four extra ``-inf`` writers, carried exactly like ``min_tokens`` and
    ``response_format`` before them."""

    _NEW_FIELDS = {
        "allowed_token_ids": [1, 2, 3],
        "bad_words":         ["{", "\""],
        "logit_bias":        {"1": -100.0, "2": 50.0},
        "stop_token_ids":    [151645, 151643],
    }

    def _trace(self) -> Trace:
        return Trace(seed=8, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=32, **self._NEW_FIELDS)])

    def test_full_dict_roundtrip_is_stable(self):
        d = self._trace().to_dict()
        assert Trace.from_dict(d).to_dict() == d

    def test_values_survive_from_dict(self):
        rt = Trace.from_dict(self._trace().to_dict())
        for name, value in self._NEW_FIELDS.items():
            assert getattr(rt.events[0], name) == value

    def test_absent_fields_load_as_none(self):
        t = Trace.from_dict({"seed": 1, "events": [
            {"action": "send", "offset_ms": 0, "prompt_len": 8,
             "max_tokens": 8}]})
        for name in self._NEW_FIELDS:
            assert getattr(t.events[0], name) is None



class TestMaskRestrictorsOf:
    """``mask_restrictors_of`` is what the oracle reasons over, so what counts
    as a writer has to be exact."""

    def _ev(self, **kw):
        return Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                     prompt_len=32, max_tokens=16, **kw)

    def test_empty_when_nothing_declared(self):
        from fuzzer.core.types import mask_restrictors_of
        assert mask_restrictors_of(self._ev()) == {}

    def test_min_tokens_zero_is_not_a_writer(self):
        from fuzzer.core.types import mask_restrictors_of
        assert mask_restrictors_of(self._ev(min_tokens=0)) == {}
        assert mask_restrictors_of(self._ev(min_tokens=1)) == {"min_tokens": 1}

    def test_empty_collections_are_not_writers(self):
        from fuzzer.core.types import mask_restrictors_of
        assert mask_restrictors_of(
            self._ev(allowed_token_ids=[], bad_words=[], logit_bias={},
                     stop_token_ids=[])) == {}

    @pytest.mark.parametrize("kw,kind", [
        ({"guided_json": {"type": "object"}}, "json_schema"),
        ({"guided_regex": "[0-9]"}, "regex"),
        ({"guided_choice": ["y", "n"]}, "choice"),
        ({"guided_grammar": 'root ::= "a"'}, "grammar"),
        ({"response_format": {"type": "json_object"}}, "json_object"),
        ({"response_format": {"type": "json_schema",
                              "json_schema": {"schema": {}}}}, "json_schema"),
    ])
    def test_structured_output_kinds(self, kw, kind):
        from fuzzer.core.types import mask_restrictors_of
        assert mask_restrictors_of(self._ev(**kw)) == {"structured_output": kind}

    def test_response_format_text_is_not_a_constraint(self):
        from fuzzer.core.types import mask_restrictors_of
        assert mask_restrictors_of(
            self._ev(response_format={"type": "text"})) == {}

    def test_only_the_grammar_precedes_the_logprob_snapshot(self):
        """vLLM takes its reported-logprob snapshot after the grammar bitmask
        but before every logits processor, which is the whole reason the
        sentinel predicate can only see grammar-containing compositions."""
        from fuzzer.core.types import (MASK_RESTRICTORS,
                                       PRE_LOGPROB_RESTRICTORS)
        assert PRE_LOGPROB_RESTRICTORS == {"structured_output"}
        assert PRE_LOGPROB_RESTRICTORS <= set(MASK_RESTRICTORS)


class TestMaskComposeSeeds:
    def test_mask_restrictor_seeds_load(self):
        """The three mask-composition seeds must survive a corpus reload."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parent.parent
        seeds = sorted((repo / "fuzzer" / "data" / "seeds" / "mask_compose").glob("*.json"))
        assert len(seeds) == 3
        for path in seeds:
            t = Trace.from_json(path.read_text())
            t.validate()
            assert t.is_deterministic()
            sends = t.send_events()
            assert sends
            # every SEND asks for logprobs — the sentinel predicate needs the
            # sampled token's own logprob and the top-k beside it
            assert all(e.logprobs for e in sends)
            assert Trace.from_dict(t.to_dict()).to_dict() == t.to_dict()

    def test_mask_restrictor_seeds_declare_two_or_more_writers(self):
        """A seed that engages only one restrictor cannot compose a mask.  At
        least one SEND per file must, and each file keeps a control SEND that
        does not (so a finding is attributable to the composition)."""
        import pathlib
        from fuzzer.core.types import mask_restrictors_of
        repo = pathlib.Path(__file__).resolve().parent.parent
        for path in sorted((repo / "fuzzer" / "data" / "seeds" / "mask_compose").glob("*.json")):
            counts = [len(mask_restrictors_of(e))
                      for e in Trace.from_json(path.read_text()).send_events()]
            assert max(counts) >= 2, path.name
            assert min(counts) <= 1, path.name


class TestSchedulerAndAmplificationEventFields:
    """``priority`` / ``prompt_logprobs`` / ``seed`` / ``truncate_prompt_tokens``
    ride the same opt-in path as every optional field before them."""

    _NEW_FIELDS = {
        "priority":               3,
        "prompt_logprobs":        5,
        "seed":                   20260826,
        "truncate_prompt_tokens": 128,
    }

    def _trace(self) -> Trace:
        return Trace(seed=9, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=32, **self._NEW_FIELDS)])

    def test_full_dict_roundtrip_is_stable(self):
        d = self._trace().to_dict()
        assert Trace.from_dict(d).to_dict() == d

    def test_values_survive_from_dict(self):
        rt = Trace.from_dict(self._trace().to_dict())
        for name, value in self._NEW_FIELDS.items():
            assert getattr(rt.events[0], name) == value

    def test_absent_fields_load_as_none(self):
        t = Trace.from_dict({"seed": 1, "events": [
            {"action": "send", "offset_ms": 0, "prompt_len": 8,
             "max_tokens": 8}]})
        for name in self._NEW_FIELDS:
            assert getattr(t.events[0], name) is None


    def test_event_seed_does_not_collide_with_trace_seed(self):
        """``Trace.seed`` (the mutation RNG seed) and ``Event.seed`` (the
        per-request sampling seed) are different things that share a name at
        two nesting levels of the same JSON document.  ``from_dict`` reads one
        from the outer object and one from each event, and getting that
        backwards would silently make every request in a trace share the
        trace's RNG seed."""
        t = Trace.from_dict({"seed": 42, "events": [
            {"action": "send", "offset_ms": 0, "prompt_len": 8,
             "max_tokens": 8, "seed": 777}]})
        assert t.seed == 42
        assert t.events[0].seed == 777

    def test_priority_zero_survives(self):
        """0 is the DEFAULT priority and also the HIGHEST one -- a falsy value
        that has to reach the wire.  ``from_dict`` uses ``.get()`` (not ``or``)
        precisely so a priority of 0 is not silently turned into None and
        dropped from the request body."""
        t = Trace.from_dict({"seed": 1, "events": [
            {"action": "send", "offset_ms": 0, "prompt_len": 8,
             "max_tokens": 8, "priority": 0, "prompt_logprobs": 0,
             "seed": 0}]})
        ev = t.events[0]
        assert ev.priority == 0 and ev.prompt_logprobs == 0 and ev.seed == 0

    def test_n_is_not_duplicated_as_its_own_field(self):
        """``n_completions`` already serialises to the body's ``n``.  A second
        Event field writing the same wire key would let two attributes of one
        Event disagree about one value, so ``n`` is deliberately absent."""
        import dataclasses
        names = {f.name for f in dataclasses.fields(Event)}
        assert "n_completions" in names
        assert "n" not in names
        assert "best_of" not in names


class TestPriorityInversionSeed:
    def test_seed_loads_and_inverts_priority_against_arrival(self):
        import pathlib
        repo = pathlib.Path(__file__).resolve().parent.parent
        t = Trace.from_json(
            (repo / "fuzzer" / "data" / "seeds" / "priority_inversion.json").read_text())
        t.validate()
        assert Trace.from_dict(t.to_dict()).to_dict() == t.to_dict()

        sends = sorted(t.send_events(), key=lambda e: e.offset_ms)
        assert len(sends) >= 8
        assert all(e.priority is not None for e in sends)

        # The inverted block: priority strictly DECREASES as arrival advances,
        # so the last request to arrive is the first one a correct scheduler
        # must admit.  Under fcfs -- or a broken priority queue -- the order is
        # unchanged, and that difference is the whole point of the seed.
        inverted = [e for e in sends if e.request_id.startswith("p")]
        pri = [e.priority for e in inverted]
        assert pri == sorted(pri, reverse=True), pri
        assert len(set(pri)) == len(pri), "ties cannot force a reordering"

        # ...and a same-priority control block, so a run that reorders the
        # control block too is showing something other than priority.
        control = [e for e in sends if e.request_id.startswith("q")]
        assert control and len({e.priority for e in control}) == 1

    def test_seed_respects_the_protocol_constraints_on_the_amplifiers(self):
        """vLLM rejects ``prompt_logprobs`` on a streaming request outright,
        and rejects ``n > 1`` under greedy sampling.  A seed that trips either
        one produces a 400 instead of a schedule."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parent.parent
        t = Trace.from_json(
            (repo / "fuzzer" / "data" / "seeds" / "priority_inversion.json").read_text())
        amplified = 0
        for e in t.send_events():
            if e.prompt_logprobs is not None:
                amplified += 1
                assert e.prompt_logprobs > 0
                assert e.stream is False, e.request_id
            if e.n_completions and e.n_completions > 1:
                amplified += 1
                assert e.temperature and e.temperature > 0, e.request_id
                assert e.seed is not None, e.request_id
        assert amplified, "the seed exercises no amplification at all"


class TestGuidedSmokeSeeds:
    def test_guided_smoke_seeds_load(self):
        """The three W2 smoke seeds must survive a corpus-style reload."""
        import pathlib
        repo = pathlib.Path(__file__).resolve().parent.parent
        seeds = sorted((repo / "fuzzer" / "data" / "seeds" / "guided_smoke").glob("*.json"))
        assert len(seeds) == 3
        for path in seeds:
            t = Trace.from_json(path.read_text())
            t.validate()
            assert t.is_deterministic()
            sends = t.send_events()
            assert sends and all(e.response_format is not None for e in sends)
            assert Trace.from_dict(t.to_dict()).to_dict() == t.to_dict()
