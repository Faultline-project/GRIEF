import argparse
import random

from fuzzer.adapters.vllm_embeds.adapter import VllmEmbedsAdapter, compose_embeds_sampling
from fuzzer.trace import ActionType, Event, Trace


def _trace(n=4, chat_first=False):
    ev = [Event(action=ActionType.SEND, offset_ms=10 * i, request_id=f"r{i}", prompt_len=64 + i)
          for i in range(n)]
    if chat_first:
        ev[0].messages = [{"role": "user", "content": "x"}]
    return Trace(seed=0, events=ev)


def test_adds_an_embeds_request_when_none():
    t = compose_embeds_sampling(_trace(), random.Random(0))
    assert any(e.prompt_embeds for e in t.events)


def test_penalty_lands_on_embeds_requests_only_and_input_untouched():
    src = _trace()
    hits = 0
    for s in range(200):
        t = compose_embeds_sampling(src, random.Random(s))
        for e in t.events:
            if e.repetition_penalty:
                assert e.prompt_embeds
                hits += 1
    assert hits > 50
    assert not any(e.prompt_embeds or e.repetition_penalty for e in src.events)


def test_chat_requests_never_become_embeds():
    for s in range(50):
        t = compose_embeds_sampling(_trace(n=1, chat_first=True), random.Random(s))
        assert not t.events[0].prompt_embeds


def test_nongreedy_gets_a_seed():
    for s in range(100):
        for e in compose_embeds_sampling(_trace(), random.Random(s)).events:
            if e.temperature:
                assert e.seed is not None


def test_adapter_registry_and_weights():
    a = VllmEmbedsAdapter()
    names = [f.__name__ for f in a.extra_mutations()]
    assert "compose_embeds_sampling" in names and "inject_prompt_embeds" in names
    assert a.mutation_weights()["compose_embeds_sampling"] > 1.0


def test_cli_selects_embeds_adapter():
    from fuzzer.backends.vllm import cli
    p = argparse.ArgumentParser()
    cli.register_args(p)
    args = p.parse_args(["--embeds"])
    assert isinstance(cli.build_adapter(args), VllmEmbedsAdapter)
    args = p.parse_args([])
    assert not isinstance(cli.build_adapter(args), VllmEmbedsAdapter)
