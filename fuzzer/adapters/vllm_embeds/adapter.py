"""
VllmEmbedsAdapter: extends VllmBaseAdapter for servers that accept
``prompt_embeds`` (``--enable-prompt-embeds``).

Like the LoRA / spec / MoE adapters, it steers the operator mix toward its
surface: the embeddings surface is exercised together with the rest of the
request (sampling parameters, log-probs, parallel completions), instead of
relying on two independent operators landing on the same request.

Adds:
  - ``compose_embeds_sampling``: sampling parameters on embedding requests
  - weights that favour the embeddings and sampling operators
"""

from __future__ import annotations

import copy
import random

from ...trace import ActionType, Trace
from ..vllm_base.adapter import VllmBaseAdapter
from ..vllm_base.mutators import VLLM_MUTATIONS


def compose_embeds_sampling(trace: Trace, rng: random.Random) -> Trace:
    """Put sampling parameters on 1-2 embedding requests.

    A trace without an embedding request first gets one (a completions SEND
    turned into ``prompt_embeds``).  Each chosen request then draws the same
    parameter mix as ``mutate_sampling_params``: temperature, repetition
    penalty, log-probs and parallel completions.
    """
    t = copy.deepcopy(trace)
    sends = [ev for ev in t.events
             if ev.action == ActionType.SEND and not ev.messages and ev.prompt_len]
    if not sends:
        return t
    emb = [ev for ev in sends if ev.prompt_embeds]
    if not emb:
        ev = rng.choice(sends)
        ev.prompt_embeds = True
        emb = [ev]
    for ev in rng.sample(emb, min(2, len(emb))):
        choice = rng.random()
        ev.temperature = 0.0 if choice < 0.5 else round(rng.uniform(0.01, 1.0), 2)
        if rng.random() < 0.6:
            ev.repetition_penalty = round(rng.uniform(1.05, 1.5), 2)
        if rng.random() < 0.3:
            ev.logprobs = rng.choice([1, 5, 20])
        if rng.random() < 0.2:
            ev.n_completions = rng.choice([2, 4])
        if ev.temperature and ev.seed is None:
            ev.seed = rng.randrange(1, 2 ** 31)
    return t


class VllmEmbedsAdapter(VllmBaseAdapter):
    """TargetAdapter for vLLM with the prompt-embeddings surface enabled."""

    def extra_mutations(self) -> list:
        return list(VLLM_MUTATIONS) + [compose_embeds_sampling]

    def mutation_weights(self) -> dict:
        return {
            "compose_embeds_sampling": 4.0,
            "inject_prompt_embeds":    2.0,
            "mutate_sampling_params":  2.0,
        }
