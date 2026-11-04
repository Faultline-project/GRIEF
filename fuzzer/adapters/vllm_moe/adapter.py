"""
VllmMoEAdapter: extends VllmBaseAdapter for MoE model serving.

Adds:
  - MoE batch-isolation oracle: output(R in batch) should equal output(R alone)
    at temperature=0 — divergence proves batch-sensitive serving behavior
  - Three MoE-specific mutations targeting expert dispatch density and timing
  - Suspicion queue + confirmation loop (solo vs batch on same server)

No second server is needed: both the batch trace and per-request solo traces
run against the same vLLM instance.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
import uuid

from ...trace import ActionType, Event, Trace
from ...core.types import BugClass, OracleResult
from ..vllm_base.adapter import VllmBaseAdapter
from .mutators import MOE_MUTATIONS
from .oracles import MoECandidate, compare_batch_vs_solo

MAX_QUEUE_SIZE = 50


class VllmMoEAdapter(VllmBaseAdapter):
    """
    TargetAdapter for vLLM serving MoE models.

    Batch isolation oracle: at temperature=0, request R's output must be
    independent of what else is in the batch.  If output(R in batch) diverges
    from output(R alone), that is batch-sensitive serving behavior.

    Parameters
    ----------
    confirm_k : int
        Number of (batch, solo) replay rounds per candidate.
    batch_size : int
        Maximum candidates drained per run_confirmation_batch() call.
    interval_min : float
        Minimum minutes between confirmation runs.
    enqueue_cooldown_s : float
        Per-signature rate limit for enqueue (seconds).
    confirm_cooldown_s : float
        Per-signature cooldown after a candidate is confirmed or exhausted.
    """

    def __init__(
        self,
        confirm_k:          int   = 3,
        batch_size:         int   = 3,
        interval_min:       float = 5.0,
        enqueue_cooldown_s: float = 60.0,
        confirm_cooldown_s: float = 600.0,
    ) -> None:
        super().__init__()
        self._queue:             list[MoECandidate] = []
        self._enqueue_times:     dict[str, float]   = {}
        self._confirm_cooldowns: dict[str, float]   = {}
        self._last_confirm_at:   float              = 0.0
        self._iteration:         int                = 0
        # params
        self.confirm_k          = confirm_k
        self.batch_size         = batch_size
        self.interval_min       = interval_min
        self.enqueue_cooldown_s = enqueue_cooldown_s
        self.confirm_cooldown_s = confirm_cooldown_s

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def extra_mutations(self) -> list:
        return list(super().extra_mutations()) + list(MOE_MUTATIONS)

    def mutation_weights(self) -> dict:
        return {
            # Upweight — expert dispatch stress
            "concurrent_expert_stress": 3.0,
            "jitter_offsets":           2.0,
            "batch_size_step":          2.0,
            "inject_cancel":            1.5,
            # Disable — KV prefix/cache mutations irrelevant for compute-path isolation
            "inject_max_prefix":    0.0,
            "chunk_boundary_align": 0.0,
            # Keep but downweight — fused dispatch behavior changes under memory load
            "memory_pressure":         0.3,
            # Downweight — tangential
            "inject_mixed_endpoints": 0.5,
            "retry_storm":            0.5,
        }

    # ------------------------------------------------------------------
    # Oracle hooks
    # ------------------------------------------------------------------

    def evaluate_extra(self, result, session, trace=None) -> list:
        """Enqueue batch-isolation candidates for deferred confirmation."""
        self._iteration += 1
        if trace is not None and self._is_confirmable(trace):
            sig = self._candidate_sig(trace)
            now = time.time()
            if (now - self._enqueue_times.get(sig, 0) >= self.enqueue_cooldown_s
                    and len(self._queue) < MAX_QUEUE_SIZE):
                self._queue.append(MoECandidate(
                    trace     = trace,
                    iteration = self._iteration,
                    sig       = sig,
                    timestamp = now,
                ))
                self._enqueue_times[sig] = now
        return list(super().evaluate_extra(result, session, trace=trace))

    async def run_confirmation_batch(self, executor, base_url: str) -> list:
        """Drain suspicion queue with solo-vs-batch confirmation runs."""
        if not self._queue:
            return []
        elapsed = time.time() - self._last_confirm_at
        if elapsed < self.interval_min * 60 and len(self._queue) < self.batch_size:
            return []
        return await self._run_confirmation(executor, base_url)

    # ------------------------------------------------------------------
    # Confirmation internals
    # ------------------------------------------------------------------

    async def _run_confirmation(self, executor, base_url: str) -> list:
        # Dedup by sig (keep freshest), take up to batch_size
        seen_sigs: set[str] = set()
        candidates: list[MoECandidate] = []
        for c in sorted(self._queue, key=lambda c: c.timestamp, reverse=True):
            if c.sig not in seen_sigs:
                seen_sigs.add(c.sig)
                candidates.append(c)
            if len(candidates) >= self.batch_size:
                break

        # Remove selected from queue
        selected_sigs = {c.sig for c in candidates}
        self._queue = [c for c in self._queue if c.sig not in selected_sigs]

        findings = []
        now = time.time()
        for candidate in candidates:
            cooldown_until = self._confirm_cooldowns.get(candidate.sig, 0)
            if now < cooldown_until:
                continue

            all_diverged: list[dict] = []
            valid_rounds = 0

            for _ in range(self.confirm_k):
                if not await self.check_liveness(base_url):
                    continue
                await self.flush_state(base_url)

                # Run full batch trace
                batch_result = await executor.run(candidate.trace, base_url)

                # Run each SEND request solo — only requests with prompt_family_id
                # are reproducible (byte-identical prompt across runs), so only
                # those are meaningful for solo-vs-batch comparison.
                sends = [ev for ev in candidate.trace.events
                         if ev.action == ActionType.SEND
                         and ev.prompt_family_id is not None]
                solo_results: dict = {}
                all_solo_ok = True
                for send_ev in sends:
                    if not await self.check_liveness(base_url):
                        all_solo_ok = False
                        break
                    await self.flush_state(base_url)
                    solo_trace = self._make_solo_trace(candidate.trace, send_ev)
                    solo_result = await executor.run(solo_trace, base_url)
                    solo_results[send_ev.request_id] = solo_result

                if not all_solo_ok:
                    continue

                valid_rounds += 1
                round_divs = compare_batch_vs_solo(batch_result, solo_results)
                if round_divs:
                    all_diverged.append(round_divs)  # one entry per round (a list of dicts)

            # Require at least 2 valid rounds before filing
            if valid_rounds < 2:
                self._confirm_cooldowns[candidate.sig] = time.time() + self.confirm_cooldown_s
                continue

            # Count rounds with at least one divergence (not total divergent requests)
            diverged_rounds = len(all_diverged)
            needed = math.ceil(valid_rounds * 2 / 3)

            if diverged_rounds >= needed:
                # Sample one divergence pair from the first diverging round
                sample = all_diverged[0][0]
                finding = OracleResult(
                    passed    = False,
                    bug_class = BugClass.INCONSISTENCY,
                    score     = 7.0,
                    detail    = (
                        f"batch isolation: output diverged solo vs batch "
                        f"in {diverged_rounds}/{valid_rounds} runs"
                    ),
                    probe_metadata = {
                        "sig":              candidate.sig,
                        "iteration":        candidate.iteration,
                        "sample_request_id": sample["request_id"],
                        "sample_batch_output": sample["batch_output"][:100],
                        "sample_solo_output":  sample["solo_output"][:100],
                    },
                )
                findings.append((candidate.trace, finding))

            self._confirm_cooldowns[candidate.sig] = time.time() + self.confirm_cooldown_s

        self._last_confirm_at = time.time()
        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_confirmable(self, trace) -> bool:
        """Trace must have ≥2 SEND events, all greedy, at least one with family_id."""
        sends = [ev for ev in trace.events if ev.action == ActionType.SEND]
        if len(sends) < 2:
            return False
        if not all(
            ev.temperature in (None, 0.0)
            and (ev.n_completions is None or ev.n_completions == 1)
            for ev in sends
        ):
            return False
        if not any(ev.prompt_family_id is not None for ev in sends):
            return False
        return True

    def _make_solo_trace(self, trace: Trace, target_ev: Event) -> Trace:
        """Build a single-request trace containing only target_ev."""
        solo_ev = copy.deepcopy(target_ev)
        return Trace(
            seed   = trace.seed,
            events = [solo_ev],
            tags   = list(trace.tags),
        )

    def _candidate_sig(self, trace) -> str:
        """Stable hash over timing + shape (not request_ids)."""
        parts = []
        for ev in sorted(trace.events, key=lambda e: e.offset_ms):
            if ev.action != ActionType.SEND:
                continue
            parts.append(
                f"{ev.action.value}:{ev.offset_ms}"
                f":pl{ev.prompt_len}:mt{ev.max_tokens}"
                f":st{int(bool(ev.stream))}"
                f":fam{ev.prompt_family_id or ''}"
            )
        key = "|".join(parts)
        return hashlib.md5(key.encode()).hexdigest()[:16]
