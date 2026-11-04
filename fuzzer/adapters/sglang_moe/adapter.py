"""SglangMoEAdapter: new-shape MoE adapter for SGLang.

Phase 4 of the adapter / strategy split refactor.

Behavior preserved from the legacy adapter:

* Same 3 MoE mutators (``MOE_MUTATIONS``), same mutator-weight dict,
  same suspicion-queue state machine (``_queue``, ``_enqueue_times``,
  ``_confirm_cooldowns``, ``_iteration``, ``_last_confirm_at``).
* ``evaluate_extra`` and ``run_confirmation_batch`` bodies copy
  verbatim — only the umbrella class changes (from
  ``SglangBaseAdapter`` to ``SglangAdapter``).

Behavior added by Phase 4:

* The adapter now exposes ``self.capabilities`` (an
  ``SglangMoeCapabilities``) and ``self.strategy``
  (a ``MoeExplorationStrategy``) — the new-shape umbrella structure.
* ``extra_mutations()`` and ``mutation_weights()`` are inherited from
  ``SglangAdapter``: they read directly off ``self.capabilities`` /
  ``self.strategy`` so the legacy override-the-method pattern is no
  longer needed.

"""

from __future__ import annotations

import copy
import hashlib
import math
import time

from ...trace import ActionType, Event, Trace
from ...core.types import BugClass, OracleResult

from ..sglang.adapter import SglangAdapter

from .capabilities import SglangMoeCapabilities
from .strategies import MoeExplorationStrategy
from .mutators import MOE_MUTATIONS   # re-exported for legacy callers
from .oracles import MoECandidate, compare_batch_vs_solo


# Hard cap on the suspicion queue size so a single very-busy campaign
# can't blow memory.
MAX_QUEUE_SIZE = 50


class SglangMoEAdapter(SglangAdapter):
    """New-shape MoE adapter.

    Extends ``SglangAdapter`` (the new-shape SGLang base) and swaps in
    ``SglangMoeCapabilities`` + ``MoeExplorationStrategy``. The
    runtime-hook methods that drive the batch-isolation oracle
    (``evaluate_extra`` / ``run_confirmation_batch`` /
    ``_run_confirmation``) and their helpers are preserved verbatim
    from the legacy adapter so the 264-line existing test suite passes
    unchanged.

    Batch isolation oracle: at ``temperature=0``, request R's output
    must be independent of what else is in the batch. The suspicion
    queue accumulates candidates, then ``run_confirmation_batch``
    drains them in solo-vs-batch replay rounds.

    Parameters
    ----------
    confirm_k : int
        Number of (batch, solo) replay rounds per candidate.
    batch_size : int
        Maximum candidates drained per ``run_confirmation_batch`` call.
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
        super().__init__(
            capabilities = SglangMoeCapabilities(),
            strategy     = MoeExplorationStrategy(),
        )
        # Suspicion-queue state — preserved verbatim from the legacy
        # adapter. These are per-instance fields (not strategy state)
        # because they hold execution-time runtime data, not policy
        # configuration.
        self._queue:             list[MoECandidate] = []
        self._enqueue_times:     dict[str, float]   = {}
        self._confirm_cooldowns: dict[str, float]   = {}
        self._last_confirm_at:   float              = 0.0
        self._iteration:         int                = 0
        # Tunables — also preserved verbatim.
        self.confirm_k          = confirm_k
        self.batch_size         = batch_size
        self.interval_min       = interval_min
        self.enqueue_cooldown_s = enqueue_cooldown_s
        self.confirm_cooldown_s = confirm_cooldown_s

    # ------------------------------------------------------------------
    # Oracle hooks
    #
    # evaluate_extra() and run_confirmation_batch() are copied verbatim
    # from the legacy adapter. Only the parent class changed; the
    # method bodies did not.
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
    # Confirmation internals (copied verbatim from legacy)
    # ------------------------------------------------------------------

    async def _run_confirmation(self, executor, base_url: str) -> list:
        seen_sigs: set[str] = set()
        candidates: list[MoECandidate] = []
        for c in sorted(self._queue, key=lambda c: c.timestamp, reverse=True):
            if c.sig not in seen_sigs:
                seen_sigs.add(c.sig)
                candidates.append(c)
            if len(candidates) >= self.batch_size:
                break

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

                batch_result = await executor.run(candidate.trace, base_url)

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
                    solo_trace  = self._make_solo_trace(candidate.trace, send_ev)
                    solo_result = await executor.run(solo_trace, base_url)
                    solo_results[send_ev.request_id] = solo_result

                if not all_solo_ok:
                    continue

                valid_rounds += 1
                round_divs = compare_batch_vs_solo(batch_result, solo_results)
                if round_divs:
                    all_diverged.append(round_divs)

            if valid_rounds < 2:
                self._confirm_cooldowns[candidate.sig] = (
                    time.time() + self.confirm_cooldown_s
                )
                continue

            diverged_rounds = len(all_diverged)
            needed = math.ceil(valid_rounds * 2 / 3)

            if diverged_rounds >= needed:
                sample  = all_diverged[0][0]
                finding = OracleResult(
                    passed    = False,
                    bug_class = BugClass.INCONSISTENCY,
                    score     = 7.0,
                    detail    = (
                        f"batch isolation: output diverged solo vs batch "
                        f"in {diverged_rounds}/{valid_rounds} runs"
                    ),
                    probe_metadata = {
                        "sig":               candidate.sig,
                        "iteration":         candidate.iteration,
                        "sample_request_id": sample["request_id"],
                        "sample_batch_output": sample["batch_output"][:100],
                        "sample_solo_output":  sample["solo_output"][:100],
                    },
                )
                findings.append((candidate.trace, finding))

            self._confirm_cooldowns[candidate.sig] = (
                time.time() + self.confirm_cooldown_s
            )

        self._last_confirm_at = time.time()
        return findings

    # ------------------------------------------------------------------
    # Helpers (copied verbatim from legacy)
    # ------------------------------------------------------------------

    def _is_confirmable(self, trace) -> bool:
        """≥2 SEND events, all greedy, at least one with family_id."""
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
        solo_ev = copy.deepcopy(target_ev)
        return Trace(
            seed   = trace.seed,
            events = [solo_ev],
            tags   = list(trace.tags),
        )

    def _candidate_sig(self, trace) -> str:
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
