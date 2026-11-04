"""
SglangSpecAdapter: extends SglangBaseAdapter for speculative decoding serving.

Adds:
  - Server flags via server_flags_extra()
  - Spec-targeting mutations (cancel_during_spec, spec_shared_prefix_race,
    spec_mixed_chunk_race, spec_budget_exhaust)
  - Suspicion queue + paired confirmation against a no-spec SGLang server

Key difference from VllmSpecAdapter:
  - No-extra-weights algorithm is NGRAM (not NEXTN).  NEXTN is an internal SGLang
    alias that maps to EAGLE and requires a draft model checkpoint.
    NGRAM uses --speculative-num-draft-tokens N (not --speculative-num-steps).
    EAGLE uses --speculative-num-steps N + --speculative-draft-model-path.
    Both verified against SGLang 0.5.10.post1 server_args.py.
  - No KV block events available; _has_suspicion_signal relies on empty-completion
    and mutation-shape signals only (no reuse_too_fast).
  - Inherits from SglangBaseAdapter instead of VllmBaseAdapter.
"""

from __future__ import annotations

import hashlib
import math
import time

from ...trace import ActionType
from ...core.types import BugClass, OracleResult
from ..sglang.legacy import SglangBaseAdapter
from .mutators import SPEC_MUTATIONS
from .oracles import SpecCandidate, compare_outputs

MAX_QUEUE_SIZE = 50


class SglangSpecAdapter(SglangBaseAdapter):
    """
    TargetAdapter for SGLang with speculative decoding enabled.

    Parameters
    ----------
    algorithm : str
        Speculative algorithm — ``"NGRAM"`` (n-gram draft, no extra weights,
        default) or ``"EAGLE"`` (requires a draft model checkpoint at
        draft_model_path).  Do not use ``"NEXTN"``; it is an internal SGLang
        alias that maps to EAGLE and requires a draft model.
    num_steps : int
        Draft token budget.  Interpretation depends on algorithm:
          NGRAM  → --speculative-num-draft-tokens (total draft tokens per step)
          EAGLE  → --speculative-num-steps (number of draft rounds)
    draft_model_path : str
        Path to the EAGLE draft model checkpoint.  Required when
        algorithm="EAGLE"; unused for NGRAM.  Emitted as
        --speculative-draft-model-path in server_flags_extra().
    confirm_url : str
        Base URL of a second SGLang server running WITHOUT speculative decoding.
        When empty, no candidates are enqueued and confirmation is disabled.
    confirm_k : int
        Number of paired (spec, no-spec) replay runs per candidate.
    confirm_needed_frac : float
        Fraction of valid runs that must diverge to file a finding.
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
        algorithm:           str   = "NGRAM",
        num_steps:           int   = 5,
        draft_model_path:    str   = "",
        confirm_url:         str   = "",
        confirm_k:           int   = 3,
        confirm_needed_frac: float = 2 / 3,
        batch_size:          int   = 3,
        interval_min:        float = 5.0,
        enqueue_cooldown_s:  float = 60.0,
        confirm_cooldown_s:  float = 600.0,
    ) -> None:
        if algorithm in ("EAGLE", "EAGLE3") and not draft_model_path:
            raise ValueError(
                f"SglangSpecAdapter: algorithm={algorithm!r} requires draft_model_path "
                "to be set (path to the draft model checkpoint)."
            )
        super().__init__()
        self.algorithm           = algorithm
        self.num_steps           = num_steps
        self.draft_model_path    = draft_model_path
        self.confirm_url         = confirm_url
        self.confirm_k           = confirm_k
        self.confirm_needed_frac = confirm_needed_frac
        self.batch_size          = batch_size
        self.interval_min        = interval_min
        self.enqueue_cooldown_s  = enqueue_cooldown_s
        self.confirm_cooldown_s  = confirm_cooldown_s

        self._queue:             list[SpecCandidate] = []
        self._enqueue_times:     dict[str, float]    = {}
        self._confirm_cooldowns: dict[str, float]    = {}
        self._last_confirm_at:   float               = 0.0
        self._iteration:         int                 = 0

    # ------------------------------------------------------------------
    # Server flags
    # ------------------------------------------------------------------

    def server_flags_extra(self) -> str:
        """Return SGLang speculative decoding flags.

        NGRAM:  --speculative-algorithm NGRAM --speculative-num-draft-tokens N
        EAGLE:  --speculative-algorithm EAGLE --speculative-num-steps N
                  [+ --speculative-draft-model-path <path>]

        Flag names verified against SGLang 0.5.10.post1 server_args.py:
          NGRAM uses --speculative-num-draft-tokens (total draft tokens per step).
          EAGLE uses --speculative-num-steps (number of draft rounds).
          NEXTN is an internal alias that maps to EAGLE and requires a draft model
          — do not use NEXTN for the no-extra-weights path.
        """
        if self.algorithm == "NGRAM":
            flags = (
                f"--speculative-algorithm NGRAM "
                f"--speculative-num-draft-tokens {self.num_steps}"
            )
        else:
            flags = (
                f"--speculative-algorithm {self.algorithm} "
                f"--speculative-num-steps {self.num_steps}"
            )
            if self.draft_model_path:
                flags += f" --speculative-draft-model-path {self.draft_model_path}"
        return flags

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def extra_mutations(self) -> list:
        return list(super().extra_mutations()) + list(SPEC_MUTATIONS)

    def mutation_weights(self) -> dict:
        return {
            # Upweight — rollback + cancellation pressure
            "cancel_during_spec":       4.0,
            "inject_cancel":            2.0,
            "inject_retry":             2.0,
            "retry_storm":              1.5,
            "stagger_arrivals":         2.0,
            # Spec-specific state transition stress
            "spec_shared_prefix_race":  2.0,
            "spec_mixed_chunk_race":    3.0,
            "spec_budget_exhaust":      3.0,
            # Downweight — tangential noise
            "inject_max_prefix":        0.5,
            "chunk_boundary_align":     0.5,
            "inject_mixed_endpoints":   0.5,
            "memory_pressure":          0.5,
        }

    # ------------------------------------------------------------------
    # Oracle hooks
    # ------------------------------------------------------------------

    def filter_findings(self, findings: list) -> list:
        from ...core.types import BugClass
        suppressed = {BugClass.LATENCY_SPIKE, BugClass.MEMORY_LEAK, BugClass.TOKEN_RATE_STALL}
        return [f for f in findings if f.bug_class not in suppressed]

    def evaluate_extra(self, result, session, trace=None) -> list:
        """Enqueue confirmable suspicious traces for deferred confirmation.

        Enqueuing only happens when confirm_url is set.  Without a comparison
        server there is nothing to confirm against.

        SGLang has no KV block events, so suspicion is based on empty-completion
        and mutation-shape signals only.
        """
        self._iteration += 1
        if (self.confirm_url
                and trace is not None
                and self._is_confirmable(trace)
                and self._has_suspicion_signal(result, trace)):
            sig = self._candidate_sig(trace)
            now = time.time()
            if (now - self._enqueue_times.get(sig, 0) >= self.enqueue_cooldown_s
                    and len(self._queue) < MAX_QUEUE_SIZE):
                self._queue.append(SpecCandidate(
                    trace     = trace,
                    iteration = self._iteration,
                    sig       = sig,
                    timestamp = now,
                ))
                self._enqueue_times[sig] = now
        return list(super().evaluate_extra(result, session, trace=trace))

    async def run_confirmation_batch(self, executor, base_url: str) -> list:
        """Drain suspicion queue with paired spec vs no-spec confirmation runs."""
        if not self.confirm_url or not self._queue:
            return []
        elapsed = time.time() - self._last_confirm_at
        if elapsed < self.interval_min * 60 and len(self._queue) < self.batch_size:
            return []
        return await self._run_confirmation(executor, base_url, self.confirm_url)

    # ------------------------------------------------------------------
    # Confirmation internals
    # ------------------------------------------------------------------

    async def _run_confirmation(
        self, executor, base_url: str, confirm_url: str
    ) -> list:
        seen_sigs: set[str] = set()
        candidates: list[SpecCandidate] = []
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

            diverged_count = 0
            valid_runs     = 0

            for _ in range(self.confirm_k):
                if not await self.check_liveness(base_url):
                    continue
                await self.flush_state(base_url)
                spec_result = await executor.run(candidate.trace, base_url)

                if not await self.check_liveness(confirm_url):
                    continue
                await self.flush_state(confirm_url)
                nospec_result = await executor.run(candidate.trace, confirm_url)

                valid_runs += 1
                if compare_outputs(spec_result, nospec_result):
                    diverged_count += 1

            if valid_runs < 2:
                self._confirm_cooldowns[candidate.sig] = (
                    time.time() + self.confirm_cooldown_s
                )
                continue

            needed = math.ceil(valid_runs * self.confirm_needed_frac)
            if diverged_count >= needed:
                finding = OracleResult(
                    passed    = False,
                    bug_class = BugClass.INCONSISTENCY,
                    score     = 7.0,
                    detail    = (
                        f"spec vs no-spec: output divergence confirmed "
                        f"{diverged_count}/{valid_runs} runs"
                    ),
                    probe_metadata = {
                        "sig":       candidate.sig,
                        "iteration": candidate.iteration,
                        "algorithm": self.algorithm,
                    },
                )
                findings.append((candidate.trace, finding))

            self._confirm_cooldowns[candidate.sig] = (
                time.time() + self.confirm_cooldown_s
            )

        self._last_confirm_at = time.time()
        return findings

    # ------------------------------------------------------------------
    # Suspicion signal helpers
    # ------------------------------------------------------------------

    def _is_confirmable(self, trace) -> bool:
        """All SEND events must use greedy decoding and single completion."""
        return all(
            ev.temperature in (None, 0.0)
            and (ev.n_completions is None or ev.n_completions == 1)
            for ev in trace.events
            if ev.action == ActionType.SEND
        )

    def _has_suspicion_signal(self, result, trace) -> bool:
        """At least one spec-specific signal must be present.

        SGLang has no KV block events, so reuse_too_fast is not available.
        """
        if self._has_empty_completion(result):
            return True
        if self._has_cancel_mutation(trace):
            return True
        if self._has_prefix_race_mutation(trace):
            return True
        return False

    def _has_empty_completion(self, result) -> bool:
        if result is None:
            return False
        for r in result.requests:
            if r.cancelled_at is not None:
                continue
            if r.completed_at is None:
                continue
            if r.tokens_received == 0 or not r.output_text:
                return True
        return False

    def _has_cancel_mutation(self, trace) -> bool:
        return any(
            ev.action == ActionType.CANCEL and ev.origin == "mutated"
            for ev in trace.events
        )

    def _has_prefix_race_mutation(self, trace) -> bool:
        return any(
            ev.action == ActionType.SEND
            and ev.origin == "mutated"
            and ev.prompt_family_id is not None
            for ev in trace.events
        )

    def _candidate_sig(self, trace) -> str:
        parts = []
        for ev in sorted(trace.events, key=lambda e: e.offset_ms):
            parts.append(
                f"{ev.action.value}:{ev.offset_ms}"
                f":pl{ev.prompt_len}:xl{ev.prefix_len}"
                f":mt{ev.max_tokens}:st{int(bool(ev.stream))}"
            )
            if ev.prompt_family_id:
                parts.append(f"fam:{ev.prompt_family_id}")
        cancel_count = sum(1 for e in trace.events if e.action == ActionType.CANCEL)
        parts.append(f"cancels:{cancel_count}")
        key = "|".join(parts)
        return hashlib.md5(key.encode()).hexdigest()[:16]
