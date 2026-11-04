"""
VllmSpecAdapter: extends VllmBaseAdapter for speculative decoding serving.

Adds:
  - --speculative-config JSON flag via server_flags_extra()
  - Spec-targeting mutations (cancel_during_spec, spec_shared_prefix_race)
  - Suspicion queue + paired confirmation against a no-spec server

Per-request spec toggle (skip_speculative_decoding, speculative_num_tokens)
does not exist in the vLLM OpenAI-compatible API (verified Step 0).  Detection
relies on existing behavioral oracles and the generic reuse_too_fast KV
anomaly.  Confirmation compares outputs from a spec server vs a no-spec server.
"""

from __future__ import annotations

import hashlib
import json
import math
import time

from ...trace import ActionType
from ...core.types import BugClass, OracleResult
from ..vllm_base.adapter import VllmBaseAdapter
from .mutators import SPEC_MUTATIONS
from .oracles import SpecCandidate, compare_outputs

MAX_QUEUE_SIZE = 50


class VllmSpecAdapter(VllmBaseAdapter):
    """
    TargetAdapter for vLLM with speculative decoding enabled.

    Parameters
    ----------
    draft_model : str
        ``"[ngram]"`` uses n-gram draft — no extra model weights required,
        isolates serving-layer correctness.  ``"[ngram_gpu]"`` is the
        GPU-accelerated variant of n-gram (fully vectorized tensor ops,
        async D2H — same config shape, different code path).
        ``"[extract_hidden_states]"`` targets the extract_hidden_states
        spec method which extracts hidden states for KV transfer.
        ``"[eagle3]"`` uses the Eagle3 speculative method with an
        external eagle3 draft model (``eagle3_model`` parameter).
        Otherwise pass a HuggingFace model ID to use a real draft model.
    default_k : int
        Number of speculative tokens (``num_speculative_tokens``).
    eagle3_model : str
        HuggingFace model ID for the Eagle3 draft model.  Only used when
        ``draft_model == "[eagle3]"``.
    eagle3_k : int
        Number of speculative tokens for Eagle3 (default 5).
    eh_num_layers : int
        Number of aux hidden state layers to extract when
        ``draft_model == "[extract_hidden_states]"``.  More layers widen the
        race window; 35 is what reliably crashes vLLM 0.19.0 Qwen3-8B.
    confirm_url : str
        Base URL of a second vLLM server running WITHOUT speculative decoding.
        When empty (default), no candidates are enqueued and confirmation is
        entirely disabled — zero overhead.
    confirm_k : int
        Number of paired (spec, no-spec) replay runs per candidate.
    confirm_needed_frac : float
        Fraction of valid runs that must diverge to file a finding.
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
        draft_model:         str   = "[ngram]",
        default_k:           int   = 12,
        confirm_url:         str   = "",
        confirm_k:           int   = 3,
        confirm_needed_frac: float = 2 / 3,
        batch_size:          int   = 3,
        interval_min:        float = 5.0,
        enqueue_cooldown_s:  float = 60.0,
        confirm_cooldown_s:  float = 600.0,
        eh_num_layers:       int   = 35,
        eagle3_model:        str   = "AngelSlim/Qwen3-8B_eagle3",
        eagle3_k:            int   = 5,
    ) -> None:
        super().__init__()
        self.draft_model          = draft_model
        self.default_k            = default_k
        self.eagle3_model         = eagle3_model
        self.eagle3_k             = eagle3_k
        self.confirm_url          = confirm_url
        self.confirm_k            = confirm_k
        self.confirm_needed_frac  = confirm_needed_frac
        self.batch_size           = batch_size
        self.interval_min         = interval_min
        self.enqueue_cooldown_s   = enqueue_cooldown_s
        self.confirm_cooldown_s   = confirm_cooldown_s
        self.eh_num_layers        = eh_num_layers

        self._queue:             list[SpecCandidate] = []
        self._enqueue_times:     dict[str, float]    = {}
        self._confirm_cooldowns: dict[str, float]    = {}
        self._last_confirm_at:   float               = 0.0
        self._iteration:         int                 = 0

    # ------------------------------------------------------------------
    # Server configuration
    # ------------------------------------------------------------------

    def server_flags_extra(self) -> str:
        """Return the --speculative-config flag for recording in findings.

        --enable-prefix-caching and --enable-chunked-prefill are already
        injected by the vLLM job script (hardcoded and via EXTRA_SERVER_FLAGS
        respectively), so omitting them here prevents double-recording in
        the stored server_flags string and duplicate-key warnings on repro.

        The JSON is compact (no spaces) so it round-trips safely through
        whitespace-split flag parsing in reproduction scripts.
        """
        # Single-quote the JSON so shlex.split() in reproduction scripts treats it
        # as one token and preserves the inner double quotes.
        spec_json = json.dumps(self._build_spec_config(), separators=(',', ':'))
        return f"--speculative-config '{spec_json}'"

    def _build_spec_config(self) -> dict:
        if self.draft_model in ("[ngram]", "[ngram_gpu]"):
            method = "ngram_gpu" if self.draft_model == "[ngram_gpu]" else "ngram"
            return {
                "method":                 method,
                "num_speculative_tokens": self.default_k,
                "prompt_lookup_max":      self.default_k,
                "prompt_lookup_min":      2,
            }
        if self.draft_model == "[eagle3]":
            return {
                "model":                  self.eagle3_model,
                "method":                 "eagle3",
                "num_speculative_tokens": self.eagle3_k,
            }
        if self.draft_model == "[extract_hidden_states]":
            layer_ids = list(range(1, self.eh_num_layers + 1))
            return {
                "method":                 "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": layer_ids,
                    },
                },
            }
        return {
            "model":                  self.draft_model,
            "num_speculative_tokens": self.default_k,
        }

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def extra_mutations(self) -> list:
        return list(super().extra_mutations()) + list(SPEC_MUTATIONS)

    def mutation_weights(self) -> dict:
        # For extract_hidden_states the bug is a race between NEW requests
        # joining a decode batch and scatter overwriting `-1` placeholders.
        # We need high concurrency with arriving requests — NOT shared prefix.
        # Upweight add_burst (concurrent new SENDs, no prefix sharing) and
        # stagger_arrivals (interleaves prefill with in-flight decodes).
        # spec_shared_prefix_race stays suppressed: its prefix-sharing aspect
        # is orthogonal to this bug and just adds KV-corruption noise.
        is_eh = self.draft_model == "[extract_hidden_states]"
        return {
            # Upweight — lifecycle/rollback pressure
            "cancel_during_spec":    4.0,
            "inject_cancel":         2.0,
            "inject_retry":          2.0,
            "retry_storm":           1.5,
            "stagger_arrivals":      5.0 if is_eh else 2.0,
            "add_burst":             5.0 if is_eh else 1.0,
            # Block/chunk boundary — exercises KV block_table edge cases
            "kv_block_boundary":       2.0,
            "chunk_boundary_align":    1.5,
            # Constraint conflict — spec decode + early stop
            "min_tokens_conflict":     1.5,
            # Memory pressure — forces eviction/preemption under spec
            "memory_pressure":         2.0,
            # Suppress — incompatible with no-APC spec discovery
            "inject_max_prefix":       0.0,
            "spec_shared_prefix_race": 0.0,
            # Downweight — tangential noise
            "inject_mixed_endpoints":  0.5,
        }

    # ------------------------------------------------------------------
    # Oracle hooks
    # ------------------------------------------------------------------

    def evaluate_extra(self, result, session, trace=None) -> list:
        """Enqueue confirmable suspicious traces; detection deferred to confirmation.

        Enqueuing only happens when confirm_url is set.  Without a comparison
        server there is nothing to confirm against, so candidates are never
        enqueued and the queue stays empty.

        KV probe summary is accessed via self._kv_collector.last_summary (set by
        make_collector) rather than passed as a parameter — the collector is
        adapter-owned, so the adapter reads it directly.
        """
        self._iteration += 1
        kv_summary = getattr(getattr(self, "_kv_collector", None), "last_summary", None)
        if (self.confirm_url
                and trace is not None
                and self._is_confirmable(trace)
                and self._has_suspicion_signal(result, kv_summary, trace)):
            sig = self._candidate_sig(trace)
            now = time.time()
            if (now - self._enqueue_times.get(sig, 0) >= self.enqueue_cooldown_s
                    and len(self._queue) < MAX_QUEUE_SIZE):
                self._queue.append(SpecCandidate(
                    trace      = trace,
                    iteration  = self._iteration,
                    sig        = sig,
                    timestamp  = now,
                    kv_anomaly = self._has_reuse_too_fast(kv_summary),
                ))
                self._enqueue_times[sig] = now
        return list(super().evaluate_extra(result, session, trace=trace))

    def analyze_kv_summary(self, summary) -> list:
        """No adapter-specific KV anomaly types for spec.

        The generic reuse_too_fast from analyze_kv_anomalies() is the correct
        signal for spec rollback (very short remove→store gap during speculation).
        """
        return []

    async def run_confirmation_batch(self, executor, base_url: str) -> list:
        """Drain suspicion queue with paired spec vs no-spec confirmation runs.

        Returns an empty list when confirm_url is not set — the queue will
        also be empty in that case because evaluate_extra never enqueues
        without a comparison server.
        """
        if not self.confirm_url or not self._queue:
            return []
        elapsed = time.time() - self._last_confirm_at
        if elapsed < self.interval_min * 60 and len(self._queue) < self.batch_size:
            return []
        return await self._run_confirmation(executor, base_url, self.confirm_url)

    # ------------------------------------------------------------------
    # Confirmation internals
    # ------------------------------------------------------------------

    async def _run_confirmation(self, executor, base_url: str, confirm_url: str) -> list:
        # Dedup by sig (keep freshest), take up to batch_size
        seen_sigs: set[str] = set()
        candidates: list[SpecCandidate] = []
        for c in sorted(self._queue, key=lambda c: c.timestamp, reverse=True):
            if c.sig not in seen_sigs:
                seen_sigs.add(c.sig)
                candidates.append(c)
            if len(candidates) >= self.batch_size:
                break

        # Remove selected candidates from queue
        selected_sigs = {c.sig for c in candidates}
        self._queue = [c for c in self._queue if c.sig not in selected_sigs]

        findings = []
        now = time.time()
        for candidate in candidates:
            # Skip if still in confirm cooldown
            cooldown_until = self._confirm_cooldowns.get(candidate.sig, 0)
            if now < cooldown_until:
                continue

            diverged_count = 0
            valid_runs     = 0

            for _ in range(self.confirm_k):
                # Gate: spec server
                if not await self.check_liveness(base_url):
                    continue
                await self.flush_state(base_url)
                spec_result = await executor.run(candidate.trace, base_url)

                # Gate: comparison server (always the no-spec server — run_extra
                # returns early at the `not self.confirm_url` guard, so this is
                # unreachable without one)
                if not await self.check_liveness(confirm_url):
                    continue
                await self.flush_state(confirm_url)
                nospec_result = await executor.run(candidate.trace, confirm_url)

                valid_runs += 1
                if compare_outputs(spec_result, nospec_result):
                    diverged_count += 1

            # Fix 1: require at least 2 valid pairs before confirming.
            # ceil(1 * 2/3) == 1, so a single surviving run would otherwise
            # be enough — too weak when infrastructure is flaky.
            if valid_runs < 2:
                self._confirm_cooldowns[candidate.sig] = time.time() + self.confirm_cooldown_s
                continue

            needed = math.ceil(valid_runs * self.confirm_needed_frac)
            if diverged_count >= needed:
                bug_class = BugClass.KV_CORRUPTION if candidate.kv_anomaly else BugClass.INCONSISTENCY
                score     = 8.0 if candidate.kv_anomaly else 7.0
                mode   = "spec vs no-spec"
                finding   = OracleResult(
                    passed    = False,
                    bug_class = bug_class,
                    score     = score,
                    detail    = (
                        f"{mode}: output divergence confirmed "
                        f"{diverged_count}/{valid_runs} runs"
                    ),
                    probe_metadata = {
                        "sig":        candidate.sig,
                        "iteration":  candidate.iteration,
                        "kv_anomaly": candidate.kv_anomaly,
                    },
                )
                findings.append((candidate.trace, finding))

            # Fix 3: set confirm cooldown on all exits (confirmed or exhausted),
            # not just on findings — suppresses re-enqueue of known-noisy sigs.
            self._confirm_cooldowns[candidate.sig] = time.time() + self.confirm_cooldown_s

        self._last_confirm_at = time.time()
        return findings

    # ------------------------------------------------------------------
    # Suspicion signal helpers
    # ------------------------------------------------------------------

    def _is_confirmable(self, trace) -> bool:
        """All SEND events must use greedy decoding and single completion.

        n_completions > 1 causes vLLM to force diversity even at temperature=0,
        so replays are inherently non-deterministic — not confirmable.
        """
        return all(
            ev.temperature in (None, 0.0)
            and (ev.n_completions is None or ev.n_completions == 1)
            for ev in trace.events
            if ev.action == ActionType.SEND
        )

    def _has_suspicion_signal(self, result, kv_summary, trace) -> bool:
        """At least one spec-specific signal must be present."""
        if self._has_reuse_too_fast(kv_summary):
            return True
        if self._has_empty_completion(result):
            return True
        if self._has_cancel_mutation(trace):
            return True
        if self._has_prefix_race_mutation(trace):
            return True
        return False

    def _has_reuse_too_fast(self, kv_summary) -> bool:
        return kv_summary is not None and kv_summary.reuse_too_fast_count > 0

    def _has_empty_completion(self, result) -> bool:
        """Any non-cancelled, completed request returned no output."""
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
        """Trace contains a CANCEL event injected by cancel_during_spec."""
        return any(
            ev.action == ActionType.CANCEL and ev.origin == "mutated"
            for ev in trace.events
        )

    def _has_prefix_race_mutation(self, trace) -> bool:
        """Trace contains a SEND event injected by spec_shared_prefix_race."""
        return any(
            ev.action == ActionType.SEND
            and ev.origin == "mutated"
            and ev.prompt_family_id is not None
            for ev in trace.events
        )

    def _candidate_sig(self, trace) -> str:
        """Stable hash over timing + shape (not request_ids).

        Fix 2: include prompt_len, prefix_len, max_tokens, stream so that
        traces with identical timing but different shapes (e.g. a 512-token
        prefix cancel vs a 64-token prefix cancel at the same offset) get
        distinct signatures and don't share an enqueue cooldown.
        """
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
