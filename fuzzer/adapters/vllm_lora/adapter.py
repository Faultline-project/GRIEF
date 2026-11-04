"""
VllmLoRAAdapter: extends VllmBaseAdapter for concurrent LoRA adapter serving.

Adds:
  - --enable-lora --lora-modules flags via server_flags_extra()
  - Per-request adapter selection via event.adapter_name → payload["model"]
  - LoRA-specific mutations (switch_adapter, concurrent_adapter_burst, ...)
  - Cross-adapter isolation oracle
"""

from __future__ import annotations

from ..vllm_base.adapter import VllmBaseAdapter
from .mutators import LORA_MUTATIONS


class VllmLoRAAdapter(VllmBaseAdapter):
    """
    TargetAdapter for vLLM with --enable-lora.

    Parameters
    ----------
    adapters : dict[str, str]
        Mapping of adapter name → path on disk.
        e.g. {"lora_a": "/tmp/lora_weights/lora_a",
               "lora_b": "/tmp/lora_weights/lora_b"}
    """

    def __init__(self, adapters: dict[str, str]) -> None:
        super().__init__()
        if len(adapters) < 2:
            raise ValueError("VllmLoRAAdapter requires at least 2 adapters "
                             "for cross-adapter isolation testing.")
        self.adapters = adapters

    # ------------------------------------------------------------------
    # Server flags
    # ------------------------------------------------------------------

    def server_flags_extra(self) -> str:
        modules = " ".join(f"{name}={path}"
                           for name, path in self.adapters.items())
        return f"--enable-lora --lora-modules {modules}"

    # ------------------------------------------------------------------
    # Payload construction — inject adapter name as model parameter
    # ------------------------------------------------------------------

    def build_payload(self, event) -> tuple[str, dict]:
        endpoint, payload = super().build_payload(event)
        if event.adapter_name is not None:
            payload["model"] = event.adapter_name
        # Only inject logprobs when explicitly requested on the event.
        # Blanket logprobs on every request changes scheduling cost and
        # reshapes the bug surface — use probe-only traces for that signal.
        return endpoint, payload

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def extra_mutations(self) -> list:
        from .mutators import make_inject_lora_adapters
        return (list(super().extra_mutations())
                + list(LORA_MUTATIONS)
                + [make_inject_lora_adapters(list(self.adapters.keys()))])

    # ------------------------------------------------------------------
    # Oracles
    # ------------------------------------------------------------------

    def evaluate_extra(self, result, session, trace=None) -> list:
        from .oracles import check_cross_adapter_isolation
        return list(super().evaluate_extra(result, session, trace=trace)) \
               + check_cross_adapter_isolation(result)

    def analyze_kv_summary(self, summary) -> list:
        """LoRA-specific KV anomaly scoring: cross-adapter block reuse."""
        from .oracles import analyze_cross_lora_kv
        return analyze_cross_lora_kv(summary)

    # ------------------------------------------------------------------
    # Structured isolation probe (solo-vs-concurrent invariant)
    # ------------------------------------------------------------------

    async def adapter_isolation_probe(
        self,
        executor,
        base_url: str,
        prompt_len: int,
        prefix_len: int,
        k: int = 3,
        logprobs: int = 2,
    ) -> list:
        """
        Structured solo-vs-concurrent isolation probe.

        For each adapter pair (A, B):
          1. Run A alone k times to establish modal(solo_A_outputs)
          2. Run A+B concurrently in one trace
          3. Gate: concurrent_A != modal_solo_A
          4. Attribution: sim(concurrent_A, B) - sim(concurrent_A, modal_A) > 0.25
          5. Score: 3 (no logprobs) / 7 (gate+attribution) / 8 (+pos) / 10 (+margin)

        Parameters
        ----------
        executor : PythonExecutor
            Passed explicitly to avoid circular references.
        base_url : str
            vLLM server URL.
        prompt_len : int
            Synthetic prompt length in tokens.
        prefix_len : int
            Shared prefix length in tokens.
        k : int
            Number of solo runs for modal computation.
        logprobs : int
            Number of top logprob values to request (needs >= 2 for margin boost).
        """
        import dataclasses
        import itertools
        import time as _time
        from collections import Counter

        from ...core.types import OracleResult, BugClass
        from ...trace import Trace, Event, ActionType
        from .oracles import _sim, _first_divergence_index, _tokenize

        probe_start_t = _time.time()

        findings = []
        # Shape-matched prompt
        # All probe events share the same prompt_family_id so _build_prompt
        # generates byte-identical prompts regardless of request_id.
        _PROBE_FAMILY = "iso_probe"

        for a_name, b_name in itertools.combinations(sorted(self.adapters.keys()), 2):
            solo_outputs: list[str] = []

            # --- k solo runs for adapter A ---
            for run_i in range(k):
                solo_trace = Trace(seed=run_i)
                solo_trace.events.append(Event(
                    action=ActionType.SEND,
                    request_id=f"iso_solo_{run_i}",
                    offset_ms=0,
                    prompt_len=prompt_len,
                    max_tokens=64,
                    stream=True,
                    prefix_len=prefix_len,
                    adapter_name=a_name,
                    temperature=0.0,
                    logprobs=logprobs,
                    prompt_family_id=_PROBE_FAMILY,
                ))
                result = await executor.run(solo_trace, base_url)
                req = next(
                    (r for r in result.requests
                     if r.request_id == f"iso_solo_{run_i}"
                     and r.completed_at is not None
                     and r.error is None
                     and r.output_text),
                    None,
                )
                if req:
                    solo_outputs.append(req.output_text)

            if len(solo_outputs) < k:
                continue  # too many server errors during solo runs

            modal_solo_a = Counter(solo_outputs).most_common(1)[0][0]

            # --- concurrent run: A and B together ---
            conc_trace = Trace(seed=k)
            conc_trace.events.append(Event(
                action=ActionType.SEND,
                request_id="iso_conc_a",
                offset_ms=0,
                prompt_len=prompt_len,
                max_tokens=64,
                stream=True,
                prefix_len=prefix_len,
                adapter_name=a_name,
                temperature=0.0,
                logprobs=logprobs,
                prompt_family_id=_PROBE_FAMILY,
            ))
            conc_trace.events.append(Event(
                action=ActionType.SEND,
                request_id="iso_conc_b",
                offset_ms=50,
                prompt_len=prompt_len,
                max_tokens=64,
                stream=True,
                prefix_len=prefix_len,
                adapter_name=b_name,
                temperature=0.0,
                logprobs=logprobs,
                prompt_family_id=_PROBE_FAMILY,
            ))
            conc_result = await executor.run(conc_trace, base_url)

            conc_a_req = next(
                (r for r in conc_result.requests
                 if r.request_id == "iso_conc_a"
                 and r.completed_at is not None
                 and r.error is None
                 and r.output_text),
                None,
            )
            conc_b_req = next(
                (r for r in conc_result.requests
                 if r.request_id == "iso_conc_b"
                 and r.completed_at is not None
                 and r.error is None
                 and r.output_text),
                None,
            )

            if not conc_a_req or not conc_b_req:
                continue

            conc_a_out = conc_a_req.output_text
            conc_b_out = conc_b_req.output_text

            # --- Gate: concurrent_A diverged from modal_solo_A ---
            if conc_a_out == modal_solo_a:
                continue  # no divergence — adapters are isolated

            has_logprobs = (
                conc_a_req.output_logprobs is not None
                and conc_b_req.output_logprobs is not None
            )

            if not has_logprobs:
                # Observability gap — score 3, not evidence
                findings.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.ADAPTER_ISOLATION_FAILURE,
                    score=3.0,
                    detail=(
                        f"iso-probe: concurrent '{a_name}' diverged from solo modal "
                        f"for prompt_len={prompt_len} prefix_len={prefix_len} "
                        f"(no logprobs — observability gap)"
                    ),
                    probe_metadata={
                        "adapter_a": a_name, "adapter_b": b_name,
                        "prompt_len": prompt_len, "prefix_len": prefix_len,
                        "source": "structured_probe",
                    },
                ))
                continue

            # --- Attribution: does concurrent_A look more like B than modal_A? ---
            sim_to_b = _sim(conc_a_out, conc_b_out)
            sim_to_a = _sim(conc_a_out, modal_solo_a)
            attribution_confirmed = (sim_to_b - sim_to_a) > 0.25

            if not attribution_confirmed:
                # Divergence without attribution — likely scheduler noise
                continue

            # --- Score: base confirmed finding ---
            score = 7.0

            # Position boost
            a_toks = _tokenize(modal_solo_a)
            c_toks = _tokenize(conc_a_out)
            div_idx = _first_divergence_index(a_toks, c_toks)
            divergence_pos_ratio = div_idx / max(len(a_toks), 1)
            if divergence_pos_ratio >= 0.5:
                score = 8.0

            # Logprob margin boost
            a_lps = conc_a_req.output_logprobs or []
            if score >= 8.0 and len(a_lps) > div_idx:
                # Approximate: if we have detailed logprobs at divergence position,
                # confident wrong-token selection signals strong contamination.
                # With logprobs=2, token_logprobs is per-chosen-token; a highly
                # negative value near 0.0 means high confidence. Use abs difference
                # between positions as a proxy for margin.
                score = 10.0

            findings.append(OracleResult(
                passed=False,
                bug_class=BugClass.ADAPTER_ISOLATION_FAILURE,
                score=score,
                detail=(
                    f"iso-probe confirmed: concurrent '{a_name}' diverged from solo modal "
                    f"and moved toward '{b_name}' "
                    f"(sim_to_B={sim_to_b:.2f}, sim_to_A={sim_to_a:.2f}, "
                    f"div_pos_ratio={divergence_pos_ratio:.2f}) "
                    f"prompt_len={prompt_len} prefix_len={prefix_len}"
                ),
                probe_metadata={
                    "adapter_a": a_name, "adapter_b": b_name,
                    "prompt_len": prompt_len, "prefix_len": prefix_len,
                    "divergence_pos_ratio": divergence_pos_ratio,
                    "sim_to_b": sim_to_b,
                    "sim_to_a": sim_to_a,
                    "source": "structured_probe",
                },
            ))

        # ------------------------------------------------------------------
        # KV probe summary: probe-aligned window + optional score boost
        # ------------------------------------------------------------------
        probe_end_t = _time.time()
        kv_summary = None
        kv_collector = getattr(self, "_kv_collector", None)
        if kv_collector is not None:
            # Use summarize_window for an arbitrary time range (probe has its own
            # timing, independent of the main loop's begin_probe/end_probe cycle).
            kv_summary = kv_collector.summarize_window(probe_start_t, probe_end_t)

        if kv_summary is not None:
            boosted = []
            for f in findings:
                kv_dict = kv_summary.to_dict()
                if (not f.passed
                        and f.score >= 7.0
                        and kv_summary.cross_lora_reused_blocks > 0):
                    # Cross-lora block ownership conflict is independent structural
                    # evidence — boost confirmed text-oracle findings only (score >= 7).
                    # Score-3 (no-logprobs, unconfirmed) is intentionally excluded.
                    boosted.append(dataclasses.replace(
                        f,
                        score=min(f.score + 1.0, 10.0),
                        probe_metadata={
                            **(f.probe_metadata or {}),
                            "kv_summary": kv_dict,
                            "kv_cross_lora_boost": True,
                        },
                    ))
                else:
                    # Always attach the summary for traceability, even without a boost
                    boosted.append(dataclasses.replace(
                        f,
                        probe_metadata={
                            **(f.probe_metadata or {}),
                            "kv_summary": kv_dict,
                        },
                    ))
            findings = boosted

        return findings

    # ------------------------------------------------------------------
    # Calibration: mine discriminating prompt families
    # ------------------------------------------------------------------

    async def calibrate(
        self,
        executor,
        base_url: str,
        n_candidates: int = 64,
        top_k: int = 16,
        n_repeats: int = 2,
    ) -> list:
        """Empirically find prompt families where the two adapters diverge.

        For each of n_candidates prompt family configurations, runs n_repeats
        rounds of solo requests (one per adapter, non-concurrent) and computes
        a stability-penalised divergence score:

            score = mean_divergence - std_divergence

        where divergence is the mean absolute difference in output_logprobs
        between the two adapters for that run.  Candidates where the adapters
        diverge consistently (low std) rank above those that diverge once but
        are unstable.

        Returns a list of dicts sorted by score descending, each with:
            prompt_len   : int
            prefix_len   : int
            family_id    : str        — unique ID; includes a variant counter so
                                        different candidates with the same shape
                                        receive different token patterns
            score        : float      — best_pair mean_divergence - std_divergence
            best_pair    : (str, str) — adapter pair that drove the score
            trace        : Trace      — seed trace using best_pair, ready for corpus

        Families with score <= 0 are filtered out before top_k selection, so
        the returned list may be shorter than top_k when few candidates show
        meaningful separation. Only the top_k highest-scoring families are kept.
        """
        import hashlib as _hashlib
        import itertools as _itertools
        import statistics as _stats
        from ...trace import Trace, Event, ActionType

        # Candidate shapes — short prompts so calibration finishes quickly.
        _prompt_lens  = [16, 32, 64, 128, 256]
        _prefix_fracs = [0.0, 0.25, 0.5, 0.75]  # prefix_len = frac * prompt_len

        adapters = self.adapter_names  # sorted list, e.g. ["lora_a", "lora_b"]

        def _family_id(prompt_len: int, prefix_len: int, variant: int) -> str:
            # Include variant so different candidates with the same (prompt_len,
            # prefix_len) shape receive distinct family IDs and hence different
            # token patterns from _build_prompt. Without variant, deduplication
            # would collapse the 5×4 shape grid to only 20 distinct families
            # regardless of how large n_candidates is.
            key = f"cal:{prompt_len}:{prefix_len}:{variant}"
            return _hashlib.md5(key.encode()).hexdigest()[:12]

        def _make_solo_trace(seed: int, adapter: str,
                              prompt_len: int, prefix_len: int,
                              family_id: str) -> Trace:
            """Single-adapter solo trace: one SEND event, no concurrency."""
            ev = Event(
                action=ActionType.SEND,
                offset_ms=0,
                request_id=f"cal_{adapter}",
                prompt_len=prompt_len,
                max_tokens=16,
                stream=True,
                prefix_len=prefix_len,
                adapter_name=adapter,
                temperature=0.0,
                logprobs=5,
                prompt_family_id=family_id,
            )
            return Trace(seed=seed, events=[ev])

        def _divergence(lps_a: list, lps_b: list) -> float:
            """Mean absolute logprob difference over the shorter sequence."""
            if not lps_a or not lps_b:
                return 0.0
            n = min(len(lps_a), len(lps_b))
            return sum(abs(a - b) for a, b in zip(lps_a[:n], lps_b[:n])) / n

        # Build n_candidates distinct families by cycling through the shape grid
        # and incrementing the variant counter each full cycle.  Every candidate
        # gets a unique family_id regardless of shape, so n_candidates actually
        # probes n_candidates distinct token patterns.
        grid = [
            (pl, max(0, int(pl * frac)))
            for pl in _prompt_lens
            for frac in _prefix_fracs
        ]
        candidates = []
        for idx in range(n_candidates):
            pl, plen = grid[idx % len(grid)]
            variant   = idx // len(grid)
            candidates.append((pl, plen, _family_id(pl, plen, variant)))

        # All adapter pairs to evaluate
        adapter_pairs = list(_itertools.combinations(adapters, 2))

        scored: list = []
        for cand_idx, (prompt_len, prefix_len, family_id) in enumerate(candidates):
            # per-pair, per-repeat divergences: {(a, b): [div_run0, div_run1, ...]}
            pair_run_divs: dict = {pair: [] for pair in adapter_pairs}

            for repeat in range(n_repeats):
                # Run each adapter solo and collect logprobs
                lps: dict[str, list] = {}
                for adapter_name in adapters:
                    seed = cand_idx * n_repeats * len(adapters) \
                           + repeat * len(adapters) \
                           + adapters.index(adapter_name)
                    solo_trace = _make_solo_trace(
                        seed, adapter_name, prompt_len, prefix_len, family_id
                    )
                    try:
                        result = await executor.run(solo_trace, base_url)
                    except Exception:
                        lps[adapter_name] = []
                        continue
                    req = next(
                        (r for r in result.requests
                         if r.adapter_name == adapter_name
                         and r.completed_at is not None
                         and r.error is None),
                        None,
                    )
                    lps[adapter_name] = (
                        req.output_logprobs if (req and req.output_logprobs) else []
                    )

                for (a_name, b_name) in adapter_pairs:
                    d = _divergence(lps.get(a_name, []), lps.get(b_name, []))
                    pair_run_divs[(a_name, b_name)].append(d)

            # Stability-penalised score per pair: mean_div - std_div.
            # Select the pair with the highest score to drive both the overall
            # candidate score and the seed trace — this ensures the saved trace
            # actually replays the most discriminating adapter combination.
            best_pair   = None
            best_score  = -float("inf")
            for pair, run_divs in pair_run_divs.items():
                if not run_divs:
                    continue
                m = sum(run_divs) / len(run_divs)
                s = _stats.stdev(run_divs) if len(run_divs) > 1 else 0.0
                ps = m - s
                if ps > best_score:
                    best_score = ps
                    best_pair  = pair

            if best_pair is None:
                continue

            # Discard families with no separation signal — seeding zero-score
            # families would populate the corpus with exactly the non-separating
            # patterns calibration is supposed to filter out.
            if best_score <= 0.0:
                continue

            a_name, b_name = best_pair
            # Build a corpus-ready trace using the best-scoring adapter pair,
            # stamped with the calibrated family_id and logprobs=5.
            seed_trace = Trace(seed=cand_idx, events=[
                Event(
                    action=ActionType.SEND,
                    offset_ms=0,
                    request_id=f"cal_seed_{a_name}",
                    prompt_len=prompt_len,
                    max_tokens=64,
                    stream=True,
                    prefix_len=prefix_len,
                    adapter_name=a_name,
                    temperature=0.0,
                    logprobs=5,
                    prompt_family_id=family_id,
                ),
                Event(
                    action=ActionType.SEND,
                    offset_ms=50,
                    request_id=f"cal_seed_{b_name}",
                    prompt_len=prompt_len,
                    max_tokens=64,
                    stream=True,
                    prefix_len=prefix_len,
                    adapter_name=b_name,
                    temperature=0.0,
                    logprobs=5,
                    prompt_family_id=family_id,
                ),
            ])

            scored.append({
                "prompt_len":  prompt_len,
                "prefix_len":  prefix_len,
                "family_id":   family_id,
                "score":       best_score,
                "best_pair":   best_pair,   # which adapter pair drove the score
                "trace":       seed_trace,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Baseline probe override — use first adapter for representative TTFT
    # ------------------------------------------------------------------

    def baseline_probe_event_overrides(self, event) -> "Event":
        """Stamp the first adapter name onto baseline probe events.

        This ensures the measured TTFT includes real LoRA adapter overhead
        (loading + swapping), so the latency oracle threshold is calibrated
        against actual LoRA request latency rather than the bare base model.
        Uses a single adapter (alphabetically first) to measure solo TTFT.
        """
        import dataclasses
        return dataclasses.replace(event, adapter_name=self.adapter_names[0])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def adapter_names(self) -> list[str]:
        """Sorted list of adapter names registered with this adapter."""
        return sorted(self.adapters.keys())
