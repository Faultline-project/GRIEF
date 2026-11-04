"""
SglangLoRAAdapter: extends SglangBaseAdapter for concurrent LoRA adapter serving.

Adds:
  - --lora-paths flags via server_flags_extra()
  - Per-request adapter selection via event.adapter_name → payload["lora_path"]
  - LoRA-specific mutations (switch_adapter, concurrent_adapter_burst, ...)
  - Solo-vs-concurrent isolation oracle (primary) + cross-adapter fast-path (secondary)
  - Calibration: mine discriminating prompt families

Key difference from VllmLoRAAdapter:
  - SGLang resolves lora_path by the registered adapter name (the key from
    --lora-paths name=path at server startup), NOT by the filesystem path.
    build_payload injects payload["lora_path"] = event.adapter_name directly.
  - KV block events require --kv-events-config on the server AND
    --kv-events-endpoint on the fuzzer (sglang_lora_fuzz_job.slurm wires both
    automatically when --kv-events-endpoint is passed to submit_sglang_lora.sh).
  - Server flag is --lora-paths (SGLang) vs --lora-modules (vLLM).

Probe strategy (adapter_isolation_probe):
  Primary invariant: same request, same adapter → same output regardless of
  concurrent load, batch composition, or adapter slot pressure.
  Four passes per adapter pair:
    0. Base model solo  → modal_solo_base  (enables base_model_drift detection)
    1. Adapter A solo   → modal_solo_a     (deterministic baseline)
    2. Light concurrent: A + B at 50 ms gap
    3. High-pressure:   A + 6 B at 0 ms
    4. Preemption:      A + 4 B at 0 ms, max_tokens=128 (only if prompt_len >= 256)
  Attribution: toward_B (7–9), base_model_drift (6), unknown_drift (4).
"""

from __future__ import annotations

from ..sglang.legacy import SglangBaseAdapter
from .mutators import LORA_MUTATIONS


class SglangLoRAAdapter(SglangBaseAdapter):
    """
    TargetAdapter for SGLang with --lora-paths.

    Parameters
    ----------
    adapters : dict[str, str]
        Mapping of adapter name → path on disk.
        e.g. {"lora_a": "/tmp/lora_weights/lora_a",
               "lora_b": "/tmp/lora_weights/lora_b"}
    """

    # Bug classes produced by the generic oracle that are not meaningful for a
    # LoRA-correctness campaign.  Latency and memory findings fire on base-model
    # traces and drown the adapter-isolation signal.
    _SUPPRESSED_CLASSES: frozenset = frozenset()  # populated after BugClass import

    def __init__(self, adapters: dict[str, str]) -> None:
        super().__init__()
        if len(adapters) < 2:
            raise ValueError("SglangLoRAAdapter requires at least 2 adapters "
                             "for isolation testing.")
        self.adapters = adapters
        self._pair_rotation_idx: int = 0  # round-robin pair sampling across probe calls

    # ------------------------------------------------------------------
    # Server flags
    # ------------------------------------------------------------------

    def server_flags_extra(self) -> str:
        modules = " ".join(f"{name}={path}"
                           for name, path in self.adapters.items())
        return f"--lora-paths {modules}"

    # ------------------------------------------------------------------
    # Payload construction — inject lora_path
    # ------------------------------------------------------------------

    def build_payload(self, event) -> tuple[str, dict]:
        endpoint, payload = super().build_payload(event)
        if event.adapter_name is not None:
            if event.adapter_name in self.adapters:
                payload["lora_path"] = event.adapter_name
        return endpoint, payload

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Finding suppression
    # ------------------------------------------------------------------

    def filter_findings(self, findings: list) -> list:
        from ...core.types import BugClass
        suppressed = {BugClass.LATENCY_SPIKE, BugClass.MEMORY_LEAK, BugClass.TOKEN_RATE_STALL}
        return [
            f for f in findings
            if f.bug_class not in suppressed
            and not (
                f.bug_class == BugClass.ADAPTER_ISOLATION_FAILURE and f.score < 7.0
            )
        ]

    def extra_mutations(self) -> list:
        from .mutators import make_inject_lora_adapters
        return (list(super().extra_mutations())
                + list(LORA_MUTATIONS)
                + [make_inject_lora_adapters(list(self.adapters.keys()))])

    # ------------------------------------------------------------------
    # Oracles (fast-path secondary + probe bootstrap)
    # ------------------------------------------------------------------

    def _probe_bootstrap_trigger(self, result) -> list:
        """Emit passed=True triggers when concurrent LoRA sends appear in a result.

        These are not bug findings (passed=True is filtered out before the finding
        queue) but ARE scanned by runner.py's probe enqueue loop, which iterates
        over findings + extra. This decouples probe scheduling from the rare
        cross-adapter equality event and lets the probe run on every trace that
        contains concurrent multi-adapter activity.
        """
        import itertools as _itertools
        from ...core.types import OracleResult, BugClass

        adapter_reqs: dict = {}
        for req in result.requests:
            if (req.adapter_name in self.adapters
                    and req.completed_at is not None
                    and req.error is None
                    and req.output_text):
                adapter_reqs.setdefault(req.adapter_name, []).append(req)

        if len(adapter_reqs) < 2:
            return []

        # Emit at most 2 triggers per result (first 2 pairs alphabetically).
        # With N adapters, all C(N,2) pairs would create a trigger storm; the
        # probe itself samples pairs via round-robin so full coverage is maintained.
        _MAX_TRIGGERS = 2
        all_pairs = list(_itertools.combinations(sorted(adapter_reqs.keys()), 2))

        triggers = []
        for a_name, b_name in all_pairs[:_MAX_TRIGGERS]:
            a_req = adapter_reqs[a_name][0]
            triggers.append(OracleResult(
                passed=True,
                bug_class=BugClass.ADAPTER_ISOLATION_FAILURE,
                score=3.0,
                detail=f"probe-bootstrap: concurrent {a_name}+{b_name} observed",
                probe_metadata={
                    "adapter_a":  a_name,
                    "adapter_b":  b_name,
                    "prompt_len": getattr(a_req, "prompt_len", 64) or 64,
                    "prefix_len": getattr(a_req, "prefix_len", 0) or 0,
                    "source":     "bootstrap_trigger",
                },
            ))
        return triggers

    def evaluate_extra(self, result, session, trace=None) -> list:
        from .oracles import check_cross_adapter_isolation
        return (list(super().evaluate_extra(result, session, trace=trace))
                + check_cross_adapter_isolation(result)
                + self._probe_bootstrap_trigger(result))

    def analyze_kv_summary(self, summary) -> list:
        """LoRA-specific KV anomaly scoring: cross-adapter block reuse."""
        from .oracles import analyze_cross_lora_kv
        return analyze_cross_lora_kv(summary)

    # ------------------------------------------------------------------
    # Probe helpers — internal, one per pass
    # ------------------------------------------------------------------

    async def _run_solo_pass(
        self,
        executor,
        base_url: str,
        adapter_name,       # str | None  (None = base model)
        prompt_len: int,
        prefix_len: int,
        k: int,
        logprobs: int,
        family_id: str,
    ):
        """Run k solo requests; return modal output text or None if < k succeeded."""
        from collections import Counter
        from ...trace import Trace, Event, ActionType

        outputs: list[str] = []
        for run_i in range(k):
            trace = Trace(seed=run_i)
            trace.events.append(Event(
                action=ActionType.SEND,
                request_id=f"iso_solo_{run_i}",
                offset_ms=0,
                prompt_len=prompt_len,
                max_tokens=64,
                stream=True,
                prefix_len=prefix_len,
                adapter_name=adapter_name,
                temperature=0.0,
                logprobs=logprobs,
                prompt_family_id=family_id,
            ))
            result = await executor.run(trace, base_url)
            req = next(
                (r for r in result.requests
                 if r.request_id == f"iso_solo_{run_i}"
                 and r.completed_at is not None
                 and r.error is None
                 and r.output_text),
                None,
            )
            if req:
                outputs.append(req.output_text)

        if len(outputs) < k:
            return None
        return Counter(outputs).most_common(1)[0][0]

    def _score_concurrent_finding(
        self,
        conc_a_out: str,
        modal_solo_a: str,
        conc_b_out,           # str | None
        modal_solo_base,      # str | None
        conc_a_req,
        a_name: str,
        b_name: str,
        prompt_len: int,
        prefix_len: int,
        pass_label: str,
    ):
        """Gate + classify a concurrent divergence. Returns OracleResult or None.

        Attribution:
          toward_B       sim_to_B − sim_to_A > 0.25   → score 7 / 8 / 9
          base_model_drift sim_to_base > sim_to_A      → score 6
          unknown_drift  (neither)                     → score 4
        """
        from ...core.types import OracleResult, BugClass
        from .oracles import _sim, _first_divergence_index, _tokenize

        if conc_a_out == modal_solo_a:
            return None

        sim_to_b    = _sim(conc_a_out, conc_b_out)    if conc_b_out    else 0.0
        sim_to_a    = _sim(conc_a_out, modal_solo_a)
        sim_to_base = _sim(conc_a_out, modal_solo_base) if modal_solo_base else 0.0

        a_toks  = _tokenize(modal_solo_a)
        c_toks  = _tokenize(conc_a_out)
        div_idx = _first_divergence_index(a_toks, c_toks)
        div_pos = div_idx / max(len(a_toks), 1)

        # High-pressure and preemption passes carry +1 confidence over light.
        _pressure = {"high_pressure": 1, "preemption": 1}.get(pass_label, 0)

        if sim_to_b - sim_to_a > 0.25:
            attribution = "toward_B"
            score = (8.0 if div_pos >= 0.5 else 7.0) + _pressure
            a_lps = conc_a_req.output_logprobs or []
            if score >= 8.0 and len(a_lps) > div_idx:
                score = min(score + 1.0, 10.0)
        elif modal_solo_base and sim_to_base > sim_to_a:
            attribution = "base_model_drift"
            score = 6.0 + _pressure
        else:
            attribution = "unknown_drift"
            score = 4.0 + _pressure

        return OracleResult(
            passed=False,
            bug_class=BugClass.ADAPTER_ISOLATION_FAILURE,
            score=score,
            detail=(
                f"iso-probe [{pass_label}]: concurrent '{a_name}' diverged from solo modal "
                f"attribution={attribution} "
                f"(sim_to_B={sim_to_b:.2f}, sim_to_A={sim_to_a:.2f}, "
                f"sim_to_base={sim_to_base:.2f}, div_pos={div_pos:.2f}) "
                f"prompt_len={prompt_len} prefix_len={prefix_len}"
            ),
            probe_metadata={
                "adapter_a":   a_name,
                "adapter_b":   b_name,
                "prompt_len":  prompt_len,
                "prefix_len":  prefix_len,
                "attribution": attribution,
                "sim_to_b":    round(sim_to_b, 3),
                "sim_to_a":    round(sim_to_a, 3),
                "sim_to_base": round(sim_to_base, 3),
                "div_pos":     round(div_pos, 3),
                "pass":        pass_label,
                "source":      "structured_probe",
            },
        )

    async def _run_light_concurrent_pass(
        self,
        executor,
        base_url: str,
        a_name: str,
        b_name: str,
        prompt_len: int,
        prefix_len: int,
        modal_solo_a: str,
        modal_solo_base,
        logprobs: int,
        family_id: str,
        seed: int,
    ) -> list:
        """Pass 2: A at 0 ms, B at 50 ms — minimal overlap."""
        from ...trace import Trace, Event, ActionType

        trace = Trace(seed=seed)
        for rid, adapter, offset in [
            ("iso_lc_a", a_name, 0),
            ("iso_lc_b", b_name, 50),
        ]:
            trace.events.append(Event(
                action=ActionType.SEND,
                request_id=rid,
                offset_ms=offset,
                prompt_len=prompt_len,
                max_tokens=64,
                stream=True,
                prefix_len=prefix_len,
                adapter_name=adapter,
                temperature=0.0,
                logprobs=logprobs,
                prompt_family_id=family_id,
            ))
        result = await executor.run(trace, base_url)

        def _get(rid):
            return next((r for r in result.requests
                         if r.request_id == rid
                         and r.completed_at and not r.error and r.output_text), None)

        conc_a = _get("iso_lc_a")
        conc_b = _get("iso_lc_b")
        if not conc_a:
            return []
        f = self._score_concurrent_finding(
            conc_a.output_text, modal_solo_a,
            conc_b.output_text if conc_b else None,
            modal_solo_base, conc_a,
            a_name, b_name, prompt_len, prefix_len, "light",
        )
        return [f] if f else []

    async def _run_high_pressure_pass(
        self,
        executor,
        base_url: str,
        a_name: str,
        b_name: str,
        prompt_len: int,
        prefix_len: int,
        modal_solo_a: str,
        modal_solo_base,
        logprobs: int,
        family_id: str,
        seed: int,
        n_stress: int = 6,
        stress_adapters: "list[str] | None" = None,
    ) -> list:
        """Pass 3: A + n_stress stress requests at 0 ms — stresses batch scheduler.

        stress_adapters cycles through all adapters other than A so that with a
        small adapter cache (--max-loras-per-batch 2) every stress request forces
        an eviction/reload.  Falls back to repeating b_name when not provided.
        """
        from ...trace import Trace, Event, ActionType

        _stress = stress_adapters if stress_adapters else [b_name]

        trace = Trace(seed=seed)
        trace.events.append(Event(
            action=ActionType.SEND,
            request_id="iso_hp_a",
            offset_ms=0,
            prompt_len=prompt_len,
            max_tokens=64,
            stream=True,
            prefix_len=prefix_len,
            adapter_name=a_name,
            temperature=0.0,
            logprobs=logprobs,
            prompt_family_id=family_id,
        ))
        for i in range(n_stress):
            trace.events.append(Event(
                action=ActionType.SEND,
                request_id=f"iso_hp_b{i}",
                offset_ms=0,
                prompt_len=prompt_len,
                max_tokens=64,
                stream=True,
                prefix_len=prefix_len,
                adapter_name=_stress[i % len(_stress)],
                temperature=0.0,
                logprobs=logprobs,
                prompt_family_id=family_id,
            ))
        result = await executor.run(trace, base_url)

        conc_a = next((r for r in result.requests
                       if r.request_id == "iso_hp_a"
                       and r.completed_at and not r.error and r.output_text), None)
        conc_b = next((r for r in result.requests
                       if r.request_id.startswith("iso_hp_b")
                       and r.completed_at and not r.error and r.output_text), None)
        if not conc_a:
            return []
        f = self._score_concurrent_finding(
            conc_a.output_text, modal_solo_a,
            conc_b.output_text if conc_b else None,
            modal_solo_base, conc_a,
            a_name, b_name, prompt_len, prefix_len, "high_pressure",
        )
        return [f] if f else []

    async def _run_preemption_pass(
        self,
        executor,
        base_url: str,
        a_name: str,
        b_name: str,
        prompt_len: int,
        prefix_len: int,
        modal_solo_a: str,
        modal_solo_base,
        logprobs: int,
        family_id: str,
        seed: int,
        stress_adapters: "list[str] | None" = None,
    ) -> list:
        """Pass 4: A + 4 stress requests at 0 ms, max_tokens=128.

        stress_adapters cycles through all adapters other than A.  Only runs
        if prompt_len >= 256 (ensures long enough sequences to trigger preemption).
        """
        if prompt_len < 256:
            return []
        from ...trace import Trace, Event, ActionType

        _stress = stress_adapters if stress_adapters else [b_name]

        trace = Trace(seed=seed)
        trace.events.append(Event(
            action=ActionType.SEND,
            request_id="iso_pre_a",
            offset_ms=0,
            prompt_len=prompt_len,
            max_tokens=128,
            stream=True,
            prefix_len=prefix_len,
            adapter_name=a_name,
            temperature=0.0,
            logprobs=logprobs,
            prompt_family_id=family_id,
        ))
        for i in range(4):
            trace.events.append(Event(
                action=ActionType.SEND,
                request_id=f"iso_pre_b{i}",
                offset_ms=0,
                prompt_len=prompt_len,
                max_tokens=128,
                stream=True,
                prefix_len=prefix_len,
                adapter_name=_stress[i % len(_stress)],
                temperature=0.0,
                logprobs=logprobs,
                prompt_family_id=family_id,
            ))
        result = await executor.run(trace, base_url)

        conc_a = next((r for r in result.requests
                       if r.request_id == "iso_pre_a"
                       and r.completed_at and not r.error and r.output_text), None)
        conc_b = next((r for r in result.requests
                       if r.request_id.startswith("iso_pre_b")
                       and r.completed_at and not r.error and r.output_text), None)
        if not conc_a:
            return []
        f = self._score_concurrent_finding(
            conc_a.output_text, modal_solo_a,
            conc_b.output_text if conc_b else None,
            modal_solo_base, conc_a,
            a_name, b_name, prompt_len, prefix_len, "preemption",
        )
        return [f] if f else []

    # ------------------------------------------------------------------
    # Structured isolation probe — coordinator
    # ------------------------------------------------------------------

    async def adapter_isolation_probe(
        self,
        executor,
        base_url: str,
        prompt_len: int,
        prefix_len: int,
        k: int = 3,
        logprobs: int = 2,
        max_pairs: int = 3,
    ) -> list:
        """
        Solo-vs-concurrent isolation probe.

        For each sampled adapter pair (A, B):
          Pass 0  base model solo k times → modal_solo_base
          Pass 1  adapter A solo k times  → modal_solo_a
          Pass 2  light concurrent: A at 0 ms, B at 50 ms
          Pass 3  high-pressure: A + stress_adapters cycling at 0 ms
          Pass 4  preemption: A + stress_adapters cycling at 0 ms, max_tokens=128

        Pairs are sampled via round-robin rotation (self._pair_rotation_idx) so
        that with many adapters every pair gets probed across invocations without
        running all C(N,2) pairs per call.

        stress_adapters = all adapters except A, cycled in Passes 3 and 4 so that
        with a small --max-loras-per-batch the cache evicts and reloads on every
        stress request.

        Scoring:
          toward_B       (sim_to_B − sim_to_A > 0.25)  → 7 / 8 / 9
          base_model_drift (sim_to_base > sim_to_A)     → 6
          unknown_drift                                  → 4
          +1 KV boost when cross_lora_reused_blocks > 0 (score >= 7 only)
        """
        import dataclasses
        import itertools
        import time as _time

        probe_start_t = _time.time()
        findings: list = []
        _PROBE_FAMILY = "iso_probe"

        # Round-robin pair selection: take max_pairs consecutive pairs starting
        # from _pair_rotation_idx, then advance the index for next invocation.
        all_pairs = list(itertools.combinations(sorted(self.adapters.keys()), 2))
        n_pairs = len(all_pairs)
        start = self._pair_rotation_idx % n_pairs
        pairs_to_probe = [
            all_pairs[(start + i) % n_pairs]
            for i in range(min(max_pairs, n_pairs))
        ]
        self._pair_rotation_idx = (self._pair_rotation_idx + max_pairs) % n_pairs

        # Pass 0: base model baseline (once per probe invocation)
        modal_solo_base = await self._run_solo_pass(
            executor, base_url, None,
            prompt_len, prefix_len, k, logprobs, _PROBE_FAMILY,
        )

        for pair_idx, (a_name, b_name) in enumerate(pairs_to_probe):
            # All adapters except A cycle through the stress slots so that each
            # stress request targets a different adapter, forcing cache eviction.
            stress_adapters = [n for n in sorted(self.adapters.keys()) if n != a_name]

            # Pass 1: solo baseline for adapter A
            modal_solo_a = await self._run_solo_pass(
                executor, base_url, a_name,
                prompt_len, prefix_len, k, logprobs, _PROBE_FAMILY,
            )
            if modal_solo_a is None:
                continue

            seed_base = k + pair_idx * 10

            # Pass 2: light concurrent
            findings += await self._run_light_concurrent_pass(
                executor, base_url, a_name, b_name,
                prompt_len, prefix_len, modal_solo_a, modal_solo_base,
                logprobs, _PROBE_FAMILY, seed_base,
            )

            # Pass 3: high-pressure concurrent (stress adapters cycle for eviction)
            findings += await self._run_high_pressure_pass(
                executor, base_url, a_name, b_name,
                prompt_len, prefix_len, modal_solo_a, modal_solo_base,
                logprobs, _PROBE_FAMILY, seed_base + 1,
                stress_adapters=stress_adapters,
            )

            # Pass 4: preemption-oriented (stress adapters cycle for eviction)
            findings += await self._run_preemption_pass(
                executor, base_url, a_name, b_name,
                prompt_len, prefix_len, modal_solo_a, modal_solo_base,
                logprobs, _PROBE_FAMILY, seed_base + 2,
                stress_adapters=stress_adapters,
            )

        # KV boost: cross_lora_reused_blocks is independent structural evidence
        probe_end_t = _time.time()
        kv_collector = getattr(self, "_kv_collector", None)
        if kv_collector is not None and findings:
            kv_summary = kv_collector.summarize_window(probe_start_t, probe_end_t)
            if kv_summary is not None and kv_summary.cross_lora_reused_blocks > 0:
                boosted = []
                for f in findings:
                    kv_dict = kv_summary.to_dict()
                    if not f.passed and f.score >= 7.0:
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

        Returns a list of dicts sorted by score descending, each with:
            prompt_len, prefix_len, family_id, score, best_pair, trace
        """
        import hashlib as _hashlib
        import itertools as _itertools
        import statistics as _stats
        from ...trace import Trace, Event, ActionType

        _prompt_lens  = [16, 32, 64, 128, 256]
        _prefix_fracs = [0.0, 0.25, 0.5, 0.75]

        adapters = self.adapter_names

        def _family_id(prompt_len: int, prefix_len: int, variant: int) -> str:
            key = f"cal:{prompt_len}:{prefix_len}:{variant}"
            return _hashlib.md5(key.encode()).hexdigest()[:12]

        def _make_solo_trace(seed, adapter, prompt_len, prefix_len, family_id):
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

        def _divergence(lps_a, lps_b) -> float:
            if not lps_a or not lps_b:
                return 0.0
            n = min(len(lps_a), len(lps_b))
            return sum(abs(a - b) for a, b in zip(lps_a[:n], lps_b[:n])) / n

        grid = [
            (pl, max(0, int(pl * frac)))
            for pl in _prompt_lens
            for frac in _prefix_fracs
        ]
        candidates = []
        for idx in range(n_candidates):
            pl, plen = grid[idx % len(grid)]
            variant  = idx // len(grid)
            candidates.append((pl, plen, _family_id(pl, plen, variant)))

        adapter_pairs = list(_itertools.combinations(adapters, 2))

        scored: list = []
        for cand_idx, (prompt_len, prefix_len, family_id) in enumerate(candidates):
            pair_run_divs: dict = {pair: [] for pair in adapter_pairs}

            for repeat in range(n_repeats):
                lps: dict[str, list] = {}
                for adapter_name in adapters:
                    seed = (cand_idx * n_repeats * len(adapters)
                            + repeat * len(adapters)
                            + adapters.index(adapter_name))
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

            best_pair  = None
            best_score = -float("inf")
            for pair, run_divs in pair_run_divs.items():
                if not run_divs:
                    continue
                m  = sum(run_divs) / len(run_divs)
                s  = _stats.stdev(run_divs) if len(run_divs) > 1 else 0.0
                ps = m - s
                if ps > best_score:
                    best_score = ps
                    best_pair  = pair

            if best_pair is None or best_score <= 0.0:
                continue

            a_name, b_name = best_pair
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
                "prompt_len": prompt_len,
                "prefix_len": prefix_len,
                "family_id":  family_id,
                "score":      best_score,
                "best_pair":  best_pair,
                "trace":      seed_trace,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Baseline probe override
    # ------------------------------------------------------------------

    def baseline_probe_event_overrides(self, event) -> "Event":
        """Stamp the first adapter onto baseline probe events for realistic TTFT."""
        import dataclasses
        return dataclasses.replace(event, adapter_name=self.adapter_names[0])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def adapter_names(self) -> list[str]:
        return sorted(self.adapters.keys())
