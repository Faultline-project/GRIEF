"""Headless GRIEF runner. Use ``grief --help`` for the public CLI."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .preflight import check_server

from .runner  import Runner, RunnerConfig
from .monitor import DEFAULT_POLL_INTERVAL_S, EmptyMonitor, NvmlMonitor
from .backends.vllm   import cli as _vllm_cli
from .backends.sglang import cli as _sglang_cli

# Backends are checked in order; first non-None adapter wins.
# vLLM is last because it is the fallback and always returns non-None.
_BACKENDS = [_sglang_cli, _vllm_cli]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m fuzzer",
        description="Workload fuzzer for vLLM, SGLang, and OpenAI-compatible inference servers.",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--base-url", default="http://localhost:8000", metavar="URL",
        help="Base URL of the running inference server (default: http://localhost:8000)",
    )
    p.add_argument(
        "--model-name", default="", metavar="NAME",
        help="Model name for completions requests (auto-discovered from /v1/models if omitted)",
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    p.add_argument("--seeds-dir",    default=str(Path(__file__).parent / "data" / "seeds"),    metavar="DIR",
                   help="Directory containing seed trace JSON files (default: bundled seeds)")
    p.add_argument("--adapter-seeds-dir", default=None, metavar="DIR",
                   help="Optional additional seed directory for the active adapter. "
                        "Auto-resolved per backend if omitted.")
    p.add_argument("--corpus-dir",   default="./corpus",   metavar="DIR",
                   help="Directory for runtime corpus (default: ./corpus)")
    p.add_argument("--findings-dir", default="./findings", metavar="DIR",
                   help="Directory for bug reports (default: ./findings)")

    # ── Fuzzing parameters ────────────────────────────────────────────────────
    p.add_argument("--max-iterations", type=int, default=10_000, metavar="N",
                   help="Maximum fuzzing iterations (default: 10000)")
    p.add_argument("--mutations-per-seed", type=int, default=1, metavar="N",
                   help="Mutations applied per selected seed (default: 1)")
    p.add_argument("--seed", type=int, default=42, metavar="INT",
                   help="RNG seed for reproducibility (default: 42)")
    p.add_argument("--no-minimize", action="store_true",
                   help="Skip delta-debugging minimization when a bug is found")
    p.add_argument("--reverify-minimized", action="store_true",
                   help="Recheck reduced traces and record whether the mechanism signature survives.")
    p.add_argument("--events-file", default="", help=argparse.SUPPRESS)
    p.add_argument("--local-gpu-metrics", action="store_true",
                   help="Read NVML on this machine (only when client and server share a GPU).")
    p.add_argument("--verbose", action="store_true",
                   help="Emit one [NNNNN] verbose ... line per iteration with "
                        "mutation operators, trace complexity, and KV forensics. "
                        "KV forensics are populated only when --kv-events-endpoint "
                        "is also set.")

    # ── Monitor ───────────────────────────────────────────────────────────────
    p.add_argument("--mock-monitor", action="store_true",
                   help="Disable telemetry; still requires a running inference server")
    p.add_argument("--poll-interval", type=float,
                   default=DEFAULT_POLL_INTERVAL_S, metavar="SEC",
                   help=f"Seconds between metric polls "
                        f"(default: {DEFAULT_POLL_INTERVAL_S}). Lower costs API-server "
                        f"event-loop time: a /metrics render is ~3.8 ms and is "
                        f"synchronous on the loop that streams tokens — see "
                        f"fuzzer/monitor.py:DEFAULT_POLL_INTERVAL_S.")

    # ── Probes ────────────────────────────────────────────────────────────────
    p.add_argument("--prefix-equiv-probe-every", type=int, default=0, metavar="N",
                   help="Run the three-arm prefix-cache equivalence probe every N "
                        "iterations (0 = off, the default). Adds TWO "
                        "differently-salted twins beside every greedy SEND: the "
                        "plain arm reads the prefix cache, both salted arms "
                        "recompute. Salted-vs-salted disagreement means the "
                        "engine is nondeterministic independently of the cache "
                        "and is filed as probe_self_check, never as a cache bug; "
                        "salted arms agreeing while the cached arm differs is "
                        "filed as kv_corruption / pfx_divergent. Divergences "
                        "within the logits-dtype ULP band are dropped as FP ties. "
                        "Costs 2 extra requests per greedy SEND per firing. "
                        "Needs the server started with "
                        "--enable-prompt-tokens-details for its vacuity control. "
                        "Runs in addition to the determinism probe, not instead "
                        "of it.")

    p.add_argument("--probe-determinism-every", type=int, default=5, metavar="N",
                   help="Re-run the trace and compare greedy outputs every N "
                        "iterations (default: 5; 0 = off).")

    # ── Confirmation ladder (fuzzer/confirm/ladder.py) ───────────────────────
    p.add_argument("--confirm-ladder", action="store_true",
                   help="Confirm output divergences before filing them: the "
                        "diverged request alone twice on --base-url, the whole "
                        "trace twice and the request alone once on "
                        "--confirm-bi-url. FP ties are queued at low priority "
                        "instead of dropped. Only confirmed verdicts are filed; "
                        "every verdict goes to confirmation_log.jsonl.")
    p.add_argument("--confirm-bi-url", default="", metavar="URL",
                   help="Second server, started with VLLM_BATCH_INVARIANT=1.")
    p.add_argument("--confirm-bi-log", default="", metavar="PATH",
                   help="The BI server's log file; BI engagement is proven from it "
                        "at startup or the run refuses to start.")
    p.add_argument("--confirm-bi-backend", default="", metavar="NAME",
                   help="Attention backend of the BI server, when its log does not say.")
    p.add_argument("--confirm-gpu-arch", default="", metavar="ARCH",
                   help="GPU of the BI server (e.g. sm86, A6000), when its log does not say.")
    p.add_argument("--confirm-every", type=int, default=5, metavar="N",
                   help="Drain the confirmation queue every N iterations (default: 5).")
    p.add_argument("--confirm-max-per-drain", type=int, default=2, metavar="K",
                   help="Candidates confirmed per drain (default: 2).")
    p.add_argument("--confirm-budget-s", type=float, default=120.0, metavar="SEC",
                   help="Wall-clock budget per drain (default: 120).")
    p.add_argument("--confirm-queue-max", type=int, default=64, metavar="N",
                   help="Queue cap; overflow drops the lowest-priority candidate (default: 64).")
    p.add_argument("--no-confirm-feedback", action="store_true",
                   help="Disable confirmation feedback into corpus weights.")
    p.add_argument("--confirm-ties", choices=["ladder", "log"], default="ladder",
                   help="FP-tie candidates: confirm at low priority (ladder) or only log them.")
    p.add_argument("--prompt-embeds-dim", type=int, default=0, metavar="D",
                   help="Model hidden size; lets Event.prompt_embeds go out as embeddings "
                        "(server needs --enable-prompt-embeds). 0 = off.")
    p.add_argument("--log-reports", action="store_true",
                   help="Append every oracle report, with its trace, to findings/reports.jsonl.")
    # ── Oracle thresholds ─────────────────────────────────────────────────────
    p.add_argument("--memory-leak-tolerance-mb", type=float, default=100.0,
                   metavar="MB",
                   help="GPU memory growth above this triggers a leak finding (default: 100)")
    p.add_argument("--latency-threshold-ms", type=float, default=5000.0,
                   metavar="MS",
                   help="p99 TTFT above this triggers a latency finding (default: 5000)")
    p.add_argument("--request-timeout-s", type=float, default=30.0,
                   metavar="SEC",
                   help="Requests not completed within this are flagged as hung (default: 30)")

    # ── Campaign tracking ─────────────────────────────────────────────────────
    p.add_argument("--campaign-id", default="", metavar="ID",
                   help="Campaign identifier (auto-generated from URL+seed if omitted)")
    p.add_argument("--server-flags", default="", metavar="FLAGS",
                   help="Server launch flags (recorded in findings for reproducibility)")

    # ── Backend-specific flags ────────────────────────────────────────────────
    for backend in _BACKENDS:
        backend.register_args(p)

    return p.parse_args(argv)


async def main(argv=None) -> None:
    args = parse_args(argv)

    if args.max_iterations < 1 or args.request_timeout_s <= 0:
        raise ValueError("iterations and request timeout must be positive")
    info = await check_server(args.base_url, args.model_name)
    args.model_name = info["model"]
    if not Path(args.seeds_dir).is_dir():
        raise ValueError(f"seed directory does not exist: {args.seeds_dir}")

    # ── Select adapter via backend dispatch ───────────────────────────────────
    adapter = None
    active_backend = None
    for backend in _BACKENDS:
        adapter = backend.build_adapter(args)
        if adapter is not None:
            active_backend = backend
            break

    # ── Resolve adapter-specific seeds dir ───────────────────────────────────
    adapter_seeds_dir = args.adapter_seeds_dir
    if adapter_seeds_dir is None and active_backend is not None:
        adapter_seeds_dir = active_backend.resolve_seeds_dir(args, args.seeds_dir)

    # ── Build RunnerConfig ────────────────────────────────────────────────────
    config = RunnerConfig(
        events_file          = args.events_file,
        base_url             = args.base_url,
        model_name           = args.model_name,
        seeds_dir            = args.seeds_dir,
        corpus_dir           = args.corpus_dir,
        findings_dir         = args.findings_dir,
        max_iterations       = args.max_iterations,
        mutations_per_seed   = args.mutations_per_seed,
        seed                 = args.seed,
        minimize_findings    = not args.no_minimize,
        reverify_minimized   = args.reverify_minimized,
        poll_interval_s      = args.poll_interval,
        prefix_equiv_probe_every = args.prefix_equiv_probe_every,
        probe_determinism_every  = args.probe_determinism_every,
        confirm_ladder       = args.confirm_ladder,
        confirm_bi_url       = args.confirm_bi_url,
        confirm_bi_log       = args.confirm_bi_log,
        confirm_bi_backend   = args.confirm_bi_backend,
        confirm_gpu_arch     = args.confirm_gpu_arch,
        confirm_every        = args.confirm_every,
        confirm_max_per_drain = args.confirm_max_per_drain,
        confirm_budget_s     = args.confirm_budget_s,
        confirm_queue_max    = args.confirm_queue_max,
        confirm_feedback     = not args.no_confirm_feedback,
        confirm_ties         = args.confirm_ties,
        prompt_embeds_dim    = args.prompt_embeds_dim,
        log_reports          = args.log_reports,
        campaign_id          = args.campaign_id,
        server_flags         = args.server_flags,
        kv_events_endpoint   = getattr(args, "kv_events_endpoint", ""),
        adapter_seeds_dir    = adapter_seeds_dir,
        verbose              = args.verbose,
    )

    # ── Monitor ───────────────────────────────────────────────────────────────
    if args.mock_monitor:
        monitor = EmptyMonitor()
        print("[fuzzer] telemetry disabled (no synthetic metrics)")
    else:
        monitor = NvmlMonitor(
            metrics_url     = f"{args.base_url}/metrics",
            local_gpu       = args.local_gpu_metrics,
            poll_interval_s = args.poll_interval,
        )

    runner = Runner(config=config, adapter=adapter, monitor=monitor)

    # Override oracle thresholds from CLI
    runner.executor.request_timeout_s = args.request_timeout_s
    runner.oracle.memory_leak_tolerance_mb  = args.memory_leak_tolerance_mb
    runner.oracle.latency_ttft_threshold_ms = args.latency_threshold_ms
    runner.oracle.request_timeout_s         = args.request_timeout_s

    # ── Startup info ──────────────────────────────────────────────────────────
    print(f"[fuzzer] base_url     : {config.base_url}")
    print(f"[fuzzer] seeds_dir    : {config.seeds_dir}")
    print(f"[fuzzer] corpus_dir   : {config.corpus_dir}")
    print(f"[fuzzer] findings_dir : {config.findings_dir}")
    print(f"[fuzzer] max_iters    : {config.max_iterations}")
    print(f"[fuzzer] rng_seed     : {config.seed}")
    # Both resolved inside Corpus (CLI flag -> env var -> default), so print
    # what the Corpus actually settled on rather than what was asked for.
    print(f"[fuzzer] select_policy: {runner.corpus.selection_policy}")
    print(f"[fuzzer] fingerprint  : v{runner.corpus.fingerprint_version}")
    if active_backend is not None:
        for line in active_backend.startup_lines(args, adapter):
            print(line)
    print()

    await runner.run()


if __name__ == "__main__":
    from .cli import main as entrypoint
    entrypoint(["run", *__import__("sys").argv[1:]])
