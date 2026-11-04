# GRIEF architecture

The unit of exploration is a timed, concurrent client workload. Send, Cancel,
Disconnect, Wait and Reset events specify when requests overlap and how their
lifecycles interact. An engine's scheduler determines the resulting internal
interleavings.

The core selects a corpus trace, mutates its timing, parameters or request
relationships, executes it, evaluates oracles and retains interesting inputs.
Scheduler fingerprints, KV observations and oracle scores influence selection.
Adapters provide engine and feature-specific mutations and checks.

| Module | Responsibility |
| --- | --- |
| `fuzzer/trace.py` | Trace and event serialization |
| `fuzzer/core/runner.py` | Search, probes, confirmation, finding persistence |
| `fuzzer/core/corpus.py` | Feedback, novelty and parent selection |
| `fuzzer/core/mutator.py` | Timing and lifecycle operators, splicing |
| `fuzzer/executor.py` | Concurrent HTTP request execution |
| `fuzzer/adapters/` | vLLM and SGLang feature behavior |
| `fuzzer/confirm/` | Batch-invariant guards and controlled comparisons |
| `fuzzer/replay.py` | Contract replay using the same execution path |
| `fuzzer/events.py` | Versioned append-only run events |
| `fuzzer/tui/` | Terminal workspace and worker lifecycle |

The TUI launches the headless client as a child process and incrementally reads
`events.jsonl`. It does not parse human log messages to infer counts. Events
include a schema version, run ID, sequence number, elapsed time, type and
payload. A reader waits for complete lines before decoding them.

UI refreshes do not participate in random selection. Candidate observations,
confirmation outcomes and saved finding exemplars are separate event types.
The timeline distinguishes scheduled event points from observed request
lifetimes. Missing data remains unavailable rather than becoming a zero.

The public package includes the tool and regression tests. Experimental search
conditions, benchmark matchers, campaign matrices and site-specific cluster
scripts belong to the separate research artifact. The default serving-state
guided search remains intact.
