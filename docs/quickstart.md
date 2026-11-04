# Running GRIEF

## Offline walkthrough

Install `.[tui]` and run `grief demo`. The synthetic session illustrates a
shared prefix, concurrent pressure, cancellation, retry and a JSON contract
violation. It never contacts a server. `grief demo --speed 2` doubles playback
speed. Live events use the same interface.

## Connect to an inference engine

Run a vLLM or SGLang server with an OpenAI-compatible endpoint. GRIEF expects
the origin URL (`http://host:port`), without a trailing `/v1`. The server must
advertise at least one model at `/v1/models`; use `--model-name` to select one.
The initial public client does not expose an API-key authentication option.

Start with a short run on a dedicated test server:

```bash
grief run --base-url http://localhost:8000 --max-iterations 10 --no-minimize
```

Server installation and launch flags depend on its version. The source
artifact used vLLM 0.27.1 and 0.19.0; these are provenance information, not a
claim that every feature works on every version. Check the installed server's
help for prefix caching, KV publication, structured outputs, and feature flags.
The public refactor's local validation uses CPU tests and synthetic HTTP
responses; a new real-GPU engine validation matrix has not yet been recorded.

## Features and telemetry

| Client option | Server prerequisite |
| --- | --- |
| `--lora` | vLLM LoRA enabled, matching adapter names and weights |
| `--spec` | vLLM speculative decoding configured |
| `--moe` | Suitable MoE model and confirmation configuration |
| `--embeds --prompt-embeds-dim N` | vLLM prompt embeddings enabled, correct hidden size |
| `--sglang` | SGLang endpoint |
| `--sglang-lora`, `--sglang-spec`, `--sglang-moe` | Corresponding SGLang feature and model setup |
| `--kv-events-endpoint tcp://host:5557` | vLLM KV-event publication, client `.[telemetry]` |
| `--local-gpu-metrics` | Client on the server's GPU host, client `.[telemetry]` |

Use `grief run --help` for the adapter-specific parameters. The client reports
unmeasured oracle conditions; absent telemetry is not evidence that a check
passed. Avoid enabling feature adapters against servers that do not implement
their request fields.

For custom seed files, pass `--seeds-dir DIR`. A seed is a JSON object with a
`seed` and an `events` array. Each event has an `action`, `offset_ms` and
`request_id`; SEND events also carry request parameters. The bundled examples
are in `fuzzer/data/seeds/`.

## Stopping and inspecting a run

Press Ctrl-C in the CLI or `s` in the TUI. Saved findings remain on disk, and
normal shutdown writes signature and confirmation summaries. The TUI waits
for its worker to exit before closing. Watching a saved directory never starts
requests against the recorded server.

`grief inspect RUN` returns a JSON summary. `grief inspect FINDING.json` returns
the saved report. To replay a reduced finding, use `grief replay FINDING`; add
`--original` to replay `trace_original` when present. Pass the original engine
and feature flags, especially `--sglang` and LoRA/spec/MoE options.
