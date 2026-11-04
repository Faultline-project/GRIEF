# Understanding results

An oracle observation, a signature exemplar, and a confirmed mechanism are
different things. GRIEF retains the supporting trace and evidence so that you
can inspect the distinction.

| Record | Meaning |
| --- | --- |
| Candidate observation | A check fired on an execution; further checks may reject it |
| Saved finding | A report survived its configured reporting path |
| Mechanism signature | A deduplication key over symptoms and trace features; not a root-cause proof |
| Confirmation verdict | The outcome of the configured controlled replay pipeline |
| Reduced-trace validation | Whether the stored reduced trace was rechecked, and what survived |

## Files

`findings/finding_*.json` stores the trace, original trace when reduced, oracle
results, signature, engine/model metadata and reduced-trace validation.
`signature_ledger.json` counts repeats and exemplars. `confirmation_log.jsonl`
and `confirmation_summary.json` record the batch-invariant ladder.
`latency_candidates.jsonl` records the latency path. `--log-reports` adds
`reports.jsonl`. These are runtime outputs and are ignored by Git.

## Contract replay

`grief replay` executes the stored trace through the same executor and adapter
oracles. Its per-run verdicts are deliberately narrow:

- `signature_reproduced`: the same mechanism signature recurred.
- `same_class_observed`: the oracle class recurred, but signature identity was
  absent or different.
- `not_reproduced`: that check did not reproduce in this replay.
- `inconclusive`: controlled comparisons are required, or execution failed.
- `executed`: a raw trace was run without an expected finding.

A contract replay cannot establish output isolation from a text mismatch,
nor reproduce a calibrated latency baseline merely by sending requests again.
The original report is never overwritten by replay.

## Controlled confirmation

Output divergence may reflect ordinary numerical or scheduling variation.
Enable the existing ladder for a live campaign with a separately configured
batch-invariant reference server:

```bash
grief run --base-url http://localhost:8000 \
  --confirm-ladder --confirm-bi-url http://localhost:8001 \
  --confirm-bi-log /path/to/reference-server.log
```

The reference server must be launched with `VLLM_BATCH_INVARIANT=1` and a
supported backend/GPU configuration. The guard checks engagement evidence
and refuses to run if the reference cannot be validated. A second URL alone
does not prove batch invariance. Keep model and relevant server settings
matched. Consult `fuzzer/confirm/bi_guard.py` for the explicit allowlist.

To target a saved workload, put its `trace_original` (or `trace`) object into
a dedicated seed JSON and pass `--seeds-dir DIR --mutations-per-seed 0` with
the same adapter and confirmation flags. Results remain tied to that server
configuration; this is controlled workload replay, not deterministic replay of
internal scheduler interleavings.

## Minimization

The reducer searches for a smaller trace that still fires a related oracle.
Enable `--reverify-minimized` to execute the final reduced trace again and
record whether its mechanism signature matches. The original oracle evidence
is retained even if the reduced trace fails verification. A reduced trace
marked `not_rechecked` must not be presented as a verified reproducer.
