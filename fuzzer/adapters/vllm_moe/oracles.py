"""
MoE batch isolation oracle helpers.

Core oracle: at temperature=0, request R's output should be independent of
what else is in the batch.  If output(R in batch) != output(R alone), that
proves batch-sensitive serving behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...trace import Trace


@dataclass
class MoECandidate:
    trace:     "Trace"
    iteration: int
    sig:       str
    timestamp: float


def compare_batch_vs_solo(batch_result, solo_results: dict) -> list[dict]:
    """Compare per-request outputs from a batch run vs per-request solo runs.

    Parameters
    ----------
    batch_result : TraceResult
        Result from running the full multi-request trace.
    solo_results : dict[str, TraceResult]
        Mapping of request_id -> TraceResult from running each request alone.

    Returns
    -------
    list[dict]
        Diverged pairs: {request_id, batch_output, solo_output}.
        Empty list if no divergences found.

    Skips cancelled or errored requests on either side.
    """
    batch_by_id = {r.request_id: r for r in batch_result.requests}

    diverged = []
    for rid, solo_trace_result in solo_results.items():
        batch_req = batch_by_id.get(rid)
        if batch_req is None:
            continue

        # Each solo TraceResult has exactly one request
        solo_reqs = [r for r in solo_trace_result.requests if r.request_id == rid]
        if not solo_reqs:
            # Fallback: take the first request (solo trace only has one anyway)
            if not solo_trace_result.requests:
                continue
            solo_req = solo_trace_result.requests[0]
        else:
            solo_req = solo_reqs[0]

        # Skip cancelled or errored requests on either side
        if batch_req.cancelled_at is not None or solo_req.cancelled_at is not None:
            continue
        if batch_req.error is not None or solo_req.error is not None:
            continue

        batch_text = batch_req.output_text or ""
        solo_text  = solo_req.output_text  or ""

        if batch_text == solo_text:
            continue

        diverged.append({
            "request_id":   rid,
            "batch_output": batch_text,
            "solo_output":  solo_text,
        })

    return diverged
