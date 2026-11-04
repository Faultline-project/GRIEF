"""
SGLang speculative decoding oracle helpers.

Per-request spec toggle fields do not exist in the SGLang OpenAI-compatible
API.  Detection relies on existing behavioral oracles.  Confirmation compares
outputs from a spec-enabled SGLang server against a no-spec SGLang server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...trace import Trace


@dataclass
class SpecCandidate:
    trace:      "Trace"
    iteration:  int
    sig:        str
    timestamp:  float
    run_count:  int = 0


def compare_outputs(spec_result, nospec_result) -> list[dict]:
    """Compare two TraceResults request-by-request.

    Matches on request_id.  Skips pairs where either side has an error or
    was cancelled.  Returns only diverged pairs.

    Each returned dict has:
        request_id, spec_output, nospec_output, first_divergence_pos
    """
    spec_by_id   = {r.request_id: r for r in spec_result.requests}
    nospec_by_id = {r.request_id: r for r in nospec_result.requests}

    diverged = []
    for rid, spec_req in spec_by_id.items():
        nospec_req = nospec_by_id.get(rid)
        if nospec_req is None:
            continue
        if spec_req.cancelled_at is not None or nospec_req.cancelled_at is not None:
            continue
        if spec_req.error is not None or nospec_req.error is not None:
            continue
        spec_text   = spec_req.output_text   or ""
        nospec_text = nospec_req.output_text or ""
        if spec_text == nospec_text:
            continue
        first_div = next(
            (i for i, (a, b) in enumerate(zip(spec_text, nospec_text)) if a != b),
            min(len(spec_text), len(nospec_text)),
        )
        diverged.append({
            "request_id":           rid,
            "spec_output":          spec_text,
            "nospec_output":        nospec_text,
            "first_divergence_pos": first_div,
        })
    return diverged
