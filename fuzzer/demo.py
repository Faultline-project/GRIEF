"""A deliberately simulated teaching session. Never presented as an engine finding."""
from __future__ import annotations

from .core.finding_sig import mechanism_signature
from .core.types import BugClass, OracleResult
from .trace import Trace, Event, ActionType


def demo_events():
    trace = Trace(seed=42, events=[
        Event(ActionType.SEND, 0, request_id="warm", prompt_len=256, prefix_len=192, max_tokens=48),
        Event(ActionType.SEND, 70, request_id="victim", prompt_len=256, prefix_len=192,
              max_tokens=64, temperature=0.0, response_format={"type": "json_object"}),
        Event(ActionType.SEND, 110, request_id="pressure", prompt_len=2048, max_tokens=128),
        Event(ActionType.CANCEL, 270, request_id="pressure"),
        Event(ActionType.SEND, 310, request_id="retry", prompt_len=256, prefix_len=192, max_tokens=48),
    ])
    bug = OracleResult(False, BugClass.SCHEMA_VIOLATION,
        "Simulated example: a request for a JSON object returned invalid JSON.", 8.0,
        {"request_id": "victim", "expected": "JSON object", "observed": '{"count": 3'})
    finding = {"demo": True, "trace": trace.to_dict(), "bugs": [
        {"class": bug.bug_class.value, "detail": bug.detail, "score": bug.score,
         "probe_metadata": bug.probe_metadata}], "mech_sig": mechanism_signature(trace, [bug]),
        "minimized": False, "minimized_validation": {"status": "not_minimized"},
        "backend": "simulated", "model": "teaching fixture"}
    records = [
        ("run_started", {"engine": "SIMULATED", "config": {"base_url": "offline",
            "model_name": "Teaching fixture · no inference", "max_iterations": 3}}),
        ("narration", {"text": "A request can behave differently when it shares serving state with its neighbours."}),
        ("trace_started", {"iteration": 0, "trace": trace.to_dict(), "mutations": ["shared_prefix", "inject_cancel"], "corpus": 6}),
        ("narration", {"text": "Shared prefixes, overlapping lifetimes, cancellation, then retry."}),
        ("iteration_finished", {"iteration": 0, "corpus": 7, "novel": True, "candidates": 0,
            "fingerprint": "new scheduler / cache state (simulated)", "measurement_gaps": [],
            "requests": [{"request_id": rid, "sent_ms": start, "end_ms": end,
                          "tokens": 32, "cancelled": rid == "pressure", "error": None}
                         for rid, start, end in [("warm", 0, 480), ("victim", 70, 620),
                                                 ("pressure", 110, 270), ("retry", 310, 740)]]}),
        ("narration", {"text": "Serving-state feedback retains this trace for further mutation."}),
        ("trace_started", {"iteration": 1, "trace": trace.to_dict(), "mutations": ["jitter_timing", "change_prefix_overlap"], "corpus": 7}),
        ("iteration_finished", {"iteration": 1, "corpus": 8, "novel": True, "candidates": 1,
            "fingerprint": "reused prefix + concurrent pressure (simulated)", "measurement_gaps": [],
            "requests": [{"request_id": rid, "sent_ms": start, "end_ms": end, "tokens": 24, "error": None}
                         for rid, start, end in [("warm", 0, 460), ("victim", 70, 580), ("pressure", 110, 270), ("retry", 310, 700)]], "bugs": finding["bugs"]}),
        ("finding_saved", {"path": "simulated-finding.json", "finding": finding}),
        ("narration", {"text": "The finding records the input and evidence. One signature is not a root-cause count."}),
        ("confirmation", {"verdict": "SIMULATED_CONTRACT_REPLAY", "reason": "Teaching example: the JSON contract violation recurs. No real server was contacted."}),
        ("narration", {"text": "Output divergence needs controlled confirmation. A numerical mismatch alone is not proof of corruption."}),
        ("run_finished", {"status": "completed", "signatures": 1}),
        ("narration", {"text": "Try grief tui --base-url http://localhost:8000 with your own inference server."}),
    ]
    return [{"schema_version": 1, "run_id": "teaching-demo", "seq": i + 1,
             "elapsed_s": i * 3.0, "type": kind, "data": data}
            for i, (kind, data) in enumerate(records)]
