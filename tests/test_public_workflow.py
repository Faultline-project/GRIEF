"""Public installation, event stream and replay semantics."""
import json
from pathlib import Path

import httpx
import pytest

from fuzzer.events import EventReader, EventWriter
from fuzzer.replay import judge_replay, load_finding, replay
from fuzzer.preflight import check_server
from fuzzer.demo import demo_events
from fuzzer.trace import Trace
from fuzzer.core.types import OracleResult, BugClass


def test_event_tail_waits_for_partial_line(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(str(path))
    reader = EventReader(path)
    writer.emit("run_started", model="test")
    assert len(reader.read()) == 1
    with path.open("ab") as stream:
        stream.write(b'{"schema_version":1,"type":"test",')
    assert reader.read() == []
    with path.open("ab") as stream:
        stream.write(b'"data":{}}\nnot-json\n')
    assert [e["type"] for e in reader.read()] == ["test"]
    assert reader.errors == 1
    assert reader.read() == []


def test_divergence_is_never_confirmed_by_contract_replay():
    trace = Trace.from_dict(demo_events()[2]["data"]["trace"])
    report = {"bugs": [{"class": "kv_corruption"}]}
    bug = OracleResult(False, BugClass.KV_CORRUPTION, "output mismatch", 8)
    assert judge_replay(report, trace, [bug])["verdict"] == "inconclusive"


def test_loader_preserves_original_trace_choice(tmp_path):
    trace = demo_events()[2]["data"]["trace"]
    short = {**trace, "events": trace["events"][:1]}
    path = tmp_path / "finding.json"
    path.write_text(json.dumps({"trace": short, "trace_original": trace}))
    assert len(load_finding(path)[1].events) == 1
    assert len(load_finding(path, True)[1].events) == 5


@pytest.fixture
def inference_transport(monkeypatch):
    """Exercise real HTTP payload construction and oracles, with synthetic replies."""
    requests = []
    def handle(req):
        if req.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fixture-model"}]})
        if req.method == "GET":
            return httpx.Response(200, json={"version": "fixture", "status": "ok"})
        data = json.loads(req.content)
        requests.append(data)
        text = '{"count": 3' if data.get("response_format") else "ok"
        if data.get("stream"):
            chunk = {"choices": [{"text": text, "finish_reason": "stop"}]}
            return httpx.Response(200, content=("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"text": text, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 2, "prompt_tokens": 8}})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: client(*a, transport=httpx.MockTransport(handle), **kw))
    return requests


async def test_preflight_rejects_unknown_model(inference_transport):
    assert (await check_server("http://fixture"))["model"] == "fixture-model"
    with pytest.raises(ValueError, match="not served"):
        await check_server("http://fixture", "missing")


async def test_run_writes_events_and_detects_contract(inference_transport, tmp_path):
    from fuzzer.__main__ import main
    seeds = tmp_path / "seeds"
    seeds.mkdir()
    trace = {"seed": 42, "events": [{"action": "send", "offset_ms": 0, "request_id": "victim",
        "prompt_len": 8, "max_tokens": 8, "stream": False, "response_format": {"type": "json_object"}}]}
    (seeds / "contract.json").write_text(json.dumps(trace))
    events = tmp_path / "events.jsonl"
    await main(["--base-url", "http://fixture", "--seeds-dir", str(seeds),
        "--findings-dir", str(tmp_path / "findings"), "--corpus-dir", str(tmp_path / "corpus"),
        "--events-file", str(events), "--mock-monitor", "--max-iterations", "1",
        "--mutations-per-seed", "0", "--no-minimize", "--probe-determinism-every", "0"])
    records = EventReader(events).read()
    assert records[0]["type"] == "run_started"
    assert records[-1]["type"] == "run_finished"
    saved = [e for e in records if e["type"] == "finding_saved"]
    assert saved
    assert "schema_violation" in [b["class"] for b in saved[0]["data"]["finding"]["bugs"]]
    assert any(r.get("model") == "fixture-model" for r in inference_transport)


async def test_replay_uses_same_schema_oracle(inference_transport, tmp_path):
    from fuzzer.__main__ import parse_args
    path = tmp_path / "finding.json"
    path.write_text(json.dumps({"bugs": [{"class": "schema_violation"}], "trace": {"seed": 1,
        "events": [{"action": "send", "offset_ms": 0, "request_id": "v", "prompt_len": 8,
                    "max_tokens": 8, "stream": False, "response_format": {"type": "json_object"}}]}}))
    result = await replay(path, parse_args(["--base-url", "http://fixture", "--mock-monitor"]), runs=2)
    assert [r["verdict"] for r in result["runs"]] == ["same_class_observed"] * 2
    assert "schema_violation" in result["runs"][0]["observed_classes"]
