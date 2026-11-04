"""Tests for speculative decoding oracles."""
import pytest

from fuzzer.adapters.vllm_spec.oracles import SpecCandidate, compare_outputs
from fuzzer.core.types import TraceResult, RequestResult
from fuzzer.trace import Trace


def _req(request_id, output_text, tokens_received=None, cancelled_at=None, error=None, completed_at=1.0):
    return RequestResult(
        request_id      = request_id,
        sent_at         = 0.0,
        completed_at    = completed_at,
        cancelled_at    = cancelled_at,
        output_text     = output_text,
        tokens_received = tokens_received if tokens_received is not None else (len(output_text.split()) if output_text else 0),
        error           = error,
    )


def _result(*reqs):
    return TraceResult(trace_seed=0, started_at=0.0, finished_at=1.0, requests=list(reqs))


class TestCompareOutputs:
    def test_diverged_pair_returned(self):
        spec   = _result(_req("r1", "hello world"))
        nospec = _result(_req("r1", "hello there"))
        pairs = compare_outputs(spec, nospec)
        assert len(pairs) == 1
        assert pairs[0]["request_id"] == "r1"
        assert pairs[0]["spec_output"] == "hello world"
        assert pairs[0]["nospec_output"] == "hello there"

    def test_matching_outputs_empty(self):
        spec   = _result(_req("r1", "hello world"))
        nospec = _result(_req("r1", "hello world"))
        assert compare_outputs(spec, nospec) == []

    def test_first_divergence_pos_tracked(self):
        spec   = _result(_req("r1", "abcXef"))
        nospec = _result(_req("r1", "abcYef"))
        pairs = compare_outputs(spec, nospec)
        assert pairs[0]["first_divergence_pos"] == 3

    def test_first_divergence_pos_when_one_is_prefix(self):
        spec   = _result(_req("r1", "abc"))
        nospec = _result(_req("r1", "abcde"))
        pairs = compare_outputs(spec, nospec)
        # first mismatch at position 3 (end of shorter string)
        assert pairs[0]["first_divergence_pos"] == 3

    def test_skips_cancelled_spec(self):
        spec   = _result(_req("r1", "hello", cancelled_at=0.5, completed_at=None))
        nospec = _result(_req("r1", "world"))
        assert compare_outputs(spec, nospec) == []

    def test_skips_cancelled_nospec(self):
        spec   = _result(_req("r1", "hello"))
        nospec = _result(_req("r1", "world", cancelled_at=0.5, completed_at=None))
        assert compare_outputs(spec, nospec) == []

    def test_skips_error_spec(self):
        spec   = _result(_req("r1", None, error="timeout"))
        nospec = _result(_req("r1", "world"))
        assert compare_outputs(spec, nospec) == []

    def test_skips_error_nospec(self):
        spec   = _result(_req("r1", "hello"))
        nospec = _result(_req("r1", None, error="500"))
        assert compare_outputs(spec, nospec) == []

    def test_unmatched_request_id_skipped(self):
        spec   = _result(_req("r1", "hello"))
        nospec = _result(_req("r2", "world"))
        assert compare_outputs(spec, nospec) == []

    def test_none_output_text_treated_as_empty(self):
        spec   = _result(_req("r1", None, tokens_received=0))
        nospec = _result(_req("r1", None, tokens_received=0))
        assert compare_outputs(spec, nospec) == []
