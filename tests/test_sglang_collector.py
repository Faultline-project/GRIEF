"""Tests for SGLang collector: _SglangProbeSummary, _analyze_sglang_summary,
_SglangCollector lifecycle, _parse_prometheus, and SglangBaseAdapter hooks."""

import asyncio
import pytest

from fuzzer.adapters.sglang.legacy import (
    SglangBaseAdapter,
    _SglangCollector,
    _SglangProbeSummary,
    _analyze_sglang_summary,
    _parse_prometheus,
    _MIN_RELATIVE_GROWTH,
    _MIN_ABSOLUTE_GROWTH,
)
from fuzzer.core.collector import CollectionAnomaly


# ---------------------------------------------------------------------------
# _parse_prometheus
# ---------------------------------------------------------------------------

class TestParsePrometheus:

    def test_simple_gauge(self):
        text = "sglang:token_usage 0.42\n"
        result = _parse_prometheus(text)
        assert result["sglang:token_usage"] == pytest.approx(0.42)

    def test_labelled_metric(self):
        text = 'sglang:cache_hit_rate{model="qwen"} 0.75\n'
        result = _parse_prometheus(text)
        assert result["sglang:cache_hit_rate"] == pytest.approx(0.75)

    def test_comment_and_blank_lines_skipped(self):
        text = (
            "# HELP sglang:token_usage GPU token utilisation\n"
            "# TYPE sglang:token_usage gauge\n"
            "\n"
            "sglang:token_usage 0.55\n"
        )
        result = _parse_prometheus(text)
        assert list(result.keys()) == ["sglang:token_usage"]
        assert result["sglang:token_usage"] == pytest.approx(0.55)

    def test_last_value_wins_for_duplicate_name(self):
        text = (
            'sglang:x{a="1"} 1.0\n'
            'sglang:x{a="2"} 2.0\n'
        )
        result = _parse_prometheus(text)
        assert result["sglang:x"] == pytest.approx(2.0)

    def test_invalid_value_skipped(self):
        text = "sglang:bad_metric not_a_number\n"
        result = _parse_prometheus(text)
        assert "sglang:bad_metric" not in result

    def test_empty_input(self):
        assert _parse_prometheus("") == {}


# ---------------------------------------------------------------------------
# _analyze_sglang_summary — memory_growth annotation
# ---------------------------------------------------------------------------

class TestAnalyzeSglangSummary:

    def _summary(self, pre_info, post_info):
        return _SglangProbeSummary(
            pre_info=pre_info, post_info=post_info,
            pre_metrics=None, post_metrics=None,
        )

    def test_no_growth_returns_empty(self):
        s = self._summary({"num_tokens_used": 100}, {"num_tokens_used": 100})
        assert _analyze_sglang_summary(s) == []

    def test_capacity_field_ignored(self):
        """Fields with 'max' or 'limit' in name are constants, not counters."""
        s = self._summary(
            {"max_total_num_tokens": 65536},
            {"max_total_num_tokens": 65536},
        )
        assert _analyze_sglang_summary(s) == []

    def test_non_util_field_ignored(self):
        """Fields without token|mem|usage|cache in name are skipped."""
        s = self._summary({"tp_size": 1}, {"tp_size": 2})
        assert _analyze_sglang_summary(s) == []

    def test_below_relative_threshold_no_anomaly(self):
        """Growth < 5% relative should not fire even if absolute is large."""
        pre  = 10_000
        post = pre + int(pre * (_MIN_RELATIVE_GROWTH - 0.01))  # just under 5%
        s = self._summary({"num_tokens_used": pre}, {"num_tokens_used": post})
        assert _analyze_sglang_summary(s) == []

    def test_below_absolute_threshold_no_anomaly(self):
        """Growth < absolute floor should not fire even if relative is large."""
        pre  = 10   # tiny base so relative growth is easy to hit
        post = pre + (_MIN_ABSOLUTE_GROWTH - 1)  # just under floor
        s = self._summary({"num_tokens_used": pre}, {"num_tokens_used": post})
        assert _analyze_sglang_summary(s) == []

    def test_clear_growth_emits_anomaly(self):
        """Both thresholds met → exactly one memory_growth anomaly."""
        pre  = 1000
        post = 1200   # 20% growth, delta=200
        s = self._summary({"num_tokens_used": pre}, {"num_tokens_used": post})
        anomalies = _analyze_sglang_summary(s)
        assert len(anomalies) == 1
        a = anomalies[0]
        assert isinstance(a, CollectionAnomaly)
        assert a.anomaly_type == "memory_growth"
        assert a.score == pytest.approx(1.5)
        assert a.metadata["field"] == "num_tokens_used"
        assert a.metadata["delta"] == pytest.approx(200)

    def test_score_below_structural_repro_threshold(self):
        """Score must be < 5.0 — never triggers structural repro queue."""
        pre, post = 1000, 1200
        s = self._summary({"num_tokens_used": pre}, {"num_tokens_used": post})
        anomalies = _analyze_sglang_summary(s)
        assert all(a.score < 5.0 for a in anomalies)

    def test_at_most_one_anomaly_even_with_multiple_growing_fields(self):
        """Only the single largest delta is reported."""
        s = self._summary(
            {"num_tokens_used": 1000, "mem_usage": 500},
            {"num_tokens_used": 1200, "mem_usage": 700},   # both grew
        )
        anomalies = _analyze_sglang_summary(s)
        assert len(anomalies) == 1

    def test_none_info_returns_empty(self):
        s = _SglangProbeSummary(pre_info=None, post_info=None,
                                pre_metrics=None, post_metrics=None)
        assert _analyze_sglang_summary(s) == []

    def test_decrease_returns_empty(self):
        """Shrinking utilization is not an anomaly."""
        s = self._summary({"num_tokens_used": 1200}, {"num_tokens_used": 1000})
        assert _analyze_sglang_summary(s) == []


# ---------------------------------------------------------------------------
# _SglangProbeSummary.to_dict
# ---------------------------------------------------------------------------

class TestSglangProbeSummaryToDict:

    def test_roundtrip(self):
        s = _SglangProbeSummary(
            pre_info={"version": "0.5"}, post_info={"version": "0.5"},
            pre_metrics={"sglang:x": 1.0}, post_metrics={"sglang:x": 1.1},
        )
        d = s.to_dict()
        assert d["pre_info"]  == {"version": "0.5"}
        assert d["pre_metrics"]["sglang:x"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _SglangCollector lifecycle (unit — no real HTTP)
# ---------------------------------------------------------------------------

class TestSglangCollectorLifecycle:

    def _collector_with_mock(self, server_info_seq, metrics_text=None):
        """Return a _SglangCollector with mocked HTTP calls."""
        import unittest.mock as mock

        collector = _SglangCollector(base_url="http://mock:8000")
        collector._metrics_ok = metrics_text is not None
        info_iter = iter(server_info_seq)

        async def _mock_server_info():
            return next(info_iter, None)

        async def _mock_metrics():
            return _parse_prometheus(metrics_text) if metrics_text else None

        collector._fetch_server_info = _mock_server_info
        collector._fetch_metrics     = _mock_metrics
        return collector

    def test_end_probe_returns_summary(self):
        pre  = {"num_tokens_used": 100}
        post = {"num_tokens_used": 120}
        c = self._collector_with_mock([pre, post])
        asyncio.run(c.begin_probe())
        summary = asyncio.run(c.end_probe())
        assert isinstance(summary, _SglangProbeSummary)
        assert summary.pre_info  == pre
        assert summary.post_info == post

    def test_last_summary_property(self):
        c = self._collector_with_mock([{"num_tokens_used": 50},
                                        {"num_tokens_used": 60}])
        assert c.last_summary is None   # before first probe
        asyncio.run(c.begin_probe())
        asyncio.run(c.end_probe())
        assert c.last_summary is not None

    def test_forensics_snapshot_none_before_probe(self):
        c = _SglangCollector(base_url="http://mock:8000")
        assert c.forensics_snapshot() is None

    def test_forensics_snapshot_returns_dict_after_probe(self):
        c = self._collector_with_mock([{"num_tokens_used": 100},
                                        {"num_tokens_used": 110}])
        asyncio.run(c.begin_probe())
        asyncio.run(c.end_probe())
        snap = c.forensics_snapshot()
        assert isinstance(snap, dict)
        assert "pre_info" in snap and "post_info" in snap

    def test_metrics_captured_when_available(self):
        metrics = "sglang:token_usage 0.30\n"
        c = self._collector_with_mock(
            [{"num_tokens_used": 100}, {"num_tokens_used": 110}],
            metrics_text=metrics,
        )
        asyncio.run(c.begin_probe())
        summary = asyncio.run(c.end_probe())
        assert summary.pre_metrics  is not None
        assert summary.post_metrics is not None
        assert "sglang:token_usage" in summary.pre_metrics

    def test_metrics_none_when_not_available(self):
        c = self._collector_with_mock([{"num_tokens_used": 100},
                                        {"num_tokens_used": 110}],
                                       metrics_text=None)
        asyncio.run(c.begin_probe())
        summary = asyncio.run(c.end_probe())
        assert summary.pre_metrics  is None
        assert summary.post_metrics is None


# ---------------------------------------------------------------------------
# SglangBaseAdapter.make_collector / analyze_collection integration
# ---------------------------------------------------------------------------

class TestSglangAdapterCollectorHooks:

    def _config(self, base_url="http://mock:8000"):
        import dataclasses
        from fuzzer.core.runner import RunnerConfig
        return RunnerConfig(base_url=base_url)

    def test_make_collector_returns_sglang_collector(self):
        a = SglangBaseAdapter()
        c = a.make_collector(self._config())
        assert isinstance(c, _SglangCollector)

    def test_analyze_collection_none_returns_empty(self):
        a = SglangBaseAdapter()
        assert a.analyze_collection(None) == []

    def test_analyze_collection_passes_through_to_analyzer(self):
        a = SglangBaseAdapter()
        summary = _SglangProbeSummary(
            pre_info={"num_tokens_used": 1000},
            post_info={"num_tokens_used": 1250},   # 25% growth, delta=250
            pre_metrics=None, post_metrics=None,
        )
        anomalies = a.analyze_collection(summary)
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "memory_growth"
        assert anomalies[0].score < 5.0
