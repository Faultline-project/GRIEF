"""Regression tests for the three production defects fixed on `layer12`.

Each test here exists because a specific line of production code destroyed data
or produced a wrong answer, and nothing in the suite noticed.  The oracle and
KV-events halves live with their subjects (`test_oracles.py::TestFatalErrorOracle`,
`test_kv_events.py::TestKVProbeSummarySerializable`); this file holds the
corpus and monitor halves, which have no natural home in a file that is being
trimmed concurrently.
"""

import dataclasses
import json

import pytest

from fuzzer.core.corpus import Corpus, CorpusEntry
from fuzzer.monitor import (
    ERR_CLASS_4XX,
    ERR_CLASS_5XX,
    ERR_CLASS_CONN,
    ERR_CLASS_MIXED,
    ERR_CLASS_NONE,
    SchedulerFingerprint,
    error_class_bucket,
)
from fuzzer.trace import Trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fingerprint(**overrides) -> SchedulerFingerprint:
    """A SchedulerFingerprint with every dimension zeroed unless overridden.

    Built positionally-free (by field name) so adding a dimension does not
    break this file.
    """
    values = {f.name: 0 for f in dataclasses.fields(SchedulerFingerprint)}
    values.update(overrides)
    return SchedulerFingerprint(**values)


@pytest.fixture
def corpus(tmp_path):
    seeds = tmp_path / "seeds"
    corpus_dir = tmp_path / "corpus"
    seeds.mkdir()
    corpus_dir.mkdir()
    return Corpus(seeds_dir=str(seeds), corpus_dir=str(corpus_dir))


def _saved(corpus_dir, seed: int) -> dict:
    return json.loads((corpus_dir / f"seed_{seed}.json").read_text())


class _Req:
    """Minimal stand-in for RequestResult — error_class_bucket reads `.error`."""

    def __init__(self, error=None):
        self.error = error


# ---------------------------------------------------------------------------
# Defect 3 — Corpus.boost() used to null out _corpus_peak_swapped
# ---------------------------------------------------------------------------

class TestBoostPreservesTelemetry:
    """`boost()` re-saves an entry after soft-repro confirms it.

    It called ``self._save(entry)`` with no ``peak_swapped``, and the parameter
    defaults to ``None`` — so the re-save wrote ``"_corpus_peak_swapped": null``
    straight over a real value.  ``load_seeds`` prepopulates
    ``_seen_swap_counts`` from that field, so every boosted entry silently
    stopped contributing to swap novelty on the next campaign's reload — and
    boosting is exactly what marks an entry as *confirmed interesting*.
    """

    def test_boost_does_not_null_out_peak_swapped(self, corpus, tmp_path):
        fp = _fingerprint(had_swap=1)
        entry = CorpusEntry(trace=Trace(seed=7, events=[]), score=1.0,
                            fingerprint=fp)
        corpus._entries.append(entry)
        corpus._save(entry, peak_swapped=12)
        assert _saved(tmp_path / "corpus", 7)["_corpus_peak_swapped"] == 12

        assert corpus.boost(fp) is True

        data = _saved(tmp_path / "corpus", 7)
        assert data["_corpus_peak_swapped"] == 12   # was None before the fix
        assert data["_corpus_score"] == 3.0         # the boost still applied

    def test_boost_does_not_null_out_preemption_delta(self, corpus, tmp_path):
        """The sibling field is written from the entry and had the same hole
        once the entry came back from disk (see the reload test below)."""
        fp = _fingerprint(had_preemption=1)
        entry = CorpusEntry(trace=Trace(seed=9, events=[]), score=1.0,
                            fingerprint=fp, preemption_delta=4)
        corpus._entries.append(entry)
        corpus._save(entry)

        corpus.boost(fp)

        assert _saved(tmp_path / "corpus", 9)["_corpus_preemption_delta"] == 4

    def test_add_records_peak_swapped_on_the_entry(self, corpus, tmp_path):
        """add() is the normal producer; the value has to land on the entry,
        not only in the _save() call, or the next re-save loses it."""
        class _Session:
            def preemption_delta(self):
                return 3

            def peak_swapped(self):
                return 6

        fp = _fingerprint(had_swap=1, queue_depth_bucket=2)
        # pressure_score short-circuits _compute_pressure, which wants the full
        # MonitorSession surface; only the two telemetry getters matter here.
        corpus.add(Trace(seed=11, events=[]), findings=[], fingerprint=fp,
                   session=_Session(), pressure_score=0.0)

        entry = corpus._entries[-1]
        assert entry.peak_swapped == 6
        assert entry.preemption_delta == 3
        assert _saved(tmp_path / "corpus", 11)["_corpus_peak_swapped"] == 6

    def test_swap_counts_survive_a_reload_then_boost_cycle(self, corpus, tmp_path):
        """The end-to-end shape of the bug: campaign 1 saves, campaign 2
        reloads and boosts, campaign 3 must still see the swap count."""
        fp = _fingerprint(had_swap=1)
        entry = CorpusEntry(trace=Trace(seed=13, events=[]), score=2.0,
                            fingerprint=fp, preemption_delta=5)
        corpus._entries.append(entry)
        corpus._save(entry, peak_swapped=9)

        reloaded = Corpus(seeds_dir=str(tmp_path / "seeds"),
                          corpus_dir=str(tmp_path / "corpus"))
        reloaded.load_seeds()
        # Reload must carry the telemetry onto the rebuilt entry, not merely
        # consume it into the counters — _save() persists it *from the entry*.
        assert reloaded._seen_swap_counts[9] == 1
        assert reloaded._entries[0].peak_swapped == 9
        assert reloaded._entries[0].preemption_delta == 5

        reloaded.boost(fp)

        data = _saved(tmp_path / "corpus", 13)
        assert data["_corpus_peak_swapped"] == 9
        assert data["_corpus_preemption_delta"] == 5

        third = Corpus(seeds_dir=str(tmp_path / "seeds"),
                       corpus_dir=str(tmp_path / "corpus"))
        third.load_seeds()
        assert third._seen_swap_counts[9] == 1

    def test_explicit_peak_swapped_still_wins_over_the_stored_one(self, corpus,
                                                                 tmp_path):
        """The fallback must not shadow a caller that does pass a value."""
        entry = CorpusEntry(trace=Trace(seed=15, events=[]), score=1.0,
                            peak_swapped=2)
        corpus._save(entry, peak_swapped=8)
        assert _saved(tmp_path / "corpus", 15)["_corpus_peak_swapped"] == 8
        assert entry.peak_swapped == 8


# ---------------------------------------------------------------------------
# monitor._classify_error — the dead _CONN_MARKERS branch
# ---------------------------------------------------------------------------

class TestErrorClassBucket:
    """`_classify_error` used to test the message against a `_CONN_MARKERS`
    tuple and `return ERR_CLASS_CONN`, then `return ERR_CLASS_CONN` anyway —
    the guard could not change the outcome for any input.  The branch is gone;
    these pin the behaviour it was supposed to (and did) produce, so the
    simplification is observable rather than assumed.
    """

    def test_no_errors_is_class_none(self):
        assert error_class_bucket([_Req(), _Req()]) == ERR_CLASS_NONE

    def test_recognised_and_unrecognised_transport_errors_agree(self):
        """The deleted guard's two sides: a message that matched a marker and
        one that matched nothing.  Both are transport failures — 'no HTTP
        status came back' is the definition, not the spelling."""
        recognised = error_class_bucket([_Req("[Errno 104] Connection reset by peer")])
        unrecognised = error_class_bucket([_Req("RuntimeError: something odd")])
        assert recognised == ERR_CLASS_CONN
        assert unrecognised == ERR_CLASS_CONN
        # ...so a trace with one of each is *not* "mixed".
        assert error_class_bucket([
            _Req("[Errno 104] Connection reset by peer"),
            _Req("RuntimeError: something odd"),
        ]) == ERR_CLASS_CONN

    def test_http_families_are_still_distinguished(self):
        assert error_class_bucket(
            [_Req("Client error '400 Bad Request' for url 'http://h/v1'")]
        ) == ERR_CLASS_4XX
        assert error_class_bucket(
            [_Req("Server error '503 Service Unavailable' for url 'http://h/v1'")]
        ) == ERR_CLASS_5XX
        assert error_class_bucket([
            _Req("Client error '400 Bad Request' for url 'http://h/v1'"),
            _Req("ConnectError: all connection attempts failed"),
        ]) == ERR_CLASS_MIXED
