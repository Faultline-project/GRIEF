"""Tests for fuzzer/corpus.py — no GPU or server required."""

import json
import random
import tempfile
from pathlib import Path

import pytest

from fuzzer.trace import ActionType, Event, Trace
from fuzzer.oracles import OracleResult, BugClass
from fuzzer.corpus import Corpus, CorpusEntry
from fuzzer.monitor import SchedulerFingerprint


def make_fp(bucket: int = 0, preemption: bool = False, cpu_swap: bool = False,
            kv_peak_bucket: int = 0, preemption_count_bucket: int = 0,
            memory_delta_bucket: int = 0, finished_count_bucket: int = 0,
            prefix_cache_hit_bucket: int = 0) -> SchedulerFingerprint:
    return SchedulerFingerprint(
        had_preemption=preemption,
        had_swap=cpu_swap,
        queue_depth_bucket=bucket,
        kv_peak_bucket=kv_peak_bucket,
        preemption_count_bucket=preemption_count_bucket,
        memory_delta_bucket=memory_delta_bucket,
        finished_count_bucket=finished_count_bucket,
        prefix_cache_hit_bucket=prefix_cache_hit_bucket,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trace(seed: int = 1) -> Trace:
    return Trace(
        seed=seed,
        events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=64, max_tokens=32)
        ],
    )


def passing_result() -> list[OracleResult]:
    return [OracleResult(passed=True, score=0.0)]


def failing_result(score: float = 5.0) -> list[OracleResult]:
    return [OracleResult(passed=False, bug_class=BugClass.MEMORY_LEAK,
                         score=score)]


def make_corpus(seeds_dir: str, corpus_dir: str) -> Corpus:
    return Corpus(seeds_dir=seeds_dir, corpus_dir=corpus_dir)


# ---------------------------------------------------------------------------
# load_seeds()
# ---------------------------------------------------------------------------

class TestLoadSeeds:
    def test_loads_json_files_from_dir(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        corpus_dir = tmp_path / "corpus"

        for i in range(3):
            t = make_trace(seed=i)
            (seeds / f"seed_{i}.json").write_text(t.to_json())

        c = make_corpus(str(seeds), str(corpus_dir))
        count = c.load_seeds()
        assert count == 3
        assert c.size() == 3

    def test_skips_invalid_json(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        corpus_dir = tmp_path / "corpus"

        (seeds / "good.json").write_text(make_trace(1).to_json())
        (seeds / "bad.json").write_text("not valid json{{{")

        c = make_corpus(str(seeds), str(corpus_dir))
        count = c.load_seeds()
        assert count == 1

    def test_empty_dir_returns_zero(self, tmp_path):
        seeds  = tmp_path / "seeds"
        seeds.mkdir()
        corpus = tmp_path / "corpus"
        c = make_corpus(str(seeds), str(corpus))
        assert c.load_seeds() == 0

    def test_ignores_non_json_files(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        corpus_dir = tmp_path / "corpus"

        (seeds / "trace.json").write_text(make_trace(1).to_json())
        (seeds / "notes.txt").write_text("ignore me")

        c = make_corpus(str(seeds), str(corpus_dir))
        count = c.load_seeds()
        assert count == 1


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestAdd:
    def test_high_score_saved_to_disk(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        c.add(make_trace(seed=999), failing_result(score=8.0), save=True)
        saved = list(corpus_dir.glob("*.json"))
        assert len(saved) == 1

    def test_zero_score_not_saved_to_disk(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        c.add(make_trace(seed=1), passing_result(), save=True)
        saved = list(corpus_dir.glob("*.json"))
        assert len(saved) == 0

    def test_saved_file_is_valid_trace(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        t = make_trace(seed=42)
        c.add(t, failing_result(), save=True)
        saved_path = next(corpus_dir.glob("*.json"))
        loaded = Trace.from_json(saved_path.read_text())
        assert loaded.seed == 42


# ---------------------------------------------------------------------------
# select()
# ---------------------------------------------------------------------------

class TestSelect:
    def test_empty_corpus_returns_none(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.select(random.Random(0)) is None

    def test_single_entry_always_returned(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        t = make_trace(1)
        c.add(t, passing_result())
        result = c.select(random.Random(0))
        assert result is not None
        assert result.seed == 1

    def test_select_increments_run_count(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c.add(make_trace(1), passing_result())
        c.select(random.Random(0))
        assert c._entries[0].run_count == 1

    def test_high_score_selected_more_often(self, tmp_path):
        """High-score entry should dominate selections with enough trials."""
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c.add(make_trace(seed=1), passing_result())                   # score 0
        c.add(make_trace(seed=2), failing_result(score=1000.0))       # score 1000

        rng = random.Random(0)
        counts = {1: 0, 2: 0}
        for _ in range(200):
            t = c.select(rng)
            counts[t.seed] += 1

        # The high-score entry should win overwhelmingly
        assert counts[2] > counts[1]

    def test_all_entries_reachable_with_equal_scores(self, tmp_path):
        """With equal weights, each entry should be selectable."""
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        for i in range(5):
            c.add(make_trace(seed=i), passing_result())

        rng = random.Random(0)
        seen = set()
        for _ in range(500):
            t = c.select(rng)
            seen.add(t.seed)

        assert seen == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# size(), findings_count(), summary()
# ---------------------------------------------------------------------------

class TestWarmupRoundRobin:
    def test_warmup_cycles_through_seeds_in_order(self, tmp_path):
        """During warmup, seeds are selected in round-robin order."""
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        corpus_dir = tmp_path / "corpus"

        for i in range(3):
            t = make_trace(seed=i)
            (seeds / f"seed_{i}.json").write_text(t.to_json())

        c = Corpus(str(seeds), str(corpus_dir), warmup_rounds=1)
        c.load_seeds()  # _seed_count = 3

        rng = random.Random(0)
        selected = [c.select(rng).seed for _ in range(3)]
        assert selected == [0, 1, 2]

    def test_warmup_repeats_for_multiple_rounds(self, tmp_path):
        """With warmup_rounds=2 and 2 seeds, first 4 selections are round-robin."""
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        corpus_dir = tmp_path / "corpus"

        for i in range(2):
            (seeds / f"seed_{i}.json").write_text(make_trace(seed=i).to_json())

        c = Corpus(str(seeds), str(corpus_dir), warmup_rounds=2)
        c.load_seeds()

        rng = random.Random(0)
        selected = [c.select(rng).seed for _ in range(4)]
        assert selected == [0, 1, 0, 1]

    def test_after_warmup_uses_weighted_selection(self, tmp_path):
        """After warmup phase ends, high-score entry dominates."""
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        corpus_dir = tmp_path / "corpus"

        for i in range(2):
            (seeds / f"seed_{i}.json").write_text(make_trace(seed=i).to_json())

        # warmup_rounds=0 → skip warmup entirely, go straight to weighted
        c = Corpus(str(seeds), str(corpus_dir), warmup_rounds=0)
        c.load_seeds()
        # Boost seed 1 to a very high score
        c._entries[0].score = 0.1
        c._entries[1].score = 1000.0

        rng = random.Random(0)
        counts = {0: 0, 1: 0}
        for _ in range(200):
            t = c.select(rng)
            counts[t.seed] += 1

        assert counts[1] > counts[0]

    def test_warmup_zero_seeds_falls_back_to_weighted(self, tmp_path):
        """If no seeds loaded, warmup is skipped gracefully."""
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"), warmup_rounds=4)
        c.load_seeds()  # _seed_count = 0

        c.add(make_trace(1), passing_result())
        result = c.select(random.Random(0))
        assert result is not None


class _CapturingRng(random.Random):
    """random.Random subclass that records the weights handed to choices().

    Used by TestScoreMultiplier to assert exact weight arithmetic without
    relying on sampling statistics. The returned choice is deterministic
    (always the first entry) — only the captured `last_weights` matter.
    """
    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.last_weights: list = None

    def choices(self, population, weights=None, k=1):
        self.last_weights = list(weights) if weights is not None else None
        return [population[0]]


class TestScoreMultiplier:
    """Exact-weight assertions for the score_multiplier callback added in
    Phase 1 of the adapter/strategy split. No sampling — uses _CapturingRng
    to capture the weights list passed to random.choices()."""

    def _two_entries(self, tmp_path):
        """Two entries with scores 10 and 20, zero run_count and pressure,
        and warmup skipped so select() takes the weighted path."""
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c.add(make_trace(seed=1), failing_result(score=10.0))
        c.add(make_trace(seed=2), failing_result(score=20.0))
        for e in c._entries:
            e.run_count     = 0
            e.pressure_score = 0.0
        # Skip warmup so select() runs the weighted-random branch we care about.
        c._seed_count = 0
        return c

    def test_none_callback_uses_default_weights(self, tmp_path):
        c = self._two_entries(tmp_path)
        rng = _CapturingRng()
        c.select(rng, score_multiplier=None)
        # Decay denominator = 1.0, pressure = 0 → weight = score
        assert rng.last_weights == [10.0, 20.0]

    def test_missing_callback_argument_uses_default_weights(self, tmp_path):
        """Calling select() with no score_multiplier kwarg = same as None."""
        c = self._two_entries(tmp_path)
        rng = _CapturingRng()
        c.select(rng)
        assert rng.last_weights == [10.0, 20.0]

    def test_uniform_half_multiplier_halves_each_weight(self, tmp_path):
        c = self._two_entries(tmp_path)
        rng = _CapturingRng()
        c.select(rng, score_multiplier=lambda e: 0.5)
        assert rng.last_weights == [5.0, 10.0]

    def test_all_zero_multiplier_clamps_to_floor(self, tmp_path):
        """If the multiplier returns 0 for every entry, the post-multiplication
        floor must keep weights at 0.01 so random.choices() does not raise."""
        c = self._two_entries(tmp_path)
        rng = _CapturingRng()
        c.select(rng, score_multiplier=lambda e: 0.0)
        assert rng.last_weights == [0.01, 0.01]

    def test_selective_zero_floors_only_targeted_entry(self, tmp_path):
        c = self._two_entries(tmp_path)
        rng = _CapturingRng()
        c.select(rng, score_multiplier=lambda e: 0.0 if e.trace.seed == 1 else 1.0)
        assert rng.last_weights == [0.01, 20.0]

    def test_multiplier_applied_after_run_count_decay(self, tmp_path):
        c = self._two_entries(tmp_path)
        # Entry 1: run_count=20 → decay denominator 1 + 20*0.05 = 2.0
        # Entry 2: run_count=0  → decay denominator 1.0
        c._entries[0].run_count = 20
        rng = _CapturingRng()
        c.select(rng, score_multiplier=lambda e: 2.0)
        # Entry 1: (10 / 2.0) * 2.0 = 10.0
        # Entry 2: (20 / 1.0) * 2.0 = 40.0
        assert rng.last_weights == [10.0, 40.0]


class TestStats:
    def test_findings_count_only_counts_nonzero_scores(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c.add(make_trace(1), passing_result())          # score 0 → not a finding
        c.add(make_trace(2), failing_result(5.0))       # score 5 → finding
        c.add(make_trace(3), failing_result(0.0))       # score 0 → not a finding
        assert c.findings_count() == 1

    def test_summary_top_scores_sorted_desc(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        for score in [1.0, 5.0, 3.0]:
            c.add(make_trace(), failing_result(score))
        top = c.summary()["top_scores"]
        assert top == sorted(top, reverse=True)


# ---------------------------------------------------------------------------
# Novelty tracking
# ---------------------------------------------------------------------------

class TestNoveltyTracking:
    def test_first_fingerprint_is_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=1)
        assert c.is_novel(fp) is True

    def test_same_fingerprint_not_novel_after_add(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=1)
        c.add(make_trace(1), passing_result(), fingerprint=fp)
        assert c.is_novel(fp) is False

    def test_different_fingerprint_still_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp1 = make_fp(bucket=1)
        fp2 = make_fp(bucket=2)
        c.add(make_trace(1), passing_result(), fingerprint=fp1)
        assert c.is_novel(fp2) is True

    def test_none_fingerprint_never_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel(None) is False

    def test_add_returns_true_for_novel_fingerprint(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        novel = c.add(make_trace(1), passing_result(), fingerprint=make_fp(bucket=0))
        assert novel is True

    def test_add_returns_false_for_duplicate_fingerprint(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=0)
        c.add(make_trace(1), passing_result(), fingerprint=fp)
        novel = c.add(make_trace(2), passing_result(), fingerprint=fp)
        assert novel is False

    def test_novel_fingerprint_high_score_not_lowered(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=2)
        c.add(make_trace(1), failing_result(score=8.0), fingerprint=fp)
        entry = c._entries[-1]
        assert entry.score == 8.0

    def test_duplicate_fingerprint_score_decays(self, tmp_path):
        """A repeat sighting still scores, but at 1/sqrt(n+1) of the bonus.

        Was `score == 0.0` under the binary one-shot signal; step 8 replaced
        the cliff with a decay so a second sighting is worth 0.71x and a fourth
        0.5x of _NOVELTY_SCORE_FLOOR.
        """
        from fuzzer.corpus import _NOVELTY_SCORE_FLOOR
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=0)
        c.add(make_trace(1), passing_result(), fingerprint=fp)   # 1st → 1.0
        c.add(make_trace(2), passing_result(), fingerprint=fp)   # 2nd → 0.71
        assert c._entries[-1].score == pytest.approx(
            _NOVELTY_SCORE_FLOOR / 2 ** 0.5)
        c.add(make_trace(3), passing_result(), fingerprint=fp)   # 3rd → 0.577
        c.add(make_trace(4), passing_result(), fingerprint=fp)   # 4th → 0.5
        assert c._entries[-1].score == pytest.approx(_NOVELTY_SCORE_FLOOR / 2)
        # Strictly decreasing, and never negative.
        scores = [e.score for e in c._entries]
        assert scores == sorted(scores, reverse=True)
        assert scores[-1] > 0.0

    def test_novelty_counter_accumulates_across_adds(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        for bucket in range(4):
            fp = make_fp(bucket=bucket)
            c.add(make_trace(bucket), passing_result(), fingerprint=fp)
        assert len(c._seen_fingerprints) == 4

    def test_add_without_fingerprint_returns_false(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        novel = c.add(make_trace(1), passing_result(), fingerprint=None)
        assert novel is False


class TestErrorSignatureNovelty:
    def test_novel_error_sig_returns_true(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        novel = c.add(make_trace(1), passing_result(), error_sig="abc123")
        assert novel is True

    def test_duplicate_error_sig_returns_false(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c.add(make_trace(1), passing_result(), error_sig="abc123")
        novel = c.add(make_trace(2), passing_result(), error_sig="abc123")
        assert novel is False

    def test_none_error_sig_not_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_error(None) is False

    def test_error_sig_tracked_independently_of_fingerprint(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=1)
        # Same fingerprint twice but different error sigs → both novel
        c.add(make_trace(1), passing_result(), fingerprint=fp, error_sig="aaa")
        novel = c.add(make_trace(2), passing_result(), fingerprint=fp, error_sig="bbb")
        assert novel is True  # error_sig is novel even though fp is duplicate

    def test_seen_error_sigs_accumulates(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        for sig in ["aaa", "bbb", "ccc"]:
            c.add(make_trace(1), passing_result(), error_sig=sig)
        assert len(c._seen_error_sigs) == 3


# ---------------------------------------------------------------------------
# Preemption novelty tracking
# ---------------------------------------------------------------------------

class TestPreemptionNovelty:
    def test_zero_delta_not_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_preemption(0) is False

    def test_none_delta_not_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_preemption(None) is False

    def test_positive_delta_first_time_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_preemption(3) is True

    def test_same_delta_not_novel_after_seen(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_preemption_counts[3] += 1
        assert c.is_novel_preemption(3) is False

    def test_different_delta_still_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_preemption_counts[3] += 1
        assert c.is_novel_preemption(5) is True

    def test_cap_collapses_large_deltas(self, tmp_path):
        from fuzzer.corpus import _PREEMPTION_COUNT_CAP
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_preemption_counts[_PREEMPTION_COUNT_CAP] += 1
        # Values above cap should map to same bucket and not be novel
        assert c.is_novel_preemption(_PREEMPTION_COUNT_CAP + 1) is False
        assert c.is_novel_preemption(_PREEMPTION_COUNT_CAP + 999) is False

    def test_below_cap_still_distinct(self, tmp_path):
        from fuzzer.corpus import _PREEMPTION_COUNT_CAP
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_preemption_counts[_PREEMPTION_COUNT_CAP] += 1
        # One below cap is a different bucket — still novel
        assert c.is_novel_preemption(_PREEMPTION_COUNT_CAP - 1) is True

    def test_add_returns_true_for_novel_preemption(self, tmp_path):
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, preemptions_total=0),
            MetricSnapshot(timestamp=1.0, preemptions_total=4),
        ]
        novel = c.add(make_trace(1), passing_result(), session=session)
        assert novel is True

    def test_add_returns_false_for_duplicate_preemption_delta(self, tmp_path):
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, preemptions_total=0),
            MetricSnapshot(timestamp=1.0, preemptions_total=4),
        ]
        c.add(make_trace(1), passing_result(), session=session)
        novel = c.add(make_trace(2), passing_result(), session=session)
        assert novel is False

    def test_novel_preemption_saved_to_disk(self, tmp_path):
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, preemptions_total=0),
            MetricSnapshot(timestamp=1.0, preemptions_total=3),
        ]
        c.add(make_trace(seed=77), passing_result(), save=True, session=session)
        saved = list(corpus_dir.glob("*.json"))
        assert len(saved) == 1
        data = json.loads(saved[0].read_text())
        assert data["_corpus_preemption_delta"] == 3

    def test_seen_preemption_counts_restored_on_reload(self, tmp_path):
        """After reload, previously seen preemption counts are not novel again."""
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()

        import dataclasses, json as _j
        t = make_trace(seed=50)
        data = t.to_dict()
        data["_corpus_score"] = 1.5
        data["_corpus_fingerprint"] = None
        data["_corpus_preemption_delta"] = 7
        (corpus_dir / "seed_50.json").write_text(_j.dumps(data))

        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c.is_novel_preemption(7) is False

    def test_preemption_independent_of_fingerprint(self, tmp_path):
        """Novel preemption count is kept even if fingerprint was already seen."""
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=1)
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, preemptions_total=0),
            MetricSnapshot(timestamp=1.0, preemptions_total=5),
        ]
        # First add: fp is novel (registered)
        c.add(make_trace(1), passing_result(), fingerprint=fp, session=session)
        # Second add: same fp (not novel) but different preemption count → still novel
        session2 = MonitorSession()
        session2.snapshots = [
            MetricSnapshot(timestamp=0.0, preemptions_total=0),
            MetricSnapshot(timestamp=1.0, preemptions_total=8),
        ]
        novel = c.add(make_trace(2), passing_result(), fingerprint=fp, session=session2)
        assert novel is True


# ---------------------------------------------------------------------------
# Corpus persistence: _save() embeds score + fingerprint
# ---------------------------------------------------------------------------

class TestCorpusSave:
    def test_save_embeds_corpus_score(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        c.add(make_trace(seed=7), failing_result(score=3.5), save=True)
        data = json.loads(next(corpus_dir.glob("*.json")).read_text())
        assert data["_corpus_score"] == pytest.approx(3.5)

    def test_save_embeds_fingerprint_when_provided(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        fp = make_fp(bucket=2, kv_peak_bucket=3)
        c.add(make_trace(seed=8), failing_result(), save=True, fingerprint=fp)
        data = json.loads(next(corpus_dir.glob("*.json")).read_text())
        assert data["_corpus_fingerprint"] is not None
        assert isinstance(data["_corpus_fingerprint"], list)

    def test_save_stores_null_fingerprint_when_absent(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        c.add(make_trace(seed=9), failing_result(), save=True, fingerprint=None)
        data = json.loads(next(corpus_dir.glob("*.json")).read_text())
        assert data["_corpus_fingerprint"] is None

    def test_saved_file_still_loads_as_valid_trace(self, tmp_path):
        # Extra metadata keys must not break Trace.from_dict()
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        c.add(make_trace(seed=42), failing_result(), save=True,
              fingerprint=make_fp(bucket=1))
        from fuzzer.trace import Trace
        loaded = Trace.from_json(next(corpus_dir.glob("*.json")).read_text())
        assert loaded.seed == 42


# ---------------------------------------------------------------------------
# Corpus reload: load_seeds() picks up corpus_dir entries
# ---------------------------------------------------------------------------

class TestCorpusReload:
    def _write_corpus_entry(self, corpus_dir: Path, seed: int,
                             score: float = 2.0,
                             fp: SchedulerFingerprint = None) -> None:
        """Simulate a previously saved corpus entry."""
        import dataclasses, json as _json
        from fuzzer.trace import Trace, Event, ActionType
        t = Trace(seed=seed, events=[
            Event(action=ActionType.SEND, offset_ms=0, request_id="r1",
                  prompt_len=32, max_tokens=16)
        ])
        data = t.to_dict()
        data["_corpus_score"] = score
        data["_corpus_fingerprint"] = (
            list(dataclasses.astuple(fp)) if fp is not None else None
        )
        (corpus_dir / f"seed_{seed}.json").write_text(_json.dumps(data))

    def test_corpus_entries_loaded_on_reload(self, tmp_path):
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        self._write_corpus_entry(corpus_dir, seed=100, score=3.0)
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c.size() == 1

    def test_corpus_entries_have_baseline_score(self, tmp_path):
        from fuzzer.corpus import _NOVELTY_SCORE_FLOOR
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        self._write_corpus_entry(corpus_dir, seed=101, score=_NOVELTY_SCORE_FLOOR)
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c._entries[0].score >= _NOVELTY_SCORE_FLOOR

    def test_seed_count_not_inflated_by_corpus(self, tmp_path):
        # _seed_count must reflect only hand-written seeds, not corpus entries,
        # so the warmup round-robin length is correct.
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        (seeds_dir / "seed_1.json").write_text(make_trace(seed=1).to_json())
        self._write_corpus_entry(corpus_dir, seed=200, score=2.0)
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c._seed_count == 1   # only the hand-written seed

    def test_dedup_keeps_highest_scored_entry_per_fingerprint(self, tmp_path):
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        fp = make_fp(bucket=1)
        # Two entries, same fingerprint, different scores
        self._write_corpus_entry(corpus_dir, seed=10, score=2.0, fp=fp)
        self._write_corpus_entry(corpus_dir, seed=11, score=7.0, fp=fp)
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        # Only one entry loaded (deduplication), and it should be the higher-scored one
        assert c.size() == 1
        assert c._entries[0].score == pytest.approx(7.0)

    def test_dedup_discards_lower_scored_duplicate(self, tmp_path):
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        fp = make_fp(bucket=2)
        self._write_corpus_entry(corpus_dir, seed=20, score=9.0, fp=fp)
        self._write_corpus_entry(corpus_dir, seed=21, score=1.0, fp=fp)
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c.size() == 1
        assert c._entries[0].score == pytest.approx(9.0)

    def test_different_fingerprints_both_loaded(self, tmp_path):
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        self._write_corpus_entry(corpus_dir, seed=30, score=2.0, fp=make_fp(bucket=1))
        self._write_corpus_entry(corpus_dir, seed=31, score=2.0, fp=make_fp(bucket=2))
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c.size() == 2

    def test_seen_fingerprints_prepopulated_from_corpus(self, tmp_path):
        # Fingerprints from loaded corpus entries must be marked as seen so
        # the running session doesn't re-add them as novel.
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        fp = make_fp(bucket=3)
        self._write_corpus_entry(corpus_dir, seed=40, score=2.0, fp=fp)
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert not c.is_novel(fp)

    def test_no_fp_entries_loaded_up_to_cap(self, tmp_path):
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        # Write 5 entries without fingerprints — all should load (well below cap)
        for i in range(5):
            self._write_corpus_entry(corpus_dir, seed=50 + i, score=float(i + 1))
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c.size() == 5

    def test_no_fp_entries_capped_at_200(self, tmp_path):
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        # Write 250 entries without fingerprints — only 200 should load
        for i in range(250):
            self._write_corpus_entry(corpus_dir, seed=1000 + i,
                                     score=float(i + 1))
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c.size() == 200

    def test_no_fp_cap_prefers_highest_scored(self, tmp_path):
        """The 200-entry fallback cap must keep the *highest scored* entries.

        Repaired: this test used to write 10 entries against a 200 cap (so the
        cap never bit) and defined an `original_load`/`patched_load` pair that
        was never applied — it could not fail.  It now writes 250 entries and
        asserts the surviving set is exactly the top 200 by score.

        The scores are deliberately laid out so that score order and *file*
        order disagree: seed_1000..seed_1249 sort lexicographically as
        1000,1001,...,1099,1100,...,1249, and score = index + 1, so keeping the
        first 200 files would retain scores 1-200 while keeping the top 200
        scores retains 51-250.  Dropping the `no_fp.sort(reverse=True)` at
        corpus.py:447 therefore fails this test.
        """
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        for i in range(250):
            self._write_corpus_entry(corpus_dir, seed=2000 + i,
                                     score=float(i + 1))
        c = Corpus(str(seeds_dir), str(corpus_dir), warmup_rounds=0)
        c.load_seeds()

        scores = sorted(e.score for e in c._entries)
        assert len(scores) == 200
        assert scores == [float(s) for s in range(51, 251)]


# ---------------------------------------------------------------------------
# Swap novelty tracking (is_novel_swap)
# ---------------------------------------------------------------------------

class TestSwapNovelty:
    def test_none_peak_not_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_swap(None) is False

    def test_zero_peak_not_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_swap(0) is False

    def test_positive_peak_first_time_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_swap(3) is True

    def test_same_peak_not_novel_after_seen(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_swap_counts[3] += 1
        assert c.is_novel_swap(3) is False

    def test_different_peak_still_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_swap_counts[3] += 1
        assert c.is_novel_swap(5) is True

    def test_cap_collapses_large_peaks(self, tmp_path):
        from fuzzer.corpus import _SWAP_COUNT_CAP
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_swap_counts[_SWAP_COUNT_CAP] += 1
        assert c.is_novel_swap(_SWAP_COUNT_CAP + 1) is False
        assert c.is_novel_swap(_SWAP_COUNT_CAP + 50) is False

    def test_below_cap_still_distinct(self, tmp_path):
        from fuzzer.corpus import _SWAP_COUNT_CAP
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c._seen_swap_counts[_SWAP_COUNT_CAP] += 1
        assert c.is_novel_swap(_SWAP_COUNT_CAP - 1) is True

    def test_add_returns_true_for_novel_swap(self, tmp_path):
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, requests_swapped=0),
            MetricSnapshot(timestamp=1.0, requests_swapped=5),
        ]
        novel = c.add(make_trace(1), passing_result(), session=session)
        assert novel is True

    def test_add_returns_false_for_duplicate_swap_peak(self, tmp_path):
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, requests_swapped=0),
            MetricSnapshot(timestamp=1.0, requests_swapped=5),
        ]
        c.add(make_trace(1), passing_result(), session=session)
        novel = c.add(make_trace(2), passing_result(), session=session)
        assert novel is False

    def test_novel_swap_saved_to_disk(self, tmp_path):
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, requests_swapped=0),
            MetricSnapshot(timestamp=1.0, requests_swapped=4),
        ]
        c.add(make_trace(seed=88), passing_result(), save=True, session=session)
        saved = list(corpus_dir.glob("*.json"))
        assert len(saved) == 1
        data = json.loads(saved[0].read_text())
        assert data["_corpus_peak_swapped"] == 4

    def test_seen_swap_counts_restored_on_reload(self, tmp_path):
        """After reload, previously seen swap peaks are not novel again."""
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()

        import json as _j
        t = make_trace(seed=55)
        data = t.to_dict()
        data["_corpus_score"] = 2.0
        data["_corpus_fingerprint"] = None
        data["_corpus_preemption_delta"] = None
        data["_corpus_peak_swapped"] = 6
        (corpus_dir / "seed_55.json").write_text(_j.dumps(data))

        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        assert c.is_novel_swap(6) is False

    def test_backward_compat_7_element_fingerprint_reloads(self, tmp_path):
        """Old corpus files with 7-element fingerprints load without error."""
        seeds_dir  = tmp_path / "seeds"
        corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()

        import json as _j
        t = make_trace(seed=60)
        data = t.to_dict()
        data["_corpus_score"] = 3.0
        # Simulate old 7-element fingerprint: (True, False, 1, 2, 1, 0, 2)
        data["_corpus_fingerprint"] = [True, False, 1, 2, 1, 0, 2]
        data["_corpus_preemption_delta"] = None
        (corpus_dir / "seed_60.json").write_text(_j.dumps(data))

        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()   # must not raise
        assert c.size() == 1
        fp = c._entries[0].fingerprint
        assert fp is not None
        assert fp.prefix_cache_hit_bucket == 0  # padded default


# ---------------------------------------------------------------------------
# Step 8: rarity-weighted novelty (hit counts instead of membership sets)
# ---------------------------------------------------------------------------

class TestRarityWeightedNovelty:
    """The five novelty signals return a decaying float, not a binary bonus.

    Decay is 1/sqrt(n + 1) over the number of prior sightings n:
    1st = full bonus, 2nd = 0.71x, 4th = 0.5x, 256th = 0.0625x.  See
    fuzzer/core/corpus.py::_rarity_decay for why this curve and not the
    power-of-two staircase the plan's code snippet showed.
    """

    def test_decay_curve_matches_the_documented_numbers(self):
        from fuzzer.core.corpus import _rarity_decay
        assert _rarity_decay(0)   == pytest.approx(1.0)
        assert _rarity_decay(1)   == pytest.approx(0.7071, abs=1e-4)
        assert _rarity_decay(3)   == pytest.approx(0.5)
        assert _rarity_decay(255) == pytest.approx(0.0625)

    def test_decay_is_strictly_monotone(self):
        """Unlike the power-of-two variant, no two counts pay the same."""
        from fuzzer.core.corpus import _rarity_decay
        vals = [_rarity_decay(n) for n in range(0, 40)]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_decay_never_reaches_zero(self):
        """Uncapped, but always positive — a saturated key is out-ranked, not
        erased."""
        from fuzzer.core.corpus import _rarity_decay
        assert _rarity_decay(10 ** 6) > 0.0

    def test_fingerprint_novelty_returns_full_bonus_then_decays(self, tmp_path):
        from fuzzer.corpus import _NOVELTY_SCORE_FLOOR
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=1)
        assert c.fingerprint_novelty(fp) == pytest.approx(_NOVELTY_SCORE_FLOOR)
        c.add(make_trace(1), passing_result(), fingerprint=fp)
        assert c.fingerprint_novelty(fp) == pytest.approx(
            _NOVELTY_SCORE_FLOOR / 2 ** 0.5)

    def test_fingerprint_novelty_none_is_zero(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.fingerprint_novelty(None) == 0.0

    def test_error_novelty_decays_from_its_own_constant(self, tmp_path):
        from fuzzer.corpus import _ERROR_NOVELTY_SCORE
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.error_novelty("sig") == pytest.approx(_ERROR_NOVELTY_SCORE)
        c.add(make_trace(1), passing_result(), error_sig="sig")
        assert c.error_novelty("sig") == pytest.approx(
            _ERROR_NOVELTY_SCORE / 2 ** 0.5)
        assert c.error_novelty(None) == 0.0

    def test_preemption_novelty_decays_and_respects_the_cap(self, tmp_path):
        from fuzzer.corpus import (_PREEMPTION_NOVELTY_SCORE,
                                   _PREEMPTION_COUNT_CAP)
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.preemption_novelty(0)    == 0.0
        assert c.preemption_novelty(None) == 0.0
        assert c.preemption_novelty(3) == pytest.approx(_PREEMPTION_NOVELTY_SCORE)
        c._seen_preemption_counts[3] += 1
        assert c.preemption_novelty(3) == pytest.approx(
            _PREEMPTION_NOVELTY_SCORE / 2 ** 0.5)
        # Above the cap everything shares one bucket, so it decays together.
        c._seen_preemption_counts[_PREEMPTION_COUNT_CAP] += 1
        assert (c.preemption_novelty(_PREEMPTION_COUNT_CAP + 999)
                == pytest.approx(_PREEMPTION_NOVELTY_SCORE / 2 ** 0.5))

    def test_swap_novelty_decays_and_respects_the_cap(self, tmp_path):
        from fuzzer.corpus import _SWAP_NOVELTY_SCORE, _SWAP_COUNT_CAP
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.swap_novelty(0)    == 0.0
        assert c.swap_novelty(None) == 0.0
        assert c.swap_novelty(4) == pytest.approx(_SWAP_NOVELTY_SCORE)
        c._seen_swap_counts[4] += 1
        assert c.swap_novelty(4) == pytest.approx(_SWAP_NOVELTY_SCORE / 2 ** 0.5)
        c._seen_swap_counts[_SWAP_COUNT_CAP] += 1
        assert (c.swap_novelty(_SWAP_COUNT_CAP + 5)
                == pytest.approx(_SWAP_NOVELTY_SCORE / 2 ** 0.5))

    def test_is_novel_predicates_are_count_zero_wrappers(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=2)
        assert c.is_novel(fp) is True
        c.add(make_trace(1), passing_result(), fingerprint=fp)
        assert c.is_novel(fp) is False
        # ...but the signal still pays something.
        assert c.fingerprint_novelty(fp) > 0.0

    def test_score_is_max_of_signals_not_sum(self, tmp_path):
        """A trace tripping several signals scores the rarest, not their sum."""
        from fuzzer.corpus import _SWAP_NOVELTY_SCORE
        from fuzzer.monitor import MonitorSession, MetricSnapshot
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        session = MonitorSession()
        session.snapshots = [
            MetricSnapshot(timestamp=0.0, preemptions_total=0, requests_swapped=0),
            MetricSnapshot(timestamp=1.0, preemptions_total=2, requests_swapped=3),
        ]
        c.add(make_trace(1), passing_result(), fingerprint=make_fp(bucket=1),
              error_sig=None, session=session, pressure_score=0.0)
        # swap (2.0) is the largest of fingerprint 1.0 / preemption 1.5 / swap 2.0
        assert c._entries[-1].score == pytest.approx(_SWAP_NOVELTY_SCORE)

    def test_repeated_adds_produce_a_decaying_score_series(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        fp = make_fp(bucket=3)
        for i in range(6):
            c.add(make_trace(i), passing_result(), fingerprint=fp, save=False)
        scores = [e.score for e in c._entries]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] > scores[-1] > 0.0

    def test_oracle_score_still_wins_over_a_fresh_novelty_bonus(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c.add(make_trace(1), failing_result(score=8.0), fingerprint=make_fp(bucket=1))
        assert c._entries[-1].score == pytest.approx(8.0)


class TestKVNovelty:
    """novel_kv used to be an always-novel signal (an exact-match test over a
    near-unique key).  kv_events.py now log2-buckets the counters, and the
    corpus decays repeat sightings like every other signal."""

    def test_first_kv_fingerprint_is_novel(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_kv("abc") is True
        assert c.add(make_trace(1), passing_result(), kv_fingerprint="abc") is True

    def test_duplicate_kv_fingerprint_is_not_novel_but_still_scores(self, tmp_path):
        from fuzzer.corpus import _KV_NOVELTY_SCORE
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        c.add(make_trace(1), passing_result(), kv_fingerprint="abc")
        assert c.add(make_trace(2), passing_result(), kv_fingerprint="abc") is False
        assert c._entries[-1].score == pytest.approx(
            _KV_NOVELTY_SCORE / 2 ** 0.5)

    def test_none_and_empty_kv_fingerprint_are_inert(self, tmp_path):
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.is_novel_kv(None) is False
        assert c.is_novel_kv("") is False       # disabled collector
        assert c.kv_novelty(None) == 0.0
        assert c.kv_novelty("") == 0.0
        assert c.add(make_trace(1), passing_result(), kv_fingerprint="") is False

    def test_kv_novelty_bonus_decays(self, tmp_path):
        from fuzzer.corpus import _KV_NOVELTY_SCORE
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.kv_novelty("fp") == pytest.approx(_KV_NOVELTY_SCORE)
        for _ in range(3):
            c._seen_kv_fingerprints["fp"] += 1
        assert c.kv_novelty("fp") == pytest.approx(_KV_NOVELTY_SCORE / 2)

    def test_bucketed_kv_fingerprints_collapse_to_one_key(self, tmp_path):
        """Two probe windows 100 vs 120 blocks apart share a fingerprint, so
        the second one is correctly *not* novel."""
        import collections as _c
        from fuzzer.kv_events import KVEventCollector

        def summary_fp(n: int) -> str:
            col = KVEventCollector.__new__(KVEventCollector)
            col._buf = _c.deque([{
                "type": "BlockStored", "t": 1.0,
                "block_hashes": list(range(1, n + 1)),
                "parent_block_hash": None, "token_ids": [], "block_size": 16,
                "lora_id": 1, "lora_name": "lora_a", "medium": "GPU",
            }], maxlen=2000)
            col._enabled = True
            col._lora_id_to_name = {}
            col._block_owner = {}
            return col.summarize_probe(0.5, 2.0).fingerprint

        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.add(make_trace(1), passing_result(),
                     kv_fingerprint=summary_fp(100)) is True
        assert c.add(make_trace(2), passing_result(),
                     kv_fingerprint=summary_fp(120)) is False
        assert c.add(make_trace(3), passing_result(),
                     kv_fingerprint=summary_fp(1000)) is True


class TestPersistenceGateUnchanged:
    """A decayed novelty bonus is a selection signal, not a reason to write a
    file.  The set of entries that reach disk must be the same as before
    step 8, or the corpus dir grows by one file per iteration."""

    def test_repeat_fingerprint_scores_but_is_not_saved(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        fp = make_fp(bucket=1)
        c.add(make_trace(seed=1), passing_result(), save=True, fingerprint=fp)
        c.add(make_trace(seed=2), passing_result(), save=True, fingerprint=fp)
        assert c._entries[-1].score > 0.0        # decayed bonus, still positive
        assert {p.name for p in corpus_dir.glob("*.json")} == {"seed_1.json"}

    def test_repeat_kv_fingerprint_is_not_saved(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        c.add(make_trace(seed=1), passing_result(), save=True, kv_fingerprint="k")
        c.add(make_trace(seed=2), passing_result(), save=True, kv_fingerprint="k")
        assert {p.name for p in corpus_dir.glob("*.json")} == {"seed_1.json"}

    def test_oracle_finding_is_saved_even_on_a_stale_fingerprint(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        fp = make_fp(bucket=1)
        c.add(make_trace(seed=1), passing_result(), save=True, fingerprint=fp)
        c.add(make_trace(seed=2), failing_result(score=6.0), save=True,
              fingerprint=fp)
        assert len(list(corpus_dir.glob("*.json"))) == 2

    def test_min_score_still_forces_a_save(self, tmp_path):
        corpus_dir = tmp_path / "corpus"
        c = make_corpus(str(tmp_path / "s"), str(corpus_dir))
        fp = make_fp(bucket=1)
        c.add(make_trace(seed=1), passing_result(), save=True, fingerprint=fp)
        c.add(make_trace(seed=2), passing_result(), save=True, fingerprint=fp,
              min_score=1.0)
        assert len(list(corpus_dir.glob("*.json"))) == 2


# ---------------------------------------------------------------------------
# Step 9: select() anti-collapse behind selection_policy="v2"
# ---------------------------------------------------------------------------

class TestSelectionPolicyResolution:
    def test_default_is_v1(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GRIEF_SELECTION_POLICY", raising=False)
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.selection_policy == "v1"

    def test_explicit_argument_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_SELECTION_POLICY", "v1")
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v2")
        assert c.selection_policy == "v2"

    def test_env_var_is_read_when_no_argument(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_SELECTION_POLICY", "V2")
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.selection_policy == "v2"

    def test_unknown_policy_falls_back_to_v1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_SELECTION_POLICY", "nonsense")
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.selection_policy == "v1"

    def test_empty_env_var_falls_back_to_v1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_SELECTION_POLICY", "")
        c = make_corpus(str(tmp_path / "s"), str(tmp_path / "c"))
        assert c.selection_policy == "v1"


class TestSelectV2:
    """v2 replaces the flat 0.01 floor with a least-run-first rarity floor and
    age-decays the score term."""

    def _corpus(self, tmp_path, n=1, **kw):
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v2", **kw)
        for i in range(n):
            c.add(make_trace(seed=i), passing_result())
        c._seed_count = 0          # skip warmup, go straight to weighted select
        return c

    def test_never_selected_zero_score_entry_floors_at_051(self, tmp_path):
        c = self._corpus(tmp_path)
        rng = _CapturingRng()
        c.select(rng)
        assert rng.last_weights[0] == pytest.approx(0.51, abs=1e-3)

    def test_entry_run_20_times_decays_to_0034(self, tmp_path):
        c = self._corpus(tmp_path)
        c._entries[0].run_count = 20
        rng = _CapturingRng()
        c.select(rng)
        # 0.01 + 0.5 / 21 = 0.033809...
        assert rng.last_weights[0] == pytest.approx(0.0338, abs=1e-4)

    def test_floor_is_monotonically_least_run_first(self, tmp_path):
        c = self._corpus(tmp_path, n=3)
        c._entries[0].run_count = 0
        c._entries[1].run_count = 5
        c._entries[2].run_count = 50
        rng = _CapturingRng()
        c.select(rng)
        w = rng.last_weights
        assert w[0] > w[1] > w[2]

    def test_v1_keeps_the_flat_floor(self, tmp_path):
        """Same corpus under v1 gives the historical 0.01 for every entry."""
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v1")
        for i in range(3):
            c.add(make_trace(seed=i), passing_result())
        c._entries[1].run_count = 5
        c._seed_count = 0
        rng = _CapturingRng()
        c.select(rng)
        assert rng.last_weights == [0.01, 0.01, 0.01]

    def test_age_decay_halves_a_score_after_10k_selections(self, tmp_path):
        c = self._corpus(tmp_path)
        c._entries[0].score = 10.0
        c._select_count = 10_000     # entry was added at select_count 0
        rng = _CapturingRng()
        c.select(rng)
        # 10 / (1 + 1e-4 * 10001) ≈ 4.9998
        assert rng.last_weights[0] == pytest.approx(5.0, abs=1e-3)

    def test_age_decay_does_not_apply_under_v1(self, tmp_path):
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v1")
        c.add(make_trace(seed=1), passing_result())
        c._entries[0].score = 10.0
        c._select_count = 10_000
        c._seed_count = 0
        rng = _CapturingRng()
        c.select(rng)
        assert rng.last_weights == [10.0]

    def test_newer_entry_outweighs_an_equally_scored_older_one(self, tmp_path):
        c = self._corpus(tmp_path)
        c._entries[0].score = 10.0
        c._select_count = 50_000
        c.add(make_trace(seed=99), failing_result(score=10.0))   # added now
        rng = _CapturingRng()
        c.select(rng)
        assert rng.last_weights[1] > rng.last_weights[0]

    def test_added_at_select_count_is_recorded(self, tmp_path):
        c = self._corpus(tmp_path)
        c._select_count = 7
        c.add(make_trace(seed=42), passing_result())
        assert c._entries[-1].added_at_select_count == 7

    def test_multiplier_is_applied_before_the_floor(self, tmp_path):
        c = self._corpus(tmp_path, n=2)
        c._entries[0].score = 100.0
        c._entries[1].score = 100.0
        rng = _CapturingRng()
        c.select(rng, score_multiplier=lambda e: 0.0 if e.trace.seed == 0 else 1.0)
        assert rng.last_weights[0] == pytest.approx(0.51, abs=1e-3)
        assert rng.last_weights[1] > 50.0

    def test_floor_fraction_is_computed(self, tmp_path):
        c = self._corpus(tmp_path, n=4)
        c._entries[0].score = 100.0          # well above its 0.51 floor
        rng = _CapturingRng()
        c.select(rng)
        assert c._floor_count == 3
        assert c._floor_fraction == pytest.approx(0.75)

    def test_floor_fraction_is_one_when_everything_collapses(self, tmp_path):
        c = self._corpus(tmp_path, n=5)
        rng = _CapturingRng()
        c.select(rng)
        assert c._floor_fraction == pytest.approx(1.0)

    def test_floor_fraction_tracked_under_v1_too(self, tmp_path):
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v1")
        for i in range(4):
            c.add(make_trace(seed=i), passing_result())
        c._entries[0].score = 5.0
        c._seed_count = 0
        c.select(_CapturingRng())
        assert c._floor_fraction == pytest.approx(0.75)

    def test_floor_fraction_printed_at_most_once_per_100_selections(
            self, tmp_path, capsys):
        c = self._corpus(tmp_path, n=2)
        rng = random.Random(0)
        for _ in range(250):
            c.select(rng)
        out = capsys.readouterr().out
        printed = [ln for ln in out.splitlines() if "floor_fraction" in ln]
        assert len(printed) == 2                 # at selections 100 and 200
        assert c._floor_reports == 2

    def test_v1_never_prints_floor_fraction(self, tmp_path, capsys):
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v1")
        c.add(make_trace(seed=1), passing_result())
        c._seed_count = 0
        rng = random.Random(0)
        for _ in range(250):
            c.select(rng)
        assert "floor_fraction" not in capsys.readouterr().out

    def test_all_entries_still_reachable_under_v2(self, tmp_path):
        c = self._corpus(tmp_path, n=5)
        rng = random.Random(0)
        seen = set()
        for _ in range(500):
            seen.add(c.select(rng).seed)
        assert seen == {0, 1, 2, 3, 4}

    def test_high_score_entry_still_dominates_under_v2(self, tmp_path):
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v2")
        c.add(make_trace(seed=1), passing_result())
        c.add(make_trace(seed=2), failing_result(score=1000.0))
        c._seed_count = 0
        rng = random.Random(0)
        counts = {1: 0, 2: 0}
        for _ in range(200):
            counts[c.select(rng).seed] += 1
        assert counts[2] > counts[1]

    def test_warmup_round_robin_is_unaffected_by_policy(self, tmp_path):
        seeds = tmp_path / "seeds"
        seeds.mkdir()
        for i in range(3):
            (seeds / f"seed_{i}.json").write_text(make_trace(seed=i).to_json())
        c = Corpus(str(seeds), str(tmp_path / "corpus"), warmup_rounds=1,
                   selection_policy="v2")
        c.load_seeds()
        rng = random.Random(0)
        assert [c.select(rng).seed for _ in range(3)] == [0, 1, 2]

    def test_empty_corpus_returns_none_under_v2(self, tmp_path):
        c = Corpus(str(tmp_path / "s"), str(tmp_path / "c"),
                   selection_policy="v2")
        assert c.select(random.Random(0)) is None


# ---------------------------------------------------------------------------
# Workstream 1a: transition coverage as the novelty primitive
#
# StateAFL pattern (Natella EMSE 2022, via the survey scan lens A idea 2): count
# state TRANSITIONS, not raw state vectors.  On a fixed config the distinct
# fingerprints saturate at 8 within a few hundred iterations
# (an internal analysis note), but the *sequence* of states a
# trace induces is far richer.
# ---------------------------------------------------------------------------

def v2_corpus(tmp_path) -> Corpus:
    return Corpus(seeds_dir=str(tmp_path / "seeds"),
                  corpus_dir=str(tmp_path / "corpus"),
                  fingerprint_version=2)


class TestTransitionNovelty:
    def test_first_edge_is_novel(self, tmp_path):
        c = v2_corpus(tmp_path)
        a, b = make_fp(0), make_fp(1)
        assert c.is_novel_transition(a, b) is True

    def test_edge_is_not_novel_after_it_has_been_taken(self, tmp_path):
        c = v2_corpus(tmp_path)
        a, b = make_fp(0), make_fp(1)
        c.add(make_trace(), passing_result(), fingerprint=b, prev_fingerprint=a)
        assert c.is_novel_transition(a, b) is False

    def test_edges_are_directed(self, tmp_path):
        c = v2_corpus(tmp_path)
        a, b = make_fp(0), make_fp(1)
        c.add(make_trace(), passing_result(), fingerprint=b, prev_fingerprint=a)
        assert c.is_novel_transition(b, a) is True

    def test_a_seen_state_reached_from_a_new_predecessor_is_novel(self, tmp_path):
        """The whole point: the state saturates, the edge does not."""
        c = v2_corpus(tmp_path)
        a, b, d = make_fp(0), make_fp(1), make_fp(2)
        c.add(make_trace(1), passing_result(), fingerprint=b, prev_fingerprint=a)
        assert c.is_novel(b) is False            # state already seen
        assert c.is_novel_transition(d, b) is True   # edge into it is not

    def test_none_predecessor_is_a_real_key(self, tmp_path):
        c = v2_corpus(tmp_path)
        b = make_fp(1)
        assert c.is_novel_transition(None, b) is True
        c.add(make_trace(), passing_result(), fingerprint=b, prev_fingerprint=None)
        assert c.is_novel_transition(None, b) is False

    def test_none_current_fingerprint_is_never_novel(self, tmp_path):
        c = v2_corpus(tmp_path)
        assert c.is_novel_transition(make_fp(0), None) is False
        assert c.transition_novelty(make_fp(0), None) == 0.0

    def test_bonus_decays_like_every_other_signal(self, tmp_path):
        from fuzzer.core.corpus import _TRANSITION_NOVELTY_SCORE, _rarity_decay
        c = v2_corpus(tmp_path)
        a, b = make_fp(0), make_fp(1)
        seen = []
        for _ in range(4):
            seen.append(c.transition_novelty(a, b))
            c.add(make_trace(), passing_result(), fingerprint=b,
                  prev_fingerprint=a, save=False)
        assert seen == [_TRANSITION_NOVELTY_SCORE * _rarity_decay(n)
                        for n in range(4)]
        assert seen[0] > seen[1] > seen[2] > seen[3]

    def test_add_returns_true_on_a_novel_edge_alone(self, tmp_path):
        c = v2_corpus(tmp_path)
        a, b = make_fp(0), make_fp(1)
        c.add(make_trace(1), passing_result(), fingerprint=a)
        c.add(make_trace(2), passing_result(), fingerprint=b)
        # Both states are now seen, so only the edge can fire.
        assert c.is_novel(a) is False and c.is_novel(b) is False
        assert c.add(make_trace(3), passing_result(), fingerprint=a,
                     prev_fingerprint=b) is True

    def test_add_returns_false_when_neither_state_nor_edge_is_new(self, tmp_path):
        c = v2_corpus(tmp_path)
        a, b = make_fp(0), make_fp(1)
        c.add(make_trace(1), passing_result(), fingerprint=b, prev_fingerprint=a)
        assert c.add(make_trace(2), passing_result(), fingerprint=b,
                     prev_fingerprint=a) is False

    def test_edge_bonus_reaches_the_entry_score(self, tmp_path):
        from fuzzer.core.corpus import _TRANSITION_NOVELTY_SCORE
        c = v2_corpus(tmp_path)
        a, b = make_fp(0), make_fp(1)
        c.add(make_trace(1), passing_result(), fingerprint=a)
        c.add(make_trace(2), passing_result(), fingerprint=b)
        c.add(make_trace(3), passing_result(), fingerprint=a, prev_fingerprint=b)
        assert c._entries[-1].score == pytest.approx(_TRANSITION_NOVELTY_SCORE)

    def test_oracle_score_still_wins_over_a_fresh_edge(self, tmp_path):
        c = v2_corpus(tmp_path)
        c.add(make_trace(), failing_result(score=9.0), fingerprint=make_fp(1),
              prev_fingerprint=make_fp(0))
        assert c._entries[-1].score == 9.0

    def test_edges_are_counted_but_not_scored_under_v1(self, tmp_path):
        """v1 is byte-exact: the diagnostic counter fills, the score does not."""
        c = make_corpus(str(tmp_path / "seeds"), str(tmp_path / "corpus"))
        assert c.fingerprint_version == 1
        a, b = make_fp(0), make_fp(1)
        c.add(make_trace(1), passing_result(), fingerprint=a)
        c.add(make_trace(2), passing_result(), fingerprint=b)
        # Both states seen; under v1 a fresh edge must not make this "novel".
        assert c.add(make_trace(3), passing_result(), fingerprint=a,
                     prev_fingerprint=b) is False
        # ...but the edge was still counted, so transition coverage is
        # measurable without changing behaviour.
        assert len(c._seen_transitions) == 3

    def test_distinct_edges_outnumber_distinct_states(self, tmp_path):
        c = v2_corpus(tmp_path)
        fps = [make_fp(i) for i in range(4)]
        prev = None
        rng = random.Random(3)
        for i in range(200):
            cur = fps[rng.randrange(4)]
            c.add(make_trace(i), passing_result(), fingerprint=cur,
                  prev_fingerprint=prev, save=False)
            prev = cur
        assert len(c._seen_fingerprints) == 4
        assert len(c._seen_transitions) > 4


# ---------------------------------------------------------------------------
# Workstream 1b(ii): variance-adaptive dimension weighting
#
# Zhu, ACM CSUR 2022 §3.5.1 "sensitive coverage": track each dimension's
# empirical entropy and DOWN-WEIGHT invariant dimensions rather than dropping
# them, re-probing so a dimension that starts moving recovers its weight.
# ---------------------------------------------------------------------------

class TestVarianceAdaptiveWeights:
    def _feed(self, c, n, fp_for):
        for i in range(n):
            c.add(make_trace(i), passing_result(), fingerprint=fp_for(i),
                  save=False)

    def test_weights_start_neutral(self, tmp_path):
        c = v2_corpus(tmp_path)
        assert c.dimension_weights() == [1.0] * len(c.dimension_weights())

    def test_no_reweight_before_the_first_reprobe(self, tmp_path):
        """No re-weight until the cadence fires (corpus.py:545-546).

        Repaired: this used to feed `_FP_WARMUP - 1` (99) observations and
        assert no re-probe.  That could not fail — the gate is
        `_fp_obs >= _FP_WARMUP and _fp_obs % _FP_REPROBE_EVERY == 0`, and the
        first n satisfying the modulo is 500, so **any** n < 500 passes and the
        `>= _FP_WARMUP` clause is unreachable as a gate.  Feeding one short of
        the cadence is the input that actually exercises it.
        """
        from fuzzer.core.corpus import _FP_WARMUP, _FP_REPROBE_EVERY
        # If this ever stops holding, the warmup clause becomes live and this
        # test needs the _FP_WARMUP - 1 case back.
        assert _FP_REPROBE_EVERY > _FP_WARMUP, (
            "the _FP_WARMUP clause at corpus.py:545 is now reachable")

        c = v2_corpus(tmp_path)
        self._feed(c, _FP_REPROBE_EVERY - 1, lambda i: make_fp(i % 4))
        assert c._dim_reprobes == 0
        assert c.dimension_weights() == [1.0] * len(c.dimension_weights())

    def test_first_reweight_happens_on_the_reprobe_cadence(self, tmp_path):
        from fuzzer.core.corpus import _FP_REPROBE_EVERY
        c = v2_corpus(tmp_path)
        self._feed(c, _FP_REPROBE_EVERY, lambda i: make_fp(i % 4))
        assert c._dim_reprobes == 1

    def test_constant_dimension_is_down_weighted_to_the_floor(self, tmp_path):
        from fuzzer.core.corpus import _DIM_WEIGHT_FLOOR, _FP_REPROBE_EVERY
        c = v2_corpus(tmp_path)
        # Only queue_depth_bucket moves; everything else is pinned.
        self._feed(c, _FP_REPROBE_EVERY, lambda i: make_fp(i % 4))
        assert c.dimension_weight("kv_peak_bucket") == _DIM_WEIGHT_FLOOR
        assert c.dimension_weight("had_preemption") == _DIM_WEIGHT_FLOOR
        # ...and the three dimensions the plan wanted deleted are down-weighted
        # by exactly the same mechanism instead of being removed.
        assert c.dimension_weight("preemption_count_bucket") == _DIM_WEIGHT_FLOOR
        assert c.dimension_weight("finished_count_bucket") == _DIM_WEIGHT_FLOOR
        assert c.dimension_weight("prefix_cache_hit_bucket") == _DIM_WEIGHT_FLOOR

    def test_a_dead_dimension_is_never_dropped(self, tmp_path):
        from fuzzer.core.corpus import _DIM_WEIGHT_FLOOR, _FP_REPROBE_EVERY
        c = v2_corpus(tmp_path)
        self._feed(c, _FP_REPROBE_EVERY, lambda i: make_fp(i % 4))
        assert all(w >= _DIM_WEIGHT_FLOOR for w in c.dimension_weights())

    def test_uniformly_moving_dimension_keeps_full_weight(self, tmp_path):
        from fuzzer.core.corpus import _FP_REPROBE_EVERY
        c = v2_corpus(tmp_path)
        self._feed(c, _FP_REPROBE_EVERY, lambda i: make_fp(i % 4))
        assert c.dimension_weight("queue_depth_bucket") == pytest.approx(1.0)

    def test_a_barely_moving_dimension_scores_near_the_floor(self, tmp_path):
        from fuzzer.core.corpus import _FP_REPROBE_EVERY
        c = v2_corpus(tmp_path)
        # kv_peak_bucket takes its second value once in 500 observations.
        self._feed(c, _FP_REPROBE_EVERY,
                   lambda i: make_fp(i % 4, kv_peak_bucket=1 if i == 3 else 0))
        w = c.dimension_weight("kv_peak_bucket")
        # "moved once in 500" is not "informative": normalised entropy is tiny,
        # so the weight sits at (or just above) the floor.
        assert 0.1 <= w < 0.15
        assert w < c.dimension_weight("queue_depth_bucket")

    def test_a_dimension_that_starts_moving_recovers_its_weight(self, tmp_path):
        """The re-probe is what makes down-weighting reversible."""
        from fuzzer.core.corpus import _DIM_WEIGHT_FLOOR, _FP_REPROBE_EVERY
        c = v2_corpus(tmp_path)
        self._feed(c, _FP_REPROBE_EVERY, lambda i: make_fp(0))
        assert c.dimension_weight("memory_delta_bucket") == _DIM_WEIGHT_FLOOR
        # New config: memory_delta now moves every iteration.
        self._feed(c, _FP_REPROBE_EVERY,
                   lambda i: make_fp(0, memory_delta_bucket=i % 4))
        assert c._dim_reprobes == 2
        assert c.dimension_weight("memory_delta_bucket") == pytest.approx(1.0)

    def test_weights_come_from_the_window_not_from_lifetime_counts(self, tmp_path):
        """A dimension dead for 10 windows still recovers in one."""
        from fuzzer.core.corpus import _FP_REPROBE_EVERY
        c = v2_corpus(tmp_path)
        self._feed(c, _FP_REPROBE_EVERY * 10, lambda i: make_fp(0))
        self._feed(c, _FP_REPROBE_EVERY,
                   lambda i: make_fp(0, finished_count_bucket=i % 4))
        assert c.dimension_weight("finished_count_bucket") == pytest.approx(1.0)

    def test_per_dimension_novelty_rescues_a_saturated_tuple(self, tmp_path):
        """A fresh value in one dimension scores even when the tuple is stale."""
        c = v2_corpus(tmp_path)
        stale = make_fp(0)
        for i in range(400):
            c.add(make_trace(i), passing_result(), fingerprint=stale, save=False)
        whole = c.fingerprint_novelty(stale)
        fresh_dim = make_fp(0, kv_peak_bucket=3)
        assert c.fingerprint_novelty(fresh_dim) > whole

    def test_per_dimension_term_cannot_outrank_a_genuinely_new_tuple(self, tmp_path):
        c = v2_corpus(tmp_path)
        stale = make_fp(0)
        for i in range(400):
            c.add(make_trace(i), passing_result(), fingerprint=stale, save=False)
        brand_new = make_fp(3, kv_peak_bucket=2, memory_delta_bucket=1)
        assert c.fingerprint_novelty(brand_new) == pytest.approx(1.0)
        # The per-dimension term on its own is a weighted *mean* over 11
        # dimensions, so one fresh value in an otherwise saturated fingerprint
        # can never reach the 1.0 a genuinely new tuple pays.
        assert c._per_dimension_novelty(make_fp(0, kv_peak_bucket=3)) < 0.5

    def test_v2_fingerprint_novelty_is_never_below_v1(self, tmp_path):
        """max(whole, perdim) keeps v1 as a lower bound, by construction."""
        from fuzzer.core.corpus import _NOVELTY_SCORE_FLOOR, _rarity_decay
        c = v2_corpus(tmp_path)
        rng = random.Random(9)
        for i in range(300):
            fp = make_fp(rng.randrange(4), memory_delta_bucket=rng.randrange(4))
            v1_value = _NOVELTY_SCORE_FLOOR * _rarity_decay(c._seen_fingerprints[fp])
            assert c.fingerprint_novelty(fp) >= v1_value - 1e-12
            c.add(make_trace(i), passing_result(), fingerprint=fp, save=False)

    def test_v1_fingerprint_novelty_is_the_whole_tuple_only(self, tmp_path):
        from fuzzer.core.corpus import _NOVELTY_SCORE_FLOOR, _rarity_decay
        c = make_corpus(str(tmp_path / "seeds"), str(tmp_path / "corpus"))
        fp = make_fp(0)
        for n in range(5):
            assert c.fingerprint_novelty(fp) == pytest.approx(
                _NOVELTY_SCORE_FLOOR * _rarity_decay(n))
            c.add(make_trace(n), passing_result(), fingerprint=fp, save=False)

    def test_version_is_not_treated_as_a_behavioural_dimension(self, tmp_path):
        from fuzzer.core.corpus import _FP_N_DIMS
        import dataclasses
        from fuzzer.monitor import SchedulerFingerprint
        assert _FP_N_DIMS == len(dataclasses.fields(SchedulerFingerprint)) - 1
        c = v2_corpus(tmp_path)
        with pytest.raises(ValueError):
            c.dimension_weight("version")


# ---------------------------------------------------------------------------
# Fingerprint version resolution + the persisted-tuple migration
# ---------------------------------------------------------------------------

class TestFingerprintVersionResolution:
    def test_default_is_v1(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GRIEF_FINGERPRINT_VERSION", raising=False)
        assert make_corpus(str(tmp_path), str(tmp_path)).fingerprint_version == 1

    def test_explicit_argument_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_FINGERPRINT_VERSION", "1")
        c = Corpus(seeds_dir=str(tmp_path), corpus_dir=str(tmp_path),
                   fingerprint_version=2)
        assert c.fingerprint_version == 2

    def test_env_var_is_read_when_no_argument(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_FINGERPRINT_VERSION", "2")
        assert make_corpus(str(tmp_path), str(tmp_path)).fingerprint_version == 2

    def test_unknown_version_falls_back_to_v1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_FINGERPRINT_VERSION", "7")
        assert make_corpus(str(tmp_path), str(tmp_path)).fingerprint_version == 1

    def test_garbage_env_var_falls_back_to_v1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_FINGERPRINT_VERSION", "banana")
        assert make_corpus(str(tmp_path), str(tmp_path)).fingerprint_version == 1

    def test_empty_env_var_falls_back_to_v1(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GRIEF_FINGERPRINT_VERSION", "")
        assert make_corpus(str(tmp_path), str(tmp_path)).fingerprint_version == 1


class TestFingerprintTuplePad:
    """Persisted `_corpus_fingerprint` tuples migrate forward, never sideways.

    Field order is append-only, so a 7- or 8-element tuple written by an older
    build pads with the dataclass defaults — crucially `version=1`, not 0.
    """

    def _write(self, corpus_dir, seed, key, score=3.0):
        import json as _j
        t = make_trace(seed=seed)
        data = t.to_dict()
        data["_corpus_score"] = score
        data["_corpus_fingerprint"] = key
        data["_corpus_preemption_delta"] = None
        (corpus_dir / f"seed_{seed}.json").write_text(_j.dumps(data))

    def _load(self, tmp_path, *pairs):
        seeds_dir = tmp_path / "seeds"; corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(exist_ok=True); corpus_dir.mkdir(exist_ok=True)
        for seed, key in pairs:
            self._write(corpus_dir, seed, key)
        c = Corpus(str(seeds_dir), str(corpus_dir))
        c.load_seeds()
        return c

    def test_7_element_tuple_pads_forward(self, tmp_path):
        c = self._load(tmp_path, (60, [True, False, 1, 2, 1, 0, 2]))
        fp = c._entries[0].fingerprint
        assert fp.prefix_cache_hit_bucket == 0
        assert fp.version == 1
        assert (fp.ttft_spread_bucket, fp.concurrency_peak_bucket,
                fp.error_class_bucket) == (0, 0, 0)

    def test_8_element_tuple_pads_forward(self, tmp_path):
        c = self._load(tmp_path, (61, [True, False, 1, 2, 1, 0, 2, 3]))
        fp = c._entries[0].fingerprint
        assert fp.prefix_cache_hit_bucket == 3
        assert fp.version == 1   # NOT 0 — padded with the field default
        assert fp.error_class_bucket == 0

    def test_a_padded_8_tuple_equals_a_live_v1_fingerprint(self, tmp_path):
        """The migration is value-preserving, which is what byte-exact means."""
        from fuzzer.monitor import SchedulerFingerprint
        c = self._load(tmp_path, (62, [True, False, 1, 2, 1, 0, 2, 3]))
        live = SchedulerFingerprint(True, False, 1, 2, 1, 0, 2, 3)
        assert c._entries[0].fingerprint == live
        assert c.is_novel(live) is False   # the reload prepopulated the counter

    def test_full_12_element_v2_tuple_round_trips(self, tmp_path):
        c = self._load(tmp_path, (63, [False, True, 3, 4, 0, 1, 0, 0, 2, 5, 4, 3]))
        fp = c._entries[0].fingerprint
        assert fp.version == 2
        assert (fp.ttft_spread_bucket, fp.concurrency_peak_bucket,
                fp.error_class_bucket) == (5, 4, 3)

    def test_a_v1_and_a_v2_tuple_are_different_buckets(self, tmp_path):
        c = self._load(tmp_path,
                       (64, [False, False, 0, 0, 0, 0, 0, 0]),
                       (65, [False, False, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]))
        assert c.size() == 2

    def test_a_too_short_tuple_is_skipped_not_fatal(self, tmp_path):
        c = self._load(tmp_path, (66, [True, False, 1]))
        assert c.size() == 0

    def test_save_round_trips_a_v2_fingerprint(self, tmp_path):
        from fuzzer.monitor import SchedulerFingerprint
        seeds_dir = tmp_path / "seeds"; corpus_dir = tmp_path / "corpus"
        seeds_dir.mkdir(); corpus_dir.mkdir()
        fp = SchedulerFingerprint(False, False, 1, 0, 0, 0, 0, 0,
                                  version=2, ttft_spread_bucket=3,
                                  concurrency_peak_bucket=4,
                                  error_class_bucket=1)
        c = Corpus(str(seeds_dir), str(corpus_dir), fingerprint_version=2)
        c.add(make_trace(seed=70), failing_result(), fingerprint=fp)
        c2 = Corpus(str(seeds_dir), str(corpus_dir), fingerprint_version=2)
        c2.load_seeds()
        assert c2._entries[0].fingerprint == fp

    def test_reload_prepopulates_per_dimension_counters(self, tmp_path):
        """A reloaded corpus must not make already-seen dimension values fresh."""
        c = self._load(tmp_path, (67, [True, False, 1, 2, 1, 0, 2, 3]))
        assert all(sum(ctr.values()) == 1 for ctr in c._dim_counters)
        assert c._dim_counters[0][True] == 1        # had_preemption
        # ...but the re-probe window is deliberately left empty: the weights
        # must describe this run's server, not a corpus from a prior config.
        assert sum(c._dim_window[0].values()) == 0
        assert c._fp_obs == 0
