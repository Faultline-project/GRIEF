"""
Corpus: stores Traces and selects seeds for the next mutation round.

Scoring is based on oracle signals: a trace that triggered a high-score
oracle result is retained and weighted more heavily for future selection.
Traces that produce no new signal are still kept but selected less often.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from dataclasses import astuple as _dc_astuple
from dataclasses import dataclass, field
from dataclasses import MISSING as _DC_MISSING
from dataclasses import fields as _dc_fields
from pathlib import Path
from typing import Callable, Optional

from ..trace import Trace
from .types import OracleResult
from ..monitor import (
    FP_VERSION_V1,
    FP_VERSION_V2,
    FP_VERSION_V3,
    MonitorSession,
    SchedulerFingerprint,
)

_NOVELTY_SCORE_FLOOR:      float = 1.0
_ERROR_NOVELTY_SCORE:      float = 2.0   # novel error class bonus
_PREEMPTION_NOVELTY_SCORE: float = 1.5   # novel preemption count bonus
_PREEMPTION_COUNT_CAP:     int   = 30    # counts above this are treated as equivalent
_SWAP_NOVELTY_SCORE:       float = 2.0   # novel swap count bonus (swap is more severe than preemption)
_SWAP_COUNT_CAP:           int   = 20    # swap counts above this are treated as equivalent
_KV_NOVELTY_SCORE:         float = 1.5   # novel KV cache fingerprint bonus
_RUN_COUNT_DECAY:          float = 0.05  # linear denominator penalty per run (20 runs → 2× down-weight)

# --- Rarity-weighted novelty (step 8 of an internal analysis note) ---
# Every novelty signal keeps a *hit count* instead of a membership set, and the
# bonus it pays decays with the number of times that key has already been seen
# in this session.  See _rarity_decay() for the exact curve and for why this
# particular curve was chosen.
_SELECTION_POLICY_ENV: str = "GRIEF_SELECTION_POLICY"

# --- Fingerprint v2: transition coverage + variance-adaptive dimensions ------
# (Workstream 1 of an internal analysis note.)  Everything in
# this block is inert under fingerprint_version == 1, which stays byte-exact.
_FINGERPRINT_VERSION_ENV: str = "GRIEF_FINGERPRINT_VERSION"

# Bonus paid by a first-seen (prev_fp -> cur_fp) edge.  Deliberately the same
# magnitude as _NOVELTY_SCORE_FLOOR and below _ERROR/_SWAP_NOVELTY_SCORE: with
# 11 live dimensions a first-time transition fires on almost every early
# iteration, so it must not be able to outrank a real oracle signal.  It decays
# through _rarity_decay exactly like the state bonus.
_TRANSITION_NOVELTY_SCORE: float = 1.0

# Variance-adaptive dimension weighting (Zhu, ACM CSUR 2022 §3.5.1 "sensitive
# coverage": track each dimension's empirical entropy and down-weight — never
# drop — the invariant ones, re-probing so a dimension that starts moving under
# a new config earns its weight back).
#
# _FP_WARMUP = 100 observations before the first re-weight.  At the ~2 000
# iterations/hour measured in an internal analysis note that is ~3
# minutes, and it is enough to catch any dimension that moves at >= 3 % of
# iterations with 95 % probability (1 - 0.97^100 = 0.95).  Below it every weight
# stays 1.0, i.e. exactly the unweighted behaviour.
#
# _FP_REPROBE_EVERY = 500 observations (~15 min, so ~4 re-probes inside a 30-min
# A/B slot and ~17 inside a 4.3 h campaign slot).  Weights are recomputed from
# the *window* since the last re-probe, not from lifetime counts, so a
# dimension that only starts moving in hour 3 (server restart, config rotation,
# a mutator finally reaching it) is re-weighted within one window instead of
# being drowned by 10 000 earlier constant observations.  A window of 500 keeps
# the entropy estimate stable enough not to oscillate on noise.
_FP_WARMUP:        int   = 100
_FP_REPROBE_EVERY: int   = 500
# A dead dimension is down-weighted 10x, never silenced.
_DIM_WEIGHT_FLOOR: float = 0.1

# Field layout of SchedulerFingerprint, derived from the dataclass so the pad
# and the per-dimension bookkeeping cannot drift from monitor.py.
_FP_FIELDS:          tuple = tuple(f.name for f in _dc_fields(SchedulerFingerprint))
_FP_FIELD_DEFAULTS:  tuple = tuple(f.default for f in _dc_fields(SchedulerFingerprint))
_FP_ARITY:           int   = len(_FP_FIELDS)
# "version" is schema metadata, not a behavioural dimension: it is constant
# within a run and would only dilute the weighted mean.
_FP_META_FIELDS:     frozenset = frozenset({"version"})
_FP_DIM_INDICES:     tuple = tuple(i for i, n in enumerate(_FP_FIELDS)
                                   if n not in _FP_META_FIELDS)
_FP_N_DIMS:          int   = len(_FP_DIM_INDICES)

# --- select() anti-collapse, policy "v2" (step 9) ---
_SELECT_FLOOR_BASE:     float = 0.01   # the historical flat floor (policy "v1")
_SELECT_FLOOR_RARITY:   float = 0.5    # never-run entries floor at 0.01 + 0.5 = 0.51
_AGE_DECAY_PER_SELECT:  float = 1e-4   # 10 000 selections later, an entry is 2× down-weighted
_FLOOR_LOG_EVERY:       int   = 100    # print floor_fraction at most once per N selections


def _rarity_decay(count_before: int) -> float:
    """Multiplier in (0, 1] applied to a novelty bonus after *count_before* sightings.

    ``1 / sqrt(count_before + 1)`` — first sighting pays the full bonus, the 2nd
    pays 0.71x, the 4th 0.5x, the 256th 0.0625x.

    **Which decay, and why this one.**  The design note behind this step is
    self-contradictory: its *prose* quotes "a first sighting scores
    1.0, the 2nd 0.71, the 4th 0.5, the 256th 0.06", but the *code snippet*
    beside it rounds the count up to a power of two first
    (``bucket = 1 << n.bit_length()``), which yields 0.58 / 0.45 / 0.09 instead.
    An offline replay compares both (`hitcount` = the snippet,
    `invsqrt` = the prose) and finds them indistinguishable in weight entropy
    (11.195 vs 11.196 nats) — so the tie is broken on the two things that do
    differ:

      * the prose numbers (1.0 / 0.71 / 0.5 / 0.06) are the ones quoted in the
        plan and carried into an internal analysis note, and
      * plain ``1/sqrt(n+1)`` is monotone in *n*, while the power-of-two variant
        is a staircase that pays the 5th and the 8th sighting the same amount.

    So the prose (plain inverse-square-root) wins, and this function is
    numerically identical to the replay's `novelty_invsqrt`.

    **No lower cap on the decay.**  An internal analysis note warns that this
    curve falls below the `select()` floor of 0.01 after
    ~10 000 sightings of the same key, so in a long slot a saturated
    fingerprint decays back onto a flat floor.  We deliberately do *not* clamp
    the decay to e.g. 0.1x the bonus, for two reasons:

      1. Under ``selection_policy="v2"`` the floor is no longer flat — it is
         ``0.01 + 0.5/(1 + run_count)``, i.e. a least-run-first scheduler.  A
         novelty term that has decayed to nothing therefore hands the entry to
         the run-count scheduler, which is the intended anti-collapse
         behaviour, not to uniform noise.  A hard 0.1x cap would instead keep a
         stale fingerprint outranking a never-selected entry forever.
      2. Uncapped, this function is exactly the `invsqrt` scheme that was
         replayed offline, so the floor-fraction numbers measured in
         an internal analysis note transfer to the shipped
         code without re-running the study.

    Under the default ``selection_policy="v1"`` the decay is still never worse
    than the binary signal it replaces (which saturates after *one* sighting).
    """
    if count_before <= 0:
        return 1.0
    return 1.0 / math.sqrt(count_before + 1.0)


def _pad_fingerprint_tuple(key) -> tuple:
    """Forward-pad a persisted ``_corpus_fingerprint`` tuple to the current arity.

    The fingerprint is stored as ``list(dataclasses.astuple(fp))`` (``_save``)
    and rebuilt with ``SchedulerFingerprint(*key)``, so every schema growth has
    to migrate the short tuples already on disk:

      * 7 elements — written before ``prefix_cache_hit_bucket`` existed;
      * 8 elements — a v1 fingerprint;
      * 12 elements — a v1 or v2 fingerprint written by this code.

    Padding uses the dataclass's own field defaults rather than zeros, which
    matters for exactly one field: ``version`` defaults to 1, so an old 8-tuple
    reloads as a *v1* fingerprint (equal to the value the same trace would
    produce today under ``--fingerprint-version 1``) instead of a bogus
    ``version=0``.  Field order is append-only, so no positional remapping is
    needed and the three v2 dimensions pad to 0 = "unknown", the same convention
    the v1 dimensions already use for missing telemetry.

    Tuples shorter than the 7 mandatory fields raise ValueError; ``load_seeds``
    catches it and skips the file, as it already does for any malformed entry.
    """
    key = tuple(key)
    if len(key) >= _FP_ARITY:
        return key[:_FP_ARITY]
    pad = _FP_FIELD_DEFAULTS[len(key):_FP_ARITY]
    if any(d is _DC_MISSING for d in pad):
        raise ValueError(
            f"fingerprint tuple has {len(key)} elements; the first "
            f"{sum(1 for d in _FP_FIELD_DEFAULTS if d is _DC_MISSING)} "
            f"fields have no default and cannot be padded"
        )
    return key + tuple(pad)


def _normalized_entropy(counter: Counter) -> float:
    """Shannon entropy of a dimension's observed values, normalised to [0, 1].

    Normalised by ``ln(K)`` where K is the number of *distinct* values observed
    for that dimension, so the score answers "given the alphabet this dimension
    actually uses, how evenly does it use it?".  A dimension stuck on one value
    scores 0 (K = 1); a dimension that splits evenly between two values scores
    1.0, and one that is 99.99 % a single value scores near 0 even though it
    technically moved once.  That is the property we want: "invariant" here
    means "carries no information", not "has cardinality 1".
    """
    total = sum(counter.values())
    if total <= 0 or len(counter) <= 1:
        return 0.0
    ent = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent / math.log(len(counter))


def _compute_pressure(session: MonitorSession) -> float:
    """
    Derive a telemetry-guided fitness score from Level 2 monitor data.
    Higher = trace pushed the scheduler/allocator harder → select more often.

      kv_peak     : 0–5 pts  (5 at 100% KV utilisation)
      preemptions : 0.5 pts each
      waiting q   : 0.3 pts per request stuck in queue
      swapped     : 1.0 pt per swapped request
      prefix miss : 0–3 pts  (3 at full cache miss)
    """
    score = 0.0
    kv_peak = session.peak_kv_cache_pct()
    if kv_peak is not None:
        score += kv_peak / 20.0
    delta_pre = session.preemption_delta()
    if delta_pre is not None:
        score += delta_pre * 0.5
    peak_q = session.peak_waiting_queue()
    if peak_q is not None:
        score += peak_q * 0.3
    peak_sw = session.peak_swapped()
    if peak_sw:
        score += peak_sw * 1.0
    min_hit = session.min_prefix_cache_hit_rate()
    if min_hit is not None:
        score += max(0.0, (1.0 - min_hit) * 3.0)
    return score


@dataclass
class CorpusEntry:
    trace:              Trace
    score:              float                          = 0.0
    pressure_score:     float                          = 0.0  # Level 2 telemetry fitness
    findings:           list[OracleResult]             = field(default_factory=list)
    run_count:          int                            = 0
    fingerprint:        Optional[SchedulerFingerprint] = None
    preemption_delta:   Optional[int]                  = None
    # Value of Corpus._select_count when this entry was added.  Used only by
    # selection_policy="v2" to age-decay the score term (step 9).  Entries
    # loaded from disk keep 0, i.e. they are treated as oldest.
    added_at_select_count: int                         = 0
    # Peak swapped-request count on the run that produced this entry, mirroring
    # ``preemption_delta`` above.  It lives on the entry — not only as a _save()
    # argument — so that any *re*-save (``boost()``) can write the same value
    # back instead of clobbering the persisted one with null; that silently
    # destroyed swap-count prepopulation on the next campaign's reload.
    peak_swapped:       Optional[int]                  = None


class Corpus:
    """
    In-memory corpus with optional disk persistence.
    Seeds directory is loaded at startup; interesting traces are saved there.
    """

    def __init__(self,
                 seeds_dir:          str           = "./seeds",
                 corpus_dir:         str           = "./corpus",
                 warmup_rounds:      int           = 4,
                 adapter_seeds_dir:  Optional[str] = None,
                 selection_policy:   Optional[str] = None,
                 fingerprint_version: Optional[int] = None):
        """selection_policy: "v1" (default) reproduces the historical select()
        weighting byte-exactly; "v2" enables the anti-collapse floor and age
        decay of step 9.  When None, the value of the environment variable
        GRIEF_SELECTION_POLICY is used, falling back to "v1".

        fingerprint_version: 1 (default) is the historical whole-tuple novelty
        over the 8-dimension fingerprint — byte-exact.  2 enables Workstream 1:
        transition coverage ((prev_fp -> cur_fp) edges) as a novelty signal
        beside the states, plus variance-adaptive per-dimension weighting.  When
        None, GRIEF_FINGERPRINT_VERSION is used, falling back to 1.  The runner
        reads ``corpus.fingerprint_version`` back to decide whether to build a
        v1 or a v2 fingerprint in the first place.

        Both env vars are the same stopgap: RunnerConfig has no field for
        either, and the Corpus is constructed inside fuzzer/core/runner.py.
        `--selection-policy` / `--fingerprint-version` on the backend CLIs set
        them.
        """
        self.seeds_dir          = Path(seeds_dir)
        self.corpus_dir         = Path(corpus_dir)
        self.adapter_seeds_dir  = Path(adapter_seeds_dir) if adapter_seeds_dir else None
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.warmup_rounds  = warmup_rounds
        self.selection_policy = self._resolve_selection_policy(selection_policy)
        self.fingerprint_version = self._resolve_fingerprint_version(
            fingerprint_version)
        self._entries:           list[CorpusEntry] = []
        self._seed_count:        int               = 0   # set after load_seeds()
        self._select_count:      int               = 0   # total calls to select()
        # Hit counts, not membership sets: every novelty signal pays a bonus
        # that decays with the number of prior sightings (see _rarity_decay).
        self._seen_fingerprints:      Counter = Counter()
        self._seen_error_sigs:        Counter = Counter()
        self._seen_preemption_counts: Counter = Counter()
        self._seen_swap_counts:       Counter = Counter()
        self._seen_kv_fingerprints:   Counter = Counter()
        # Transition coverage (StateAFL pattern via Natella EMSE 2022 / Zhang
        # CSUR 2025 §5): the novelty primitive is the *edge* (prev_fp -> cur_fp),
        # not the state.  On a fixed config the distinct states saturate at ~8
        # (an internal analysis note) but the sequence of
        # states a trace induces is far richer.  Counted under both fingerprint
        # versions (it is a free diagnostic); only *scored* under v2.
        self._seen_transitions:       Counter = Counter()
        # Variance-adaptive per-dimension bookkeeping.  Two structures on
        # purpose: _dim_counters is lifetime (it feeds the per-dimension rarity
        # term, which must not reset or every value would look fresh again every
        # window), _dim_window is reset at every re-probe (it feeds the weights,
        # which must reflect *recent* behaviour so a dimension can recover).
        self._dim_counters: list[Counter] = [Counter() for _ in range(_FP_N_DIMS)]
        self._dim_window:   list[Counter] = [Counter() for _ in range(_FP_N_DIMS)]
        self._dim_weights:  list[float]   = [1.0] * _FP_N_DIMS
        self._fp_obs:       int = 0   # fingerprints observed since construction
        self._dim_reprobes: int = 0   # number of completed re-weightings
        # select() collapse diagnostics (step 9).  Updated on every weighted
        # selection under both policies; only printed under "v2".
        self._floor_fraction:  float = 0.0   # share of weights clamped to the floor
        self._floor_count:     int   = 0     # entries clamped in the last selection
        self._floor_reports:   int   = 0     # number of floor_fraction lines printed

    @staticmethod
    def _resolve_selection_policy(value: Optional[str]) -> str:
        policy = (value if value is not None
                  else os.environ.get(_SELECTION_POLICY_ENV, "v1")) or "v1"
        policy = policy.strip().lower()
        if policy not in ("v1", "v2"):
            print(f"[corpus] unknown selection_policy {policy!r} — using 'v1'")
            return "v1"
        return policy

    @staticmethod
    def _resolve_fingerprint_version(value: Optional[int]) -> int:
        raw = (value if value is not None
               else os.environ.get(_FINGERPRINT_VERSION_ENV, ""))
        if raw is None or raw == "":
            return FP_VERSION_V1
        try:
            version = int(raw)
        except (TypeError, ValueError):
            print(f"[corpus] unknown fingerprint_version {raw!r} — using 1")
            return FP_VERSION_V1
        if version not in (FP_VERSION_V1, FP_VERSION_V2, FP_VERSION_V3):
            print(f"[corpus] unknown fingerprint_version {version!r} — using 1")
            return FP_VERSION_V1
        return version

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_seeds(self) -> int:
        """Load JSON traces from seeds_dir and corpus_dir. Returns count loaded.

        Seeds are loaded first (they anchor the warmup round-robin). Corpus
        entries saved from prior runs are loaded after with a small score
        bonus so the selector prefers them over fresh seeds during weighted
        selection.
        """
        count = 0
        seen_seed_ints: set[int] = set()
        seed_dirs = [self.seeds_dir]
        if self.adapter_seeds_dir is not None and self.adapter_seeds_dir.is_dir():
            seed_dirs.append(self.adapter_seeds_dir)
        for seed_dir in seed_dirs:
            for path in sorted(seed_dir.glob("*.json")):
                try:
                    trace = Trace.from_json(path.read_text())
                    if trace.seed in seen_seed_ints:
                        continue
                    seen_seed_ints.add(trace.seed)
                    self._entries.append(CorpusEntry(trace=trace))
                    count += 1
                except Exception as exc:
                    print(f"[corpus] skipping seed {path.name}: {exc}")
        self._seed_count = count

        corpus_count = 0
        if self.corpus_dir != self.seeds_dir:
            # AFL-style top-rated culling: for each fingerprint bucket keep
            # only the highest-scored entry.  Entries without a saved
            # fingerprint go into a separate "no-fp" bucket and are capped.
            import json as _json
            # fingerprint_key -> (score, trace, preemption_delta, peak_swapped).
            # The two telemetry counts are carried onto the rebuilt CorpusEntry,
            # not just consumed into the _seen_* counters: _save() persists them
            # from the entry, so an entry that is re-saved after reload (via
            # boost()) would otherwise write null over its own on-disk values.
            best: dict = {}
            no_fp: list = []
            for path in sorted(self.corpus_dir.glob("*.json")):
                try:
                    data  = _json.loads(path.read_text())
                    score = float(data.pop("_corpus_score", _NOVELTY_SCORE_FLOOR))
                    fp_raw = data.pop("_corpus_fingerprint", None)
                    pre_delta = data.pop("_corpus_preemption_delta", None)
                    peak_swapped = data.pop("_corpus_peak_swapped", None)
                    trace = Trace.from_dict(data)
                    if pre_delta is not None and int(pre_delta) > 0:
                        self._seen_preemption_counts[
                            min(int(pre_delta), _PREEMPTION_COUNT_CAP)
                        ] += 1
                    if peak_swapped is not None and int(peak_swapped) > 0:
                        self._seen_swap_counts[
                            min(int(peak_swapped), _SWAP_COUNT_CAP)
                        ] += 1
                    if fp_raw is not None:
                        # Backward compat: 7-element tuples predate
                        # prefix_cache_hit_bucket, 8-element tuples predate the
                        # v2 fields.  Both pad forward with the dataclass
                        # defaults (version -> 1).
                        key = _pad_fingerprint_tuple(fp_raw)
                        prev_score = best.get(key, (-1,))[0]
                        if score > prev_score:
                            best[key] = (score, trace, pre_delta, peak_swapped)
                    else:
                        no_fp.append((score, trace, pre_delta, peak_swapped))
                except Exception as exc:
                    print(f"[corpus] skipping corpus {path.name}: {exc}")

            for key, (score, trace, pre_delta, peak_swapped) in best.items():
                fp = SchedulerFingerprint(*key)
                self._entries.append(CorpusEntry(trace=trace, score=score,
                                                 fingerprint=fp,
                                                 preemption_delta=pre_delta,
                                                 peak_swapped=peak_swapped))
                self._seen_fingerprints[fp] += 1
                # Lifetime per-dimension counts are prepopulated like
                # _seen_fingerprints so a reloaded corpus does not make every
                # already-seen dimension value look fresh.  The re-probe
                # *window* is deliberately not prepopulated: the weights must
                # describe what this run's server is doing now.
                self._register_fingerprint_dims(fp, window=False)
                corpus_count += 1

            # Fallback bucket: cap at 200, prefer highest scores
            no_fp.sort(key=lambda t: t[0], reverse=True)
            for score, trace, pre_delta, peak_swapped in no_fp[:200]:
                self._entries.append(CorpusEntry(trace=trace, score=score,
                                                 preemption_delta=pre_delta,
                                                 peak_swapped=peak_swapped))
                corpus_count += 1

        if corpus_count:
            print(f"[corpus] loaded {corpus_count} corpus entries "
                  f"(deduplicated by fingerprint) from {self.corpus_dir}")
        return count

    # ------------------------------------------------------------------
    # Adding: novelty as rarity-weighted hit counts (step 8)
    #
    # Each *_novelty() method returns the bonus that *this* observation earns:
    # the full constant on a first sighting, multiplied by _rarity_decay() of
    # the number of prior sightings thereafter.  The is_novel*() predicates are
    # thin "count == 0" wrappers, kept so add()'s boolean return value ("did
    # any signal fire for the first time?"), runner.py's `[novel ...]` tag and
    # the existing tests all keep their original meaning.
    # ------------------------------------------------------------------

    def fingerprint_novelty(self, fp: Optional[SchedulerFingerprint]) -> float:
        """Bonus earned by this scheduler fingerprint (0.0 if there is none).

        Under ``fingerprint_version == 1`` this is the whole-tuple rarity bonus
        and nothing else — byte-exact with the pre-Workstream-1 behaviour.

        Under version 2 it is ``max(whole_tuple, per_dimension)``, where the
        per-dimension term is a variance-weighted mean of each dimension's own
        rarity.  Keeping the whole-tuple term as a lower bound means v2 can only
        ever score a trace *higher* than v1 would, never lower; the per-dimension
        term is what lets a single moving dimension earn novelty on a tuple that
        has already been seen thousands of times — the "sensitive coverage" idea
        from Zhu, ACM CSUR 2022 §3.5.1.  It is self-limiting: with 11 dimensions
        one fresh value in an otherwise saturated fingerprint contributes at most
        ``w_i / sum(w)`` (roughly 0.1-0.3), well below the 1.0 a genuinely new
        tuple pays.
        """
        if fp is None:
            return 0.0
        whole = _NOVELTY_SCORE_FLOOR * _rarity_decay(self._seen_fingerprints[fp])
        if self.fingerprint_version < FP_VERSION_V2:
            return whole
        return max(whole, self._per_dimension_novelty(fp))

    def _per_dimension_novelty(self, fp: SchedulerFingerprint) -> float:
        """Variance-weighted mean of the per-dimension rarity decays."""
        values = _dc_astuple(fp)
        total_w = sum(self._dim_weights)
        if total_w <= 0:
            return 0.0
        acc = 0.0
        for slot, idx in enumerate(_FP_DIM_INDICES):
            acc += (self._dim_weights[slot]
                    * _rarity_decay(self._dim_counters[slot][values[idx]]))
        return _NOVELTY_SCORE_FLOOR * (acc / total_w)

    def transition_novelty(self,
                           prev_fp: Optional[SchedulerFingerprint],
                           cur_fp: Optional[SchedulerFingerprint]) -> float:
        """Bonus earned by the fingerprint *edge* ``prev_fp -> cur_fp``.

        ``prev_fp`` is None on the first iteration of a run; that is kept as a
        real key (the "entry" edge) rather than skipped, so the very first
        fingerprint of a slot is still credited once and then decays like any
        other edge.  Telemetry is polled at ~1 Hz and the fingerprint collapses a
        whole trace into one value, so the only granularity available is
        iteration-to-iteration — there is no per-reply telemetry to build a
        finer state machine from.
        """
        if cur_fp is None:
            return 0.0
        key = (prev_fp, cur_fp)
        return _TRANSITION_NOVELTY_SCORE * _rarity_decay(
            self._seen_transitions[key])

    def dimension_weights(self) -> list[float]:
        """Current per-dimension weights, in ``_FP_DIM_INDICES`` order."""
        return list(self._dim_weights)

    def dimension_weight(self, name: str) -> float:
        """Current weight of one fingerprint dimension, by field name."""
        idx = _FP_FIELDS.index(name)
        return self._dim_weights[_FP_DIM_INDICES.index(idx)]

    def _register_fingerprint_dims(self, fp: SchedulerFingerprint,
                                   window: bool = True) -> None:
        """Count this fingerprint's per-dimension values; re-probe on cadence."""
        values = _dc_astuple(fp)
        for slot, idx in enumerate(_FP_DIM_INDICES):
            self._dim_counters[slot][values[idx]] += 1
            if window:
                self._dim_window[slot][values[idx]] += 1
        if not window:
            return
        self._fp_obs += 1
        if (self._fp_obs >= _FP_WARMUP
                and self._fp_obs % _FP_REPROBE_EVERY == 0):
            self._recompute_dim_weights()

    def _recompute_dim_weights(self) -> None:
        """Re-weight every dimension from the observations in this window.

        ``w_i = clamp(normalized_entropy(window_i), _DIM_WEIGHT_FLOOR, 1.0)``.
        The window is then cleared, which is what makes this a *re-probe* rather
        than a one-off warm-up measurement: weights track the current server
        config, and a dimension that was dead for the first three hours recovers
        its weight one window after it starts moving.  Nothing is ever dropped —
        the floor guarantees a dead dimension keeps 10 % of a live one's say.
        """
        for slot in range(_FP_N_DIMS):
            h = _normalized_entropy(self._dim_window[slot])
            self._dim_weights[slot] = min(1.0, max(_DIM_WEIGHT_FLOOR, h))
            self._dim_window[slot] = Counter()
        self._dim_reprobes += 1

    def error_novelty(self, sig: Optional[str]) -> float:
        """Bonus earned by this error signature (0.0 if there is none)."""
        if sig is None:
            return 0.0
        return _ERROR_NOVELTY_SCORE * _rarity_decay(self._seen_error_sigs[sig])

    def swap_novelty(self, peak_swapped: Optional[int]) -> float:
        """Bonus earned by this peak swap count.

        Swap events are more severe than preemptions — they indicate the
        scheduler had to spill KV blocks to CPU memory.  Counts above
        _SWAP_COUNT_CAP are collapsed to one bucket.
        """
        if not peak_swapped:
            return 0.0
        key = min(peak_swapped, _SWAP_COUNT_CAP)
        return _SWAP_NOVELTY_SCORE * _rarity_decay(self._seen_swap_counts[key])

    def preemption_novelty(self, delta: Optional[int]) -> float:
        """Bonus earned by this preemption count.

        Preemptions are the precondition for KV block lifecycle bugs — a trace
        that causes a rarely-seen count of preemptions is worth keeping even
        when no oracle fires, because it exercises the block-free/reallocate
        path in an under-explored way.  Counts above _PREEMPTION_COUNT_CAP are
        collapsed to one bucket so the key space stays small.
        """
        if not delta:
            return 0.0
        key = min(delta, _PREEMPTION_COUNT_CAP)
        return _PREEMPTION_NOVELTY_SCORE * _rarity_decay(
            self._seen_preemption_counts[key])

    def kv_novelty(self, kv_fingerprint: Optional[str]) -> float:
        """Bonus earned by this KV-probe fingerprint (0.0 if there is none).

        KVProbeSummary.fingerprint buckets its counters to log2 before hashing
        (see fuzzer/kv_events.py::_build_fingerprint), so this key space is
        small and comparable to the scheduler fingerprint.  Before that change
        the fingerprint was near-unique per trace and this signal fired on
        roughly every other iteration — see
        an internal analysis note.
        """
        if not kv_fingerprint:
            return 0.0
        return _KV_NOVELTY_SCORE * _rarity_decay(
            self._seen_kv_fingerprints[kv_fingerprint])

    # --- first-sighting predicates (count == 0) ---

    def is_novel(self, fp: Optional[SchedulerFingerprint]) -> bool:
        """Return True if this fingerprint has not been seen this session."""
        if fp is None:
            return False
        return self._seen_fingerprints[fp] == 0

    def is_novel_transition(self,
                            prev_fp: Optional[SchedulerFingerprint],
                            cur_fp: Optional[SchedulerFingerprint]) -> bool:
        """Return True if this fingerprint edge has not been seen this session."""
        if cur_fp is None:
            return False
        return self._seen_transitions[(prev_fp, cur_fp)] == 0

    def is_novel_error(self, sig: Optional[str]) -> bool:
        """Return True if this error signature has not been seen this session."""
        if sig is None:
            return False
        return self._seen_error_sigs[sig] == 0

    def is_novel_swap(self, peak_swapped: Optional[int]) -> bool:
        """Return True if this exact peak swap count has not been seen this session."""
        if not peak_swapped:
            return False
        return self._seen_swap_counts[min(peak_swapped, _SWAP_COUNT_CAP)] == 0

    def is_novel_preemption(self, delta: Optional[int]) -> bool:
        """Return True if this exact preemption count has not been seen this session."""
        if not delta:
            return False
        return self._seen_preemption_counts[
            min(delta, _PREEMPTION_COUNT_CAP)] == 0

    def is_novel_kv(self, kv_fingerprint: Optional[str]) -> bool:
        """Return True if this KV fingerprint has not been seen this session."""
        if not kv_fingerprint:
            return False
        return self._seen_kv_fingerprints[kv_fingerprint] == 0

    def add(self, trace: Trace,
            findings: list[OracleResult],
            save: bool = True,
            fingerprint: Optional[SchedulerFingerprint] = None,
            error_sig: Optional[str] = None,
            session: Optional[MonitorSession] = None,
            kv_fingerprint: Optional[str] = None,
            min_score: float = 0.0,
            pressure_score: Optional[float] = None,
            prev_fingerprint: Optional[SchedulerFingerprint] = None) -> bool:
        """Add a trace to the corpus. Returns True if any novelty signal fired
        for the *first* time (an already-seen signal still contributes a decayed
        score, but does not make the trace "novel").

        min_score: floor applied after all novelty bonuses are computed.  Use
        this when a trace is "promising but unconfirmed" — it ensures the entry
        gets a non-zero selection weight without masquerading as a real finding.

        pressure_score: if provided, overrides _compute_pressure(session).  Use
        when there is no live MonitorSession but a synthetic pressure estimate is
        available (e.g. derived from a latency ratio).

        prev_fingerprint: the fingerprint of the *previous* iteration, forming
        the transition ``prev -> fingerprint``.  Edges are always counted (a free
        diagnostic: ``len(corpus._seen_transitions)`` is the transition-coverage
        number), but they only contribute to the score and to the "novel" return
        value under ``fingerprint_version == 2``, so v1 stays byte-exact.
        """
        oracle_score = sum(f.score for f in findings)

        pre_delta = session.preemption_delta() if session is not None else None
        peak_sw   = session.peak_swapped()     if session is not None else None

        v2 = self.fingerprint_version >= FP_VERSION_V2

        # First-sighting predicates, evaluated BEFORE the counters are bumped.
        novel      = self.is_novel(fingerprint)
        novel_err  = self.is_novel_error(error_sig)
        novel_pre  = self.is_novel_preemption(pre_delta)
        novel_swap = self.is_novel_swap(peak_sw)
        novel_kv   = self.is_novel_kv(kv_fingerprint)
        novel_trans = (self.is_novel_transition(prev_fingerprint, fingerprint)
                       if v2 else False)

        # Rarity-weighted bonuses, likewise from the counts before this add().
        # Score is still a max(), not a sum: a trace that trips five signals is
        # not five times as interesting as one that trips the rarest of them.
        signals = [
            oracle_score,
            self.fingerprint_novelty(fingerprint),
            self.error_novelty(error_sig),
            self.preemption_novelty(pre_delta),
            self.swap_novelty(peak_sw),
            self.kv_novelty(kv_fingerprint),
        ]
        if v2:
            signals.append(self.transition_novelty(prev_fingerprint, fingerprint))
        score = max(signals)

        # Register the sightings.
        if fingerprint is not None:
            self._seen_fingerprints[fingerprint] += 1
            self._seen_transitions[(prev_fingerprint, fingerprint)] += 1
            self._register_fingerprint_dims(fingerprint)
        if error_sig is not None:
            self._seen_error_sigs[error_sig] += 1
        if pre_delta:
            self._seen_preemption_counts[min(pre_delta, _PREEMPTION_COUNT_CAP)] += 1
        if peak_sw:
            self._seen_swap_counts[min(peak_sw, _SWAP_COUNT_CAP)] += 1
        if kv_fingerprint:
            self._seen_kv_fingerprints[kv_fingerprint] += 1

        novelty_count = sum([novel, novel_err, novel_pre, novel_swap, novel_kv,
                             novel_trans])
        if novelty_count > 1:
            trans_tag = f" trans={novel_trans}" if v2 else ""
            print(f"[corpus] multi-novelty trace: {novelty_count} signals "
                  f"(sched={novel} err={novel_err} pre={novel_pre} "
                  f"swap={novel_swap} kv={novel_kv}{trans_tag})")

        score = max(score, min_score)   # caller-supplied floor (e.g. weak latency signal)
        if pressure_score is not None:
            pressure = pressure_score
        else:
            pressure = _compute_pressure(session) if session is not None else 0.0
        entry = CorpusEntry(trace=trace, score=score, pressure_score=pressure,
                            findings=findings, fingerprint=fingerprint,
                            preemption_delta=pre_delta,
                            peak_swapped=peak_sw,
                            added_at_select_count=self._select_count)
        self._entries.append(entry)
        # Persistence gate — deliberately NOT "score > 0" alone.  A decayed
        # novelty bonus is a selection signal, not a reason to write a file: a
        # repeat sighting of the modal fingerprint now scores > 0, so a plain
        # "score > 0" test would persist essentially every iteration instead of
        # the historical ~4%.  This condition is exactly equivalent to the old
        # one (old score > 0 iff oracle sum > 0, or min_score > 0, or some
        # signal fired for the first time), so the on-disk corpus is unchanged
        # under fingerprint_version 1.  Under version 2 the transition signal is
        # part of "some signal fired for the first time", so early iterations of
        # a run persist more often — that is the point (a new edge is a new
        # behaviour worth keeping), and it self-limits as the edge set fills.
        first_sighting = (novel or novel_err or novel_pre or novel_swap
                          or novel_kv or novel_trans)
        if (save and score > 0
                and (oracle_score > 0 or min_score > 0 or first_sighting)):
            self._save(entry, peak_swapped=peak_sw)
        return first_sighting

    def boost(self, fingerprint: "SchedulerFingerprint", delta: float = 2.0) -> bool:
        """Retroactively raise the score of a corpus entry after soft-repro confirms it.

        Finds the first entry matching fingerprint, increments its score by delta,
        and re-saves it to disk. Returns True if found, False if no match exists
        (e.g., trace was never added to corpus or fingerprint was None).

        The re-save deliberately passes no ``peak_swapped``: ``_save`` falls
        back to ``entry.peak_swapped``.  It used to write the parameter default
        (``None``) straight over the real value, so any boosted entry lost its
        ``_corpus_peak_swapped`` and stopped prepopulating ``_seen_swap_counts``
        on the next campaign's reload.
        """
        return self.adjust(fingerprint, delta)

    def adjust(self, fingerprint: "SchedulerFingerprint", delta: float,
               floor: Optional[float] = None) -> bool:
        """Add ``delta`` (may be negative) to the first entry with ``fingerprint``.

        ``floor``, when given, clamps the new score from below; the
        confirmation ladder passes 0.0 when it down-weights a trace whose
        divergence turned out to be batch variance.  Returns True if an entry
        was found.  ``boost`` is ``adjust`` with a positive delta and no floor.
        """
        if fingerprint is None:
            return False
        for entry in self._entries:
            if entry.fingerprint == fingerprint:
                entry.score += delta
                if floor is not None and entry.score < floor:
                    entry.score = floor
                self._save(entry)
                return True
        return False

    def _save(self, entry: CorpusEntry,
              peak_swapped: Optional[int] = None) -> None:
        """Write *entry* to ``corpus_dir/seed_<seed>.json``.

        ``peak_swapped`` is remembered on the entry, so a later re-save that
        does not supply it (``boost()``) still persists the original value
        rather than nulling it out.
        """
        import dataclasses
        import json as _json
        if peak_swapped is None:
            peak_swapped = entry.peak_swapped
        else:
            entry.peak_swapped = peak_swapped
        path = self.corpus_dir / f"seed_{entry.trace.seed}.json"
        # Store score and fingerprint alongside the trace so we can
        # deduplicate and rank on reload without re-executing anything.
        data = entry.trace.to_dict()
        data["_corpus_score"]             = entry.score
        data["_corpus_fingerprint"]       = (
            list(dataclasses.astuple(entry.fingerprint))
            if entry.fingerprint is not None else None
        )
        data["_corpus_preemption_delta"]  = entry.preemption_delta
        data["_corpus_peak_swapped"]      = peak_swapped
        path.write_text(_json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        rng: random.Random,
        score_multiplier: Optional[Callable[["CorpusEntry"], float]] = None,
    ) -> Optional[Trace]:
        """
        Selection strategy:
        - Warmup phase (first warmup_rounds × seed_count calls): round-robin
          through seeds in order, ensuring every seed is exercised.
        - After warmup: weighted random by score (higher = picked more often).

        score_multiplier (optional): callable applied per-entry to bias
        selection. The floor is enforced AFTER multiplication, so an
        all-zero multiplier cannot cause random.choices() to raise.
        Default (None) preserves byte-exact pre-refactor behavior.

        Two policies (self.selection_policy):

        - "v1" (default): weight = max((score + 0.5·pressure) / (1 + 0.05·runs)
          [· multiplier], 0.01).  Byte-exact historical behaviour.
        - "v2": anti-collapse (step 9 of the dedup/novelty plan).  Two changes:
            * the constant 0.01 floor becomes 0.01 + 0.5/(1 + run_count), so
              the floor is itself a least-run-first scheduler instead of
              uniform noise — a never-selected entry floors at 0.51, an entry
              run 20× at 0.034;
            * the score term is age-decayed by
              1 / (1 + 1e-4·(select_count - entry.added_at_select_count)) so a
              10 k-entry corpus does not let hour-1 traces outvote hour-4 ones
              forever.

        Both policies record self._floor_fraction — the share of weights that
        were clamped to their floor, which is the number that makes collapse
        visible.  Only "v2" prints it (once per _FLOOR_LOG_EVERY selections).
        """
        if not self._entries:
            return None

        warmup_total = self.warmup_rounds * self._seed_count
        if self._seed_count > 0 and self._select_count < warmup_total:
            idx   = self._select_count % self._seed_count
            entry = self._entries[idx]
            entry.run_count    += 1
            self._select_count += 1
            return entry.trace

        self._select_count += 1
        # Combined weight: oracle score + half of telemetry pressure score,
        # divided by a linear run-count decay factor.
        # Decay: each run adds 5% to the denominator (20 runs → 2× down-weight,
        # 40 runs → 3× down-weight). The floor keeps every entry selectable.
        # Pressure is half-weighted so high-pressure non-bug traces stay useful
        # but don't monopolise selection over actual bug traces.
        raws:   list[float] = []
        floors: list[float] = []
        v2 = self.selection_policy == "v2"
        for e in self._entries:
            raw = (e.score + e.pressure_score * 0.5) / (1.0 + e.run_count * _RUN_COUNT_DECAY)
            if v2:
                age = self._select_count - e.added_at_select_count
                raw *= 1.0 / (1.0 + _AGE_DECAY_PER_SELECT * max(0, age))
            if score_multiplier is not None:
                raw *= score_multiplier(e)
            raws.append(raw)
            floors.append(
                _SELECT_FLOOR_BASE + _SELECT_FLOOR_RARITY / (1.0 + e.run_count)
                if v2 else _SELECT_FLOOR_BASE
            )

        weights = [r if r > f else f for r, f in zip(raws, floors)]
        self._floor_count    = sum(1 for r, f in zip(raws, floors) if r <= f)
        self._floor_fraction = self._floor_count / len(weights)
        if v2 and self._select_count % _FLOOR_LOG_EVERY == 0:
            self._floor_reports += 1
            print(f"[corpus] select #{self._select_count}: "
                  f"floor_fraction={self._floor_fraction:.3f} "
                  f"({self._floor_count}/{len(weights)} entries at their floor)")

        [entry]  = rng.choices(self._entries, weights=weights, k=1)
        entry.run_count += 1
        return entry.trace

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._entries)

    def findings_count(self) -> int:
        return sum(1 for e in self._entries if e.score > 0)

    def summary(self) -> dict:
        return {
            "total":    self.size(),
            "findings": self.findings_count(),
            "top_scores": sorted(
                [e.score for e in self._entries], reverse=True
            )[:5],
        }
