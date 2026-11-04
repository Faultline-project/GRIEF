"""
Core Oracle class: determines whether a scenario run constitutes a bug finding.

Each oracle takes a TraceResult + MonitorSession and returns an OracleResult
indicating pass/fail and a severity/classification.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from .types import (
    BugClass,
    OracleResult,
    TraceResult,
    _CORRUPT_RE,
    is_fatal_error,
    http_status_from_error,
    MIN_LATENCY_SAMPLE,
    _LATENCY_SPIKE_MIN_MS,
    NEG_INF_LOGPROB_SENTINEL,
    PRE_LOGPROB_RESTRICTORS,
)
from ..trace import ActionType


#: Lower bound a *finite* rival logprob must clear for the ``dm_sentinel``
#: predicate to accept it as evidence.  Real top-k candidates live within a few
#: tens of nats of 0; anything below this is either a second sentinel spelling
#: or a numerically broken row, neither of which is the composition defect.
_FINITE_ALT_FLOOR = -1000.0

#: vLLM's downstream grammar guard, verbatim from the 0.27.1 image:
#:   ``v1/structured_output/backend_xgrammar.py:163-166``
#:     "Failed to advance FSM for request %s for tokens %s. Please file an issue."
#:     (``%s`` for the token is a *single* int despite the plural)
#:   ``v1/core/sched/scheduler.py:1836-1839``
#:     "Unexpected: grammar rejected tokens %s for request %s. Terminating request."
#:     (``%s`` for the tokens is a ``list[int]``, so it renders as "[0]")
#: Note the two put request and token in opposite orders, and that the second
#: fires without the first whenever the matcher was already terminated
#: (``backend_xgrammar.py:158-159`` returns False with no log line).
#:
#: The client only ever sees vLLM's generic ``InternalServerError`` envelope, so
#: these are *corroboration* when a caller has stitched a server-log line into
#: the error string — never the thing that creates the finding.
import re as _re_oracles

_RE_GRAMMAR_GUARD_FSM = _re_oracles.compile(
    r"Failed to advance FSM for request (?P<req_id>\S+) "
    r"for tokens (?P<token>-?\d+)\. Please file an issue\.",
    _re_oracles.IGNORECASE)
_RE_GRAMMAR_GUARD_REJECT = _re_oracles.compile(
    r"Unexpected: grammar rejected tokens (?P<tokens>\[[^\]]*\]) "
    r"for request (?P<req_id>\S+)\. Terminating request\.",
    _re_oracles.IGNORECASE)


def grammar_guard_signature(text: str) -> Optional[str]:
    """``"fsm"`` / ``"reject"`` when *text* carries a grammar-guard log line."""
    if not text:
        return None
    if _RE_GRAMMAR_GUARD_FSM.search(text):
        return "fsm"
    if _RE_GRAMMAR_GUARD_REJECT.search(text):
        return "reject"
    return None


#: Server-side serialisation failures of non-finite floats (Starlette/JSON:
#: "Out of range float values are not JSON compliant: nan").
_NONFINITE_ERROR = _re_oracles.compile(r"not JSON compliant|Out of range float", _re_oracles.I)


class Oracle:
    """Run all oracles and return the most severe finding (or a pass)."""

    def __init__(self,
                 memory_leak_tolerance_mb:  float          = 100.0,
                 latency_ttft_threshold_ms: float          = 5000.0,
                 latency_multiplier:        float          = 3.0,
                 baseline_ttft_ms:          Optional[float] = None,
                 baseline_prompt_len:       int            = 32,
                 request_timeout_s:         float          = 30.0,
                 kv_block_leak_pct:         float          = 5.0,
                 repetition_window:         int            = 20,
                 repetition_threshold:      float          = 0.60,
                 token_rate_floor_tps:      float = 0.5,
                 token_rate_min_elapsed_s:  float = 5.0,
                 token_rate_min_tokens:     int   = 3,
                 ):
        self.memory_leak_tolerance_mb  = memory_leak_tolerance_mb
        self.latency_ttft_threshold_ms = latency_ttft_threshold_ms
        self.latency_multiplier        = latency_multiplier
        self.baseline_ttft_ms          = baseline_ttft_ms
        self.baseline_prompt_len       = baseline_prompt_len
        self.request_timeout_s         = request_timeout_s
        self.kv_block_leak_pct         = kv_block_leak_pct
        self.repetition_window         = repetition_window
        self.repetition_threshold      = repetition_threshold
        self.token_rate_floor_tps     = token_rate_floor_tps
        self.token_rate_min_elapsed_s = token_rate_min_elapsed_s
        self.token_rate_min_tokens    = token_rate_min_tokens
        #: Results from the most recent :meth:`evaluate` whose oracle declined
        #: to judge (see ``OracleResult.unmeasured``).  Reset at the top of
        #: every ``evaluate``; read by ``Runner`` so a check that could not run
        #: is recorded as a gap rather than disappearing into the pass path.
        #: Kept off the ``evaluate`` return value on purpose — that list feeds
        #: ``corpus.add`` scoring and ``adapter.filter_findings``, and a gap is
        #: not a finding.
        self.last_measurement_gaps: list[OracleResult] = []

    def evaluate(self,
                 result: TraceResult,
                 session) -> list[OracleResult]:
        findings = []
        self.last_measurement_gaps = []

        def _record(res: OracleResult) -> OracleResult:
            if res.passed and res.unmeasured:
                self.last_measurement_gaps.append(res)
            return res

        r = _record(self._check_memory_leak(session))
        if not r.passed:
            findings.append(r)

        r = self._check_post_cancel_tokens(result)
        if not r.passed:
            findings.append(r)

        r = self._check_hung_requests(result)
        if not r.passed:
            findings.append(r)

        r = self._check_scheduler_hang(session)
        if not r.passed:
            findings.append(r)

        r = self._check_latency_spike(result)
        if not r.passed:
            findings.append(r)

        r = _record(self._check_kv_block_leak(session))
        if not r.passed:
            findings.append(r)

        r = self._check_token_overflow(result)
        if not r.passed:
            findings.append(r)

        r = self._check_error_rate(result)
        if not r.passed:
            findings.append(r)

        for r in self._check_feature_failure(result):
            if not r.passed:
                findings.append(r)

        for r in self._check_token_rate(result):
            if not r.passed:
                findings.append(r)

        for r in self._check_degenerate_mask(result):
            if not r.passed:
                findings.append(r)

        return findings if findings else [OracleResult(passed=True, score=0.0)]

    def _check_reset_honored(self, result: TraceResult, trace) -> list[OracleResult]:
        """A prefix-cache reset that reported success must leave nothing cached.

        For every RESET the server answered with success while no request of
        the trace was still in flight, the first request sent after it must
        read zero prompt tokens from cache (``usage.prompt_tokens_details.
        cached_tokens``).  Being first, it cannot have hit anything another
        post-reset request wrote.  Requests with a ``cache_salt`` are skipped
        (their namespace is fresh anyway), as are requests whose usage did not
        report a cache count.
        """
        resets = getattr(result, "resets", None) or []
        if not resets:
            return []
        salted = {ev.request_id for ev in trace.events
                  if ev.action == ActionType.SEND and getattr(ev, "cache_salt", None)}
        out = []
        for rec in resets:
            if not rec.get("success") or rec.get("in_flight"):
                continue
            done = rec.get("done_at")
            after = [r for r in result.requests
                     if done is not None and r.sent_at >= done]
            if not after:
                continue
            first = min(after, key=lambda r: r.sent_at)
            if first.request_id in salted or first.error is not None:
                continue
            if first.cache_read_tokens and first.cache_read_tokens > 0:
                out.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.CACHE_RESET_IGNORED,
                    detail=(f"request {first.request_id} read "
                            f"{first.cache_read_tokens} cached prompt tokens right "
                            f"after POST /reset_prefix_cache reported success "
                            f"(status {rec.get('status')}, nothing in flight)"),
                    score=6.0,
                ))
        return out

    def annotate(self, result: "TraceResult") -> "list[OracleResult]":
        """Run output-quality checks that never trigger standalone findings.

        Returns annotation OracleResults for repetition and char_corruption.
        Only called when a primary oracle has already fired; results are embedded
        as output_annotations in the finding JSON.
        """
        out: list[OracleResult] = []
        out.extend(self._check_repetition(result))
        out.extend(self._check_char_corruption(result))
        return out

    # ------------------------------------------------------------------

    def _check_memory_leak(self, session) -> OracleResult:
        """GPU memory still held by a **drained** engine at end of trace.

        Same defect as ``_check_kv_block_leak`` and the same gate.  The detail
        string has always claimed "after scenario drained" while nothing
        established that a drain happened, and NVML's reading of a busy engine
        includes activation and workspace allocations that are released when
        generation finishes — so a client-cancelled trace's final sample is
        transient allocation being reported as a leak.  This oracle produced
        exactly 1 finding in each of the last campaign's six arms; that
        uniformity is explained by per-signature exemplar capping rather than
        by drain state, so unlike the KV case it is *not* established that
        every historical ``memory_leak`` is a false positive.  The gate is
        applied because the reasoning defect is identical, not because the
        findings have been refuted.

        ``memory_leaked_mb`` / ``baseline_memory_mb`` are deliberately left
        alone: ``MonitorSession.fingerprint`` derives ``memory_delta_bucket``
        from them, so changing their semantics would silently re-bucket every
        fingerprint and break comparability with archived corpora.  The gate
        lives here, where it only affects the finding.
        """
        leaked = session.memory_leaked_mb(self.memory_leak_tolerance_mb)
        if leaked is None:
            return OracleResult(passed=True, detail="no gpu data",
                                unmeasured="memory_leak: no gpu data")
        if leaked > 0:
            drained = session.engine_drained()
            if drained is None:
                return OracleResult(
                    passed=True,
                    detail=f"GPU memory grew by {leaked:.1f} MB but drain state "
                           f"is unknown — no queue telemetry in this session",
                    unmeasured="memory_leak: no queue telemetry",
                )
            if not drained:
                q = session.final_queue_state() or (0, 0)
                return OracleResult(
                    passed=True,
                    detail=f"GPU memory grew by {leaked:.1f} MB but the engine "
                           f"was not drained (running={q[0]} waiting={q[1]}) — "
                           f"retention not measurable",
                    unmeasured=f"memory_leak: engine not drained "
                               f"(running={q[0]} waiting={q[1]})",
                )
            return OracleResult(
                passed=False,
                bug_class=BugClass.MEMORY_LEAK,
                detail=f"GPU memory grew by {leaked:.1f} MB after scenario drained "
                       f"(engine idle: running=0 waiting=0)",
                score=min(leaked / 100.0, 10.0),
            )
        return OracleResult(passed=True)

    def _check_post_cancel_tokens(self, result: TraceResult) -> OracleResult:
        bad = result.post_cancel_tokens()
        if bad:
            ids = [r.request_id for r in bad]
            return OracleResult(
                passed=False,
                bug_class=BugClass.LIFECYCLE_ERROR,
                detail=f"tokens received after cancel on requests: {ids}",
                score=5.0 * len(bad),
            )
        return OracleResult(passed=True)

    def _check_hung_requests(self, result: TraceResult) -> OracleResult:
        hung = result.hung_requests(self.request_timeout_s)
        if hung:
            ids = [r.request_id for r in hung]
            return OracleResult(
                passed=False,
                bug_class=BugClass.LIFECYCLE_ERROR,
                detail=f"requests never completed: {ids}",
                score=3.0 * len(hung),
            )
        return OracleResult(passed=True)

    def _check_scheduler_hang(self, session) -> OracleResult:
        if session.requests_ever_stuck():
            return OracleResult(
                passed=False,
                bug_class=BugClass.SCHEDULER_HANG,
                detail="waiting queue non-empty after running queue drained",
                score=8.0,
            )
        return OracleResult(passed=True)

    def _check_latency_spike(self, result: TraceResult) -> OracleResult:
        # Non-streaming: first_token_at == completion time, not first token.
        # Exclude them to avoid flagging normal inference time as a latency bug.
        streaming = [r for r in result.requests
                     if r.ttft_ms() is not None and r.stream]
        if not streaming:
            return OracleResult(passed=True)

        # p99 is meaningless with fewer than MIN_LATENCY_SAMPLE requests;
        # small traces produce p100 with the floor-index formula, generating
        # false positives from any single slow request.
        if len(streaming) < MIN_LATENCY_SAMPLE:
            return OracleResult(passed=True,
                                detail="too few samples for latency judgment")

        if self.baseline_ttft_ms is not None:
            baseline_per_token = self.baseline_ttft_ms / max(self.baseline_prompt_len, 1)
            ratios = []
            for r in streaming:
                ttft = r.ttft_ms()
                # Queue depth when this request was dispatched: count requests
                # that were sent at or before this one and hadn't yet received
                # their first token.  This scales the threshold with actual
                # concurrency so normal burst-induced queuing is not flagged.
                in_flight = sum(
                    1 for other in streaming
                    if other.sent_at <= r.sent_at
                    and (other.first_token_at is None
                         or other.first_token_at > r.sent_at)
                )
                expected  = baseline_per_token * max(r.prompt_len, self.baseline_prompt_len)
                threshold = max(
                    expected * self.latency_multiplier * max(1, in_flight),
                    _LATENCY_SPIKE_MIN_MS,
                )
                ratios.append((ttft / threshold, ttft, threshold, in_flight))
            ratios.sort(key=lambda x: x[0])
            # p99 inclusive: index ceil(n*0.99)-1; guard against n=1 edge case
            idx = max(0, int(len(ratios) * 0.99) - 1)
            worst_ratio, worst_ttft, worst_threshold, worst_concurrency = ratios[idx]
            if worst_ratio > 2.0:
                return OracleResult(
                    passed=False,
                    bug_class=BugClass.LATENCY_SPIKE,
                    detail=f"p99 TTFT {worst_ttft:.0f} ms exceeds expected "
                           f"{worst_threshold:.0f} ms ({worst_ratio:.1f}x) "
                           f"at concurrency={worst_concurrency}",
                    score=worst_ratio,
                )
        else:
            # No baseline: fall back to fixed threshold.
            ttfts = sorted(r.ttft_ms() for r in streaming)
            idx   = max(0, int(len(ttfts) * 0.99) - 1)
            p99   = ttfts[idx]
            if p99 > self.latency_ttft_threshold_ms:
                return OracleResult(
                    passed=False,
                    bug_class=BugClass.LATENCY_SPIKE,
                    detail=f"p99 TTFT {p99:.0f} ms exceeds threshold "
                           f"{self.latency_ttft_threshold_ms:.0f} ms",
                    score=p99 / self.latency_ttft_threshold_ms,
                )
        return OracleResult(passed=True)

    def _check_kv_block_leak(self, session) -> OracleResult:
        """KV blocks still held by a **drained** engine at end of trace.

        The drain gate is the whole check.  Without it this fired on the last
        KV reading of whatever state the engine happened to be in when the
        client stopped watching — and since ``AsyncExecutor`` client-cancels at
        ``trace_timeout_s`` while the server keeps generating, that reading is
        routinely mid-trace occupancy.  Every one of the 39 ``kv_block_leak``
        findings this fuzzer has ever produced was that: joining their
        ``collection_forensics.end_t`` against vLLM's own stat log puts 37 of
        them at ``Running>0`` or ``Waiting>0`` (as bad as ``Running=3
        Waiting=7`` at 89.5% KV), and the remaining 2 sit against
        ``Running=0 Waiting=0, GPU KV cache usage 0.0%`` log lines that
        contradict the finding's own "final 30–58%".  Meanwhile vLLM's stat
        logger produced 164 idle samples across three arms, every one reading
        exactly 0.0% — with a 1152-block pool a *single* retained block would
        render as 0.1%.  Nothing was ever retained.

        A non-drained session must not pass silently either: it returns
        ``passed=True`` with ``unmeasured`` set, which the runner records as a
        measurement gap.  The failure being fixed here is an unmeasured
        quantity reported as a measurement; replacing it with an unmeasured
        quantity reported as a clean bill of health would be the same mistake
        wearing the opposite sign.
        """
        baseline = session.kv_cache_baseline_pct()
        final    = session.kv_cache_final_pct()
        if baseline is None or final is None:
            return OracleResult(passed=True, detail="no kv cache data",
                                unmeasured="kv_block_leak: no kv cache data")

        drained = session.kv_cache_final_drained()
        if drained is None:
            return OracleResult(
                passed=True,
                detail="drain state unknown — no queue telemetry in the "
                       "snapshot that supplied the final KV reading",
                unmeasured="kv_block_leak: no queue telemetry",
            )
        if not drained:
            q = session.final_queue_state() or (0, 0)
            return OracleResult(
                passed=True,
                detail=f"engine not drained (running={q[0]} waiting={q[1]}) — "
                       f"KV retention not measurable; final reading "
                       f"{final:.1f}% is in-flight occupancy, not a leak",
                unmeasured=f"kv_block_leak: engine not drained "
                           f"(running={q[0]} waiting={q[1]})",
            )

        delta = final - baseline
        if delta > self.kv_block_leak_pct:
            return OracleResult(
                passed=False,
                bug_class=BugClass.KV_BLOCK_LEAK,
                detail=f"KV cache usage grew {delta:.1f}% after scenario drained "
                       f"(baseline {baseline:.1f}% → final {final:.1f}%, "
                       f"engine idle: running=0 waiting=0)",
                score=delta / 10.0,
            )
        return OracleResult(passed=True)

    def _check_token_overflow(self, result: TraceResult) -> OracleResult:
        """Detect requests that produced more tokens than max_tokens_requested."""
        bad = [r for r in result.requests
               if r.max_tokens_requested is not None
               and r.tokens_received > r.max_tokens_requested]
        if bad:
            ids = [r.request_id for r in bad]
            return OracleResult(
                passed=False,
                bug_class=BugClass.LIFECYCLE_ERROR,
                detail=f"tokens_received exceeded max_tokens_requested on requests: {ids}",
                score=3.0 * len(bad),
            )
        return OracleResult(passed=True)

    def _check_error_rate(self, result: TraceResult) -> OracleResult:
        """Flag fatal server errors (5xx, connection failures) regardless of rate.

        HTTP 400 (context length) and 429 (rate limit) are expected under fuzzing
        load and are ignored. Timeouts are handled by the latency oracle.
        A single 5xx or connection failure is sufficient to fire.

        Classification lives in ``types.is_fatal_error``, which keys on the
        HTTP status the message actually reports.  The predicate used to be a
        bare substring test against ``_FATAL_PATTERNS = ("500", "502", ...)``,
        so "400 Bad Request: max_tokens 8500 exceeds context length" and
        "429 Too Many Requests (retry after 5040ms)" both filed a
        LIVENESS_FAILURE off an incidental digit run.
        """
        fatal = [
            r for r in result.requests
            if is_fatal_error(r.error)
        ]
        if fatal:
            ids = [r.request_id for r in fatal]
            return OracleResult(
                passed=False,
                bug_class=BugClass.LIVENESS_FAILURE,
                detail=(f"fatal server error on {len(fatal)} request(s): {ids} "
                        f"— {fatal[0].error[:120]}"),
                score=min(5.0 * len(fatal), 10.0),
            )
        return OracleResult(passed=True)

    # ------------------------------------------------------------------
    # New output quality oracles
    # ------------------------------------------------------------------

    def _check_repetition(self, result: TraceResult) -> list[OracleResult]:
        findings = []
        for r in result.requests:
            if r.error is not None:
                continue
            if r.cancelled_at is not None:
                continue
            if r.output_text is None:
                continue
            tokens = r.output_text.split()
            if len(tokens) < self.repetition_window:
                continue
            window_size = self.repetition_window
            fired = False
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i:i + window_size]
                unique_count = len(set(window))
                if (window_size - unique_count) / window_size > self.repetition_threshold:
                    fired = True
                    break
            if fired:
                findings.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.REPETITION,
                    detail=f"repetitive output detected in request {r.request_id}",
                    score=6.0,
                ))
        return findings if findings else [OracleResult(passed=True)]

    def _check_char_corruption(self, result: TraceResult) -> list[OracleResult]:
        bad = []
        for r in result.requests:
            if r.error is not None:
                continue
            if r.output_text is None:
                continue
            m = _CORRUPT_RE.search(r.output_text.replace("\x1f", ""))
            if m:
                bad.append((r, m.group()))
        if bad:
            findings = []
            for r, matched_char in bad:
                findings.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.CHAR_CORRUPTION,
                    detail=f"corrupt character in request {r.request_id}: {repr(matched_char)}",
                    score=7.0,
                ))
            return findings
        return [OracleResult(passed=True)]

    def _check_feature_failure(self, result: TraceResult) -> list[OracleResult]:
        findings = []
        for r in result.requests:
            # A request the server answered with an error because it could not
            # serialise its OWN numbers (NaN/inf logprobs -> "Out of range float
            # values are not JSON compliant") is the NaN check below, surfaced
            # as a 4xx instead of a body.  The status is the server's mistake,
            # not a verdict on the request.
            if r.error is not None and _NONFINITE_ERROR.search(r.error):
                findings.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.FEATURE_FAILURE,
                    detail=(f"request {r.request_id}: server produced non-finite "
                            f"values it could not serialise — {r.error[:160]}"),
                    score=5.0,
                ))
                continue
            if r.error is not None:
                continue
            if r.cancelled_at is not None:
                continue
            violated = False
            detail_parts = []
            # Logprobs presence is unreliable for streaming: vLLM may not include
            # logprobs in every SSE chunk, and some models/configs don't support
            # streaming logprobs at all. Only check non-streaming responses.
            if not r.stream and r.logprobs_requested is not None and r.logprobs_requested > 0:
                if r.response_has_logprobs is not True:
                    violated = True
                    detail_parts.append(
                        f"logprobs requested={r.logprobs_requested} but not in response"
                    )
            # Invalid logprob values (NaN/inf) are always a numeric bug.
            # Only populated for non-streaming responses where we parse the body.
            if r.logprob_has_invalid:
                violated = True
                detail_parts.append("invalid logprob value (NaN or inf) in response")

            # n_completions check is unreliable for streaming requests: vLLM may
            # interleave or deduplicate sequences in ways that make choice-counting
            # inaccurate without server-side usage data. Only check non-streaming.
            if not r.stream and r.n_completions_requested is not None and r.n_completions_requested > 1:
                if r.response_n_choices != r.n_completions_requested:
                    violated = True
                    detail_parts.append(
                        f"n_completions requested={r.n_completions_requested} "
                        f"but got {r.response_n_choices} choices"
                    )
            # finish_reason == "error" is a server-side failure mid-generation.
            # Only fire when the request otherwise completed (no client error/cancel).
            if r.finish_reason == "error":
                violated = True
                detail_parts.append("finish_reason=error reported by server")

            if violated:
                findings.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.FEATURE_FAILURE,
                    detail=f"feature failure in request {r.request_id}: {'; '.join(detail_parts)}",
                    score=5.0,
                ))
        return findings if findings else [OracleResult(passed=True)]

    def _check_token_rate(self, result: TraceResult) -> list[OracleResult]:
        """Detect mid-stream stalls: streaming requests whose generation rate
        (tokens/sec after first token) falls below a floor.  Catches a class
        of bugs distinct from TTFT — a request can start promptly but stall
        mid-generation due to repeated preemption or scheduler starvation."""
        findings = []
        for r in result.requests:
            if not r.stream:
                continue
            if r.error is not None or r.cancelled_at is not None:
                continue
            if r.first_token_at is None or r.completed_at is None:
                continue
            elapsed = r.completed_at - r.first_token_at
            if elapsed < self.token_rate_min_elapsed_s:
                continue
            if r.tokens_received < self.token_rate_min_tokens:
                continue
            rate = r.tokens_received / elapsed
            if rate < self.token_rate_floor_tps:
                # Score scales with severity: floor/rate capped at 10.
                score = min(self.token_rate_floor_tps / max(rate, 1e-6), 10.0)
                findings.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.TOKEN_RATE_STALL,
                    detail=(
                        f"token rate {rate:.3f} tok/s on request {r.request_id} "
                        f"(floor={self.token_rate_floor_tps} tok/s, "
                        f"elapsed={elapsed:.1f}s, tokens={r.tokens_received})"
                    ),
                    score=score,
                ))
        return findings if findings else [OracleResult(passed=True)]

    # ------------------------------------------------------------------
    # Degenerate composed logits mask (mask-composition surface)
    # ------------------------------------------------------------------

    #: how much of a degenerate output is kept on the finding
    _DEGENERATE_OUTPUT_EXCERPT = 400

    def _check_degenerate_mask(self, result: TraceResult) -> list:
        """Specified oracle: **no composed token mask may be empty.**

        vLLM writes ``-inf`` into the logits row from at least six independent
        places — the structured-output bitmask, ``min_tokens``,
        ``allowed_token_ids``, ``bad_words``, ``logit_bias``, and the stop set
        that ``stop_token_ids`` feeds — and composes them in sequence with *no
        emptiness check anywhere on the path*.  When two of them mask disjoint
        sets the whole row becomes ``-inf`` and greedy ``argmax`` returns
        vocabulary index 0, a token no mask allowed.  The archived
        ``min_tokens`` + grammar finding is one cell of that table; this oracle
        is deliberately written to decide *any* cell, and it never needs to know
        which pair caused it.

        Three exact predicates, each with its own sub-type:

        ``dm_sentinel`` — **the generic one.**  vLLM serialises an ``-inf``
            logprob as ``-9999.0``, and it computes the logprobs it *reports*
            from the logits **before** the logits processors run (``v1/sample/
            sampler.py``: ``raw_logprobs`` first, ``apply_logits_processors``
            after).  So a degenerate row is directly observable over HTTP: the
            **sampled** token carries the sentinel while some **other**
            candidate at the same position is still at a finite value.  A
            sampled token whose own logprob is ``-inf`` is not a sampling
            decision — it is ``argmax`` over a row that was already fully
            masked by the time the sampler saw it.  No schema, no JSON parsing,
            no statistics, no baseline.

            Because the snapshot precedes the processors, this predicate sees a
            composition only when at least one member is applied *before* it —
            in practice the grammar bitmask (``PRE_LOGPROB_RESTRICTORS``).  A
            purely processor-vs-processor empty mask leaves the reported
            logprobs untouched and is caught by ``dm_allowlist_escape`` instead.

        ``dm_grammar_guard`` — the same defect's symptom on releases that added
            the downstream guard (0.27.1 and later): the illegal token is fed
            back into the grammar FSM, rejected, and the request is terminated
            with **HTTP 500** (``backend_xgrammar.py``: "Failed to advance FSM
            … for tokens 0. Please file an issue." → ``scheduler.py``:
            "Unexpected: grammar rejected tokens [0] … Terminating request.").
            The empty mask is still produced; only the handling changed.

        ``dm_allowlist_escape`` — the tokenizer-free one.  When a request
            declares ``allowed_token_ids`` and *every* id in it is also in the
            stop set (``stop_token_ids``) that ``min_tokens`` bans below the
            floor, no legal token exists at all for the first ``min_tokens``
            positions.  Each id maps to exactly one token string, so emitting
            **more distinct token strings than the allowlist has ids** proves
            tokens outside the allowlist were emitted — decided exactly, with
            no vocabulary knowledge and no threshold.

        The false-positive guards, enumerated:

        1. **A legitimately banned token is not a bug.**  Every predicate
           anchors on the *sampled* token, never on a top-k entry.  A banned
           candidate sitting at the sentinel in ``top_logprobs`` while a legal
           token was actually sampled is exactly what a *working* mask looks
           like, and is silent here.
        2. **The sentinel could in principle be a real logprob.**  For *output*
           logprobs -9999.0 is a ``max()`` **clamp**, not an exact ``-inf`` map
           (``chat_completion/serving.py:1215`` and ``:1158``;
           ``completion/serving.py:703`` and ``:721``), so two non-``-inf``
           paths can produce it and both are excluded here:

           * a genuine finite logprob at or below -9999 clamps onto it.  That
             needs a 1e4-nat spread inside one row, which no model produces,
             and it is unreachable through ``logit_bias`` — clamped to +/-100
             on every HTTP path (``sampling_params.py:399``, ``:411``) and
             *additive* rather than a mask (``builtin.py:161``), so it cannot
             write ``-inf`` at all.  The test is nevertheless exact equality,
             not ``<=``, so a real -9998.7 does not match; and a finite rival
             above ``_FINITE_ALT_FLOOR`` must exist at the same position, so a
             row whose alternatives are *also* at the sentinel (a missing
             top-k, or a different defect) is not filed as this one.
           * ``ChatCompletionLogProb.logprob`` **defaults** to -9999.0
             (``chat_completion/protocol.py:83``) and the chat serialiser omits
             the kwarg exactly when the sampled token is absent from that
             step's top-k (``serving.py:1194-1207``).  Requiring the sampled
             token to be *present* in the position's top-k, carrying the value
             itself, removes that path entirely.
        3. **No logprobs, no verdict.**  A request that did not ask for
           logprobs, or whose top-k never arrived, is skipped rather than
           guessed at.  This is why ``compose_mask_restrictors`` always sets
           ``logprobs``.
        4. **Truncation / cancellation / transport failure.**  Cancelled
           requests are skipped everywhere.  ``dm_sentinel`` and
           ``dm_allowlist_escape`` skip requests with ``r.error`` set: a 4xx is
           the server *rejecting* the combination up front, which is the honest
           outcome and a capability gap, not a defect.
        5. **A 500 is not automatically this bug.**  ``dm_grammar_guard``
           requires the status to be exactly 500 *and* the request to have
           composed two or more restrictors *including* a structured-output
           constraint — the guard lives inside the grammar backend, so a 500 on
           a request with no grammar cannot be it.  (Such a request still files
           a ``liveness_failure`` from ``_check_error_rate``; that is
           deliberate, and the two carry different sub-types so the ledger keeps
           them apart.)
        6. **Alignment.**  ``output_logprobs`` / ``output_top_logprobs`` /
           ``output_tokens`` are choice-index-0 only and index-aligned; nothing
           past the shortest list is examined, so an ``n>1`` request can never
           mix sequences.
        """
        findings: list[OracleResult] = []
        for r in result.requests:
            if r.cancelled_at is not None:               # guard 4
                continue
            restrictors = r.mask_restrictors or {}
            if r.error is not None:                      # guards 4 + 5
                f = self._degenerate_mask_from_error(r, restrictors)
                if f is not None:
                    findings.append(f)
                continue
            f = self._degenerate_mask_from_logprobs(r, restrictors)
            if f is not None:
                findings.append(f)
            f = self._degenerate_mask_from_allowlist(r, restrictors)
            if f is not None:
                findings.append(f)
        return findings if findings else [OracleResult(passed=True)]

    def _degenerate_mask_from_logprobs(self, r, restrictors: dict):
        """``dm_sentinel``: the sampled token is at ``-inf``, a rival is not."""
        import math as _math

        lps  = r.output_logprobs
        tops = r.output_top_logprobs
        toks = r.output_tokens
        if not lps or not tops or not toks:              # guard 3
            return None
        for i in range(min(len(lps), len(tops), len(toks))):   # guard 6
            sampled = lps[i]
            # guard 1 + 2: the SAMPLED token, at exactly the sentinel
            if not isinstance(sampled, (int, float)) or isinstance(sampled, bool):
                continue
            if float(sampled) != NEG_INF_LOGPROB_SENTINEL:
                continue
            top = tops[i]
            if not isinstance(top, dict):
                continue
            # guard 2b: the sampled token must itself appear in this position's
            # top-k *carrying the value*.  ``ChatCompletionLogProb.logprob``
            # defaults to -9999.0 and the chat serialiser omits the kwarg
            # exactly when the sampled token is missing from the top-k
            # (chat_completion/protocol.py:83, serving.py:1194-1207), so
            # "sampled token absent from top-k" is the one way to read -9999.0
            # with no -inf behind it.  Requiring its presence removes that path.
            token = toks[i]
            if token not in top:
                continue
            entry = top[token]
            if not isinstance(entry, (int, float)) or isinstance(entry, bool):
                continue
            if float(entry) != NEG_INF_LOGPROB_SENTINEL:
                continue
            # guard 2: some *other* candidate at a plausible finite value.
            # Entries at the sentinel are excluded, which also excludes the
            # sampled token's own entry.
            alts = [v for v in top.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                    and _math.isfinite(v)
                    and float(v) != NEG_INF_LOGPROB_SENTINEL
                    and float(v) > _FINITE_ALT_FLOOR]
            if not alts:
                continue
            best = max(alts)
            return OracleResult(
                passed=False,
                bug_class=BugClass.DEGENERATE_MASK,
                detail=(
                    f"empty composed logits mask on request {r.request_id}: "
                    f"the sampled token at output position {i} carries the "
                    f"-inf sentinel ({NEG_INF_LOGPROB_SENTINEL}) while a rival "
                    f"candidate is finite at {best:.4f} "
                    f"(restrictors={sorted(restrictors) or 'none declared'})"
                ),
                score=9.0,
                probe_metadata={
                    "predicate": "sentinel",
                    "position": i,
                    "sampled_token": token,
                    "sampled_logprob": float(sampled),
                    "best_alternative_logprob": float(best),
                    "restrictors": sorted(restrictors),
                    "pre_logprob_restrictors": sorted(
                        set(restrictors) & PRE_LOGPROB_RESTRICTORS),
                    "request_id": r.request_id,
                    "output": (r.output_text or "")[:self._DEGENERATE_OUTPUT_EXCERPT],
                },
            )
        return None

    def _degenerate_mask_from_error(self, r, restrictors: dict):
        """``dm_grammar_guard``: the post-0.27 downstream guard's HTTP 500."""
        if http_status_from_error(r.error) != 500:       # guard 5
            return None
        if len(restrictors) < 2:                         # guard 5
            return None
        if not (set(restrictors) & PRE_LOGPROB_RESTRICTORS):
            return None
        corroborated = grammar_guard_signature(r.error or "")
        return OracleResult(
            passed=False,
            bug_class=BugClass.DEGENERATE_MASK,
            detail=(
                f"grammar guard terminated request {r.request_id} with HTTP 500 "
                f"after composing {len(restrictors)} mask restrictors "
                f"({sorted(restrictors)}) — the post-0.27 symptom of an empty "
                f"composed mask"
            ),
            score=9.0,
            probe_metadata={
                "predicate": "grammar_guard",
                "http_status": 500,
                "log_corroborated": corroborated,
                "restrictors": sorted(restrictors),
                "request_id": r.request_id,
                "error": (r.error or "")[:self._DEGENERATE_OUTPUT_EXCERPT],
            },
        )

    def _degenerate_mask_from_allowlist(self, r, restrictors: dict):
        """``dm_allowlist_escape``: more distinct tokens than the allowlist has.

        Exact by construction and free of any vocabulary knowledge: when every
        allowed id is also in the banned stop set, *nothing* is legal below the
        floor, and one id can only ever render as one token string.
        """
        allow = restrictors.get("allowed_token_ids")
        stops = restrictors.get("stop_token_ids")
        floor = restrictors.get("min_tokens")
        if not allow or not stops or not floor:
            return None
        try:
            allow_set = {int(i) for i in allow}
            stop_set  = {int(i) for i in stops}
            floor     = int(floor)
        except (TypeError, ValueError):
            return None
        if floor < 1 or not allow_set or not allow_set <= stop_set:
            return None
        if r.finish_reason not in ("stop", "length"):    # guard 4
            return None
        tokens = r.output_tokens
        if not tokens:                                   # guard 3
            return None
        window = [t for t in tokens[:floor] if t is not None]
        distinct = set(window)
        if len(distinct) <= len(allow_set):
            return None
        return OracleResult(
            passed=False,
            bug_class=BugClass.DEGENERATE_MASK,
            detail=(
                f"allowed_token_ids escaped on request {r.request_id}: the "
                f"allowlist declares {len(allow_set)} id(s), every one of them "
                f"is in the stop set that min_tokens={floor} bans, yet "
                f"{len(distinct)} distinct token strings were emitted in the "
                f"first {len(window)} output positions"
            ),
            score=9.0,
            probe_metadata={
                "predicate": "allowlist_escape",
                "allowed_token_ids": sorted(allow_set),
                "stop_token_ids": sorted(stop_set),
                "min_tokens": floor,
                "distinct_tokens_in_window": len(distinct),
                "window_len": len(window),
                "restrictors": sorted(restrictors),
                "request_id": r.request_id,
                "output": (r.output_text or "")[:self._DEGENERATE_OUTPUT_EXCERPT],
            },
        )

    # ------------------------------------------------------------------
    # Structured-output exact oracle (needs the Trace; called from the
    # adapter's evaluate_extra, which is the only hook that receives it)
    # ------------------------------------------------------------------

    #: how much of a violating output is kept on the finding
    _SCHEMA_OUTPUT_EXCERPT = 400

    def _check_schema_violation(self, result: TraceResult,
                                trace) -> list[OracleResult]:
        """Specified oracle: a finished output must satisfy its declared constraint.

        When a SEND carries ``response_format`` / ``guided_json`` /
        ``guided_regex`` / ``guided_choice``, the engine's structured-output FSM
        is *contractually* obliged to make every emitted token keep the output
        inside the grammar.  So a request that ran to ``finish_reason == "stop"``
        and still fails validation is a defect, decided exactly — no baseline,
        no threshold, no statistics.  (``guided_grammar`` is accepted on the wire
        but not checked here: exact EBNF parsing is not cheap and the
        stateless grammar-compilation path is DIRGE's, not GRIEF's.)

        The six false-positive sources this deliberately excludes:

        1. **Truncation.** ``finish_reason != "stop"`` — a ``"length"`` stop
           cuts the object mid-grammar, so the FSM never got to close it.
        2. **Server rejection.** ``r.error`` set (4xx for a schema keyword
           xgrammar does not implement, 5xx, connection reset) means no
           constrained generation happened.  That is a capability gap; it is
           counted in ``_capability_gaps`` and never filed as a bug.  An invalid
           *schema* (jsonschema ``SchemaError``) is treated the same way.
        3. **Trailing content.** Templates emit a trailing newline/space after
           the object; ``raw_decode`` accepts a valid JSON prefix whose
           remainder is whitespace.
        4. **Prompt/model capability.** Only constraints the engine enforces
           structurally are checked — bare ``{"type": "json_object"}`` is
           checked for parseability only, never against a schema — and an empty
           output is skipped rather than blamed on the FSM.  The rest of this
           guard is a seed-design constraint: the smoke seeds
           (``seeds/guided_smoke/``) use a schema the base model satisfies
           unconstrained, so a failure is FSM state, not capability.
        5. **n>1 join artifact.** ``output_text`` is ``\x1f``-joined for n>1;
           each segment is validated separately, never the joined string.
        6. **Streaming reassembly.** ``output_text`` is the fully concatenated
           stream, never a partial SSE chunk.

        Cancelled requests are skipped for the same reason as (1).
        """
        if trace is None:
            return [OracleResult(passed=True)]

        events = {}
        for ev in getattr(trace, "events", []) or []:
            action = getattr(ev.action, "value", ev.action)
            if action == "send":
                events[ev.request_id] = ev
        if not events:
            return [OracleResult(passed=True)]

        findings: list[OracleResult] = []
        for r in result.requests:
            ev = events.get(r.request_id)
            if ev is None:
                continue
            kind, spec = _constraint_of(ev)
            if kind is None:
                continue
            # (2) server rejected / never generated under the constraint
            if r.error is not None:
                _note_capability_gap(kind, r.error)
                continue
            if r.cancelled_at is not None:
                continue
            # (1a) A choice constraint is checkable even when truncated: every
            # prefix the FSM admits is a prefix of one of the choices, so a
            # length-stopped output that is not one was never constrained.
            if (kind == "choice" and r.finish_reason == "length"
                    and r.output_text is not None):
                choices = [str(c) for c in spec]
                for seq_idx, segment in enumerate(r.output_text.split("\x1f")):
                    if segment and not any(c.startswith(segment) for c in choices):
                        findings.append(OracleResult(
                            passed=False,
                            bug_class=BugClass.SCHEMA_VIOLATION,
                            detail=(
                                f"structured output violated its choice constraint "
                                f"on request {r.request_id} (finish_reason=length, "
                                f"failure=choice_prefix): output is not a prefix "
                                f"of any of {choices[:4]}"
                            ),
                            score=8.0,
                            probe_metadata={
                                "constraint": kind, "failure": "choice_prefix",
                                "reason": "not a prefix of any choice",
                                "sequence_index": seq_idx,
                                "output": segment[:self._SCHEMA_OUTPUT_EXCERPT],
                                "request_id": r.request_id,
                            },
                        ))
                continue
            # (1) truncated mid-grammar.  finish_reason alone is NOT proof the
            # FSM reached an accepting state — see _terminated_voluntarily.
            if r.finish_reason != "stop":
                continue
            if not self._terminated_voluntarily(r, ev):
                continue
            if r.output_text is None:
                continue

            # (5) validate each completion separately
            for seq_idx, segment in enumerate(r.output_text.split("\x1f")):
                # (4) nothing generated — not an FSM failure
                if not segment.strip():
                    continue
                mode, reason = _validate_constraint(kind, spec, segment)
                if mode is None:
                    continue                       # satisfied
                if mode == "capability":
                    _note_capability_gap(kind, reason)
                    continue
                findings.append(OracleResult(
                    passed=False,
                    bug_class=BugClass.SCHEMA_VIOLATION,
                    detail=(
                        f"structured output violated its {kind} constraint on "
                        f"request {r.request_id} (finish_reason=stop, "
                        f"failure={mode}): {reason}"
                    ),
                    score=8.0,
                    probe_metadata={
                        "constraint": kind,
                        "failure": mode,
                        "reason": reason,
                        "sequence_index": seq_idx,
                        "output": segment[:self._SCHEMA_OUTPUT_EXCERPT],
                        "request_id": r.request_id,
                    },
                ))

        return findings if findings else [OracleResult(passed=True)]

    @staticmethod
    def _terminated_voluntarily(r, ev) -> bool:
        """True iff the request ended without stop-string/stop-token truncation.

        ``finish_reason == "stop"`` does **not** mean the structured-output FSM
        reached an accepting state.  Three paths on vLLM 0.27.1 produce it while
        the grammar is still mid-production, all read out of the 0.27.1 image:

        1. **A stop string matched.**  ``v1/engine/detokenizer.py:131`` scans for
           stop strings; on a hit ``:138-141`` truncates ``output_text`` at
           ``truncate_to``, and ``v1/engine/output_processor.py:659-661``
           *overwrites* ``finish_reason`` to ``STOP`` and sets
           ``stop_reason = stop_string``.  The overwrite is unconditional, so a
           request the scheduler had already length-capped is relabelled "stop".
        2. **A stop token id fired.**  ``v1/core/sched/utils.py:108-111`` sets
           ``stop_reason = last_token_id`` and is checked *before* the length cap
           at ``:112-117``.
        3. Neither is the FSM's doing: once the grammar terminates, the bitmask
           is replaced with all-ones (``v1/structured_output/__init__.py:198-205``,
           ``_full_mask = torch.tensor(-1)`` at ``:58``), so generation past the
           accepting state is *unconstrained* and (1)/(2) are the only things
           left to end the request.

        The concrete false positive this guard removes (3 of them in the
        7352699_tight15_s7 campaign, all request ``p7``): ``min_tokens=32`` +
        ``bad_words=['{','"',':','}']`` + a ``json_schema`` response_format +
        ``stop=['In ','Second','\\n\\n']``.  ``bad_words`` banned the quote, so
        whitespace was the only grammar-legal continuation; ``min_tokens``
        suppressed stop-string checking below the floor and — because
        ``detokenizer.py:117-123`` advances ``stop_check_offset`` past every
        pre-floor token and ``check_stop_strings`` only ever looks back
        ``new_char_count + len(stop) - 1`` chars (``:340``) — that pre-floor text
        is never rescanned.  Literal ``\\n\\n`` therefore accumulated *inside*
        the returned string; the first post-floor ``\\n\\n`` then truncated it
        and stamped ``finish_reason="stop"`` on ``{\\n\\n  \\n\\n\\n...``.

        ``stop_reason is None`` is necessary but not sufficient on its own,
        because ``sampling_params.py:661-674`` silently folds the model's
        *secondary* EOS ids from ``generation_config.json`` into
        ``stop_token_ids``.  Qwen2.5-0.5B-Instruct declares
        ``eos_token_id: [151645, 151643]`` with primary ``151645``, so a
        voluntary ``<|endoftext|>`` (151643) takes the ``utils.py:108-111``
        branch and reports an ``int``.  Requiring a bare ``None`` would turn a
        false positive into a false negative there.  So an ``int`` is accepted
        when it is *not* one of the ids the request itself asked for: the only
        other way to get one is the generation_config augmentation, i.e. a
        model-native EOS.

        A ``str`` is always rejected: a stop string truncated the text.  (The
        one other str, ``"repetition_detected"`` at ``sched/utils.py:127``, ships
        with ``finish_reason == "repetition"`` and never reaches here.)
        """
        reported = getattr(r, "stop_reason_reported", None)
        sr = getattr(r, "stop_reason", None)

        if sr is None:
            if reported is False:
                # Backend never reports the field (pre-0.19 vLLM, SGLang, a
                # mock).  We cannot prove the stop was voluntary, so fall back
                # to the only other signal the executor records: whether any
                # stop condition was in play at all.  This is the first reader
                # of ``had_stop_sequences``.
                if getattr(r, "had_stop_sequences", False):
                    return False
                if getattr(ev, "stop_token_ids", None):
                    return False
            return True

        if isinstance(sr, bool):                   # bool is an int subclass
            return False
        if isinstance(sr, str):
            return False                           # (1) a stop string truncated
        if isinstance(sr, int):
            try:
                requested = {int(i) for i in (getattr(ev, "stop_token_ids", None) or [])}
            except (TypeError, ValueError):
                return False
            return sr not in requested             # not ours ⇒ generation_config EOS
        return False

    # ------------------------------------------------------------------
    # LLM-specific black-box probes (called by Runner, not evaluate())
    # ------------------------------------------------------------------

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Token-level Jaccard similarity between two output strings.

        Splits on whitespace; returns 1.0 for identical strings, 0.0 for
        completely disjoint token sets.  Fast enough for inline use and
        robust to minor token-boundary differences.
        """
        if a == b:
            return 1.0
        ta, tb = set(a.split()), set(b.split())
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @staticmethod
    def check_determinism(actual: str, expected: str) -> OracleResult:
        """Determinism oracle: temperature=0 identical prompts must produce
        identical output.

        Returns passed=True only when outputs are identical.  Any divergence
        returns passed=False with similarity in probe_metadata (for logging).
        FP filtering is handled by the caller via logprob tie-break check.
        """
        if actual == expected:
            return OracleResult(passed=True)
        sim = Oracle._text_similarity(actual, expected)
        return OracleResult(
            passed=False,
            bug_class=BugClass.KV_CORRUPTION,
            detail="Non-deterministic output for identical prompt/temperature=0 "
                   f"(possible KV cache corruption, similarity={sim:.2f})",
            score=10.0,
            probe_metadata={"similarity": sim},
        )

    @staticmethod
    def check_stability(
        target_outputs: "list[str]",
        baseline_outputs: "list[str]",
        min_target_rate: float = 0.6,
        min_rate_delta: float = 0.4,
    ) -> OracleResult:
        """Stability oracle: validated output instability above the baseline noise floor.

        Runs the concurrent trace k times (target) and the same request in isolation
        k times (baseline), then compares modal-divergence rates.

        Rule A: target diverge_rate >= min_target_rate (high instability under load)
        Rule B: (target_rate - baseline_rate) >= min_rate_delta (excess over baseline)

        Both must hold to fire. Score is proportional to the excess rate delta.
        Unlike check_determinism(), a single pairwise mismatch is not enough.
        """
        if not target_outputs or not baseline_outputs:
            return OracleResult(passed=True)

        def _diverge_rate(outputs: list[str]) -> float:
            if len(outputs) <= 1:
                return 0.0
            modal = Counter(outputs).most_common(1)[0][0]
            return sum(1 for o in outputs if o != modal) / len(outputs)

        target_rate   = _diverge_rate(target_outputs)
        baseline_rate = _diverge_rate(baseline_outputs)
        rate_delta    = target_rate - baseline_rate

        if target_rate >= min_target_rate and rate_delta >= min_rate_delta:
            return OracleResult(
                passed=False,
                bug_class=BugClass.KV_CORRUPTION,
                detail=(
                    f"Validated output instability: target_rate={target_rate:.2f} "
                    f"baseline_rate={baseline_rate:.2f} delta={rate_delta:.2f}"
                ),
                score=min(10.0, rate_delta * 10.0),
            )
        return OracleResult(passed=True)

    @staticmethod
    def check_cache_hit_proxy(warm_ttft_ms: float,
                               probe_ttft_ms: float,
                               multiplier: float = 4.0) -> OracleResult:
        """
        Cache-hit proxy oracle: after warming the prefix cache, a re-probe of
        the same prompt should be fast. A TTFT regression suggests incorrect
        KV block eviction (blocks freed when they shouldn't be).
        """
        threshold = max(warm_ttft_ms * multiplier, 500.0)
        if probe_ttft_ms > threshold:
            return OracleResult(
                passed=False,
                bug_class=BugClass.CACHE_EVICTION,
                detail=f"Cache probe TTFT {probe_ttft_ms:.0f} ms vs warm baseline "
                       f"{warm_ttft_ms:.0f} ms — possible incorrect block eviction",
                score=probe_ttft_ms / max(warm_ttft_ms, 1.0),
            )
        return OracleResult(passed=True)


# ---------------------------------------------------------------------------
# Structured-output constraint helpers (module level: pure, no Oracle state)
# ---------------------------------------------------------------------------

try:                                   # optional dependency
    import jsonschema as _jsonschema
except Exception:                      # pragma: no cover - env without jsonschema
    _jsonschema = None

#: True when full JSON Schema validation is available.  Without it the json
#: constraint degrades to a parseability check (``json.loads``), which still
#: catches the FSM failures that matter most (unclosed object, stray token)
#: but cannot see a structurally-valid document that violates the schema.
SCHEMA_VALIDATION_AVAILABLE = _jsonschema is not None

_SCHEMA_DEGRADED_LOGGED = False

#: counts of constraints the server could not honour, by (kind, reason-head).
#: A capability gap is not a bug — it is recorded so a smoke run can tell
#: "xgrammar rejects $ref" apart from "the oracle never fired".
_CAPABILITY_GAPS: "Counter" = Counter()


def _note_capability_gap(kind: str, reason) -> None:
    _CAPABILITY_GAPS[(kind, str(reason)[:80])] += 1


def capability_gaps() -> dict:
    """Snapshot of the capability gaps seen so far (for run reporting)."""
    return {f"{k}:{r}": n for (k, r), n in _CAPABILITY_GAPS.items()}


def _constraint_of(event) -> tuple:
    """Return ``(kind, spec)`` for a SEND event, or ``(None, None)``.

    ``kind`` is one of ``"json_schema"`` / ``"json_object"`` / ``"regex"`` /
    ``"choice"``.  ``guided_grammar`` yields ``(None, None)``: it goes on the
    wire but is not exactly checkable here.  ``guided_*`` wins over
    ``response_format`` when both are present, matching vLLM's own precedence
    of the explicit extension over the OpenAI-compatible field.
    """
    gj = getattr(event, "guided_json", None)
    if isinstance(gj, dict) and gj:
        return "json_schema", gj
    gr = getattr(event, "guided_regex", None)
    if isinstance(gr, str) and gr:
        return "regex", gr
    gc = getattr(event, "guided_choice", None)
    if isinstance(gc, (list, tuple)) and gc:
        return "choice", list(gc)

    rf = getattr(event, "response_format", None)
    if isinstance(rf, dict) and rf:
        rf_type = rf.get("type")
        if rf_type == "json_schema":
            js = rf.get("json_schema") or {}
            schema = js.get("schema") if isinstance(js, dict) else None
            if isinstance(schema, dict) and schema:
                return "json_schema", schema
            # declared json_schema with no usable schema body → parseability only
            return "json_object", None
        if rf_type == "json_object":
            return "json_object", None
    return None, None


def _json_prefix(text: str):
    """``(obj, None)`` when *text* is one JSON value plus only whitespace.

    Guard (3): a trailing newline or space after a complete object is a
    template artifact, not a grammar violation.  Anything else after the value
    (a second object, prose, a stray brace) is a real violation.
    """
    import json as _json
    stripped = text.strip()
    try:
        obj, end = _json.JSONDecoder().raw_decode(stripped)
    except ValueError as exc:
        return None, str(exc)
    if stripped[end:].strip():
        return None, ("trailing content after the JSON value: "
                      + stripped[end:].strip()[:80])
    return obj, None


def _validate_constraint(kind: str, spec, text: str) -> tuple:
    """``(None, "")`` when satisfied, else ``(failure_mode, reason)``.

    ``failure_mode`` is ``"json_parse"`` / ``"json_schema"`` / ``"regex"`` /
    ``"choice"``, or the sentinel ``"capability"`` for "the constraint itself
    could not be evaluated" (invalid schema, uncompilable regex) — which is a
    capability gap, not a finding.
    """
    global _SCHEMA_DEGRADED_LOGGED

    if kind in ("json_schema", "json_object"):
        obj, err = _json_prefix(text)
        if err is not None:
            return "json_parse", err
        if kind == "json_object" or not isinstance(spec, dict):
            return None, ""
        if _jsonschema is None:
            if not _SCHEMA_DEGRADED_LOGGED:
                _SCHEMA_DEGRADED_LOGGED = True
                print("[oracle] jsonschema not importable — the schema_violation "
                      "oracle is degraded to json.loads parseability only; "
                      "schema-valid-but-wrong outputs will not be detected")
            return None, ""
        try:
            _jsonschema.validate(obj, spec)
        except _jsonschema.exceptions.SchemaError as exc:
            return "capability", f"invalid schema: {exc.message}"
        except _jsonschema.exceptions.ValidationError as exc:
            path = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            return "json_schema", f"at {path}: {exc.message}"
        except Exception as exc:                    # defensive
            return "capability", f"validator error: {exc}"
        return None, ""

    if kind == "regex":
        import re as _re2
        try:
            pattern = _re2.compile(spec, _re2.DOTALL)
        except _re2.error as exc:
            return "capability", f"uncompilable regex: {exc}"
        if pattern.fullmatch(text) or pattern.fullmatch(text.strip()):
            return None, ""
        return "regex", f"output does not fullmatch {spec!r}"

    if kind == "choice":
        choices = [str(c) for c in spec]
        if text in choices or text.strip() in choices:
            return None, ""
        return "choice", f"output is not one of {choices!r}"

    return None, ""
