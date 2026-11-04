"""Confirmation of output-divergence candidates inside the fuzzing loop.

``bi_guard``  proves a batch-invariant server is engaged and its attention
              backend is BI-sound (moved from ``scripts/bi_confirm.py``).
``ladder``    the arms, the comparisons and the verdict.  Pure.

The runner (``fuzzer.core.runner``) owns the queue, the execution and the
feedback to the corpus.
"""
