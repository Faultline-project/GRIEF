"""Fixtures shared across the test suite."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_grief_env():
    """Restore every ``GRIEF_*`` environment variable after each test.

    ``build_adapter()`` writes ``GRIEF_FINGERPRINT_VERSION`` into ``os.environ``
    itself (that is the mechanism the CLI uses to reach ``Corpus``), so a test
    that calls it leaks the value to every later test.  ``monkeypatch.delenv(...,
    raising=False)`` does *not* protect against this: when the variable is
    absent it has nothing to record, so its teardown restores nothing.

    Concretely, before this fixture existed:

        pytest tests/test_runner.py tests/test_corpus.py

    failed on ``TestTransitionNovelty::test_edges_are_counted_but_not_scored_
    under_v1`` -- it asserts ``corpus.fingerprint_version == 1`` and picked up
    the ``"2"`` left behind by ``TestFingerprintVersionWiring``.  The full-suite
    run happened to pass only because ``test_corpus`` sorts before
    ``test_runner``.
    """
    import os
    before = {k: v for k, v in os.environ.items() if k.startswith("GRIEF_")}
    try:
        yield
    finally:
        for k in [k for k in os.environ if k.startswith("GRIEF_")]:
            if k not in before:
                del os.environ[k]
        os.environ.update(before)
