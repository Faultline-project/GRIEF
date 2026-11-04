"""
Runner shim: re-exports from core.runner with VllmBaseAdapter as default.

This module preserves existing import paths. The core Runner lives in
fuzzer/core/runner.py. When Runner is instantiated without an adapter,
VllmBaseAdapter is injected automatically.
"""

from __future__ import annotations

from .core.runner import Runner as _CoreRunner, RunnerConfig
from .adapters.vllm_base.adapter import VllmBaseAdapter
from .executor import Executor, PythonExecutor
from .monitor import Monitor


class Runner(_CoreRunner):
    """Runner with VllmBaseAdapter injected as default adapter."""

    def __init__(self, config, adapter=None, executor=None, monitor=None):
        if adapter is None:
            adapter = VllmBaseAdapter()
        super().__init__(config, adapter=adapter, executor=executor, monitor=monitor)


# Re-export RunnerConfig for callers who do: from fuzzer.runner import RunnerConfig
RunnerConfig = RunnerConfig

__all__ = ["Runner", "RunnerConfig"]
