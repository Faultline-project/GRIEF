"""
Core Executor ABC — mode-agnostic interface for running a Trace.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Executor(ABC):
    @abstractmethod
    async def run(self, trace, base_url: str):
        ...
