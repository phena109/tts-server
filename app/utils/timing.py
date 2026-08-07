"""Timing helpers for structured generation metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimerResult:
    elapsed_ms: float


@contextmanager
def timer() -> Iterator[TimerResult]:
    """Context manager that records wall-clock elapsed milliseconds."""
    result = TimerResult(elapsed_ms=0.0)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000.0


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
