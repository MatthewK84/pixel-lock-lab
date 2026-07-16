"""Timing instrumentation.

Uses perf_counter, records per-stage samples, and reports percentiles.
In the field the tail matters more than the mean, so p95 and p99 are
first-class outputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from types import TracebackType

MS_PER_SECOND: Final[float] = 1000.0


@dataclass(frozen=True)
class StageStats:
    """Latency summary for one pipeline stage, in milliseconds."""

    stage: str
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class LatencyRecorder:
    """Collects per-stage timing samples and summarizes them."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {}

    def record(self, stage: str, elapsed_ms: float) -> None:
        """Add one timing sample for `stage`."""
        if elapsed_ms < 0.0:
            raise ValueError(f"elapsed_ms must be non-negative, got {elapsed_ms}")
        self._samples.setdefault(stage, []).append(elapsed_ms)

    def measure(self, stage: str) -> StageTimer:
        """Context manager that records the wrapped block's duration."""
        return StageTimer(self, stage)

    def stages(self) -> list[str]:
        """Names of all recorded stages."""
        return sorted(self._samples)

    def stats(self, stage: str) -> StageStats:
        """Percentile summary for one stage."""
        samples: list[float] = self._samples.get(stage, [])
        if not samples:
            return StageStats(stage, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        array: np.ndarray = np.asarray(samples, dtype=np.float64)
        return StageStats(
            stage=stage,
            count=int(array.size),
            mean_ms=float(array.mean()),
            p50_ms=float(np.percentile(array, 50)),
            p95_ms=float(np.percentile(array, 95)),
            p99_ms=float(np.percentile(array, 99)),
            max_ms=float(array.max()),
        )

    def summary(self) -> list[StageStats]:
        """Percentile summaries for every recorded stage."""
        return [self.stats(name) for name in self.stages()]

    def total_mean_ms(self) -> float:
        """Sum of per-stage means, i.e. the mean end-to-end frame cost."""
        return float(sum(s.mean_ms for s in self.summary()))


class StageTimer:
    """Context manager returned by `LatencyRecorder.measure`."""

    def __init__(self, recorder: LatencyRecorder, stage: str) -> None:
        self._recorder: LatencyRecorder = recorder
        self._stage: str = stage
        self._start: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> StageTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * MS_PER_SECOND
        self._recorder.record(self._stage, self.elapsed_ms)
