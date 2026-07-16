"""Drop event extraction.

A drop is where the lock score collapses: either it falls below the
configured floor for N consecutive frames, or it falls faster than the
configured slope. Each event carries a window of pre-event frames and a
heuristic cause, which is the first thing you want when asking why the
tracker let go.

Causes are ranked evidence, not ground truth. They point you at the frames
worth watching.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

from pixel_lock_lab.config.schemas import DiagnosticsConfig, TrackStatus

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pixel_lock_lab.diagnostics.log_schema import TrackLogRecord

HIGH_RESIDUAL_PX: Final[float] = 6.0
LOW_CONTRAST_INTENSITY: Final[float] = 35.0
FAST_MOTION_PX: Final[float] = 12.0


class DropCause(str, Enum):
    """Heuristic attribution for a drop event."""

    OCCLUSION = "occlusion"
    MOTION = "motion"
    LOW_CONTRAST = "low_contrast"
    LATENCY = "latency"
    SCORE_DECAY = "score_decay"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DropEvent:
    """One lock collapse, with context and attribution."""

    start_frame: int
    end_frame: int
    lowest_score: float
    score_at_start: float
    max_slope: float
    cause: DropCause
    recovered: bool
    context: tuple[TrackLogRecord, ...]

    @property
    def duration_frames(self) -> int:
        """Number of frames the drop spanned."""
        return self.end_frame - self.start_frame + 1


def _max_slope(records: Sequence[TrackLogRecord]) -> float:
    """Largest single-frame score decrease across the window."""
    if len(records) < 2:
        return 0.0
    drops: list[float] = [records[i - 1].score - records[i].score for i in range(1, len(records))]
    return max(max(drops), 0.0)


def _is_dropped(record: TrackLogRecord, config: DiagnosticsConfig) -> bool:
    """True when this frame counts as part of a collapse."""
    if record.status in (TrackStatus.COASTING, TrackStatus.LOST):
        return True
    return record.score < config.drop_score_threshold


def _find_runs(
    records: Sequence[TrackLogRecord], config: DiagnosticsConfig
) -> list[tuple[int, int]]:
    """Index ranges where the drop condition holds long enough to count."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, record in enumerate(records):
        if _is_dropped(record, config):
            start = index if start is None else start
            continue
        if start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(records) - 1))
    return [r for r in runs if r[1] - r[0] + 1 >= config.drop_consecutive_frames]


def attribute_cause(window: Sequence[TrackLogRecord], latency_budget_ms: float) -> DropCause:
    """Rank the available evidence and name the most likely cause."""
    if not window:
        return DropCause.UNKNOWN
    if any(r.occluded for r in window):
        return DropCause.OCCLUSION
    if max(r.residual_motion_px for r in window) > HIGH_RESIDUAL_PX:
        return DropCause.MOTION
    if max(_speed(r) for r in window) > FAST_MOTION_PX:
        return DropCause.MOTION
    if min(r.mean_intensity for r in window) < LOW_CONTRAST_INTENSITY:
        return DropCause.LOW_CONTRAST
    if latency_budget_ms > 0.0 and max(r.latency_ms for r in window) > latency_budget_ms:
        return DropCause.LATENCY
    return DropCause.SCORE_DECAY


def _speed(record: TrackLogRecord) -> float:
    vx, vy = record.velocity
    return float((vx * vx + vy * vy) ** 0.5)


def _build_event(
    records: Sequence[TrackLogRecord],
    run: tuple[int, int],
    config: DiagnosticsConfig,
    latency_budget_ms: float,
) -> DropEvent:
    begin, end = run
    span: Sequence[TrackLogRecord] = records[begin : end + 1]
    context_start: int = max(0, begin - config.pre_event_frames)
    context: Sequence[TrackLogRecord] = records[context_start : end + 1]
    recovered: bool = end + 1 < len(records) and records[end + 1].status is TrackStatus.LOCKED
    return DropEvent(
        start_frame=span[0].frame_index,
        end_frame=span[-1].frame_index,
        lowest_score=min(r.score for r in span),
        score_at_start=records[context_start].score,
        max_slope=_max_slope(context),
        cause=attribute_cause(context, latency_budget_ms),
        recovered=recovered,
        context=tuple(context),
    )


def find_drop_events(
    records: Sequence[TrackLogRecord],
    config: DiagnosticsConfig,
    latency_budget_ms: float = 0.0,
) -> list[DropEvent]:
    """Extract every drop event from a track log."""
    if not records:
        return []
    runs: list[tuple[int, int]] = _find_runs(records, config)
    return [_build_event(records, run, config, latency_budget_ms) for run in runs]


def summarize(events: Sequence[DropEvent]) -> dict[str, int | float]:
    """Aggregate counts and durations across drop events."""
    if not events:
        return {"event_count": 0, "total_dropped_frames": 0, "recovered": 0, "mean_duration": 0.0}
    durations: list[int] = [e.duration_frames for e in events]
    return {
        "event_count": len(events),
        "total_dropped_frames": sum(durations),
        "recovered": sum(1 for e in events if e.recovered),
        "mean_duration": sum(durations) / len(durations),
    }
