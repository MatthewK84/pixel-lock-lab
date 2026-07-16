"""Structured track logs.

JSONL is the on-disk default: it is append-only, survives a crash mid-run,
and needs no dependencies. Parquet export is available when Polars is
installed, which is worth it once logs get large.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict, Field

from pixel_lock_lab.config.schemas import TrackStatus
from pixel_lock_lab.errors import DiagnosticsError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from types import TracebackType

    from pixel_lock_lab.trackers.base import TrackState

LOG_SCHEMA_VERSION: Final[str] = "1.0"


class TrackLogRecord(BaseModel):
    """One row of the track log."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = LOG_SCHEMA_VERSION
    frame_index: int
    timestamp: float
    status: TrackStatus
    score: float
    bbox: tuple[float, float, float, float] | None = None
    predicted_bbox: tuple[float, float, float, float] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    coast_frames: int = 0
    latency_ms: float = 0.0
    residual_motion_px: float = 0.0
    occluded: bool = False
    mean_intensity: float = Field(default=0.0, ge=0.0, le=255.0)


def record_from_state(
    state: TrackState,
    timestamp: float,
    residual_motion_px: float = 0.0,
    occluded: bool = False,
    mean_intensity: float = 0.0,
) -> TrackLogRecord:
    """Build a log record from a TrackState plus per-frame context."""
    return TrackLogRecord(
        frame_index=state.frame_index,
        timestamp=timestamp,
        status=state.status,
        score=state.score,
        bbox=None if state.bbox is None else _box_tuple(state.bbox),
        predicted_bbox=None if state.predicted_bbox is None else _box_tuple(state.predicted_bbox),
        velocity=state.velocity,
        coast_frames=state.coast_frames,
        latency_ms=state.latency_ms,
        residual_motion_px=residual_motion_px,
        occluded=occluded,
        mean_intensity=mean_intensity,
    )


def _box_tuple(box: Any) -> tuple[float, float, float, float]:
    return (float(box.x), float(box.y), float(box.width), float(box.height))


class TrackLogWriter:
    """Append-only JSONL writer usable as a context manager."""

    def __init__(self, path: Path) -> None:
        self._path: Path = path
        self._handle: Any = None

    def open(self) -> None:
        """Open the log file for appending, creating parent directories."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("w", encoding="utf-8")
        except OSError as exc:
            raise DiagnosticsError(f"cannot open log {self._path}: {exc}") from exc

    def write(self, record: TrackLogRecord) -> None:
        """Append one record."""
        if self._handle is None:
            raise DiagnosticsError("TrackLogWriter.open() must be called before write()")
        try:
            self._handle.write(record.model_dump_json() + "\n")
        except OSError as exc:
            raise DiagnosticsError(f"cannot write to log {self._path}: {exc}") from exc

    def close(self) -> None:
        """Flush and close the log file."""
        if self._handle is None:
            return
        try:
            self._handle.close()
        except OSError as exc:
            raise DiagnosticsError(f"cannot close log {self._path}: {exc}") from exc
        self._handle = None

    def __enter__(self) -> TrackLogWriter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_log(path: Path) -> list[TrackLogRecord]:
    """Load and validate a JSONL track log."""
    try:
        lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DiagnosticsError(f"cannot read log {path}: {exc}") from exc
    return list(_parse_lines(lines, path))


def _parse_lines(lines: Sequence[str], path: Path) -> Iterator[TrackLogRecord]:
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            yield TrackLogRecord.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DiagnosticsError(f"invalid log record at {path}:{number}: {exc}") from exc


def to_parquet(records: Sequence[TrackLogRecord], path: Path) -> None:
    """Write records to Parquet. Requires the optional 'data' extra."""
    try:
        import polars as pl
    except ImportError as exc:
        raise DiagnosticsError(
            "Parquet export requires Polars: pip install 'pixel-lock-lab[data]'"
        ) from exc
    rows: list[dict[str, Any]] = [r.model_dump(mode="json") for r in records]
    try:
        pl.DataFrame(rows).write_parquet(path)
    except OSError as exc:
        raise DiagnosticsError(f"cannot write parquet {path}: {exc}") from exc
