"""Tests for logging, drop analysis, and overlay rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixel_lock_lab.config.schemas import DiagnosticsConfig, TrackerConfig, TrackStatus
from pixel_lock_lab.diagnostics.drop_analyzer import (
    DropCause,
    attribute_cause,
    find_drop_events,
    summarize,
)
from pixel_lock_lab.diagnostics.log_schema import (
    TrackLogRecord,
    TrackLogWriter,
    read_log,
    record_from_state,
)
from pixel_lock_lab.diagnostics.overlay import annotate, render_score_plot
from pixel_lock_lab.errors import DiagnosticsError
from pixel_lock_lab.geometry import BoundingBox
from pixel_lock_lab.trackers.base import TrackState


def _record(
    index: int,
    score: float,
    status: TrackStatus = TrackStatus.LOCKED,
    **kwargs: float | bool,
) -> TrackLogRecord:
    return TrackLogRecord(
        frame_index=index,
        timestamp=index / 30.0,
        status=status,
        score=score,
        mean_intensity=float(kwargs.get("mean_intensity", 120.0)),
        residual_motion_px=float(kwargs.get("residual_motion_px", 0.0)),
        occluded=bool(kwargs.get("occluded", False)),
        latency_ms=float(kwargs.get("latency_ms", 5.0)),
    )


def _healthy(count: int, start: int = 0) -> list[TrackLogRecord]:
    return [_record(start + i, 0.9) for i in range(count)]


def test_record_from_state_maps_boxes() -> None:
    state = TrackState(
        frame_index=4,
        timestamp=0.0,
        status=TrackStatus.LOCKED,
        score=0.8,
        bbox=BoundingBox(1.0, 2.0, 3.0, 4.0),
        predicted_bbox=BoundingBox(5.0, 6.0, 7.0, 8.0),
        velocity=(1.0, -1.0),
        coast_frames=0,
        latency_ms=3.5,
    )
    record = record_from_state(state, timestamp=0.13)
    assert record.bbox == (1.0, 2.0, 3.0, 4.0)
    assert record.predicted_bbox == (5.0, 6.0, 7.0, 8.0)
    assert record.velocity == (1.0, -1.0)


def test_log_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "track.jsonl"
    records = _healthy(5)
    with TrackLogWriter(path) as writer:
        for record in records:
            writer.write(record)
    loaded = read_log(path)
    assert len(loaded) == 5
    assert loaded[0].score == pytest.approx(0.9)


def test_write_before_open_raises(tmp_path: Path) -> None:
    writer = TrackLogWriter(tmp_path / "x.jsonl")
    with pytest.raises(DiagnosticsError, match="open"):
        writer.write(_record(0, 0.5))


def test_read_missing_log_raises(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticsError, match="cannot read"):
        read_log(tmp_path / "missing.jsonl")


def test_read_corrupt_log_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="invalid log record"):
        read_log(path)


def test_no_events_when_healthy() -> None:
    assert find_drop_events(_healthy(30), DiagnosticsConfig()) == []


def test_empty_log_yields_no_events() -> None:
    assert find_drop_events([], DiagnosticsConfig()) == []


def test_detects_score_collapse() -> None:
    records = (
        _healthy(10)
        + [_record(10 + i, 0.1, TrackStatus.COASTING) for i in range(5)]
        + _healthy(10, start=15)
    )
    events = find_drop_events(records, DiagnosticsConfig(drop_consecutive_frames=3))
    assert len(events) == 1
    assert events[0].start_frame == 10
    assert events[0].end_frame == 14
    assert events[0].duration_frames == 5
    assert events[0].recovered is True


def test_short_dip_below_min_duration_ignored() -> None:
    records = [*_healthy(10), _record(10, 0.1, TrackStatus.COASTING), *_healthy(10, start=11)]
    assert find_drop_events(records, DiagnosticsConfig(drop_consecutive_frames=3)) == []


def test_unrecovered_drop_at_end() -> None:
    records = _healthy(10) + [_record(10 + i, 0.0, TrackStatus.LOST) for i in range(5)]
    events = find_drop_events(records, DiagnosticsConfig(drop_consecutive_frames=3))
    assert len(events) == 1
    assert events[0].recovered is False


def test_occlusion_cause_wins() -> None:
    window = [_record(i, 0.2, TrackStatus.COASTING, occluded=True) for i in range(4)]
    assert attribute_cause(window, 0.0) is DropCause.OCCLUSION


def test_motion_cause_from_residual() -> None:
    window = [_record(i, 0.2, TrackStatus.COASTING, residual_motion_px=20.0) for i in range(4)]
    assert attribute_cause(window, 0.0) is DropCause.MOTION


def test_low_contrast_cause() -> None:
    window = [_record(i, 0.2, TrackStatus.COASTING, mean_intensity=5.0) for i in range(4)]
    assert attribute_cause(window, 0.0) is DropCause.LOW_CONTRAST


def test_latency_cause_when_over_budget() -> None:
    window = [_record(i, 0.2, TrackStatus.COASTING, latency_ms=90.0) for i in range(4)]
    assert attribute_cause(window, latency_budget_ms=33.0) is DropCause.LATENCY


def test_score_decay_is_the_fallback() -> None:
    window = [_record(i, 0.2, TrackStatus.COASTING) for i in range(4)]
    assert attribute_cause(window, 0.0) is DropCause.SCORE_DECAY


def test_empty_window_is_unknown() -> None:
    assert attribute_cause([], 0.0) is DropCause.UNKNOWN


def test_summarize_counts() -> None:
    records = [
        *_healthy(10),
        *[_record(10 + i, 0.1, TrackStatus.COASTING) for i in range(4)],
        *_healthy(5, start=14),
    ]
    events = find_drop_events(records, DiagnosticsConfig(drop_consecutive_frames=3))
    stats = summarize(events)
    assert stats["event_count"] == 1
    assert stats["recovered"] == 1


def test_summarize_empty() -> None:
    assert summarize([])["event_count"] == 0


def test_annotate_returns_bgr_of_same_size() -> None:
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    state = TrackState(
        frame_index=1,
        timestamp=0.0,
        status=TrackStatus.LOCKED,
        score=0.8,
        bbox=BoundingBox(10.0, 10.0, 20.0, 20.0),
        predicted_bbox=BoundingBox(12.0, 12.0, 20.0, 20.0),
        velocity=(0.0, 0.0),
        coast_frames=0,
        latency_ms=1.0,
    )
    annotated = annotate(frame, state)
    assert annotated.shape == (100, 120, 3)


def test_render_score_plot_dimensions() -> None:
    plot = render_score_plot([0.9, 0.5, 0.2], width=200, height=80, tracker=TrackerConfig())
    assert plot.shape == (80, 200, 3)


def test_render_score_plot_handles_short_history() -> None:
    plot = render_score_plot([0.5], width=100, height=60, tracker=TrackerConfig())
    assert plot.shape == (60, 100, 3)
