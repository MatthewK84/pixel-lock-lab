"""Tests for the tracker interface, lock policy, and backends."""

from __future__ import annotations

import numpy as np
import pytest

from pixel_lock_lab.array_types import Array
from pixel_lock_lab.config.schemas import SourceConfig, TrackerBackend, TrackerConfig, TrackStatus
from pixel_lock_lab.errors import TrackerError
from pixel_lock_lab.geometry import BoundingBox, center_distance
from pixel_lock_lab.sources import SyntheticSource
from pixel_lock_lab.trackers import TemplateMatchTracker, build_tracker
from pixel_lock_lab.trackers.base import BaseTracker, Measurement
from pixel_lock_lab.trackers.deep import Detection, DetectionTracker, associate


class ScriptedTracker(BaseTracker):
    """Test double that replays a fixed sequence of measurements."""

    def __init__(self, config: TrackerConfig, script: list[Measurement]) -> None:
        super().__init__(config)
        self._script: list[Measurement] = script
        self._step: int = 0

    def _initialize(self, frame: Array, bbox: BoundingBox) -> None:
        self._step = 0

    def _measure(self, frame: Array, search_box: BoundingBox) -> Measurement:
        if self._step >= len(self._script):
            return Measurement(None, 0.0)
        result = self._script[self._step]
        self._step += 1
        return result


class StubDetector:
    """Detector returning a fixed list every frame."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, frame: Array) -> list[Detection]:
        return self._detections


@pytest.fixture
def blank_frame() -> Array:
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_frames() -> list[Array]:
    source = SyntheticSource(SourceConfig(synthetic_frames=40, frame_width=320, frame_height=240))
    return list(source.frames())


def test_update_before_init_raises(blank_frame: Array) -> None:
    tracker = TemplateMatchTracker(TrackerConfig())
    with pytest.raises(TrackerError, match="init"):
        tracker.update(blank_frame)


def test_init_rejects_offscreen_bbox(blank_frame: Array) -> None:
    tracker = TemplateMatchTracker(TrackerConfig())
    with pytest.raises(TrackerError, match="outside the frame"):
        tracker.init(blank_frame, BoundingBox(900.0, 900.0, 20.0, 20.0))


def test_init_rejects_tiny_bbox(blank_frame: Array) -> None:
    tracker = TemplateMatchTracker(TrackerConfig())
    with pytest.raises(TrackerError, match="too small"):
        tracker.init(blank_frame, BoundingBox(10.0, 10.0, 3.0, 3.0))


def test_init_sets_locked(blank_frame: Array) -> None:
    tracker = TemplateMatchTracker(TrackerConfig())
    state = tracker.init(blank_frame, BoundingBox(100.0, 100.0, 30.0, 30.0))
    assert state.status is TrackStatus.LOCKED
    assert state.score == 1.0


def test_template_tracker_follows_synthetic_target(synthetic_frames: list[Array]) -> None:
    source = SyntheticSource(SourceConfig(synthetic_frames=40, frame_width=320, frame_height=240))
    tracker = TemplateMatchTracker(TrackerConfig(lock_threshold=0.4, reacquire_threshold=0.5))
    truth = source.ground_truth(0)
    assert truth is not None
    tracker.init(synthetic_frames[0], truth)
    for index in range(1, len(synthetic_frames)):
        state = tracker.update(synthetic_frames[index])
    expected = source.ground_truth(len(synthetic_frames) - 1)
    assert expected is not None
    assert state.bbox is not None
    assert center_distance(state.bbox, expected) < 8.0


def test_low_score_triggers_coasting(blank_frame: Array) -> None:
    config = TrackerConfig(lock_threshold=0.5, reacquire_threshold=0.8, max_coast_frames=3)
    script = [Measurement(None, 0.0)] * 2
    tracker = ScriptedTracker(config, script)
    tracker.init(blank_frame, BoundingBox(100.0, 100.0, 20.0, 20.0))
    state = tracker.update(blank_frame)
    assert state.status is TrackStatus.COASTING
    assert state.coast_frames == 1


def test_coast_budget_exhaustion_declares_lost(blank_frame: Array) -> None:
    config = TrackerConfig(max_coast_frames=2)
    tracker = ScriptedTracker(config, [Measurement(None, 0.0)] * 5)
    tracker.init(blank_frame, BoundingBox(100.0, 100.0, 20.0, 20.0))
    statuses = [tracker.update(blank_frame).status for _ in range(3)]
    assert statuses == [TrackStatus.COASTING, TrackStatus.COASTING, TrackStatus.LOST]


def test_hysteresis_blocks_weak_reacquire(blank_frame: Array) -> None:
    config = TrackerConfig(lock_threshold=0.4, reacquire_threshold=0.9, max_coast_frames=10)
    box = BoundingBox(100.0, 100.0, 20.0, 20.0)
    script = [Measurement(None, 0.0), Measurement(box, 0.6), Measurement(box, 0.95)]
    tracker = ScriptedTracker(config, script)
    tracker.init(blank_frame, box)
    assert tracker.update(blank_frame).status is TrackStatus.COASTING
    # 0.6 clears lock_threshold but not reacquire_threshold, so still coasting.
    assert tracker.update(blank_frame).status is TrackStatus.COASTING
    assert tracker.update(blank_frame).status is TrackStatus.LOCKED


def test_reset_clears_state(blank_frame: Array) -> None:
    tracker = TemplateMatchTracker(TrackerConfig())
    tracker.init(blank_frame, BoundingBox(100.0, 100.0, 20.0, 20.0))
    tracker.reset()
    assert tracker.status is TrackStatus.UNINITIALIZED
    assert tracker.get_score() == 0.0


def test_build_tracker_returns_requested_backend() -> None:
    tracker = build_tracker(TrackerConfig(backend=TrackerBackend.TEMPLATE))
    assert isinstance(tracker, TemplateMatchTracker)


def test_build_detection_tracker_without_detector_raises() -> None:
    with pytest.raises(TrackerError, match="requires a detector"):
        build_tracker(TrackerConfig(backend=TrackerBackend.DETECTION))


def test_associate_picks_best_overlap() -> None:
    reference = BoundingBox(0.0, 0.0, 10.0, 10.0)
    far = Detection(BoundingBox(50.0, 50.0, 10.0, 10.0), 0.9)
    near = Detection(BoundingBox(1.0, 1.0, 10.0, 10.0), 0.5)
    assert associate([far, near], reference, min_iou=0.3) is near


def test_associate_returns_none_below_threshold() -> None:
    reference = BoundingBox(0.0, 0.0, 10.0, 10.0)
    far = Detection(BoundingBox(80.0, 80.0, 10.0, 10.0), 0.9)
    assert associate([far], reference, min_iou=0.3) is None


def test_detection_tracker_tracks_matching_detection(blank_frame: Array) -> None:
    box = BoundingBox(100.0, 100.0, 20.0, 20.0)
    detector = StubDetector([Detection(box, 0.95)])
    tracker = DetectionTracker(TrackerConfig(backend=TrackerBackend.DETECTION), detector)
    tracker.init(blank_frame, box)
    state = tracker.update(blank_frame)
    assert state.status is TrackStatus.LOCKED
    assert state.score == pytest.approx(0.95)


def test_detection_tracker_rejects_bad_detector() -> None:
    with pytest.raises(TrackerError, match="detect"):
        DetectionTracker(TrackerConfig(), object())  # type: ignore[arg-type]
