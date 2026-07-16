"""Common tracker interface and the shared lock / coast state machine.

Backends implement `_initialize` and `_measure` only. The lock threshold,
re-acquire hysteresis, and coasting logic live here so every backend behaves
identically with respect to those parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pixel_lock_lab.config.schemas import TrackerConfig, TrackStatus
from pixel_lock_lab.errors import TrackerError
from pixel_lock_lab.geometry import BoundingBox, clip_to_frame

if TYPE_CHECKING:
    import numpy as np

MIN_TEMPLATE_PX: Final[int] = 4


@dataclass(frozen=True)
class Measurement:
    """Raw backend output before lock policy is applied."""

    bbox: BoundingBox | None
    score: float


@dataclass(frozen=True)
class TrackState:
    """Full public state of a track for one frame."""

    frame_index: int
    timestamp: float
    status: TrackStatus
    score: float
    bbox: BoundingBox | None
    predicted_bbox: BoundingBox | None
    velocity: tuple[float, float]
    coast_frames: int
    latency_ms: float


def _validate_frame(frame: np.ndarray) -> None:
    if frame.ndim not in (2, 3):
        raise TrackerError(f"frame must be 2D or 3D, got shape {frame.shape}")
    if frame.size == 0:
        raise TrackerError("frame is empty")


class BaseTracker(ABC):
    """Abstract tracker with shared lock, hysteresis, and coasting behavior."""

    def __init__(self, config: TrackerConfig) -> None:
        self._config: TrackerConfig = config
        self._status: TrackStatus = TrackStatus.UNINITIALIZED
        self._bbox: BoundingBox | None = None
        self._score: float = 0.0
        self._velocity: tuple[float, float] = (0.0, 0.0)
        self._coast_frames: int = 0
        self._frame_index: int = -1
        self._anchor: BoundingBox | None = None

    @property
    def config(self) -> TrackerConfig:
        """The tracker configuration in use."""
        return self._config

    @property
    def status(self) -> TrackStatus:
        """Current lifecycle status."""
        return self._status

    def get_score(self) -> float:
        """Most recent lock score in [0, 1]."""
        return self._score

    @abstractmethod
    def _initialize(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        """Backend-specific setup for a new target."""

    @abstractmethod
    def _measure(self, frame: np.ndarray, search_box: BoundingBox) -> Measurement:
        """Backend-specific single-frame measurement within `search_box`."""

    def init(self, frame: np.ndarray, bbox: BoundingBox) -> TrackState:
        """Start tracking `bbox` in `frame`. Resets all internal state."""
        _validate_frame(frame)
        clipped: BoundingBox | None = clip_to_frame(bbox, frame.shape[1], frame.shape[0])
        if clipped is None:
            raise TrackerError(f"initial bbox {bbox} lies outside the frame")
        if clipped.width < MIN_TEMPLATE_PX or clipped.height < MIN_TEMPLATE_PX:
            raise TrackerError(f"initial bbox too small: {clipped.width}x{clipped.height}")
        self.reset()
        self._initialize(frame, clipped)
        self._bbox = clipped
        self._anchor = clipped
        self._score = 1.0
        self._status = TrackStatus.LOCKED
        self._frame_index = 0
        return self._build_state(0.0, clipped)

    def reset(self) -> None:
        """Clear all track state back to uninitialized."""
        self._status = TrackStatus.UNINITIALIZED
        self._bbox = None
        self._score = 0.0
        self._velocity = (0.0, 0.0)
        self._coast_frames = 0
        self._frame_index = -1
        self._anchor = None

    def update(self, frame: np.ndarray, latency_ms: float = 0.0) -> TrackState:
        """Advance the track by one frame and apply the lock policy."""
        _validate_frame(frame)
        if self._status is TrackStatus.UNINITIALIZED:
            raise TrackerError("tracker.init() must be called before update()")
        self._frame_index += 1
        predicted: BoundingBox | None = self._predict(frame)
        measurement: Measurement = self._measure_safely(frame, predicted)
        self._apply_policy(frame, measurement, predicted)
        return self._build_state(latency_ms, predicted)

    def _measure_safely(self, frame: np.ndarray, predicted: BoundingBox | None) -> Measurement:
        if predicted is None:
            return Measurement(None, 0.0)
        search: BoundingBox | None = clip_to_frame(
            predicted.scaled(self._search_scale()), frame.shape[1], frame.shape[0]
        )
        if search is None:
            return Measurement(None, 0.0)
        return self._measure(frame, search)

    def _search_scale(self) -> float:
        """Search window size, growing with time since loss as uncertainty grows.

        A lost target keeps moving, so a fixed window only ever re-acquires
        targets that stopped near where the lock broke.
        """
        if self._status is not TrackStatus.LOST:
            return self._config.search_window_scale
        lost_frames: int = max(self._coast_frames - self._config.max_coast_frames, 0)
        grown: float = self._config.lost_search_scale * (
            1.0 + self._config.lost_search_growth * lost_frames
        )
        return min(grown, self._config.max_lost_search_scale)

    def _predict(self, frame: np.ndarray) -> BoundingBox | None:
        """Where to look next: velocity-projected when tracking, the anchor when lost."""
        if self._bbox is None:
            return self._anchor_region(frame)
        vx, vy = self._velocity
        moved: BoundingBox = self._bbox.translated(vx, vy)
        return clip_to_frame(moved, frame.shape[1], frame.shape[0])

    def _anchor_region(self, frame: np.ndarray) -> BoundingBox | None:
        """Last known position, kept so a LOST track can still be re-acquired."""
        if self._anchor is None:
            return None
        return clip_to_frame(self._anchor, frame.shape[1], frame.shape[0])

    def _apply_policy(
        self, frame: np.ndarray, measurement: Measurement, predicted: BoundingBox | None
    ) -> None:
        """Route the measurement through lock, coast, or lost handling."""
        self._score = measurement.score
        if self._accepts(measurement):
            self._accept_measurement(measurement)
            return
        self._coast(frame, predicted)

    def _accepts(self, measurement: Measurement) -> bool:
        """Apply hysteresis: re-acquire needs a higher score than holding lock."""
        if measurement.bbox is None:
            return False
        if self._status is TrackStatus.LOCKED:
            return measurement.score >= self._config.lock_threshold
        return measurement.score >= self._config.reacquire_threshold

    def _accept_measurement(self, measurement: Measurement) -> None:
        if measurement.bbox is None:
            raise TrackerError("_accept_measurement requires a measurement with a bbox")
        new_box: BoundingBox = measurement.bbox
        if self._bbox is not None:
            self._velocity = self._smooth_velocity(self._bbox, new_box)
        self._bbox = new_box
        self._anchor = new_box
        self._coast_frames = 0
        self._status = TrackStatus.LOCKED

    def _smooth_velocity(self, old: BoundingBox, new: BoundingBox) -> tuple[float, float]:
        alpha: float = self._config.velocity_smoothing
        old_cx, old_cy = old.center
        new_cx, new_cy = new.center
        vx, vy = self._velocity
        return (
            alpha * vx + (1.0 - alpha) * (new_cx - old_cx),
            alpha * vy + (1.0 - alpha) * (new_cy - old_cy),
        )

    def _coast(self, frame: np.ndarray, predicted: BoundingBox | None) -> None:
        """Dead-reckon through a dropout until the coast budget expires."""
        self._coast_frames += 1
        if self._coast_frames > self._config.max_coast_frames or predicted is None:
            self._status = TrackStatus.LOST
            self._bbox = None
            self._velocity = (0.0, 0.0)
            return
        decay: float = self._config.coast_velocity_decay
        vx, vy = self._velocity
        self._velocity = (vx * decay, vy * decay)
        self._bbox = clip_to_frame(predicted, frame.shape[1], frame.shape[0])
        if self._bbox is not None:
            self._anchor = self._bbox
        self._status = TrackStatus.COASTING

    def _build_state(self, latency_ms: float, predicted: BoundingBox | None) -> TrackState:
        return TrackState(
            frame_index=self._frame_index,
            timestamp=0.0,
            status=self._status,
            score=self._score,
            bbox=self._bbox,
            predicted_bbox=predicted,
            velocity=self._velocity,
            coast_frames=self._coast_frames,
            latency_ms=latency_ms,
        )

    def get_state(self) -> TrackState:
        """Return the current state without advancing the track."""
        return self._build_state(0.0, self._bbox)
