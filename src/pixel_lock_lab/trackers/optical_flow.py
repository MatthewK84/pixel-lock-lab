"""Sparse Lucas-Kanade optical flow tracker.

The lock score is the fraction of seeded feature points that survive a
forward-backward consistency check. That makes the score fall off smoothly
under motion blur and occlusion rather than collapsing in a single step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from pixel_lock_lab.cv_compat import calc_optical_flow, empty_points, good_features
from pixel_lock_lab.errors import TrackerError
from pixel_lock_lab.imageutil import to_gray
from pixel_lock_lab.trackers.base import BaseTracker, Measurement

if TYPE_CHECKING:
    from pixel_lock_lab.array_types import Array
    from pixel_lock_lab.config.schemas import TrackerConfig
    from pixel_lock_lab.geometry import BoundingBox

LK_WINDOW: Final[tuple[int, int]] = (21, 21)
LK_LEVELS: Final[int] = 3
LK_CRITERIA: Final[tuple[int, int, float]] = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    30,
    0.01,
)
FEATURE_QUALITY: Final[float] = 0.01
FEATURE_MIN_DISTANCE: Final[float] = 3.0


def _seed_features(gray: Array, box: BoundingBox, max_points: int) -> Array:
    """Find trackable corners inside `box`, returned in full-frame coordinates."""
    mask: Array = np.zeros(gray.shape[:2], dtype=np.uint8)
    x, y, w, h = box.as_int_tuple()
    mask[y : y + h, x : x + w] = 255
    return good_features(gray, max_points, FEATURE_QUALITY, FEATURE_MIN_DISTANCE, mask)


def _flow_once(prev_gray: Array, next_gray: Array, points: Array) -> tuple[Array, Array]:
    """Run one LK pass, returning (tracked_points, status_flags)."""
    return calc_optical_flow(prev_gray, next_gray, points, LK_WINDOW, LK_LEVELS, LK_CRITERIA)


def _survivors(prev_gray: Array, next_gray: Array, points: Array, max_error_px: float) -> Array:
    """Points that pass a forward-backward consistency check, in new coordinates."""
    forward, fwd_status = _flow_once(prev_gray, next_gray, points)
    if forward.shape[0] == 0:
        return empty_points()
    backward, bwd_status = _flow_once(next_gray, prev_gray, forward)
    if backward.shape[0] == 0:
        return empty_points()
    error: Array = np.linalg.norm(points - backward, axis=2).reshape(-1)
    good: Array = (
        (fwd_status.reshape(-1) == 1) & (bwd_status.reshape(-1) == 1) & (error < max_error_px)
    )
    return np.asarray(forward[good], dtype=np.float32)


class LucasKanadeTracker(BaseTracker):
    """Sparse optical flow tracker with a survival-fraction lock score."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._prev_gray: Array | None = None
        self._points: Array = np.empty((0, 1, 2), dtype=np.float32)
        self._seed_count: int = 0

    def _initialize(self, frame: Array, bbox: BoundingBox) -> None:
        gray: Array = to_gray(frame)
        points: Array = _seed_features(gray, bbox, max_points=100)
        if points.shape[0] < self._config.min_flow_points:
            raise TrackerError(
                f"only {points.shape[0]} features in bbox; need {self._config.min_flow_points}"
            )
        self._prev_gray = gray
        self._points = points
        self._seed_count = points.shape[0]

    def _measure(self, frame: Array, _search_box: BoundingBox) -> Measurement:
        if self._prev_gray is None or self._bbox is None:
            return Measurement(None, 0.0)
        gray: Array = to_gray(frame)
        kept: Array = _survivors(
            self._prev_gray, gray, self._points, self._config.flow_fb_error_px
        )
        self._prev_gray = gray
        score: float = kept.shape[0] / float(max(self._seed_count, 1))
        if kept.shape[0] < self._config.min_flow_points:
            return Measurement(None, min(score, 1.0))
        self._points = kept
        centroid: Array = kept.reshape(-1, 2).mean(axis=0)
        box: BoundingBox = self._bbox.moved_to_center(float(centroid[0]), float(centroid[1]))
        self._maybe_reseed(gray, box, score)
        return Measurement(box, min(score, 1.0))

    def _maybe_reseed(self, gray: Array, box: BoundingBox, score: float) -> None:
        """Top the point set back up once attrition passes the update rate."""
        rate: float = self._config.template_update_rate
        if rate <= 0.0 or score > 0.75:
            return
        fresh: Array = _seed_features(gray, box, max_points=100)
        if fresh.shape[0] >= self._config.min_flow_points:
            self._points = fresh
            self._seed_count = fresh.shape[0]
