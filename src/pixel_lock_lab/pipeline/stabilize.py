"""Motion compensation: IMU feed-forward and visual EIS.

IMU mode converts angular rates to a pixel shift via the pinhole relation
shift_px = focal_length_px * tan(angle_rad), which is well conditioned for
the small angles seen between consecutive frames.

Both modes report residual motion so you can tell whether a drop followed
under-compensation or the stabilizer itself fighting the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from pixel_lock_lab.config.schemas import StabilizeConfig, StabilizeMode
from pixel_lock_lab.cv_compat import calc_optical_flow, good_features
from pixel_lock_lab.imageutil import to_gray

if TYPE_CHECKING:
    from pixel_lock_lab.array_types import Array

MIN_MATCH_POINTS: Final[int] = 6
FEATURE_QUALITY: Final[float] = 0.01
FEATURE_MIN_DISTANCE: Final[float] = 8.0


@dataclass(frozen=True)
class ImuSample:
    """Angular rates in degrees per second at a point in time."""

    timestamp: float
    yaw_rate_dps: float
    pitch_rate_dps: float
    roll_rate_dps: float = 0.0


@dataclass(frozen=True)
class StabilizationResult:
    """Stabilized frame plus the shift applied and what remained."""

    frame: Array
    applied_shift_px: tuple[float, float]
    residual_px: float
    mode: StabilizeMode


def rates_to_shift(sample: ImuSample, focal_length_px: float, dt: float) -> tuple[float, float]:
    """Convert angular rates over `dt` seconds into an image-plane shift."""
    yaw_rad: float = math.radians(sample.yaw_rate_dps * dt)
    pitch_rad: float = math.radians(sample.pitch_rate_dps * dt)
    return (-focal_length_px * math.tan(yaw_rad), focal_length_px * math.tan(pitch_rad))


def translate(frame: Array, dx: float, dy: float) -> Array:
    """Shift a frame by (dx, dy) pixels with replicated borders."""
    matrix: Array = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def estimate_global_shift(prev_gray: Array, gray: Array, max_features: int) -> tuple[float, float]:
    """Estimate frame-to-frame translation from sparse feature correspondences."""
    prev_points: Array = good_features(
        prev_gray, max_features, FEATURE_QUALITY, FEATURE_MIN_DISTANCE
    )
    if prev_points.shape[0] < MIN_MATCH_POINTS:
        return (0.0, 0.0)
    moved, status = calc_optical_flow(prev_gray, gray, prev_points)
    if moved.shape[0] == 0:
        return (0.0, 0.0)
    good: Array = status.reshape(-1) == 1
    if int(good.sum()) < MIN_MATCH_POINTS:
        return (0.0, 0.0)
    deltas: Array = (moved[good] - prev_points[good]).reshape(-1, 2)
    median: Array = np.median(deltas, axis=0)
    return (float(median[0]), float(median[1]))


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class Stabilizer:
    """Applies IMU feed-forward or visual EIS and reports residual motion."""

    def __init__(self, config: StabilizeConfig) -> None:
        self._config: StabilizeConfig = config
        self._prev_gray: Array | None = None
        self._smoothed: tuple[float, float] = (0.0, 0.0)

    def reset(self) -> None:
        """Clear accumulated stabilization state."""
        self._prev_gray = None
        self._smoothed = (0.0, 0.0)

    def apply(
        self, frame: Array, imu: ImuSample | None = None, dt: float = 1.0 / 30.0
    ) -> StabilizationResult:
        """Stabilize one frame according to the configured mode."""
        if self._config.mode is StabilizeMode.NONE:
            return StabilizationResult(frame, (0.0, 0.0), 0.0, StabilizeMode.NONE)
        if self._config.mode is StabilizeMode.IMU:
            return self._apply_imu(frame, imu, dt)
        return self._apply_visual(frame)

    def _apply_imu(self, frame: Array, imu: ImuSample | None, dt: float) -> StabilizationResult:
        if imu is None:
            return StabilizationResult(frame, (0.0, 0.0), 0.0, StabilizeMode.IMU)
        lead: float = dt + self._config.imu_lead_seconds
        raw_dx, raw_dy = rates_to_shift(imu, self._config.focal_length_px, lead)
        dx: float = _clamp(-raw_dx, self._config.max_shift_px)
        dy: float = _clamp(-raw_dy, self._config.max_shift_px)
        stabilized: Array = translate(frame, dx, dy)
        residual: float = self._measure_residual(stabilized)
        return StabilizationResult(stabilized, (dx, dy), residual, StabilizeMode.IMU)

    def _apply_visual(self, frame: Array) -> StabilizationResult:
        gray: Array = to_gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return StabilizationResult(frame, (0.0, 0.0), 0.0, StabilizeMode.VISUAL)
        raw_dx, raw_dy = estimate_global_shift(self._prev_gray, gray, self._config.max_features)
        alpha: float = self._config.smoothing_alpha
        prev_x, prev_y = self._smoothed
        self._smoothed = (
            alpha * prev_x + (1.0 - alpha) * raw_dx,
            alpha * prev_y + (1.0 - alpha) * raw_dy,
        )
        dx: float = _clamp(-(raw_dx - self._smoothed[0]), self._config.max_shift_px)
        dy: float = _clamp(-(raw_dy - self._smoothed[1]), self._config.max_shift_px)
        stabilized: Array = translate(frame, dx, dy)
        self._prev_gray = to_gray(stabilized)
        residual: float = math.hypot(raw_dx + dx, raw_dy + dy)
        return StabilizationResult(stabilized, (dx, dy), residual, StabilizeMode.VISUAL)

    def _measure_residual(self, stabilized: Array) -> float:
        """Residual motion left after compensation, measured visually."""
        gray: Array = to_gray(stabilized)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        res_x, res_y = estimate_global_shift(self._prev_gray, gray, self._config.max_features)
        self._prev_gray = gray
        return math.hypot(res_x, res_y)
