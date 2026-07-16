"""Shoulder-fired motion model.

Generates angular rate streams that resemble a hand-held or shoulder-braced
sensor: physiological tremor around 8-12 Hz, a slow breathing component, a
constant slew term, and an optional recoil transient with exponential decay.

The generated ImuSample stream can be fed straight to the IMU stabilizer,
which is the point: you can test compensation against motion you control.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pixel_lock_lab.pipeline.stabilize import ImuSample, translate

if TYPE_CHECKING:
    from pixel_lock_lab.config.schemas import MotionConfig


def _tremor(config: MotionConfig, t: float, phase: float) -> float:
    return config.tremor_amplitude_dps * math.sin(2.0 * math.pi * config.tremor_hz * t + phase)


def _breathing(config: MotionConfig, t: float, phase: float) -> float:
    return config.breathing_amplitude_dps * math.sin(
        2.0 * math.pi * config.breathing_hz * t + phase
    )


def _recoil(config: MotionConfig, frame_index: int) -> float:
    """Exponentially decaying transient starting at `recoil_frame`."""
    start: int | None = config.recoil_frame
    if start is None or frame_index < start:
        return 0.0
    elapsed: int = frame_index - start
    decay: float = math.exp(-elapsed / float(config.recoil_decay_frames))
    return config.recoil_amplitude_dps * decay * math.cos(elapsed * 0.9)


class MotionGenerator:
    """Produces a deterministic angular rate stream and can warp frames by it."""

    def __init__(self, config: MotionConfig, fps: float = 30.0) -> None:
        if fps <= 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        self._config: MotionConfig = config
        self._dt: float = 1.0 / fps
        rng: np.random.Generator = np.random.default_rng(config.seed)
        drawn: np.ndarray = rng.uniform(0.0, 2.0 * math.pi, size=4)
        self._phases: tuple[float, float, float, float] = (
            float(drawn[0]),
            float(drawn[1]),
            float(drawn[2]),
            float(drawn[3]),
        )

    @property
    def dt(self) -> float:
        """Seconds between frames."""
        return self._dt

    def sample(self, frame_index: int) -> ImuSample:
        """Angular rates for `frame_index`, in degrees per second."""
        if not self._config.enabled:
            return ImuSample(frame_index * self._dt, 0.0, 0.0, 0.0)
        t: float = frame_index * self._dt
        yaw: float = (
            _tremor(self._config, t, self._phases[0])
            + _breathing(self._config, t, self._phases[1])
            + self._config.drift_dps
            + _recoil(self._config, frame_index)
        )
        pitch: float = (
            _tremor(self._config, t, self._phases[2])
            + _breathing(self._config, t, self._phases[3])
            + _recoil(self._config, frame_index) * 0.6
        )
        return ImuSample(t, yaw, pitch, 0.0)

    def stream(self, frame_count: int) -> list[ImuSample]:
        """Angular rate samples for a whole run."""
        if frame_count < 0:
            raise ValueError(f"frame_count must be non-negative, got {frame_count}")
        return [self.sample(i) for i in range(frame_count)]

    def shake(self, frame: np.ndarray, frame_index: int, focal_length_px: float) -> np.ndarray:
        """Warp a frame by the motion at `frame_index`, simulating camera shake."""
        if not self._config.enabled:
            return frame
        imu: ImuSample = self.sample(frame_index)
        dx: float = focal_length_px * math.tan(math.radians(imu.yaw_rate_dps * self._dt))
        dy: float = focal_length_px * math.tan(math.radians(imu.pitch_rate_dps * self._dt))
        return translate(frame, dx, dy)
