"""Clutter and occlusion injection.

All injectors are seeded and deterministic: the same config and frame index
always produce the same corruption, so a drop event can be replayed exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

if TYPE_CHECKING:
    from pixel_lock_lab.array_types import Array
    from pixel_lock_lab.config.schemas import ClutterConfig
    from pixel_lock_lab.geometry import BoundingBox

GLARE_MAX: Final[int] = 255
BLOB_GRAY: Final[int] = 200


def _rng_for_frame(seed: int, frame_index: int) -> np.random.Generator:
    """Independent, reproducible generator per frame."""
    return np.random.default_rng(seed + frame_index * 7919)


def add_blobs(frame: Array, count: int, radius: int, rng: np.random.Generator) -> Array:
    """Draw `count` distractor blobs at random positions."""
    if count <= 0:
        return frame
    out: Array = frame.copy()
    height, width = out.shape[:2]
    xs: Array = rng.integers(0, width, size=count)
    ys: Array = rng.integers(0, height, size=count)
    shades: Array = rng.integers(BLOB_GRAY - 60, BLOB_GRAY + 55, size=count)
    for i in range(count):
        color: tuple[int, int, int] = (int(shades[i]), int(shades[i]), int(shades[i]))
        cv2.circle(out, (int(xs[i]), int(ys[i])), radius, color, thickness=-1)
    return out


def add_noise(frame: Array, sigma: float, rng: np.random.Generator) -> Array:
    """Add zero-mean Gaussian sensor noise."""
    if sigma <= 0.0:
        return frame
    noise: Array = rng.normal(0.0, sigma, size=frame.shape)
    noisy: Array = frame.astype(np.float32) + noise
    return np.asarray(np.clip(noisy, 0, 255).astype(np.uint8))


def add_glare(frame: Array, intensity: float, rng: np.random.Generator) -> Array:
    """Blend in a soft radial glare patch."""
    if intensity <= 0.0:
        return frame
    height, width = frame.shape[:2]
    mask: Array = np.zeros((height, width), dtype=np.uint8)
    center: tuple[int, int] = (int(rng.integers(0, width)), int(rng.integers(0, height)))
    radius: int = int(min(height, width) * 0.25)
    cv2.circle(mask, center, radius, GLARE_MAX, thickness=-1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=radius / 2.0)
    scaled: Array = mask.astype(np.float32) * intensity
    layer: Array = scaled if frame.ndim == 2 else cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    return np.asarray(np.clip(frame.astype(np.float32) + layer, 0, 255).astype(np.uint8))


def occlude(frame: Array, target: BoundingBox, coverage: float) -> Array:
    """Cover `coverage` of the target box with an opaque rectangle."""
    if coverage <= 0.0:
        return frame
    out: Array = frame.copy()
    covered_h: int = max(1, int(target.height * min(coverage, 1.0)))
    x, y, w, _ = target.as_int_tuple()
    y_end: int = min(out.shape[0], y + covered_h)
    x_end: int = min(out.shape[1], x + w)
    if y_end <= max(y, 0) or x_end <= max(x, 0):
        return out
    out[max(y, 0) : y_end, max(x, 0) : x_end] = 40
    return out


class ClutterInjector:
    """Applies the configured corruption chain to each frame."""

    def __init__(self, config: ClutterConfig) -> None:
        self._config: ClutterConfig = config

    def is_occluding(self, frame_index: int) -> bool:
        """True when `frame_index` falls inside the configured occlusion window."""
        start: int | None = self._config.occlusion_start_frame
        if start is None or self._config.occlusion_frames <= 0:
            return False
        return start <= frame_index < start + self._config.occlusion_frames

    def apply(self, frame: Array, frame_index: int, target: BoundingBox | None = None) -> Array:
        """Inject clutter, noise, glare, and occlusion for one frame."""
        if not self._config.enabled:
            return frame
        rng: np.random.Generator = _rng_for_frame(self._config.seed, frame_index)
        out: Array = add_blobs(frame, self._config.blob_count, self._config.blob_radius_px, rng)
        out = add_noise(out, self._config.noise_sigma, rng)
        out = add_glare(out, self._config.glare_intensity, rng)
        if target is not None and self.is_occluding(frame_index):
            out = occlude(out, target, self._config.occlusion_coverage)
        return out
