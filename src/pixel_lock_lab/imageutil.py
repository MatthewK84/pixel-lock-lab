"""Small, pure image helpers shared by trackers and the pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from pixel_lock_lab.geometry import BoundingBox, clip_to_frame

if TYPE_CHECKING:
    from pixel_lock_lab.array_types import Array


def to_gray(frame: Array) -> Array:
    """Return a single-channel uint8 view of `frame` as a new array."""
    if frame.ndim == 2:
        return np.asarray(frame.astype(np.uint8, copy=True))
    if frame.shape[2] == 3:
        return np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if frame.shape[2] == 4:
        return np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY))
    raise ValueError(f"unsupported channel count: {frame.shape[2]}")


def to_bgr(frame: Array) -> Array:
    """Return a 3-channel BGR copy of `frame`."""
    if frame.ndim == 2:
        return np.asarray(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    if frame.shape[2] == 4:
        return np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
    return np.asarray(frame.copy())


def crop(frame: Array, box: BoundingBox) -> Array | None:
    """Crop `box` from `frame`. Returns None if the box falls outside the frame."""
    clipped: BoundingBox | None = clip_to_frame(box, frame.shape[1], frame.shape[0])
    if clipped is None:
        return None
    x, y, w, h = clipped.as_int_tuple()
    patch: Array = frame[y : y + h, x : x + w]
    if patch.size == 0:
        return None
    return np.asarray(patch.copy())
