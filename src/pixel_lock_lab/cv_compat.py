"""Typed adapters over OpenCV calls whose published stubs do not match the real API.

Every place where the opencv-python type stubs disagree with the documented
runtime behavior lives here, so the rest of the package stays strictly typed
with no inline suppressions.

The one case today is `calcOpticalFlowPyrLK`: the documented API accepts
`nextPts=None` to let OpenCV allocate the output, but the stubs declare that
parameter as a required array.
"""

from __future__ import annotations

from typing import Any, Final

import cv2
import numpy as np

EMPTY_POINTS: Final[np.ndarray] = np.empty((0, 1, 2), dtype=np.float32)
EMPTY_STATUS: Final[np.ndarray] = np.zeros((0, 1), dtype=np.uint8)


def empty_points() -> np.ndarray:
    """A fresh empty point array, so callers never share the module constant."""
    return np.empty((0, 1, 2), dtype=np.float32)


def _empty_status() -> np.ndarray:
    """A fresh empty status array."""
    return np.zeros((0, 1), dtype=np.uint8)


def calc_optical_flow(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    points: np.ndarray,
    win_size: tuple[int, int] | None = None,
    max_level: int | None = None,
    criteria: tuple[int, int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run pyramidal Lucas-Kanade, returning (tracked_points, status_flags).

    Returns empty arrays rather than None when OpenCV finds nothing to track.
    """
    if points.size == 0:
        return empty_points(), _empty_status()
    kwargs: dict[str, Any] = {}
    if win_size is not None:
        kwargs["winSize"] = win_size
    if max_level is not None:
        kwargs["maxLevel"] = max_level
    if criteria is not None:
        kwargs["criteria"] = criteria
    flow: Any = cv2.calcOpticalFlowPyrLK
    tracked, status, _error = flow(prev_gray, next_gray, points, None, **kwargs)
    if tracked is None or status is None:
        return empty_points(), _empty_status()
    moved: np.ndarray = np.asarray(tracked, dtype=np.float32)
    flags: np.ndarray = np.asarray(status, dtype=np.uint8)
    return moved, flags


def good_features(
    gray: np.ndarray,
    max_corners: int,
    quality_level: float,
    min_distance: float,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Find trackable corners, returning an empty array when none are found."""
    points: np.ndarray | None = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
    )
    if points is None:
        return empty_points()
    found: np.ndarray = np.asarray(points, dtype=np.float32)
    return found
