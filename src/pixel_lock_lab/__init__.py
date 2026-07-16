"""pixel_lock_lab: a diagnostic and reference tracker toolkit for EO/IR lock analysis.

This is a bench instrument, not a control loop. It exists to answer one
question well: when a tracker let go of a target, why did it let go?
"""

from __future__ import annotations

from typing import Final

from pixel_lock_lab.config.schemas import LabConfig, TrackerBackend, TrackerConfig, TrackStatus
from pixel_lock_lab.errors import (
    BackendUnavailableError,
    ConfigError,
    DiagnosticsError,
    FrameSourceError,
    PixelLockLabError,
    TrackerError,
)
from pixel_lock_lab.geometry import BoundingBox, iou
from pixel_lock_lab.session import Session, SessionResult
from pixel_lock_lab.trackers import BaseTracker, TrackState, build_tracker

__version__: Final[str] = "0.1.0"

__all__ = [
    "BackendUnavailableError",
    "BaseTracker",
    "BoundingBox",
    "ConfigError",
    "DiagnosticsError",
    "FrameSourceError",
    "LabConfig",
    "PixelLockLabError",
    "Session",
    "SessionResult",
    "TrackState",
    "TrackStatus",
    "TrackerBackend",
    "TrackerConfig",
    "TrackerError",
    "__version__",
    "build_tracker",
    "iou",
]
