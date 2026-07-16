"""Tracker backends and the factory that builds them from config."""

from __future__ import annotations

from pixel_lock_lab.config.schemas import TrackerBackend, TrackerConfig
from pixel_lock_lab.errors import TrackerError
from pixel_lock_lab.trackers.base import BaseTracker, Measurement, TrackState
from pixel_lock_lab.trackers.classical import OpenCVTracker, TemplateMatchTracker
from pixel_lock_lab.trackers.deep import Detection, DetectionTracker, Detector
from pixel_lock_lab.trackers.optical_flow import LucasKanadeTracker

__all__ = [
    "BaseTracker",
    "Detection",
    "DetectionTracker",
    "Detector",
    "LucasKanadeTracker",
    "Measurement",
    "OpenCVTracker",
    "TemplateMatchTracker",
    "TrackState",
    "build_tracker",
]


def build_tracker(config: TrackerConfig, detector: Detector | None = None) -> BaseTracker:
    """Construct the tracker named by `config.backend`."""
    if config.backend is TrackerBackend.TEMPLATE:
        return TemplateMatchTracker(config)
    if config.backend is TrackerBackend.OPTICAL_FLOW:
        return LucasKanadeTracker(config)
    if config.backend is TrackerBackend.DETECTION:
        if detector is None:
            raise TrackerError("backend 'detection' requires a detector instance")
        return DetectionTracker(config, detector)
    return OpenCVTracker(config)
