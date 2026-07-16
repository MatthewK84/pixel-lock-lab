"""Detection-driven tracking with a pluggable detector.

The lab core stays pure NumPy/OpenCV. Any detector -- Ultralytics, a
TensorRT engine on a Jetson, or a vendor SDK -- can be adapted to the
`Detector` protocol and swapped in without touching the lock policy.
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pixel_lock_lab.array_types import Array
    from pixel_lock_lab.config.schemas import TrackerConfig
from pixel_lock_lab.errors import BackendUnavailableError, TrackerError
from pixel_lock_lab.geometry import BoundingBox, iou
from pixel_lock_lab.trackers.base import BaseTracker, Measurement

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detector output."""

    bbox: BoundingBox
    confidence: float
    class_id: int = 0


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a frame into detections."""

    def detect(self, frame: Array) -> list[Detection]:
        """Return detections for a single frame."""
        ...


def associate(
    candidates: list[Detection], reference: BoundingBox, min_iou: float
) -> Detection | None:
    """Pick the detection best overlapping `reference` above `min_iou`."""
    best: Detection | None = None
    best_iou: float = min_iou
    for candidate in candidates:
        overlap: float = iou(candidate.bbox, reference)
        if overlap >= best_iou:
            best = candidate
            best_iou = overlap
    return best


class DetectionTracker(BaseTracker):
    """Tracks by associating detector output with the predicted box each frame."""

    def __init__(self, config: TrackerConfig, detector: Detector) -> None:
        super().__init__(config)
        if not isinstance(detector, Detector):
            raise TrackerError("detector must implement detect(frame) -> list[Detection]")
        self._detector: Detector = detector

    def _initialize(self, frame: Array, bbox: BoundingBox) -> None:
        """No warm-up needed; detection is stateless per frame."""

    def _measure(self, frame: Array, search_box: BoundingBox) -> Measurement:
        try:
            detections: list[Detection] = self._detector.detect(frame)
        except (RuntimeError, ValueError) as exc:
            logger.warning("detector failed on frame %d: %s", self._frame_index, exc)
            return Measurement(None, 0.0)
        reference: BoundingBox = self._bbox if self._bbox is not None else search_box
        matched: Detection | None = associate(detections, reference, self._config.association_iou)
        if matched is None:
            return Measurement(None, 0.0)
        return Measurement(matched.bbox, float(np.clip(matched.confidence, 0.0, 1.0)))


class UltralyticsDetector:
    """Thin adapter for an Ultralytics YOLO model. Requires the optional extra."""

    def __init__(
        self, weights: str, confidence: float = 0.25, class_id: int | None = None
    ) -> None:
        self._model: Any = self._load(weights)
        self._confidence: float = confidence
        self._class_id: int | None = class_id

    @staticmethod
    def _load(weights: str) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailableError(
                "Ultralytics not installed: pip install 'pixel-lock-lab[deep]'"
            ) from exc
        return YOLO(weights)

    def detect(self, frame: Array) -> list[Detection]:
        """Run the model and adapt its output to `Detection` objects."""
        results: Any = self._model.predict(frame, conf=self._confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            detections.extend(self._adapt(result))
        return detections

    def _adapt(self, result: object) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        adapted: list[Detection] = []
        for row in boxes:
            class_id = int(row.cls.item())
            if self._class_id is not None and class_id != self._class_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in row.xyxy[0].tolist())
            adapted.append(
                Detection(BoundingBox(x1, y1, x2 - x1, y2 - y1), float(row.conf.item()), class_id)
            )
        return adapted


def is_deep_available() -> bool:
    """True when the optional deep-learning extra is installed."""
    return importlib.util.find_spec("ultralytics") is not None
