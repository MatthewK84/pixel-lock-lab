"""Classical trackers: NCC template matching and OpenCV backends.

TemplateMatchTracker is the reference implementation. Its score is a true
normalized cross-correlation peak, which makes score collapse directly
interpretable when diagnosing a drop.

OpenCVTracker wraps CSRT / KCF / MOSSE. Those backends do not expose a
confidence value, so the score is an NCC of the current patch against the
maintained template. This keeps the lock policy comparable across backends.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import cv2
import numpy as np

from pixel_lock_lab.config.schemas import TrackerBackend, TrackerConfig
from pixel_lock_lab.errors import BackendUnavailableError, TrackerError
from pixel_lock_lab.geometry import BoundingBox
from pixel_lock_lab.imageutil import crop, to_gray
from pixel_lock_lab.trackers.base import BaseTracker, Measurement

logger: logging.Logger = logging.getLogger(__name__)

MIN_TEMPLATE_PX: Final[int] = 4
OPENCV_BACKENDS: Final[frozenset[TrackerBackend]] = frozenset(
    {TrackerBackend.CSRT, TrackerBackend.KCF, TrackerBackend.MOSSE}
)


def _ncc(patch: np.ndarray, template: np.ndarray) -> float:
    """Single normalized cross-correlation value, clamped to [0, 1]."""
    if patch.shape != template.shape:
        patch = cv2.resize(patch, (template.shape[1], template.shape[0]))
    result: np.ndarray = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
    return float(np.clip(result.max(), 0.0, 1.0))


class TemplateMatchTracker(BaseTracker):
    """Pure NCC template matcher with a configurable template update rate."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._template: np.ndarray | None = None

    def _initialize(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        patch: np.ndarray | None = crop(to_gray(frame), bbox)
        if patch is None or patch.shape[0] < MIN_TEMPLATE_PX or patch.shape[1] < MIN_TEMPLATE_PX:
            raise TrackerError(f"cannot build template from bbox {bbox}")
        self._template = patch.astype(np.float32)

    def _measure(self, frame: np.ndarray, search_box: BoundingBox) -> Measurement:
        if self._template is None:
            return Measurement(None, 0.0)
        region: np.ndarray | None = crop(to_gray(frame), search_box)
        if region is None:
            return Measurement(None, 0.0)
        template_u8: np.ndarray = self._template.astype(np.uint8)
        if region.shape[0] < template_u8.shape[0] or region.shape[1] < template_u8.shape[1]:
            return Measurement(None, 0.0)
        return self._locate(region, template_u8, search_box, frame)

    def _locate(
        self,
        region: np.ndarray,
        template: np.ndarray,
        search_box: BoundingBox,
        frame: np.ndarray,
    ) -> Measurement:
        result: np.ndarray = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        score: float = float(np.clip(max_val, 0.0, 1.0))
        found: BoundingBox = BoundingBox(
            search_box.x + float(max_loc[0]),
            search_box.y + float(max_loc[1]),
            float(template.shape[1]),
            float(template.shape[0]),
        )
        if score >= self._config.lock_threshold:
            self._blend_template(frame, found)
        return Measurement(found, score)

    def _blend_template(self, frame: np.ndarray, box: BoundingBox) -> None:
        """Exponentially blend the observed patch into the template."""
        rate: float = self._config.template_update_rate
        if rate <= 0.0 or self._template is None:
            return
        patch: np.ndarray | None = crop(to_gray(frame), box)
        if patch is None or patch.shape != self._template.shape:
            return
        self._template = (1.0 - rate) * self._template + rate * patch.astype(np.float32)


def _create_opencv_tracker(backend: TrackerBackend) -> Any:
    """Instantiate an OpenCV tracker, tolerating the legacy namespace split."""
    factories: dict[TrackerBackend, tuple[str, ...]] = {
        TrackerBackend.CSRT: ("TrackerCSRT_create", "legacy.TrackerCSRT_create"),
        TrackerBackend.KCF: ("TrackerKCF_create", "legacy.TrackerKCF_create"),
        TrackerBackend.MOSSE: ("legacy.TrackerMOSSE_create", "TrackerMOSSE_create"),
    }
    for name in factories[backend]:
        factory: Any = _resolve_attr(cv2, name)
        if factory is not None:
            return factory()
    raise BackendUnavailableError(
        f"OpenCV tracker '{backend.value}' unavailable; install opencv-contrib-python"
    )


def _resolve_attr(root: Any, dotted: str) -> Any:
    """Resolve a dotted attribute path, returning None if any part is missing."""
    node: Any = root
    for part in dotted.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


class OpenCVTracker(BaseTracker):
    """Wrapper around OpenCV CSRT / KCF / MOSSE with an NCC-derived score."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        if config.backend not in OPENCV_BACKENDS:
            raise TrackerError(f"{config.backend.value} is not an OpenCV backend")
        self._impl: Any = None
        self._template: np.ndarray | None = None

    def _initialize(self, frame: np.ndarray, bbox: BoundingBox) -> None:
        self._impl = _create_opencv_tracker(self._config.backend)
        try:
            self._impl.init(frame, bbox.as_int_tuple())
        except cv2.error as exc:
            raise TrackerError(f"OpenCV tracker init failed: {exc}") from exc
        patch: np.ndarray | None = crop(to_gray(frame), bbox)
        if patch is None:
            raise TrackerError(f"cannot build template from bbox {bbox}")
        self._template = patch.astype(np.float32)

    def _measure(self, frame: np.ndarray, _search_box: BoundingBox) -> Measurement:
        if self._impl is None:
            return Measurement(None, 0.0)
        try:
            ok, raw = self._impl.update(frame)
        except cv2.error as exc:
            logger.warning("OpenCV tracker update failed: %s", exc)
            return Measurement(None, 0.0)
        if not ok:
            return Measurement(None, 0.0)
        box: BoundingBox = BoundingBox(float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        return Measurement(box, self._score_box(frame, box))

    def _score_box(self, frame: np.ndarray, box: BoundingBox) -> float:
        if self._template is None:
            return 0.0
        patch: np.ndarray | None = crop(to_gray(frame), box)
        if patch is None or patch.size == 0:
            return 0.0
        score: float = _ncc(patch, self._template.astype(np.uint8))
        if score >= self._config.lock_threshold:
            self._blend_template(patch)
        return score

    def _blend_template(self, patch: np.ndarray) -> None:
        rate: float = self._config.template_update_rate
        if rate <= 0.0 or self._template is None:
            return
        if patch.shape != self._template.shape:
            patch = cv2.resize(patch, (self._template.shape[1], self._template.shape[0]))
        self._template = (1.0 - rate) * self._template + rate * patch.astype(np.float32)
