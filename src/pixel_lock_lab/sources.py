"""Frame sources.

The synthetic source generates a textured background with a moving target
and exposes ground truth, so the whole lab is testable and demoable with no
input media.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from pixel_lock_lab.errors import FrameSourceError
from pixel_lock_lab.geometry import BoundingBox

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pixel_lock_lab.array_types import Array
    from pixel_lock_lab.config.schemas import SourceConfig

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
)
TARGET_SIZE_PX: Final[int] = 28
BACKGROUND_SEED: Final[int] = 4242


class FrameSource(ABC):
    """Yields frames and optionally knows ground truth."""

    @abstractmethod
    def frames(self) -> Iterator[Array]:
        """Yield frames in order."""

    def ground_truth(self, _frame_index: int) -> BoundingBox | None:
        """Ground-truth target box for a frame, when the source knows it."""
        return None


def _make_background(width: int, height: int) -> Array:
    """Textured static background that gives optical flow something to hold."""
    rng: np.random.Generator = np.random.default_rng(BACKGROUND_SEED)
    noise: Array = rng.integers(60, 120, size=(height, width), dtype=np.uint8)
    blurred: Array = cv2.GaussianBlur(noise, (0, 0), sigmaX=3.0)
    canvas: Array = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    for i in range(0, width, 64):
        cv2.line(canvas, (i, 0), (i, height), (90, 95, 100), 1)
    for j in range(0, height, 64):
        cv2.line(canvas, (0, j), (width, j), (90, 95, 100), 1)
    return canvas


class SyntheticSource(FrameSource):
    """A bright target on a sinusoidal path across a textured background."""

    def __init__(self, config: SourceConfig) -> None:
        self._config: SourceConfig = config
        self._background: Array = _make_background(config.frame_width, config.frame_height)

    def ground_truth(self, frame_index: int) -> BoundingBox | None:
        """Exact target box for `frame_index`."""
        width: int = self._config.frame_width
        height: int = self._config.frame_height
        total: int = max(self._config.synthetic_frames, 1)
        progress: float = frame_index / float(total)
        cx: float = 0.1 * width + 0.8 * width * progress
        cy: float = height / 2.0 + 0.18 * height * math.sin(progress * 4.0 * math.pi)
        half: float = TARGET_SIZE_PX / 2.0
        return BoundingBox(cx - half, cy - half, float(TARGET_SIZE_PX), float(TARGET_SIZE_PX))

    def _render(self, frame_index: int) -> Array:
        frame: Array = self._background.copy()
        box: BoundingBox | None = self.ground_truth(frame_index)
        if box is None:
            return frame
        cx, cy = box.center
        cv2.circle(frame, (int(cx), int(cy)), TARGET_SIZE_PX // 2, (235, 240, 245), thickness=-1)
        cv2.circle(frame, (int(cx), int(cy)), TARGET_SIZE_PX // 4, (40, 40, 60), thickness=-1)
        return frame

    def frames(self) -> Iterator[Array]:
        """Yield the configured number of synthetic frames."""
        for index in range(self._config.synthetic_frames):
            yield self._render(index)


class VideoSource(FrameSource):
    """Reads frames from a video file via OpenCV."""

    def __init__(self, config: SourceConfig) -> None:
        if config.video_path is None:
            raise FrameSourceError("VideoSource requires source.video_path")
        if not config.video_path.exists():
            raise FrameSourceError(f"video not found: {config.video_path}")
        self._config: SourceConfig = config
        self._path: Path = config.video_path

    def frames(self) -> Iterator[Array]:
        """Yield frames, honoring start_frame and max_frames."""
        capture: cv2.VideoCapture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            raise FrameSourceError(f"cannot open video: {self._path}")
        try:
            yield from self._read_all(capture)
        finally:
            capture.release()

    def _read_all(self, capture: cv2.VideoCapture) -> Iterator[Array]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(self._config.start_frame))
        emitted: int = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
            emitted += 1
            if self._config.max_frames is not None and emitted >= self._config.max_frames:
                return


class ImageDirSource(FrameSource):
    """Reads a lexically sorted directory of images."""

    def __init__(self, config: SourceConfig) -> None:
        if config.image_dir is None:
            raise FrameSourceError("ImageDirSource requires source.image_dir")
        if not config.image_dir.is_dir():
            raise FrameSourceError(f"not a directory: {config.image_dir}")
        self._config: SourceConfig = config
        self._paths: list[Path] = self._collect(config.image_dir)
        if not self._paths:
            raise FrameSourceError(f"no images found in {config.image_dir}")

    @staticmethod
    def _collect(directory: Path) -> list[Path]:
        return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

    def frames(self) -> Iterator[Array]:
        """Yield decoded images, honoring start_frame and max_frames."""
        end: int | None = (
            None
            if self._config.max_frames is None
            else self._config.start_frame + self._config.max_frames
        )
        for path in self._paths[self._config.start_frame : end]:
            frame: Array | None = cv2.imread(str(path))
            if frame is None:
                raise FrameSourceError(f"cannot decode image: {path}")
            yield frame


def build_source(config: SourceConfig) -> FrameSource:
    """Construct the frame source described by `config`."""
    if config.video_path is not None:
        return VideoSource(config)
    if config.image_dir is not None:
        return ImageDirSource(config)
    return SyntheticSource(config)
