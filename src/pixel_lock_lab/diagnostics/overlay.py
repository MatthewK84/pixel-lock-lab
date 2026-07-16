"""Annotated overlay rendering.

Draws the measured box, the predicted box, status, and a rolling score plot
with the lock and re-acquire thresholds marked. The plot is drawn with
OpenCV primitives rather than matplotlib to keep the field install small
and the per-frame cost bounded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import cv2
import numpy as np

from pixel_lock_lab.config.schemas import DiagnosticsConfig, TrackerConfig, TrackStatus
from pixel_lock_lab.errors import DiagnosticsError
from pixel_lock_lab.imageutil import to_bgr

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import TracebackType

    from pixel_lock_lab.array_types import Array
    from pixel_lock_lab.geometry import BoundingBox
    from pixel_lock_lab.trackers.base import TrackState

STATUS_COLORS: Final[dict[TrackStatus, tuple[int, int, int]]] = {
    TrackStatus.LOCKED: (0, 220, 0),
    TrackStatus.COASTING: (0, 200, 255),
    TrackStatus.LOST: (0, 0, 255),
    TrackStatus.UNINITIALIZED: (150, 150, 150),
}
PREDICTED_COLOR: Final[tuple[int, int, int]] = (255, 180, 0)
PLOT_BG: Final[tuple[int, int, int]] = (25, 25, 25)
FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: Final[float] = 0.5


def draw_box(
    frame: Array, box: BoundingBox, color: tuple[int, int, int], thickness: int = 2
) -> Array:
    """Draw a rectangle for `box` on a copy of `frame`."""
    out: Array = frame.copy()
    x, y, w, h = box.as_int_tuple()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_status(frame: Array, state: TrackState) -> Array:
    """Write frame index, status, score, and latency into the top-left corner."""
    out: Array = frame.copy()
    color: tuple[int, int, int] = STATUS_COLORS[state.status]
    lines: list[str] = [
        f"frame {state.frame_index}  {state.status.value.upper()}",
        f"score {state.score:.3f}  coast {state.coast_frames}",
        f"latency {state.latency_ms:.1f} ms",
    ]
    for index, text in enumerate(lines):
        cv2.putText(out, text, (8, 20 + index * 18), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)
    return out


def annotate(frame: Array, state: TrackState) -> Array:
    """Draw predicted box, measured box, and the status block."""
    out: Array = to_bgr(frame)
    if state.predicted_bbox is not None:
        out = draw_box(out, state.predicted_bbox, PREDICTED_COLOR, thickness=1)
    if state.bbox is not None:
        out = draw_box(out, state.bbox, STATUS_COLORS[state.status], thickness=2)
    return draw_status(out, state)


def _plot_threshold(canvas: Array, value: float, color: tuple[int, int, int]) -> None:
    height: int = canvas.shape[0]
    y: int = int(height - value * height)
    cv2.line(canvas, (0, y), (canvas.shape[1], y), color, 1, cv2.LINE_AA)


def render_score_plot(
    scores: Sequence[float], width: int, height: int, tracker: TrackerConfig
) -> Array:
    """Render a rolling score trace with threshold lines."""
    canvas: Array = np.full((height, width, 3), PLOT_BG, dtype=np.uint8)
    _plot_threshold(canvas, tracker.lock_threshold, (0, 140, 255))
    _plot_threshold(canvas, tracker.reacquire_threshold, (0, 220, 0))
    if len(scores) < 2:
        return canvas
    window: Sequence[float] = scores[-width:]
    step: float = width / float(max(len(window) - 1, 1))
    points: Array = np.array(
        [
            [int(i * step), int(height - np.clip(s, 0.0, 1.0) * height)]
            for i, s in enumerate(window)
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [points], isClosed=False, color=(255, 255, 255), thickness=1)
    cv2.putText(canvas, f"score {window[-1]:.2f}", (6, 14), FONT, 0.4, (200, 200, 200), 1)
    return canvas


def compose(frame: Array, plot: Array) -> Array:
    """Stack an annotated frame above its score plot."""
    if frame.shape[1] != plot.shape[1]:
        plot = cv2.resize(plot, (frame.shape[1], plot.shape[0]))
    return np.vstack([frame, plot])


class OverlayWriter:
    """Writes an annotated video with a score plot strip underneath."""

    def __init__(self, path: Path, config: DiagnosticsConfig, tracker: TrackerConfig) -> None:
        self._path: Path = path
        self._config: DiagnosticsConfig = config
        self._tracker: TrackerConfig = tracker
        self._writer: cv2.VideoWriter | None = None
        self._scores: list[float] = []

    def write(self, frame: Array, state: TrackState) -> Array:
        """Annotate one frame, append it to the video, and return the composite."""
        self._scores.append(state.score)
        annotated: Array = annotate(frame, state)
        plot: Array = render_score_plot(
            self._scores, annotated.shape[1], self._config.plot_height_px, self._tracker
        )
        composite: Array = compose(annotated, plot)
        self._ensure_writer(composite.shape[1], composite.shape[0])
        if self._writer is not None:
            self._writer.write(composite)
        return composite

    def _ensure_writer(self, width: int, height: int) -> None:
        if self._writer is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc: int = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(self._path), fourcc, self._config.overlay_fps, (width, height)
        )
        if not writer.isOpened():
            raise DiagnosticsError(f"cannot open video writer for {self._path}")
        self._writer = writer

    def close(self) -> None:
        """Finalize the video file."""
        if self._writer is None:
            return
        self._writer.release()
        self._writer = None

    def __enter__(self) -> OverlayWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
