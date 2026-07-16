"""Session orchestration.

Wires the full chain for one run:

    source -> motion shake -> clutter -> preprocess -> stabilize -> tracker -> log -> overlay

Each stage is timed separately so a latency regression can be pinned to the
stage that caused it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from pixel_lock_lab.config.schemas import LabConfig, StabilizeMode, TrackStatus
from pixel_lock_lab.diagnostics.drop_analyzer import DropEvent, find_drop_events
from pixel_lock_lab.diagnostics.log_schema import TrackLogRecord, TrackLogWriter, record_from_state
from pixel_lock_lab.diagnostics.overlay import OverlayWriter
from pixel_lock_lab.errors import ConfigError
from pixel_lock_lab.geometry import BoundingBox
from pixel_lock_lab.imageutil import crop
from pixel_lock_lab.pipeline.latency import LatencyRecorder, StageStats
from pixel_lock_lab.pipeline.preprocess import Preprocessor
from pixel_lock_lab.pipeline.stabilize import ImuSample, StabilizationResult, Stabilizer
from pixel_lock_lab.simulation.clutter import ClutterInjector
from pixel_lock_lab.simulation.motion import MotionGenerator
from pixel_lock_lab.sources import FrameSource, build_source
from pixel_lock_lab.trackers import BaseTracker, Detector, build_tracker
from pixel_lock_lab.trackers.base import TrackState

logger: logging.Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionResult:
    """Everything a run produced."""

    records: tuple[TrackLogRecord, ...]
    drop_events: tuple[DropEvent, ...]
    latency: tuple[StageStats, ...]
    frames_processed: int

    @property
    def locked_fraction(self) -> float:
        """Share of frames that held lock."""
        if not self.records:
            return 0.0
        locked: int = sum(1 for r in self.records if r.status is TrackStatus.LOCKED)
        return locked / len(self.records)


def _mean_intensity(frame: np.ndarray, box: BoundingBox | None) -> float:
    """Mean intensity inside the track box, used for low-contrast attribution."""
    if box is None:
        return 0.0
    patch: np.ndarray | None = crop(frame, box)
    if patch is None or patch.size == 0:
        return 0.0
    return float(np.clip(patch.mean(), 0.0, 255.0))


class Session:
    """Runs one tracking session end to end."""

    def __init__(self, config: LabConfig, detector: Detector | None = None) -> None:
        self._config: LabConfig = config
        self._source: FrameSource = build_source(config.source)
        self._tracker: BaseTracker = build_tracker(config.tracker, detector)
        self._preprocessor: Preprocessor = Preprocessor(config.preprocess)
        self._stabilizer: Stabilizer = Stabilizer(config.stabilize)
        self._clutter: ClutterInjector = ClutterInjector(config.clutter)
        self._motion: MotionGenerator = MotionGenerator(
            config.motion, fps=config.diagnostics.overlay_fps
        )
        self._latency: LatencyRecorder = LatencyRecorder()

    def _initial_box(self) -> BoundingBox:
        """Configured start box, or the source's ground truth for frame zero."""
        if self._config.initial_bbox is not None:
            x, y, w, h = self._config.initial_bbox
            return BoundingBox(float(x), float(y), float(w), float(h))
        truth: BoundingBox | None = self._source.ground_truth(0)
        if truth is None:
            raise ConfigError("initial_bbox is required for sources without ground truth")
        return truth

    def _corrupt(self, frame: np.ndarray, index: int) -> np.ndarray:
        """Apply the simulated motion and clutter for one frame."""
        with self._latency.measure("simulate"):
            shaken: np.ndarray = self._motion.shake(
                frame, index, self._config.stabilize.focal_length_px
            )
            truth: BoundingBox | None = self._source.ground_truth(index)
            return self._clutter.apply(shaken, index, truth)

    def _condition(self, frame: np.ndarray, index: int) -> tuple[np.ndarray, float]:
        """Preprocess and stabilize, returning the frame and residual motion."""
        with self._latency.measure("preprocess"):
            processed: np.ndarray = self._preprocessor.apply(frame)
        with self._latency.measure("stabilize"):
            imu: ImuSample | None = self._imu_for(index)
            result: StabilizationResult = self._stabilizer.apply(processed, imu, self._motion.dt)
        return result.frame, result.residual_px

    def _imu_for(self, index: int) -> ImuSample | None:
        if self._config.stabilize.mode is not StabilizeMode.IMU:
            return None
        return self._motion.sample(index)

    def _offset_box(self, box: BoundingBox) -> BoundingBox:
        """Shift a frame-space box into ROI-cropped coordinates."""
        dx, dy = self._preprocessor.roi_offset()
        return box.translated(-float(dx), -float(dy))

    def _track(self, frame: np.ndarray, index: int) -> TrackState:
        """Initialize on the first frame, otherwise update."""
        with self._latency.measure("track") as timer:
            if index == 0:
                state: TrackState = self._tracker.init(
                    frame, self._offset_box(self._initial_box())
                )
            else:
                state = self._tracker.update(frame)
        return self._retime(state, timer.elapsed_ms)

    @staticmethod
    def _retime(state: TrackState, latency_ms: float) -> TrackState:
        return TrackState(
            frame_index=state.frame_index,
            timestamp=state.timestamp,
            status=state.status,
            score=state.score,
            bbox=state.bbox,
            predicted_bbox=state.predicted_bbox,
            velocity=state.velocity,
            coast_frames=state.coast_frames,
            latency_ms=latency_ms,
        )

    def _log(
        self, state: TrackState, frame: np.ndarray, index: int, residual: float
    ) -> TrackLogRecord:
        return record_from_state(
            state,
            timestamp=index * self._motion.dt,
            residual_motion_px=residual,
            occluded=self._clutter.is_occluding(index),
            mean_intensity=_mean_intensity(frame, state.bbox),
        )

    def run(self, log_path: Path | None = None, overlay_path: Path | None = None) -> SessionResult:
        """Execute the session, optionally writing a track log and overlay video."""
        records: list[TrackLogRecord] = []
        writer: TrackLogWriter | None = TrackLogWriter(log_path) if log_path else None
        overlay: OverlayWriter | None = self._make_overlay(overlay_path)
        try:
            if writer is not None:
                writer.open()
            records = self._loop(writer, overlay)
        finally:
            if writer is not None:
                writer.close()
            if overlay is not None:
                overlay.close()
        return self._finalize(records)

    def _make_overlay(self, path: Path | None) -> OverlayWriter | None:
        if path is None or not self._config.diagnostics.overlay_enabled:
            return None
        return OverlayWriter(path, self._config.diagnostics, self._config.tracker)

    def _loop(
        self, writer: TrackLogWriter | None, overlay: OverlayWriter | None
    ) -> list[TrackLogRecord]:
        records: list[TrackLogRecord] = []
        for index, raw in enumerate(self._source.frames()):
            corrupted: np.ndarray = self._corrupt(raw, index)
            conditioned, residual = self._condition(corrupted, index)
            state: TrackState = self._track(conditioned, index)
            record: TrackLogRecord = self._log(state, conditioned, index, residual)
            records.append(record)
            if writer is not None:
                writer.write(record)
            if overlay is not None:
                overlay.write(conditioned, state)
        return records

    def _finalize(self, records: list[TrackLogRecord]) -> SessionResult:
        events: list[DropEvent] = find_drop_events(records, self._config.diagnostics)
        return SessionResult(
            records=tuple(records),
            drop_events=tuple(events),
            latency=tuple(self._latency.summary()),
            frames_processed=len(records),
        )
