"""Tests for the image pipeline: preprocess, stabilize, latency."""

from __future__ import annotations

import numpy as np
import pytest

from pixel_lock_lab.config.schemas import (
    DenoiseMethod,
    PreprocessConfig,
    RegionOfInterest,
    StabilizeConfig,
    StabilizeMode,
)
from pixel_lock_lab.pipeline.latency import LatencyRecorder
from pixel_lock_lab.pipeline.preprocess import (
    Preprocessor,
    apply_exposure,
    apply_roi,
    apply_sharpen,
)
from pixel_lock_lab.pipeline.stabilize import ImuSample, Stabilizer, rates_to_shift, translate


@pytest.fixture
def gray_frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(80, 160, size=(120, 160, 3), dtype=np.uint8)


def test_apply_roi_crops(gray_frame: np.ndarray) -> None:
    roi = RegionOfInterest(x=10, y=20, width=50, height=40)
    cropped = apply_roi(gray_frame, roi)
    assert cropped.shape[:2] == (40, 50)


def test_apply_roi_none_is_passthrough(gray_frame: np.ndarray) -> None:
    assert apply_roi(gray_frame, None) is gray_frame


def test_apply_roi_out_of_bounds_returns_original(gray_frame: np.ndarray) -> None:
    roi = RegionOfInterest(x=900, y=900, width=10, height=10)
    assert apply_roi(gray_frame, roi).shape == gray_frame.shape


def test_exposure_gain_brightens(gray_frame: np.ndarray) -> None:
    brighter = apply_exposure(gray_frame, 1.6)
    assert brighter.mean() > gray_frame.mean()


def test_exposure_unity_is_passthrough(gray_frame: np.ndarray) -> None:
    assert apply_exposure(gray_frame, 1.0) is gray_frame


def test_exposure_saturates_not_wraps() -> None:
    frame = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert int(apply_exposure(frame, 4.0).max()) == 255


def test_sharpen_zero_is_passthrough(gray_frame: np.ndarray) -> None:
    assert apply_sharpen(gray_frame, 0.0) is gray_frame


def test_preprocessor_does_not_mutate_input(gray_frame: np.ndarray) -> None:
    original = gray_frame.copy()
    config = PreprocessConfig(
        exposure_gain=1.5, denoise=DenoiseMethod.GAUSSIAN, denoise_strength=3
    )
    Preprocessor(config).apply(gray_frame)
    assert np.array_equal(gray_frame, original)


def test_preprocessor_disabled_is_passthrough(gray_frame: np.ndarray) -> None:
    processed = Preprocessor(PreprocessConfig(enabled=False)).apply(gray_frame)
    assert processed is gray_frame


def test_preprocessor_clahe_runs(gray_frame: np.ndarray) -> None:
    config = PreprocessConfig(clahe_enabled=True, clahe_clip_limit=3.0, clahe_grid=4)
    assert Preprocessor(config).apply(gray_frame).shape == gray_frame.shape


def test_roi_offset_reported() -> None:
    config = PreprocessConfig(roi=RegionOfInterest(x=15, y=25, width=10, height=10))
    assert Preprocessor(config).roi_offset() == (15, 25)


def test_roi_offset_zero_without_roi() -> None:
    assert Preprocessor(PreprocessConfig()).roi_offset() == (0, 0)


def test_rates_to_shift_zero_rate_is_zero() -> None:
    sample = ImuSample(0.0, 0.0, 0.0)
    assert rates_to_shift(sample, 900.0, 1.0 / 30.0) == (0.0, 0.0)


def test_rates_to_shift_sign_and_scale() -> None:
    sample = ImuSample(0.0, yaw_rate_dps=10.0, pitch_rate_dps=0.0)
    dx, dy = rates_to_shift(sample, focal_length_px=900.0, dt=0.1)
    assert dx < 0.0
    assert dy == pytest.approx(0.0)


def test_translate_shifts_content() -> None:
    frame = np.zeros((50, 50), dtype=np.uint8)
    frame[25, 25] = 255
    shifted = translate(frame, 5.0, 0.0)
    assert int(shifted[25, 30]) > int(shifted[25, 25])


def test_stabilizer_none_mode_is_passthrough(gray_frame: np.ndarray) -> None:
    result = Stabilizer(StabilizeConfig(mode=StabilizeMode.NONE)).apply(gray_frame)
    assert result.frame is gray_frame
    assert result.applied_shift_px == (0.0, 0.0)


def test_stabilizer_imu_without_sample_is_passthrough(gray_frame: np.ndarray) -> None:
    result = Stabilizer(StabilizeConfig(mode=StabilizeMode.IMU)).apply(gray_frame, imu=None)
    assert result.applied_shift_px == (0.0, 0.0)


def test_stabilizer_imu_counteracts_motion(gray_frame: np.ndarray) -> None:
    config = StabilizeConfig(mode=StabilizeMode.IMU, focal_length_px=900.0)
    sample = ImuSample(0.0, yaw_rate_dps=20.0, pitch_rate_dps=0.0)
    result = Stabilizer(config).apply(gray_frame, imu=sample, dt=0.05)
    assert result.applied_shift_px[0] > 0.0


def test_stabilizer_clamps_to_max_shift(gray_frame: np.ndarray) -> None:
    config = StabilizeConfig(mode=StabilizeMode.IMU, focal_length_px=900.0, max_shift_px=3.0)
    sample = ImuSample(0.0, yaw_rate_dps=80.0, pitch_rate_dps=80.0)
    result = Stabilizer(config).apply(gray_frame, imu=sample, dt=0.2)
    assert abs(result.applied_shift_px[0]) <= 3.0
    assert abs(result.applied_shift_px[1]) <= 3.0


def test_stabilizer_visual_first_frame_is_passthrough(gray_frame: np.ndarray) -> None:
    result = Stabilizer(StabilizeConfig(mode=StabilizeMode.VISUAL)).apply(gray_frame)
    assert result.applied_shift_px == (0.0, 0.0)


def test_latency_recorder_percentiles() -> None:
    recorder = LatencyRecorder()
    for value in range(1, 101):
        recorder.record("track", float(value))
    stats = recorder.stats("track")
    assert stats.count == 100
    assert stats.p50_ms == pytest.approx(50.5)
    assert stats.max_ms == 100.0


def test_latency_recorder_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LatencyRecorder().record("track", -1.0)


def test_latency_unknown_stage_is_empty() -> None:
    assert LatencyRecorder().stats("nothing").count == 0


def test_latency_measure_context_manager() -> None:
    recorder = LatencyRecorder()
    with recorder.measure("stage") as timer:
        sum(range(1000))
    assert timer.elapsed_ms >= 0.0
    assert recorder.stats("stage").count == 1
    assert recorder.stages() == ["stage"]
