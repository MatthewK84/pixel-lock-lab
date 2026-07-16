"""Tests for clutter injection and the motion model."""

from __future__ import annotations

import numpy as np
import pytest

from pixel_lock_lab.config.schemas import ClutterConfig, MotionConfig
from pixel_lock_lab.geometry import BoundingBox
from pixel_lock_lab.simulation.clutter import ClutterInjector, add_noise, occlude
from pixel_lock_lab.simulation.motion import MotionGenerator


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((120, 160, 3), 128, dtype=np.uint8)


def test_disabled_injector_is_passthrough(frame: np.ndarray) -> None:
    injector = ClutterInjector(ClutterConfig(enabled=False, blob_count=50))
    assert injector.apply(frame, 0) is frame


def test_injection_is_deterministic(frame: np.ndarray) -> None:
    config = ClutterConfig(enabled=True, seed=99, blob_count=10, noise_sigma=5.0)
    first = ClutterInjector(config).apply(frame, 3)
    second = ClutterInjector(config).apply(frame, 3)
    assert np.array_equal(first, second)


def test_different_frames_differ(frame: np.ndarray) -> None:
    config = ClutterConfig(enabled=True, seed=99, blob_count=10)
    assert not np.array_equal(
        ClutterInjector(config).apply(frame, 1), ClutterInjector(config).apply(frame, 2)
    )


def test_injection_does_not_mutate_input(frame: np.ndarray) -> None:
    original = frame.copy()
    config = ClutterConfig(enabled=True, blob_count=5, noise_sigma=3.0, glare_intensity=0.5)
    ClutterInjector(config).apply(frame, 0)
    assert np.array_equal(frame, original)


def test_noise_changes_pixels_but_stays_in_range(frame: np.ndarray) -> None:
    noisy = add_noise(frame, 10.0, np.random.default_rng(0))
    assert not np.array_equal(noisy, frame)
    assert noisy.min() >= 0
    assert noisy.max() <= 255


def test_noise_zero_sigma_is_passthrough(frame: np.ndarray) -> None:
    assert add_noise(frame, 0.0, np.random.default_rng(0)) is frame


def test_occlude_darkens_target(frame: np.ndarray) -> None:
    box = BoundingBox(40.0, 40.0, 20.0, 20.0)
    occluded = occlude(frame, box, coverage=1.0)
    assert occluded[45, 45].mean() < frame[45, 45].mean()


def test_occlusion_window_membership() -> None:
    config = ClutterConfig(enabled=True, occlusion_start_frame=10, occlusion_frames=5)
    injector = ClutterInjector(config)
    assert not injector.is_occluding(9)
    assert injector.is_occluding(10)
    assert injector.is_occluding(14)
    assert not injector.is_occluding(15)


def test_no_occlusion_window_when_unset() -> None:
    injector = ClutterInjector(ClutterConfig(enabled=True))
    assert not injector.is_occluding(0)


def test_motion_disabled_returns_zero_rates() -> None:
    sample = MotionGenerator(MotionConfig(enabled=False)).sample(10)
    assert sample.yaw_rate_dps == 0.0
    assert sample.pitch_rate_dps == 0.0


def test_motion_is_deterministic() -> None:
    config = MotionConfig(enabled=True, seed=5)
    first = MotionGenerator(config).sample(7)
    second = MotionGenerator(config).sample(7)
    assert first.yaw_rate_dps == pytest.approx(second.yaw_rate_dps)


def test_motion_respects_tremor_amplitude() -> None:
    config = MotionConfig(
        enabled=True, tremor_amplitude_dps=2.0, breathing_amplitude_dps=0.0, drift_dps=0.0
    )
    samples = MotionGenerator(config).stream(200)
    assert max(abs(s.yaw_rate_dps) for s in samples) <= 2.0 + 1e-6


def test_drift_biases_yaw() -> None:
    config = MotionConfig(
        enabled=True, tremor_amplitude_dps=0.0, breathing_amplitude_dps=0.0, drift_dps=4.0
    )
    assert MotionGenerator(config).sample(0).yaw_rate_dps == pytest.approx(4.0)


def test_recoil_transient_decays() -> None:
    config = MotionConfig(
        enabled=True,
        tremor_amplitude_dps=0.0,
        breathing_amplitude_dps=0.0,
        recoil_frame=10,
        recoil_amplitude_dps=30.0,
        recoil_decay_frames=4,
    )
    generator = MotionGenerator(config)
    assert abs(generator.sample(9).yaw_rate_dps) < 1e-9
    assert abs(generator.sample(10).yaw_rate_dps) == pytest.approx(30.0)
    assert abs(generator.sample(30).yaw_rate_dps) < 1.0


def test_stream_length() -> None:
    assert len(MotionGenerator(MotionConfig(enabled=True)).stream(25)) == 25


def test_stream_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MotionGenerator(MotionConfig()).stream(-1)


def test_generator_rejects_bad_fps() -> None:
    with pytest.raises(ValueError, match="fps must be positive"):
        MotionGenerator(MotionConfig(), fps=0.0)


def test_shake_moves_frame(frame: np.ndarray) -> None:
    marked = frame.copy()
    marked[60, 80] = 255
    config = MotionConfig(enabled=True, tremor_amplitude_dps=20.0, drift_dps=10.0)
    shaken = MotionGenerator(config).shake(marked, 3, focal_length_px=900.0)
    assert not np.array_equal(shaken, marked)
