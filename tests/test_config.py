"""Tests for configuration validation and round-tripping."""

from __future__ import annotations

from pathlib import Path

import pytest

from pixel_lock_lab.config.schemas import (
    DenoiseMethod,
    LabConfig,
    PreprocessConfig,
    SourceConfig,
    TrackerConfig,
    load_config,
    save_config,
)
from pixel_lock_lab.errors import ConfigError


def test_defaults_are_valid() -> None:
    config = LabConfig()
    assert config.tracker.lock_threshold <= config.tracker.reacquire_threshold


def test_hysteresis_is_enforced() -> None:
    with pytest.raises(ValueError, match="hysteresis"):
        TrackerConfig(lock_threshold=0.9, reacquire_threshold=0.2)


def test_threshold_bounds_enforced() -> None:
    with pytest.raises(ValueError, match=r"less than or equal to 1"):
        TrackerConfig(lock_threshold=1.5)


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValueError, match=r"extra_forbidden|Extra inputs"):
        TrackerConfig(bogus_field=3)  # type: ignore[call-arg]


def test_even_denoise_kernel_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        PreprocessConfig(denoise=DenoiseMethod.MEDIAN, denoise_strength=4)


def test_odd_denoise_kernel_accepted() -> None:
    config = PreprocessConfig(denoise=DenoiseMethod.MEDIAN, denoise_strength=5)
    assert config.denoise_strength == 5


def test_bilateral_allows_even_strength() -> None:
    config = PreprocessConfig(denoise=DenoiseMethod.BILATERAL, denoise_strength=8)
    assert config.denoise_strength == 8


def test_source_requires_exactly_one_input() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SourceConfig(synthetic_frames=10, video_path=Path("a.mp4"))


def test_source_requires_at_least_one_input() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SourceConfig()


def test_json_roundtrip(tmp_path: Path) -> None:
    original = LabConfig(name="roundtrip")
    original.tracker.lock_threshold = 0.4
    path = tmp_path / "cfg.json"
    save_config(original, path)
    loaded = load_config(path)
    assert loaded.name == "roundtrip"
    assert loaded.tracker.lock_threshold == pytest.approx(0.4)


def test_load_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "nope.json")


def test_load_invalid_json_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(path)


def test_load_invalid_values_raises_config_error(tmp_path: Path) -> None:
    path = tmp_path / "bad_values.json"
    path.write_text('{"tracker": {"lock_threshold": 5.0}}', encoding="utf-8")
    with pytest.raises(ConfigError, match="validation failed"):
        load_config(path)


def test_non_object_root_rejected(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be an object"):
        load_config(path)
