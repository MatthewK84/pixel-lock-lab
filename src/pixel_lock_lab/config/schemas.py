"""Pydantic v2 models for every tunable parameter in the lab.

Every knob that affects a track is a first-class, serializable field here.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pixel_lock_lab.errors import ConfigError

SCHEMA_VERSION: Final[str] = "1.0"


class TrackerBackend(str, Enum):
    """Selectable tracker implementations."""

    TEMPLATE = "template"
    CSRT = "csrt"
    KCF = "kcf"
    MOSSE = "mosse"
    OPTICAL_FLOW = "optical_flow"
    DETECTION = "detection"


class DenoiseMethod(str, Enum):
    """Denoising stage options."""

    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


class StabilizeMode(str, Enum):
    """Motion compensation strategy."""

    NONE = "none"
    IMU = "imu"
    VISUAL = "visual"


class TrackStatus(str, Enum):
    """Lifecycle state of a track."""

    UNINITIALIZED = "uninitialized"
    LOCKED = "locked"
    COASTING = "coasting"
    LOST = "lost"


class StrictModel(BaseModel):
    """Base model: reject unknown fields and validate on assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


class TrackerConfig(StrictModel):
    """Lock, re-acquire, search, template update and coasting parameters."""

    backend: TrackerBackend = TrackerBackend.TEMPLATE
    lock_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    reacquire_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    search_window_scale: float = Field(default=2.5, ge=1.0, le=10.0)
    lost_search_scale: float = Field(default=4.0, ge=1.0, le=20.0)
    lost_search_growth: float = Field(default=0.15, ge=0.0, le=2.0)
    max_lost_search_scale: float = Field(default=12.0, ge=1.0, le=40.0)
    template_update_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    max_coast_frames: int = Field(default=12, ge=0, le=600)
    coast_velocity_decay: float = Field(default=0.90, ge=0.0, le=1.0)
    velocity_smoothing: float = Field(default=0.60, ge=0.0, le=1.0)
    min_flow_points: int = Field(default=8, ge=1, le=500)
    flow_fb_error_px: float = Field(default=2.0, ge=0.1, le=50.0)
    association_iou: float = Field(default=0.30, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_hysteresis(self) -> TrackerConfig:
        if self.reacquire_threshold < self.lock_threshold:
            raise ValueError("reacquire_threshold must be >= lock_threshold (hysteresis)")
        return self

    @model_validator(mode="after")
    def _check_search_scales(self) -> TrackerConfig:
        if self.max_lost_search_scale < self.lost_search_scale:
            raise ValueError("max_lost_search_scale must be >= lost_search_scale")
        return self


class RegionOfInterest(StrictModel):
    """Pixel-space crop applied before tracking."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class PreprocessConfig(StrictModel):
    """ISP-like controlled image processing stages."""

    enabled: bool = True
    exposure_gain: float = Field(default=1.0, gt=0.0, le=8.0)
    contrast_alpha: float = Field(default=1.0, ge=0.0, le=4.0)
    brightness_beta: float = Field(default=0.0, ge=-128.0, le=128.0)
    clahe_enabled: bool = False
    clahe_clip_limit: float = Field(default=2.0, gt=0.0, le=40.0)
    clahe_grid: int = Field(default=8, ge=1, le=64)
    denoise: DenoiseMethod = DenoiseMethod.NONE
    denoise_strength: int = Field(default=3, ge=1, le=31)
    sharpen_amount: float = Field(default=0.0, ge=0.0, le=4.0)
    roi: RegionOfInterest | None = None

    @model_validator(mode="after")
    def _check_kernel_odd(self) -> PreprocessConfig:
        needs_odd = self.denoise in (DenoiseMethod.GAUSSIAN, DenoiseMethod.MEDIAN)
        if needs_odd and self.denoise_strength % 2 == 0:
            raise ValueError("denoise_strength must be odd for gaussian/median kernels")
        return self


class StabilizeConfig(StrictModel):
    """IMU feed-forward or visual electronic image stabilization."""

    mode: StabilizeMode = StabilizeMode.NONE
    focal_length_px: float = Field(default=900.0, gt=0.0)
    max_shift_px: float = Field(default=120.0, gt=0.0)
    smoothing_alpha: float = Field(default=0.80, ge=0.0, le=1.0)
    imu_lead_seconds: float = Field(default=0.0, ge=-0.5, le=0.5)
    max_features: int = Field(default=200, ge=10, le=5000)


class ClutterConfig(StrictModel):
    """Synthetic clutter and occlusion injection."""

    enabled: bool = False
    seed: int = Field(default=1337, ge=0)
    blob_count: int = Field(default=0, ge=0, le=200)
    blob_radius_px: int = Field(default=6, ge=1, le=200)
    noise_sigma: float = Field(default=0.0, ge=0.0, le=128.0)
    glare_intensity: float = Field(default=0.0, ge=0.0, le=1.0)
    occlusion_start_frame: int | None = Field(default=None, ge=0)
    occlusion_frames: int = Field(default=0, ge=0, le=10000)
    occlusion_coverage: float = Field(default=0.7, ge=0.0, le=1.0)


class MotionConfig(StrictModel):
    """Shoulder-fired angular rate and vibration model."""

    enabled: bool = False
    seed: int = Field(default=7, ge=0)
    tremor_hz: float = Field(default=9.0, gt=0.0, le=200.0)
    tremor_amplitude_dps: float = Field(default=1.2, ge=0.0, le=90.0)
    breathing_hz: float = Field(default=0.25, gt=0.0, le=10.0)
    breathing_amplitude_dps: float = Field(default=0.6, ge=0.0, le=90.0)
    drift_dps: float = Field(default=0.0, ge=-90.0, le=90.0)
    recoil_frame: int | None = Field(default=None, ge=0)
    recoil_amplitude_dps: float = Field(default=25.0, ge=0.0, le=500.0)
    recoil_decay_frames: int = Field(default=8, ge=1, le=500)


class DiagnosticsConfig(StrictModel):
    """Drop-event detection and overlay rendering."""

    drop_score_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    drop_consecutive_frames: int = Field(default=3, ge=1, le=600)
    drop_slope_per_frame: float = Field(default=0.12, ge=0.0, le=1.0)
    pre_event_frames: int = Field(default=15, ge=0, le=600)
    overlay_enabled: bool = False
    overlay_fps: float = Field(default=30.0, gt=0.0, le=1000.0)
    plot_height_px: int = Field(default=120, ge=40, le=800)


class SourceConfig(StrictModel):
    """Frame source definition."""

    video_path: Path | None = None
    image_dir: Path | None = None
    synthetic_frames: int = Field(default=0, ge=0, le=100000)
    frame_width: int = Field(default=640, ge=32, le=8192)
    frame_height: int = Field(default=480, ge=32, le=8192)
    start_frame: int = Field(default=0, ge=0)
    max_frames: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_exactly_one(self) -> SourceConfig:
        chosen = [
            self.video_path is not None,
            self.image_dir is not None,
            self.synthetic_frames > 0,
        ]
        if sum(chosen) != 1:
            raise ValueError("set exactly one of video_path, image_dir, or synthetic_frames")
        return self


class LabConfig(StrictModel):
    """Root configuration for a tracking session."""

    schema_version: str = SCHEMA_VERSION
    name: str = "session"
    initial_bbox: tuple[int, int, int, int] | None = None
    source: SourceConfig = Field(default_factory=lambda: SourceConfig(synthetic_frames=180))
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    stabilize: StabilizeConfig = Field(default_factory=StabilizeConfig)
    clutter: ClutterConfig = Field(default_factory=ClutterConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full config tree to JSON."""
        return self.model_dump_json(indent=indent)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc


def _parse_mapping(text: str, path: Path) -> dict[str, Any]:
    if path.suffix.lower() in (".yaml", ".yml"):
        return _parse_yaml(text, path)
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"config root must be an object, got {type(parsed).__name__}")
    return parsed


def _parse_yaml(text: str, path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ConfigError(
            "YAML config requires PyYAML: pip install 'pixel-lock-lab[yaml]'"
        ) from exc
    try:
        parsed: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"config root must be a mapping, got {type(parsed).__name__}")
    return parsed


def load_config(path: Path) -> LabConfig:
    """Load and validate a LabConfig from a .json, .yaml, or .yml file."""
    mapping: dict[str, Any] = _parse_mapping(_read_text(path), path)
    try:
        return LabConfig.model_validate(mapping)
    except ValueError as exc:
        raise ConfigError(f"config validation failed for {path}: {exc}") from exc


def save_config(config: LabConfig, path: Path) -> None:
    """Write a LabConfig to disk as JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(config.to_json(), encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot write config to {path}: {exc}") from exc
