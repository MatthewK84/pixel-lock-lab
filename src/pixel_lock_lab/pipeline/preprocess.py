"""Controlled ISP-like processing stages.

Every stage is a pure function returning a new array. The Preprocessor
composes them in a fixed order: ROI -> exposure -> contrast -> denoise -> sharpen.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

from pixel_lock_lab.config.schemas import DenoiseMethod, PreprocessConfig, RegionOfInterest

BILATERAL_SIGMA: Final[float] = 50.0


def apply_roi(frame: np.ndarray, roi: RegionOfInterest | None) -> np.ndarray:
    """Crop to the configured ROI, clipped to frame bounds."""
    if roi is None:
        return frame
    height, width = frame.shape[:2]
    x2: int = min(roi.x + roi.width, width)
    y2: int = min(roi.y + roi.height, height)
    if roi.x >= width or roi.y >= height or x2 <= roi.x or y2 <= roi.y:
        return frame
    return np.asarray(frame[roi.y : y2, roi.x : x2].copy())


def apply_exposure(frame: np.ndarray, gain: float) -> np.ndarray:
    """Simulate sensor exposure gain with saturation."""
    if gain == 1.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=gain, beta=0.0)


def apply_contrast(frame: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Linear contrast and brightness adjustment."""
    if alpha == 1.0 and beta == 0.0:
        return frame
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def apply_clahe(frame: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization on luminance only."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    if frame.ndim == 2:
        return clahe.apply(frame)
    lab: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_denoise(frame: np.ndarray, method: DenoiseMethod, strength: int) -> np.ndarray:
    """Apply the selected denoise kernel."""
    if method is DenoiseMethod.NONE:
        return frame
    if method is DenoiseMethod.GAUSSIAN:
        return cv2.GaussianBlur(frame, (strength, strength), 0)
    if method is DenoiseMethod.MEDIAN:
        return cv2.medianBlur(frame, strength)
    return cv2.bilateralFilter(frame, strength, BILATERAL_SIGMA, BILATERAL_SIGMA)


def apply_sharpen(frame: np.ndarray, amount: float) -> np.ndarray:
    """Unsharp mask edge enhancement."""
    if amount <= 0.0:
        return frame
    blurred: np.ndarray = cv2.GaussianBlur(frame, (0, 0), sigmaX=2.0)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0.0)


class Preprocessor:
    """Applies the configured ISP stages in a fixed, inspectable order."""

    def __init__(self, config: PreprocessConfig) -> None:
        self._config: PreprocessConfig = config

    @property
    def config(self) -> PreprocessConfig:
        """The preprocessing configuration in use."""
        return self._config

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Run the full stage chain, returning a new array."""
        if not self._config.enabled:
            return frame
        stage: np.ndarray = apply_roi(frame, self._config.roi)
        stage = apply_exposure(stage, self._config.exposure_gain)
        stage = apply_contrast(stage, self._config.contrast_alpha, self._config.brightness_beta)
        if self._config.clahe_enabled:
            stage = apply_clahe(stage, self._config.clahe_clip_limit, self._config.clahe_grid)
        stage = apply_denoise(stage, self._config.denoise, self._config.denoise_strength)
        return apply_sharpen(stage, self._config.sharpen_amount)

    def roi_offset(self) -> tuple[int, int]:
        """Pixel offset introduced by ROI cropping, for coordinate mapping."""
        if self._config.roi is None or not self._config.enabled:
            return (0, 0)
        return (self._config.roi.x, self._config.roi.y)
