"""Domain exceptions. Every module raises from this hierarchy."""

from __future__ import annotations


class PixelLockLabError(Exception):
    """Base exception for all pixel_lock_lab failures."""


class ConfigError(PixelLockLabError):
    """Raised when configuration is missing, malformed, or invalid."""


class TrackerError(PixelLockLabError):
    """Raised when a tracker is misused or fails to initialize."""


class BackendUnavailableError(TrackerError):
    """Raised when an optional tracker backend is not installed."""


class FrameSourceError(PixelLockLabError):
    """Raised when frames cannot be read from a source."""


class DiagnosticsError(PixelLockLabError):
    """Raised when logging, analysis, or overlay rendering fails."""
