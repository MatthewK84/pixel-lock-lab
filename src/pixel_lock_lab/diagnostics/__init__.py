"""Diagnostics: structured logs, drop analysis, and overlay rendering."""

from __future__ import annotations

from pixel_lock_lab.diagnostics.drop_analyzer import (
    DropCause,
    DropEvent,
    find_drop_events,
    summarize,
)
from pixel_lock_lab.diagnostics.log_schema import (
    TrackLogRecord,
    TrackLogWriter,
    read_log,
    record_from_state,
)
from pixel_lock_lab.diagnostics.overlay import OverlayWriter, annotate, render_score_plot

__all__ = [
    "DropCause",
    "DropEvent",
    "OverlayWriter",
    "TrackLogRecord",
    "TrackLogWriter",
    "annotate",
    "find_drop_events",
    "read_log",
    "record_from_state",
    "render_score_plot",
    "summarize",
]
