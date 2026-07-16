"""End-to-end session tests and CLI smoke tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from pixel_lock_lab.array_types import Array
from pixel_lock_lab.cli import main
from pixel_lock_lab.config.schemas import (
    ClutterConfig,
    LabConfig,
    SourceConfig,
    TrackerBackend,
    TrackerConfig,
    TrackStatus,
    save_config,
)
from pixel_lock_lab.diagnostics.drop_analyzer import DropCause
from pixel_lock_lab.errors import ConfigError
from pixel_lock_lab.session import Session
from pixel_lock_lab.sources import SyntheticSource, build_source


def _base_config(frames: int = 40) -> LabConfig:
    config = LabConfig(name="test")
    config.source = SourceConfig(synthetic_frames=frames, frame_width=320, frame_height=240)
    config.tracker = TrackerConfig(
        backend=TrackerBackend.TEMPLATE, lock_threshold=0.4, reacquire_threshold=0.5
    )
    return config


def test_synthetic_source_yields_expected_frames() -> None:
    source = SyntheticSource(SourceConfig(synthetic_frames=12, frame_width=64, frame_height=48))
    frames = list(source.frames())
    assert len(frames) == 12
    assert frames[0].shape == (48, 64, 3)


def test_synthetic_ground_truth_moves() -> None:
    source = SyntheticSource(SourceConfig(synthetic_frames=30, frame_width=320, frame_height=240))
    first = source.ground_truth(0)
    last = source.ground_truth(29)
    assert first is not None
    assert last is not None
    assert last.center[0] > first.center[0]


def test_build_source_defaults_to_synthetic() -> None:
    assert isinstance(build_source(SourceConfig(synthetic_frames=5)), SyntheticSource)


def test_session_holds_lock_on_clean_synthetic() -> None:
    result = Session(_base_config()).run()
    assert result.frames_processed == 40
    assert result.locked_fraction > 0.9
    assert result.drop_events == ()


def test_session_records_latency_stages() -> None:
    result = Session(_base_config(10)).run()
    stages = {s.stage for s in result.latency}
    assert {"simulate", "preprocess", "stabilize", "track"} <= stages


def test_session_writes_log(tmp_path: Path) -> None:
    log_path = tmp_path / "track.jsonl"
    result = Session(_base_config(15)).run(log_path=log_path)
    assert log_path.exists()
    assert (
        len(log_path.read_text(encoding="utf-8").strip().splitlines()) == result.frames_processed
    )


class _BlindSource:
    """Source with no ground truth, used to test the config guard."""

    def frames(self) -> Iterator[Array]:
        yield np.zeros((240, 320, 3), dtype=np.uint8)

    def ground_truth(self, frame_index: int) -> None:
        return None


def test_session_requires_bbox_without_ground_truth() -> None:
    config = _base_config()
    config.initial_bbox = None
    session = Session(config)
    object.__setattr__(session, "_source", _BlindSource())
    with pytest.raises(ConfigError, match="initial_bbox is required"):
        session.run()


def test_session_uses_explicit_initial_bbox() -> None:
    config = _base_config()
    config.initial_bbox = (140, 100, 30, 30)
    session = Session(config)
    object.__setattr__(session, "_source", _BlindSource())
    result = session.run()
    assert result.frames_processed == 1


def test_occlusion_produces_attributed_drop_event() -> None:
    config = _base_config(50)
    config.clutter = ClutterConfig(
        enabled=True,
        occlusion_start_frame=20,
        occlusion_frames=12,
        occlusion_coverage=1.0,
    )
    config.tracker.max_coast_frames = 4
    result = Session(config).run()
    assert len(result.drop_events) >= 1
    assert result.drop_events[0].cause is DropCause.OCCLUSION


def test_lost_status_appears_when_coast_budget_short() -> None:
    config = _base_config(50)
    config.clutter = ClutterConfig(
        enabled=True, occlusion_start_frame=15, occlusion_frames=20, occlusion_coverage=1.0
    )
    config.tracker.max_coast_frames = 2
    result = Session(config).run()
    assert any(r.status is TrackStatus.LOST for r in result.records)


def test_cli_config_template_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config-template"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "1.0"


def test_cli_config_template_to_file(tmp_path: Path) -> None:
    out = tmp_path / "cfg.json"
    assert main(["config-template", "--output", str(out)]) == 0
    assert out.exists()


def test_cli_list_trackers(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list-trackers"]) == 0
    assert "template" in capsys.readouterr().out


def test_cli_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = tmp_path / "cfg.json"
    save_config(_base_config(12), cfg_path)
    log_path = tmp_path / "log.jsonl"
    assert main(["run", "--config", str(cfg_path), "--log", str(log_path)]) == 0
    assert "Frames processed: 12" in capsys.readouterr().out
    assert log_path.exists()


def test_cli_run_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = tmp_path / "cfg.json"
    save_config(_base_config(10), cfg_path)
    assert main(["run", "--config", str(cfg_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "events" in payload
    assert "summary" in payload


def test_cli_analyze(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = tmp_path / "cfg.json"
    save_config(_base_config(12), cfg_path)
    log_path = tmp_path / "log.jsonl"
    main(["run", "--config", str(cfg_path), "--log", str(log_path)])
    capsys.readouterr()
    assert main(["analyze", "--log", str(log_path)]) == 0
    assert "drop event" in capsys.readouterr().out.lower()


def test_cli_missing_config_returns_error_code(tmp_path: Path) -> None:
    assert main(["run", "--config", str(tmp_path / "nope.json")]) == 1


def test_cli_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        main([])
