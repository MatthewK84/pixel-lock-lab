"""Command-line entry points.

Subcommands:
    run              run a session from a config file
    replay           re-run a recorded video and explain every drop
    analyze          extract drop events from an existing track log
    config-template  emit a fully populated default config
    list-trackers    show available backends
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pixel_lock_lab import __version__
from pixel_lock_lab.config.schemas import (
    LabConfig,
    SourceConfig,
    TrackerBackend,
    load_config,
    save_config,
)
from pixel_lock_lab.diagnostics.drop_analyzer import DropEvent, find_drop_events, summarize
from pixel_lock_lab.diagnostics.log_schema import TrackLogRecord, read_log
from pixel_lock_lab.errors import PixelLockLabError
from pixel_lock_lab.session import Session, SessionResult
from pixel_lock_lab.trackers.deep import is_deep_available

if TYPE_CHECKING:
    # argparse exposes no public type for what add_subparsers() returns.
    _SubParsers = argparse._SubParsersAction[argparse.ArgumentParser]

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1

logger: logging.Logger = logging.getLogger(__name__)


def _write(text: str) -> None:
    sys.stdout.write(text + "\n")


def _event_dict(event: DropEvent) -> dict[str, object]:
    return {
        "start_frame": event.start_frame,
        "end_frame": event.end_frame,
        "duration_frames": event.duration_frames,
        "lowest_score": round(event.lowest_score, 4),
        "score_at_start": round(event.score_at_start, 4),
        "max_slope": round(event.max_slope, 4),
        "cause": event.cause.value,
        "recovered": event.recovered,
    }


def _report_events(events: list[DropEvent], as_json: bool) -> None:
    if as_json:
        payload = {"events": [_event_dict(e) for e in events], "summary": summarize(events)}
        _write(json.dumps(payload, indent=2))
        return
    if not events:
        _write("No drop events detected.")
        return
    _write(f"{len(events)} drop event(s):")
    for event in events:
        recovered: str = "recovered" if event.recovered else "not recovered"
        _write(
            f"  frames {event.start_frame}-{event.end_frame} "
            f"({event.duration_frames}f)  cause={event.cause.value}  "
            f"score {event.score_at_start:.2f} -> {event.lowest_score:.2f}  {recovered}"
        )


def _report_session(result: SessionResult, as_json: bool) -> None:
    if as_json:
        _report_events(list(result.drop_events), as_json=True)
        return
    _write(f"Frames processed: {result.frames_processed}")
    _write(f"Locked fraction:  {result.locked_fraction:.1%}")
    _write("Latency by stage (ms):")
    for stats in result.latency:
        _write(
            f"  {stats.stage:<11} mean {stats.mean_ms:6.2f}  "
            f"p95 {stats.p95_ms:6.2f}  p99 {stats.p99_ms:6.2f}  max {stats.max_ms:6.2f}"
        )
    _report_events(list(result.drop_events), as_json=False)


def _cmd_run(args: argparse.Namespace) -> int:
    config: LabConfig = load_config(args.config)
    if args.overlay is not None:
        config.diagnostics.overlay_enabled = True
    result: SessionResult = Session(config).run(log_path=args.log, overlay_path=args.overlay)
    _report_session(result, args.json)
    return EXIT_OK


def _cmd_replay(args: argparse.Namespace) -> int:
    config: LabConfig = load_config(args.config)
    config.source = SourceConfig(video_path=args.video, max_frames=config.source.max_frames)
    if args.overlay is not None:
        config.diagnostics.overlay_enabled = True
    result: SessionResult = Session(config).run(log_path=args.log_out, overlay_path=args.overlay)
    _report_session(result, args.json)
    if args.log_in is not None:
        _compare_to_prior(args.log_in, config, result)
    return EXIT_OK


def _compare_to_prior(log_in: Path, config: LabConfig, result: SessionResult) -> None:
    prior: list[TrackLogRecord] = read_log(log_in)
    prior_events: list[DropEvent] = find_drop_events(prior, config.diagnostics)
    _write("")
    _write(f"Prior log: {len(prior_events)} drop(s) over {len(prior)} frames")
    _write(f"This run:  {len(result.drop_events)} drop(s) over {result.frames_processed} frames")


def _cmd_analyze(args: argparse.Namespace) -> int:
    records: list[TrackLogRecord] = read_log(args.log)
    config: LabConfig = LabConfig() if args.config is None else load_config(args.config)
    events: list[DropEvent] = find_drop_events(records, config.diagnostics, args.latency_budget_ms)
    _report_events(events, args.json)
    return EXIT_OK


def _cmd_config_template(args: argparse.Namespace) -> int:
    config: LabConfig = LabConfig()
    if args.output is None:
        _write(config.to_json())
        return EXIT_OK
    save_config(config, args.output)
    _write(f"Wrote default config to {args.output}")
    return EXIT_OK


def _cmd_list_trackers(_args: argparse.Namespace) -> int:
    _write("Available tracker backends:")
    for backend in TrackerBackend:
        note: str = "" if backend is not TrackerBackend.DETECTION else "  (needs a detector)"
        _write(f"  {backend.value}{note}")
    _write(f"Deep extra installed: {is_deep_available()}")
    return EXIT_OK


def _add_run_parser(sub: _SubParsers) -> None:
    parser = sub.add_parser("run", help="run a session from a config file")
    parser.add_argument("--config", type=Path, required=True, help="path to a JSON/YAML config")
    parser.add_argument("--log", type=Path, default=None, help="write a JSONL track log here")
    parser.add_argument("--overlay", type=Path, default=None, help="write an overlay video here")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.set_defaults(func=_cmd_run)


def _add_replay_parser(sub: _SubParsers) -> None:
    parser = sub.add_parser("replay", help="re-run a recorded video and explain every drop")
    parser.add_argument("--config", type=Path, required=True, help="path to a JSON/YAML config")
    parser.add_argument("--video", type=Path, required=True, help="recorded video to replay")
    parser.add_argument("--log-in", type=Path, default=None, help="previous track log to compare")
    parser.add_argument("--log-out", type=Path, default=None, help="write this run's log here")
    parser.add_argument("--overlay", type=Path, default=None, help="write an overlay video here")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.set_defaults(func=_cmd_replay)


def _add_analyze_parser(sub: _SubParsers) -> None:
    parser = sub.add_parser("analyze", help="extract drop events from a track log")
    parser.add_argument("--log", type=Path, required=True, help="JSONL track log to analyze")
    parser.add_argument("--config", type=Path, default=None, help="config for drop thresholds")
    parser.add_argument(
        "--latency-budget-ms", type=float, default=0.0, help="flag frames exceeding this budget"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    parser.set_defaults(func=_cmd_analyze)


def _add_misc_parsers(sub: _SubParsers) -> None:
    template = sub.add_parser("config-template", help="emit a fully populated default config")
    template.add_argument("--output", type=Path, default=None, help="write config here")
    template.set_defaults(func=_cmd_config_template)
    listing = sub.add_parser("list-trackers", help="show available tracker backends")
    listing.set_defaults(func=_cmd_list_trackers)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser."""
    parser = argparse.ArgumentParser(prog="pixel-lock-lab", description=__doc__)
    parser.add_argument("--version", action="version", version=f"pixel-lock-lab {__version__}")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    _add_replay_parser(sub)
    _add_analyze_parser(sub)
    _add_misc_parsers(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args: argparse.Namespace = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        exit_code: int = int(args.func(args))
    except PixelLockLabError as exc:
        logger.error("%s", exc)
        return EXIT_ERROR
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return EXIT_ERROR
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
