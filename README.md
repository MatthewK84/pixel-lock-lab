# pixel-lock-lab

A diagnostic and reference tracker toolkit for EO/IR pixel-lock analysis.

**BLUF:** This package exists to answer one question well. When a tracker let go of a target, why did it let go? It is a bench instrument, not a fielded tracker and not a control loop.

## Scope

This is deliberately a diagnostic and reference toolkit. It does not replace a fielded C-sUAS tracker, it does not close a control loop, and it does not command anything. That keeps the license simple and the engineering honest. Use it to characterize behavior, tune parameters against recorded data, and produce evidence about drop causes.

Licensed Apache-2.0.

## Install

```bash
pip install -e ".[dev]"
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `yaml` | PyYAML | YAML config files |
| `data` | Polars | Parquet log export |
| `deep` | Ultralytics | YOLO detector adapter |

The core stays pure Python + NumPy + OpenCV so it can later be accelerated with ONNX or TensorRT on Jetson-class hardware without restructuring.

## Quickstart

Everything runs with no input media. The synthetic source generates a moving target on a textured background and knows its own ground truth.

```bash
# Write a fully populated default config
pixel-lock-lab config-template --output config.json

# Run a session, write a track log and an annotated overlay video
pixel-lock-lab run --config config.json --log track.jsonl --overlay overlay.mp4

# Extract drop events from an existing log
pixel-lock-lab analyze --log track.jsonl --latency-budget-ms 33

# Replay mode: recorded video + previous log, see exactly why lock was lost
pixel-lock-lab replay --config config.json --video flight.mp4 \
    --log-in old_track.jsonl --overlay explained.mp4
```

Library use:

```python
from pixel_lock_lab import LabConfig, Session

config = LabConfig()
config.tracker.lock_threshold = 0.45
config.clutter.enabled = True
config.clutter.occlusion_start_frame = 60
config.clutter.occlusion_frames = 15

result = Session(config).run()
print(f"locked {result.locked_fraction:.1%} of frames")
for event in result.drop_events:
    print(f"frames {event.start_frame}-{event.end_frame}: {event.cause.value}")
```

## Architecture

```
src/pixel_lock_lab/
├── trackers/
│   ├── base.py           # common interface + shared lock/coast state machine
│   ├── classical.py      # NCC template matching, OpenCV CSRT/KCF/MOSSE
│   ├── optical_flow.py   # Lucas-Kanade with forward-backward gating
│   └── deep.py           # Detector protocol, association, Ultralytics adapter
├── pipeline/
│   ├── preprocess.py     # ROI, exposure, contrast, CLAHE, denoise, sharpen
│   ├── stabilize.py      # IMU feed-forward and visual EIS + residual logging
│   └── latency.py        # per-stage timing with p50/p95/p99
├── diagnostics/
│   ├── drop_analyzer.py  # drop event extraction + cause attribution
│   ├── overlay.py        # annotated video with score-collapse plot
│   └── log_schema.py     # structured JSONL / Parquet logs
├── simulation/
│   ├── clutter.py        # blobs, noise, glare, occlusion
│   └── motion.py         # tremor, breathing, drift, recoil transients
├── config/schemas.py     # Pydantic v2 models for every tunable parameter
├── sources.py            # video, image directory, synthetic scene
├── session.py            # orchestration
└── cli.py                # command-line entry points
```

### One design decision worth knowing

The lock threshold, re-acquire hysteresis, and coasting logic live in `BaseTracker`, not in each backend. Backends only implement `_initialize` and `_measure`. This means a change to `lock_threshold` means the same thing whether you are running CSRT or optical flow, so backend comparisons are actually comparisons of the backend rather than of four different interpretations of "lock."

The cost is that backends must produce a score in `[0, 1]`. Template matching gives a true NCC peak. Optical flow uses the surviving fraction of seeded points. OpenCV's CSRT/KCF/MOSSE expose no confidence, so the score there is an NCC of the current patch against a maintained template. That is a proxy, and it is worth remembering it is a proxy when reading CSRT scores.

## Tunable parameters

All fields are Pydantic-validated and serializable. Invalid combinations fail at load time, not mid-run.

### `tracker`

| Field | Default | Meaning |
|---|---|---|
| `backend` | `template` | `template`, `csrt`, `kcf`, `mosse`, `optical_flow`, `detection` |
| `lock_threshold` | 0.55 | Score needed to hold an existing lock |
| `reacquire_threshold` | 0.70 | Score needed to re-acquire after a drop; must be >= `lock_threshold` |
| `search_window_scale` | 2.5 | Search window as a multiple of the target box |
| `template_update_rate` | 0.05 | EMA rate for template drift; 0 disables updates |
| `max_coast_frames` | 12 | Dead-reckoning budget before declaring LOST |
| `coast_velocity_decay` | 0.90 | Per-frame velocity decay while coasting |
| `velocity_smoothing` | 0.60 | EMA factor on measured velocity |
| `association_iou` | 0.30 | Minimum IoU for detection association |

The hysteresis constraint (`reacquire_threshold >= lock_threshold`) is enforced by the schema. Without it you get lock chatter at the threshold boundary.

### `preprocess`

`exposure_gain`, `contrast_alpha`, `brightness_beta`, `clahe_enabled`, `clahe_clip_limit`, `clahe_grid`, `denoise` (`none`/`gaussian`/`median`/`bilateral`), `denoise_strength`, `sharpen_amount`, `roi`.

Stages apply in a fixed order: ROI → exposure → contrast → CLAHE → denoise → sharpen.

### `stabilize`

`mode` (`none`/`imu`/`visual`), `focal_length_px`, `max_shift_px`, `smoothing_alpha`, `imu_lead_seconds`, `max_features`.

IMU mode maps angular rates to pixels via `shift_px = focal_length_px * tan(angle_rad)`. `imu_lead_seconds` is the knob for IMU-to-image synchronization, which in practice is the hard part of this whole area — set it wrong and the stabilizer adds motion instead of removing it. Both modes report `residual_px` so you can tell under-compensation from over-compensation.

### `simulation`

`clutter`: blob count and radius, Gaussian noise sigma, glare intensity, occlusion window and coverage.
`motion`: tremor (Hz + amplitude), breathing, constant drift, and an optional recoil transient with exponential decay.

Both are seeded and deterministic. The same config and frame index always produce the same corruption, so a drop event replays exactly.

### `diagnostics`

`drop_score_threshold`, `drop_consecutive_frames`, `drop_slope_per_frame`, `pre_event_frames`, `overlay_enabled`, `overlay_fps`, `plot_height_px`.

## Drop events

A drop is a run of frames where the score falls below the floor, or the tracker enters COASTING/LOST, lasting at least `drop_consecutive_frames`. Each event carries `pre_event_frames` of context, the score at the start, the steepest single-frame decline, and a heuristic cause:

| Cause | Evidence |
|---|---|
| `occlusion` | Frames flagged occluded in the window |
| `motion` | Residual motion or track velocity above threshold |
| `low_contrast` | Mean intensity in the box collapsed |
| `latency` | Frame cost exceeded the supplied budget |
| `score_decay` | None of the above; the score simply eroded |

**Causes are ranked evidence, not ground truth.** They point you at the frames worth watching. The overlay video is where you confirm.

## Logs

JSONL by default: append-only, survives a crash mid-run, no dependencies. `to_parquet()` is available with the `data` extra once logs get large. Every record is a validated `TrackLogRecord` with a `schema_version`, so old logs stay readable.

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest -q
```

The package follows NASA Power-of-Ten-inspired standards: full type hints, no recursion, functions under 30 lines, no mutable global state, specific exceptions only, and zero static-analysis suppressions. CI enforces all of it on 3.10 through 3.12.

## Extending

Swap in a vendor tracker by subclassing `BaseTracker` and implementing two methods:

```python
from pixel_lock_lab.trackers.base import BaseTracker, Measurement

class VendorTracker(BaseTracker):
    def _initialize(self, frame, bbox) -> None:
        self._handle = vendor_sdk.open(frame, bbox.as_int_tuple())

    def _measure(self, frame, search_box) -> Measurement:
        box, confidence = self._handle.step(frame)
        return Measurement(box, confidence)
```

The lock policy, coasting, logging, drop analysis, and overlays all come for free.

Or plug in any detector via the `Detector` protocol:

```python
class MyDetector:
    def detect(self, frame) -> list[Detection]:
        return [Detection(BoundingBox(x, y, w, h), confidence)]
```
