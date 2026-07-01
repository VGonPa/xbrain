"""Content-type-aware key-frame extraction for `digest-video --frames` (#44 PR4).

The **opt-in** visual layer: for a slide/screen/demo-heavy talk the visual
carries as much as the audio; for an interview the scene frames are camera cuts =
noise. This module makes the choice **content-aware** and keeps xbrain core
**ML-free** — the heavy vision (per-frame description) is an EXTERNAL subprocess
(`xbrain.vision`), never bundled. Two public entry points:

- `extract_key_frames(video_path, *, threshold, max_frames)` shells out to the
  external `ffmpeg` CLI (argv `shlex`-split, run WITHOUT a shell — the
  `transcribe.py` shape) and returns downscaled, timestamped `KeyFrame`s. It
  guards the **"scene detection stops mid-video"** gap: the ffmpeg `select`
  expression combines scene-change detection with a **periodic interval term**
  keyed on `prev_selected_t`, so a long STATIC TAIL (a 30-min static Q&A after a
  slide deck) is still sampled every `interval_seconds` — coverage spans the
  WHOLE video, in one pass, no duration probe. An over-`max_frames` result is
  subsampled EVENLY across the timeline (front + tail), never truncated to the
  first N (which would re-open the very gap).

- `classify_visual(frames)` decides `"slides"` vs `"talking_head"` from the
  fraction of frames whose **edge density** is high (text / sharp lines →
  high FIND_EDGES energy; smooth faces / bokeh → low). It uses Pillow — classic
  image processing, NOT a vision/ML library — so the digest can skip vision calls
  on interviews (skip + log, never a silent drop). A test asserts this module
  imports no ML/vision lib.

`ffmpeg` failures surface as clear operator errors: a **missing binary**
(`FrameExtractionToolNotFound`) aborts the run (a global config error, like
`TranscriberNotFound`); a **non-zero exit / timeout** (`FrameExtractionFailed`)
is per-video (the digest records it and continues the batch). The `runner` (a
`subprocess.run` stand-in) is injectable so tests run offline against a fake.

The extracted frames are written under a temp subdir of the video's parent; the
CALLER owns cleanup (in `digest-video` the enclosing ephemeral `TemporaryDirectory`
reclaims them once the kept slides have been persisted). Nothing here writes to
the store.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess  # nosec B404 - ffmpeg is an external subprocess by design (#44)
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)

# The external frame-extraction CLI. `ffmpeg` is already in the #44 tech stack;
# it is invoked as a subprocess, never imported.
DEFAULT_FFMPEG_COMMAND = "ffmpeg"

# Scene-change score above which ffmpeg's `select` filter keeps a frame. 0.4 is a
# middle-of-the-road cut sensitivity — high enough to ignore minor motion, low
# enough to catch a slide transition.
DEFAULT_SCENE_THRESHOLD = 0.4

# Upper bound on kept frames. A 72-min slide deck can yield hundreds of scene +
# interval samples; 40 is enough to represent a talk's slides without bloating
# the note or the vision-call budget.
DEFAULT_MAX_FRAMES = 40

# The periodic interval (seconds) for the static-tail fallback: the `select`
# filter also keeps a frame whenever this long has elapsed since the last kept
# one, so no stretch of the video — however static — goes unsampled.
DEFAULT_INTERVAL_SECONDS = 15.0

# Downscale width (px); height auto (`-2` keeps aspect + even dimension). Slides
# are legible at 640px and the vision step + vault embed stay lightweight.
_FRAME_WIDTH = 640

# A generous wall-clock cap: extracting frames from a 72-min talk is fast, but a
# wedged ffmpeg must not hang the run forever.
_DEFAULT_TIMEOUT_SECONDS = 900

# Per-frame edge-density cutoff (mean FIND_EDGES magnitude ÷ 255). Measured:
# text/line slides ~0.07, smooth gradients / bokeh ~0.004-0.011 — 0.03 cleanly
# separates them.
_SLIDE_EDGE_DENSITY_THRESHOLD = 0.03

# Fraction of frames that must read as slide-like for the WHOLE video to classify
# as "slides". A majority keeps one stray camera cut in a slide talk from tipping
# it to talking_head.
_SLIDE_FRACTION_THRESHOLD = 0.5

# Parse `pts_time:<seconds>` out of ffmpeg's `showinfo` stderr lines.
_PTS_TIME_RE = re.compile(r"pts_time:\s*([0-9]+(?:\.[0-9]+)?)")

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

VisualKind = Literal["slides", "talking_head"]


class FrameExtractionError(RuntimeError):
    """Base class for every key-frame-extraction failure (a clean CLI exit-1)."""


class FrameExtractionToolNotFound(FrameExtractionError):
    """The `ffmpeg` binary is missing / not executable (a global config error)."""


class FrameExtractionFailed(FrameExtractionError):
    """ffmpeg ran but failed: non-zero exit, timeout, or unreadable output."""


@dataclass(frozen=True)
class KeyFrame:
    """One extracted key frame: its `timestamp` (seconds) + the image on disk.

    `path` points at a downscaled PNG under the caller-owned temp tree. The image
    is decoded lazily by `classify_visual` (edge density) and read by the digest
    when a kept slide is persisted — `KeyFrame` itself stays lean (no bytes held).
    """

    timestamp: float
    path: Path


def _build_select_expr(threshold: float, interval_seconds: float) -> str:
    """The ffmpeg `select` expression: scene-change OR periodic interval.

    A frame is kept when ANY term is non-zero: `gt(scene,T)` (a cut), the first
    frame (`isnan(prev_selected_t)`), or `interval_seconds` elapsed since the last
    kept frame (`gte(t-prev_selected_t,interval)`). The interval term is what
    covers a long static tail scene detection alone would miss.
    """
    return f"gt(scene,{threshold})+isnan(prev_selected_t)+gte(t-prev_selected_t,{interval_seconds})"


def _build_argv(command: str, video: Path, out_pattern: Path, select_expr: str) -> list[str]:
    """Assemble the ffmpeg argv (shlex-split command, downscale, showinfo).

    `-vf select=...,scale=W:-2,showinfo` filters to the selected frames, downscales
    them, and logs each kept frame's `pts_time` to stderr; `-vsync vfr` writes one
    image per SELECTED frame (not per source frame). Run WITHOUT a shell.
    """
    filtergraph = f"select='{select_expr}',scale={_FRAME_WIDTH}:-2,showinfo"
    argv = shlex.split(command)
    argv += [
        "-nostdin",
        "-i",
        str(video),
        "-vf",
        filtergraph,
        "-vsync",
        "vfr",
        str(out_pattern),
    ]
    return argv


def _run_ffmpeg(argv: list[str], runner: Runner, timeout_seconds: int) -> str:
    """Run ffmpeg; return its stderr (carrying showinfo), or a clear operator error."""
    try:
        completed = runner(argv, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise FrameExtractionToolNotFound(
            f"frame extractor {argv[0]!r} not found — install ffmpeg to use "
            f"`digest-video --frames` ({exc})"
        ) from exc
    except OSError as exc:  # not-executable, permission denied, etc.
        raise FrameExtractionToolNotFound(
            f"frame extractor {argv[0]!r} could not be executed ({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise FrameExtractionFailed(
            f"frame extractor {argv[0]!r} timed out after {timeout_seconds}s"
        ) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise FrameExtractionFailed(
            f"frame extractor {argv[0]!r} exited {completed.returncode}: {stderr or '(no stderr)'}"
        )
    return completed.stderr or ""


def _parse_timestamps(stderr: str) -> list[float]:
    """Every `pts_time` from ffmpeg's showinfo stderr, in output order."""
    return [float(match) for match in _PTS_TIME_RE.findall(stderr)]


def _pair_frames(out_dir: Path, timestamps: list[float]) -> list[KeyFrame]:
    """Pair the produced image files (sorted) with their showinfo timestamps.

    ffmpeg emits frames and their showinfo lines in the same temporal order, so a
    positional zip is correct. A count mismatch (defensive — should not happen)
    pairs up to the shorter and logs, never crashes.
    """
    files = sorted(out_dir.glob("frame-*.png"))
    if len(files) != len(timestamps):
        logger.warning(
            "digest-video: ffmpeg produced %d frame(s) but %d timestamp(s); pairing %d.",
            len(files),
            len(timestamps),
            min(len(files), len(timestamps)),
        )
    return [KeyFrame(timestamp=ts, path=path) for ts, path in zip(timestamps, files)]


def _cap_evenly(frames: list[KeyFrame], max_frames: int) -> list[KeyFrame]:
    """Subsample `frames` to at most `max_frames`, spread ACROSS the timeline.

    Picks evenly-spaced indices including the first and last, so the cap keeps
    both the front AND the tail — truncating to the first N would re-open the
    static-tail-coverage gap the extractor works to close.
    """
    if max_frames < 1 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    last = len(frames) - 1
    indices = sorted({round(i * last / (max_frames - 1)) for i in range(max_frames)})
    return [frames[index] for index in indices]


def extract_key_frames(
    video_path: Path | str,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    max_frames: int = DEFAULT_MAX_FRAMES,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    command: str = DEFAULT_FFMPEG_COMMAND,
    runner: Runner | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> list[KeyFrame]:
    """Extract downscaled, timestamped key frames from `video_path` via ffmpeg.

    Combines scene-change detection (`threshold`) with periodic interval sampling
    (`interval_seconds`) in ONE ffmpeg `select` pass, so extraction covers the
    WHOLE video — including a long static tail — not just the front. The result is
    subsampled EVENLY to at most `max_frames` (front + tail preserved). Frames are
    written under a temp subdir of the video's parent and returned as `KeyFrame`s;
    the CALLER owns cleanup (the digest's ephemeral temp dir reclaims them).

    A missing ffmpeg raises `FrameExtractionToolNotFound` (aborts the run); a
    non-zero exit / timeout raises `FrameExtractionFailed` (per-video — the digest
    records it and continues). `runner` defaults to `subprocess.run`, resolved at
    call time so it stays monkeypatchable.
    """
    video = Path(video_path)
    active_runner: Runner = runner if runner is not None else subprocess.run
    out_dir = Path(tempfile.mkdtemp(prefix="xbrain-frames-", dir=video.parent))
    out_pattern = out_dir / "frame-%05d.png"
    select_expr = _build_select_expr(threshold, interval_seconds)
    argv = _build_argv(command, video, out_pattern, select_expr)
    stderr = _run_ffmpeg(argv, active_runner, timeout_seconds)
    frames = _pair_frames(out_dir, _parse_timestamps(stderr))
    return _cap_evenly(frames, max_frames)


def _edge_density(path: Path) -> float:
    """Mean FIND_EDGES magnitude of the frame (0..1) — high for text/line slides.

    A corrupt/unreadable frame reads as 0.0 (biases toward talking_head → skip,
    the safe direction) with a warning, never a crash mid-classification.
    """
    try:
        with Image.open(path) as img:
            edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
            return ImageStat.Stat(edges).mean[0] / 255.0
    except (OSError, ValueError) as exc:
        logger.warning("digest-video: could not read frame %s for classification: %s", path, exc)
        return 0.0


def classify_visual(frames: list[KeyFrame]) -> VisualKind:
    """Classify a frame set as `"slides"` or `"talking_head"`.

    Returns `"slides"` when at least `_SLIDE_FRACTION_THRESHOLD` of the frames read
    as slide-like (edge density ≥ `_SLIDE_EDGE_DENSITY_THRESHOLD`) — the visual
    layer is worth describing + embedding. Otherwise `"talking_head"`: the scene
    frames are camera cuts (noise), so the caller skips the visual layer (and
    logs the reason). An empty set has no visual signal → `"talking_head"`.
    """
    if not frames:
        return "talking_head"
    slide_like = sum(
        1 for frame in frames if _edge_density(frame.path) >= _SLIDE_EDGE_DENSITY_THRESHOLD
    )
    if slide_like / len(frames) >= _SLIDE_FRACTION_THRESHOLD:
        return "slides"
    return "talking_head"
