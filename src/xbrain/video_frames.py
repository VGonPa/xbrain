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
  on interviews (skip + log, never a silent drop). Each frame's bytes are read
  fully into memory before decoding (no lazy-FS decode race), and a frame that
  cannot be decoded is EXCLUDED from the vote rather than silently counted as a
  low-edge vote; an all-unreadable set returns the distinct `"unreadable"` so the
  caller surfaces it instead of masquerading it as talking-head. A test asserts
  this module imports no ML/vision lib.

Key-frame pairing is DETERMINISTIC: `_pair_frames` maps the i-th `showinfo`
timestamp onto `frame-{i:05d}.png` by parsed index (ffmpeg writes + logs frames
in the same order), never by globbing the directory back — so a cold/loaded
filesystem can't truncate or reorder the readback. A real count mismatch (a
logged frame with no file, or a surplus file with no showinfo line) is a LOUD
`FrameExtractionFailed`, never a silent truncation.

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

import io
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

# SAFETY ceiling on kept frames — applied AFTER perceptual dedup, which is the
# real reducer. Dedup drops near-identical slides so the budget covers DISTINCT
# ones; this cap only stops a pathological continuous-motion clip (gameplay) from
# emitting hundreds of frames. Raise it (`[frames].max_frames`) for very long decks.
DEFAULT_MAX_FRAMES = 60

# Perceptual-hash (dHash) near-duplicate removal. Two frames whose 64-bit dHashes
# differ by <= this Hamming distance are "the same slide" — the later one is
# dropped. 6/64 tolerates JPEG/scale noise + a cursor/laser-pointer moving on a
# held slide, without merging two genuinely different slides.
DEFAULT_DEDUPE_DISTANCE = 6
_DHASH_SIZE = 8  # 8x8 difference hash → 64-bit fingerprint

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

VisualKind = Literal["slides", "talking_head", "unreadable"]


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
    except UnicodeDecodeError as exc:  # subprocess.run(text=True) on non-UTF-8 stderr
        raise FrameExtractionFailed(
            f"frame extractor {argv[0]!r} produced non-UTF-8 output: {exc}"
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
    """Pair each showinfo timestamp with its INDEXED frame file — deterministically.

    ffmpeg writes one image per SELECTED frame and logs one `showinfo` line per
    selected frame in the SAME order, so the i-th `pts_time` belongs to
    `frame-{i:05d}.png` **by construction**. Pairing by that parsed index — never
    by `glob`-ing the directory back and zipping positionally — is deterministic:
    a cold/loaded filesystem that returns the listing truncated or reordered can no
    longer drop a slide or misalign a timestamp onto the wrong image.

    A genuine count mismatch is a LOUD `FrameExtractionFailed`, never a silent
    warn-and-truncate: an expected frame file MISSING (ffmpeg logged it but no
    image landed) or a SURPLUS file beyond the logged count (an image with no
    showinfo line) both raise, so the slide count the run reports always matches
    what ffmpeg actually produced.
    """
    frames: list[KeyFrame] = []
    for index, ts in enumerate(timestamps, start=1):
        path = out_dir / f"frame-{index:05d}.png"
        if not path.exists():
            raise FrameExtractionFailed(
                f"ffmpeg logged {len(timestamps)} selected frame(s) but {path.name} "
                "is missing — frame/timestamp pairing broke"
            )
        frames.append(KeyFrame(timestamp=ts, path=path))
    surplus = out_dir / f"frame-{len(timestamps) + 1:05d}.png"
    if surplus.exists():
        raise FrameExtractionFailed(
            f"ffmpeg wrote more frame files than the {len(timestamps)} it logged "
            f"({surplus.name} present with no showinfo line) — frame/timestamp pairing broke"
        )
    return frames


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


def _dhash(path: Path, hash_size: int = _DHASH_SIZE) -> int | None:
    """Perceptual difference-hash of a frame as an int, or None if unreadable.

    Downscales to (hash_size+1) x hash_size grayscale and encodes each row's
    left>right gradient as one bit — a compact fingerprint robust to scale / JPEG
    noise. Reads bytes fully before decode (same cold-FS guard as `_edge_density`).
    """
    try:
        with Image.open(io.BytesIO(path.read_bytes())) as img:
            small = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    except (OSError, ValueError) as exc:
        logger.warning("digest-video: could not hash frame %s for dedup: %s", path, exc)
        return None
    px = small.tobytes()  # "L" mode → one byte per pixel, row-major (width = hash_size+1)
    bits = 0
    for row in range(hash_size):
        base = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | int(px[base + col] > px[base + col + 1])
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def dedupe_frames(
    frames: list[KeyFrame], *, max_distance: int = DEFAULT_DEDUPE_DISTANCE
) -> list[KeyFrame]:
    """Drop consecutive near-duplicate frames (a held slide) by perceptual hash.

    Keeps a frame when its dHash differs from the LAST KEPT frame's by more than
    `max_distance` bits — so a slide held across several interval samples costs one
    kept frame, not many, and the frame budget covers DISTINCT slides. An
    unreadable frame is kept (never silently dropped) and resets the comparison.
    Order-preserving.
    """
    kept: list[KeyFrame] = []
    last_hash: int | None = None
    for frame in frames:
        h = _dhash(frame.path)
        if h is None or last_hash is None or _hamming(h, last_hash) > max_distance:
            kept.append(frame)
            last_hash = h
    return kept


def extract_key_frames(
    video_path: Path | str,
    *,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    max_frames: int = DEFAULT_MAX_FRAMES,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    dedupe: bool = True,
    dedupe_distance: int = DEFAULT_DEDUPE_DISTANCE,
    command: str = DEFAULT_FFMPEG_COMMAND,
    runner: Runner | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> list[KeyFrame]:
    """Extract downscaled, timestamped key frames from `video_path` via ffmpeg.

    Combines scene-change detection (`threshold`) with periodic interval sampling
    (`interval_seconds`) in ONE ffmpeg `select` pass, so extraction covers the
    WHOLE video — including a long static tail — not just the front. The pipeline
    is **extract → dedupe → cap**: `dedupe` (default on) drops near-identical
    consecutive frames (a held slide) by perceptual hash so the budget covers
    DISTINCT slides, then the result is subsampled EVENLY to at most `max_frames`
    (a safety ceiling; front + tail preserved). Frames are written under a temp
    subdir of the video's parent and returned as `KeyFrame`s; the CALLER owns
    cleanup (the digest's ephemeral temp dir reclaims them).

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
    if dedupe:
        before = len(frames)
        frames = dedupe_frames(frames, max_distance=dedupe_distance)
        logger.debug("digest-video: dedup kept %d/%d frames", len(frames), before)
    return _cap_evenly(frames, max_frames)


def _edge_density(path: Path) -> float | None:
    """Mean FIND_EDGES magnitude of the frame (0..1), or `None` if unreadable.

    High for text/line slides, low for smooth faces / bokeh. The image bytes are
    read FULLY into memory first and decoded from an in-memory buffer, so the
    decode never races a lazy filesystem read of a just-written frame (a cold-FS
    flake source) — the whole file is materialised before PIL touches it.

    A genuinely corrupt / unreadable frame returns `None` — DISTINCT from a real
    low-edge `0.0` — with a warning, so `classify_visual` can tell "no readable
    visual signal" from "smooth video" and never silently miscount a decode
    failure as a talking-head vote.
    """
    try:
        with Image.open(io.BytesIO(path.read_bytes())) as img:
            edges = img.convert("L").filter(ImageFilter.FIND_EDGES)
            return ImageStat.Stat(edges).mean[0] / 255.0
    except (OSError, ValueError) as exc:
        logger.warning("digest-video: could not read frame %s for classification: %s", path, exc)
        return None


def classify_visual(frames: list[KeyFrame]) -> VisualKind:
    """Classify a frame set as `"slides"`, `"talking_head"`, or `"unreadable"`.

    Returns `"slides"` when at least `_SLIDE_FRACTION_THRESHOLD` of the READABLE
    frames read as slide-like (edge density ≥ `_SLIDE_EDGE_DENSITY_THRESHOLD`) —
    the visual layer is worth describing + embedding. Otherwise `"talking_head"`:
    the scene frames are camera cuts (noise), so the caller skips the visual layer
    (and logs the reason).

    Unreadable frames are EXCLUDED from the vote (never counted as low-edge), so a
    couple of corrupt frames can't drag a real slide deck to talking_head. When
    EVERY frame is unreadable the result is `"unreadable"` — a distinct, systemic
    signal the caller surfaces + logs rather than silently treating as a content
    decision. An empty set has no visual signal → `"talking_head"`.
    """
    if not frames:
        return "talking_head"
    densities = [d for d in (_edge_density(frame.path) for frame in frames) if d is not None]
    if not densities:
        return "unreadable"
    slide_like = sum(1 for density in densities if density >= _SLIDE_EDGE_DENSITY_THRESHOLD)
    if slide_like / len(densities) >= _SLIDE_FRACTION_THRESHOLD:
        return "slides"
    return "talking_head"
