"""Tests for `xbrain.video_frames` — ffmpeg key-frame extraction + classification.

`extract_key_frames` shells out to the EXTERNAL `ffmpeg` CLI (subprocess, argv
`shlex`-split, no shell — the `transcribe.py` shape) for scene-change key frames,
combining the scene filter with a periodic interval term in ONE filter expression
so a long static tail can never go uncovered (the "scene detection stops
mid-video" gap). `classify_visual` decides slides vs talking_head from the
fraction of frames whose edge density is high (Pillow FIND_EDGES — classic image
processing, NOT a vision/ML library; a test asserts the module imports none).

Every test injects a fake ffmpeg `runner` (a `subprocess.run` stand-in) so NO
real ffmpeg runs; the fake writes frame files + emits `showinfo` `pts_time:` lines
exactly as the real tool does.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from xbrain.video_frames import (
    FrameExtractionFailed,
    FrameExtractionToolNotFound,
    KeyFrame,
    _dhash,
    classify_visual,
    dedupe_frames,
    extract_key_frames,
    select_frames,
)


def _completed(stderr: str, *, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["ffmpeg"], returncode=returncode, stdout="", stderr=stderr
    )


def _ffmpeg_runner(timestamps: list[float], *, returncode: int = 0, stderr: str = ""):
    """A fake `subprocess.run` mirroring real ffmpeg: writes one PNG per timestamp
    to the output pattern's dir and emits a `showinfo` `pts_time:` line per frame.

    The output pattern is ffmpeg's last argv token (`<dir>/frame-%05d.png`)."""
    calls: list[list[str]] = []

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        pattern = Path(argv[-1])
        out_dir = pattern.parent
        lines = []
        for index, ts in enumerate(timestamps, start=1):
            (out_dir / f"frame-{index:05d}.png").write_bytes(b"\x89PNG\r\n\x1a\n fake")
            lines.append(f"[Parsed_showinfo @ 0x0] n:{index - 1} pts:0 pts_time:{ts} duration:1")
        emitted = stderr or "\n".join(lines)
        return _completed(emitted, returncode=returncode)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def _slide_png(path: Path) -> Path:
    """A high-edge, text/line-heavy image — what a slide/screen frame looks like."""
    img = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, 320, 24):
        draw.line([(40, y), (600, y)], fill="black", width=3)
    draw.rectangle([300, 200, 560, 330], outline="black", width=4)
    img.save(path)
    return path


def _photo_png(path: Path) -> Path:
    """A low-edge, smooth image — what a talking-head / bokeh frame looks like."""
    img = Image.new("L", (640, 360))
    for x in range(640):
        for y in range(360):
            img.putpixel((x, y), int((x / 640) * 200 + (y / 360) * 40))
    img.convert("RGB").save(path)
    return path


# ------------------------------------------------------------ extract_key_frames


def test_extract_returns_timestamped_frames(tmp_path: Path):
    """Each produced frame becomes a `KeyFrame` with its `pts_time` timestamp and
    an on-disk image path."""
    runner = _ffmpeg_runner([2.0, 5.0, 9.0])
    frames = extract_key_frames(tmp_path / "vid.mp4", runner=runner)
    assert [f.timestamp for f in frames] == [2.0, 5.0, 9.0]
    assert all(isinstance(f, KeyFrame) and f.path.exists() for f in frames)


def test_extract_wires_scene_and_interval_for_whole_video_coverage(tmp_path: Path):
    """The select filter carries BOTH the scene-change term AND a periodic
    interval term keyed on `prev_selected_t` — so a long STATIC TAIL is still
    sampled (the 'scene detection stops mid-video' guard), plus a downscale."""
    runner = _ffmpeg_runner([1.0])
    extract_key_frames(tmp_path / "vid.mp4", threshold=0.4, runner=runner)
    argv = runner.calls[0]  # type: ignore[attr-defined]
    joined = " ".join(argv)
    assert argv[0] == "ffmpeg"
    assert "-i" in argv
    assert "gt(scene,0.4)" in joined  # scene-change detection
    assert "prev_selected_t" in joined  # interval-sampling fallback for static tails
    assert "scale=" in joined  # downscaled
    assert "showinfo" in joined  # timestamps


def test_extract_multi_token_ffmpeg_command_is_split(tmp_path: Path):
    """A multi-token `command` (a wrapper) is `shlex`-split into argv, not shelled."""
    runner = _ffmpeg_runner([1.0])
    extract_key_frames(tmp_path / "vid.mp4", command="nice ffmpeg", runner=runner)
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[:2] == ["nice", "ffmpeg"]


def test_extract_covers_the_static_tail_not_just_the_front(tmp_path: Path):
    """A video whose only later frame is a far static-tail sample keeps that
    tail frame — extraction covers the WHOLE video, not just the front."""
    runner = _ffmpeg_runner([1.0, 3.0, 3600.0])
    frames = extract_key_frames(tmp_path / "vid.mp4", runner=runner)
    assert max(f.timestamp for f in frames) == 3600.0  # the tail is represented


def test_extract_caps_evenly_and_keeps_the_tail(tmp_path: Path):
    """Over-`max_frames` extraction subsamples ACROSS the timeline (front + tail),
    never just the first N — else the cap would re-open the tail-coverage gap."""
    runner = _ffmpeg_runner([float(t) for t in range(10)])  # t = 0..9
    frames = extract_key_frames(tmp_path / "vid.mp4", max_frames=3, runner=runner)
    times = [f.timestamp for f in frames]
    assert len(frames) == 3
    assert times[0] == 0.0  # front kept
    assert times[-1] == 9.0  # tail kept


def test_extract_caps_to_single_front_frame(tmp_path: Path):
    """`max_frames=1` keeps exactly the first frame (the degenerate cap branch)."""
    runner = _ffmpeg_runner([float(t) for t in range(5)])
    frames = extract_key_frames(tmp_path / "vid.mp4", max_frames=1, runner=runner)
    assert [f.timestamp for f in frames] == [0.0]


def test_extract_pairs_timestamps_to_indexed_files_not_glob_order(tmp_path: Path):
    """Pairing is by the frame INDEX (`frame-<i>.png` ↔ the i-th `pts_time`), never
    by a directory listing — so it is deterministic even if a cold/loaded FS would
    return the glob in a different or truncated order. Each timestamp lands on its
    OWN indexed image, never misaligned onto a neighbour's file."""
    runner = _ffmpeg_runner([2.0, 5.0, 9.0])
    frames = extract_key_frames(tmp_path / "vid.mp4", runner=runner)
    assert [(f.timestamp, f.path.name) for f in frames] == [
        (2.0, "frame-00001.png"),
        (5.0, "frame-00002.png"),
        (9.0, "frame-00003.png"),
    ]


def test_extract_missing_frame_file_is_loud_failure(tmp_path: Path):
    """ffmpeg logs 3 selected frames but only 2 files materialise → the third's
    expected file is missing → a LOUD `FrameExtractionFailed`, never a silent
    warn-and-truncate that would drop a slide from the reported set."""

    def _run(argv, **_kwargs):
        out_dir = Path(argv[-1]).parent
        for index in range(1, 3):  # only 2 of the 3 logged frames written
            (out_dir / f"frame-{index:05d}.png").write_bytes(b"\x89PNG fake")
        lines = [f"[Parsed_showinfo @ 0x0] pts_time:{ts}" for ts in (1.0, 2.0, 3.0)]
        return _completed("\n".join(lines))

    with pytest.raises(FrameExtractionFailed) as excinfo:
        extract_key_frames(tmp_path / "vid.mp4", runner=_run)
    assert "frame-00003.png" in str(excinfo.value)


def test_extract_surplus_frame_file_is_loud_failure(tmp_path: Path):
    """ffmpeg writes MORE image files than the frames it logged → a surplus file
    beyond the logged count → a LOUD `FrameExtractionFailed`, never silently
    dropping the surplus (which would understate the slide set)."""

    def _run(argv, **_kwargs):
        out_dir = Path(argv[-1]).parent
        for index in range(1, 4):  # 3 files written
            (out_dir / f"frame-{index:05d}.png").write_bytes(b"\x89PNG fake")
        return _completed("[Parsed_showinfo @ 0x0] pts_time:2.0")  # only 1 logged

    with pytest.raises(FrameExtractionFailed):
        extract_key_frames(tmp_path / "vid.mp4", runner=_run)


def test_extract_non_utf8_ffmpeg_stderr_raises_failed(tmp_path: Path):
    """`subprocess.run(text=True)` raises `UnicodeDecodeError` decoding a non-UTF-8
    ffmpeg banner (Latin-1 / Shift-JIS container tags); it is a per-video
    `FrameExtractionFailed` (the batch continues), never an uncaught abort — the
    same contract `vision.py` / `transcribe.py` already honour."""

    def _run(_argv, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(FrameExtractionFailed) as excinfo:
        extract_key_frames(tmp_path / "vid.mp4", runner=_run)
    assert "non-UTF-8" in str(excinfo.value)


def test_extract_missing_ffmpeg_raises_tool_not_found(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    with pytest.raises(FrameExtractionToolNotFound) as excinfo:
        extract_key_frames(tmp_path / "vid.mp4", runner=_run)
    assert "ffmpeg" in str(excinfo.value)


def test_extract_non_executable_ffmpeg_raises_tool_not_found(tmp_path: Path):
    """A present-but-non-executable ffmpeg (a `PermissionError`/`OSError`, not a
    `FileNotFoundError`) is still a global config error that aborts the run —
    `FrameExtractionToolNotFound`, mirroring the vision wrapper."""

    def _run(_argv, **_kwargs):
        raise PermissionError(13, "Permission denied", "ffmpeg")

    with pytest.raises(FrameExtractionToolNotFound) as excinfo:
        extract_key_frames(tmp_path / "vid.mp4", runner=_run)
    assert "ffmpeg" in str(excinfo.value)


def test_extract_timeout_raises_failed(tmp_path: Path):
    """A wedged ffmpeg (`TimeoutExpired`) is a per-video `FrameExtractionFailed`
    (the batch continues), not a global abort."""

    def _run(_argv, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1)

    with pytest.raises(FrameExtractionFailed):
        extract_key_frames(tmp_path / "vid.mp4", runner=_run)


def test_extract_nonzero_exit_raises_failed(tmp_path: Path):
    runner = _ffmpeg_runner([], returncode=1, stderr="Invalid data found")
    with pytest.raises(FrameExtractionFailed) as excinfo:
        extract_key_frames(tmp_path / "vid.mp4", runner=runner)
    assert "Invalid data found" in str(excinfo.value)


def test_extract_no_frames_returns_empty(tmp_path: Path):
    """A video with no selected frames (exit 0, no files) yields an empty list —
    the caller (classify) then treats it as no visual signal, not a crash."""
    runner = _ffmpeg_runner([])
    assert extract_key_frames(tmp_path / "vid.mp4", runner=runner) == []


# ------------------------------------------------------------ classify_visual


def test_classify_slides_for_high_edge_set(tmp_path: Path):
    """A set dominated by text/line-heavy frames classifies as 'slides' — the
    visual layer is worth keeping."""
    frames = [
        KeyFrame(timestamp=float(i), path=_slide_png(tmp_path / f"s{i}.png")) for i in range(4)
    ]
    assert classify_visual(frames) == "slides"


def test_classify_talking_head_for_low_edge_set(tmp_path: Path):
    """A set of smooth (bokeh / face) frames classifies as 'talking_head' — the
    scene frames are camera cuts = noise, so the visual layer is skipped."""
    frames = [
        KeyFrame(timestamp=float(i), path=_photo_png(tmp_path / f"p{i}.png")) for i in range(4)
    ]
    assert classify_visual(frames) == "talking_head"


def test_classify_mostly_slides_still_slides(tmp_path: Path):
    """A talk that is mostly slides with one camera cut is still 'slides'."""
    frames = [
        KeyFrame(timestamp=float(i), path=_slide_png(tmp_path / f"s{i}.png")) for i in range(3)
    ]
    frames.append(KeyFrame(timestamp=9.0, path=_photo_png(tmp_path / "p.png")))
    assert classify_visual(frames) == "slides"


def test_classify_empty_is_talking_head(tmp_path: Path):
    """No frames → no visual signal → 'talking_head' (skip the visual layer)."""
    assert classify_visual([]) == "talking_head"


def test_classify_all_unreadable_returns_unreadable(tmp_path: Path):
    """A frame set where EVERY frame fails to decode classifies as 'unreadable' — a
    DISTINCT signal (a systemic extraction problem), never silently degraded to
    'talking_head' (which would look like a content decision to the operator)."""
    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"not a real image")
    frames = [KeyFrame(timestamp=float(i), path=bad) for i in range(3)]
    assert classify_visual(frames) == "unreadable"


def test_classify_ignores_unreadable_frames_among_readable_slides(tmp_path: Path):
    """An unreadable frame is NOT counted as a low-edge (talking-head) vote — it is
    excluded, so a mostly-slide set with one corrupt frame still classifies
    'slides' rather than being dragged toward talking_head."""
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a real image")
    frames = [
        KeyFrame(timestamp=float(i), path=_slide_png(tmp_path / f"s{i}.png")) for i in range(3)
    ]
    frames.append(KeyFrame(timestamp=9.0, path=corrupt))
    assert classify_visual(frames) == "slides"


def test_video_frames_imports_no_ml_or_vision_library():
    """xbrain core stays ML-free: the vision step is EXTERNAL. `video_frames.py`
    may use Pillow (classic image processing) but must import no vision/ML lib."""
    import xbrain.video_frames as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import torch",
        "import mlx",
        "import cv2",
        "import tensorflow",
        "transformers",
    ):
        assert forbidden not in source


def _frame(tmp_path: Path, name: str, boxes: list[tuple[int, int, int, int]]) -> KeyFrame:
    """A structured 'slide' PNG (white with black boxes) → a KeyFrame.

    dHash captures STRUCTURE, not brightness (a uniform image hashes to 0), so the
    fixtures draw distinct box layouts to exercise the perceptual hash meaningfully.
    """
    img = Image.new("RGB", (256, 144), "white")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(box, fill="black")
    path = tmp_path / name
    img.save(path)
    return KeyFrame(timestamp=0.0, path=path)


def test_dhash_is_stable_and_discriminates(tmp_path: Path):
    a = _frame(tmp_path, "a.png", [(10, 10, 100, 130)]).path
    a_copy = _frame(tmp_path, "a2.png", [(10, 10, 100, 130)]).path  # identical drawing
    b = _frame(tmp_path, "b.png", [(150, 10, 240, 130)]).path  # box on the far side
    assert _dhash(a) == _dhash(a_copy)  # identical structure → same fingerprint
    assert _dhash(a) != _dhash(b)  # different structure → different fingerprint
    assert _dhash(tmp_path / "missing.png") is None  # unreadable → None


def test_dedupe_drops_consecutive_near_duplicates(tmp_path: Path):
    a = _frame(tmp_path, "a.png", [(10, 10, 100, 130)])
    a_dup = _frame(tmp_path, "a2.png", [(10, 10, 100, 130)])  # a held slide, re-sampled
    b = _frame(tmp_path, "b.png", [(150, 10, 240, 130)])
    kept = dedupe_frames([a, a_dup, b], max_distance=6)
    assert [f.path.name for f in kept] == ["a.png", "b.png"]  # the duplicate is dropped


def test_dedupe_keeps_distinct_slides(tmp_path: Path):
    frames = [
        _frame(tmp_path, "tl.png", [(10, 10, 90, 60)]),  # top-left
        _frame(tmp_path, "br.png", [(170, 90, 250, 135)]),  # bottom-right
        _frame(tmp_path, "wide.png", [(10, 10, 250, 40)]),  # wide top bar
    ]
    assert len(dedupe_frames(frames, max_distance=6)) == 3


def test_dedupe_keeps_unreadable_frames(tmp_path: Path):
    good = _frame(tmp_path, "good.png", [(10, 10, 100, 130)])
    bad = KeyFrame(timestamp=1.0, path=tmp_path / "nope.png")  # no file → unreadable
    kept = dedupe_frames([good, bad, good], max_distance=6)
    assert bad in kept  # unreadable is never silently dropped


def test_extract_cap_false_returns_raw_frames(tmp_path: Path):
    """`cap=False` returns the RAW extraction (no even-subsample) — the digest
    classifies slides-vs-talking-head on this full distribution before reducing."""
    runner = _ffmpeg_runner([float(t) for t in range(10)])
    frames = extract_key_frames(tmp_path / "vid.mp4", max_frames=3, cap=False, runner=runner)
    assert len(frames) == 10  # not capped to 3


def test_select_frames_dedupes_then_caps(tmp_path: Path):
    """`select_frames` = dedupe (drop held-slide near-dups) → cap evenly."""
    a = _frame(tmp_path, "a.png", [(10, 10, 100, 130)])
    a_dup = _frame(tmp_path, "a2.png", [(10, 10, 100, 130)])  # dropped by dedup
    frames = [
        a,
        a_dup,
        _frame(tmp_path, "b.png", [(10, 10, 90, 40)]),
        _frame(tmp_path, "c.png", [(160, 100, 250, 140)]),
        _frame(tmp_path, "d.png", [(10, 100, 250, 140)]),
    ]
    # dedup → [a, b, c, d] (4 distinct); cap 2 → 2, front+tail preserved.
    kept = select_frames(frames, dedupe=True, dedupe_distance=6, max_frames=2)
    assert len(kept) == 2
    assert kept[0].path.name == "a.png"  # front kept
    # dedupe=False → no dedup, cap 3 of the 5.
    assert len(select_frames(frames, dedupe=False, max_frames=3)) == 3
