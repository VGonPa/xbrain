"""Tests for `xbrain.redescribe` — offline re-captioning of stored key frames.

The engine reads pixels from `data/media/<id>/frames/` and NEVER touches the
network, X or ffmpeg. Every test injects a fake `describe_fn`, so no vision model
runs. Staleness is keyed on the source's `caption_contract`, so a corpus already
at the current contract costs nothing to re-run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.models import (
    FRAME_CAPTION_CONTRACT,
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    VideoFrame,
)
from xbrain.redescribe import (
    RedescribeReport,
    _frame_bearing_sources,
    format_redescribe_summary,
    redescribe_frames,
    stale_video_sources,
)
from xbrain.vision import VisionFailed, VisionNotFound


def _item(item_id: str, *, contract: str = "", descriptions: tuple[str, ...] = ("old",)) -> Item:
    """An item whose `x_video` source carries `descriptions` as frame captions.

    `Item` requires `source`, `author`, `created_at` and `captured_at` — all four
    are mandatory on the model, so none of them can be omitted here.
    """
    frames = [
        VideoFrame(timestamp=float(n), local_path=f"{item_id}/frames/{n}.png", description=text)
        for n, text in enumerate(descriptions)
    ]
    video_source = ContentSourceSuccess(
        kind="x_video",
        url=f"https://x.com/a/status/{item_id}",
        text="transcript",
        frames=frames,
        caption_contract=contract,
    )
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text="tweet",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        content=Content(
            fetched_at=datetime(2020, 1, 1, tzinfo=timezone.utc), sources=[video_source]
        ),
    )


def _media(tmp_path: Path, item: Item) -> Path:
    """Write a real byte for every frame the item references."""
    for source in item.content.sources:
        for frame in source.frames:
            path = tmp_path / frame.local_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x89PNG")
    return tmp_path


def _bare_item(item_id: str, *, content: Content | None) -> Item:
    """An item carrying an arbitrary (or absent) `content` block — for exercising
    `_frame_bearing_sources`'s guards directly, bypassing `_item`'s fixed
    single-`x_video`-source shape."""
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text="tweet",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        content=content,
    )


def test_frame_bearing_sources_returns_nothing_when_content_is_none():
    """M1 (review): per CLAUDE.md rule 6, 1551/2168 real items carry no `content`
    block at all — this is the module's most-executed guard in production, and
    the mutation deleting the `content is None` early return left all 17 tests
    green."""
    item = _bare_item("1", content=None)
    assert _frame_bearing_sources(item) == []


def test_frame_bearing_sources_skips_a_content_source_failure():
    """M1 (review): a `ContentSourceFailure` on `item.content.sources` must never
    reach the vision engine — the `isinstance(ContentSourceSuccess)` narrowing
    guards it. Deleting that narrowing left all 17 tests green."""
    failure = ContentSourceFailure(
        kind="x_video", url="https://x.com/a/status/1", failure_reason="not_found"
    )
    item = _bare_item(
        "1",
        content=Content(fetched_at=datetime(2020, 1, 1, tzinfo=timezone.utc), sources=[failure]),
    )
    assert _frame_bearing_sources(item) == []


def test_frame_bearing_sources_skips_a_non_video_source_with_frames():
    """M1 (review): only `kind == "x_video"` sources are frame-bearing in
    practice, but nothing stops a differently-kinded `ContentSourceSuccess` from
    carrying a (nonsensical) `frames` list — the `kind == "x_video"` filter guards
    against treating it as one. Deleting that filter left all 17 tests green."""
    article_with_frames = ContentSourceSuccess(
        kind="external_article",
        url="https://example.com/a",
        text="article body",
        frames=[VideoFrame(timestamp=0.0, local_path="1/frames/0.png", description="old")],
    )
    item = _bare_item(
        "1",
        content=Content(
            fetched_at=datetime(2020, 1, 1, tzinfo=timezone.utc), sources=[article_with_frames]
        ),
    )
    assert _frame_bearing_sources(item) == []


def test_a_source_without_the_current_contract_is_stale():
    store = {"1": _item("1", contract="")}
    assert [item_id for item_id, _ in stale_video_sources(store)] == ["1"]


def test_a_source_at_the_current_contract_is_not_stale():
    store = {"1": _item("1", contract=FRAME_CAPTION_CONTRACT)}
    assert stale_video_sources(store) == []


def test_force_makes_a_current_source_stale_again():
    store = {"1": _item("1", contract=FRAME_CAPTION_CONTRACT)}
    assert [item_id for item_id, _ in stale_video_sources(store, force=True)] == ["1"]


def test_a_source_with_no_frames_is_never_stale():
    """An audio-only video has no captions to fix — it must not be re-described,
    or a backfill run would call the vision model zero times but still report work."""
    store = {"1": _item("1", contract="", descriptions=())}
    assert stale_video_sources(store) == []


def test_redescribe_replaces_every_caption_and_stamps_the_contract(tmp_path: Path):
    item = _item("1", descriptions=("old a", "old b"))
    store = {"1": item}
    _media(tmp_path, item)

    report = redescribe_frames(
        store, media_root=tmp_path, describe_fn=lambda path: f"new {path.stem}"
    )

    source = store["1"].content.sources[0]
    assert [f.description for f in source.frames] == ["new 0", "new 1"]
    assert source.caption_contract == FRAME_CAPTION_CONTRACT
    assert report.videos_selected == 1
    assert report.frames_described == 2
    assert report.frames_failed == 0


def test_the_described_path_is_resolved_under_the_injected_media_root(tmp_path: Path):
    """PIN: `describe_fn` is handed a path built from the INJECTED `media_root`,
    never from a hardcoded `data/media` or from cwd.

    This does NOT defend against path traversal — `VideoFrame.local_path` already
    rejects an escaping value at construction time
    (`_reject_local_path_traversal`, `src/xbrain/models.py`), so an adversarial
    `local_path` is unconstructible and cannot reach this engine at all; that
    defence is exercised by
    `tests/test_models.py::test_video_frame_rejects_absolute_and_traversal_local_path`.
    What this test actually pins is the resolution seam: `tmp_path` is a fresh
    random directory every run, so an implementation that reads from a
    hardcoded `data/media` or from the process's cwd instead of the `media_root`
    argument would fail here even though nothing traversed anything.
    """
    item = _item("1")
    store = {"1": item}
    _media(tmp_path, item)
    seen: list[Path] = []

    redescribe_frames(
        store,
        media_root=tmp_path,
        describe_fn=lambda path: seen.append(path) or "new",
    )
    assert seen == [tmp_path / "1/frames/0.png"]


def test_a_missing_image_is_a_per_frame_failure_not_a_run_abort(tmp_path: Path):
    """One deleted PNG must not cost the other 2076 frames their run."""
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")

    source = store["1"].content.sources[0]
    assert report.frames_failed == 1
    assert report.frames_described == 1
    # The surviving frame got its new caption; the missing one kept its old text.
    assert source.frames[0].description == "a"
    assert source.frames[1].description == "new"


def test_a_vision_failure_is_a_per_frame_failure_not_a_run_abort(tmp_path: Path):
    """C1 (review): `describe_fn` raises `VisionFailed` on non-zero exit, a
    timeout, AND on an exit-0-but-empty-output run (`vision.py:147-151`) — exactly
    the corrupt-frame case a 2077-frame backfill will meet. It must be handled the
    same way a missing file is: logged, the frame keeps its old caption, the run
    CONTINUES to the other frames (and other items), and the source is left
    unstamped so a later run retries it."""
    item = _item("1", descriptions=("a", "b"))
    other = _item("2", descriptions=("c",))
    store = {"1": item, "2": other}
    _media(tmp_path, item)
    _media(tmp_path, other)

    corrupt_frame = tmp_path / "1/frames/0.png"

    def _describe(path: Path) -> str:
        if path == corrupt_frame:
            raise VisionFailed("vision command exited 1: corrupt frame")
        return f"new {path.stem}"

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=_describe)

    source = store["1"].content.sources[0]
    assert report.frames_failed == 1
    assert report.frames_described == 2
    # The frame whose vision call raised kept its old caption; the rest were
    # actually re-described, INCLUDING the frame on a later item — the raise did
    # not abort the run.
    assert source.frames[0].description == "a"
    assert source.frames[1].description == "new 1"
    assert store["2"].content.sources[0].frames[0].description == "new 0"
    # All-or-nothing stamping composes with per-frame failure: one failed frame
    # means the source is NOT stamped, so a later run retries it.
    assert source.caption_contract == ""


def test_a_partial_failure_does_not_stamp_the_contract(tmp_path: Path):
    """Stamping is all-or-nothing: a source with one un-redescribed frame is still
    stale, so the next run retries it instead of declaring it done."""
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()

    redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")
    assert store["1"].content.sources[0].caption_contract == ""


def test_dry_run_mutates_nothing(tmp_path: Path):
    item = _item("1", descriptions=("a",))
    store = {"1": item}
    _media(tmp_path, item)

    report = redescribe_frames(
        store, media_root=tmp_path, describe_fn=lambda path: "new", dry_run=True
    )

    source = store["1"].content.sources[0]
    assert source.frames[0].description == "a"
    assert source.caption_contract == ""
    # It still reports what it WOULD do, which is the whole point of --dry-run.
    assert report.videos_selected == 1
    assert report.frames_described == 1


def test_dry_run_calls_no_vision_model(tmp_path: Path):
    """`--dry-run` must be free. Calling the model and discarding the result would
    make a 2077-frame preview cost exactly as much as the real run."""
    item = _item("1")
    store = {"1": item}
    _media(tmp_path, item)
    calls: list[Path] = []

    redescribe_frames(
        store,
        media_root=tmp_path,
        describe_fn=lambda path: calls.append(path) or "new",
        dry_run=True,
    )
    assert calls == []


def test_dry_run_reports_a_missing_image_as_a_failure(tmp_path: Path):
    """A `--dry-run` preview must not OVERSTATE the work a real run would do: a
    frame whose PNG was already deleted can never be described, so the preview
    must already report it as a failure rather than as work still to be done —
    while still calling the vision model ZERO times (dry-run stays free).

    Checking `is_file()` costs nothing and touches no vision model, so there is
    no tension between "the preview is accurate" and "the preview is free".
    """
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()
    calls: list[Path] = []

    report = redescribe_frames(
        store,
        media_root=tmp_path,
        describe_fn=lambda path: calls.append(path) or "new",
        dry_run=True,
    )

    assert report.frames_failed == 1
    assert report.frames_described == 1
    assert calls == []


def test_item_ids_narrows_the_selection(tmp_path: Path):
    a, b = _item("1"), _item("2")
    store = {"1": a, "2": b}
    _media(tmp_path, a)
    _media(tmp_path, b)

    redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new", item_ids=["2"])
    assert store["1"].content.sources[0].frames[0].description == "old"
    assert store["2"].content.sources[0].frames[0].description == "new"


def test_a_changed_caption_bumps_fetched_at(tmp_path: Path):
    """The propagation trigger. New evidence must reach `enrich` → `video-digest`
    → `generate`, and `enrich._needs_reenrichment` keys on
    `content.fetched_at > enriched.enriched_at` — so the bump IS the wiring."""
    item = _item("1", descriptions=("old",))
    store = {"1": item}
    _media(tmp_path, item)
    before = item.content.fetched_at

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")

    assert store["1"].content.fetched_at > before
    assert report.items_repropagated == 1


def test_items_repropagated_counts_distinct_items_not_sources(tmp_path: Path):
    """PIN: `items_repropagated` counts ITEMS, not frame-bearing sources.

    `_frame_bearing_sources` returns a list, so one item CAN carry two
    frame-bearing `x_video` sources (e.g. two separate videos quoted/embedded in
    the same bookmark). `format_redescribe_summary` prints this counter as
    "N items" — if the engine incremented it once per changed SOURCE instead of
    once per changed ITEM, an item with two changing videos would be counted
    twice, and the summary would lie about how many items got new evidence.
    """
    frame_a = VideoFrame(timestamp=0.0, local_path="1/frames/a0.png", description="old a")
    frame_b = VideoFrame(timestamp=0.0, local_path="1/frames/b0.png", description="old b")
    source_a = ContentSourceSuccess(
        kind="x_video", url="https://x.com/a/status/1", text="video a", frames=[frame_a]
    )
    source_b = ContentSourceSuccess(
        kind="x_video", url="https://x.com/a/status/1", text="video b", frames=[frame_b]
    )
    item = Item(
        id="1",
        source="bookmark",
        url="https://x.com/a/status/1",
        author=Author(handle="alice", name="Alice"),
        text="tweet",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        content=Content(
            fetched_at=datetime(2020, 1, 1, tzinfo=timezone.utc), sources=[source_a, source_b]
        ),
    )
    store = {"1": item}
    for frame in (frame_a, frame_b):
        path = tmp_path / frame.local_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG")

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")

    # Both sources changed, but they live on the SAME item.
    assert report.items_repropagated == 1


def test_an_identical_caption_does_not_bump_fetched_at(tmp_path: Path):
    """The cost guard. A model that returns the same caption is not new evidence,
    and re-enriching 142 items on a no-op backfill is exactly the LLM spend this
    design exists to avoid. Note the contract IS still stamped — the source is
    now known-current even though nothing changed."""
    item = _item("1", descriptions=("same",))
    store = {"1": item}
    _media(tmp_path, item)
    before = item.content.fetched_at

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "same")

    assert store["1"].content.fetched_at == before
    assert report.items_repropagated == 0
    assert store["1"].content.sources[0].caption_contract == FRAME_CAPTION_CONTRACT


def test_dry_run_never_bumps_fetched_at(tmp_path: Path):
    item = _item("1", descriptions=("old",))
    store = {"1": item}
    _media(tmp_path, item)
    before = item.content.fetched_at

    redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new", dry_run=True)
    assert store["1"].content.fetched_at == before


def test_videos_current_is_counted_from_a_real_mixed_run(tmp_path: Path):
    """I2 (review): `report.videos_current = 0` left all 17 tests green — nothing
    exercised the counter substantiating the module's headline claim ("re-running
    on a current corpus costs zero vision calls", rendered as "N ya al día"). Build
    a store where a DIFFERENT answer was reachable: one stale source, one
    already-current source, and one source with no frames at all (never counted
    either way — a real corpus has audio-only videos)."""
    stale = _item("1", contract="")
    current = _item("2", contract=FRAME_CAPTION_CONTRACT)
    no_frames = _item("3", contract="", descriptions=())
    store = {"1": stale, "2": current, "3": no_frames}
    _media(tmp_path, stale)
    _media(tmp_path, current)

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")

    assert report.videos_current == 1
    assert report.videos_selected == 1


def test_summary_line_reports_the_counters():
    """I1 (review): the old assertion (`"3" in line and "140" in line and "44" in
    line and "1" in line`) is satisfied even if every counter is printed under the
    wrong label, or if `items_repropagated` is dropped from the output entirely
    (`"1"` is already a substring of `"140"`). Assert the EXACT formatted string,
    with every counter distinct and non-zero (including `items_repropagated`, left
    at 0 — and therefore unasserted — by the old fixture) so no counter's digits
    can hide inside another's."""
    report = RedescribeReport(
        videos_selected=3,
        videos_current=140,
        frames_described=44,
        frames_failed=7,
        items_repropagated=2,
    )
    line = format_redescribe_summary(report)
    assert line == (
        "Frames: 3 vídeos seleccionados, 140 ya al día. "
        "Captions: 44 regeneradas, 7 fallidas. "
        "Re-propagados: 2 items."
    )


def test_a_permanently_missing_frame_makes_the_source_a_tarpit(tmp_path: Path):
    """I3 (review): retry is per-SOURCE, not per-frame, and this is deliberate
    (see the module docstring + WHY-comment). PIN it as intentional: a second
    consecutive run, with the missing PNG still missing, re-describes the
    SURVIVING frame again (paying for it twice) and STILL does not stamp the
    contract — this is the tarpit the design consciously accepts rather than
    building a per-frame stamp (which would land inside `fetch._source_signature`'s
    FLAT deny-list and re-introduce the spurious ~142-item re-enrichment this
    design exists to avoid)."""
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()
    calls: list[Path] = []

    def _describe(path: Path) -> str:
        calls.append(path)
        return "new"

    redescribe_frames(store, media_root=tmp_path, describe_fn=_describe)
    redescribe_frames(store, media_root=tmp_path, describe_fn=_describe)

    source = store["1"].content.sources[0]
    # The surviving frame (index 1) was described on BOTH runs — paid for twice.
    assert calls == [tmp_path / "1/frames/1.png", tmp_path / "1/frames/1.png"]
    assert source.caption_contract == ""
    assert source.frames[0].description == "a"  # missing frame: never touched
    assert source.frames[1].description == "new"  # surviving frame: re-described


def test_redescribe_imports_no_network_or_subprocess_machinery():
    """M2 (review): the module's own docstring makes the STRONGEST offline claim
    in the repo ('NO network, NO X and NO ffmpeg'), and nothing pinned it. Mirrors
    the idiom in `test_vision.py`/`test_transcribe.py`/`test_video_frames.py`:
    guard the module's OWN source against heavy/ML/network machinery pulled in at
    import time.

    Deliberately does NOT forbid `from xbrain.vision import VisionFailed` (or
    `xbrain.vision` generally) — C1's fix imports it on purpose, to catch the
    exact exception `vision.describe_image` raises, and `vision.py` itself spawns
    no subprocess at import time (only when `describe_image` is called)."""
    import xbrain.redescribe as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import subprocess",
        "import socket",
        "import requests",
        "import urllib",
        "import playwright",
        "import ffmpeg",
        "import torch",
        "import mlx",
        "import cv2",
    ):
        assert forbidden not in source


# ---------------------------------------------------------------------------
# Re-review item 1: the `VisionFailed` catch is NARROW — `VisionNotFound`
# (missing/unconfigured `[vision].command`) is a CONFIG error and must ABORT
# the run, never become a per-frame failure.
# ---------------------------------------------------------------------------


def test_vision_not_found_propagates_instead_of_becoming_a_per_frame_failure(tmp_path: Path):
    """Re-review item 1: `VisionNotFound` is what an unconfigured or missing
    `[vision].command` raises (`vision.describe_image`). It must PROPAGATE out of
    `redescribe_frames`, not be swallowed as a per-frame failure — mirroring
    `digest._extract_described_slides`'s comment on the identical split. Widening
    the catch to `except VisionError` (its own base class) or `except Exception`
    both leave this red: a 2077-frame backfill against an unconfigured vision
    command would otherwise report `frames_failed=2077`, exit 0, and have changed
    nothing, instead of failing fast on what is plainly an operator error."""
    item = _item("1", descriptions=("a",))
    store = {"1": item}
    _media(tmp_path, item)

    def _describe(path: Path) -> str:
        raise VisionNotFound("no [vision].command configured")

    with pytest.raises(VisionNotFound):
        redescribe_frames(store, media_root=tmp_path, describe_fn=_describe)


# ---------------------------------------------------------------------------
# Re-review item 2: a total-failure circuit breaker.
# ---------------------------------------------------------------------------


def test_total_frame_failure_raises_runtime_error(tmp_path: Path):
    """Re-review item 2: 9 of 9 (here, 1 of 1) frames failing must not complete
    silently — the caller would read the returned report as success. Mirrors
    `media.py:245-256`'s total-failure short-circuit."""
    item = _item("1", descriptions=("a",))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()

    with pytest.raises(RuntimeError, match="All 1 frame re-caption attempts failed"):
        redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")


# ---------------------------------------------------------------------------
# #90 review M1: the circuit breaker must blame the right subsystem — a
# missing image file (pruned/unmounted `data/media/`, an ORDINARY per-worktree
# state) never even reaches `describe_fn`, so it must not be reported as a
# vision failure.
# ---------------------------------------------------------------------------


def test_total_failure_from_missing_images_blames_data_media_not_vision(tmp_path: Path):
    """M1: every PNG deleted, with a perfectly healthy `describe_fn` (it is never
    even called), must name `data/media/` — NOT `[vision].command`/the vision
    model — as the thing to check."""
    item = _item("1", descriptions=("a",))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()

    with pytest.raises(RuntimeError) as excinfo:
        redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")

    message = str(excinfo.value)
    assert "data/media" in message
    assert "[vision]" not in message
    assert "vision model" not in message


def test_total_failure_from_vision_errors_blames_the_vision_model(tmp_path: Path):
    """M1: every frame's IMAGE is present but `describe_fn` always raises
    `VisionFailed` — the message must point at `[vision].command`/the vision
    model, NOT at `data/media/` (nothing is missing there)."""
    item = _item("1", descriptions=("a",))
    store = {"1": item}
    _media(tmp_path, item)

    def _always_fails(path: Path) -> str:
        raise VisionFailed("vision command exited 1")

    with pytest.raises(RuntimeError) as excinfo:
        redescribe_frames(store, media_root=tmp_path, describe_fn=_always_fails)

    message = str(excinfo.value)
    assert "[vision].command" in message
    assert "data/media" not in message


def test_total_failure_with_mixed_causes_names_both(tmp_path: Path):
    """M1: one frame missing, the other's `describe_fn` call fails — a run that
    mixes both causes must name BOTH, not just whichever the implementation
    happens to check first."""
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()

    def _always_fails(path: Path) -> str:
        raise VisionFailed("vision command exited 1")

    with pytest.raises(RuntimeError) as excinfo:
        redescribe_frames(store, media_root=tmp_path, describe_fn=_always_fails)

    message = str(excinfo.value)
    assert "data/media" in message
    assert "[vision].command" in message


def test_partial_frame_failure_does_not_raise(tmp_path: Path):
    """A PARTIAL failure (some frames really described) is real progress, not a
    total failure — it must not raise, unlike the all-failed case above."""
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")
    assert report.frames_described == 1
    assert report.frames_failed == 1


def test_nothing_stale_does_not_raise(tmp_path: Path):
    """Zero attempted is not a failure: a run against an up-to-date corpus must
    not trip the total-failure breaker just because nothing was selected."""
    item = _item("1", contract=FRAME_CAPTION_CONTRACT)
    store = {"1": item}
    _media(tmp_path, item)

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")
    assert report.videos_selected == 0
    assert report.frames_described == 0
    assert report.frames_failed == 0


def test_dry_run_with_everything_failing_does_not_raise(tmp_path: Path):
    """REAL RUNS ONLY: a `--dry-run` preview must still report its counts rather
    than raise — a preview that fails is not a preview."""
    item = _item("1", descriptions=("a",))
    store = {"1": item}
    _media(tmp_path, item)
    (tmp_path / "1/frames/0.png").unlink()

    report = redescribe_frames(
        store, media_root=tmp_path, describe_fn=lambda path: "new", dry_run=True
    )
    assert report.frames_failed == 1
    assert report.frames_described == 0


# ---------------------------------------------------------------------------
# Re-review item 5: two pre-existing behaviours that are correct but were
# unasserted — each is now live via the CLI's `--ids` wiring.
# ---------------------------------------------------------------------------


def test_stale_video_sources_ignores_an_unknown_id_instead_of_raising():
    """Re-review item 5: `_warn_missing_ids` (cli.py) only WARNS about an unknown
    `--ids` entry — it never removes it from the list handed to
    `stale_video_sources`. Dropping the `if i in store` membership filter there
    would raise `KeyError` on the very first unknown id, turning a typo into a
    crash instead of a warning + "nothing found for it"."""
    store = {"1": _item("1", contract="")}
    result = stale_video_sources(store, item_ids=["1", "does-not-exist"])
    assert [item_id for item_id, _ in result] == ["1"]


def test_videos_current_is_scoped_to_the_item_ids_selection(tmp_path: Path):
    """Re-review item 5: `--ids 1` against a corpus that also has OTHER
    already-current videos must not count those others as "ya al día" — the
    counter is user-visible (`format_redescribe_summary`) and must be scoped to
    the SELECTION, not the whole store. Computing it over the whole store would
    make `--ids 1` print "1 ya al día" here (and "141 ya al día" on the real
    142-video corpus) instead of "0"."""
    stale = _item("1", contract="")
    current = _item("2", contract=FRAME_CAPTION_CONTRACT)
    store = {"1": stale, "2": current}
    _media(tmp_path, stale)
    _media(tmp_path, current)

    report = redescribe_frames(
        store, media_root=tmp_path, describe_fn=lambda path: "new", item_ids=["1"]
    )

    assert report.videos_current == 0
    assert report.videos_selected == 1


# ---------------------------------------------------------------------------
# Re-review item 6: the per-frame warning must name the failed frame — it is
# the only thing telling an operator WHICH frame failed in a 2077-frame run.
# ---------------------------------------------------------------------------


def test_missing_image_warning_names_the_frame_path(tmp_path: Path, caplog):
    # A second, surviving frame keeps this a PARTIAL failure — not a total one —
    # so the item-2 circuit breaker does not also fire here.
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    frame_path = tmp_path / "1/frames/0.png"
    frame_path.unlink()

    with caplog.at_level(logging.WARNING):
        redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")

    assert str(frame_path) in caplog.text


def test_vision_failure_warning_names_the_frame_path(tmp_path: Path, caplog):
    # A second, surviving frame keeps this a PARTIAL failure — not a total one —
    # so the item-2 circuit breaker does not also fire here.
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    frame_path = tmp_path / "1/frames/0.png"

    def _describe(path: Path) -> str:
        if path == frame_path:
            raise VisionFailed("vision command exited 1: corrupt frame")
        return "new"

    with caplog.at_level(logging.WARNING):
        redescribe_frames(store, media_root=tmp_path, describe_fn=_describe)

    assert str(frame_path) in caplog.text


# ---------------------------------------------------------------------------
# #90 review I1 + M2: an injectable `report` so a caller can read (and persist)
# partial progress even when the run raises before returning.
# ---------------------------------------------------------------------------


def test_redescribe_frames_populates_and_returns_the_passed_in_report(tmp_path: Path):
    """The `report` object the CALLER passed in is the SAME instance returned —
    not a fresh one the caller has to remember to swap in."""
    item = _item("1", descriptions=("a", "b"))
    store = {"1": item}
    _media(tmp_path, item)
    report = RedescribeReport()

    result = redescribe_frames(
        store, media_root=tmp_path, describe_fn=lambda path: "new", report=report
    )

    assert result is report
    assert report.frames_described == 2


def test_a_mid_run_exception_leaves_partial_progress_on_the_passed_in_report(tmp_path: Path):
    """PIN for #90 review I1: a `VisionNotFound` raised on the SECOND item must
    not erase the first item's already-completed, already-paid-for work from the
    caller's `report` — this is what lets the CLI's `finally` persist it.

    Measured on the real bug: a 2-item store where item `42` was fully
    re-described (2 real vision calls) and item `43` then raised
    `VisionNotFound` left `STORE UNCHANGED: True` because the caller never got
    ANY report back (the function raises before its final `return`). Passing a
    pre-created `report` in sidesteps that: it is mutated IN PLACE, frame by
    frame, so whatever landed before the raise is still there afterward.
    """
    first = _item("1", descriptions=("a", "b"))
    second = _item("2", descriptions=("c",))
    store = {"1": first, "2": second}
    _media(tmp_path, first)
    _media(tmp_path, second)

    def _describe(path: Path) -> str:
        if "2/frames" in str(path):
            raise VisionNotFound("vision command vanished")
        return "new"

    report = RedescribeReport()
    with pytest.raises(VisionNotFound):
        redescribe_frames(store, media_root=tmp_path, describe_fn=_describe, report=report)

    # Item 1's two frames were both really re-described BEFORE item 2 raised —
    # that work must still show up on the report the caller is holding.
    assert report.frames_described == 2
    assert [f.description for f in store["1"].content.sources[0].frames] == ["new", "new"]
    # Item 2 never got as far as `source.frames = new_frames` — it kept its
    # original caption, which is correct (it was never actually re-described).
    assert store["2"].content.sources[0].frames[0].description == "c"


def test_report_defaults_to_a_fresh_instance_when_omitted(tmp_path: Path):
    """Backward compatibility: every existing caller (this module's own test
    suite included) that does not pass `report=` must keep working exactly as
    before."""
    item = _item("1", descriptions=("a",))
    store = {"1": item}
    _media(tmp_path, item)

    report = redescribe_frames(store, media_root=tmp_path, describe_fn=lambda path: "new")
    assert report.frames_described == 1


def test_report_counters_all_accumulate_across_repeated_calls_sharing_one_report(
    tmp_path: Path,
):
    """#90 re-review item 3: reusing ONE `report=` across two engine calls must
    accumulate EVERY counter, not just the three that already did
    (`videos_selected`, `frames_described`, `frames_failed`). `videos_current`
    and `items_repropagated` used to be plain ASSIGNMENTS — the second call's
    count silently REPLACED the first call's instead of adding to it, an
    inconsistent contract on an otherwise-uniform seam.

    Two independent batches (distinct stores, one shared `report`), each
    contributing to every counter: a stale item that changes cleanly, an
    already-current item (`videos_current`), and — in the second batch — one
    frame missing on disk (a real `frames_failed`), so nothing here is 0 in
    a way that could hide a broken counter.
    """
    first_stale = _item("1", contract="", descriptions=("old", "old"))
    first_current = _item("cur1", contract=FRAME_CAPTION_CONTRACT, descriptions=("x",))
    first_store = {"1": first_stale, "cur1": first_current}
    _media(tmp_path, first_stale)
    _media(tmp_path, first_current)

    report = RedescribeReport()
    redescribe_frames(
        first_store, media_root=tmp_path, describe_fn=lambda path: "new", report=report
    )

    assert (
        report.videos_selected,
        report.videos_current,
        report.frames_described,
        report.frames_failed,
        report.items_repropagated,
    ) == (1, 1, 2, 0, 1)

    second_stale = _item("2", contract="", descriptions=("old", "old"))
    second_current = _item("cur2", contract=FRAME_CAPTION_CONTRACT, descriptions=("x",))
    second_store = {"2": second_stale, "cur2": second_current}
    _media(tmp_path, second_stale)
    _media(tmp_path, second_current)
    # Delete one of item 2's frame images so this batch also feeds `frames_failed`.
    (tmp_path / second_stale.content.sources[0].frames[0].local_path).unlink()

    redescribe_frames(
        second_store, media_root=tmp_path, describe_fn=lambda path: "new", report=report
    )

    # Every counter is now the SUM of both batches — never just the second's
    # (which alone would read (1, 1, 1, 1, 1), the exact shape a regressed
    # assignment on `videos_current`/`items_repropagated` would produce).
    assert (
        report.videos_selected,
        report.videos_current,
        report.frames_described,
        report.frames_failed,
        report.items_repropagated,
    ) == (2, 2, 3, 1, 2)


# ---------------------------------------------------------------------------
# #90 review I3: `--dry-run`'s summary wording must be predictive, not
# past-tense, and must be textually distinct from a real run's.
# ---------------------------------------------------------------------------


def test_dry_run_summary_wording_is_predictive_not_past_tense():
    """The real-run summary says "regeneradas"/"fallidas" — PAST TENSE, things
    that HAPPENED. A dry run happened nothing, so its wording must not claim
    otherwise (#90 review I3)."""
    report = RedescribeReport(
        videos_selected=1, videos_current=0, frames_described=2, frames_failed=0
    )
    dry_line = format_redescribe_summary(report, dry_run=True)
    real_line = format_redescribe_summary(report, dry_run=False)

    assert dry_line != real_line
    assert "regeneradas" not in dry_line
    assert "fallidas" not in dry_line
    assert "regeneradas" in real_line


def test_dry_run_summary_flags_repropagated_as_not_a_prediction():
    """`Re-propagados` is explicitly NOT a prediction (the module docstring says
    so) — that caveat must reach the OUTPUT itself, not just a docstring the
    operator never reads (#90 review I3)."""
    report = RedescribeReport(videos_selected=1, frames_described=2)
    dry_line = format_redescribe_summary(report, dry_run=True)
    assert "predic" in dry_line.lower()
