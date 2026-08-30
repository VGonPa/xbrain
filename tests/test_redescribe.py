"""Tests for `xbrain.redescribe` — offline re-captioning of stored key frames.

The engine reads pixels from `data/media/<id>/frames/` and NEVER touches the
network, X or ffmpeg. Every test injects a fake `describe_fn`, so no vision model
runs. Staleness is keyed on the source's `caption_contract`, so a corpus already
at the current contract costs nothing to re-run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from xbrain.models import (
    FRAME_CAPTION_CONTRACT,
    Author,
    Content,
    ContentSourceSuccess,
    Item,
    VideoFrame,
)
from xbrain.redescribe import (
    RedescribeReport,
    format_redescribe_summary,
    redescribe_frames,
    stale_video_sources,
)


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
    assert report.videos_redescribed == 1
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
    assert report.videos_redescribed == 1
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


def test_summary_line_reports_the_counters():
    report = RedescribeReport(
        videos_redescribed=3, videos_current=140, frames_described=44, frames_failed=1
    )
    line = format_redescribe_summary(report)
    assert "3" in line and "140" in line and "44" in line and "1" in line
