"""Re-caption key frames already on disk, offline (#90).

WHY THIS EXISTS. `digest-video --frames` is the only producer of frame captions,
and it re-fetches the video from X to run. When the caption CONTRACT changes —
as it did in #90, where captions were translating on-screen text instead of
transcribing it — every stored caption is stale, but the pixels are not: they sit
under `data/media/<id>/frames/`. This module re-describes those bytes with the
current rubric and touches NO network, NO X and NO ffmpeg. On the corpus that
motivated it, that is 2077 frames across 142 videos whose videos may no longer
be downloadable at all.

STALENESS IS A CONTRACT COMPARISON, NOT A TIMESTAMP. A source carries the
`caption_contract` its captions were produced under (see `models.FRAME_CAPTION_
CONTRACT`). Anything other than the current value is stale, so re-running the
command on an up-to-date corpus costs zero vision calls. `force=True` ignores it.

STAMPING IS ALL-OR-NOTHING. A source is stamped only when every one of its frames
was re-described. A partial run leaves the old stamp, so the next run retries the
frames it missed rather than declaring a half-fixed video done.

The `describe_fn` seam (`vision.describe_image` pre-bound to the configured
command/model/language) is injected, so tests run offline against a fake and no
vision model is ever required by the suite.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from xbrain.models import (
    FRAME_CAPTION_CONTRACT,
    ContentSourceSuccess,
    Item,
    VideoFrame,
)

logger = logging.getLogger(__name__)

DescribeFn = Callable[[Path], str]


@dataclass
class RedescribeReport:
    """Structured outcome of a `redescribe_frames` run (drives the CLI summary).

    `videos_redescribed` counts sources whose frames were (or, on a dry run, would
    be) re-captioned; `videos_current` counts sources skipped because they already
    carry the current contract; `frames_described` / `frames_failed` are frame-
    granular; `items_repropagated` counts items whose `content.fetched_at` was
    bumped because at least one caption actually changed.
    """

    videos_redescribed: int = 0
    videos_current: int = 0
    frames_described: int = 0
    frames_failed: int = 0
    items_repropagated: int = 0


def _frame_bearing_sources(item: Item) -> list[ContentSourceSuccess]:
    """Every `x_video` success on the item that actually carries key frames.

    Narrowing with `isinstance` inside the comprehension (rather than a boolean
    predicate plus a later `assert`) keeps mypy happy without an `assert` in
    production code, which bandit flags as B101.
    """
    if item.content is None:
        return []
    return [
        source
        for source in item.content.sources
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_video" and source.frames
    ]


def stale_video_sources(
    store: dict[str, Item],
    item_ids: list[str] | None = None,
    *,
    force: bool = False,
) -> list[tuple[str, ContentSourceSuccess]]:
    """Return `(item_id, source)` for every frame-bearing source needing re-caption.

    A source is stale when its `caption_contract` differs from the current one —
    `""` (produced before #90) included. `force` selects every frame-bearing
    source regardless. `item_ids` narrows the walk; `None` walks the whole store.
    """
    selected = store if item_ids is None else {i: store[i] for i in item_ids if i in store}
    stale: list[tuple[str, ContentSourceSuccess]] = []
    for item_id, item in selected.items():
        for source in _frame_bearing_sources(item):
            if force or source.caption_contract != FRAME_CAPTION_CONTRACT:
                stale.append((item_id, source))
    return stale


def redescribe_frames(
    store: dict[str, Item],
    *,
    media_root: Path,
    describe_fn: DescribeFn,
    item_ids: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> RedescribeReport:
    """Re-caption every stale frame from its stored image; return the outcome.

    Mutates `store` in place (the CALLER persists and snapshots). `dry_run`
    reports what would happen and calls `describe_fn` ZERO times — a preview of a
    2000-frame backfill must not cost what the backfill costs.

    A missing or unreadable image is a PER-FRAME failure: it is logged, the frame
    keeps its old caption, and the run continues. That frame's source is then left
    unstamped, so a later run retries it.
    """
    report = RedescribeReport()
    now = datetime.now(timezone.utc)
    all_frame_sources = stale_video_sources(store, item_ids, force=True)
    stale = stale_video_sources(store, item_ids, force=force)
    stale_keys = {(item_id, id(source)) for item_id, source in stale}
    report.videos_current = sum(
        1 for item_id, source in all_frame_sources if (item_id, id(source)) not in stale_keys
    )

    # Items whose `content.fetched_at` was actually bumped, so a real re-caption
    # of TWO frame-bearing sources on the SAME item (one bookmark embedding two
    # videos) is reported as ONE re-propagated item — `format_redescribe_summary`
    # prints this counter as "N items", and counting per source would double-
    # count it, lying about how many items received new evidence (#90 review).
    repropagated_items: set[str] = set()

    for item_id, source in stale:
        report.videos_redescribed += 1
        if dry_run:
            # A preview must not OVERSTATE the work a real run would do: stat
            # each frame (free, no vision model) so a corpus with deleted PNGs
            # previews the same described/failed split the real run would
            # record, instead of promising work the real run cannot perform.
            described, failed = _preview_source(source, media_root)
            report.frames_described += described
            report.frames_failed += failed
            continue
        new_frames, described, failed = _redescribe_source(source, media_root, describe_fn)
        report.frames_described += described
        report.frames_failed += failed
        changed = [f.description for f in new_frames] != [f.description for f in source.frames]
        source.frames = new_frames
        # All-or-nothing: a source with an un-redescribed frame stays stale.
        if failed == 0:
            source.caption_contract = FRAME_CAPTION_CONTRACT
        if changed:
            item = store[item_id]
            if item.content is not None:
                # The SAME re-enrichment trigger `digest.attach_transcript` uses:
                # new evidence must reach enrich → video-digest → generate. Bumped
                # only on a real caption change, so a no-op backfill spends nothing.
                item.content.fetched_at = now
                repropagated_items.add(item_id)
    report.items_repropagated = len(repropagated_items)
    return report


def _preview_source(source: ContentSourceSuccess, media_root: Path) -> tuple[int, int]:
    """Predict `_redescribe_source`'s described/failed split WITHOUT describing.

    Stats each frame's file — free, and calls `describe_fn` zero times — so a
    `--dry-run` preview reports a frame whose PNG is already gone as a FAILURE,
    the same way the real run would, rather than as work still to be done.
    """
    failed = sum(1 for frame in source.frames if not (media_root / frame.local_path).is_file())
    return len(source.frames) - failed, failed


def _redescribe_source(
    source: ContentSourceSuccess, media_root: Path, describe_fn: DescribeFn
) -> tuple[list[VideoFrame], int, int]:
    """Re-caption one source's frames; return (frames, described, failed)."""
    frames: list[VideoFrame] = []
    described = failed = 0
    for frame in source.frames:
        path = media_root / frame.local_path
        if not path.is_file():
            logger.warning(
                "redescribe-frames: image missing for %s — keeping the old caption", path
            )
            frames.append(frame)
            failed += 1
            continue
        frames.append(frame.model_copy(update={"description": describe_fn(path)}))
        described += 1
    return frames, described, failed


def format_redescribe_summary(report: RedescribeReport) -> str:
    """One-line human SUMMARY of a re-description run."""
    return (
        f"Frames: {report.videos_redescribed} vídeos re-descritos, "
        f"{report.videos_current} ya al día. "
        f"Captions: {report.frames_described} regeneradas, "
        f"{report.frames_failed} fallidas. "
        f"Re-propagados: {report.items_repropagated} items."
    )
