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

STAMPING IS ALL-OR-NOTHING, PER SOURCE — AND THAT IS A DELIBERATE TARPIT. A
source is stamped only when every one of its frames was re-described. A partial
run leaves the old stamp, so the next run retries the WHOLE SOURCE — re-paying
for frames that were already correctly re-described — not just the frames it
missed. A source with a PERMANENTLY missing frame image (deleted, never
downloaded, media root pruned) therefore never converges: every future run
re-describes its surviving frames again and never stamps.

THE TRUE COST IS NOT JUST RE-PAYING VISION CALLS (re-review, #90). A real
vision model is NON-deterministic: the surviving frames come back reworded
slightly differently on every run even though the pixels never changed. That
reads as a genuine caption change (`changed` is True), so `content.fetched_at`
is bumped and downstream `enrich` (and `video-digest`) re-run for that item —
not once, but on EVERY future run, forever, for as long as the tarpit persists.
This module's OWN test suite hides that cost: every `describe_fn` fake in
`tests/test_redescribe.py` is deterministic (same path in, same string out), so
`items_repropagated` drops to 0 after the first run there — that is an artifact
of the test fake, not evidence the real tarpit is cheap. Do not read the green
suite as proof of the cheaper story.

This is accepted on purpose, not an oversight — the alternatives are worse:
  - a per-frame stamp would have to live on `VideoFrame`, which sits INSIDE
    `fetch._source_signature`'s fingerprint. That deny-list is FLAT (it excludes
    top-level field names on the source, e.g. `caption_contract` itself) and does
    not descend into nested models, so a per-frame stamp would read as a material
    content change on every run and re-introduce the spurious ~142-item
    re-enrichment this whole design exists to avoid;
  - stamping the source as done despite the failed frame loses the retry
    entirely — the permanently-missing frame's stale caption would never be
    revisited even if the file later reappears;
  - the module has no way to distinguish a PERMANENTLY-gone file from a
    TEMPORARILY-unmounted media root, so it cannot special-case the former.

The `describe_fn` seam (`vision.describe_image` pre-bound to the configured
command/model/language) is injected, so tests run offline against a fake and no
vision model is ever required by the suite. This module imports `VisionFailed`
from `xbrain.vision` — a deliberate coupling, matching the sibling producer
`digest.py`, purely to catch that one exception type; `vision.py` spawns no
subprocess and touches no network at IMPORT time (only when `describe_image` is
actually called), so this stays offline.
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
from xbrain.vision import VisionFailed

logger = logging.getLogger(__name__)

DescribeFn = Callable[[Path], str]


@dataclass
class RedescribeReport:
    """Structured outcome of a `redescribe_frames` run (drives the CLI summary).

    `videos_selected` counts sources SELECTED for re-captioning (stale, or every
    frame-bearing source under `force`) — NOT how many were actually
    re-described; a per-frame `VisionFailed` can leave some (or, with the media
    root gone, all) of a selected source's frames un-re-described while it still
    counts as selected (#90 review M4 — the prior name `videos_redescribed`
    claimed a past-tense outcome the field never verified, so a run against a
    missing media root read as "N vídeos re-descritos … 0 regeneradas",
    self-contradictory). `frames_described` / `frames_failed` are what actually
    happened, frame-granular — read those for the real story.
    `videos_current` counts sources skipped because they already carry the
    current contract; `items_repropagated` counts items whose `content.fetched_at`
    was bumped because at least one caption actually changed.
    """

    videos_selected: int = 0
    videos_current: int = 0
    frames_described: int = 0
    frames_failed: int = 0
    items_repropagated: int = 0
    # Split of `frames_failed` by CAUSE (#90 review M1) — used only by
    # `_raise_on_total_failure` to name the right subsystem in its error
    # message. A missing image file never even reaches `describe_fn`, so
    # lumping it in with a real `VisionFailed` sent operators chasing
    # `[vision].command` for what was actually a pruned/unmounted `data/media/`
    # (gitignored and per-worktree — an ORDINARY state, not exotic).
    frames_failed_missing_image: int = 0
    frames_failed_vision_error: int = 0


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
    report: RedescribeReport | None = None,
) -> RedescribeReport:
    """Re-caption every stale frame from its stored image; return the outcome.

    Mutates `store` in place (the CALLER persists and snapshots). `dry_run`
    reports what would happen and calls `describe_fn` ZERO times — a preview of a
    2000-frame backfill must not cost what the backfill costs. `dry_run` CANNOT
    predict `items_repropagated`: whether a caption actually changes is only known
    by calling the model, and a dry run never does — so the field stays at its
    default (0) on every dry-run report, and is not a prediction of what the real
    run will bump (#90 review M3).

    `report`, when passed, is the object POPULATED in place and returned — the
    caller then holds the SAME instance the engine is filling in, frame by
    frame (#90 review I1). Without this seam, a mid-run exception (the
    total-failure `RuntimeError` below, a propagating `VisionNotFound`, an
    `OSError`, a Ctrl-C) would discard every already-paid-for vision call: the
    caller's own `RedescribeReport()` never gets assigned because the `return`
    that would hand it over never executes. Passing a pre-created `report` in
    lets a `try/finally` in the caller read (and persist) whatever landed before
    the exception, even though the exception still propagates unmodified. A
    fresh `RedescribeReport()` is created when `report` is omitted, so every
    existing caller (including this module's own test suite) is unaffected.

    A missing image, or a `describe_fn` call that raises `vision.VisionFailed`
    (non-zero exit, timeout, or an exit-0-but-empty-output run), is a PER-FRAME
    failure: it is logged, the frame keeps its old caption, and the run continues
    — to the source's other frames, and to the rest of the store. Retry is
    per-SOURCE, not per-frame: that frame's WHOLE source is then left unstamped,
    so a later run retries every one of the source's frames, not just the one(s)
    it missed (see the module WHY-comment for why this is deliberate, and its
    tarpit cost when a frame image is permanently gone).

    A TOTAL failure — every attempted frame failed — is a different case from a
    per-frame one and is NOT swallowed: see `_raise_on_total_failure`, called
    unconditionally at the end of a real (non-dry) run.
    """
    if report is None:
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
        report.videos_selected += 1
        if dry_run:
            # A preview must not OVERSTATE the work a real run would do: stat
            # each frame (free, no vision model) so a corpus with deleted PNGs
            # previews the same described/failed split the real run would
            # record, instead of promising work the real run cannot perform.
            described, failed = _preview_source(source, media_root)
            report.frames_described += described
            report.frames_failed += failed
            # A preview can only ever detect the missing-file cause (it never
            # calls `describe_fn`, so a `VisionFailed` can never surface here).
            report.frames_failed_missing_image += failed
            continue
        _apply_real_redescription(
            item_id,
            source,
            media_root=media_root,
            describe_fn=describe_fn,
            report=report,
            store=store,
            now=now,
            repropagated_items=repropagated_items,
        )
    report.items_repropagated = len(repropagated_items)
    _raise_on_total_failure(report, dry_run=dry_run)
    return report


def _apply_real_redescription(
    item_id: str,
    source: ContentSourceSuccess,
    *,
    media_root: Path,
    describe_fn: DescribeFn,
    report: RedescribeReport,
    store: dict[str, Item],
    now: datetime,
    repropagated_items: set[str],
) -> None:
    """Re-caption ONE stale source's frames for a REAL (non-dry) run.

    Updates `report`'s counters and, when a caption actually changed, the
    item's `content.fetched_at` + `repropagated_items` — extracted out of
    `redescribe_frames`'s main loop to keep that loop's own branching at a
    glance (radon: this split is what keeps `redescribe_frames` from
    compounding a grade with every future counter it needs to track).
    """
    new_frames, described, failed, missing, vision_errors = _redescribe_source(
        source, media_root, describe_fn
    )
    report.frames_described += described
    report.frames_failed += failed
    report.frames_failed_missing_image += missing
    report.frames_failed_vision_error += vision_errors
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


def _raise_on_total_failure(report: RedescribeReport, *, dry_run: bool) -> None:
    """Circuit breaker (#90 re-review item 2): raise when EVERY attempted frame
    failed, mirroring `media.py`'s total-failure short-circuit (`media.py:245-256`)
    exactly — same shape, same "nothing landed" test, same operator-facing tone.

    Without this, 9 of 9 frames failing completes silently and returns a report
    the caller reads as success. That is not hypothetical: `VisionFailed` also
    covers a TIMEOUT (`vision._DEFAULT_TIMEOUT_SECONDS = 300`), so a wedged vision
    model now costs 300s × N frames before this raises — on the real 2077-frame
    corpus, ~173 hours reported as a successful run instead of failing fast.

    REAL RUNS ONLY: `dry_run` must still report its counts rather than raise — a
    preview that raises is not a preview. A PARTIAL failure (some frames really
    described) is real progress, not a total failure, so it does not raise
    either — only `total_attempted > 0 and frames_described == 0` does. Zero
    attempted (nothing stale) is not a failure. `RuntimeError` is already in
    `cli._OPERATOR_ERRORS`, so this surfaces as a clean exit-1, not a traceback.
    """
    total_attempted = report.frames_described + report.frames_failed
    if dry_run or total_attempted == 0 or report.frames_described > 0:
        return
    raise RuntimeError(_total_failure_message(report, total_attempted))


def _total_failure_message(report: RedescribeReport, total_attempted: int) -> str:
    """Name the right subsystem for a total failure (#90 review M1).

    A missing image file (a pruned or unmounted `data/media/` — gitignored and
    per-worktree, so "items.json present, media absent" is an ORDINARY state,
    not an exotic one) never even reaches `describe_fn`. Blaming
    `[vision].command`/the vision model for it — the OLD, single message below —
    sends the operator chasing the wrong subsystem: measured, deleting every PNG
    against a perfectly healthy `describe_fn` raised exactly that message.
    Distinguishes the two causes (and names both when a run somehow mixes them),
    while keeping `media.py:245-256`'s shape and tone: "All N … failed; check
    X, and the per-frame warnings above."
    """
    missing = report.frames_failed_missing_image
    vision_errors = report.frames_failed_vision_error
    if vision_errors == 0:
        check = (
            "check data/media/ — the image files are missing (a pruned or "
            "unmounted media root looks exactly like this)"
        )
    elif missing == 0:
        check = "check [vision].command / the vision model"
    else:
        check = (
            f"check data/media/ ({missing} missing image file(s)) and "
            f"[vision].command / the vision model ({vision_errors} vision call failure(s))"
        )
    return (
        f"All {total_attempted} frame re-caption attempts failed; {check}, "
        "and the per-frame warnings above."
    )


def _preview_source(source: ContentSourceSuccess, media_root: Path) -> tuple[int, int]:
    """A best-effort PREDICTION of `_redescribe_source`'s described/failed split
    WITHOUT describing — not a guaranteed match (#90 re-review item 3).

    Stats each frame's file — free, and calls `describe_fn` zero times — so a
    `--dry-run` preview reports a frame whose PNG is already gone as a FAILURE,
    the same way the real run would, rather than as work still to be done. It
    CANNOT see a frame whose PNG EXISTS but is corrupt/unreadable: that previews
    here as "described", yet the real run can still fail it with `VisionFailed`
    once `describe_fn` actually opens the bytes — a free stat can only confirm a
    file is present, not that its content is decodable.
    """
    failed = sum(1 for frame in source.frames if not (media_root / frame.local_path).is_file())
    return len(source.frames) - failed, failed


def _redescribe_source(
    source: ContentSourceSuccess, media_root: Path, describe_fn: DescribeFn
) -> tuple[list[VideoFrame], int, int, int, int]:
    """Re-caption one source's frames; return (frames, described, failed,
    missing_image, vision_error) — the last two split `failed` by CAUSE
    (#90 review M1) so the caller's circuit breaker can name the right
    subsystem instead of always blaming the vision model.

    Two DISTINCT per-frame failure modes are handled the same way — logged, the
    frame keeps its old caption, `failed` incremented, the loop continues to the
    source's remaining frames: a missing/unreadable image (never reaches
    `describe_fn`), and a `describe_fn` call that raises `VisionFailed` — which
    `vision.describe_image` does on a non-zero exit, a timeout, AND on an
    exit-0-but-empty-output run (`vision.py`'s "a slide's content would be lost"
    guard). Both are exactly the corrupt/unreadable-frame case a multi-thousand-
    frame backfill will meet, and neither may abort the whole run (#90 review C1)
    — `redescribe_frames`'s all-or-nothing stamping (per SOURCE, not per run)
    already means `failed > 0` leaves this source unstamped for a later retry, so
    the two mechanisms compose without double-bookkeeping.

    The catch is deliberately `VisionFailed`, NOT its base `VisionError` (and
    NOT bare `Exception`) — `VisionNotFound` (an unconfigured / missing
    `[vision].command`) is a global CONFIG error, not a per-frame data problem,
    exactly like `digest._extract_described_slides`'s comment on the same split:
    it must ABORT the run, not become a spurious `frames_failed` count (#90
    re-review item 1). Widening this catch would turn a 2077-frame backfill
    against an unconfigured vision command into `frames_failed=2077`, exit 0,
    having changed nothing — instead of failing fast on an operator error.
    """
    frames: list[VideoFrame] = []
    described = failed = missing_image = vision_error = 0
    for frame in source.frames:
        path = media_root / frame.local_path
        if not path.is_file():
            logger.warning(
                "redescribe-frames: image missing for %s — keeping the old caption", path
            )
            frames.append(frame)
            failed += 1
            missing_image += 1
            continue
        try:
            description = describe_fn(path)
        except VisionFailed as exc:
            logger.warning(
                "redescribe-frames: vision call failed for %s (%s) — keeping the old caption",
                path,
                exc,
            )
            frames.append(frame)
            failed += 1
            vision_error += 1
            continue
        # `model_copy(update=...)` bypasses pydantic validation — a conscious,
        # NOT-built defence-in-depth gap: `describe_fn` is typed `Callable[[Path],
        # str]`, and the real `vision.describe_image` always returns a stripped
        # non-empty `str` (empty output raises `VisionFailed`, caught above), so
        # an invalid `description` reaching this line is unreachable via the real
        # implementation — only a misbehaving injected fake could trigger it
        # (#90 review M5).
        frames.append(frame.model_copy(update={"description": description}))
        described += 1
    return frames, described, failed, missing_image, vision_error


def format_redescribe_summary(report: RedescribeReport, *, dry_run: bool = False) -> str:
    """One-line human SUMMARY of a re-description run.

    "vídeos seleccionados" (not "re-descritos"): `videos_selected` counts sources
    SELECTED for re-captioning, not sources actually finished — with the media
    root gone, a run would otherwise print the self-contradictory "N vídeos
    re-descritos … 0 regeneradas" (#90 review M4). `Captions:` carries the real
    described/failed story.

    `dry_run=True` renders a DIFFERENT, predictive wording (#90 review I3): the
    real-run string above is past tense ("regeneradas", "fallidas" — things that
    HAPPENED), and a `--dry-run` report reusing it verbatim is textually
    indistinguishable from a real run's — plus it lies grammatically, since a
    preview regenerates zero captions. The dry-run string below uses future/
    conditional wording ("se regenerarían", "fallarían") and spells out, IN the
    output itself (not just a docstring the operator never sees), that
    `Re-propagados` is not a prediction at all — `redescribe_frames`'s own
    docstring already says a dry run cannot know whether a caption will
    actually CHANGE, only whether it can be attempted.
    """
    if dry_run:
        return (
            f"Frames: {report.videos_selected} vídeos seleccionados, "
            f"{report.videos_current} ya al día. "
            f"Captions: {report.frames_described} se regenerarían, "
            f"{report.frames_failed} fallarían. "
            "Re-propagados: no es una predicción — sólo se sabe corriendo el "
            "modelo de verdad (sin --dry-run)."
        )
    return (
        f"Frames: {report.videos_selected} vídeos seleccionados, "
        f"{report.videos_current} ya al día. "
        f"Captions: {report.frames_described} regeneradas, "
        f"{report.frames_failed} fallidas. "
        f"Re-propagados: {report.items_repropagated} items."
    )
