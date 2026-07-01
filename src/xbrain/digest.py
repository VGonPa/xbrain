"""`digest-video` orchestration — transcript as an `x_video` content source (#44).

Turns bookmarked videos into readable, topic-linkable notes by MANUFACTURING
text: for each selected video it does an **ephemeral** fetch (reusing PR1's
`video_fetch.fetch_videos`), shells out to the **external** transcriber
(`xbrain.transcribe`), attaches the transcript to the item as a
`ContentSourceSuccess(kind="x_video")`, and **discards** the bytes. Everything
downstream (enrich → topics → generate) is xbrain's existing pipeline, reused
unchanged.

Two invariants carry the design:

- **Dedup by video identity.** The full mp4 URL is unstable (`?tag=` + rotating
  signing/filename), so we key on the stable id parsed from the URL *path*
  (`amplify_video/<id>` / `ext_tw_video/<id>` / `tweet_video/<id>`). N bookmarks
  of the same video are fetched + transcribed **once**; every referencing item
  gets the same transcript source.
- **Ephemeral, one video at a time.** Each video is fetched into a temp dir,
  transcribed, then its bytes are deleted immediately — and the whole temp dir is
  removed even if transcription raises. Never more than one video on disk; the
  ~140 GB corpus never lands in the store.

This module mutates the in-memory `store` in place and returns a `DigestReport`;
persisting (with the destructive auto-snapshot) is the CLI's job. The per-group
work returns a small `_GroupOutcome` and the run sums them into the report once,
so the tally lives in a single place.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from xbrain.models import Content, ContentSource, ContentSourceSuccess, Item, VideoFrame
from xbrain.transcribe import Transcript, TranscriberFailed, transcribe_media
from xbrain.video_fetch import FetchReport, _select_entry, fetch_videos
from xbrain.video_frames import (
    FrameExtractionFailed,
    KeyFrame,
    classify_visual,
)
from xbrain.vision import VisionFailed

logger = logging.getLogger(__name__)

# A stable per-video identity parsed from the mp4 URL path (NOT the full URL).
VideoKey = str

# The X CDN media categories whose next path segment is the stable video id.
_VIDEO_CATEGORIES = ("amplify_video", "ext_tw_video", "tweet_video")

# `fetch_videos`-shaped callable (store, ids, dest_dir) -> FetchReport, and the
# transcriber (path -> Transcript). Both are injectable so tests run offline and
# the CLI can bind config (session / command / model / language).
FetchFn = Callable[[dict[str, Item], list[str], Path], FetchReport]
TranscribeFn = Callable[[Path], Transcript]

# The visual-layer collaborators (#44 PR4). `extract_fn` is `extract_key_frames`
# pre-bound with threshold/max_frames; `describe_fn` is `vision.describe_image`
# pre-bound with the `[vision]` command/model; `classify_fn` is `classify_visual`.
# All injectable so tests run offline (no real ffmpeg / vision).
ExtractFn = Callable[[Path], list[KeyFrame]]
DescribeFn = Callable[[Path], str]
ClassifyFn = Callable[[list[KeyFrame]], str]


@dataclass(frozen=True)
class VisualConfig:
    """The opt-in `--frames` visual-layer configuration (#44 PR4).

    Its PRESENCE on `digest_videos` enables the layer; `None` (the default) leaves
    the audio-only path byte-for-byte unchanged. `media_root` is where kept slides
    are persisted (`<media_root>/<item-id>/frames/<n>.png`) so `generate` mirrors
    them into the vault's `_media/` tree and embeds them like downloaded photos.
    """

    media_root: Path
    extract_fn: ExtractFn
    describe_fn: DescribeFn
    classify_fn: ClassifyFn = classify_visual


@dataclass(frozen=True)
class _DescribedSlide:
    """A kept key frame + its EXTERNAL-vision description, before per-item persist."""

    timestamp: float
    path: Path
    description: str


@dataclass
class _VisualResult:
    """The per-video outcome of the visual layer.

    `classification` is one of `disabled` (no `--frames`), `slides` (kept +
    described), `talking_head` (skipped + logged), or `skipped` (extraction /
    vision failed, per-video — logged, layer dropped). `slides` is non-empty only
    for `classification == "slides"`.
    """

    slides: list[_DescribedSlide] = field(default_factory=list)
    classification: str = "disabled"


@dataclass
class _MediaAnalysis:
    """The transcript + visual result for one fetched video, before attach."""

    transcript: Transcript
    visual: _VisualResult


@dataclass
class DigestReport:
    """Structured outcome of a `digest_videos` run (drives the CLI summary).

    Item-granular counters: `transcribed` (got a with-speech source), `no_speech`
    (got a `has_speech=False` marker), `already` (skipped — already carried an
    `x_video` source), `failed` (its video's fetch/transcribe failed),
    `skipped_no_video` (in the store but no fetchable mp4), `skipped_unknown` (id
    absent from the store). `videos_transcribed` is the distinct videos that
    actually produced a transcript this run; `groups` is the dedup grouping so the
    summary can report "N items ← M videos".
    """

    transcribed: int = 0
    no_speech: int = 0
    already: int = 0
    failed: int = 0
    skipped_no_video: int = 0
    skipped_unknown: int = 0
    videos_transcribed: int = 0
    # Visual layer (`--frames`, #44 PR4): distinct videos whose slides were
    # extracted + embedded, and distinct videos skipped as talking-head (both 0 on
    # a non-`--frames` run).
    visual_slides: int = 0
    visual_skipped: int = 0
    groups: dict[VideoKey, list[str]] = field(default_factory=dict)

    @property
    def total_items(self) -> int:
        """N — the items across every dedup group (the fetchable-video items)."""
        return sum(len(ids) for ids in self.groups.values())

    @property
    def video_count(self) -> int:
        """M — the distinct videos the selection resolved to."""
        return len(self.groups)

    @property
    def changed(self) -> int:
        """Items that received an `x_video` source this run — drives whether the
        CLI takes a snapshot + rewrites `items.json`."""
        return self.transcribed + self.no_speech


@dataclass
class _GroupOutcome:
    """The per-video-group result, summed into the `DigestReport` once.

    Keeping the tally out of the per-group helpers (they only decide + return)
    means the counters are combined in exactly one place — no counter is bumped
    across three functions.
    """

    transcribed: int = 0
    no_speech: int = 0
    already: int = 0
    failed: int = 0
    did_transcribe: bool = False
    visual_slides: bool = False
    visual_skipped: bool = False


def _video_key(url: str) -> VideoKey:
    """The stable video identity for `url` — the dedup key.

    Prefers `<category>/<id>` parsed from the path (`amplify_video/<id>` etc.),
    which survives the rotating `?tag=` / signing / filename on the full URL.
    Falls back to `<netloc><path>` (query stripped) for an unrecognised pattern —
    the safe direction: identical media paths still de-dup, and different videos
    never collide.
    """
    parsed = urlparse(url)
    segments = [seg for seg in parsed.path.split("/") if seg]
    for index, segment in enumerate(segments):
        if segment in _VIDEO_CATEGORIES and index + 1 < len(segments):
            return f"{segment}/{segments[index + 1]}"
    return f"{parsed.netloc}{parsed.path}"


def group_items_by_video(store: dict[str, Item], item_ids: list[str]) -> dict[VideoKey, list[str]]:
    """Group `item_ids` by the stable identity of their referenced video.

    Only items with a fetchable **mp4** entry are grouped — unknown ids and
    HLS / poster-era / no-video items are dropped (the caller reports them). Each
    group preserves first-seen order and is de-duplicated, so the same video is
    fetched + transcribed once and every referencing item gets the transcript.
    """
    groups: dict[VideoKey, list[str]] = {}
    for item_id in item_ids:
        item = store.get(item_id)
        if item is None:
            continue
        entry, _reason = _select_entry(item)
        if entry is None:
            continue
        members = groups.setdefault(_video_key(entry.url), [])
        if item_id not in members:
            members.append(item_id)
    return groups


def _is_x_video_source(source: ContentSource) -> bool:
    """True for an `x_video` content source (both union variants carry `kind`)."""
    return source.kind == "x_video"


def _has_x_video_source(item: Item) -> bool:
    """True when the item already carries an `x_video` transcript source."""
    if item.content is None:
        return False
    return any(_is_x_video_source(source) for source in item.content.sources)


def _source_url_for(item: Item) -> str:
    """The URL to record on the `x_video` source: the item's mp4 stream, else its
    permalink (a hand-edited store might lack a resolvable entry)."""
    entry, _reason = _select_entry(item)
    return entry.url if entry is not None else item.url


def attach_transcript(
    store: dict[str, Item],
    item_ids: list[str],
    transcript: Transcript,
    *,
    frames_by_item: dict[str, list[VideoFrame]] | None = None,
) -> int:
    """Attach `transcript` as an `x_video` source to each item; return the count.

    Idempotent per item: an existing `x_video` source is REPLACED (not
    duplicated), so a `--force` re-digest refreshes it. Any other content source
    (article body, thread) is preserved. A no-speech transcript is attached with
    empty text + `has_speech=False` — the marker `generate` renders as a silent
    video and `enrich` skips.

    `frames_by_item` (`--frames`, #44 PR4) carries the per-item key-frame slides
    (each item's own `<id>/frames/` paths); its default of no frames keeps the
    audio-only attach unchanged. `content.fetched_at` is bumped to attach time in
    every case — including when appending to an existing `Content`. This is
    load-bearing for PR3's re-enrichment trigger (`enrich._needs_reenrichment`): a
    video enriched from its tweet BEFORE the transcript landed must re-enrich, and
    that hinges on `fetched_at` moving past the earlier `enriched_at`. Without the
    bump the new transcript would look already-processed and the video keeps "—".
    """
    now = datetime.now(timezone.utc)
    frames_map = frames_by_item or {}
    attached = 0
    for item_id in item_ids:
        item = store.get(item_id)
        if item is None:
            continue
        source = ContentSourceSuccess(
            kind="x_video",
            url=_source_url_for(item),
            title=transcript.title,
            text=transcript.text,
            has_speech=transcript.has_speech,
            language=transcript.language,
            frames=frames_map.get(item_id, []),
        )
        if item.content is None:
            item.content = Content(fetched_at=now, sources=[source])
        else:
            kept = [s for s in item.content.sources if not _is_x_video_source(s)]
            item.content.sources = [*kept, source]
            item.content.fetched_at = now
        attached += 1
    return attached


def _fetched_path(report: FetchReport, item_id: str) -> Path | None:
    """The local path of `item_id`'s successful fetch in `report`, else None."""
    for result in report.results:
        if result.id == item_id and result.outcome == "fetched" and result.path is not None:
            return Path(result.path)
    return None


def _extract_described_slides(path: Path, visual: VisualConfig, *, item_id: str) -> _VisualResult:
    """Extract → classify → describe the video's key frames (`--frames`, #44 PR4).

    A talking-head classification SKIPS the visual layer and LOGS the reason (never
    a silent drop); a slide classification describes every kept frame via the
    EXTERNAL vision step. A per-video `FrameExtractionFailed` (bad mp4) or
    `VisionFailed` (a describe failure) drops the layer for THIS video (logged) and
    the batch continues — the tool-not-found variants (`FrameExtractionToolNotFound`
    / `VisionNotFound`) are NOT caught here: they are global config errors that
    abort the run, exactly like a missing transcriber.
    """
    try:
        frames = visual.extract_fn(path)
    except FrameExtractionFailed as exc:
        logger.warning("digest-video: frame extraction failed for item %s: %s", item_id, exc)
        return _VisualResult(classification="skipped")
    if not frames:
        return _VisualResult(classification="talking_head")
    if visual.classify_fn(frames) == "talking_head":
        logger.info("digest-video: visual layer skipped (talking-head) for item %s", item_id)
        return _VisualResult(classification="talking_head")
    try:
        slides = [
            _DescribedSlide(frame.timestamp, frame.path, visual.describe_fn(frame.path))
            for frame in frames
        ]
    except VisionFailed as exc:
        logger.warning("digest-video: visual layer failed for item %s: %s", item_id, exc)
        return _VisualResult(classification="skipped")
    return _VisualResult(slides=slides, classification="slides")


def _analyze_media(
    path: Path, transcribe_fn: TranscribeFn, visual: VisualConfig | None, *, item_id: str
) -> _MediaAnalysis | None:
    """Transcribe (and optionally extract slides from) `path`, then discard the bytes.

    A per-video `TranscriberFailed` (malformed output for this one video) is logged
    and returns None so the batch continues; a missing-binary `TranscriberNotFound`
    is NOT caught — it aborts the whole run. The visual layer runs BEFORE the mp4 is
    discarded (it needs the bytes). The mp4 is unlinked in every case, so at most
    one video is on disk at a time; the extracted frame images live in a sibling
    temp dir reclaimed by the enclosing ephemeral `TemporaryDirectory`.
    """
    try:
        try:
            transcript = transcribe_fn(path)
        except TranscriberFailed as exc:
            logger.warning("digest-video: transcription failed for item %s: %s", item_id, exc)
            return None
        result = _VisualResult()
        if visual is not None:
            result = _extract_described_slides(path, visual, item_id=item_id)
        return _MediaAnalysis(transcript=transcript, visual=result)
    finally:
        path.unlink(missing_ok=True)


def _persist_slides_for_item(
    slides: list[_DescribedSlide], media_root: Path, item_id: str
) -> list[VideoFrame]:
    """Copy each kept slide under `media_root/<item_id>/frames/<n>.<ext>` + build
    the per-item `VideoFrame` list.

    The image is persisted where `generate` mirrors from — exactly like a
    downloaded photo — so the note embeds it with no extra wiring. `local_path` is
    per item (each referencing item embeds its own copy) so the dedup grouping and
    the per-item `_media/` embed stay consistent.
    """
    frames: list[VideoFrame] = []
    for index, slide in enumerate(slides):
        local_path = f"{item_id}/frames/{index}{slide.path.suffix}"
        destination = media_root / local_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(slide.path, destination)
        frames.append(
            VideoFrame(
                timestamp=slide.timestamp, local_path=local_path, description=slide.description
            )
        )
    return frames


def _frames_by_item(
    slides: list[_DescribedSlide], media_root: Path, item_ids: list[str]
) -> dict[str, list[VideoFrame]]:
    """Persist the shared slides once per referencing item → per-item `VideoFrame`s.

    The frames are described ONCE (in `_extract_described_slides`); here each
    needing item gets its own persisted copy + `<id>/frames/` paths."""
    return {item_id: _persist_slides_for_item(slides, media_root, item_id) for item_id in item_ids}


def _group_outcome(analysis: _MediaAnalysis, needing: list[str], already: int) -> _GroupOutcome:
    """Map one video-group's analysis to its counters (the single decision site).

    A with-speech transcript counts `transcribed`, a no-speech one `no_speech`
    (both `did_transcribe`). Orthogonally, the visual layer counts `visual_slides`
    (kept + embedded) or `visual_skipped` (talking-head) — a silent slide deck is
    both `no_speech` and `visual_slides`.
    """
    count = len(needing)
    visual = analysis.visual.classification
    return _GroupOutcome(
        already=already,
        transcribed=count if analysis.transcript.has_speech else 0,
        no_speech=0 if analysis.transcript.has_speech else count,
        did_transcribe=True,
        visual_slides=visual == "slides",
        visual_skipped=visual == "talking_head",
    )


def _process_group(
    store: dict[str, Item],
    ids: list[str],
    dest_dir: Path,
    *,
    force: bool,
    fetch_fn: FetchFn,
    transcribe_fn: TranscribeFn,
    visual: VisualConfig | None,
) -> _GroupOutcome:
    """Fetch + transcribe (+ optionally slide-describe) one video ONCE, attach it
    to every item that needs it.

    `needing` is the subset of the group without an `x_video` source (all of it
    under `--force`); the rest are `already`. The video is fetched via a NEEDING
    item (`needing[0]`), never an already-digested member whose signed URL may be
    stale/expired. On a fetch failure the needing items are `failed` (nothing
    attached). The visual layer (`--frames`) describes the slides ONCE and persists
    them PER item; a no-speech transcript is still attached (as the marker).
    Returns the counts; the caller sums them.
    """
    needing = [item_id for item_id in ids if force or not _has_x_video_source(store[item_id])]
    already = len(ids) - len(needing)
    if not needing:
        return _GroupOutcome(already=already)
    representative = needing[0]
    fetch_report = fetch_fn(store, [representative], dest_dir)
    fetched = _fetched_path(fetch_report, representative)
    if fetched is None:
        return _GroupOutcome(already=already, failed=len(needing))
    analysis = _analyze_media(fetched, transcribe_fn, visual, item_id=representative)
    if analysis is None:
        return _GroupOutcome(already=already, failed=len(needing))
    frames_by_item = (
        _frames_by_item(analysis.visual.slides, visual.media_root, needing)
        if visual is not None and analysis.visual.slides
        else None
    )
    attach_transcript(store, needing, analysis.transcript, frames_by_item=frames_by_item)
    return _group_outcome(analysis, needing, already)


def _tally(report: DigestReport, outcome: _GroupOutcome) -> None:
    """Sum one group's outcome into the run report (the single tally site)."""
    report.transcribed += outcome.transcribed
    report.no_speech += outcome.no_speech
    report.already += outcome.already
    report.failed += outcome.failed
    if outcome.did_transcribe:
        report.videos_transcribed += 1
    report.visual_slides += int(outcome.visual_slides)
    report.visual_skipped += int(outcome.visual_skipped)


def _count_unselectable(
    store: dict[str, Item], unique_ids: list[str], grouped: set[str], report: DigestReport
) -> None:
    """Record requested ids that never made a group: unknown (absent from the
    store) vs no-video (present but no fetchable mp4) — distinct, never lumped."""
    for item_id in unique_ids:
        if item_id in grouped:
            continue
        if item_id in store:
            report.skipped_no_video += 1
        else:
            report.skipped_unknown += 1


def _default_transcribe(path: Path) -> Transcript:
    """Fallback transcriber (default config): the external `parakeet-mlx`.

    The CLI injects a config-bound `transcribe_fn` (command / model / language);
    this default keeps `digest_videos` callable without wiring for simple use.
    """
    return transcribe_media(path)


def digest_videos(
    store: dict[str, Item],
    item_ids: list[str],
    *,
    force: bool = False,
    fetch_fn: FetchFn = fetch_videos,
    transcribe_fn: TranscribeFn = _default_transcribe,
    temp_root: Path | str | None = None,
    visual: VisualConfig | None = None,
) -> DigestReport:
    """Digest each selected video into an `x_video` transcript source.

    Groups `item_ids` by video identity (dedup), then for each group fetches the
    video ONCE into an ephemeral temp dir, transcribes it, attaches the transcript
    to every referencing item that needs it, and discards the bytes. Idempotent
    (already-digested items are skipped unless `force`); no video byte survives the
    call (the temp dir is removed even if transcription raises). Mutates `store` in
    place and returns a `DigestReport`; the caller persists.

    When `visual` is provided (`--frames`, #44 PR4), each slide-classified video
    also has its key frames extracted, described via the EXTERNAL vision step, and
    attached (+ the slide images persisted under `visual.media_root`); a
    talking-head video's visual layer is skipped and logged. `visual=None` (the
    default) leaves the audio-only path unchanged.
    """
    unique_ids = list(dict.fromkeys(item_ids))
    groups = group_items_by_video(store, unique_ids)
    grouped = {item_id for members in groups.values() for item_id in members}
    report = DigestReport(groups=groups)
    _count_unselectable(store, unique_ids, grouped, report)

    with tempfile.TemporaryDirectory(prefix="xbrain-digest-", dir=temp_root) as tmp:
        dest_dir = Path(tmp)
        for ids in groups.values():
            outcome = _process_group(
                store,
                ids,
                dest_dir,
                force=force,
                fetch_fn=fetch_fn,
                transcribe_fn=transcribe_fn,
                visual=visual,
            )
            _tally(report, outcome)
    return report


def format_digest_summary(report: DigestReport) -> str:
    """One-line human SUMMARY of a digest run (mirrors the fetch/download lines).

    The visual-layer segment is appended ONLY when `--frames` actually did
    something (kept slides or skipped a talking-head), so a non-`--frames` run's
    summary is unchanged.
    """
    summary = (
        f"Vídeos: transcritos {report.transcribed}, sin voz {report.no_speech}, "
        f"ya digeridos {report.already}, fallidos {report.failed}, "
        f"sin vídeo {report.skipped_no_video}, desconocidos {report.skipped_unknown}. "
        f"Dedup: {report.total_items} items ← {report.video_count} vídeos "
        f"({report.videos_transcribed} transcritos este run)."
    )
    if report.visual_slides or report.visual_skipped:
        summary += (
            f" Visual: {report.visual_slides} con slides, "
            f"{report.visual_skipped} talking-head (saltados)."
        )
    return summary
