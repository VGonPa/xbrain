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
  (`amplify_video/<id>` / `ext_tw_video/<id>` / `tweet_video/<id>`). On the
  default path, N bookmarks of the same video are fetched + transcribed **once**;
  every referencing item gets the same transcript source. `keep_transcript`
  instead partitions that video by each item's stored transcript (plus one
  missing-transcript batch): one fetch per batch, zero ASR for stored batches,
  and one ASR for the missing batch. That extra fetch is what prevents one
  item's stored transcript from moving to another.
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
from typing import Literal
from urllib.parse import urlparse

from xbrain.models import (
    Content,
    ContentSource,
    ContentSourceSuccess,
    FRAME_CAPTION_CONTRACT,
    Item,
    VideoFrame,
)
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
# `reduce_fn` runs on the RAW extracted frames AFTER classification (so dedup can't
# skew the slides-vs-talking-head vote) and BEFORE describe — it dedupes + caps the
# set the vision model actually captions. Default = identity (describe them all).
ReduceFn = Callable[[list[KeyFrame]], list[KeyFrame]]


def _no_reduce(frames: list[KeyFrame]) -> list[KeyFrame]:
    """Default `reduce_fn` AND `footage_reduce_fn` — identity (describe every frame
    `extract_fn` returned); the CLI binds the real `[frames]` budgets."""
    return frames


# The per-video visual-layer outcome. `disabled` = no `--frames`; `slides` = kept
# + described + embedded (counts as a "with slides" video); `footage` = a SILENT
# video whose readable frames do not look like slides — kept + described +
# embedded under its own, smaller budget, because with no speech the frames are
# the only evidence of what the video shows (counts as silent footage);
# `talking_head` = the same non-slide verdict on a video WITH speech, whose
# transcript already carries the content (counts as a skipped talking-head);
# `skipped` = a non-content drop (extraction/vision failed, zero frames selected,
# or every frame unreadable) — logged with its reason, counted as NONE of slides,
# footage or talking-head, so no visual tally conflates failures with real content
# decisions (a silent `skipped` video is counted `hollow` instead).
Classification = Literal["disabled", "slides", "footage", "talking_head", "skipped"]


@dataclass(frozen=True)
class VisualConfig:
    """The opt-in `--frames` visual-layer configuration (#44 PR4).

    Its PRESENCE on `digest_videos` enables the layer; `None` (the default) leaves
    the audio-only path byte-for-byte unchanged. `media_root` is where kept slides
    are persisted (`<media_root>/<item-id>/frames/<n>.png`) so `generate` mirrors
    them into the vault's `_media/` tree and embeds them like downloaded photos.

    `reduce_fn` trims the frames of a slide deck before they are described;
    `footage_reduce_fn` trims the frames of SILENT non-slide footage (screen
    recordings, demos, animations). They are separate so footage gets its own,
    smaller budget: a slide deck needs one frame per distinct slide, while a
    silent clip is covered by a handful of frames — and each one is a vision call.
    """

    media_root: Path
    extract_fn: ExtractFn
    describe_fn: DescribeFn
    classify_fn: ClassifyFn = classify_visual
    reduce_fn: ReduceFn = _no_reduce
    footage_reduce_fn: ReduceFn = _no_reduce


@dataclass(frozen=True)
class _DescribedSlide:
    """A kept key frame + its EXTERNAL-vision description, before per-item persist."""

    timestamp: float
    path: Path
    description: str


@dataclass(frozen=True)
class _VisualResult:
    """The per-video outcome of the visual layer (constructed once, never mutated).

    `classification` is one of `disabled` (no `--frames`), `slides` (kept +
    described + embedded), `footage` (a silent non-slide video — its frames kept +
    described + embedded like slides), `talking_head` (a non-slide video WITH
    speech — skipped + logged, the transcript carries it), or `skipped` (a
    non-content drop: extraction / vision failed, ZERO frames selected, or every
    frame unreadable — logged with its reason, counted as none of slides, footage
    or talking-head). `slides` holds the described frames and is non-empty only for
    `classification` `slides` or `footage`.
    """

    slides: list[_DescribedSlide] = field(default_factory=list)
    classification: Classification = "disabled"


@dataclass(frozen=True)
class _MediaAnalysis:
    """The transcript + visual result for one fetched video, before attach.

    `reused_transcript` is True when `transcript` is an item's stored one
    (`--keep-transcript`) rather than fresh ASR output.
    """

    transcript: Transcript
    visual: _VisualResult
    reused_transcript: bool = False


@dataclass
class DigestReport:
    """Structured outcome of a `digest_videos` run (drives the CLI summary).

    Item-granular counters: `transcribed` (got a with-speech source), `no_speech`
    (got a `has_speech=False` marker), `already` (skipped — already carried an
    `x_video` source), `failed` (its video's fetch/transcribe failed),
    `skipped_no_video` (in the store but no fetchable mp4), `skipped_unknown` (id
    absent from the store), `hollow` (attached with no words AND no frames — it
    overlaps `transcribed` / `no_speech`, it is not a separate bucket of the
    total). `videos_transcribed` is the distinct videos the ASR actually
    transcribed this run, and `videos_reused` the distinct videos whose stored
    transcript was reused instead (`--keep-transcript`); a video can be both when
    its items are split (see `_transcript_batches`). `groups` is the dedup
    grouping so the summary can report "N items ← M videos".
    """

    transcribed: int = 0
    no_speech: int = 0
    already: int = 0
    failed: int = 0
    skipped_no_video: int = 0
    skipped_unknown: int = 0
    videos_transcribed: int = 0
    videos_reused: int = 0
    # Visual layer (`--frames`, #44 PR4): distinct videos whose slides were
    # extracted + embedded, distinct videos skipped as talking-head, and distinct
    # SILENT non-slide videos whose frames were described as footage (all 0 on a
    # non-`--frames` run).
    visual_slides: int = 0
    visual_skipped: int = 0
    visual_footage: int = 0
    # Items attached with neither words (`_carries_speech`) nor frames — an
    # x_video source that carries no content at all. Item-granular like
    # `no_speech`, and counted on EVERY run, whatever the reason the frames are
    # missing (no `--frames`, extraction failed, unreadable frames, vision
    # failed), so a hollow entry is never silent.
    hollow: int = 0
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
    across three functions. A group processed in several batches
    (`_transcript_batches`) combines their outcomes with `merged` first.
    """

    transcribed: int = 0
    no_speech: int = 0
    already: int = 0
    failed: int = 0
    did_transcribe: bool = False
    reused_transcript: bool = False
    visual_slides: bool = False
    visual_skipped: bool = False
    visual_footage: bool = False
    hollow: int = 0

    def merged(self, other: _GroupOutcome) -> _GroupOutcome:
        """This outcome plus `other`, another batch of the SAME video group.

        Item counters add up; the per-video flags are set when either batch set
        them, so the video still counts once in each per-video tally.
        """
        return _GroupOutcome(
            transcribed=self.transcribed + other.transcribed,
            no_speech=self.no_speech + other.no_speech,
            already=self.already + other.already,
            failed=self.failed + other.failed,
            did_transcribe=self.did_transcribe or other.did_transcribe,
            reused_transcript=self.reused_transcript or other.reused_transcript,
            visual_slides=self.visual_slides or other.visual_slides,
            visual_skipped=self.visual_skipped or other.visual_skipped,
            visual_footage=self.visual_footage or other.visual_footage,
            hollow=self.hollow + other.hollow,
        )


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
    group preserves first-seen order and is de-duplicated. On the default path
    the group is fetched + transcribed once and every referencing item gets the
    transcript; `keep_transcript` may later split it into transcript-specific
    fetch batches so each item keeps its own stored metadata.
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


def _stored_transcript(item: Item) -> Transcript | None:
    """The transcript stored on `item`'s first `x_video` success source, else None.

    `--keep-transcript` reuses it instead of re-running the ASR: the ASR can
    invent words on audio with no usable speech ("you you"), and a hollow video
    that gains invented words can be skipped as a talking-head with no frames —
    and is then no longer counted hollow. Only a success source carries a
    transcript, so failed `x_video` sources are passed over; with no success
    source there is nothing to keep. The stored source has no segments. A legacy
    `has_speech=None` is read from the text (words ⇒ speech), as `generate`
    treats it as speech rather than as a silent video.
    """
    if item.content is None:
        return None
    source = next(
        (
            s
            for s in item.content.sources
            if isinstance(s, ContentSourceSuccess) and _is_x_video_source(s)
        ),
        None,
    )
    if source is None:
        return None
    has_speech = source.has_speech if source.has_speech is not None else bool(source.text.strip())
    return Transcript(
        text=source.text,
        segments=[],
        language=source.language,
        has_speech=has_speech,
        title=source.title,
    )


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
    video. Enrich still re-runs once for it (the `fetched_at` bump below re-flags
    the item as pending), but the empty transcript is excluded from the enrichment
    prompt everywhere, so the result reflects the tweet signal — only a wasted call
    on the api/manual tracks.

    `frames_by_item` (`--frames`, #44 PR4) carries the per-item key-frame slides
    (each item's own `<id>/frames/` paths); its default of no frames keeps the
    audio-only attach unchanged. `content.fetched_at` is bumped to attach time in
    every case — including when appending to an existing `Content`. This is
    load-bearing for PR3's re-enrichment trigger (`enrich._needs_reenrichment`): a
    video enriched from its tweet BEFORE the transcript landed must re-enrich, and
    that hinges on `fetched_at` moving past the earlier `enriched_at`. Without the
    bump the new transcript would look already-processed and the video keeps "—".

    When this run actually described frames, the source is also stamped with
    `FRAME_CAPTION_CONTRACT` (`caption_contract`, #90) — which rubric produced
    those captions, so `redescribe-frames` can tell a current caption from a
    stale pre-#90 one and skip it. An audio-only digest attaches `""`: it made
    no captions, so it has nothing to vouch for.
    """
    now = datetime.now(timezone.utc)
    frames_map = frames_by_item or {}
    attached = 0
    for item_id in item_ids:
        item = store.get(item_id)
        if item is None:
            continue
        item_frames = frames_map.get(item_id, [])
        source = ContentSourceSuccess(
            kind="x_video",
            url=_source_url_for(item),
            title=transcript.title,
            text=transcript.text,
            has_speech=transcript.has_speech,
            language=transcript.language,
            frames=item_frames,
            # Claim the contract only when this run actually described frames; an
            # audio-only digest has no captions to vouch for.
            caption_contract=FRAME_CAPTION_CONTRACT if item_frames else "",
        )
        if item.content is None:
            item.content = Content(fetched_at=now, sources=[source])
        else:
            prior = next((s for s in item.content.sources if _is_x_video_source(s)), None)
            kept = [s for s in item.content.sources if not _is_x_video_source(s)]
            item.content.sources = [*kept, source]
            item.content.fetched_at = now
            _log_dropped_visual_layer(item_id, prior, source)
        attached += 1
    return attached


def _log_dropped_visual_layer(
    item_id: str, prior: ContentSource | None, new_source: ContentSourceSuccess
) -> None:
    """Log when a re-digest strips a prior kept visual layer, so it is never silent.

    A `--force` re-digest replaces the `x_video` source: if the old one carried
    slides and the new one carries none (a `--force` run WITHOUT `--frames`, or one
    where the video flipped to talking-head / skipped), the kept slides are dropped
    — an operator-visible change that must be logged, not swallowed.
    """
    prior_frames = len(prior.frames) if isinstance(prior, ContentSourceSuccess) else 0
    if prior_frames and not new_source.frames:
        logger.info(
            "digest-video: re-digest dropped %d prior slide(s) from item %s "
            "(rebuilt without a kept visual layer)",
            prior_frames,
            item_id,
        )


def _fetched_path(report: FetchReport, item_id: str) -> Path | None:
    """The local path of `item_id`'s successful fetch in `report`, else None."""
    for result in report.results:
        if result.id == item_id and result.outcome == "fetched" and result.path is not None:
            return Path(result.path)
    return None


def _describe_frames(
    frames: list[KeyFrame],
    visual: VisualConfig,
    *,
    item_id: str,
    classification: Literal["slides", "footage"],
) -> _VisualResult:
    """Describe each already-reduced frame via the EXTERNAL vision step.

    Slides and silent footage share this one path, so a footage frame is
    described — and fails — exactly like a slide: a per-frame `VisionFailed` drops
    the whole visual layer for the video (`skipped`, warned), never a silent
    partial set.
    """
    try:
        slides = [
            _DescribedSlide(frame.timestamp, frame.path, visual.describe_fn(frame.path))
            for frame in frames
        ]
    except VisionFailed as exc:
        logger.warning("digest-video: visual layer failed for item %s: %s", item_id, exc)
        return _VisualResult(classification="skipped")
    return _VisualResult(slides=slides, classification=classification)


def _describe_silent_footage(
    frames: list[KeyFrame], visual: VisualConfig, *, item_id: str
) -> _VisualResult:
    """Describe a silent non-slide video's frames as `footage`.

    Like the slides path, the classifier saw the RAW frames and only the describe
    set is reduced — but with `footage_reduce_fn`, footage's own smaller budget.
    """
    footage = visual.footage_reduce_fn(frames)
    logger.info(
        "digest-video: silent non-slide video — describing %d frame(s) as footage for item %s",
        len(footage),
        item_id,
    )
    return _describe_frames(footage, visual, item_id=item_id, classification="footage")


def _extract_described_slides(
    path: Path, visual: VisualConfig, *, item_id: str, has_speech: bool
) -> _VisualResult:
    """Extract → classify → describe the video's key frames (`--frames`, #44 PR4).

    Distinguishes a genuine content decision from a failure so an operator
    debugging "why was my slide deck skipped?" is never misled:

    - `slides` → reduce with `reduce_fn`, then describe every kept frame via the
      EXTERNAL vision step — with or without speech.
    - a non-slide verdict (`classify_fn` says `talking_head`) depends on
      `has_speech` — whether the transcript `_analyze_media` already produced
      carries words (`_carries_speech`, not the bare flag):
      - WITH speech → `talking_head`: SKIP + `info` log; the transcript carries
        the content, so describing camera cuts would be wasted vision calls.
      - WITHOUT speech → `footage`: the frames are the only evidence left (screen
        recordings of light UIs, charts, robots, animations score below the
        slide edge threshold, but none is a talking head), so reduce them with
        `footage_reduce_fn` and describe them like slides. Skipping them would
        attach an empty, frameless source — a hollow entry.
    - `skipped` (a NON-content drop, whatever the speech) → logged with its
      specific reason, counted as neither slides, footage nor talking-head: a
      per-video `FrameExtractionFailed`
      (bad mp4), ZERO frames selected (ffmpeg found nothing — logged, not silently
      bucketed as talking-head), every frame `unreadable` (a systemic decode
      problem — surfaced, not degraded), or a `VisionFailed` describe failure.

    The tool-not-found variants (`FrameExtractionToolNotFound` / `VisionNotFound`)
    are NOT caught here — they are global config errors that abort the run, exactly
    like a missing transcriber.
    """
    try:
        frames = visual.extract_fn(path)
    except FrameExtractionFailed as exc:
        logger.warning("digest-video: frame extraction failed for item %s: %s", item_id, exc)
        return _VisualResult(classification="skipped")
    if not frames:
        logger.info(
            "digest-video: no key frames extracted for item %s — visual layer skipped", item_id
        )
        return _VisualResult(classification="skipped")
    classification = visual.classify_fn(frames)
    if classification == "talking_head":
        if has_speech:
            logger.info("digest-video: visual layer skipped (talking-head) for item %s", item_id)
            return _VisualResult(classification="talking_head")
        return _describe_silent_footage(frames, visual, item_id=item_id)
    if classification == "unreadable":
        logger.warning(
            "digest-video: all %d extracted frame(s) unreadable for item %s — visual layer skipped",
            len(frames),
            item_id,
        )
        return _VisualResult(classification="skipped")
    # Classification saw the RAW frames; now dedupe + cap only the describe set.
    slides = visual.reduce_fn(frames)
    return _describe_frames(slides, visual, item_id=item_id, classification="slides")


def _carries_speech(transcript: Transcript) -> bool:
    """Whether the transcript carries words a reader gets — the digest's ONE speech test.

    `has_speech` alone is looser than every consumer's check:
    `transcribe._derive_has_speech` returns True for blank segments
    (`{"segments": [{"text": " "}]}`) and trusts `{"text": "", "has_speech": true}`,
    both with empty text. Every consumer of the attached source requires
    non-empty text on top of the flag, so a wordless source is silent there:
    `video_digest._has_digestible_content`, `worksheet._video_transcript` and
    `executors.api._video_transcript_section` require a truthy `has_speech`;
    `generate._video_digest_lines` requires `has_speech is not False`, so a
    `None` flag counts there — but `attach_transcript` always copies the
    transcript's bool, so digest never writes `None`. Gating on the flag alone
    would skip such a video's frames as a talking-head and never count it
    hollow: the exact silent, frameless entry the footage path exists to
    prevent. Whitespace-only text counts as blank, so this is never looser than
    those consumers.

    It gates the footage decision and the `hollow` count only; the
    `transcribed` / `no_speech` counters keep reporting the transcriber's flag.
    """
    return transcript.has_speech and bool(transcript.text.strip())


def _analyze_media(
    path: Path,
    transcribe_fn: TranscribeFn,
    visual: VisualConfig | None,
    *,
    item_id: str,
    stored_transcript: Transcript | None = None,
) -> _MediaAnalysis | None:
    """Transcribe (and optionally extract slides from) `path`, then discard the bytes.

    A `stored_transcript` (`--keep-transcript`) is used as-is and `transcribe_fn`
    is not called; everything after the transcription step is the same.

    A per-video `TranscriberFailed` (malformed output for this one video) is logged
    and returns None so the batch continues; a missing-binary `TranscriberNotFound`
    is NOT caught — it aborts the whole run. The visual layer runs BEFORE the mp4 is
    discarded (it needs the bytes) and AFTER transcription, whose words
    (`_carries_speech`) decide whether a non-slide video is skipped or described
    as footage. The mp4 is unlinked in every case, so at most one video is on
    disk at a time; the extracted frame images live in a sibling temp dir
    reclaimed by the enclosing ephemeral `TemporaryDirectory`.
    """
    try:
        try:
            transcript = transcribe_fn(path) if stored_transcript is None else stored_transcript
        except TranscriberFailed as exc:
            logger.warning("digest-video: transcription failed for item %s: %s", item_id, exc)
            return None
        result = _VisualResult()
        if visual is not None:
            result = _extract_described_slides(
                path, visual, item_id=item_id, has_speech=_carries_speech(transcript)
            )
        return _MediaAnalysis(
            transcript=transcript,
            visual=result,
            reused_transcript=stored_transcript is not None,
        )
    finally:
        path.unlink(missing_ok=True)


def _reset_item_frames(media_root: Path, item_ids: list[str]) -> None:
    """Clear each item's persisted slide dir before a (re-)digest writes the current set.

    A re-digest that yields FEWER slides — or flips to talking-head / skipped —
    must not leave stale higher-index PNGs (`<id>/frames/5.png`) orphaned on disk:
    `generate` would keep mirroring them into the vault though nothing references
    them. Clearing `<media_root>/<id>/frames/` first makes the on-disk set match the
    current result exactly. Called only on the `--frames` path, for the items about
    to be (re)written.
    """
    for item_id in item_ids:
        frames_dir = media_root / item_id / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)


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


def _group_outcome(analysis: _MediaAnalysis, needing: list[str]) -> _GroupOutcome:
    """Map one batch's analysis to its counters (the single decision site).

    A with-speech transcript counts `transcribed`, a no-speech one `no_speech`.
    The batch sets `did_transcribe` when the ASR produced that transcript and
    `reused_transcript` when it was a stored one. Orthogonally, the visual layer counts `visual_slides`
    (kept + embedded), `visual_footage` (silent non-slide frames described) or
    `visual_skipped` (talking-head) — a silent slide deck is both `no_speech` and
    `visual_slides`. A group whose transcript carries no words (`_carries_speech`
    — stricter than the flag the two transcript counters read) and that ends up
    with NO described frames — for any reason, `--frames` or not — counts its
    items as `hollow`.
    """
    count = len(needing)
    has_speech = analysis.transcript.has_speech
    visual = analysis.visual
    return _GroupOutcome(
        transcribed=count if has_speech else 0,
        no_speech=0 if has_speech else count,
        did_transcribe=not analysis.reused_transcript,
        reused_transcript=analysis.reused_transcript,
        visual_slides=visual.classification == "slides",
        visual_skipped=visual.classification == "talking_head",
        visual_footage=visual.classification == "footage",
        hollow=0 if _carries_speech(analysis.transcript) or visual.slides else count,
    )


# One batch of a video group: the items that share a transcript, and the stored
# transcript they reuse (`None` = run the ASR).
_Batch = tuple[list[str], Transcript | None]


def _transcript_batches(
    store: dict[str, Item], needing: list[str], *, keep_transcript: bool
) -> list[_Batch]:
    """Split a group's `needing` items by where their transcript comes from.

    Without `keep_transcript` the whole group is one ASR batch, as it always was.
    With it, no transcript ever moves from one item to another: items whose
    stored transcripts are identical share a batch that reuses it, and the items
    with none share one batch that runs the ASR. Batches keep first-seen order,
    and each one fetches the video, so a group that mixes stored and missing (or
    different) transcripts costs one fetch per batch.
    """
    if not keep_transcript:
        return [(needing, None)]
    batches: dict[tuple[str, bool, str | None, str | None] | None, _Batch] = {}
    for item_id in needing:
        stored = _stored_transcript(store[item_id])
        key = (
            None
            if stored is None
            else (stored.text, stored.has_speech, stored.language, stored.title)
        )
        batches.setdefault(key, ([], stored))[0].append(item_id)
    return list(batches.values())


def _process_batch(
    store: dict[str, Item],
    batch: list[str],
    dest_dir: Path,
    *,
    stored_transcript: Transcript | None,
    fetch_fn: FetchFn,
    transcribe_fn: TranscribeFn,
    visual: VisualConfig | None,
) -> _GroupOutcome:
    """Fetch + transcribe (+ optionally slide-describe) one video ONCE, attach it
    to every item of `batch`.

    The video is fetched via `batch[0]` — a NEEDING item, never an
    already-digested member whose signed URL may be stale/expired. On a fetch
    failure the batch's items are `failed` (nothing attached). A
    `stored_transcript` is reused instead of calling the ASR; the video is still
    fetched, because the frames need it. The visual layer (`--frames`) describes
    the frames ONCE and persists them PER item whenever it kept any — slides or
    silent footage alike; a no-speech transcript is still attached (as the
    marker).
    """
    representative = batch[0]
    fetch_report = fetch_fn(store, [representative], dest_dir)
    fetched = _fetched_path(fetch_report, representative)
    if fetched is None:
        return _GroupOutcome(failed=len(batch))
    analysis = _analyze_media(
        fetched,
        transcribe_fn,
        visual,
        item_id=representative,
        stored_transcript=stored_transcript,
    )
    if analysis is None:
        return _GroupOutcome(failed=len(batch))
    frames_by_item = None
    if visual is not None:
        # Clear stale slide dirs for every batch item BEFORE persisting the
        # current result — a re-digest with fewer slides (or none) must not leave
        # orphaned higher-index PNGs behind.
        _reset_item_frames(visual.media_root, batch)
        if analysis.visual.slides:
            frames_by_item = _frames_by_item(analysis.visual.slides, visual.media_root, batch)
    attach_transcript(store, batch, analysis.transcript, frames_by_item=frames_by_item)
    return _group_outcome(analysis, batch)


def _process_group(
    store: dict[str, Item],
    ids: list[str],
    dest_dir: Path,
    *,
    force: bool,
    keep_transcript: bool,
    fetch_fn: FetchFn,
    transcribe_fn: TranscribeFn,
    visual: VisualConfig | None,
) -> _GroupOutcome:
    """Digest one video group: attach a transcript to every item that needs it.

    `needing` is the subset of the group without an `x_video` source (all of it
    under `--force`); the rest are `already`. `_transcript_batches` splits
    `needing` (one batch unless `keep_transcript`), and each batch is fetched
    and processed once by `_process_batch`. Returns the group's combined
    counts; the caller sums them.
    """
    needing = [item_id for item_id in ids if force or not _has_x_video_source(store[item_id])]
    outcome = _GroupOutcome(already=len(ids) - len(needing))
    if not needing:
        return outcome
    for batch, stored_transcript in _transcript_batches(
        store, needing, keep_transcript=keep_transcript
    ):
        batch_outcome = _process_batch(
            store,
            batch,
            dest_dir,
            stored_transcript=stored_transcript,
            fetch_fn=fetch_fn,
            transcribe_fn=transcribe_fn,
            visual=visual,
        )
        outcome = outcome.merged(batch_outcome)
    return outcome


def _tally(report: DigestReport, outcome: _GroupOutcome) -> None:
    """Sum one group's outcome into the run report (the single tally site)."""
    report.transcribed += outcome.transcribed
    report.no_speech += outcome.no_speech
    report.already += outcome.already
    report.failed += outcome.failed
    if outcome.did_transcribe:
        report.videos_transcribed += 1
    report.videos_reused += int(outcome.reused_transcript)
    report.visual_slides += int(outcome.visual_slides)
    report.visual_skipped += int(outcome.visual_skipped)
    report.visual_footage += int(outcome.visual_footage)
    report.hollow += outcome.hollow


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


def _check_keep_transcript(
    keep_transcript: bool, *, force: bool, visual: VisualConfig | None
) -> None:
    """Reject a `keep_transcript` run that cannot do what the flag is for.

    The same preconditions the CLI checks: without `visual` a keep run would
    only strip each item's stored frames and long-form digest, and without
    `force` every item with a stored transcript is skipped.
    """
    if keep_transcript and visual is None:
        raise ValueError("keep_transcript requires visual (it re-runs the visual layer)")
    if keep_transcript and not force:
        raise ValueError("keep_transcript requires force (it re-digests already-digested videos)")


def digest_videos(
    store: dict[str, Item],
    item_ids: list[str],
    *,
    force: bool = False,
    keep_transcript: bool = False,
    fetch_fn: FetchFn = fetch_videos,
    transcribe_fn: TranscribeFn = _default_transcribe,
    temp_root: Path | str | None = None,
    visual: VisualConfig | None = None,
) -> DigestReport:
    """Digest each selected video into an `x_video` transcript source.

    Groups `item_ids` by video identity (dedup). On the default path each group
    is fetched once into an ephemeral temp dir, transcribed once, attached to
    every referencing item that needs it, and discarded. Idempotent
    (already-digested items are skipped unless `force`); no video byte survives
    the call (the temp dir is removed even if transcription raises). Mutates
    `store` in place and returns a `DigestReport`; the caller persists.

    When `visual` is provided (`--frames`, #44 PR4), each slide-classified video
    also has its key frames extracted, described via the EXTERNAL vision step, and
    attached (+ the slide images persisted under `visual.media_root`). A video
    that does not look like slides is skipped and logged as a talking head only
    when its transcript carries speech; a SILENT one has its frames described and
    attached as footage instead, under `visual.footage_reduce_fn`'s budget.
    `visual=None` (the default) leaves the audio-only path unchanged; a silent
    video attached without frames is counted in `DigestReport.hollow` either way.

    `keep_transcript` (`--frames --force --keep-transcript`) reuses each item's
    own stored transcript instead of re-running the ASR, and never gives one
    item's transcript to another (`_transcript_batches`); an item with no stored
    transcript is transcribed as usual. One `VideoKey` may therefore cost one
    fetch per distinct stored transcript plus one for the missing-transcript
    batch, while stored batches cost zero ASR. The visual layer is redone, and,
    as on any forced re-digest, a completed batch replaces the source: its
    long-form `digest` is cleared and `content.fetched_at` is bumped, so
    `video-digest` and `enrich` pick the item up again. A visual failure or a
    talking-head reclassification consequently drops prior frames; the CLI's
    pre-write snapshot is the undo boundary. It requires `visual` and `force`
    (`ValueError` otherwise).
    """
    _check_keep_transcript(keep_transcript, force=force, visual=visual)
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
                keep_transcript=keep_transcript,
                fetch_fn=fetch_fn,
                transcribe_fn=transcribe_fn,
                visual=visual,
            )
            _tally(report, outcome)
    return report


def format_digest_summary(report: DigestReport) -> str:
    """One-line human SUMMARY of a digest run (mirrors the fetch/download lines).

    The visual-layer segment is appended ONLY when `--frames` actually did
    something (kept slides, described silent footage or skipped a talking-head).
    The hollow segment follows whenever items were attached with neither speech
    nor frames — on any run, `--frames` or not — so such an entry is never silent.
    A run with neither prints the same line as before. A run that reused stored
    transcripts (`--keep-transcript`) names them inside the Dedup parenthesis;
    without reuse that parenthesis is unchanged.
    """
    reused = f", {report.videos_reused} con transcripción guardada" if report.videos_reused else ""
    summary = (
        f"Vídeos: transcritos {report.transcribed}, sin voz {report.no_speech}, "
        f"ya digeridos {report.already}, fallidos {report.failed}, "
        f"sin vídeo {report.skipped_no_video}, desconocidos {report.skipped_unknown}. "
        f"Dedup: {report.total_items} items ← {report.video_count} vídeos "
        f"({report.videos_transcribed} transcritos este run{reused})."
    )
    if report.visual_slides or report.visual_footage or report.visual_skipped:
        summary += (
            f" Visual: {report.visual_slides} con slides, "
            f"{report.visual_footage} metraje mudo descrito, "
            f"{report.visual_skipped} talking-head (saltados)."
        )
    if report.hollow:
        summary += f" Huecos (sin voz ni frames): {report.hollow}."
    return summary
