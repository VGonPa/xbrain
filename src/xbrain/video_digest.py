"""`digest-video` orchestration.

The normal XBrain video path must not persist MP4/audio/frame bytes. This module
turns a captured text-track, or a temporary ASR transcript fallback, into an
`x_video` executive summary source and stores the raw transcript separately on
that source for vault rendering only. Downstream analysis consumes only the
summary (`ContentSourceSuccess.text`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypeGuard
from urllib.parse import urlparse

from xbrain.llm_client import LlmProvider
from xbrain.models import (
    Content,
    ContentSource,
    ContentSourceSuccess,
    Item,
    MediaEntry,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)
from xbrain.video_transcript import (
    VIDEO_SUMMARY_VERSION,
    VideoExecutiveSummary,
    VideoSummaryFailed,
    VideoTranscript,
    VideoTranscriptFetchFailed,
    VideoTranscriptFetchSkipped,
    VideoTranscriptUnavailable,
    fetch_video_transcript,
    summarize_video_transcript,
)

logger = logging.getLogger(__name__)

VideoEntry = MediaVideoPending | MediaVideoDownloaded | MediaVideoFailed
VideoKey = str
TranscriptFn = Callable[[Item, VideoEntry], VideoTranscript]
SummaryFn = Callable[[Item, VideoTranscript], VideoExecutiveSummary]

_VIDEO_TYPES = (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)
_VIDEO_CATEGORIES = ("amplify_video", "ext_tw_video", "tweet_video")


@dataclass
class VideoDigestReport:
    """Structured outcome of a video digest run."""

    summarized: int = 0
    already: int = 0
    no_transcript: int = 0
    failed: int = 0
    skipped_too_large: int = 0
    skipped_size_unknown: int = 0
    skipped_no_video: int = 0
    skipped_unknown: int = 0
    videos_summarized: int = 0
    groups: dict[VideoKey, list[str]] = field(default_factory=dict)

    @property
    def total_items(self) -> int:
        return sum(len(ids) for ids in self.groups.values())

    @property
    def video_count(self) -> int:
        return len(self.groups)

    @property
    def changed(self) -> int:
        return self.summarized


@dataclass
class _GroupOutcome:
    summarized: int = 0
    already: int = 0
    no_transcript: int = 0
    failed: int = 0
    skipped_too_large: int = 0
    skipped_size_unknown: int = 0
    did_summarize: bool = False


def _is_video_entry(entry: MediaEntry) -> TypeGuard[VideoEntry]:
    return isinstance(entry, _VIDEO_TYPES)


def _first_video_entry(item: Item) -> VideoEntry | None:
    return next((entry for entry in item.media if _is_video_entry(entry)), None)


def _video_key(url: str) -> VideoKey:
    parsed = urlparse(url)
    segments = [seg for seg in parsed.path.split("/") if seg]
    for index, segment in enumerate(segments):
        if segment in _VIDEO_CATEGORIES and index + 1 < len(segments):
            return f"{segment}/{segments[index + 1]}"
    return f"{parsed.netloc}{parsed.path}"


def _key_for_item(item: Item) -> VideoKey | None:
    entry = _first_video_entry(item)
    if entry is None:
        return None
    return _video_key(entry.url)


def _is_x_video_source(source: ContentSource) -> bool:
    return source.kind == "x_video"


def _has_x_video_source(item: Item) -> bool:
    return item.content is not None and any(_is_x_video_source(src) for src in item.content.sources)


def group_items_by_video(store: dict[str, Item], item_ids: list[str]) -> dict[VideoKey, list[str]]:
    """Group selected ids by stable video identity without fetching bytes."""
    groups: dict[VideoKey, list[str]] = {}
    for item_id in item_ids:
        item = store.get(item_id)
        if item is None:
            continue
        key = _key_for_item(item)
        if key is None:
            continue
        members = groups.setdefault(key, [])
        if item_id not in members:
            members.append(item_id)
    return groups


def _representative_video(
    store: dict[str, Item], item_ids: list[str]
) -> tuple[Item, VideoEntry] | None:
    """Pick a needing video item, preferring one that already has captions."""
    fallback: tuple[Item, VideoEntry] | None = None
    for item_id in item_ids:
        item = store[item_id]
        entry = _first_video_entry(item)
        if entry is None:
            continue
        if entry.transcript_url:
            return item, entry
        if fallback is None:
            fallback = (item, entry)
    return fallback


def attach_video_summary(
    store: dict[str, Item],
    item_ids: list[str],
    transcript: VideoTranscript,
    summary: VideoExecutiveSummary,
) -> int:
    """Attach executive summary as `x_video`; keep raw transcript off analysis paths."""
    now = datetime.now(timezone.utc)
    attached = 0
    for item_id in item_ids:
        item = store.get(item_id)
        if item is None:
            continue
        source = ContentSourceSuccess(
            kind="x_video",
            url=transcript.source_url,
            title=summary.title,
            text=summary.markdown,
            has_speech=True,
            language=transcript.language,
            raw_transcript=transcript.text,
            raw_transcript_url=transcript.source_url,
            raw_transcript_format=transcript.format,
            executive_summary_version=VIDEO_SUMMARY_VERSION,
            frames=[],
        )
        if item.content is None:
            item.content = Content(fetched_at=now, sources=[source])
        else:
            kept = [src for src in item.content.sources if not _is_x_video_source(src)]
            item.content.sources = [*kept, source]
            item.content.fetched_at = now
        attached += 1
    return attached


def _process_group(
    store: dict[str, Item],
    ids: list[str],
    *,
    force: bool,
    transcript_fn: TranscriptFn,
    summary_fn: SummaryFn,
) -> _GroupOutcome:
    needing = [item_id for item_id in ids if force or not _has_x_video_source(store[item_id])]
    already = len(ids) - len(needing)
    if not needing:
        return _GroupOutcome(already=already)
    selected = _representative_video(store, needing)
    if selected is None:
        return _GroupOutcome(already=already, no_transcript=len(needing))
    item, entry = selected
    try:
        transcript = transcript_fn(item, entry)
        summary = summary_fn(item, transcript)
    except VideoTranscriptFetchSkipped as exc:
        if exc.reason == "too_large":
            return _GroupOutcome(already=already, skipped_too_large=len(needing))
        if exc.reason == "size_unknown":
            return _GroupOutcome(already=already, skipped_size_unknown=len(needing))
        return _GroupOutcome(already=already, no_transcript=len(needing))
    except VideoTranscriptUnavailable:
        return _GroupOutcome(already=already, no_transcript=len(needing))
    except (VideoTranscriptFetchFailed, VideoSummaryFailed) as exc:
        logger.warning("digest-video: transcript digest failed for item %s: %s", item.id, exc)
        return _GroupOutcome(already=already, failed=len(needing))
    attach_video_summary(store, needing, transcript, summary)
    return _GroupOutcome(summarized=len(needing), already=already, did_summarize=True)


def _count_unselectable(
    store: dict[str, Item], unique_ids: list[str], grouped: set[str], report: VideoDigestReport
) -> None:
    for item_id in unique_ids:
        if item_id in grouped:
            continue
        if item_id in store:
            report.skipped_no_video += 1
        else:
            report.skipped_unknown += 1


def _tally(report: VideoDigestReport, outcome: _GroupOutcome) -> None:
    report.summarized += outcome.summarized
    report.already += outcome.already
    report.no_transcript += outcome.no_transcript
    report.failed += outcome.failed
    report.skipped_too_large += outcome.skipped_too_large
    report.skipped_size_unknown += outcome.skipped_size_unknown
    report.videos_summarized += int(outcome.did_summarize)


def _caption_transcript_fn(_item: Item, entry: VideoEntry) -> VideoTranscript:
    return fetch_video_transcript(entry)


def digest_video_transcripts(
    store: dict[str, Item],
    item_ids: list[str],
    *,
    force: bool = False,
    summary_fn: SummaryFn,
    transcript_fn: TranscriptFn = _caption_transcript_fn,
) -> VideoDigestReport:
    """Create x_video executive summaries from captions or temporary ASR fallback."""
    unique_ids = list(dict.fromkeys(item_ids))
    groups = group_items_by_video(store, unique_ids)
    grouped = {item_id for members in groups.values() for item_id in members}
    report = VideoDigestReport(groups=groups)
    _count_unselectable(store, unique_ids, grouped, report)
    for ids in groups.values():
        _tally(
            report,
            _process_group(
                store,
                ids,
                force=force,
                transcript_fn=transcript_fn,
                summary_fn=summary_fn,
            ),
        )
    return report


def format_video_digest_summary(report: VideoDigestReport) -> str:
    """One-line human summary for video digest runs."""
    return (
        f"Vídeos: resumidos {report.summarized}, ya digeridos {report.already}, "
        f"sin transcript {report.no_transcript}, fallidos {report.failed}, "
        f"> --max-size {report.skipped_too_large}, sin tamaño {report.skipped_size_unknown}, "
        f"sin vídeo {report.skipped_no_video}, desconocidos {report.skipped_unknown}. "
        f"Dedup: {report.total_items} items ← {report.video_count} vídeos "
        f"({report.videos_summarized} procesados este run)."
    )


def configured_summary_fn(
    *,
    provider: LlmProvider,
    model: str,
    output_language: str,
    base_url: str | None = None,
    client: object = None,
) -> SummaryFn:
    """Bind configured LLM settings into a `SummaryFn`."""

    def _summarize(item: Item, transcript: VideoTranscript) -> VideoExecutiveSummary:
        return summarize_video_transcript(
            item.text,
            item.author.handle,
            transcript,
            provider=provider,
            model=model,
            output_language=output_language,
            base_url=base_url,
            client=client,
        )

    return _summarize
