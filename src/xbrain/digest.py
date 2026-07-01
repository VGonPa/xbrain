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
persisting (with the destructive auto-snapshot) is the CLI's job.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from xbrain.models import Content, ContentSource, ContentSourceSuccess, Item
from xbrain.transcribe import Transcript, TranscriberFailed, transcribe_media
from xbrain.video_fetch import FetchReport, _select_entry, fetch_videos

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


@dataclass
class DigestReport:
    """Structured outcome of a `digest_videos` run (drives the CLI summary).

    Item-granular counters: `transcribed` (got a with-speech source),
    `no_speech` (got a `has_speech=False` marker), `already` (skipped — already
    carried a fresh `x_video` source), `failed` (its video's fetch/transcribe
    failed), `skipped_no_video` (requested but no fetchable mp4). `videos_fetched`
    is the distinct videos actually processed; `groups` is the dedup grouping so
    the summary can report "N items ← M videos".
    """

    transcribed: int = 0
    no_speech: int = 0
    already: int = 0
    failed: int = 0
    skipped_no_video: int = 0
    videos_fetched: int = 0
    failed_videos: int = 0
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
    """True for an `x_video` content source (success or, defensively, any)."""
    return getattr(source, "kind", None) == "x_video"


def _has_x_video_source(item: Item) -> bool:
    """True when the item already carries a fresh `x_video` transcript source."""
    if item.content is None:
        return False
    return any(_is_x_video_source(source) for source in item.content.sources)


def _source_url_for(item: Item) -> str:
    """The URL to record on the `x_video` source: the item's mp4 stream, else its
    permalink (a hand-edited store might lack a resolvable entry)."""
    entry, _reason = _select_entry(item)
    return entry.url if entry is not None else item.url


def attach_transcript(store: dict[str, Item], item_ids: list[str], transcript: Transcript) -> int:
    """Attach `transcript` as an `x_video` source to each item; return the count.

    Idempotent per item: an existing `x_video` source is REPLACED (not
    duplicated), so a `--force` re-digest refreshes it. Any other content source
    (article body, thread) is preserved. A no-speech transcript is attached with
    empty text + `has_speech=False` — the marker `generate` renders as a silent
    video and `enrich` skips.
    """
    now = datetime.now(timezone.utc)
    attached = 0
    for item_id in item_ids:
        item = store.get(item_id)
        if item is None:
            continue
        source = ContentSourceSuccess(
            kind="x_video",
            url=_source_url_for(item),
            text=transcript.text,
            has_speech=transcript.has_speech,
            language=transcript.language,
        )
        if item.content is None:
            item.content = Content(fetched_at=now, sources=[source])
        else:
            kept = [s for s in item.content.sources if not _is_x_video_source(s)]
            item.content.sources = [*kept, source]
        attached += 1
    return attached


def _fetched_path(report: FetchReport, item_id: str) -> Path | None:
    """The local path of `item_id`'s successful fetch in `report`, else None."""
    for result in report.results:
        if result.id == item_id and result.outcome == "fetched" and result.path is not None:
            return Path(result.path)
    return None


def _transcribe_and_discard(
    path: Path, transcribe_fn: TranscribeFn, *, item_id: str, count: int, report: DigestReport
) -> Transcript | None:
    """Transcribe `path`, then delete the bytes (even on failure).

    A per-video `TranscriberFailed` (malformed output for this one video) is
    recorded and returns None so the batch continues; a missing-binary
    `TranscriberNotFound` is NOT caught — it aborts the whole run (a global
    config error). The temp file is unlinked in every case.
    """
    try:
        return transcribe_fn(path)
    except TranscriberFailed as exc:
        logger.warning("digest-video: transcription failed for item %s: %s", item_id, exc)
        report.failed += count
        report.failed_videos += 1
        return None
    finally:
        path.unlink(missing_ok=True)


def _process_group(
    store: dict[str, Item],
    ids: list[str],
    dest_dir: Path,
    *,
    force: bool,
    fetch_fn: FetchFn,
    transcribe_fn: TranscribeFn,
    report: DigestReport,
) -> None:
    """Fetch + transcribe one video ONCE, attach it to every item that needs it.

    `needing` is the subset of the group without a fresh `x_video` source (all of
    it under `--force`); the rest count as `already`. On a fetch failure the
    needing items count as `failed` (nothing attached). A no-speech transcript is
    still attached (as the marker) and counted under `no_speech`.
    """
    needing = [item_id for item_id in ids if force or not _has_x_video_source(store[item_id])]
    report.already += len(ids) - len(needing)
    if not needing:
        return
    representative = ids[0]
    fetch_report = fetch_fn(store, [representative], dest_dir)
    fetched = _fetched_path(fetch_report, representative)
    if fetched is None:
        report.failed += len(needing)
        report.failed_videos += 1
        return
    transcript = _transcribe_and_discard(
        fetched, transcribe_fn, item_id=representative, count=len(needing), report=report
    )
    if transcript is None:
        return
    attach_transcript(store, needing, transcript)
    report.videos_fetched += 1
    if transcript.has_speech:
        report.transcribed += len(needing)
    else:
        report.no_speech += len(needing)


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
) -> DigestReport:
    """Digest each selected video into an `x_video` transcript source.

    Groups `item_ids` by video identity (dedup), then for each group fetches the
    video ONCE into an ephemeral temp dir, transcribes it, attaches the
    transcript to every referencing item that needs it, and discards the bytes.
    Idempotent (already-digested items are skipped unless `force`); no video byte
    survives the call (the temp dir is removed even if transcription raises).
    Mutates `store` in place and returns a `DigestReport`; the caller persists.
    """
    unique_ids = list(dict.fromkeys(item_ids))
    groups = group_items_by_video(store, unique_ids)
    grouped = {item_id for members in groups.values() for item_id in members}
    report = DigestReport(groups=groups)
    report.skipped_no_video = sum(1 for item_id in unique_ids if item_id not in grouped)

    with tempfile.TemporaryDirectory(prefix="xbrain-digest-", dir=temp_root) as tmp:
        dest_dir = Path(tmp)
        for ids in groups.values():
            _process_group(
                store,
                ids,
                dest_dir,
                force=force,
                fetch_fn=fetch_fn,
                transcribe_fn=transcribe_fn,
                report=report,
            )
    return report


def format_digest_summary(report: DigestReport) -> str:
    """One-line human SUMMARY of a digest run (mirrors the fetch/download lines)."""
    return (
        f"Vídeos: transcritos {report.transcribed}, sin voz {report.no_speech}, "
        f"ya digeridos {report.already}, fallidos {report.failed}, "
        f"sin vídeo {report.skipped_no_video} "
        f"({report.total_items} items ← {report.video_count} vídeos)"
    )
