"""Read-only video catalog for `xbrain list-videos`.

Pure selection/derivation over the in-memory item store: one `VideoRow` per
video media entry, carrying a derived state (downloaded / failed / pending /
poster-era), an estimated size (bitrate × duration; the exact on-disk size for a
downloaded entry; `None` when unknown), the item's `primary_topic`, the resolved
stream URL and a short text snippet. **Zero writes, no network, no snapshot** —
`list_video_entries` is a function of the store alone. `fetch-video` (see
`xbrain.video_fetch`) consumes the same selection surface.

The mp4/HLS/poster discriminator (`_video_class`) and the size estimator
(`_estimated_bytes`) are REUSED from `xbrain.video_media` so the catalog's notion
of "what is a real mp4" and "how big is it" matches the downloader exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from xbrain.models import (
    Item,
    MediaEntry,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)
from xbrain.video_media import _SIZE_UNITS, _estimated_bytes, _video_class

# The four catalog states surfaced by `list-videos` (and the `--status` filter).
# `poster-era` is an un-backfilled pending entry (`url == thumbnail_url`); run
# `xbrain refresh-media` to resolve it into a real stream first.
VideoState = Literal["downloaded", "failed", "pending", "poster-era"]
_VIDEO_STATES: tuple[VideoState, ...] = ("downloaded", "failed", "pending", "poster-era")

# The video media variants — a photo entry is not a video row.
_VIDEO_VARIANTS = (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)
_VideoEntry = MediaVideoPending | MediaVideoDownloaded | MediaVideoFailed

# `--source` → item source set, identical to `extract` / `download-videos`.
_SOURCE_SETS: dict[str, set[str]] = {
    "bookmarks": {"bookmark"},
    "tweets": {"own_tweet"},
    "all": {"bookmark", "own_tweet"},
}

# Text-snippet cap: a one-line preview so the human table stays scannable. The
# full text lives on the item; the digest pipeline (PR 3) uses the transcript,
# not this snippet.
_SNIPPET_MAX = 80


@dataclass(frozen=True)
class VideoRow:
    """One catalog row for a single video media entry.

    `url` is the item permalink; `mp4_url` is the resolved stream URL of the
    video entry (the `.mp4`, or the `.m3u8` for an HLS entry) — `None` for a
    poster-era entry, which has no resolved stream. `size_bytes` is the exact
    on-disk size for a downloaded entry, else the bitrate × duration estimate,
    else `None` when neither is knowable. `topic` is the item's `primary_topic`
    or `None`.
    """

    id: str
    url: str
    state: VideoState
    topic: str | None
    size_bytes: int | None
    mp4_url: str | None
    text: str


def _is_video_entry(entry: MediaEntry) -> bool:
    """True when `entry` is one of the three video variants (photos elided)."""
    return isinstance(entry, _VIDEO_VARIANTS)


def _video_state(entry: _VideoEntry) -> VideoState:
    """Derive the catalog state from the entry variant + stream class.

    A downloaded/failed entry maps straight to its state; a pending entry is
    `poster-era` when its URL is the un-backfilled poster, else `pending`
    (a real mp4 or an HLS manifest awaiting download).
    """
    if isinstance(entry, MediaVideoDownloaded):
        return "downloaded"
    if isinstance(entry, MediaVideoFailed):
        return "failed"
    if _video_class(entry) == "poster":
        return "poster-era"
    return "pending"


def _row_size(entry: _VideoEntry) -> int | None:
    """Exact on-disk size for a downloaded entry, else the estimate (or None)."""
    if isinstance(entry, MediaVideoDownloaded):
        return entry.bytes_size
    return _estimated_bytes(entry)


def _mp4_url(entry: _VideoEntry) -> str | None:
    """The resolved stream URL, or None for a poster-era (un-backfilled) entry."""
    if _video_class(entry) == "poster":
        return None
    return entry.url


def _snippet(text: str) -> str:
    """A one-line, length-capped preview of the item text."""
    flattened = " ".join(text.split())
    if len(flattened) <= _SNIPPET_MAX:
        return flattened
    return flattened[:_SNIPPET_MAX] + "…"


def _primary_topic(item: Item) -> str | None:
    """The item's assigned `primary_topic`, or None when not enriched."""
    if item.enriched is None:
        return None
    return item.enriched.primary_topic


def _build_row(item_id: str, item: Item, entry: _VideoEntry) -> VideoRow:
    """Assemble the `VideoRow` for one (item, video-entry) pair."""
    return VideoRow(
        id=item_id,
        url=item.url,
        state=_video_state(entry),
        topic=_primary_topic(item),
        size_bytes=_row_size(entry),
        mp4_url=_mp4_url(entry),
        text=_snippet(item.text),
    )


def _scope_by_source(store: dict[str, Item], source: str) -> dict[str, Item]:
    """Restrict the store to items whose source matches `source` (or raise)."""
    if source not in _SOURCE_SETS:
        raise ValueError(f"source must be one of {', '.join(_SOURCE_SETS)}, got {source!r}")
    chosen = _SOURCE_SETS[source]
    return {item_id: item for item_id, item in store.items() if item.source in chosen}


def _passes_size_cap(size_bytes: int | None, max_size_bytes: int | None) -> bool:
    """True when the row is within the `--max-size` cap.

    No cap → always True. With a cap, an UNKNOWN size (no bitrate/duration) is
    excluded — the same conservative rule `download-videos` applies, so the
    catalog and the fetch select the same under-cap set.
    """
    if max_size_bytes is None:
        return True
    return size_bytes is not None and size_bytes <= max_size_bytes


def _row_matches(
    row: VideoRow,
    *,
    topic: str | None,
    status: str | None,
    max_size_bytes: int | None,
) -> bool:
    """True when `row` passes the topic / status / size filters (all optional)."""
    if topic is not None and row.topic != topic:
        return False
    if status is not None and row.state != status:
        return False
    return _passes_size_cap(row.size_bytes, max_size_bytes)


def list_video_entries(
    store: dict[str, Item],
    *,
    topic: str | None = None,
    status: str | None = None,
    max_size_bytes: int | None = None,
    source: str = "all",
    limit: int | None = None,
) -> list[VideoRow]:
    """Catalog every video media entry across the store, filtered + capped.

    Read-only: no writes, no network, no snapshot. Rows are yielded in store
    order (one per video entry). Filters compose: `topic` (exact
    `primary_topic` match), `status` (one of the four `VideoState`s),
    `max_size_bytes` (known size ≤ cap; unknown-size excluded), `source`
    (bookmark / own-tweet), and `limit` (first N after filtering).
    """
    if status is not None and status not in _VIDEO_STATES:
        raise ValueError(f"--status must be one of {', '.join(_VIDEO_STATES)}, got {status!r}")
    rows: list[VideoRow] = []
    for item_id, item in _scope_by_source(store, source).items():
        for entry in item.media:
            if not _is_video_entry(entry):
                continue
            row = _build_row(item_id, item, entry)  # type: ignore[arg-type]
            if not _row_matches(row, topic=topic, status=status, max_size_bytes=max_size_bytes):
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def row_to_json(row: VideoRow) -> dict[str, object]:
    """Serialise a `VideoRow` to the stable machine schema for `--json`.

    Fields: `id, url, state, topic, size_bytes, mp4_url, text`. `size_bytes`
    and `mp4_url` are `null` when unknown / poster-era; `topic` renders as `—`
    when the item carries no `primary_topic` (matching the human column).
    """
    return {
        "id": row.id,
        "url": row.url,
        "state": row.state,
        "topic": row.topic if row.topic is not None else "—",
        "size_bytes": row.size_bytes,
        "mp4_url": row.mp4_url,
        "text": row.text,
    }


def _format_size(size_bytes: int | None) -> str:
    """Human-readable decimal size (`12.3 MB`), or `unknown` when None."""
    if size_bytes is None:
        return "unknown"
    for unit, multiplier in _SIZE_UNITS:
        if size_bytes >= multiplier:
            if unit == "B":
                return f"{size_bytes} B"
            return f"{size_bytes / multiplier:.1f} {unit}"
    return f"{size_bytes} B"


def format_video_table(rows: list[VideoRow]) -> str:
    """Render the catalog as an aligned human table (headers + one row each)."""
    if not rows:
        return "No hay vídeos."
    headers = ("ID", "STATE", "SIZE", "TOPIC", "TEXT")
    display = [
        (row.id, row.state, _format_size(row.size_bytes), row.topic or "—", row.text)
        for row in rows
    ]
    widths = [
        max(len(headers[col]), *(len(r[col]) for r in display)) for col in range(len(headers))
    ]

    def _fmt(cells: tuple[str, ...]) -> str:
        return "  ".join(cell.ljust(widths[col]) for col, cell in enumerate(cells))

    lines = [_fmt(headers)]
    lines.extend(_fmt(row) for row in display)
    return "\n".join(lines)
