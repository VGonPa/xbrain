"""Tests for `xbrain.video_select` — the read-only video catalog for `list-videos`.

Pure selection/derivation logic: given a store of `Item`s, produce one
`VideoRow` per video media entry, with a derived state (downloaded / failed /
pending / poster-era), an estimated (or exact, for downloaded) size, the item's
`primary_topic`, the resolved stream URL and a short text snippet. No network,
no writes — `list_video_entries` is a pure function over the in-memory store.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from xbrain.models import (
    Author,
    Enrichment,
    Item,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
)
from xbrain.video_select import (
    VideoRow,
    _format_size,
    format_video_table,
    list_video_entries,
    row_to_json,
)

_MP4_URL = "https://video.twimg.com/ext_tw_video/1/vid/720/A.mp4?tag=12"
_POSTER = "https://pbs.twimg.com/ext_tw_video_thumb/1/img/P.jpg"


def _item(
    item_id: str,
    *,
    media: list | None = None,
    source: str = "bookmark",
    text: str = "some note",
    primary_topic: str | None = None,
) -> Item:
    item = Item(
        id=item_id,
        source=source,  # type: ignore[arg-type]
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text,
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    if media is not None:
        item.media = media
    if primary_topic is not None:
        item.enriched = Enrichment(
            enriched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            executor="manual",
            primary_topic=primary_topic,
        )
    return item


def _pending(
    url: str = _MP4_URL, *, bitrate: int | None = 2_176_000, duration: int | None = 30_000
):
    return MediaVideoPending(
        url=url, thumbnail_url=_POSTER, bitrate=bitrate, duration_millis=duration
    )


def _poster_pending():
    # url == thumbnail_url ⇒ un-backfilled poster-era entry.
    return MediaVideoPending(url=_POSTER, thumbnail_url=_POSTER)


def _downloaded(url: str = _MP4_URL, *, bytes_size: int = 1234):
    return MediaVideoDownloaded(
        url=url,
        thumbnail_url=_POSTER,
        bitrate=2_176_000,
        duration_millis=30_000,
        local_path="42/0.mp4",
        bytes_size=bytes_size,
        downloaded_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


def _failed(url: str = _MP4_URL):
    return MediaVideoFailed(
        url=url,
        thumbnail_url=_POSTER,
        bitrate=2_176_000,
        duration_millis=30_000,
        failure_reason="http_5xx",
        error="boom",
        attempts=1,
        last_attempt_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )


# ------------------------------------------------------------ row derivation


def test_pending_mp4_row_is_pending_state_with_estimated_size():
    store = {"1": _item("1", media=[_pending()], primary_topic="ai")}
    rows = list_video_entries(store)
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, VideoRow)
    assert row.id == "1"
    assert row.url == "https://x.com/a/status/1"
    assert row.state == "pending"
    assert row.topic == "ai"
    assert row.mp4_url == _MP4_URL
    # bitrate 2_176_000 bits/s × 30_000 ms / 8000 = 8_160_000 bytes
    assert row.size_bytes == 2_176_000 * 30_000 // 8000
    assert row.text == "some note"


def test_downloaded_row_uses_exact_bytes_not_estimate():
    store = {"42": _item("42", media=[_downloaded(bytes_size=999)])}
    row = list_video_entries(store)[0]
    assert row.state == "downloaded"
    assert row.size_bytes == 999  # exact on-disk size, not the bitrate estimate


def test_failed_row_is_failed_state():
    store = {"7": _item("7", media=[_failed()])}
    row = list_video_entries(store)[0]
    assert row.state == "failed"


def test_poster_era_row_has_no_mp4_url_and_pending_poster_state():
    store = {"9": _item("9", media=[_poster_pending()])}
    row = list_video_entries(store)[0]
    assert row.state == "poster-era"
    assert row.mp4_url is None


def test_unknown_size_pending_reports_none():
    store = {"1": _item("1", media=[_pending(bitrate=None, duration=None)])}
    row = list_video_entries(store)[0]
    assert row.size_bytes is None


def test_missing_topic_is_none():
    store = {"1": _item("1", media=[_pending()])}
    row = list_video_entries(store)[0]
    assert row.topic is None


def test_photos_are_ignored():
    store = {"1": _item("1", media=[MediaPhotoPending(url="https://pbs.twimg.com/x.jpg")])}
    assert list_video_entries(store) == []


def test_snippet_is_truncated_and_single_line():
    long_text = "first line\nsecond line " + "x" * 200
    store = {"1": _item("1", media=[_pending()], text=long_text)}
    row = list_video_entries(store)[0]
    assert "\n" not in row.text
    assert len(row.text) <= 81  # snippet cap + ellipsis
    assert row.text.endswith("…")


# ------------------------------------------------------------ filters


def test_filter_by_topic():
    store = {
        "1": _item("1", media=[_pending()], primary_topic="ai"),
        "2": _item("2", media=[_pending()], primary_topic="climate"),
    }
    rows = list_video_entries(store, topic="ai")
    assert [r.id for r in rows] == ["1"]


def test_filter_by_status():
    store = {
        "1": _item("1", media=[_pending()]),
        "2": _item("2", media=[_downloaded()]),
        "3": _item("3", media=[_failed()]),
    }
    assert [r.id for r in list_video_entries(store, status="downloaded")] == ["2"]
    assert [r.id for r in list_video_entries(store, status="failed")] == ["3"]
    assert [r.id for r in list_video_entries(store, status="pending")] == ["1"]


def test_invalid_status_raises():
    store = {"1": _item("1", media=[_pending()])}
    with pytest.raises(ValueError, match="status"):
        list_video_entries(store, status="bogus")


def test_filter_by_max_size_excludes_too_large_and_unknown():
    store = {
        "small": _item("small", media=[_pending(bitrate=8000, duration=1000)]),  # 1000 bytes
        "big": _item("big", media=[_pending(bitrate=2_176_000, duration=30_000)]),  # ~8 MB
        "unknown": _item("unknown", media=[_pending(bitrate=None, duration=None)]),
    }
    rows = list_video_entries(store, max_size_bytes=5000)
    # only the small, known-size video is under the cap; unknown is excluded.
    assert [r.id for r in rows] == ["small"]


def test_filter_by_source():
    store = {
        "b": _item("b", media=[_pending()], source="bookmark"),
        "t": _item("t", media=[_pending()], source="own_tweet"),
    }
    assert [r.id for r in list_video_entries(store, source="bookmarks")] == ["b"]
    assert [r.id for r in list_video_entries(store, source="tweets")] == ["t"]
    assert {r.id for r in list_video_entries(store, source="all")} == {"b", "t"}


def test_invalid_source_raises():
    store = {"1": _item("1", media=[_pending()])}
    with pytest.raises(ValueError, match="source"):
        list_video_entries(store, source="nope")


def test_limit_caps_row_count():
    store = {str(i): _item(str(i), media=[_pending()]) for i in range(5)}
    assert len(list_video_entries(store, limit=2)) == 2


# ------------------------------------------------------------ json + table


def test_row_to_json_schema_is_stable():
    store = {"1": _item("1", media=[_pending()], primary_topic="ai")}
    row = list_video_entries(store)[0]
    payload = row_to_json(row)
    assert set(payload) == {"id", "url", "state", "topic", "size_bytes", "mp4_url", "text"}
    # round-trips through json cleanly
    assert json.loads(json.dumps(payload)) == payload
    assert payload["topic"] == "ai"


def test_row_to_json_missing_topic_is_null():
    """The machine schema emits JSON null for an absent topic (the human table
    renders "—"; the array PR2/PR3 lock onto must stay null, not a sentinel)."""
    store = {"1": _item("1", media=[_pending()])}
    payload = row_to_json(list_video_entries(store)[0])
    assert payload["topic"] is None


def test_row_to_json_poster_era_mp4_url_is_null():
    store = {"9": _item("9", media=[_poster_pending()])}
    payload = row_to_json(list_video_entries(store)[0])
    assert payload["mp4_url"] is None
    assert payload["size_bytes"] is None


def test_format_video_table_has_headers_and_rows():
    store = {"1": _item("1", media=[_pending()], primary_topic="ai")}
    table = format_video_table(list_video_entries(store))
    assert "ID" in table and "STATE" in table and "TOPIC" in table
    assert "pending" in table
    assert "ai" in table


def test_format_video_table_missing_topic_shows_dash():
    """The human table keeps the "—" sentinel even though --json emits null."""
    store = {"1": _item("1", media=[_pending()])}
    assert "—" in format_video_table(list_video_entries(store))


def test_format_video_table_empty():
    assert "No" in format_video_table([])


def test_format_size_rendering():
    assert _format_size(None) == "unknown"
    assert _format_size(2_000_000_000) == "2.0 GB"
    assert _format_size(1_500_000) == "1.5 MB"
    assert _format_size(2_048) == "2.0 KB"
    assert _format_size(500) == "500 B"
