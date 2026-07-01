"""Tests for `xbrain.digest` — the `digest-video` orchestration (#44 PR2).

`digest_videos` turns bookmarked videos into transcripts attached to the item as
an `x_video` content source, via: ephemeral fetch (reusing PR1's `fetch_videos`)
→ external transcribe → `attach_transcript` → discard the bytes. The load-bearing
behaviours are exercised here with INJECTED fakes (no real network, no real
subprocess, no real downloads):

- **Dedup by video identity** — N bookmarks of the same video fetch + transcribe
  ONCE; every referencing item gets the same transcript source.
- **Idempotency** — an item already carrying a fresh `x_video` source is skipped
  unless `--force`.
- **No-speech** — a `has_speech=False` transcript is attached (empty text +
  marker), never a hard failure.
- **Ephemeral cleanup** — the temp video is discarded even when transcription
  fails; no bytes persist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.digest import (
    DigestReport,
    _video_key,
    attach_transcript,
    digest_videos,
    group_items_by_video,
)
from xbrain.models import (
    Author,
    ContentSourceSuccess,
    Item,
    MediaVideoPending,
)
from xbrain.transcribe import Segment, Transcript, TranscriberFailed, TranscriberNotFound
from xbrain.video_fetch import FetchReport, FetchResult

# Two DISTINCT signed URLs for the SAME underlying video (rotating `?tag=` +
# filename) — the whole point of keying on the stable path id, not the URL.
_VIDEO_A_URL_1 = "https://video.twimg.com/amplify_video/1500/vid/720/aaa.mp4?tag=16"
_VIDEO_A_URL_2 = "https://video.twimg.com/amplify_video/1500/vid/1080/bbb.mp4?tag=21"
_VIDEO_B_URL = "https://video.twimg.com/ext_tw_video/2600/vid/480/ccc.mp4?tag=12"
_POSTER = "https://pbs.twimg.com/ext_tw_video_thumb/1/img/P.jpg"


def _item(item_id: str, url: str, *, source: str = "bookmark") -> Item:
    item = Item(
        id=item_id,
        source=source,  # type: ignore[arg-type]
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )
    item.media = [MediaVideoPending(url=url, thumbnail_url=_POSTER)]
    return item


def _speech(text: str = "the transcript") -> Transcript:
    return Transcript(
        text=text,
        segments=[Segment(0.0, 1.0, text)],
        language="en",
        has_speech=True,
    )


def _silence() -> Transcript:
    return Transcript(text="", segments=[], language=None, has_speech=False)


class _FakeFetch:
    """A `fetch_videos` stand-in: writes a dummy mp4 per requested id, records the
    dest dir it was handed, and returns a `FetchReport`. `fail_ids` mark ids whose
    download 'fails' (skipped write + a failed result)."""

    def __init__(self, fail_ids: set[str] | None = None):
        self.fail_ids = fail_ids or set()
        self.dest_dirs: list[Path] = []
        self.fetched_ids: list[str] = []

    def __call__(self, store: dict, ids: list[str], dest_dir: Path) -> FetchReport:
        dest_dir = Path(dest_dir)
        self.dest_dirs.append(dest_dir)
        report = FetchReport()
        for item_id in ids:
            if item_id in self.fail_ids:
                report.results.append(
                    FetchResult(item_id, "failed", reason="http_5xx", error="boom")
                )
                continue
            path = dest_dir / f"{item_id}.mp4"
            path.write_bytes(b"\x00\x00\x00\x18ftypmp42 fake video bytes")
            self.fetched_ids.append(item_id)
            report.results.append(
                FetchResult(item_id, "fetched", path=str(path), size_bytes=path.stat().st_size)
            )
        return report


# ------------------------------------------------------------ _video_key


def test_video_key_is_stable_across_rotating_url():
    """Two signed URLs for the SAME amplify_video id resolve to ONE key — the
    dedup contract. The full URL (query/signing/filename) is unstable."""
    assert _video_key(_VIDEO_A_URL_1) == _video_key(_VIDEO_A_URL_2) == "amplify_video/1500"


def test_video_key_distinguishes_categories_and_ids():
    assert _video_key(_VIDEO_B_URL) == "ext_tw_video/2600"
    assert _video_key("https://video.twimg.com/tweet_video/GABC.mp4") == "tweet_video/GABC.mp4"


def test_video_key_fallback_strips_query_for_unknown_pattern():
    """An unrecognised host/path still de-dups on the path (query stripped), the
    safe direction: identical media path → same key even if signing rotates."""
    a = _video_key("https://cdn.example.com/media/clip.mp4?sig=1")
    b = _video_key("https://cdn.example.com/media/clip.mp4?sig=2")
    assert a == b == "cdn.example.com/media/clip.mp4"


# ------------------------------------------------------------ group_items_by_video


def test_group_items_by_video_dedups_same_video():
    store = {
        "a1": _item("a1", _VIDEO_A_URL_1),
        "a2": _item("a2", _VIDEO_A_URL_2),  # same video, different URL
        "b1": _item("b1", _VIDEO_B_URL),
    }
    groups = group_items_by_video(store, ["a1", "a2", "b1"])
    assert groups == {"amplify_video/1500": ["a1", "a2"], "ext_tw_video/2600": ["b1"]}


def test_group_items_by_video_skips_unknown_and_non_mp4():
    from xbrain.models import MediaVideoPending

    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    store["hls"] = _item("hls", "https://video.twimg.com/ext_tw_video/9/pl/x.m3u8?c=1")
    poster = _item("poster", _POSTER)
    poster.media = [MediaVideoPending(url=_POSTER, thumbnail_url=_POSTER)]
    store["poster"] = poster
    groups = group_items_by_video(store, ["a1", "hls", "poster", "ghost"])
    assert groups == {"amplify_video/1500": ["a1"]}


def test_group_items_by_video_dedups_repeated_ids():
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    assert group_items_by_video(store, ["a1", "a1"]) == {"amplify_video/1500": ["a1"]}


# ------------------------------------------------------------ attach_transcript


def test_attach_transcript_adds_x_video_source_to_each_item():
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    count = attach_transcript(store, ["a1", "a2"], _speech("shared transcript"))
    assert count == 2
    for item_id in ("a1", "a2"):
        sources = store[item_id].content.sources
        assert len(sources) == 1
        src = sources[0]
        assert isinstance(src, ContentSourceSuccess)
        assert src.kind == "x_video"
        assert src.text == "shared transcript"
        assert src.has_speech is True
        assert src.language == "en"


def test_attach_transcript_no_speech_carries_empty_text_and_marker():
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    attach_transcript(store, ["a1"], _silence())
    src = store["a1"].content.sources[0]
    assert src.text == ""
    assert src.has_speech is False


def test_attach_transcript_preserves_existing_article_source():
    """Attaching a transcript must not clobber an already-fetched article body —
    the x_video source is added alongside it."""
    from xbrain.models import Content

    item = _item("a1", _VIDEO_A_URL_1)
    item.content = Content(
        fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        sources=[ContentSourceSuccess(kind="external_article", url="u", text="article body")],
    )
    store = {"a1": item}
    attach_transcript(store, ["a1"], _speech())
    kinds = [s.kind for s in store["a1"].content.sources]
    assert kinds == ["external_article", "x_video"]


def test_attach_transcript_replaces_prior_x_video_source():
    """A re-attach (force re-run) REPLACES the stale x_video source rather than
    appending a duplicate — one transcript per item."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    attach_transcript(store, ["a1"], _speech("v1"))
    attach_transcript(store, ["a1"], _speech("v2"))
    sources = store["a1"].content.sources
    assert len(sources) == 1
    assert sources[0].text == "v2"


# ------------------------------------------------------------ digest_videos (orchestration)


def test_dedup_fetches_and_transcribes_once_attaches_to_all(tmp_path: Path):
    """Two items bookmarking the SAME video: ONE fetch + ONE transcribe, both
    items carry the transcript (the core #44 success criterion)."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    fetch = _FakeFetch()
    transcribe_calls: list[Path] = []

    def _transcribe(path: Path) -> Transcript:
        transcribe_calls.append(Path(path))
        return _speech("one talk")

    report = digest_videos(
        store, ["a1", "a2"], fetch_fn=fetch, transcribe_fn=_transcribe, temp_root=tmp_path
    )
    assert len(fetch.fetched_ids) == 1  # fetched once
    assert len(transcribe_calls) == 1  # transcribed once
    assert report.transcribed == 2  # both items carry it
    assert report.videos_fetched == 1
    assert store["a1"].content.sources[0].text == "one talk"
    assert store["a2"].content.sources[0].text == "one talk"


def test_no_speech_video_is_attached_not_failed(tmp_path: Path):
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
    )
    assert report.no_speech == 1
    assert report.transcribed == 0
    assert report.failed == 0
    src = store["a1"].content.sources[0]
    assert src.kind == "x_video"
    assert src.has_speech is False


def test_idempotent_skips_already_digested(tmp_path: Path):
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    fetch = _FakeFetch()
    digest_videos(
        store, ["a1"], fetch_fn=fetch, transcribe_fn=lambda _p: _speech(), temp_root=tmp_path
    )
    # Second run: already has a fresh x_video source → skipped, no re-fetch.
    fetch2 = _FakeFetch()
    report = digest_videos(
        store, ["a1"], fetch_fn=fetch2, transcribe_fn=lambda _p: _speech(), temp_root=tmp_path
    )
    assert report.already == 1
    assert report.transcribed == 0
    assert fetch2.fetched_ids == []  # nothing fetched on the idempotent re-run


def test_force_redigests_already_digested(tmp_path: Path):
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech("old"),
        temp_root=tmp_path,
    )
    report = digest_videos(
        store,
        ["a1"],
        force=True,
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech("new"),
        temp_root=tmp_path,
    )
    assert report.transcribed == 1
    assert store["a1"].content.sources[0].text == "new"


def test_partial_group_transcribes_once_for_the_missing_item(tmp_path: Path):
    """A group where item A is already digested and item B (same video) is not:
    fetch+transcribe once, attach to B; A counts as already."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech("t"),
        temp_root=tmp_path,
    )
    fetch = _FakeFetch()
    report = digest_videos(
        store,
        ["a1", "a2"],
        fetch_fn=fetch,
        transcribe_fn=lambda _p: _speech("t"),
        temp_root=tmp_path,
    )
    assert report.already == 1
    assert report.transcribed == 1
    assert len(fetch.fetched_ids) == 1
    assert store["a2"].content.sources[0].text == "t"


def test_fetch_failure_is_recorded_not_fatal(tmp_path: Path):
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "b1": _item("b1", _VIDEO_B_URL)}
    fetch = _FakeFetch(fail_ids={"a1"})
    report = digest_videos(
        store, ["a1", "b1"], fetch_fn=fetch, transcribe_fn=lambda _p: _speech(), temp_root=tmp_path
    )
    assert report.failed == 1
    assert report.transcribed == 1  # b1 still processed
    assert store["a1"].content is None  # nothing attached to the failed one
    assert store["b1"].content.sources[0].kind == "x_video"


def test_temp_video_discarded_after_transcription(tmp_path: Path):
    """No video bytes persist: the fetched mp4 exists AT transcribe time, and the
    whole temp dir is gone after the run (ephemeral, one-at-a-time)."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    seen: dict[str, bool] = {}

    def _transcribe(path: Path) -> Transcript:
        seen["existed_during"] = Path(path).exists()
        return _speech()

    fetch = _FakeFetch()
    digest_videos(store, ["a1"], fetch_fn=fetch, transcribe_fn=_transcribe, temp_root=tmp_path)
    assert seen["existed_during"] is True  # the fetch really landed the bytes
    assert not fetch.dest_dirs[0].exists()  # temp dir cleaned up
    assert list(tmp_path.rglob("*.mp4")) == []  # no bytes left anywhere


def test_temp_dir_cleaned_even_when_transcription_raises(tmp_path: Path):
    """A missing transcriber (TranscriberNotFound) aborts the run — but the temp
    video is still discarded (cleanup on failure)."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    fetch = _FakeFetch()

    def _boom(_path: Path) -> Transcript:
        raise TranscriberNotFound("parakeet-mlx not found")

    with pytest.raises(TranscriberNotFound):
        digest_videos(store, ["a1"], fetch_fn=fetch, transcribe_fn=_boom, temp_root=tmp_path)
    assert not fetch.dest_dirs[0].exists()
    assert list(tmp_path.rglob("*.mp4")) == []


def test_per_video_transcriber_failure_is_recorded_not_fatal(tmp_path: Path):
    """A malformed-output TranscriberFailed for ONE video is per-video (recorded +
    the batch continues), unlike a missing binary which aborts."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "b1": _item("b1", _VIDEO_B_URL)}

    def _transcribe(path: Path) -> Transcript:
        if "a1" in str(path):
            raise TranscriberFailed("garbage output")
        return _speech()

    report = digest_videos(
        store, ["a1", "b1"], fetch_fn=_FakeFetch(), transcribe_fn=_transcribe, temp_root=tmp_path
    )
    assert report.failed == 1
    assert report.transcribed == 1
    assert store["a1"].content is None


def test_report_grouping_counts(tmp_path: Path):
    store = {
        "a1": _item("a1", _VIDEO_A_URL_1),
        "a2": _item("a2", _VIDEO_A_URL_2),
        "b1": _item("b1", _VIDEO_B_URL),
    }
    report = digest_videos(
        store,
        ["a1", "a2", "b1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
    )
    assert isinstance(report, DigestReport)
    assert report.total_items == 3
    assert report.video_count == 2  # M videos ← N items


def test_no_video_items_reported(tmp_path: Path):
    """An item with no fetchable mp4 (poster-era) is reported as skipped_no_video,
    not silently dropped."""
    from xbrain.models import MediaVideoPending

    poster = _item("p", _POSTER)
    poster.media = [MediaVideoPending(url=_POSTER, thumbnail_url=_POSTER)]
    store = {"p": poster}
    report = digest_videos(
        store, ["p"], fetch_fn=_FakeFetch(), transcribe_fn=lambda _p: _speech(), temp_root=tmp_path
    )
    assert report.skipped_no_video == 1
    assert report.transcribed == 0
