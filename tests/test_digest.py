"""Tests for `xbrain.digest` — the `digest-video` orchestration (#44 PR2).

`digest_videos` turns bookmarked videos into transcripts attached to the item as
an `x_video` content source, via: ephemeral fetch (reusing PR1's `fetch_videos`)
→ external transcribe → `attach_transcript` → discard the bytes. The load-bearing
behaviours are exercised here with INJECTED fakes (no real network, no real
subprocess, no real downloads):

- **Dedup by video identity** — N bookmarks of the same video fetch + transcribe
  ONCE; every referencing item gets the same transcript source.
- **Idempotency** — an item already carrying an `x_video` source is skipped
  unless `--force`.
- **No-speech** — a `has_speech=False` transcript is attached (empty text +
  marker), never a hard failure.
- **Ephemeral cleanup** — the temp video is discarded even when transcription
  fails; no bytes persist.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.digest import (
    DigestReport,
    VisualConfig,
    _video_key,
    attach_transcript,
    digest_videos,
    format_digest_summary,
    group_items_by_video,
)
from xbrain.models import (
    FRAME_CAPTION_CONTRACT,
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    MediaVideoPending,
)
from xbrain.transcribe import Segment, Transcript, TranscriberFailed, TranscriberNotFound
from xbrain.video_fetch import FetchReport, FetchResult
from xbrain.video_frames import (
    FrameExtractionFailed,
    FrameExtractionToolNotFound,
    KeyFrame,
)
from xbrain.vision import VisionFailed, VisionNotFound

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


def test_attach_transcript_passes_title(tmp_path: Path):
    """A transcript `title` (item 14) is carried onto the `x_video` source's
    `title` field for PR3's digest rendering; None when the transcript has none."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    titled = Transcript(text="body", segments=[], language="en", has_speech=True, title="Ep 12")
    attach_transcript(store, ["a1"], titled)
    attach_transcript(store, ["a2"], _speech())  # no title
    assert store["a1"].content.sources[0].title == "Ep 12"
    assert store["a2"].content.sources[0].title is None


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


def test_attach_transcript_bumps_fetched_at_on_existing_content():
    """Appending a transcript to already-fetched content bumps `content.fetched_at`
    so `enrich` treats the new transcript as unprocessed and RE-ENRICHES the item
    (PR3 re-enrichment trigger). Without the bump the transcript looks already
    processed and the video keeps topic "—"."""
    from xbrain.models import Content, ContentSourceSuccess

    old = datetime(2026, 5, 16, tzinfo=timezone.utc)
    item = _item("a1", _VIDEO_A_URL_1)
    item.content = Content(
        fetched_at=old,
        sources=[ContentSourceSuccess(kind="external_article", url="u", text="body")],
    )
    store = {"a1": item}
    attach_transcript(store, ["a1"], _speech())
    assert store["a1"].content.fetched_at > old


def test_attach_transcript_sets_fetched_at_when_no_prior_content():
    """A first-ever attach (content was None) records a fresh, UTC-aware fetch time.

    `Content.fetched_at` is a required field, so `is not None` is vacuous — assert
    it is timezone-aware (the enrich re-enrichment comparison needs aware datetimes)
    and stamped at attach time (recent)."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    attach_transcript(store, ["a1"], _speech())
    fetched_at = store["a1"].content.fetched_at
    assert fetched_at.tzinfo is not None  # UTC-aware, not a naive datetime
    assert abs((datetime.now(timezone.utc) - fetched_at).total_seconds()) < 5  # stamped now


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
    assert report.videos_transcribed == 1
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


def test_representative_fetch_uses_a_needing_item_not_a_stale_already_digested_one(tmp_path: Path):
    """A group's video is fetched via a NEEDING item (`needing[0]`), never an
    already-digested member first in the group. Fetching via an already-digested
    item risks its stale/expired signed URL 403-ing the whole group even when a
    needing item carries a fresh one — so `representative` must be a needing id."""
    # a1 is already digested (its signed URL is the "stale" one); a2 needs it.
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    digest_videos(
        store, ["a1"], fetch_fn=_FakeFetch(), transcribe_fn=lambda _p: _speech(), temp_root=tmp_path
    )
    fetch = _FakeFetch()
    # Group order puts the already-digested a1 first; the fetch must still target a2.
    digest_videos(
        store, ["a1", "a2"], fetch_fn=fetch, transcribe_fn=lambda _p: _speech(), temp_root=tmp_path
    )
    assert fetch.fetched_ids == ["a2"]  # needing[0], not the stale a1


def test_at_most_one_video_on_disk_across_groups(tmp_path: Path):
    """Ephemeral, one-at-a-time (test 4a): while transcribing the 2nd video, the
    1st group's mp4 is already gone — at most one *.mp4 exists at any moment.
    Deleting the per-file unlink in `_transcribe_and_discard` must fail this."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "b1": _item("b1", _VIDEO_B_URL)}
    fetch = _FakeFetch()
    max_seen = 0

    def _transcribe(path: Path) -> Transcript:
        nonlocal max_seen
        # Count the mp4s present in the shared dest dir at transcribe time.
        present = len(list(Path(path).parent.glob("*.mp4")))
        max_seen = max(max_seen, present)
        return _speech()

    digest_videos(
        store, ["a1", "b1"], fetch_fn=fetch, transcribe_fn=_transcribe, temp_root=tmp_path
    )
    assert max_seen == 1  # never two videos on disk simultaneously
    assert list(tmp_path.rglob("*.mp4")) == []


def test_unknown_id_and_no_video_are_distinguished(tmp_path: Path):
    """An id absent from the store (`skipped_unknown`) is reported separately from
    a stored item with no fetchable mp4 (`skipped_no_video`) — not lumped."""
    poster = _item("p", _POSTER)
    poster.media = [MediaVideoPending(url=_POSTER, thumbnail_url=_POSTER)]
    store = {"p": poster}
    report = digest_videos(
        store,
        ["p", "ghost"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
    )
    assert report.skipped_no_video == 1  # p: in store, no mp4
    assert report.skipped_unknown == 1  # ghost: absent from store


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
        # Key on the EXACT fetched filename (`<id>.mp4`), never a substring of the
        # full path — the random `xbrain-digest-XXXXXX` temp dir can itself contain
        # "a1"/"b1", which would misfire the failure onto both videos (a flake).
        if path.stem == "a1":
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
    assert report.videos_transcribed == 2  # both distinct videos processed this run


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


# ------------------------------------------------------------ visual layer (--frames, PR4)


def _make_frame(directory: Path, index: int, timestamp: float) -> KeyFrame:
    """A KeyFrame whose image file really exists (so persistence can copy it).

    The bytes are dummy — the digest tests inject a fake classifier/describer, so
    no real image decode or vision runs here."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"frame-{index:05d}.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n slide bytes")
    return KeyFrame(timestamp=timestamp, path=path)


class _FakeVisual:
    """Injected `extract_fn` + `describe_fn` + `classify_fn` for the visual layer.

    `extract_fn` writes its frame files INTO the fetched video's parent (the
    ephemeral temp dir) — mirroring the real ffmpeg path — so the outer cleanup
    reclaims them. Records the calls so tests can assert extract/describe ran (or
    did NOT, on the non-frames / talking-head paths).

    The two reducers keep the FIRST `slides_keep` / `footage_keep` frames (`None`
    = keep all) and record which one ran in `reduce_calls`, so a test can tell the
    slide budget from the footage budget by the frames that come out."""

    def __init__(
        self,
        *,
        classification: str = "slides",
        n_frames: int = 2,
        describe=None,
        slides_keep: int | None = None,
        footage_keep: int | None = None,
    ):
        self.classification = classification
        self.n_frames = n_frames
        self._describe = describe or (lambda path: f"description of {Path(path).name}")
        self.slides_keep = slides_keep
        self.footage_keep = footage_keep
        self.extract_calls: list[Path] = []
        self.describe_calls: list[Path] = []
        self.reduce_calls: list[str] = []
        self.classify_calls = 0

    def extract(self, path: Path) -> list[KeyFrame]:
        self.extract_calls.append(Path(path))
        frames_dir = Path(path).parent / "xbrain-frames-fake"
        return [_make_frame(frames_dir, i, float(i * 10)) for i in range(self.n_frames)]

    def classify(self, frames):
        self.classify_calls += 1
        return self.classification

    def describe(self, path: Path) -> str:
        self.describe_calls.append(Path(path))
        return self._describe(path)

    def reduce_slides(self, frames: list[KeyFrame]) -> list[KeyFrame]:
        self.reduce_calls.append("slides")
        return frames if self.slides_keep is None else frames[: self.slides_keep]

    def reduce_footage(self, frames: list[KeyFrame]) -> list[KeyFrame]:
        self.reduce_calls.append("footage")
        return frames if self.footage_keep is None else frames[: self.footage_keep]

    def config(self, media_root: Path) -> VisualConfig:
        return VisualConfig(
            media_root=media_root,
            extract_fn=self.extract,
            describe_fn=self.describe,
            classify_fn=self.classify,
            reduce_fn=self.reduce_slides,
            footage_reduce_fn=self.reduce_footage,
        )


def test_slide_video_describes_and_attaches_frames(tmp_path: Path):
    """A slide-classified video: its key frames are described and recorded as
    `VideoFrame`s on the item's `x_video` source, and the slide images are
    persisted under `media_root/<id>/frames/<n>.png` for the generator to embed."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    media_root = tmp_path / "media"
    visual = _FakeVisual(classification="slides", n_frames=2)

    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech("the talk"),
        temp_root=tmp_path,
        visual=visual.config(media_root),
    )
    frames = store["a1"].content.sources[0].frames
    assert [f.description for f in frames] == [
        "description of frame-00000.png",
        "description of frame-00001.png",
    ]
    assert [f.local_path for f in frames] == ["a1/frames/0.png", "a1/frames/1.png"]
    # the slide bytes are persisted where generate mirrors from
    assert (media_root / "a1" / "frames" / "0.png").exists()
    assert (media_root / "a1" / "frames" / "1.png").exists()
    assert report.visual_slides == 1
    assert report.visual_skipped == 0
    assert visual.describe_calls  # vision ran on the slides


def test_talking_head_video_skips_visual_and_logs(tmp_path: Path, caplog):
    """A talking-head-classified video: the visual layer is SKIPPED and the reason
    logged — never a silent drop — and no vision call is wasted."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual(classification="talking_head", n_frames=3)

    with caplog.at_level(logging.INFO):
        report = digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _speech(),
            temp_root=tmp_path,
            visual=visual.config(tmp_path / "media"),
        )
    assert store["a1"].content.sources[0].frames == []  # no slides embedded
    assert visual.describe_calls == []  # vision NOT wasted on an interview
    assert report.visual_skipped == 1
    assert report.visual_slides == 0
    assert "visual layer skipped (talking-head)" in caplog.text
    assert store["a1"].content.sources[0].text  # the transcript is still attached


def test_non_frames_run_never_touches_ffmpeg_or_vision(tmp_path: Path):
    """The visual layer is fully opt-in: with no `visual` config, ffmpeg/vision are
    never invoked and the source carries no frames (the default path is inert)."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual()

    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=None,
    )
    assert visual.extract_calls == []
    assert visual.describe_calls == []
    assert store["a1"].content.sources[0].frames == []


def test_dedup_describes_slides_once_persists_per_item(tmp_path: Path):
    """Two items bookmarking the same slide video: the frames are extracted +
    described ONCE, but persisted + attached PER item (each with its own
    `<id>/frames/` path so the per-item embed resolves)."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    media_root = tmp_path / "media"
    visual = _FakeVisual(classification="slides", n_frames=2)

    digest_videos(
        store,
        ["a1", "a2"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=visual.config(media_root),
    )
    assert len(visual.describe_calls) == 2  # 2 frames described ONCE (not per item)
    assert [f.local_path for f in store["a1"].content.sources[0].frames] == [
        "a1/frames/0.png",
        "a1/frames/1.png",
    ]
    assert [f.local_path for f in store["a2"].content.sources[0].frames] == [
        "a2/frames/0.png",
        "a2/frames/1.png",
    ]
    assert (media_root / "a2" / "frames" / "1.png").exists()


def test_vision_failure_drops_visual_layer_but_keeps_transcript(tmp_path: Path, caplog):
    """A per-image `VisionFailed` is recorded and the visual layer dropped for that
    video — never a silent partial — while the transcript still attaches (the audio
    digest is independent of the visual layer)."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}

    def _boom(_path):
        raise VisionFailed("model crashed")

    visual = _FakeVisual(classification="slides", describe=_boom)
    with caplog.at_level(logging.WARNING):
        report = digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _speech("kept"),
            temp_root=tmp_path,
            visual=visual.config(tmp_path / "media"),
        )
    assert store["a1"].content.sources[0].frames == []  # visual dropped
    assert store["a1"].content.sources[0].text == "kept"  # transcript survives
    assert report.transcribed == 1
    assert report.visual_slides == 0
    assert report.visual_skipped == 0  # a per-video failure is NOT a talking-head skip


def test_frame_extraction_failure_is_per_video_not_fatal(tmp_path: Path):
    """A per-video `FrameExtractionFailed` (a bad mp4 ffmpeg rejects) drops that
    video's visual layer and continues — the transcript still attaches."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}

    def _boom(_path):
        raise FrameExtractionFailed("Invalid data")

    visual = VisualConfig(
        media_root=tmp_path / "media",
        extract_fn=_boom,
        describe_fn=lambda _p: "x",
        classify_fn=lambda _f: "slides",
    )
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=visual,
    )
    assert store["a1"].content.sources[0].frames == []
    assert report.transcribed == 1  # transcript still landed
    assert report.visual_skipped == 0  # a per-video failure is NOT a talking-head skip


def test_missing_ffmpeg_aborts_the_run(tmp_path: Path):
    """A missing ffmpeg (`FrameExtractionToolNotFound`) is a global config error —
    it ABORTS the whole run, like a missing transcriber, not a per-video skip."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}

    def _boom(_path):
        raise FrameExtractionToolNotFound("ffmpeg not found")

    visual = VisualConfig(
        media_root=tmp_path / "media",
        extract_fn=_boom,
        describe_fn=lambda _p: "x",
        classify_fn=lambda _f: "slides",
    )
    with pytest.raises(FrameExtractionToolNotFound):
        digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _speech(),
            temp_root=tmp_path,
            visual=visual,
        )


def test_missing_vision_binary_aborts_the_run(tmp_path: Path):
    """A missing/unconfigured vision binary (`VisionNotFound`) aborts the run — a
    global config error, not a per-video skip."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}

    def _boom(_path):
        raise VisionNotFound("no [vision].command configured")

    visual = _FakeVisual(classification="slides", describe=_boom)
    with pytest.raises(VisionNotFound):
        digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _speech(),
            temp_root=tmp_path,
            visual=visual.config(tmp_path / "media"),
        )


def test_frame_temp_files_discarded_after_run(tmp_path: Path):
    """The extracted frame images are ephemeral: only the KEPT slides persist under
    media_root; nothing lingers in the digest temp tree."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual(classification="slides", n_frames=2)
    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    # the ephemeral frame files are gone; the persisted slides live under media/
    assert list((tmp_path).glob("xbrain-digest-*")) == []
    assert (tmp_path / "media" / "a1" / "frames" / "0.png").exists()


def test_empty_extraction_skips_and_logs_not_talking_head(tmp_path: Path, caplog):
    """When ffmpeg selects NO frames, the visual layer is a NON-content `skipped`
    (logged), NOT bucketed as a talking-head content decision — so an operator can
    tell "no frames found" from "genuine interview". No classifier/vision call is
    made and the transcript still attaches. `visual_skipped` (the talking-head
    tally) stays 0."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    describe_calls: list = []
    classify_calls: list = []

    visual = VisualConfig(
        media_root=tmp_path / "media",
        extract_fn=lambda _p: [],  # ffmpeg found nothing to select
        describe_fn=lambda p: describe_calls.append(p) or "x",
        classify_fn=lambda f: classify_calls.append(f) or "slides",
    )
    with caplog.at_level(logging.INFO):
        report = digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _speech("kept"),
            temp_root=tmp_path,
            visual=visual,
        )
    assert store["a1"].content.sources[0].frames == []
    assert store["a1"].content.sources[0].text == "kept"  # transcript survives
    assert describe_calls == []  # no vision call on an empty extraction
    assert classify_calls == []  # classifier not even consulted
    assert report.visual_slides == 0
    assert report.visual_skipped == 0  # NOT a talking-head decision
    assert "no key frames extracted" in caplog.text


def test_all_unreadable_frames_skips_and_logs_not_talking_head(tmp_path: Path, caplog):
    """Every extracted frame unreadable → classify returns 'unreadable' → a
    non-content `skipped` (logged with the count), NOT a talking-head. No vision
    call is wasted; `visual_skipped` stays 0; the transcript still attaches."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    describe_calls: list = []
    visual = _FakeVisual(classification="unreadable", n_frames=3)
    visual._describe = lambda p: describe_calls.append(p) or "x"

    with caplog.at_level(logging.WARNING):
        report = digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _speech("kept"),
            temp_root=tmp_path,
            visual=visual.config(tmp_path / "media"),
        )
    assert store["a1"].content.sources[0].frames == []
    assert describe_calls == []  # vision NOT wasted on unreadable frames
    assert report.visual_slides == 0
    assert report.visual_skipped == 0  # NOT a talking-head decision
    assert "unreadable" in caplog.text


def test_silent_slide_deck_keeps_frames(tmp_path: Path):
    """A SILENT slide deck (no speech) still gets its slides described + embedded —
    the visual layer is exactly where a screen-only video carries its content."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual(classification="slides", n_frames=1)
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    src = store["a1"].content.sources[0]
    assert src.has_speech is False
    assert len(src.frames) == 1  # slides kept despite no speech
    assert report.no_speech == 1
    assert report.visual_slides == 1


# ------------------------------------------------------------ silent footage (non-slide, no speech)


def _digest_messages(caplog) -> list[str]:
    """The fully-formatted messages `xbrain.digest` logged — compared whole, never
    searched for a fragment."""
    return [r.getMessage() for r in caplog.records if r.name == "xbrain.digest"]


def test_silent_non_slide_video_describes_its_frames_as_footage(tmp_path: Path, caplog):
    """A SILENT video the classifier calls `talking_head` is not a talking head: with
    no speech the frames are the only evidence, so they are kept, described and
    embedded as `footage` — never attached as a hollow empty source."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    media_root = tmp_path / "media"
    visual = _FakeVisual(classification="talking_head", n_frames=3, footage_keep=2)

    with caplog.at_level(logging.INFO, logger="xbrain.digest"):
        report = digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _silence(),
            temp_root=tmp_path,
            visual=visual.config(media_root),
        )
    src = store["a1"].content.sources[0]
    assert src.kind == "x_video"
    assert src.has_speech is False
    assert src.text == ""
    assert [(f.timestamp, f.local_path, f.description) for f in src.frames] == [
        (0.0, "a1/frames/0.png", "description of frame-00000.png"),
        (10.0, "a1/frames/1.png", "description of frame-00001.png"),
    ]
    assert src.caption_contract == FRAME_CAPTION_CONTRACT
    assert sorted(p.name for p in (media_root / "a1" / "frames").iterdir()) == ["0.png", "1.png"]
    assert [p.name for p in visual.describe_calls] == ["frame-00000.png", "frame-00001.png"]
    assert (report.no_speech, report.visual_footage) == (1, 1)
    assert (report.visual_slides, report.visual_skipped, report.hollow) == (0, 0, 0)
    assert (
        "digest-video: silent non-slide video — describing 2 frame(s) as footage for item a1"
        in _digest_messages(caplog)
    )
    assert "digest-video: visual layer skipped (talking-head) for item a1" not in (
        _digest_messages(caplog)
    )


def test_footage_is_capped_by_the_footage_reducer_not_the_slide_reducer(tmp_path: Path):
    """Footage has its own, smaller budget: the slide reducer would keep 3 of 4
    frames, the footage reducer keeps 1 — exactly 1 frame is described + attached,
    and the slide reducer never runs."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual(classification="talking_head", n_frames=4, slides_keep=3, footage_keep=1)
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    assert [f.local_path for f in store["a1"].content.sources[0].frames] == ["a1/frames/0.png"]
    assert visual.reduce_calls == ["footage"]
    assert len(visual.describe_calls) == 1
    assert report.visual_footage == 1


@pytest.mark.parametrize(
    ("classification", "transcript"),
    [
        pytest.param("slides", _speech(), id="slides"),
        pytest.param("talking_head", _silence(), id="footage"),
    ],
)
def test_default_reducers_describe_every_extracted_frame(
    tmp_path: Path, classification: str, transcript: Transcript
):
    """A `VisualConfig` built without reducers keeps every frame on BOTH kept paths
    — the budgets are the CLI's to bind, never a hidden default cap."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    fake = _FakeVisual(classification=classification, n_frames=3)
    visual = VisualConfig(
        media_root=tmp_path / "media",
        extract_fn=fake.extract,
        describe_fn=fake.describe,
        classify_fn=fake.classify,
    )
    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: transcript,
        temp_root=tmp_path,
        visual=visual,
    )
    assert [f.local_path for f in store["a1"].content.sources[0].frames] == [
        "a1/frames/0.png",
        "a1/frames/1.png",
        "a1/frames/2.png",
    ]


def test_talking_head_with_speech_still_skips_the_visual_layer(tmp_path: Path):
    """With speech the transcript carries the content, so a `talking_head` video keeps
    today's skip: no reducer, no vision call, no frame files, counted as a skipped
    talking-head — and neither as footage nor as hollow."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    media_root = tmp_path / "media"
    visual = _FakeVisual(classification="talking_head", n_frames=3, footage_keep=2)
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech("an interview"),
        temp_root=tmp_path,
        visual=visual.config(media_root),
    )
    src = store["a1"].content.sources[0]
    assert (src.text, src.has_speech, src.frames, src.caption_contract) == (
        "an interview",
        True,
        [],
        "",
    )
    assert visual.describe_calls == []
    assert visual.reduce_calls == []
    assert not (media_root / "a1" / "frames").exists()
    assert (report.transcribed, report.visual_skipped) == (1, 1)
    assert (report.visual_footage, report.visual_slides, report.hollow) == (0, 0, 0)


def test_silent_footage_vision_failure_is_skipped_and_counted_hollow(tmp_path: Path, caplog):
    """A `VisionFailed` while describing footage drops the visual layer exactly like
    the slides path (`skipped`, warned) — and the item, now silent AND frameless, is
    counted as hollow rather than as described footage."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}

    def _boom(_path):
        raise VisionFailed("model crashed")

    visual = _FakeVisual(classification="talking_head", n_frames=2, describe=_boom)
    with caplog.at_level(logging.INFO, logger="xbrain.digest"):
        report = digest_videos(
            store,
            ["a1"],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _silence(),
            temp_root=tmp_path,
            visual=visual.config(tmp_path / "media"),
        )
    src = store["a1"].content.sources[0]
    assert (src.has_speech, src.frames, src.caption_contract) == (False, [], "")
    assert (report.no_speech, report.hollow) == (1, 1)
    assert (report.visual_footage, report.visual_slides, report.visual_skipped) == (0, 0, 0)
    assert "digest-video: visual layer failed for item a1: model crashed" in (
        _digest_messages(caplog)
    )


def _extraction_fails(_path: Path) -> list[KeyFrame]:
    raise FrameExtractionFailed("Invalid data")


@pytest.mark.parametrize(
    ("extract", "classification"),
    [
        pytest.param(_extraction_fails, "talking_head", id="extraction-failed"),
        pytest.param(lambda _p: [], "talking_head", id="zero-frames"),
        pytest.param(None, "unreadable", id="all-unreadable"),
    ],
)
def test_silent_video_non_content_drops_stay_skipped_and_hollow(
    tmp_path: Path, extract, classification
):
    """Extraction failure, zero frames and all-unreadable frames stay `skipped` even
    for a silent video — the footage path never describes what could not be read —
    and each leaves the item hollow."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual(classification=classification, n_frames=2)
    config = visual.config(tmp_path / "media")
    if extract is not None:
        config = replace(config, extract_fn=extract)
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=config,
    )
    assert store["a1"].content.sources[0].frames == []
    assert visual.describe_calls == []
    assert visual.reduce_calls == []
    assert (report.no_speech, report.hollow) == (1, 1)
    assert (report.visual_footage, report.visual_slides, report.visual_skipped) == (0, 0, 0)


def test_silent_video_on_a_non_frames_run_is_counted_hollow_per_item(tmp_path: Path):
    """Without `--frames` a silent video is attached with no text and no frames — a
    hollow entry, counted per ITEM (two bookmarks of one video = 2), so the run
    summary never hides it."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    report = digest_videos(
        store,
        ["a1", "a2"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=None,
    )
    assert (report.no_speech, report.videos_transcribed, report.hollow) == (2, 1, 2)
    assert (report.visual_footage, report.visual_slides, report.visual_skipped) == (0, 0, 0)


def test_speaking_video_on_a_non_frames_run_is_not_hollow(tmp_path: Path):
    """A frameless video WITH speech is not hollow — its transcript is the content."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=None,
    )
    assert (report.transcribed, report.hollow) == (1, 0)


# `transcribe._derive_has_speech` trusts an explicit `has_speech` and infers it from a
# non-empty segment list, so the ASR can hand back `has_speech=True` with no words:
# `{"text": "", "has_speech": true}`, or `{"segments": [{"text": " "}]}` (text
# backfilled to ""). Every consumer (generate, video_digest, worksheet, the api
# executor) reads such a source as having NO transcript, so the digest must too.
_BLANK_SPEECH = [
    pytest.param("", id="empty-text"),
    pytest.param("   ", id="whitespace-text"),
]


def _blank_speech(text: str) -> Transcript:
    return Transcript(text=text, segments=[Segment(0.0, 1.0, " ")], language="en", has_speech=True)


@pytest.mark.parametrize("text", _BLANK_SPEECH)
def test_blank_transcript_flagged_as_speech_describes_its_frames_as_footage(
    tmp_path: Path, text: str
):
    """`has_speech=True` with blank text carries no words, so a non-slide verdict is
    footage — frames described + attached — not a talking-head skip that would
    leave a frameless, wordless (hollow) source."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual(classification="talking_head", n_frames=2)
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _blank_speech(text),
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    src = store["a1"].content.sources[0]
    assert [(f.timestamp, f.local_path) for f in src.frames] == [
        (0.0, "a1/frames/0.png"),
        (10.0, "a1/frames/1.png"),
    ]
    assert visual.reduce_calls == ["footage"]
    assert (report.visual_footage, report.visual_skipped, report.hollow) == (1, 0, 0)
    # The transcript counters keep reading the raw flag — only the gate changed.
    assert (report.transcribed, report.no_speech) == (1, 0)


@pytest.mark.parametrize("text", _BLANK_SPEECH)
def test_blank_transcript_flagged_as_speech_is_hollow_on_a_non_frames_run(
    tmp_path: Path, text: str
):
    """Without `--frames` the same blank-but-flagged transcript attaches a source
    with no words and no frames — counted as hollow, never silent."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _blank_speech(text),
        temp_root=tmp_path,
        visual=None,
    )
    assert store["a1"].content.sources[0].frames == []
    assert report.hollow == 1
    assert (report.transcribed, report.no_speech) == (1, 0)


def _hollow_store(tmp_path: Path) -> dict[str, Item]:
    """One item already carrying a hollow source (silent, frameless), from a prior
    non-`--frames` run."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    first = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=None,
    )
    assert first.hollow == 1
    return store


def _transcription_fails(_path: Path) -> Transcript:
    raise TranscriberFailed("garbage output")


@pytest.mark.parametrize(
    ("fetch_fn", "transcribe_fn"),
    [
        pytest.param(_FakeFetch(fail_ids={"a1"}), lambda _p: _silence(), id="fetch-failed"),
        pytest.param(_FakeFetch(), _transcription_fails, id="transcription-failed"),
    ],
)
def test_failed_forced_redigest_keeps_the_old_hollow_source_as_failed(
    tmp_path: Path, fetch_fn, transcribe_fn
):
    """A `--force --frames` re-digest of a hollow item whose fetch or transcription
    fails attaches nothing: the OLD hollow source stays, and the item is counted
    under `fallidos` — not under `Huecos`, which counts only what this run
    attached (docs/digest-video.md, hollow items)."""
    store = _hollow_store(tmp_path)
    old = store["a1"].content.sources[0]
    visual = _FakeVisual(classification="talking_head", n_frames=2)
    report = digest_videos(
        store,
        ["a1"],
        force=True,
        fetch_fn=fetch_fn,
        transcribe_fn=transcribe_fn,
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    # Identity, not equality: a re-attached silent transcript compares EQUAL to the
    # old hollow source, so only `is` proves nothing was attached.
    assert len(store["a1"].content.sources) == 1
    assert store["a1"].content.sources[0] is old
    assert visual.describe_calls == []
    assert (report.failed, report.hollow, report.visual_footage) == (1, 0, 0)


def test_already_stored_hollow_item_is_not_counted_hollow_again(tmp_path: Path):
    """Without `--force` a stored hollow item is `already` and is not re-counted, so
    `Huecos` never reports the backlog — the documented jq recipe finds that."""
    store = _hollow_store(tmp_path)
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=None,
    )
    assert (report.already, report.hollow, report.videos_transcribed) == (1, 0, 0)


def test_silent_slide_deck_uses_the_slide_reducer_not_the_footage_one(tmp_path: Path):
    """A silent SLIDE deck keeps today's path: the slide reducer (keeps 2) decides the
    set, the footage reducer (keeps 1) never runs, and it counts as slides — not as
    footage, not as hollow."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    visual = _FakeVisual(classification="slides", n_frames=3, slides_keep=2, footage_keep=1)
    report = digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    assert [f.local_path for f in store["a1"].content.sources[0].frames] == [
        "a1/frames/0.png",
        "a1/frames/1.png",
    ]
    assert visual.reduce_calls == ["slides"]
    assert (report.no_speech, report.visual_slides) == (1, 1)
    assert (report.visual_footage, report.visual_skipped, report.hollow) == (0, 0, 0)


def test_redigest_with_fewer_slides_clears_stale_frame_files(tmp_path: Path):
    """A `--force` re-digest that yields FEWER slides must not leave stale
    higher-index PNGs orphaned on disk: `<id>/frames/` is cleared before the new
    (smaller) set is written, so the persisted files match the current result."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    media_root = tmp_path / "media"
    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=_FakeVisual(classification="slides", n_frames=3).config(media_root),
    )
    assert (media_root / "a1" / "frames" / "2.png").exists()  # 3 slides persisted

    digest_videos(
        store,
        ["a1"],
        force=True,
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=_FakeVisual(classification="slides", n_frames=1).config(media_root),
    )
    assert [f.local_path for f in store["a1"].content.sources[0].frames] == ["a1/frames/0.png"]
    assert (media_root / "a1" / "frames" / "0.png").exists()
    assert not (media_root / "a1" / "frames" / "1.png").exists()  # stale cleared
    assert not (media_root / "a1" / "frames" / "2.png").exists()  # stale cleared


def test_force_without_frames_logs_dropped_visual_layer(tmp_path: Path, caplog):
    """A `--force` re-digest WITHOUT `--frames` strips a prior kept visual layer —
    that is an operator-visible change, so it is LOGGED, never a silent drop."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _speech(),
        temp_root=tmp_path,
        visual=_FakeVisual(classification="slides", n_frames=2).config(tmp_path / "media"),
    )
    assert len(store["a1"].content.sources[0].frames) == 2  # precondition: framed

    with caplog.at_level(logging.INFO):
        digest_videos(
            store,
            ["a1"],
            force=True,
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p: _speech(),
            temp_root=tmp_path,
            visual=None,  # no --frames on the re-run
        )
    assert store["a1"].content.sources[0].frames == []  # visual layer stripped
    assert "dropped 2 prior slide(s)" in caplog.text


def test_format_digest_summary_renders_both_visual_segments():
    """The summary's Visual segment reports BOTH kept-slide and talking-head counts
    when `--frames` did something — so the operator sees the split at a glance."""
    report = DigestReport(
        transcribed=2, visual_slides=1, visual_skipped=1, groups={"amplify_video/1": ["a", "b"]}
    )
    summary = format_digest_summary(report)
    assert "1 con slides" in summary
    assert "1 talking-head (saltados)" in summary


def test_format_digest_summary_omits_visual_on_non_frames_run():
    """A run where the visual layer did nothing (a non-`--frames` run) appends no
    Visual segment — the summary is byte-unchanged from the PR2/PR3 shape."""
    report = DigestReport(transcribed=1, groups={"amplify_video/1": ["a"]})
    assert "Visual:" not in format_digest_summary(report)


# Today's summary line for the reports below, written out literally — never
# rebuilt with `format_digest_summary`, which is the code under test.
_BASE_SUMMARY = (
    "Vídeos: transcritos 1, sin voz 2, ya digeridos 0, fallidos 0, "
    "sin vídeo 0, desconocidos 0. "
    "Dedup: 3 items ← 3 vídeos (3 transcritos este run)."
)
_THREE_GROUPS = {"amplify_video/1": ["a"], "amplify_video/2": ["b"], "amplify_video/3": ["c"]}


def test_format_digest_summary_renders_the_full_visual_segment():
    """Slides, silent footage and talking-head skips each get their own count, in
    that order, as one exact segment right after the Dedup sentence."""
    report = DigestReport(
        transcribed=1,
        no_speech=2,
        videos_transcribed=3,
        visual_slides=1,
        visual_footage=1,
        visual_skipped=1,
        groups=_THREE_GROUPS,
    )
    assert format_digest_summary(report) == (
        _BASE_SUMMARY + " Visual: 1 con slides, 1 metraje mudo descrito, 1 talking-head (saltados)."
    )


def test_format_digest_summary_shows_a_footage_only_visual_segment():
    """A run whose only visual work was silent footage still prints the segment."""
    report = DigestReport(
        transcribed=1, no_speech=2, videos_transcribed=3, visual_footage=2, groups=_THREE_GROUPS
    )
    assert format_digest_summary(report) == (
        _BASE_SUMMARY + " Visual: 0 con slides, 2 metraje mudo descrito, 0 talking-head (saltados)."
    )


def test_format_digest_summary_appends_hollow_to_todays_line():
    """With no visual work, hollow items add exactly one sentence to today's line."""
    report = DigestReport(
        transcribed=1, no_speech=2, videos_transcribed=3, hollow=2, groups=_THREE_GROUPS
    )
    assert format_digest_summary(report) == (_BASE_SUMMARY + " Huecos (sin voz ni frames): 2.")


def test_format_digest_summary_puts_hollow_after_the_visual_segment():
    report = DigestReport(
        transcribed=1,
        no_speech=2,
        videos_transcribed=3,
        visual_slides=1,
        visual_skipped=1,
        hollow=1,
        groups=_THREE_GROUPS,
    )
    assert format_digest_summary(report) == (
        _BASE_SUMMARY
        + " Visual: 1 con slides, 0 metraje mudo descrito, 1 talking-head (saltados)."
        + " Huecos (sin voz ni frames): 1."
    )


def test_format_digest_summary_is_todays_line_without_hollow_or_visual_work():
    report = DigestReport(transcribed=1, no_speech=2, videos_transcribed=3, groups=_THREE_GROUPS)
    assert format_digest_summary(report) == _BASE_SUMMARY


def test_digest_stamps_the_caption_contract_when_frames_were_described():
    """A `--frames` run produces captions under the CURRENT contract, so it must
    say so — otherwise `redescribe-frames` re-captions frames that are already
    fresh, at full cost."""
    from xbrain.models import FRAME_CAPTION_CONTRACT, VideoFrame

    store = {"1": _item("1", _VIDEO_A_URL_1)}
    frames = {"1": [VideoFrame(timestamp=0.0, local_path="1/frames/0.png", description="d")]}
    attach_transcript(store, ["1"], _speech(), frames_by_item=frames)
    source = store["1"].content.sources[-1]
    assert source.caption_contract == FRAME_CAPTION_CONTRACT


def test_digest_leaves_the_contract_empty_when_no_frames_were_described():
    """An audio-only digest describes nothing, so it must claim no contract."""
    store = {"1": _item("1", _VIDEO_A_URL_1)}
    attach_transcript(store, ["1"], _speech())
    assert store["1"].content.sources[-1].caption_contract == ""


# ------------------------------------------------------------ keep_transcript (frames-only re-digest)


class _TranscriberMustNotRun:
    """A transcriber that fails the test the moment it runs.

    `keep_transcript` must reuse the stored transcript and never re-run the ASR.
    It raises `AssertionError`, which `_analyze_media` does not catch (it only
    catches `TranscriberFailed`), so a call aborts the run instead of being
    quietly counted as `fallidos`; `calls` records it as well."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, path: Path) -> Transcript:
        self.calls.append(Path(path).name)
        raise AssertionError(f"the transcriber ran on {Path(path).name} under keep_transcript")


def _stored_speech_store(tmp_path: Path, transcript: Transcript) -> dict[str, Item]:
    """One item already carrying a with-speech `x_video` source, from a prior
    non-`--frames` run."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1)}
    digest_videos(
        store,
        ["a1"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: transcript,
        temp_root=tmp_path,
        visual=None,
    )
    source = store["a1"].content.sources[0]
    assert (source.text, source.has_speech) == (transcript.text, True)
    return store


def test_keep_transcript_redigests_a_stored_hollow_item_without_the_asr(tmp_path: Path):
    """The hollow-recovery run: the stored silent transcript is reused (the ASR
    never runs), the mp4 is still fetched because the frames need it, and the
    silent non-slide video comes back with its frames as footage — no longer
    hollow."""
    store = _hollow_store(tmp_path)
    fetch = _FakeFetch()
    transcriber = _TranscriberMustNotRun()
    visual = _FakeVisual(classification="talking_head", n_frames=2)

    report = digest_videos(
        store,
        ["a1"],
        force=True,
        keep_transcript=True,
        fetch_fn=fetch,
        transcribe_fn=transcriber,
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    assert transcriber.calls == []
    assert fetch.fetched_ids == ["a1"]
    assert len(store["a1"].content.sources) == 1
    source = store["a1"].content.sources[0]
    assert source.kind == "x_video"
    assert [(f.local_path, f.description) for f in source.frames] == [
        ("a1/frames/0.png", "description of frame-00000.png"),
        ("a1/frames/1.png", "description of frame-00001.png"),
    ]
    assert source.has_speech is False
    assert source.text == ""
    assert (report.no_speech, report.visual_footage, report.hollow) == (1, 1, 0)
    assert report.videos_transcribed == 0
    assert report.videos_reused == 1


def test_keep_transcript_keeps_stored_speech_and_skips_a_talking_head(tmp_path: Path):
    """A stored transcript WITH speech is kept exactly (text, flag, language,
    title) and still decides the visual layer: a talking head is skipped, so no
    frames and no vision call."""
    stored = replace(_speech("hello world"), title="The talk")
    store = _stored_speech_store(tmp_path, stored)
    transcriber = _TranscriberMustNotRun()
    visual = _FakeVisual(classification="talking_head", n_frames=2)

    report = digest_videos(
        store,
        ["a1"],
        force=True,
        keep_transcript=True,
        fetch_fn=_FakeFetch(),
        transcribe_fn=transcriber,
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    assert transcriber.calls == []
    source = store["a1"].content.sources[0]
    assert source.frames == []
    assert (source.text, source.has_speech, source.language, source.title) == (
        "hello world",
        True,
        "en",
        "The talk",
    )
    assert visual.describe_calls == []
    assert (report.transcribed, report.visual_skipped, report.visual_footage, report.hollow) == (
        1,
        1,
        0,
        0,
    )


@pytest.mark.parametrize(
    "stored_sources",
    [
        pytest.param(None, id="no-content"),
        pytest.param(
            [ContentSourceSuccess(kind="external_article", url="https://ex.com/a", text="body")],
            id="article-only",
        ),
        pytest.param(
            [
                ContentSourceFailure(
                    kind="x_video", url=_VIDEO_A_URL_1, failure_reason="unknown_error"
                )
            ],
            id="failed-x-video",
        ),
    ],
)
def test_keep_transcript_transcribes_when_no_transcript_is_stored(tmp_path: Path, stored_sources):
    """With no stored `x_video` success source there is nothing to keep, so the
    ASR runs once, as on a normal run."""
    item = _item("a1", _VIDEO_A_URL_1)
    if stored_sources is not None:
        item.content = Content(
            fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc), sources=stored_sources
        )
    store = {"a1": item}
    calls: list[str] = []

    def _transcribe(path: Path) -> Transcript:
        calls.append(Path(path).name)
        return _speech("fresh words")

    report = digest_videos(
        store,
        ["a1"],
        force=True,
        keep_transcript=True,
        fetch_fn=_FakeFetch(),
        transcribe_fn=_transcribe,
        temp_root=tmp_path,
        visual=_FakeVisual(classification="talking_head").config(tmp_path / "media"),
    )
    assert calls == ["a1.mp4"]
    video_sources = [s for s in store["a1"].content.sources if s.kind == "x_video"]
    assert [(s.outcome, s.text) for s in video_sources] == [("success", "fresh words")]
    assert (report.transcribed, report.visual_skipped) == (1, 1)


# A whisper hallucination stored in the real corpus (item 2095233745167282602).
_STORED_HALLUCINATION = Transcript(text="you you", has_speech=True)


@pytest.mark.parametrize(
    ("keep_transcript", "text", "has_speech", "frame_paths", "counts"),
    [
        # Re-transcribing invents a quote: the words skip the frames as a talking
        # head, and the item is no longer counted hollow — nothing flags it again.
        pytest.param(False, "you you", True, [], (1, 0, 1, 0), id="re-transcribed"),
        pytest.param(
            True,
            "",
            False,
            ["a1/frames/0.png", "a1/frames/1.png"],
            (0, 1, 0, 0),
            id="kept-transcript",
        ),
    ],
)
def test_keep_transcript_blocks_an_asr_hallucination_on_a_hollow_item(
    tmp_path: Path, keep_transcript, text, has_speech, frame_paths, counts
):
    """Regression pin for the whisper-on-music hazard on a stored hollow item.
    `counts` is (transcribed, visual_footage, visual_skipped, hollow)."""
    store = _hollow_store(tmp_path)
    visual = _FakeVisual(classification="talking_head", n_frames=2)

    report = digest_videos(
        store,
        ["a1"],
        force=True,
        keep_transcript=keep_transcript,
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _STORED_HALLUCINATION,
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    source = store["a1"].content.sources[0]
    assert (source.text, source.has_speech) == (text, has_speech)
    assert [f.local_path for f in source.frames] == frame_paths
    assert (
        report.transcribed,
        report.visual_footage,
        report.visual_skipped,
        report.hollow,
    ) == counts


def _hollow_pair_store(tmp_path: Path) -> dict[str, Item]:
    """`a1` already carries a hollow source; `p1` bookmarks the SAME video
    (another signed URL for `amplify_video/1500`) and was never digested."""
    store = _hollow_store(tmp_path)
    store["p1"] = _item("p1", _VIDEO_A_URL_2)
    return store


@pytest.mark.parametrize(
    "order", [["a1", "p1"], ["p1", "a1"]], ids=["hollow-first", "pending-first"]
)
def test_keep_transcript_never_moves_a_transcript_between_items_of_one_video(
    tmp_path: Path, order: list[str]
):
    """One video, two items: `a1` has a stored hollow transcript, `p1` has none.
    Whatever the id order, the ASR runs once, for `p1` only, and its invented
    words land on `p1` alone; `a1` keeps its empty transcript and gets frames."""
    store = _hollow_pair_store(tmp_path)
    fetch = _FakeFetch()
    calls: list[str] = []

    def _transcribe(path: Path) -> Transcript:
        calls.append(Path(path).name)
        return _STORED_HALLUCINATION

    report = digest_videos(
        store,
        order,
        force=True,
        keep_transcript=True,
        fetch_fn=fetch,
        transcribe_fn=_transcribe,
        temp_root=tmp_path,
        visual=_FakeVisual(classification="talking_head", n_frames=2).config(tmp_path / "media"),
    )
    hollow = store["a1"].content.sources[0]
    pending = store["p1"].content.sources[0]
    assert calls == ["p1.mp4"]
    assert sorted(fetch.fetched_ids) == ["a1", "p1"]
    assert (hollow.text, hollow.has_speech) == ("", False)
    assert [f.local_path for f in hollow.frames] == ["a1/frames/0.png", "a1/frames/1.png"]
    assert (pending.text, pending.has_speech, pending.frames) == ("you you", True, [])
    assert (report.videos_transcribed, report.videos_reused) == (1, 1)
    assert (
        report.transcribed,
        report.no_speech,
        report.visual_footage,
        report.visual_skipped,
        report.hollow,
    ) == (1, 1, 1, 1, 0)


@pytest.mark.parametrize(
    "order", [["a1", "a2"], ["a2", "a1"]], ids=["speech-first", "silent-first"]
)
def test_keep_transcript_keeps_each_items_own_stored_transcript(tmp_path: Path, order: list[str]):
    """Two items of one video stored DIFFERENT transcripts. Each keeps its own,
    byte for byte, and its own visual verdict: the speaking one is a skipped
    talking head, the silent one gets its frames as footage."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    for item_id, transcript in (("a1", _speech("hello world")), ("a2", _silence())):
        digest_videos(
            store,
            [item_id],
            fetch_fn=_FakeFetch(),
            transcribe_fn=lambda _p, t=transcript: t,
            temp_root=tmp_path,
            visual=None,
        )
    fetch = _FakeFetch()
    transcriber = _TranscriberMustNotRun()

    report = digest_videos(
        store,
        order,
        force=True,
        keep_transcript=True,
        fetch_fn=fetch,
        transcribe_fn=transcriber,
        temp_root=tmp_path,
        visual=_FakeVisual(classification="talking_head", n_frames=2).config(tmp_path / "media"),
    )
    speaking = store["a1"].content.sources[0]
    silent = store["a2"].content.sources[0]
    assert transcriber.calls == []
    assert sorted(fetch.fetched_ids) == ["a1", "a2"]
    assert (speaking.text, speaking.has_speech, speaking.language, speaking.frames) == (
        "hello world",
        True,
        "en",
        [],
    )
    assert (silent.text, silent.has_speech, silent.language) == ("", False, None)
    assert [f.local_path for f in silent.frames] == ["a2/frames/0.png", "a2/frames/1.png"]
    assert (report.videos_transcribed, report.videos_reused) == (0, 1)
    assert (
        report.transcribed,
        report.no_speech,
        report.visual_footage,
        report.visual_skipped,
        report.hollow,
    ) == (1, 1, 1, 1, 0)


def test_keep_transcript_shares_one_fetch_between_identical_stored_transcripts(tmp_path: Path):
    """Two hollow items of one video store the same empty transcript: the video is
    fetched and described once, and each item gets its own copy of the frames."""
    store = {"a1": _item("a1", _VIDEO_A_URL_1), "a2": _item("a2", _VIDEO_A_URL_2)}
    digest_videos(
        store,
        ["a1", "a2"],
        fetch_fn=_FakeFetch(),
        transcribe_fn=lambda _p: _silence(),
        temp_root=tmp_path,
        visual=None,
    )
    fetch = _FakeFetch()
    transcriber = _TranscriberMustNotRun()
    visual = _FakeVisual(classification="talking_head", n_frames=2)

    report = digest_videos(
        store,
        ["a1", "a2"],
        force=True,
        keep_transcript=True,
        fetch_fn=fetch,
        transcribe_fn=transcriber,
        temp_root=tmp_path,
        visual=visual.config(tmp_path / "media"),
    )
    assert transcriber.calls == []
    assert fetch.fetched_ids == ["a1"]
    assert len(visual.describe_calls) == 2
    for item_id in ("a1", "a2"):
        source = store[item_id].content.sources[0]
        assert (source.text, source.has_speech) == ("", False)
        assert [f.local_path for f in source.frames] == [
            f"{item_id}/frames/0.png",
            f"{item_id}/frames/1.png",
        ]
    assert (report.no_speech, report.visual_footage, report.hollow) == (2, 1, 0)
    assert (report.videos_transcribed, report.videos_reused) == (0, 1)


@pytest.mark.parametrize(
    ("force", "with_visual", "message"),
    [
        pytest.param(
            True,
            False,
            "keep_transcript requires visual (it re-runs the visual layer)",
            id="without-visual",
        ),
        pytest.param(
            False,
            True,
            "keep_transcript requires force (it re-digests already-digested videos)",
            id="without-force",
        ),
    ],
)
def test_keep_transcript_rejects_a_run_without_visual_or_force(
    tmp_path: Path, force: bool, with_visual: bool, message: str
):
    """The library guards the flag like the CLI does: a keep run without the visual
    layer would strip stored frames and digests, and without `force` it does
    nothing. Both raise before anything is fetched or written."""
    store = _hollow_store(tmp_path)
    before = store["a1"].content.sources[0]
    fetch = _FakeFetch()
    visual = _FakeVisual().config(tmp_path / "media") if with_visual else None

    with pytest.raises(ValueError) as excinfo:
        digest_videos(
            store,
            ["a1"],
            force=force,
            keep_transcript=True,
            fetch_fn=fetch,
            transcribe_fn=_TranscriberMustNotRun(),
            temp_root=tmp_path,
            visual=visual,
        )
    assert str(excinfo.value) == message
    assert fetch.fetched_ids == []
    assert store["a1"].content.sources == [before]
    assert store["a1"].content.sources[0] is before


def _item_with_sources(sources: list) -> dict[str, Item]:
    item = _item("a1", _VIDEO_A_URL_1)
    item.content = Content(fetched_at=datetime(2026, 5, 17, tzinfo=timezone.utc), sources=sources)
    return {"a1": item}


def test_keep_transcript_reads_the_first_x_video_success_source(tmp_path: Path):
    """A failed `x_video` source listed before the stored hollow transcript does not
    hide it: the transcript is still reused and the ASR does not run."""
    store = _item_with_sources(
        [
            ContentSourceFailure(kind="x_video", url=_VIDEO_A_URL_1, failure_reason="timeout"),
            ContentSourceSuccess(kind="x_video", url=_VIDEO_A_URL_1, text="", has_speech=False),
        ]
    )
    transcriber = _TranscriberMustNotRun()

    report = digest_videos(
        store,
        ["a1"],
        force=True,
        keep_transcript=True,
        fetch_fn=_FakeFetch(),
        transcribe_fn=transcriber,
        temp_root=tmp_path,
        visual=_FakeVisual(classification="talking_head", n_frames=2).config(tmp_path / "media"),
    )
    sources = store["a1"].content.sources
    assert transcriber.calls == []
    assert [(s.outcome, s.text, s.has_speech) for s in sources] == [("success", "", False)]
    assert [f.local_path for f in sources[0].frames] == ["a1/frames/0.png", "a1/frames/1.png"]
    assert (report.videos_transcribed, report.videos_reused) == (0, 1)


@pytest.mark.parametrize(
    ("text", "has_speech", "frame_paths", "visual_skipped"),
    [
        pytest.param("real words", True, [], 1, id="legacy-speech"),
        pytest.param("", False, ["a1/frames/0.png", "a1/frames/1.png"], 0, id="legacy-empty"),
        pytest.param("  ", False, ["a1/frames/0.png", "a1/frames/1.png"], 0, id="legacy-blank"),
    ],
)
def test_keep_transcript_reads_a_legacy_speech_flag_from_the_text(
    tmp_path: Path, text: str, has_speech: bool, frame_paths: list[str], visual_skipped: int
):
    """A legacy source with `has_speech=None` counts as speech when its text has
    words (as `generate` reads it) and as silent otherwise; the text is kept."""
    store = _item_with_sources(
        [ContentSourceSuccess(kind="x_video", url=_VIDEO_A_URL_1, text=text, has_speech=None)]
    )

    report = digest_videos(
        store,
        ["a1"],
        force=True,
        keep_transcript=True,
        fetch_fn=_FakeFetch(),
        transcribe_fn=_TranscriberMustNotRun(),
        temp_root=tmp_path,
        visual=_FakeVisual(classification="talking_head", n_frames=2).config(tmp_path / "media"),
    )
    source = store["a1"].content.sources[0]
    assert (source.text, source.has_speech) == (text, has_speech)
    assert [f.local_path for f in source.frames] == frame_paths
    assert report.visual_skipped == visual_skipped


def test_format_digest_summary_names_reused_transcripts():
    """A run that reused stored transcripts says how many, inside the Dedup
    parenthesis; without reuse the line is today's (tested above)."""
    report = DigestReport(no_speech=3, videos_transcribed=1, videos_reused=2, groups=_THREE_GROUPS)
    assert format_digest_summary(report) == (
        "Vídeos: transcritos 0, sin voz 3, ya digeridos 0, fallidos 0, "
        "sin vídeo 0, desconocidos 0. "
        "Dedup: 3 items ← 3 vídeos (1 transcritos este run, 2 con transcripción guardada)."
    )
