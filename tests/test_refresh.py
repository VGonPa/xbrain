# tests/test_refresh.py
"""Unit tests for the pure video-refresh backfill (`xbrain.refresh`).

`refresh_video_media` rewrites the VIDEO media on items already in the store —
swapping each poster-era `MediaVideoPending` for the freshly-parsed one that
carries the playable stream URL + bitrate + duration — WITHOUT touching photos
or any enrichment/description state. `estimate_download_size` is a pure,
network-free size pre-flight over the stored videos. Both are exercised here
in isolation; the CLI wiring is covered in `tests/test_cli.py`.
"""

from __future__ import annotations

from datetime import datetime, timezone

from xbrain.models import (
    Author,
    Item,
    MediaEntry,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoPending,
)
from xbrain.refresh import RefreshReport, estimate_download_size, refresh_video_media


def _item(item_id: str, media: list[MediaEntry]) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        media=media,
    )


def _poster_video() -> MediaVideoPending:
    """A poster-era video record: the poster image as the URL, no metadata."""
    return MediaVideoPending(url="https://pbs.twimg.com/poster.jpg")


def _playable_video(url: str = "https://v/high.mp4") -> MediaVideoPending:
    """A freshly-parsed video record: the playable stream + bitrate + duration."""
    return MediaVideoPending(
        url=url,
        thumbnail_url="https://pbs.twimg.com/poster.jpg",
        bitrate=2_176_000,
        duration_millis=30_000,
    )


def _poster_fallback_video() -> MediaVideoPending:
    """A fresh capture where X served NO usable variant: `build_video_media`
    falls back to the poster, so url == thumbnail_url and there is no metadata.
    Swapping this onto a stored record would DEGRADE it — it must be rejected.
    """
    poster = "https://pbs.twimg.com/poster.jpg"
    return MediaVideoPending(url=poster, thumbnail_url=poster)


def _downloaded_photo() -> MediaPhotoDownloaded:
    return MediaPhotoDownloaded(
        url="https://pbs.twimg.com/media/dl.jpg",
        local_path="1/0.jpg",
        width=8,
        height=6,
        bytes_size=200,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )


def _described_photo() -> MediaPhotoDescribed:
    return MediaPhotoDescribed(
        url="https://pbs.twimg.com/media/desc.jpg",
        local_path="1/1.jpg",
        width=8,
        height=6,
        bytes_size=200,
        downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
        is_decorative=False,
        description="A bar chart.",
        description_lang="English",
        description_version="v1",
        described_at=datetime(2026, 5, 25, tzinfo=timezone.utc),
    )


def _failed_photo() -> MediaPhotoFailed:
    return MediaPhotoFailed(
        url="https://pbs.twimg.com/media/fail.jpg",
        failure_reason="http_4xx",
        attempts=1,
        last_attempt_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
    )


# --------------------------------------------------------------------- merge


def test_refresh_swaps_video_url_and_metadata():
    """A poster-era video gains the playable URL + bitrate + duration."""
    store = {"1": _item("1", [_poster_video()])}
    fresh = [_item("1", [_playable_video()])]

    report = refresh_video_media(store, fresh)

    video = store["1"].media[0]
    assert isinstance(video, MediaVideoPending)
    assert video.url == "https://v/high.mp4"
    assert video.bitrate == 2_176_000
    assert video.duration_millis == 30_000
    assert report.items_seen == 1
    assert report.items_refreshed == 1
    assert report.videos_updated == 1


def test_refresh_preserves_downloaded_and_described_photos_untouched():
    """A downloaded AND a described photo survive a refresh byte-for-byte."""
    downloaded = _downloaded_photo()
    described = _described_photo()
    store = {"1": _item("1", [downloaded, _poster_video(), described])}
    fresh = [_item("1", [_playable_video()])]

    refresh_video_media(store, fresh)

    media = store["1"].media
    # The photo entries are the very same objects, unmodified.
    assert media[0] is downloaded
    assert media[2] is described
    assert isinstance(media[0], MediaPhotoDownloaded)
    assert isinstance(media[2], MediaPhotoDescribed)
    assert media[2].description == "A bar chart."
    # Only the middle video entry changed.
    assert isinstance(media[1], MediaVideoPending)
    assert media[1].url == "https://v/high.mp4"


def test_refresh_preserves_every_photo_variant_and_order():
    """Pending / Failed photo variants are also left exactly as-is, in order."""
    pending = MediaPhotoPending(url="https://pbs.twimg.com/media/p.jpg")
    failed = _failed_photo()
    store = {"1": _item("1", [pending, _poster_video(), failed])}
    fresh = [_item("1", [_playable_video()])]

    refresh_video_media(store, fresh)

    media = store["1"].media
    assert media[0] is pending
    assert media[2] is failed
    assert isinstance(media[1], MediaVideoPending)
    assert media[1].url == "https://v/high.mp4"


def test_refresh_replaces_multiple_videos_positionally():
    """Two videos in one item map positionally to the two fresh videos."""
    store = {"1": _item("1", [_poster_video(), _poster_video()])}
    fresh = [
        _item(
            "1",
            [_playable_video("https://v/a.mp4"), _playable_video("https://v/b.mp4")],
        )
    ]

    report = refresh_video_media(store, fresh)

    media = store["1"].media
    assert [m.url for m in media] == ["https://v/a.mp4", "https://v/b.mp4"]
    assert report.videos_updated == 2
    assert report.items_refreshed == 1


def test_refresh_keeps_extra_store_video_when_fresh_has_fewer():
    """A store video with no fresh counterpart is left as-is (no crash, no drop)."""
    keep = _poster_video()
    store = {"1": _item("1", [_poster_video(), keep])}
    fresh = [_item("1", [_playable_video("https://v/only.mp4")])]

    report = refresh_video_media(store, fresh)

    media = store["1"].media
    assert media[0].url == "https://v/only.mp4"
    assert media[1] is keep  # untouched — no fresh video for this slot
    assert report.videos_updated == 1


def test_refresh_skips_fresh_item_not_in_store():
    """Backfill only touches known ids — an unknown fresh item is ignored."""
    store = {"1": _item("1", [_poster_video()])}
    fresh = [_item("999", [_playable_video()])]

    report = refresh_video_media(store, fresh)

    assert store["1"].media[0].url == "https://pbs.twimg.com/poster.jpg"
    assert "999" not in store
    assert report.items_seen == 0
    assert report.items_refreshed == 0
    assert report.videos_updated == 0


def test_refresh_leaves_store_untouched_when_fresh_item_has_no_video():
    """A re-seen item whose fresh capture has no video is counted but not changed."""
    poster = _poster_video()
    store = {"1": _item("1", [poster])}
    fresh = [_item("1", [MediaPhotoPending(url="https://pbs.twimg.com/media/x.jpg")])]

    report = refresh_video_media(store, fresh)

    assert store["1"].media[0] is poster  # unchanged
    assert report.items_seen == 1
    assert report.items_refreshed == 0
    assert report.videos_updated == 0


def test_refresh_counts_video_items_not_re_seen():
    """A video item absent from the fresh capture is reported as still poster-era."""
    store = {
        "seen": _item("seen", [_poster_video()]),
        "missed": _item("missed", [_poster_video()]),
        "photo-only": _item(
            "photo-only", [MediaPhotoPending(url="https://pbs.twimg.com/media/y.jpg")]
        ),
    }
    fresh = [_item("seen", [_playable_video()])]

    report = refresh_video_media(store, fresh)

    # `missed` has a video and was not re-seen → still poster-era.
    # `photo-only` has no video → not counted. `seen` was re-seen → not counted.
    assert report.items_with_video_not_seen == 1
    assert report.items_seen == 1
    assert report.items_refreshed == 1


def test_refresh_first_fresh_entry_wins_on_duplicate_id():
    """Duplicate fresh ids (e.g. an item captured from two sources) dedupe."""
    store = {"1": _item("1", [_poster_video()])}
    fresh = [
        _item("1", [_playable_video("https://v/first.mp4")]),
        _item("1", [_playable_video("https://v/second.mp4")]),
    ]

    report = refresh_video_media(store, fresh)

    assert store["1"].media[0].url == "https://v/first.mp4"
    assert report.items_seen == 1
    assert report.videos_updated == 1


def test_refresh_rejects_poster_fallback_keeping_existing_poster_era():
    """A fresh poster-only capture (drift) does NOT overwrite a poster-era record."""
    existing = _poster_video()
    store = {"1": _item("1", [existing])}
    fresh = [_item("1", [_poster_fallback_video()])]

    report = refresh_video_media(store, fresh)

    # The store entry is left as-is — no replacement with the poster fallback.
    assert store["1"].media[0] is existing
    assert report.items_seen == 1
    assert report.items_refreshed == 0
    assert report.videos_updated == 0


def test_refresh_never_degrades_a_good_video_to_a_poster_fallback():
    """Second-run regression guard: a good mp4 survives a poster-only re-capture.

    `refresh_video_media` is the first overwriting path in the repo. If X drifts
    and serves no usable variant on a later run, `build_video_media` yields a
    poster fallback (url == thumbnail_url). That must NOT replace an already-good
    playable record — otherwise a re-run would silently undo a prior refresh.
    """
    good = _playable_video("https://v/good.mp4")
    store = {"1": _item("1", [good])}
    fresh = [_item("1", [_poster_fallback_video()])]

    report = refresh_video_media(store, fresh)

    survivor = store["1"].media[0]
    assert survivor is good
    assert survivor.url == "https://v/good.mp4"
    assert survivor.bitrate == 2_176_000
    assert report.items_refreshed == 0
    assert report.videos_updated == 0


def test_refresh_report_defaults_to_zero():
    """An empty refresh yields an all-zero report (no false positives)."""
    report = refresh_video_media({}, [])
    assert report == RefreshReport(
        items_seen=0,
        items_refreshed=0,
        videos_updated=0,
        items_with_video_not_seen=0,
    )


# ----------------------------------------------------------------- estimate


def test_estimate_sums_a_known_mp4():
    """bytes = bitrate * duration_millis / 1000 / 8 for a fully-specified mp4."""
    store = {"1": _item("1", [_playable_video()])}

    estimated, n_estimable, n_unknown = estimate_download_size(store)

    # 2_176_000 b/s * 30 s / 8 = 8_160_000 bytes.
    assert estimated == 8_160_000
    assert n_estimable == 1
    assert n_unknown == 0


def test_estimate_counts_bitrate_zero_gif_as_unknown():
    """An animated GIF reports bitrate 0 — unknown size, never 0 bytes."""
    gif = MediaVideoPending(url="https://v/gif.mp4", bitrate=0, duration_millis=5_000)
    store = {"1": _item("1", [gif])}

    estimated, n_estimable, n_unknown = estimate_download_size(store)

    assert estimated == 0
    assert n_estimable == 0
    assert n_unknown == 1


def test_estimate_counts_none_bitrate_as_unknown():
    """An HLS-only variant has no bitrate — unknown."""
    hls = MediaVideoPending(url="https://v/play.m3u8", bitrate=None, duration_millis=5_000)
    store = {"1": _item("1", [hls])}

    estimated, n_estimable, n_unknown = estimate_download_size(store)

    assert estimated == 0
    assert n_unknown == 1


def test_estimate_counts_missing_duration_as_unknown():
    """A bitrate with no duration cannot be estimated — unknown."""
    no_dur = MediaVideoPending(url="https://v/x.mp4", bitrate=1_000_000, duration_millis=None)
    store = {"1": _item("1", [no_dur])}

    estimated, n_estimable, n_unknown = estimate_download_size(store)

    assert estimated == 0
    assert n_unknown == 1


def test_estimate_ignores_photo_entries():
    """Photos never contribute to the video download estimate."""
    store = {"1": _item("1", [_downloaded_photo(), _described_photo()])}

    estimated, n_estimable, n_unknown = estimate_download_size(store)

    assert estimated == 0
    assert n_estimable == 0
    assert n_unknown == 0


def test_estimate_mixes_estimable_and_unknown_across_items():
    """The estimate sums estimable videos and counts the unknown separately."""
    store = {
        "1": _item("1", [_playable_video()]),  # estimable: 8_160_000
        "2": _item("2", [MediaVideoPending(url="https://v/gif.mp4", bitrate=0)]),  # unknown
        "3": _item("3", [_downloaded_photo()]),  # ignored
    }

    estimated, n_estimable, n_unknown = estimate_download_size(store)

    assert estimated == 8_160_000
    assert n_estimable == 1
    assert n_unknown == 1


# --------------------------------------------------------- quoted-post backfill

from xbrain.models import Content, ContentSourceFailure, ContentSourceSuccess  # noqa: E402
from xbrain.refresh import backfill_quoted_sources  # noqa: E402

_T0 = datetime(2026, 5, 1, tzinfo=timezone.utc)


def _quoted_item(item_id: str, *, sources=None, quoted_id: str | None = "999") -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="Read this and you'll understand better this career move",
        created_at=_T0,
        captured_at=_T0,
        quoted_id=quoted_id,
        content=Content(fetched_at=_T0, sources=list(sources)) if sources is not None else None,
    )


def _quoted_success() -> ContentSourceSuccess:
    return ContentSourceSuccess(
        kind="quoted_tweet",
        url="https://x.com/karpathy/status/999",
        text="I am leaving OpenAI.",
        author=Author(handle="karpathy", name="Andrej Karpathy"),
    )


def test_backfill_attaches_the_quoted_source_to_a_stored_item():
    """The 762 stored quote-tweets hold only a `quoted_id`. A re-capture carries the
    quoted post in the SAME payload, so the backfill is a re-parse, not a fetch."""
    store = {"1": _quoted_item("1")}
    fresh = [_quoted_item("1", sources=[_quoted_success()])]

    report = backfill_quoted_sources(store, fresh)

    assert report.items_seen == 1
    assert report.sources_attached == 1
    assert report.readable == 1
    assert store["1"].content is not None
    assert store["1"].content.sources == [_quoted_success()]


def test_backfill_preserves_every_other_source_and_all_enrichment():
    """It appends the quoted post. An article body, a transcript or a thread already
    on the item must survive untouched — this is a backfill, not a rebuild."""
    article = ContentSourceSuccess(
        kind="external_article", url="https://example.com/p", title="T", text="the article body"
    )
    store = {"1": _quoted_item("1", sources=[article])}
    fresh = [_quoted_item("1", sources=[_quoted_success()])]

    backfill_quoted_sources(store, fresh)

    assert article in store["1"].content.sources
    assert _quoted_success() in store["1"].content.sources


def test_backfill_advances_fetched_at_so_the_item_re_enriches():
    """The item just gained the evidence its summary was missing — it must flow back
    through `enrich` (`content.fetched_at > enriched_at`), or the defective summary
    stands and the whole fix ships without repairing anything."""
    store = {"1": _quoted_item("1", sources=[])}
    fresh = [_quoted_item("1", sources=[_quoted_success()])]

    backfill_quoted_sources(store, fresh)

    assert store["1"].content.fetched_at > _T0


def test_backfill_upgrades_an_unreadable_quote_to_a_readable_one():
    """A post that was deleted-at-capture but is readable now must be upgradeable —
    and the stale failure must not linger alongside the body."""
    failure = ContentSourceFailure(
        kind="quoted_tweet", url="https://x.com/i/status/999", failure_reason="not_found"
    )
    store = {"1": _quoted_item("1", sources=[failure])}
    fresh = [_quoted_item("1", sources=[_quoted_success()])]

    report = backfill_quoted_sources(store, fresh)

    assert store["1"].content.sources == [_quoted_success()]
    assert report.sources_attached == 1


def test_backfill_is_idempotent_on_an_already_readable_quote():
    store = {"1": _quoted_item("1", sources=[_quoted_success()])}
    fresh = [_quoted_item("1", sources=[_quoted_success()])]

    report = backfill_quoted_sources(store, fresh)

    assert report.already_present == 1
    assert report.sources_attached == 0
    assert store["1"].content.fetched_at == _T0  # untouched → no needless re-enrich


def test_backfill_records_an_unreadable_quote_as_a_failure_not_as_silence():
    """A deleted/protected quote still gets a source — so #86's `NOT fetched` marker
    keeps firing and the state is demonstrable, not merely absent."""
    failure = ContentSourceFailure(
        kind="quoted_tweet", url="https://x.com/i/status/999", failure_reason="forbidden"
    )
    store = {"1": _quoted_item("1")}
    fresh = [_quoted_item("1", sources=[failure])]

    report = backfill_quoted_sources(store, fresh)

    assert report.unreadable == 1
    assert report.readable == 0
    assert store["1"].content.sources == [failure]


def test_backfill_ignores_fresh_items_not_already_in_the_store():
    """Backfill only touches known ids — it never adds new items (that is `extract`)."""
    store = {"1": _quoted_item("1")}
    fresh = [_quoted_item("2", sources=[_quoted_success()])]

    report = backfill_quoted_sources(store, fresh)

    assert report.items_seen == 0
    assert "2" not in store
    assert store["1"].content is None


def test_backfill_counts_the_quote_tweets_x_never_re_surfaced():
    """The honest diagnostic: how many quote-tweets are still evidence-less because X
    did not show them again. Without it the run reports success over a silent gap."""
    store = {"1": _quoted_item("1"), "2": _quoted_item("2")}
    fresh = [_quoted_item("1", sources=[_quoted_success()])]

    report = backfill_quoted_sources(store, fresh)

    assert report.quoted_items_not_seen == 1


def test_backfill_leaves_non_quoting_items_alone():
    store = {"1": _quoted_item("1", quoted_id=None)}
    fresh = [_quoted_item("1", quoted_id=None)]

    report = backfill_quoted_sources(store, fresh)

    assert report.sources_attached == 0
    assert report.quoted_items_not_seen == 0
    assert store["1"].content is None


# ------------------------------------- offline backfill: the quote is already OURS
#
# Measured on the real store: 199 of the 762 quote-tweets (26.1%) quote a post that
# is ALREADY an item in `items.json` — we hold its body and its author right now.
# Those need no capture at all: the repair is a join on `quoted_id`.


def _plain_item(item_id: str, *, handle: str, name: str, text: str) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/{handle}/status/{item_id}",
        author=Author(handle=handle, name=name),
        text=text,
        created_at=_T0,
        captured_at=_T0,
    )


def test_offline_backfill_joins_the_quoted_post_already_in_the_store():
    """No network: the quoted post IS an item we hold. Its body and author become the
    `quoted_tweet` source on the quoting item."""
    from xbrain.refresh import backfill_quoted_from_store

    quoted = _plain_item(
        "999", handle="karpathy", name="Andrej Karpathy", text="I am leaving OpenAI."
    )
    store = {"1": _quoted_item("1"), "999": quoted}

    report = backfill_quoted_from_store(store)

    assert report.sources_attached == 1
    assert report.readable == 1
    source = store["1"].content.sources[0]
    assert isinstance(source, ContentSourceSuccess)
    assert source.kind == "quoted_tweet"
    assert source.text == "I am leaving OpenAI."
    assert source.author == Author(handle="karpathy", name="Andrej Karpathy")
    assert source.url == quoted.url


def test_offline_backfill_leaves_a_quote_we_do_not_hold_alone():
    """`quoted_id` points at a post that is not in the store — nothing to join. It is
    NOT stamped as a failure: the post may be perfectly alive, we simply never captured
    it, and the re-capture route can still repair it."""
    from xbrain.refresh import backfill_quoted_from_store

    store = {"1": _quoted_item("1")}

    report = backfill_quoted_from_store(store)

    assert report.sources_attached == 0
    assert report.quoted_items_not_seen == 1
    assert store["1"].content is None


def test_offline_backfill_skips_an_empty_quoted_body():
    from xbrain.refresh import backfill_quoted_from_store

    store = {"1": _quoted_item("1"), "999": _plain_item("999", handle="b", name="B", text="")}

    report = backfill_quoted_from_store(store)

    assert report.sources_attached == 0
    assert store["1"].content is None


def test_offline_backfill_never_quotes_the_item_itself():
    """A self-referential `quoted_id` must not make an item its own evidence."""
    from xbrain.refresh import backfill_quoted_from_store

    store = {"1": _quoted_item("1", quoted_id="1")}

    report = backfill_quoted_from_store(store)

    assert report.sources_attached == 0
    assert store["1"].content is None


def test_offline_backfill_is_idempotent_and_preserves_other_sources():
    from xbrain.refresh import backfill_quoted_from_store

    article = ContentSourceSuccess(
        kind="external_article", url="https://example.com/p", text="the article body"
    )
    quoted = _plain_item("999", handle="k", name="K", text="quoted body")
    store = {"1": _quoted_item("1", sources=[article]), "999": quoted}

    first = backfill_quoted_from_store(store)
    second = backfill_quoted_from_store(store)

    assert first.sources_attached == 1
    assert second.sources_attached == 0
    assert second.already_present == 1
    assert article in store["1"].content.sources
