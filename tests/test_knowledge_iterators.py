# tests/test_knowledge_iterators.py
"""The three atomic iterators the knowledge emitter and `evidence.py` SHARE (Plan 01 §2, M2).

THE PROBLEM. The spec (§2.2) says the knowledge contract must *reuse the atomic extractors
or factor out a shared definition, avoiding two hand-written lists that drift apart again*.
Read against the tree, three of the five selectors `evidence.py` consumes cannot be reused
as they stand:

* `first_source_text` returns **the first** source, truncated to `ARTICLE_CHAR_LIMIT` — the
  emitter needs **every** source (119 items carry more than one) and untruncated;
* `_content_image_descriptions` returns `list[str]` — the media INDEX and the media URL are
  gone, and both are needed to build a stable `surface_id` and a `locator.media_index`;
* `_video_frame_descriptions` returns a `list[str]` FLATTENED across every video source —
  which source and which frame produced a description is unrecoverable.

An emitter built on those would have to write its own walk over `content.sources` and
`item.media`: exactly the second hand-written list the spec forbids. So the shared thing is
the ITERATOR — index, URL and multiplicity preserved — and the existing selectors are
re-expressed on top of it.

WHAT THIS FILE PINS. The iterators' three load-bearing properties (index, URL,
multiplicity), plus the filters they own. That the SELECTORS still behave identically is
pinned separately, in `test_evidence_characterization.py` (the contract fingerprint) and in
`test_selectors_unchanged_by_the_shared_iterators` below.
"""

from __future__ import annotations

from datetime import datetime, timezone

from xbrain.executors.api import (
    _content_image_descriptions,
    _video_frame_descriptions,
    _video_source,
    first_source_text,
    iter_content_sources,
    iter_described_photos,
    iter_video_frames,
    quoted_source,
)
from xbrain.models import (
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaVideoPending,
    VideoFrame,
)
from xbrain.rubrics import ARTICLE_CHAR_LIMIT

UTC = timezone.utc


def _photo(
    *, url: str, index: int, description: str, decorative: bool = False
) -> MediaPhotoDescribed:
    return MediaPhotoDescribed(
        url=url,
        local_path=f"77/{index}.jpg",
        width=100,
        height=100,
        bytes_size=1000,
        downloaded_at=datetime(2026, 1, 1, tzinfo=UTC),
        is_decorative=decorative,
        description=description,
        description_lang="Spanish",
        description_version="describe/v1",
        described_at=datetime(2026, 1, 2, tzinfo=UTC),
    )


def _item(*, sources: list = None, media: list = None) -> Item:
    return Item(
        id="77",
        source="bookmark",
        url="https://x.com/a/status/77",
        author=Author(handle="a", name="A"),
        text="tweet text",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        media=media or [],
        content=(
            Content(fetched_at=datetime(2026, 1, 3, tzinfo=UTC), sources=sources)
            if sources is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# iter_content_sources — index, URL, multiplicity
# ---------------------------------------------------------------------------


def test_iter_content_sources_yields_the_index_in_content_sources() -> None:
    """The index is the position in `content.sources`, not a counter over the matches.

    A FAILURE sits at index 0 on purpose: an enumerate over the *filtered* list would
    report 0 and 1, and the emitter would write a `locator.source_index` pointing at the
    wrong source.
    """
    item = _item(
        sources=[
            ContentSourceFailure(
                kind="external_article", url="https://dead.example/a", failure_reason="not_found"
            ),
            ContentSourceSuccess(kind="external_article", url="https://ok.example/b", text="B"),
            ContentSourceSuccess(kind="external_article", url="https://ok.example/c", text="C"),
        ]
    )
    assert [index for index, _ in iter_content_sources(item, {"external_article"})] == [1, 2]


def test_iter_content_sources_preserves_multiplicity_where_the_selector_collapses_it() -> None:
    """Two sources of the same kind yield twice — `first_source_text` returns one.

    This is the whole reason the iterator exists: 119 items in the corpus carry more than
    one content source, and the knowledge emitter must surface every one of them.
    """
    item = _item(
        sources=[
            ContentSourceSuccess(kind="external_article", url="https://a.example", text="first"),
            ContentSourceSuccess(kind="external_article", url="https://b.example", text="second"),
        ]
    )
    assert [src.text for _, src in iter_content_sources(item, {"external_article"})] == [
        "first",
        "second",
    ]
    assert first_source_text(item, {"external_article"}) == "first"


def test_iter_content_sources_preserves_the_url() -> None:
    """The URL is what `source_key = sha1(kind\\0url)` hashes; losing it loses id stability."""
    item = _item(
        sources=[ContentSourceSuccess(kind="x_article", url="https://x.com/i/article/9", text="T")]
    )
    assert [src.url for _, src in iter_content_sources(item, {"x_article"})] == [
        "https://x.com/i/article/9"
    ]


def test_iter_content_sources_does_not_filter_on_text() -> None:
    """A no-speech video has an EMPTY transcript and real frames.

    `_video_source` never filtered on text, so the iterator must not either — otherwise
    re-expressing it would silently drop the 107 no-speech videos, and with them their
    frame descriptions. Callers that need text apply that filter themselves.
    """
    silent = ContentSourceSuccess(
        kind="x_video",
        url="https://video.example/v.mp4",
        text="",
        has_speech=False,
        frames=[VideoFrame(timestamp=1.0, local_path="77/frames/0.jpg", description="A slide.")],
    )
    item = _item(sources=[silent])
    assert [src for _, src in iter_content_sources(item, {"x_video"})] == [silent]
    assert _video_source(item) is silent


def test_iter_content_sources_is_empty_without_content() -> None:
    assert list(iter_content_sources(_item(), {"external_article"})) == []


# ---------------------------------------------------------------------------
# iter_described_photos — the media index survives the filters
# ---------------------------------------------------------------------------


def test_iter_described_photos_yields_the_media_index_not_a_match_counter() -> None:
    """Index 3 must be reported as 3.

    A video, a merely-downloaded photo and a decorative photo precede the only
    content-bearing one. An enumerate over the filtered list would call it 0, and the
    emitter would build `surface_id`/`locator.media_index` for the wrong entry.
    """
    described = _photo(url="https://pbs.example/real.jpg", index=3, description="A chart.")
    item = _item(
        media=[
            MediaVideoPending(url="https://video.example/v.mp4"),
            MediaPhotoDownloaded(
                url="https://pbs.example/plain.jpg",
                local_path="77/1.jpg",
                width=10,
                height=10,
                bytes_size=10,
                downloaded_at=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            _photo(url="https://pbs.example/avatar.jpg", index=2, description="", decorative=True),
            described,
        ]
    )
    assert list(iter_described_photos(item)) == [(3, described)]


def test_iter_described_photos_skips_decorative_and_empty_descriptions() -> None:
    """The non-decorative seam, owned by the iterator so both consumers share it.

    Spec §4: *decorative descriptions emit no chunks*. Keeping the filter here is what
    stops an avatar from becoming an indexable surface AND keeps
    `_content_image_descriptions` byte-identical.
    """
    item = _item(
        media=[
            _photo(url="https://pbs.example/a.jpg", index=0, description="", decorative=True),
            _photo(url="https://pbs.example/b.jpg", index=1, description="   "),
            _photo(url="https://pbs.example/c.jpg", index=2, description="Real content."),
        ]
    )
    assert [entry.url for _, entry in iter_described_photos(item)] == [
        "https://pbs.example/b.jpg",
        "https://pbs.example/c.jpg",
    ]


# ---------------------------------------------------------------------------
# iter_video_frames — which source, which frame
# ---------------------------------------------------------------------------


def test_iter_video_frames_reports_the_owning_source_and_the_frame_index() -> None:
    """The flattened `list[str]` cannot say which video or which slide a caption came from.

    `source_key` for a frame is `<the video's source_key>.f<frame index>`, so both indices
    are load-bearing. The empty-description frame at index 0 is skipped, and the surviving
    frame must still report index 1 — a positional counter would call it 0.
    """
    first_video = ContentSourceSuccess(
        kind="x_video",
        url="https://video.example/one.mp4",
        text="t1",
        frames=[
            VideoFrame(timestamp=0.0, local_path="77/frames/0.jpg", description=""),
            VideoFrame(timestamp=5.0, local_path="77/frames/1.jpg", description="Slide one."),
        ],
    )
    second_video = ContentSourceSuccess(
        kind="x_video",
        url="https://video.example/two.mp4",
        text="t2",
        frames=[VideoFrame(timestamp=2.0, local_path="77/frames/2.jpg", description="Slide two.")],
    )
    item = _item(
        sources=[
            ContentSourceSuccess(kind="external_article", url="https://a.example", text="A"),
            first_video,
            second_video,
        ]
    )
    assert [
        (source_index, source.url, frame_index, frame.description)
        for source_index, source, frame_index, frame in iter_video_frames(item)
    ] == [
        (1, "https://video.example/one.mp4", 1, "Slide one."),
        (2, "https://video.example/two.mp4", 0, "Slide two."),
    ]


# ---------------------------------------------------------------------------
# The selectors, re-expressed, still behave exactly as before
# ---------------------------------------------------------------------------


def _kitchen_sink() -> Item:
    return _item(
        sources=[
            ContentSourceFailure(
                kind="external_article", url="https://dead.example", failure_reason="not_found"
            ),
            ContentSourceSuccess(
                kind="external_article",
                url="https://a.example",
                text="A" * (ARTICLE_CHAR_LIMIT + 500),
            ),
            ContentSourceSuccess(kind="external_article", url="https://b.example", text="second"),
            ContentSourceSuccess(
                kind="quoted_tweet",
                url="https://x.com/jj/status/1",
                text="quoted body",
                author=Author(handle="JosephJacks_", name="JJ"),
            ),
            ContentSourceSuccess(
                kind="x_video",
                url="https://video.example/v.mp4",
                text="transcript",
                frames=[
                    VideoFrame(timestamp=0.0, local_path="77/frames/0.jpg", description="Slide."),
                    VideoFrame(timestamp=1.0, local_path="77/frames/1.jpg", description=""),
                ],
            ),
        ],
        media=[
            _photo(url="https://pbs.example/a.jpg", index=0, description="", decorative=True),
            _photo(url="https://pbs.example/b.jpg", index=1, description="A chart."),
        ],
    )


def test_selectors_unchanged_by_the_shared_iterators() -> None:
    """The five selectors' observable behaviour, asserted against hand-written expectations.

    NOT `selector(item) == [x for x in iterator(item)]` — that is the tautology CLAUDE.md
    rule 1 names: it is green by construction the moment delegation exists and pins
    nothing. These are the properties the SPEC of each selector promises, spelled out
    independently: first-only, truncation at `ARTICLE_CHAR_LIMIT`, the decorative filter,
    the empty-caption filter, and the flattening across sources.
    """
    item = _kitchen_sink()

    body = first_source_text(item, {"external_article"})
    assert body is not None
    assert len(body) == ARTICLE_CHAR_LIMIT, "first_source_text must still truncate"
    assert body == "A" * ARTICLE_CHAR_LIMIT, "and it must still pick the FIRST source"

    quoted = quoted_source(item)
    assert quoted is not None and quoted.author is not None
    assert quoted.author.handle == "JosephJacks_"

    video = _video_source(item)
    assert video is not None and video.url == "https://video.example/v.mp4"

    assert _content_image_descriptions(item) == ["A chart."]
    assert _video_frame_descriptions(item) == ["Slide."]
