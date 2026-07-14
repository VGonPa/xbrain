# tests/test_evidence.py
"""The ONE definition of what counts as evidence — its own spec.

The cross-component binding (generator ⊇ rubric == judge == checker) lives in
`test_evidence_contract.py`. This file pins the function itself: which surfaces each
target admits, and that an absent surface contributes nothing.
"""

from datetime import datetime, timezone

import pytest

from xbrain.evidence import SURFACE_KEYS, evidence_surfaces, evidence_text
from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Item,
    Link,
    MediaPhotoDescribed,
    VideoFrame,
)


def _item(*, video: bool = True, article: bool = True, thread: bool = True) -> Item:
    sources: list[ContentSourceSuccess] = []
    if video:
        sources.append(
            ContentSourceSuccess(
                kind="x_video",
                url="https://x.com/v",
                title="The scaling-laws talk",
                text="the transcript body",
                has_speech=True,
                frames=[
                    VideoFrame(timestamp=0.0, local_path="7/frames/0.png", description="A chart.")
                ],
            )
        )
    if article:
        sources.append(
            ContentSourceSuccess(
                kind="external_article",
                url="https://example.com/a",
                title="The TIME piece",
                text="the fetched article body",
            )
        )
    if thread:
        sources.append(
            ContentSourceSuccess(
                kind="thread",
                url="https://x.com/a/status/7",
                text="1/ the poster's own thread",
            )
        )
    return Item(
        id="7",
        source="bookmark",
        url="https://x.com/a/status/7",
        author=Author(handle="emollick", name="Ethan Mollick"),
        text="watch this",
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/a", domain="example.com")],
        media=[
            MediaPhotoDescribed(
                url="https://pbs.twimg.com/media/X.jpg",
                local_path="7/0.jpg",
                width=4,
                height=3,
                bytes_size=512,
                downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
                is_decorative=False,
                description="A bar chart of GPU prices.",
                description_lang="English",
                description_version="v1",
                described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
            )
        ],
        content=Content(fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc), sources=sources),
    )


# What each target's GENERATOR is actually handed — the whole point of the module.
DIGEST_KEYS = ("author", "video_title", "video_transcript", "video_frames", "tweet")
ENRICH_KEYS = DIGEST_KEYS + ("images", "article_title", "article", "thread")


def test_digest_evidence_is_the_video_and_the_post_only():
    """`export_video_digest_worksheet` hands the digest generator the transcript, the
    frames, the video title, the author and the tweet text — and NOTHING else. The
    article, the thread and the images are not on that worksheet, so they cannot be
    evidence for a digest: judging a digest against them excuses an invention the
    generator had no way to source."""
    keys = [s.key for s in evidence_surfaces(_item(), "digest")]
    assert set(keys) == set(DIGEST_KEYS)
    assert "article" not in keys
    assert "thread" not in keys
    assert "images" not in keys


@pytest.mark.parametrize("target", ["summary", "topics"])
def test_enrich_evidence_adds_what_the_enrich_worksheet_ships(target):
    """The enrich worksheet DOES ship the article, the thread and the image
    descriptions, and its rubric says to summarise the article's substance — so for a
    summary they are evidence. Checking a summary against the digest's narrower set
    would flag the generator for using evidence it was correctly given."""
    keys = {s.key for s in evidence_surfaces(_item(), target)}
    assert keys == set(ENRICH_KEYS)


def test_surface_keys_are_declared_per_target_without_an_item():
    """The declared set is a property of the TARGET, not of one item — the fingerprint
    (PR-D) and the rubric binding need it before any item is in hand."""
    assert set(SURFACE_KEYS["digest"]) == set(DIGEST_KEYS)
    assert set(SURFACE_KEYS["summary"]) == set(ENRICH_KEYS)
    assert SURFACE_KEYS["summary"] == SURFACE_KEYS["topics"]


def test_surfaces_carry_the_items_text_under_a_label():
    surfaces = {s.key: s for s in evidence_surfaces(_item(), "summary")}
    assert surfaces["author"].text == "@emollick (Ethan Mollick)"
    assert surfaces["video_title"].text == "The scaling-laws talk"
    assert surfaces["video_transcript"].text == "the transcript body"
    assert "A chart." in surfaces["video_frames"].text
    assert surfaces["article_title"].text == "The TIME piece"
    assert surfaces["article"].text == "the fetched article body"
    assert surfaces["thread"].text == "1/ the poster's own thread"
    assert "A bar chart of GPU prices." in surfaces["images"].text
    assert surfaces["tweet"].text == "watch this"
    assert surfaces["video_transcript"].label == "[Video transcript]"


def test_an_absent_surface_is_omitted_entirely():
    """No video, no article, no thread → those surfaces do not appear at all. An empty
    labelled block would tell the judge evidence exists where there is none."""
    item = _item(video=False, article=False, thread=False)
    keys = {s.key for s in evidence_surfaces(item, "summary")}
    assert keys == {"author", "tweet", "images"}


def test_a_handleless_author_contributes_no_author_surface():
    """The rubric calls the author block TRUSTED metadata; an empty one would present a
    garbage anchor as trustworthy (see #92)."""
    item = _item()
    item.author = Author(handle="", name="")
    assert "author" not in {s.key for s in evidence_surfaces(item, "summary")}


def test_evidence_text_is_the_flat_join_the_checker_reads():
    """The deterministic entity checker asks one question — does this name appear on ANY
    evidence surface? — so it needs the surfaces as one blob, from the SAME source of
    truth the judge and the generators use."""
    text = evidence_text(_item(), "digest")
    assert "the transcript body" in text
    assert "Ethan Mollick" in text  # the display name IS evidence (the rubric promises it)
    assert "The scaling-laws talk" in text  # the video title, which the worksheet ships
    assert "the fetched article body" not in text  # …but the article is not digest evidence
    assert "the fetched article body" in evidence_text(_item(), "summary")


def test_a_links_url_is_never_evidence_on_any_surface():
    """D1: the domain is topic signal, never a name and never content. `axios.com` must
    not be able to ground the name "Axios" — which is exactly how a summary in the
    corpus reconstructed a whole article from a URL slug."""
    for target in ("digest", "summary", "topics"):
        text = evidence_text(_item(), target)
        assert "example.com" not in text
        assert "https://" not in text
