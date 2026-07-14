# tests/test_checker_evidence_binding.py
"""The CHECKER searches exactly the shared evidence surfaces — bound BEHAVIOURALLY.

PR-F's cross-component test claimed to bind four components. Its checker leg compared
`evidence.evidence_text` to `evidence.evidence_surfaces` — both from the same module, so it
asserted `evidence.py` against ITSELF and bound nothing. Meanwhile the checker kept its own
hand-written list, which had already drifted: it admitted neither the VIDEO TITLE nor the
FETCHED ARTICLE TITLE, both of which the generators ship and the judge reads, so every name
sitting in a title was reported ungrounded — false positives manufactured by the one
component whose entire value is being a trustworthy instrument.

The checker now imports the shared `evidence_text`, which makes an identity test
(`entity_grounding.evidence_text == evidence.evidence_text`) a tautology AGAIN — the module
attribute IS the same object. So this binds through the checker's PUBLIC SCAN instead: give
it an output whose only grounding is a specific surface, and assert what it flags. No
re-exported name and no copied implementation can satisfy these; only actually searching the
right surfaces can.
"""

from datetime import datetime, timezone

import pytest

from xbrain.entity_grounding import scan_store
from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    VideoFrame,
)


def _item(
    *,
    summary: str = "Nothing notable.",
    digest: str = "",
    video_title: str = "An untitled clip",
    article_title: str = "An untitled piece",
    transcript: str = "the speaker discusses model training at length",
    article_body: str = "the article body discusses model training",
    tweet: str = "worth a read",
    folder: str | None = None,
    link_url: str = "https://example.com/a",
) -> Item:
    return Item(
        id="7",
        source="bookmark",
        url="https://x.com/a/status/7",
        author=Author(handle="someone", name="Some One"),
        text=tweet,
        created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url=link_url, domain="example.com")],
        bookmark_folder=folder,
        enriched=Enrichment(
            enriched_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
            executor="claude-code",
            summary=summary,
            primary_topic="ai-coding",
            topics=["ai-coding"],
        ),
        content=Content(
            fetched_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://x.com/v",
                    title=video_title,
                    text=transcript,
                    has_speech=True,
                    digest=digest,
                    frames=[
                        VideoFrame(
                            timestamp=0.0, local_path="7/frames/0.png", description="A slide."
                        )
                    ],
                ),
                ContentSourceSuccess(
                    kind="external_article",
                    url="https://example.com/a",
                    title=article_title,
                    text=article_body,
                ),
            ],
        ),
    )


def _ungrounded(item: Item, target: str) -> list[str]:
    return [name for record in scan_store({"7": item}, target) for name in record.ungrounded]


# ------------------------------------------------- the drift the private list actually had


def test_a_name_grounded_ONLY_in_the_video_title_is_not_flagged():
    """The hand-rolled list never searched the video title, so a summary naming the talk was
    reported ungrounded — a false positive against a generator that WAS handed the title."""
    item = _item(
        summary="Kalamazoo Dynamics explains how the training loop works.",
        video_title="Kalamazoo Dynamics on training loops",
    )
    assert "Kalamazoo Dynamics" not in _ungrounded(item, "summary")


def test_a_name_grounded_ONLY_in_the_article_title_is_not_flagged():
    """Same drift, the other title."""
    item = _item(
        summary="The piece in Zolmander Review argues the loop is the bottleneck.",
        article_title="Zolmander Review: the loop is the bottleneck",
    )
    assert "Zolmander Review" not in _ungrounded(item, "summary")


# ------------------------------------------------- the control: it CAN still flag


def test_a_name_on_no_surface_at_all_IS_flagged():
    """Without this, every test above would pass on a checker that flags nothing."""
    item = _item(summary="Quorvex Systems raised a round last week.")
    assert "Quorvex Systems" in _ungrounded(item, "summary")


# ------------------------------------------------- what is NOT evidence


def test_a_name_readable_only_off_the_URL_IS_flagged():
    """A URL is topic signal, never a name — the D1 defect, from the checker's side."""
    item = _item(
        summary="An Axios piece covers the jobs question.",
        link_url="https://www.axios.com/2025/05/28/ai-jobs-anthropic",
    )
    assert "Axios" in _ungrounded(item, "summary")


def test_a_name_readable_only_off_the_bookmark_folder_IS_flagged():
    """The folder is the USER's filing label, not content about the post."""
    item = _item(summary="Quorvex Systems is the subject here.", folder="Quorvex Systems")
    assert "Quorvex Systems" in _ungrounded(item, "summary")


# ------------------------------------------------- evidence is TARGET-dependent


def test_the_article_grounds_a_summary_but_NOT_a_digest():
    """The digest generator is never handed the article, so a name that appears only there
    must be flagged in a DIGEST and accepted in a SUMMARY. One fixture, two verdicts — the
    target-dependence, asserted from the checker's side."""
    item = _item(
        summary="Quorvex Systems built the trainer.",
        digest="Quorvex Systems built the trainer.",
        article_body="Quorvex Systems built the trainer described here.",
    )
    assert "Quorvex Systems" not in _ungrounded(item, "summary")  # the article IS summary evidence
    assert "Quorvex Systems" in _ungrounded(item, "digest")  # …and is NOT digest evidence


def test_topics_is_refused_loudly_rather_than_scored_vacuously():
    """An unscannable target must RAISE, never return zero findings that read as a clean bill
    of health from an instrument structurally incapable of finding anything."""
    with pytest.raises(ValueError, match="cannot be entity-scanned"):
        scan_store({"7": _item()}, "topics")


# ---------------------------------------------------------------- the quoted post
#
# #98/#94 made the quoted post an evidence surface for EVERY target. Its `@handle (Name)`
# is the grounding for naming the third party a quote-tweet is sharing — and the whole
# point of #86's attribution rule. If the checker cannot see it, it manufactures a false
# positive on a CORRECT attribution: the instrument whose only value is being trustworthy,
# flagging the very output the evidence supports.


def _quoting_item(summary: str) -> Item:
    item = _item(summary=summary)
    item.quoted_id = "999"
    item.content.sources = [
        *item.content.sources,
        ContentSourceSuccess(
            kind="quoted_tweet",
            url="https://x.com/karpathy/status/999",
            text="I am leaving OpenAI.",
            author=Author(handle="karpathy", name="Andrej Karpathy"),
        ),
    ]
    return item


def test_a_name_from_the_QUOTED_POST_author_is_grounded():
    """The quoted account's display name lives in the judge's LABEL, and `evidence_text`
    strips labels — so `evidence.py` puts it in the surface's VALUES. If the checker's blob
    ever loses it, this goes red: a correct attribution reported as invented."""
    records = scan_store(
        {"1": _quoting_item("Andrej Karpathy anuncia que deja OpenAI.")}, "summary"
    )
    assert records == []


def test_a_name_from_the_QUOTED_POST_body_is_grounded():
    records = scan_store({"1": _quoting_item("El post citado habla de OpenAI.")}, "summary")
    assert records == []


def test_a_name_no_surface_names_is_still_flagged_on_a_quote_tweet():
    """The other arm — the quoted surface must not become a blanket amnesty."""
    records = scan_store({"1": _quoting_item("Karpathy se une a Anthropic.")}, "summary")
    assert [e for r in records for e in r.ungrounded] == ["Anthropic"]


# ------------------------------------------------------------------ the URL hole
#
# 1,281 of the 2,168 items carry a URL inside their OWN tweet text, and `[Tweet]` is the
# post's words verbatim — so a URL rides into the evidence blob. A naive substring search
# then grounds "Anthropic" in the slug of a link that was NEVER FETCHED. That is the exact
# failure this whole workstream started from ("Axios", "Anthropic", read off
# `axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic`), and the checker is
# the component that does the matching, so the strip belongs here.


def test_a_name_read_off_a_URL_IN_THE_TWEET_is_NOT_grounded():
    """Both names sit in the slug of a link that was NEVER FETCHED. Neither is grounded.

    (Named mid-sentence on purpose: a sentence-initial capital is `uncertain` by design,
    and this test is about grounding, not about the confidence split.)"""
    item = _item(summary="Según Axios, la empresa Anthropic recorta empleos.")
    item.text = "Worth reading https://axios.com/2025/05/28/ai-jobs-white-collar-anthropic"

    flagged = [e for r in scan_store({"1": item}, "summary") for e in r.ungrounded]

    assert "Anthropic" in flagged
    assert "Axios" in flagged


def test_stripping_a_URL_does_not_swallow_the_words_around_it():
    """The strip must remove the URL, not the sentence. A name the tweet ACTUALLY says
    stays grounded even when a URL sits beside it."""
    item = _item(summary="Anthropic recorta empleos.")
    item.text = "Anthropic just posted this https://example.com/a/b-anthropic-c"

    assert scan_store({"1": item}, "summary") == []
