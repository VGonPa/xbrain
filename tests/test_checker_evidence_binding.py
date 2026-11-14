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
    MediaPhotoDescribed,
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


# ------------------------------------------- a domain-shaped token is not a URL
#
# I shipped a `strip_urls()` that removed every URL-shaped token from the evidence blob
# before matching a name. It scored **0 true positives and 5 false positives** on the real
# corpus, and it had to: the case it targeted — a name readable ONLY off an unfetched
# link's slug — was ALREADY handled, structurally, because `xbrain.evidence` derives no
# surface from `item.links` (#94). There was nothing left for a strip to catch.
#
# What it DID catch was evidence. A token that LOOKS like a domain is very often the
# CONTENT of a surface: a row in a rankings table an image description transcribes, a logo
# a vision model reads off a slide, a product name a speaker says out loud. Stripping it
# deletes the grounding and reports the generator for inventing what the item plainly
# shows.
#
# These pin both arms so the mistake cannot come back.


def test_a_domain_shaped_token_IN_AN_IMAGE_DESCRIPTION_grounds_the_name():
    """The real item that exposed this (1888651361094717717). The tweet MISSPELLS it
    ("GhatGPT"); the IMAGE — a vision-described traffic-share table — carries
    `chatgpt.com 2,33%`. The name is genuinely shown, in a table, as data."""
    item = _item(summary="ChatGPT tiene el 2,3% del tráfico web.")
    item.media = [
        MediaPhotoDescribed(
            url="https://pbs.twimg.com/media/X.jpg",
            local_path="7/0.jpg",
            width=4,
            height=3,
            bytes_size=512,
            downloaded_at=datetime(2026, 5, 24, tzinfo=timezone.utc),
            is_decorative=False,
            description="Tabla de dominios: google.com 29,21%, chatgpt.com 2,33%, x.com 1,92%.",
            description_lang="Spanish",
            description_version="v1",
            described_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc),
        )
    ]

    assert _ungrounded(item, "summary") == []


def test_a_domain_shaped_token_IN_A_TRANSCRIPT_grounds_the_name():
    """The speaker SAYS it: "a new website called Codream.ai". That is the video's content."""
    item = _item(
        digest="Presentan Codream, una web en preview.",
        transcript="What we shipped is a new website in our preview called Codream.ai.",
    )
    assert _ungrounded(item, "digest") == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN GAP, measured, deliberately not fixed here. A human typo is a mangled SOURCE, "
        "not an invention — the same species as the ASR corruption the fuzzy matcher exists "
        "for. But `_FUZZY_MIN_LEN = 8` skips the fuzzy pass entirely for a 7-character "
        "needle, so 'chatgpt' never reaches the 0.80 threshold even though 'ghatgpt' sits "
        "right there at ratio 0.857. Lowering it to 7 newly grounds 11 entities on the real "
        "corpus — and while 2 are real ('ChatGPT'), the rest are common-word collisions "
        "('Rumanía', 'Malasia', 'Cultura', 'Perdona', 'Vizcaya'), i.e. it would start HIDING "
        "inventions behind fuzzy hits. That is a precision trade to settle against the "
        "hand-labelled set, not by intuition. Pinned as xfail so it flips green the day "
        "someone does the measurement."
    ),
)
def test_a_MISSPELLED_name_grounded_ONLY_by_the_typo_is_not_yet_caught():
    """The residual false-positive shape: the typo'd tweet is the ONLY grounding.

    Note the real item that started this (1888651361094717717) does NOT hit this path — its
    IMAGE description carries `chatgpt.com 2,33%`, so it is grounded regardless (pinned
    above). This is the narrower case where nothing else names it."""
    item = _item(
        summary="ChatGPT tiene el 2,3% del tráfico web.",
        tweet="Google has nearly 1/3rd of the web traffic. GhatGPT 10x less than that (2.3%)",
        transcript="",
        article_body="",
    )

    assert _ungrounded(item, "summary") == []


def test_a_name_readable_ONLY_off_an_unfetched_LINK_is_still_flagged():
    """The other arm, and the reason no strip is needed: `item.links` is NO evidence
    surface (#94), so `axios.com/...-anthropic` never reaches the blob in the first place.
    The name is flagged structurally."""
    item = _item(
        summary="Según Axios, la empresa Anthropic recorta empleos.",
        link_url="https://www.axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic",
    )

    flagged = _ungrounded(item, "summary")

    assert "Anthropic" in flagged
    assert "Axios" in flagged
