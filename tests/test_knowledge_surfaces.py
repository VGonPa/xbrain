# tests/test_knowledge_surfaces.py
"""The surface emitter: item/topic -> KnowledgeSurface (Plan 01 §3.2, steps 7c-13c).

The emitter is where the three deliberate differences from `evidence.py` become real
(Plan 01 §2): NO target scoping (every surface, always), NO truncation (spec §2.2 — a
retrieval layer cannot be capped by a prompt's budget), and FULL multiplicity (119 items
carry more than one content source, and `evidence` takes the first).

Every test here names the store fact it protects, because the failure modes are silent: a
quoted post attributed to the poster reads perfectly, a truncated article body looks like
an article, and a decorative avatar emitted as a surface just adds noise nobody traces.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge.surfaces import (
    hydrate_verification,
    item_surfaces,
    knowledge_item,
    topic_record,
    topic_surfaces,
)
from xbrain.models import (
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoDescribed,
    MediaPhotoFailed,
    MediaPhotoPending,
    Topic,
    TopicPage,
    VerificationVerdict,
    VideoFrame,
)
from xbrain.rubrics import ARTICLE_CHAR_LIMIT

UTC = timezone.utc
FIXTURES = Path(__file__).parent / "fixtures"


def _item(**overrides) -> Item:
    defaults = dict(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        text="the post body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        captured_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    return Item(**{**defaults, **overrides})


def _by_type(surfaces, surface_type: str):
    return [s for s in surfaces if s.surface_type == surface_type]


def _load_fixture_item(name: str) -> Item:
    return Item.model_validate(json.loads((FIXTURES / name).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# 7c — producer / produced_at come from the store, not from None
# ---------------------------------------------------------------------------


def test_image_description_producer_and_produced_at_come_from_the_photo() -> None:
    """M4: spec §3.4 requires the producing component and the instant of generation.

    Both already exist in the store for a described photo, so leaving them `None` would be
    discarding data we have. Seen red by dropping `described_at` in the emitter: the
    surface then claims not to know when a description it is serving was produced.
    """
    item = _item(
        media=[
            MediaPhotoDescribed(
                url="https://pbs.example/a.jpg",
                local_path="42/0.jpg",
                width=10,
                height=10,
                bytes_size=10,
                downloaded_at=datetime(2026, 1, 3, tzinfo=UTC),
                is_decorative=False,
                description="A bar chart.",
                description_lang="Spanish",
                description_version="describe/v7",
                described_at=datetime(2026, 1, 4, tzinfo=UTC),
            )
        ]
    )
    (surface,) = _by_type(item_surfaces(item), "image_description")
    assert surface.producer == "describe/v7"
    assert surface.produced_at == datetime(2026, 1, 4, tzinfo=UTC)


def test_summary_producer_and_produced_at_come_from_the_enrichment() -> None:
    item = _item(
        enriched=Enrichment(
            enriched_at=datetime(2026, 2, 1, tzinfo=UTC),
            executor="claude-code",
            summary="Un resumen.",
        )
    )
    (surface,) = _by_type(item_surfaces(item), "summary")
    assert (surface.producer, surface.produced_at) == (
        "claude-code",
        datetime(2026, 2, 1, tzinfo=UTC),
    )


def test_configured_commands_become_the_producer_of_asr_and_vlm_surfaces() -> None:
    """The transcript's producer is the transcribe command, not a guess.

    It is the one producer that does NOT live in the store, so it is passed in. CLAUDE.md
    documents why it matters: parakeet does not fail on Spanish audio, it INVENTS — so
    which backend produced a transcript is exactly the kind of thing a reader must be able
    to recover after the fact.
    """
    item = _item(
        content=Content(
            fetched_at=datetime(2026, 3, 1, tzinfo=UTC),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://v.example/v.mp4",
                    text="transcript body",
                    has_speech=True,
                    frames=[
                        VideoFrame(
                            timestamp=1.0, local_path="42/frames/0.jpg", description="Slide."
                        )
                    ],
                )
            ],
        )
    )
    surfaces = item_surfaces(
        item,
        transcribe_command="scripts/xbrain-transcribe-auto",
        vision_command="scripts/xbrain-vision",
    )
    assert _by_type(surfaces, "video_transcript")[0].producer == "scripts/xbrain-transcribe-auto"
    assert _by_type(surfaces, "video_frame")[0].producer == "scripts/xbrain-vision"
    assert _by_type(surfaces, "video_transcript")[0].produced_at == datetime(2026, 3, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# 8 — attribution: the poster is not the author of what they quote
# ---------------------------------------------------------------------------


def test_quoted_post_attribution_is_the_quoted_author_not_the_item_author() -> None:
    """Invariant 3 of spec §3.7, on a fixture derived from the real P2 item.

    `tests/fixtures/knowledge_quoted_p2.json` keeps what makes the case: the poster is not
    the author, and the quoted body is 3 943 chars. The corpus itself is not readable in
    CI (no `data/`), so the fixture is the test and the real store is re-verified in the
    report — the two registers of Plan 01 §1.
    """
    item = _load_fixture_item("knowledge_quoted_p2.json")
    (quoted,) = _by_type(item_surfaces(item), "quoted_post")
    assert quoted.attribution is not None
    assert quoted.attribution.handle == "JosephJacks_"
    assert quoted.attribution.handle != item.author.handle
    (post,) = _by_type(item_surfaces(item), "post")
    assert post.attribution == item.author, "the poster still owns their own tweet"


# ---------------------------------------------------------------------------
# 9, 10 — multiplicity and no truncation
# ---------------------------------------------------------------------------


def test_every_source_of_a_kind_is_emitted_not_just_the_first() -> None:
    """119 items carry more than one content source; `evidence` takes the first.

    Seen red by emitting `first_source_text`: only one article surface comes back and the
    second body is unreachable by any query.
    """
    item = _item(
        content=Content(
            fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
            sources=[
                ContentSourceSuccess(
                    kind="external_article", url="https://a.example", text="first body"
                ),
                ContentSourceSuccess(
                    kind="external_article", url="https://b.example", text="second body"
                ),
            ],
        )
    )
    articles = _by_type(item_surfaces(item), "external_article")
    assert [s.text for s in articles] == ["first body", "second body"]
    assert len({s.surface_id for s in articles}) == 2


def test_article_bodies_are_never_truncated() -> None:
    """Spec §2.2: *the full article cannot be limited by the ceiling used in a prompt*.

    `tests/fixtures/knowledge_long_article.json` (derived from case S2) holds a 20 147-char
    article. Seen red by reusing `first_source_text`, which caps at ARTICLE_CHAR_LIMIT
    (4 000) — the answer to S2 sits past that cap, so the truncated surface makes a
    verified golden-set case unanswerable.
    """
    item = _load_fixture_item("knowledge_long_article.json")
    (article,) = _by_type(item_surfaces(item), "external_article")
    assert len(article.text) == 20_147
    assert len(article.text) > ARTICLE_CHAR_LIMIT


def test_article_title_rides_on_the_surface() -> None:
    item = _load_fixture_item("knowledge_long_article.json")
    (article,) = _by_type(item_surfaces(item), "external_article")
    assert article.title == "Dario Amodei — On DeepSeek and Export Controls"


# ---------------------------------------------------------------------------
# 11, 12, 13 — what must NOT be emitted
# ---------------------------------------------------------------------------


def test_a_decorative_photo_emits_no_surface() -> None:
    """Spec §4: *empty decorative descriptions emit no chunks*.

    An avatar or a reaction meme indexed as a surface drags topic signal toward whatever it
    depicts. The filter is the SAME seam `_content_image_descriptions` uses
    (`iter_described_photos`), not a second copy of the rule.
    """
    item = _item(
        media=[
            MediaPhotoDescribed(
                url="https://pbs.example/avatar.jpg",
                local_path="42/0.jpg",
                width=10,
                height=10,
                bytes_size=10,
                downloaded_at=datetime(2026, 1, 3, tzinfo=UTC),
                is_decorative=True,
                description="",
                description_lang="Spanish",
                description_version="describe/v7",
                described_at=datetime(2026, 1, 4, tzinfo=UTC),
            )
        ]
    )
    assert _by_type(item_surfaces(item), "image_description") == []


@pytest.mark.parametrize(
    "entry",
    [
        MediaPhotoPending(url="https://pbs.example/p.jpg"),
        MediaPhotoFailed(
            url="https://pbs.example/f.jpg",
            failure_reason="http_4xx",
            attempts=1,
            last_attempt_at=datetime(2026, 1, 3, tzinfo=UTC),
        ),
    ],
)
def test_a_pending_or_failed_photo_emits_no_surface(entry) -> None:
    """There is no description to index — emitting an empty surface would fake evidence."""
    assert _by_type(item_surfaces(_item(media=[entry])), "image_description") == []


def test_a_no_speech_video_emits_frames_but_no_transcript() -> None:
    """107 of 259 `x_video` sources have `has_speech=False` and an empty transcript.

    Emitting an empty `video_transcript` would put a zero-length body in the index under a
    label that promises speech. The frames are real and must survive — they are often the
    only text such a video contributes.
    """
    item = _item(
        content=Content(
            fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
            sources=[
                ContentSourceSuccess(
                    kind="x_video",
                    url="https://v.example/silent.mp4",
                    text="",
                    has_speech=False,
                    frames=[
                        VideoFrame(
                            timestamp=1.0, local_path="42/frames/0.jpg", description="A slide."
                        )
                    ],
                )
            ],
        )
    )
    surfaces = item_surfaces(item)
    assert _by_type(surfaces, "video_transcript") == []
    assert [s.text for s in _by_type(surfaces, "video_frame")] == ["A slide."]


def test_a_fetch_failure_emits_no_text_but_a_structured_failure() -> None:
    """Spec §4: a failure emits no text, and `get` reports that the source failed and why."""
    item = _item(
        content=Content(
            fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
            sources=[
                ContentSourceFailure(
                    kind="external_article",
                    url="https://dead.example/a",
                    failure_reason="not_found",
                    error="404",
                )
            ],
        )
    )
    surfaces = item_surfaces(item)
    assert [s.surface_type for s in surfaces] == ["post"], "the failed article emitted text"
    record = knowledge_item(item)
    assert "external_article" not in record.content_kinds, (
        "a kind whose fetch failed must not be advertised as available — it would send the "
        "consumer to `get` for a body that does not exist"
    )
    assert [(f.kind, f.failure_reason) for f in record.failed_sources] == [
        ("external_article", "not_found")
    ]


def test_a_blank_surface_is_never_emitted() -> None:
    """Whitespace-only text is not a surface (Plan 01 §9)."""
    assert _by_type(item_surfaces(_item(text="   \n ")), "post") == []


# ---------------------------------------------------------------------------
# 13b — unfetched links carry their reason and emit no surface
# ---------------------------------------------------------------------------


def test_a_link_with_no_body_is_visible_with_its_recorded_reason() -> None:
    """m7 + spec §4: the consumer must SEE the URL and why there is no body.

    The reason comes from the store's own `FailureReason`, mapped through the total table
    in `knowledge.models` — never invented. Naming the cause is not permission to describe
    the content, so no surface is emitted for it.
    """
    item = _item(
        links=[Link(url="https://dead.example/a", domain="dead.example")],
        content=Content(
            fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
            sources=[
                ContentSourceFailure(
                    kind="external_article",
                    url="https://dead.example/a",
                    failure_reason="not_found",
                    error="404 Not Found",
                )
            ],
        ),
    )
    record = knowledge_item(item)
    assert [(link.url, link.reason) for link in record.unfetched_links] == [
        ("https://dead.example/a", "http_error")
    ]
    assert record.unfetched_links[0].detail == "the page no longer exists (HTTP 404)"


def test_a_link_nobody_ever_tried_is_reported_as_not_attempted() -> None:
    """Distinct from a failure: no attempt was recorded, so no cause may be named.

    `executors.api._failure_clause` already refuses to name a cause it did not measure;
    this is the same rule expressed in the read model.
    """
    item = _item(links=[Link(url="https://never.example/a", domain="never.example")])
    (link,) = knowledge_item(item).unfetched_links
    assert (link.reason, link.detail) == ("not_attempted", None)


def test_a_fetched_link_is_not_listed_as_unfetched() -> None:
    item = _item(
        links=[Link(url="https://ok.example/a", domain="ok.example")],
        content=Content(
            fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
            sources=[
                ContentSourceSuccess(
                    kind="external_article", url="https://ok.example/a", text="body"
                )
            ],
        ),
    )
    assert knowledge_item(item).unfetched_links == ()


# ---------------------------------------------------------------------------
# 13c — verification is hydrated from the live store and a stale verdict is not shown
# ---------------------------------------------------------------------------


def _verified_item(summary: str, fingerprint_source: str) -> Item:
    from xbrain.verification import contract_fingerprint, fingerprint_output

    item = _item(
        enriched=Enrichment(
            enriched_at=datetime(2026, 2, 1, tzinfo=UTC),
            executor="claude-code",
            summary=fingerprint_source,
        )
    )
    verdict = VerificationVerdict(
        verdict="PASS",
        output_fingerprint=fingerprint_output(item, "summary") or ("0" * 64),
        contract_fingerprint=contract_fingerprint(item, "summary", "Spanish"),
        verified_at=datetime(2026, 2, 2, tzinfo=UTC),
    )
    return item.model_copy(
        update={
            "enriched": item.enriched.model_copy(update={"summary": summary}),
            "verification": {"summary": verdict},
        }
    )


def test_a_current_verdict_is_hydrated_from_the_live_store() -> None:
    item = _verified_item("Un resumen.", "Un resumen.")
    assert hydrate_verification(item, "Spanish")["summary"].verdict == "PASS"


def test_a_stale_verdict_is_not_shown_at_all() -> None:
    """M5: the verdict was earned by a DIFFERENT output, so it is not served.

    This is why `verification` is not a persisted field of `KnowledgeSurface`:
    `surface_fingerprint` does not depend on the verdict, so a copy stored beside the text
    would keep asserting a PASS that the current output never earned. The check is the same
    one `generate._verdict_badge` applies — one definition, not two.

    Seen red by returning `item.verification` unfiltered.
    """
    item = _verified_item("Un resumen REESCRITO.", "Un resumen.")
    assert hydrate_verification(item, "Spanish") == {}


def test_surfaces_never_carry_a_verdict() -> None:
    item = _verified_item("Un resumen.", "Un resumen.")
    assert all(not hasattr(surface, "verification") for surface in item_surfaces(item))


# ---------------------------------------------------------------------------
# The item record and the empty-content case
# ---------------------------------------------------------------------------


def test_an_item_without_content_still_emits_post_and_summary() -> None:
    """961 of 2 404 items (40 %) have no `content` at all — that is not an error.

    Plan 01 §9: they emit `post` + `summary` + topics, with `content_kinds=()`.
    """
    item = _item(
        enriched=Enrichment(
            enriched_at=datetime(2026, 2, 1, tzinfo=UTC),
            executor="api",
            summary="Un resumen.",
            primary_topic="ai-coding",
            topics=["ai-coding", "agentic-engineering"],
        )
    )
    record = knowledge_item(item)
    assert record.content_kinds == ()
    assert set(record.available_surfaces) == {"post", "summary"}
    assert record.topics == ("ai-coding", "agentic-engineering")


def test_topics_are_deduplicated_with_the_primary_first() -> None:
    """A duplicated or re-listed primary topic must not produce a duplicate entry.

    Determinism matters because `KnowledgeChunk.topics` is copied from here and feeds a
    filter; a list whose order depends on how the enrichment happened to be written would
    make two identical items rank differently.
    """
    item = _item(
        enriched=Enrichment(
            enriched_at=datetime(2026, 2, 1, tzinfo=UTC),
            executor="api",
            summary="s",
            primary_topic="ai-coding",
            topics=["agentic-engineering", "ai-coding", "agentic-engineering"],
        )
    )
    assert knowledge_item(item).topics == ("ai-coding", "agentic-engineering")


def test_note_path_is_resolved_only_when_the_note_exists(tmp_path: Path) -> None:
    """A path to a note nobody generated would be a broken promise to the consumer."""
    item = _item()
    assert knowledge_item(item, vault_dir=tmp_path).note_path is None
    notes = tmp_path / "items"
    notes.mkdir()
    from xbrain.notes_io import note_filename

    (notes / note_filename(item)).write_text("x", encoding="utf-8")
    assert knowledge_item(item, vault_dir=tmp_path).note_path == f"items/{note_filename(item)}"


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------


def test_topic_surfaces_carry_their_own_origins() -> None:
    """`topic_description` is `unknown` because the vocabulary does not record a producer.

    Spec §4 says so explicitly, and `ORIGIN_TRUST` then classifies `unknown` as synthesis —
    the fail-closed path. The overview and the notes ARE known to be LLM output.
    """
    topic = Topic(slug="ai-coding", description="Coding with AI assistance.")
    page = TopicPage(
        slug="ai-coding",
        overview="An overview paragraph.",
        notes=["A first note.", "A second note."],
        synthesized_at=datetime(2026, 4, 1, tzinfo=UTC),
        post_count_at_synth=10,
    )
    surfaces = topic_surfaces(topic, page)
    origins = {s.surface_type: s.origin for s in surfaces}
    assert origins == {
        "topic_description": "unknown",
        "topic_overview": "llm",
        "topic_note": "llm",
    }
    assert len(_by_type(surfaces, "topic_note")) == 2
    assert [s.locator.note_index for s in _by_type(surfaces, "topic_note")] == [0, 1]


def test_a_topic_without_a_page_is_stale_with_an_empty_overview() -> None:
    """Plan 01 §9: a vocabulary entry with no `TopicPage` is not an error.

    It is the normal state of a topic that has never been synthesized, and reporting it as
    stale is what tells a consumer the overview they did not get is missing, not empty.
    """
    record = topic_record(Topic(slug="new-topic", description="d"), None, (), ())
    assert (record.overview, record.synthesized_at, record.stale) == ("", None, True)
    assert topic_surfaces(Topic(slug="new-topic", description="d"), None)[0].surface_type == (
        "topic_description"
    )


def test_topic_staleness_compares_the_live_count_with_the_synthesized_one() -> None:
    topic = Topic(slug="ai-coding", description="d")
    page = TopicPage(
        slug="ai-coding",
        overview="o",
        notes=[],
        synthesized_at=datetime(2026, 4, 1, tzinfo=UTC),
        post_count_at_synth=2,
    )
    assert topic_record(topic, page, ("1", "2"), ("3",)).stale is False
    assert topic_record(topic, page, ("1", "2", "3"), ()).stale is True
