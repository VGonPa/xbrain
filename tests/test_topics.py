# tests/test_topics.py
from datetime import datetime, timezone

from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.topics import TopicPosts, compute_topic_posts


def _enriched(item_id: str, primary: str, topics: list[str], day: int = 1) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text=f"post {item_id}",
        created_at=datetime(2026, 5, day, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        enriched=Enrichment(
            enriched_at=datetime.now(timezone.utc),
            executor="api",
            summary="s",
            primary_topic=primary,
            topics=topics,
        ),
    )


_VOCAB = [Topic(slug="ai-coding", description="d"), Topic(slug="career", description="d")]


def test_compute_topic_posts_splits_primary_and_also_relevant():
    store = {
        "1": _enriched("1", "ai-coding", ["ai-coding", "career"]),
        "2": _enriched("2", "career", ["career"]),
    }
    posts = compute_topic_posts(store, _VOCAB)
    assert [i.id for i in posts["ai-coding"].primary] == ["1"]
    assert [i.id for i in posts["ai-coding"].also] == []
    assert [i.id for i in posts["career"].primary] == ["2"]
    assert [i.id for i in posts["career"].also] == ["1"]


def test_compute_topic_posts_ignores_unenriched_items():
    store = {
        "1": Item(
            id="1",
            source="bookmark",
            url="u",
            author=Author(handle="a", name="A"),
            text="t",
            created_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            captured_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
    }
    posts = compute_topic_posts(store, _VOCAB)
    assert posts["ai-coding"].total == 0


def test_compute_topic_posts_sorts_newest_first():
    store = {
        "1": _enriched("1", "ai-coding", ["ai-coding"], day=1),
        "2": _enriched("2", "ai-coding", ["ai-coding"], day=9),
    }
    posts = compute_topic_posts(store, _VOCAB)
    assert [i.id for i in posts["ai-coding"].primary] == ["2", "1"]


def test_topic_posts_total_counts_both_blocks():
    tp = TopicPosts()
    tp.primary.append(_enriched("1", "ai-coding", ["ai-coding"]))
    tp.also.append(_enriched("2", "career", ["career", "ai-coding"]))
    assert tp.total == 2


def _topic_page(slug: str, count: int):
    from datetime import datetime, timezone

    from xbrain.models import TopicPage

    return TopicPage(
        slug=slug,
        overview="El overview.",
        notes=["Nota importante."],
        synthesized_at=datetime(2026, 5, 18, tzinfo=timezone.utc),
        post_count_at_synth=count,
    )


def test_render_topic_page_has_frontmatter_overview_and_post_blocks():
    from xbrain.topics import render_topic_page

    posts = TopicPosts()
    posts.primary.append(_enriched("1", "ai-coding", ["ai-coding"]))
    posts.also.append(_enriched("2", "career", ["career", "ai-coding"]))
    page = _topic_page("ai-coding", count=2)
    rendered = render_topic_page(_VOCAB[0], posts, page)
    assert "tags: [x-knowledge-topic, ai-coding]" in rendered
    assert "posts: 2" in rendered
    assert "El overview." in rendered
    assert "## Notas importantes" in rendered
    assert "## Posts primarios (1)" in rendered
    assert "## También relevante (1)" in rendered
    assert "[[items/" in rendered


def test_render_topic_page_marks_a_stale_overview():
    from xbrain.topics import render_topic_page

    posts = TopicPosts()
    for n in range(5):
        posts.primary.append(_enriched(str(n), "ai-coding", ["ai-coding"]))
    page = _topic_page("ai-coding", count=2)  # synthesized when the topic had 2
    rendered = render_topic_page(_VOCAB[0], posts, page)
    assert "Overview desactualizado" in rendered


def test_render_topic_page_handles_a_missing_overview():
    from xbrain.topics import render_topic_page

    posts = TopicPosts()
    posts.primary.append(_enriched("1", "ai-coding", ["ai-coding"]))
    rendered = render_topic_page(_VOCAB[0], posts, None)
    assert "Overview pendiente" in rendered


def test_write_topic_pages_skips_empty_topics(tmp_path):
    from xbrain.topics import write_topic_pages

    posts = {
        "ai-coding": TopicPosts(),
        "career": TopicPosts(),
    }
    posts["ai-coding"].primary.append(_enriched("1", "ai-coding", ["ai-coding"]))
    written = write_topic_pages(tmp_path, _VOCAB, posts, {})
    assert written == 1
    assert (tmp_path / "topics" / "ai-coding.md").exists()
    assert not (tmp_path / "topics" / "career.md").exists()


def test_write_topic_pages_preserves_user_tail(tmp_path):
    from xbrain.topics import write_topic_pages

    posts = {"ai-coding": TopicPosts()}
    posts["ai-coding"].primary.append(_enriched("1", "ai-coding", ["ai-coding"]))
    write_topic_pages(tmp_path, [_VOCAB[0]], posts, {})
    page = tmp_path / "topics" / "ai-coding.md"
    page.write_text(page.read_text(encoding="utf-8") + "MI ANOTACION", encoding="utf-8")
    write_topic_pages(tmp_path, [_VOCAB[0]], posts, {})
    assert "MI ANOTACION" in page.read_text(encoding="utf-8")


def test_topics_needing_synth_picks_missing_and_grown_topics():
    from xbrain.topics import topics_needing_synth

    posts = {"ai-coding": TopicPosts(), "career": TopicPosts()}
    for n in range(30):
        posts["ai-coding"].primary.append(_enriched(f"a{n}", "ai-coding", ["ai-coding"]))
    posts["career"].primary.append(_enriched("c", "career", ["career"]))
    pages = {"ai-coding": _topic_page("ai-coding", count=2)}  # grew 2 -> 30
    needing = topics_needing_synth(_VOCAB, posts, pages, threshold=25, resynth=False)
    # ai-coding grew past the threshold; career has no page yet.
    assert set(needing) == {"ai-coding", "career"}


def test_topics_needing_synth_resynth_takes_any_changed_topic():
    from xbrain.topics import topics_needing_synth

    posts = {"ai-coding": TopicPosts()}
    for n in range(3):
        posts["ai-coding"].primary.append(_enriched(f"a{n}", "ai-coding", ["ai-coding"]))
    pages = {"ai-coding": _topic_page("ai-coding", count=2)}  # grew 2 -> 3 (below threshold)
    assert topics_needing_synth([_VOCAB[0]], posts, pages, threshold=25, resynth=False) == []
    assert topics_needing_synth([_VOCAB[0]], posts, pages, threshold=25, resynth=True) == [
        "ai-coding"
    ]


def test_build_topic_inputs_collects_post_summaries():
    from xbrain.topics import build_topic_inputs

    posts = {"ai-coding": TopicPosts()}
    posts["ai-coding"].primary.append(_enriched("1", "ai-coding", ["ai-coding"]))
    inputs = build_topic_inputs(["ai-coding"], _VOCAB, posts)
    assert inputs[0].slug == "ai-coding"
    assert inputs[0].summaries == ["s"]


def test_build_topic_inputs_rejects_an_unknown_slug():
    import pytest

    from xbrain.topics import build_topic_inputs

    with pytest.raises(ValueError, match="vocabulary"):
        build_topic_inputs(["not-a-real-topic"], _VOCAB, {})


def test_compute_topic_posts_dedups_duplicate_slugs_in_topics():
    # A store record with a duplicate slug in `enriched.topics` (pre-validation
    # data) must place the item once, not twice, in the also-relevant list.
    store = {
        "1": _enriched("1", "ai-coding", ["ai-coding", "career", "career"]),
    }
    posts = compute_topic_posts(store, _VOCAB)
    assert [i.id for i in posts["career"].also] == ["1"]


def test_merge_overviews_records_the_synthesis_count():
    from xbrain.topic_synth import OverviewJudgment
    from xbrain.topics import merge_overviews

    posts = {"ai-coding": TopicPosts()}
    posts["ai-coding"].primary.append(_enriched("1", "ai-coding", ["ai-coding"]))
    pages: dict = {}
    merge_overviews(pages, [OverviewJudgment(slug="ai-coding", overview="o", notes=["n"])], posts)
    assert pages["ai-coding"].overview == "o"
    assert pages["ai-coding"].post_count_at_synth == 1
