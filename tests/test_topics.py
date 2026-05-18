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
