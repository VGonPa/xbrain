from datetime import datetime, timezone

from xbrain.cli import _taxonomy_health_lines
from xbrain.models import Author, Enrichment, Item, Topic


def _item(item_id: str, enrichment: Enrichment | None) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="a", name="A"),
        text="t",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        enriched=enrichment,
    )


def _enrichment(
    *,
    primary_topic: str,
    topics: list[str],
    topic_confidence: str | None = None,
    suggested_new_topics: list[str] | None = None,
) -> Enrichment:
    return Enrichment(
        enriched_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        executor="api",
        summary="s",
        primary_topic=primary_topic,
        topics=topics,
        topic_confidence=topic_confidence,
        suggested_new_topics=suggested_new_topics or [],
    )


def test_taxonomy_health_surfaces_misc_low_confidence_and_suggestions():
    store = {
        "1": _item(
            "1",
            _enrichment(
                primary_topic="misc",
                topics=["misc"],
                topic_confidence="low",
                suggested_new_topics=["modern-statistics"],
            ),
        ),
        "2": _item(
            "2",
            _enrichment(primary_topic="ai-coding", topics=["ai-coding"], topic_confidence="high"),
        ),
        "3": _item("3", None),
    }
    vocab = [
        Topic(slug="ai-coding", description="d"),
        Topic(slug="misc", description="d"),
        Topic(slug="unused-topic", description="d"),
    ]

    output = "\n".join(_taxonomy_health_lines(store, vocab, top=5))

    assert "3 total · 2 enriched · 1 pending" in output
    assert "misc: 1 (50.0%)" in output
    assert "confidence high/medium/low/unknown: 1/0/1/0" in output
    assert "- modern-statistics: 1" in output
    assert "unused-topic" in output
