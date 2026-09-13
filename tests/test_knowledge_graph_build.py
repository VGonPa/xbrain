"""Plan 04.2 — graph build over a fixture of KNOWN population.

Never asserts corpus figures: every item, topic and count below is built here.
"""

from __future__ import annotations

from datetime import UTC, datetime

from xbrain.knowledge.graph_build import build_graph_edges
from xbrain.models import Author, Enrichment, Item

_T = datetime(2026, 1, 1, tzinfo=UTC)


def _item(item_id: str, primary: str | None, topics: list[str]) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/u/status/{item_id}",
        author=Author(handle="u", name="U"),
        text=f"tweet {item_id}",
        created_at=_T,
        captured_at=_T,
        enriched=Enrichment(
            enriched_at=_T,
            executor="manual",
            summary="s",
            primary_topic=primary,
            topics=topics,
        ),
    )


def test_primary_topic_and_topics_each_produce_their_own_edge_kind() -> None:
    store = {"1": _item("1", primary="agents", topics=["rag"])}

    edges = build_graph_edges(store)

    triples = {(e.relation, e.source, e.target) for e in edges}
    assert ("HAS_PRIMARY_TOPIC", "item:1", "topic:agents") in triples
    assert ("HAS_TOPIC", "item:1", "topic:rag") in triples
    # The primary topic comes from `primary_topic`, not from `topics`: "rag" is not primary.
    assert ("HAS_PRIMARY_TOPIC", "item:1", "topic:rag") not in triples


def test_primary_topic_repeated_in_topics_does_not_produce_a_double_edge() -> None:
    # The enrichment lists the primary topic inside `topics` too, and repeats a secondary one.
    store = {"1": _item("1", primary="agents", topics=["agents", "rag", "rag"])}

    edges = build_graph_edges(store)

    pairs = [(e.source, e.target) for e in edges]
    assert sorted(pairs) == [("item:1", "topic:agents"), ("item:1", "topic:rag")]
    relation_of = {(e.source, e.target): e.relation for e in edges}
    assert relation_of[("item:1", "topic:agents")] == "HAS_PRIMARY_TOPIC"
    assert relation_of[("item:1", "topic:rag")] == "HAS_TOPIC"
