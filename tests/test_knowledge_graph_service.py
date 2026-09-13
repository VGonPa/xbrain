"""Plan 04.3 — `graph_expand` over a fixture of KNOWN population.

Never asserts corpus figures: every item, topic and count below is built here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from xbrain.knowledge import index_build
from xbrain.knowledge.graph_service import graph_expand
from xbrain.knowledge.search_service import QueryContext
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.rubrics import save_vocab
from xbrain.store import save_store

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


# items(a) = {1, 2, 3} · items(b) = {1, 2, 4} · items(c) = {2, 4}
_KNOWN = {
    "1": _item("1", primary="a", topics=["b"]),
    "2": _item("2", primary="a", topics=["b", "c"]),
    "3": _item("3", primary="a", topics=[]),
    "4": _item("4", primary="b", topics=["c"]),
}
_VOCAB = [Topic(slug=s, description=f"topic {s}") for s in ("a", "b", "c")]


def _context(tmp_path: Path, store: dict[str, Item] | None = None) -> QueryContext:
    """A built index over `store` (default `_KNOWN`) and the context a query door reads."""
    store = dict(_KNOWN) if store is None else store
    data = tmp_path / "data"
    save_store(store, data / "items.json")
    save_vocab(list(_VOCAB), data / "vocab.yaml")
    inputs = index_build.load_index_inputs(
        data / "items.json", data / "vocab.yaml", data / "topics.json"
    )
    index_build.build(data / "index", inputs)
    return QueryContext(
        store=store,
        vocab=tuple(_VOCAB),
        index_dir=data / "index",
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
    )


def test_expansion_distinguishes_the_primary_topic_from_a_secondary_one(tmp_path: Path) -> None:
    # Item 1 is `primary="a"` and carries "b" only as a secondary topic.
    context = _context(tmp_path)

    response = graph_expand(("item:1",), context, max_hops=1)

    relation_of = {
        (e.source, e.target): e.relation for e in response.edges if e.source == "item:1"
    }
    assert relation_of == {
        ("item:1", "topic:a"): "HAS_PRIMARY_TOPIC",
        ("item:1", "topic:b"): "HAS_TOPIC",
    }
