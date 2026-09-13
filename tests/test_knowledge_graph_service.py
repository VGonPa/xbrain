"""Plan 04.3 — `graph_expand` over a fixture of KNOWN population.

Never asserts corpus figures: every item, topic and count below is built here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.graph_build import ASSIGNMENT_METHOD, CO_OCCURRENCE_METHOD
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


def _path_to(response, node_id: str):
    (path,) = [p for p in response.paths if p.nodes[-1] == node_id]
    return path


def test_every_path_carries_node_types_relation_method_weight_and_support(
    tmp_path: Path,
) -> None:
    # From topic:a at one hop: items 1, 2, 3 (assigned `a` as primary) and topic:b, which shares
    # items {1, 2} of the union {1, 2, 3, 4} — Jaccard 0.5. The a–c pair shares ONE item, below
    # the build's default `min_shared_items=2`, so it is not an edge.
    context = _context(tmp_path)

    response = graph_expand(("topic:a",), context, max_hops=1)

    assert {p.nodes[-1] for p in response.paths} == {"item:1", "item:2", "item:3", "topic:b"}
    node_type = {n.node_id: n.node_type for n in response.nodes}
    for path in response.paths:
        assert path.nodes[0] == "topic:a"
        assert len(path.edges) == len(path.nodes) - 1
        assert all(node_id in node_type for node_id in path.nodes)

    co = _path_to(response, "topic:b")
    assert co.nodes == ("topic:a", "topic:b")
    (edge,) = co.edges
    assert (edge.source, edge.target, edge.relation) == ("topic:a", "topic:b", "CO_OCCURS_WITH")
    assert edge.method == CO_OCCURRENCE_METHOD
    assert edge.weight == 0.5
    assert edge.supporting_item_ids == ("1", "2")
    assert node_type["topic:b"] == "topic"

    assigned = _path_to(response, "item:3")
    (edge,) = assigned.edges
    assert (edge.source, edge.target, edge.relation) == ("item:3", "topic:a", "HAS_PRIMARY_TOPIC")
    assert edge.method == ASSIGNMENT_METHOD
    assert node_type["item:3"] == "item"
    assert node_type["topic:a"] == "topic"


def test_every_served_path_rests_on_item_ids_that_resolve_in_the_live_store(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)

    response = graph_expand(("topic:a",), context, max_hops=1)

    # The support of an edge, read off the edge itself: the listed ids of a co-occurrence, the
    # item endpoint of an assignment. Written here, not borrowed from the service under test.
    assert response.paths
    for path in response.paths:
        for edge in path.edges:
            if edge.relation == "CO_OCCURS_WITH":
                support = edge.supporting_item_ids
            else:
                support = (edge.source.removeprefix("item:"),)
            assert support, path
            assert all(item_id in context.store for item_id in support), path

    # Item 2 leaves the live store after the index was built. The persisted a→b edge still lists
    # it, and so does the assignment item:2 → topic:a: serving either would hand the consumer an
    # id that `get` cannot open. The expansion refuses and names the id and the repair.
    without_2 = {k: v for k, v in _KNOWN.items() if k != "2"}
    stale = QueryContext(**{**context.__dict__, "store": without_2})
    with pytest.raises(ValueError, match="xbrain index update") as refused:
        graph_expand(("topic:a",), stale, max_hops=1)
    assert "'2'" in str(refused.value)
