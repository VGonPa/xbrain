"""Plan 04.2 — graph build over a fixture of KNOWN population.

Never asserts corpus figures: every item, topic and count below is built here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xbrain import cli
from xbrain.config import load_config
from xbrain.knowledge import graph_build, index_build
from xbrain.knowledge.contracts import GraphEdge, GraphNode
from xbrain.knowledge.graph_build import (
    ASSIGNMENT_METHOD,
    CO_OCCURRENCE_METHOD,
    RELATION_ENDPOINTS,
    build_graph_edges,
)
from xbrain.knowledge.index_schema import db_path, open_index
from xbrain.models import Author, Enrichment, Item, Topic, TopicPage
from xbrain.rubrics import save_vocab
from xbrain.store import save_store, save_topic_pages

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


def _assignments(edges: list[GraphEdge]) -> list[GraphEdge]:
    return [e for e in edges if e.relation != "CO_OCCURS_WITH"]


def _co_occurrence(edges: list[GraphEdge]) -> dict[tuple[str, str], GraphEdge]:
    return {(e.source, e.target): e for e in edges if e.relation == "CO_OCCURS_WITH"}


# items(a) = {1, 2, 3} · items(b) = {1, 2, 4} · items(c) = {2, 4}
_KNOWN = {
    "1": _item("1", primary="a", topics=["b"]),
    "2": _item("2", primary="a", topics=["b", "c"]),
    "3": _item("3", primary="a", topics=[]),
    "4": _item("4", primary="b", topics=["c"]),
}


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

    edges = _assignments(build_graph_edges(store))

    pairs = [(e.source, e.target) for e in edges]
    assert sorted(pairs) == [("item:1", "topic:agents"), ("item:1", "topic:rag")]
    relation_of = {(e.source, e.target): e.relation for e in edges}
    assert relation_of[("item:1", "topic:agents")] == "HAS_PRIMARY_TOPIC"
    assert relation_of[("item:1", "topic:rag")] == "HAS_TOPIC"


@pytest.mark.parametrize(
    ("a", "b", "shared", "union"),
    [("a", "b", 2, 4), ("a", "c", 1, 4), ("b", "c", 2, 3)],
)
def test_co_occurrence_weight_is_jaccard_and_symmetric(
    a: str, b: str, shared: int, union: int
) -> None:
    co = _co_occurrence(build_graph_edges(_KNOWN))

    forward = co[(f"topic:{a}", f"topic:{b}")]
    backward = co[(f"topic:{b}", f"topic:{a}")]
    assert forward.weight == pytest.approx(shared / union)
    assert backward.weight == forward.weight
    assert forward.shared_items == backward.shared_items == shared


def test_a_large_topic_does_not_dominate_after_normalisation() -> None:
    # `big` sits on 10 items and overlaps `small` on 2 of them; `x` and `y` share their only 2.
    # Both pairs share exactly 2 items, so a raw count ties them — only normalisation separates.
    store = {str(n): _item(str(n), primary="big", topics=[]) for n in range(1, 11)}
    store["1"] = _item("1", primary="big", topics=["small"])
    store["2"] = _item("2", primary="big", topics=["small"])
    store["11"] = _item("11", primary="x", topics=["y"])
    store["12"] = _item("12", primary="x", topics=["y"])

    co = _co_occurrence(build_graph_edges(store))

    big_small = co[("topic:big", "topic:small")]
    x_y = co[("topic:x", "topic:y")]
    assert big_small.shared_items == x_y.shared_items == 2
    assert x_y.weight > big_small.weight
    assert max(co.values(), key=lambda e: e.weight).source in {"topic:x", "topic:y"}


def test_edges_carry_method_weights_support_and_input_fingerprints() -> None:
    edges = build_graph_edges(_KNOWN)
    co = _co_occurrence(edges)

    a_b = co[("topic:a", "topic:b")]
    assert a_b.method == CO_OCCURRENCE_METHOD
    assert (a_b.weight, a_b.shared_items) == (pytest.approx(0.5), 2)
    assert a_b.supporting_item_ids == ("1", "2")
    assert len(a_b.input_fingerprints) == 1
    assert {e.method for e in _assignments(edges)} == {ASSIGNMENT_METHOD}

    # Re-assigning a SUPPORTING item's topics keeps the support set {1, 2} but must move the
    # fingerprint: an id-only hash would certify an edge whose inputs changed underneath it.
    reassigned = {**_KNOWN, "1": _item("1", primary="a", topics=["b", "c"])}
    moved = _co_occurrence(build_graph_edges(reassigned))[("topic:a", "topic:b")]
    assert moved.supporting_item_ids == a_b.supporting_item_ids
    assert moved.input_fingerprints != a_b.input_fingerprints

    # Re-assigning an item OUTSIDE the support (item 3 is only on `a`) must not move it.
    outside = {**_KNOWN, "3": _item("3", primary="a", topics=["z"])}
    still = _co_occurrence(build_graph_edges(outside))[("topic:a", "topic:b")]
    assert still.input_fingerprints == a_b.input_fingerprints


def test_truncated_supporting_item_ids_still_declare_the_total() -> None:
    # `p` and `q` share all three items; the cap keeps only two ids.
    store = {n: _item(n, primary="p", topics=["q"]) for n in ("1", "2", "3")}

    edge = _co_occurrence(build_graph_edges(store, max_supporting_item_ids=2))[
        ("topic:p", "topic:q")
    ]

    assert edge.supporting_item_ids == ("1", "2")
    assert edge.shared_items == 3  # the TOTAL, not the length of the truncated tuple
    assert edge.weight == pytest.approx(1.0)

    # The fingerprint still covers the item the cap dropped: re-assigning item 3 moves it.
    moved = {**store, "3": _item("3", primary="p", topics=["q", "r"])}
    edge_moved = _co_occurrence(build_graph_edges(moved, max_supporting_item_ids=2))[
        ("topic:p", "topic:q")
    ]
    assert edge_moved.supporting_item_ids == ("1", "2")
    assert edge_moved.input_fingerprints != edge.input_fingerprints


def _unordered_pairs(edges: list[GraphEdge]) -> set[tuple[str, str]]:
    return {tuple(sorted((e.source, e.target))) for e in edges if e.relation == "CO_OCCURS_WITH"}


def test_min_shared_items_leaves_fewer_pairs_than_no_threshold() -> None:
    # p–q share 5 items; r–s and p–r share 1 each.
    store = {n: _item(n, primary="p", topics=["q"]) for n in ("1", "2", "3", "4", "5")}
    store["6"] = _item("6", primary="r", topics=["s"])
    store["7"] = _item("7", primary="p", topics=["r"])

    unfiltered = build_graph_edges(store)
    filtered = build_graph_edges(store, min_shared_items=5)

    assert _unordered_pairs(unfiltered) == {
        ("topic:p", "topic:q"),
        ("topic:r", "topic:s"),
        ("topic:p", "topic:r"),
    }
    assert _unordered_pairs(filtered) == {("topic:p", "topic:q")}
    assert len(_unordered_pairs(filtered)) < len(_unordered_pairs(unfiltered))
    # The threshold prunes co-occurrence only; every assignment survives it.
    assert _assignments(filtered) == _assignments(unfiltered)


def test_min_weight_drops_co_occurrence_below_the_jaccard_floor() -> None:
    # On `_KNOWN`: J(a,b) = 0.5, J(a,c) = 0.25, J(b,c) = 2/3.
    filtered = build_graph_edges(_KNOWN, min_weight=0.4)

    assert _unordered_pairs(filtered) == {("topic:a", "topic:b"), ("topic:b", "topic:c")}
    assert _assignments(filtered) == _assignments(build_graph_edges(_KNOWN))


def test_max_neighbors_per_node_keeps_each_topics_strongest_co_occurrences() -> None:
    # From `b`: c (2/3) > a (1/2). From `a`: b (1/2) > c (1/4). From `c`: b (2/3) > a (1/4).
    capped = build_graph_edges(_KNOWN, max_neighbors_per_node=1)

    neighbours = {(e.source, e.target) for e in capped if e.relation == "CO_OCCURS_WITH"}
    assert neighbours == {
        ("topic:a", "topic:b"),
        ("topic:b", "topic:c"),
        ("topic:c", "topic:b"),
    }
    assert _assignments(capped) == _assignments(build_graph_edges(_KNOWN))


def test_no_item_to_item_edge_exists_in_the_schema() -> None:
    # Read the relations off the CONTRACT'S schema, not off a list written here: a relation
    # added to `GraphEdge` without declared endpoints must turn this red.
    schema_relations = set(GraphEdge.model_json_schema()["properties"]["relation"]["enum"])
    node_types = set(GraphNode.model_json_schema()["properties"]["node_type"]["enum"])

    assert set(RELATION_ENDPOINTS) == schema_relations
    assert all({s, t} <= node_types for s, t in RELATION_ENDPOINTS.values())
    assert ("item", "item") not in RELATION_ENDPOINTS.values()

    # And what the builder emits obeys the declared endpoints, on a store where every item
    # shares topics with every other — the shape that would tempt an item–item edge.
    for edge in build_graph_edges(_KNOWN):
        endpoints = (edge.source.split(":", 1)[0], edge.target.split(":", 1)[0])
        assert endpoints == RELATION_ENDPOINTS[edge.relation]


_VOCAB = [Topic(slug=s, description=f"topic {s}") for s in ("a", "b", "c")]


def _persisted(tmp_path: Path) -> Path:
    """A data/ holding `_KNOWN` and its vocabulary, written through the store's own writers."""
    data = tmp_path / "data"
    save_store(dict(_KNOWN), data / "items.json")
    save_vocab(list(_VOCAB), data / "vocab.yaml")
    return data


def _inputs(data: Path) -> index_build.IndexInputs:
    return index_build.load_index_inputs(
        data / "items.json", data / "vocab.yaml", data / "topics.json"
    )


def _graph_rows(data: Path) -> list[tuple]:
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        return [
            tuple(row)
            for row in connection.execute(
                "SELECT source, target, relation FROM graph_edges ORDER BY source, target, relation"
            )
        ]
    finally:
        connection.close()


def test_graph_build_does_not_mutate_items_json(tmp_path: Path) -> None:
    data = _persisted(tmp_path)
    before = hashlib.sha256((data / "items.json").read_bytes()).hexdigest()

    index_build.build(data / "index", _inputs(data))

    # The graph WAS built — without this, an unchanged hash would also hold for a build that
    # never wrote a single edge, and the assertion below would prove nothing.
    rows = _graph_rows(data)
    assert ("topic:a", "topic:b", "CO_OCCURS_WITH") in rows
    assert ("item:1", "topic:a", "HAS_PRIMARY_TOPIC") in rows
    assert hashlib.sha256((data / "items.json").read_bytes()).hexdigest() == before


def _a_b_edge(data: Path) -> tuple[float, list[str]]:
    """`(weight, input_fingerprints)` of the stored `topic:a → topic:b` co-occurrence edge."""
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        weight, fingerprints = connection.execute(
            "SELECT weight, input_fingerprints_json FROM graph_edges "
            "WHERE source = 'topic:a' AND target = 'topic:b' AND relation = 'CO_OCCURS_WITH'"
        ).fetchone()
    finally:
        connection.close()
    return weight, json.loads(fingerprints)


def _update(data: Path) -> None:
    index_build.update(data / "index", _inputs(data))


def test_update_recomputes_the_graph_when_topics_or_vocabulary_change(tmp_path: Path) -> None:
    data = _persisted(tmp_path)
    index_build.build(data / "index", _inputs(data))
    built = _a_b_edge(data)

    # Control: an update with nothing changed leaves the edge exactly as built.
    _update(data)
    assert _a_b_edge(data) == built

    # The vocabulary moves (a description edit): the edge's input fingerprints must move.
    save_vocab([Topic(slug="a", description="a rewritten"), *_VOCAB[1:]], data / "vocab.yaml")
    _update(data)
    after_vocab = _a_b_edge(data)
    assert after_vocab[1] != built[1]

    # The topic pages move (topics.json gains an overview): they must move again.
    page = TopicPage(slug="a", overview="o", synthesized_at=_T, post_count_at_synth=3)
    save_topic_pages({"a": page}, data / "topics.json")
    _update(data)
    after_topics = _a_b_edge(data)
    assert after_topics[1] != after_vocab[1]

    # And an item re-assignment is recomputed, not left stale: item 3 joins `b`, so
    # items(a) = {1, 2, 3} and items(b) = {1, 2, 3, 4} → Jaccard 3/4 where it was 2/4.
    assert built[0] == pytest.approx(0.5)
    save_store({**_KNOWN, "3": _item("3", primary="a", topics=["b"])}, data / "items.json")
    _update(data)
    assert _a_b_edge(data)[0] == pytest.approx(0.75)


def test_each_graph_config_field_reaches_graph_build(tmp_path: Path, monkeypatch) -> None:
    # Three DISTINCT values, none a default, so a field wired to another field's slot goes red.
    data = _persisted(tmp_path)
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "vault"\noutput_subdir = "x-knowledge"\ndata_dir = "data"\n'
        '[x]\nhandle = "u"\n'
        "[index]\n"
        "graph_min_shared_items = 4\n"
        "graph_min_weight = 0.3\n"
        "graph_max_neighbors_per_node = 7\n",
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def spy(store, **kwargs):
        seen.update(kwargs)
        return graph_build.build_graph_edges(store, **kwargs)

    monkeypatch.setattr(index_build, "build_graph_edges", spy)

    options = cli._index_options(load_config(tmp_path))
    index_build.build(data / "index", _inputs(data), options=options)

    assert seen.get("min_shared_items") == 4
    assert seen.get("min_weight") == pytest.approx(0.3)
    assert seen.get("max_neighbors_per_node") == 7
