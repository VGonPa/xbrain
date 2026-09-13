"""Plan 04.3 — `graph_expand` over a fixture of KNOWN population.

Never asserts corpus figures: every item, topic and count below is built here.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xbrain import i18n
from xbrain.i18n import SUPPORTED_LANGUAGES, strings_for
from xbrain.knowledge import graph_service, index_build
from xbrain.knowledge.graph_build import (
    ASSIGNMENT_METHOD,
    CO_OCCURRENCE_METHOD,
    DEFAULT_GRAPH_MAX_NEIGHBORS_PER_NODE,
)
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


def test_max_hops_one_returns_no_node_two_hops_away(tmp_path: Path) -> None:
    # item:3 carries only `a`. One hop: topic:a. Two hops: items 1 and 2 (also assigned `a`)
    # and topic:b (a→b co-occurs). Asserting only the one-hop half would pass on an expansion
    # that cannot walk at all, so the two-hop half proves the bound is a bound.
    context = _context(tmp_path)

    two = graph_expand(("item:3",), context, max_hops=2)
    one = graph_expand(("item:3",), context, max_hops=1)

    assert {n.node_id for n in two.nodes} == {"item:3", "topic:a", "item:1", "item:2", "topic:b"}
    assert _path_to(two, "topic:b").nodes == ("item:3", "topic:a", "topic:b")
    assert [e.relation for e in _path_to(two, "topic:b").edges] == [
        "HAS_PRIMARY_TOPIC",
        "CO_OCCURS_WITH",
    ]

    assert {n.node_id for n in one.nodes} == {"item:3", "topic:a"}
    assert all(len(p.nodes) <= 2 for p in one.paths)
    assert all({e.source, e.target} <= {"item:3", "topic:a"} for e in one.edges)


def test_a_giant_topic_is_bounded_to_max_neighbors_per_node(tmp_path: Path) -> None:
    # Fifteen more items carry `a` as primary: topic:a now has 18 item neighbours plus topic:b.
    # `graph_build` caps co-occurrence per topic, but NOT assignments — a topic's item list is
    # bounded only by the corpus, so the expansion is where the bound has to hold.
    giant = {f"g{n:02d}": _item(f"g{n:02d}", primary="a", topics=[]) for n in range(1, 16)}
    context = _context(tmp_path, {**_KNOWN, **giant})

    def neighbours(response) -> list[str]:
        return [p.nodes[-1] for p in response.paths if p.nodes[:-1] == ("topic:a",)]

    unbounded = graph_expand(("topic:a",), context, max_hops=1, max_neighbors_per_node=100)
    capped = graph_expand(("topic:a",), context, max_hops=1, max_neighbors_per_node=5)
    default = graph_expand(("topic:a",), context, max_hops=1)

    # The population is what makes the cap the thing that cuts: 19 exist, 5 are served.
    assert len(neighbours(unbounded)) == 18 + 1
    assert len(neighbours(capped)) == 5
    assert len(neighbours(default)) == DEFAULT_GRAPH_MAX_NEIGHBORS_PER_NODE
    # Strength ranks first, so the topic neighbour is not crowded out by the item list.
    assert "topic:b" in neighbours(capped)
    assert {n.node_id for n in capped.nodes} == {"topic:a", *neighbours(capped)}


def test_the_serialized_expansion_carries_semantics_and_disclaimer_key(tmp_path: Path) -> None:
    # Read off the JSON a consumer receives, not off the model's attributes: a field excluded at
    # serialization would still be readable on the object and absent from every consumer.
    context = _context(tmp_path)

    payload = json.loads(graph_expand(("topic:a",), context, max_hops=1).model_dump_json())

    assert payload["semantics"] == "co_occurrence_in_corpus"
    assert payload["disclaimer_key"] == "graph_edge_is_corpus_not_world"
    # The key names a sentence that exists, non-empty, in every supported language.
    for language in SUPPORTED_LANGUAGES:
        assert getattr(strings_for(language), payload["disclaimer_key"]).strip()


def test_the_disclaimer_sentence_comes_from_i18n_strings_in_both_languages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    response = graph_expand(("topic:a",), context, max_hops=1)

    # The real sentences are two, one per language — a single inline English sentence would
    # answer both calls with the same text.
    english = graph_service.disclaimer(response, "English")
    spanish = graph_service.disclaimer(response, "Spanish")
    assert english == strings_for("English").graph_edge_is_corpus_not_world
    assert spanish == strings_for("Spanish").graph_edge_is_corpus_not_world
    assert english != spanish

    # Replace the table the sentence lives in. An inline copy — equal by value to the real
    # sentence — cannot follow the replacement; only a read through `i18n.Strings` can.
    for language in ("English", "Spanish"):
        monkeypatch.setitem(
            i18n._STRINGS,
            language,
            dataclasses.replace(
                strings_for(language), graph_edge_is_corpus_not_world=f"sentinel {language}"
            ),
        )
    assert graph_service.disclaimer(response, "English") == "sentinel English"
    assert graph_service.disclaimer(response, "Spanish") == "sentinel Spanish"
