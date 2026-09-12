# tests/test_knowledge_search_hybrid.py
"""`search` with the vector channel wired in: `vector` and `hybrid` (Plan 03 §4, TDD 18, 21).

THROUGH THE PUBLIC SERVICE, over a REAL index built with a vector plane (rule 3). The fake
embedder maps a text to a point on the unit circle by its hash, so a query embedded as a given
chunk's text finds THAT chunk at cosine 1 — which is what lets a test choose, deterministically,
a chunk the vector channel finds and the lexical channel cannot.

What is pinned is the EXPLANATION (criterion §13.7): every match says which channels found it,
each rank is `None` exactly when its channel did not, and the response never names a strategy
whose vector channel did not run. No value of `RRF_K` or of the weights appears here.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.contracts import SearchFilters, SearchMatch, SearchResponse
from xbrain.knowledge.index_schema import db_path, open_index
from xbrain.knowledge.search_service import QueryContext, search
from xbrain.knowledge.vector_index import VectorSpec
from xbrain.models import Item, Topic, TopicPage
from xbrain.rubrics import save_vocab
from xbrain.store import save_store, save_topic_pages

FIXTURES = Path(__file__).parent / "fixtures"
QUERY = "Quillfeather"

SPEC = VectorSpec(
    model="intfloat/multilingual-e5-base",
    dimension=2,
    normalized=True,
    query_prefix="query: ",
    passage_prefix="passage: ",
)


def _vector(text: str) -> tuple[float, ...]:
    angle = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    angle *= 2 * math.pi
    return (math.cos(angle), math.sin(angle))


class QueryEmbedder:
    """Embeds every query as ONE chosen passage, and counts the calls."""

    def __init__(self, passage: str) -> None:
        self.passage = passage
        self.calls: list[str] = []

    def __call__(self, query: str) -> tuple[float, ...]:
        self.calls.append(query)
        return _vector(self.passage)


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


def _data(tmp_path: Path, corpus, *, with_plane: bool) -> Path:
    store, vocab, pages = corpus
    data = tmp_path / "data"
    save_store(store, data / "items.json")
    save_vocab(vocab, data / "vocab.yaml")
    save_topic_pages(pages, data / "topics.json")
    inputs = index_build.load_index_inputs(
        data / "items.json", data / "vocab.yaml", data / "topics.json"
    )
    vectors = (
        index_build.VectorBuild(spec=SPEC, embed=lambda texts: [_vector(text) for text in texts])
        if with_plane
        else None
    )
    index_build.build(data / "index", inputs, vectors=vectors)
    return data


def _context(data: Path, corpus, embed_query=None) -> QueryContext:
    store, vocab, pages = corpus
    return QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
        embed_query=embed_query,
    )


def _item_chunks(data: Path) -> dict[str, str]:
    """`{chunk_id: text}` of the ITEM-owned chunks, as the lexical plane serves them."""
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        rows = connection.execute("SELECT chunk_id, text FROM chunks WHERE owner_type = 'item'")
        return {str(row[0]): str(row[1]) for row in rows}
    finally:
        connection.close()


def _pick(data: Path, *, contains_query: bool) -> tuple[str, str]:
    """The first item chunk (by id) whose text does / does not carry the query word."""
    chunks = _item_chunks(data)
    for chunk_id in sorted(chunks):
        if (QUERY.lower() in chunks[chunk_id].lower()) is contains_query:
            return chunk_id, chunks[chunk_id]
    raise AssertionError(
        f"the fixture corpus has no item chunk with contains_query={contains_query}"
    )


def _matches(response: SearchResponse) -> list[SearchMatch]:
    return [match for result in response.results for match in result.matches]


def _match(response: SearchResponse, chunk_id: str) -> SearchMatch:
    found = [match for match in _matches(response) if match.chunk_id == chunk_id]
    assert found, f"{chunk_id} was not served; served: {[m.chunk_id for m in _matches(response)]}"
    return found[0]


def _assert_explained(match: SearchMatch) -> None:
    """Criterion §13.7: a rank is present exactly when its channel is named."""
    assert match.matched_by, match
    assert set(match.matched_by) <= {"lexical", "vector"}, match
    assert (match.lexical_rank is not None) is ("lexical" in match.matched_by), match
    assert (match.vector_rank is not None) is ("vector" in match.matched_by), match


def test_hybrid_runs_both_channels_and_explains_every_match(tmp_path: Path, corpus) -> None:
    """The response names `hybrid` only because the vector channel RAN, and every match says how."""
    data = _data(tmp_path, corpus, with_plane=True)
    target, text = _pick(data, contains_query=False)
    embedder = QueryEmbedder(text)

    response = search(QUERY, _context(data, corpus, embedder), strategy="hybrid", limit=50)

    assert response.strategy == "hybrid"
    assert embedder.calls == [QUERY]
    assert "no_embeddings" not in response.index.degraded
    assert not [flag for flag in response.index.degraded if flag.endswith("_not_implemented")]
    for match in _matches(response):
        _assert_explained(match)
    channels = {channel for match in _matches(response) for channel in match.matched_by}
    assert channels == {"lexical", "vector"}, "a test of fusion needs both channels to contribute"


def test_a_chunk_only_the_vector_channel_found_is_served_with_a_null_lexical_rank(
    tmp_path: Path, corpus
) -> None:
    """TDD 16 through the service: the surface, the locator and the explanation all survive."""
    data = _data(tmp_path, corpus, with_plane=True)
    target, text = _pick(data, contains_query=False)

    response = search(
        QUERY, _context(data, corpus, QueryEmbedder(text)), strategy="hybrid", limit=50
    )
    match = _match(response, target)

    assert match.matched_by == ("vector",)
    assert match.lexical_rank is None
    assert match.vector_rank is not None and match.vector_rank >= 1
    assert match.surface_type, "matched_surface must survive the fusion"
    assert match.locator is not None


def test_a_chunk_both_channels_found_carries_both_ranks(tmp_path: Path, corpus) -> None:
    """TDD 15 through the service: a lexical hit the vector channel also found names both."""
    data = _data(tmp_path, corpus, with_plane=True)
    target, text = _pick(data, contains_query=True)

    response = search(
        QUERY, _context(data, corpus, QueryEmbedder(text)), strategy="hybrid", limit=50
    )
    match = _match(response, target)

    assert match.matched_by == ("lexical", "vector")
    assert match.lexical_rank is not None
    assert match.vector_rank is not None


def test_vector_serves_only_what_the_vector_channel_found(tmp_path: Path, corpus) -> None:
    """`vector` is separable (spec §5.7): no lexical rank, and no lexical channel, on any match."""
    data = _data(tmp_path, corpus, with_plane=True)
    target, text = _pick(data, contains_query=False)

    response = search(
        QUERY, _context(data, corpus, QueryEmbedder(text)), strategy="vector", limit=50
    )

    assert response.strategy == "vector"
    assert _match(response, target).vector_rank is not None
    for match in _matches(response):
        assert match.matched_by == ("vector",), match
        assert match.lexical_rank is None, match


@pytest.mark.parametrize("with_plane, with_embedder", [(True, False), (False, True)])
def test_hybrid_whose_vector_channel_cannot_run_never_claims_it(
    tmp_path: Path, corpus, with_plane: bool, with_embedder: bool
) -> None:
    """TDD 18 / the line that is not crossed: no plane or no embedder ⇒ `lexical`, declared.

    And the embedder is not called over an index with no plane: a subprocess spent on a query
    that cannot be scored is a cost with nothing to show for it.
    """
    data = _data(tmp_path, corpus, with_plane=with_plane)
    embedder = QueryEmbedder(_pick(data, contains_query=False)[1]) if with_embedder else None

    response = search(QUERY, _context(data, corpus, embedder), strategy="hybrid")

    assert response.strategy == "lexical"
    assert response.index.degraded
    assert response.results, "lexical stays operational"
    assert not [match for match in _matches(response) if "vector" in match.matched_by]
    if embedder is not None:
        assert embedder.calls == []


def test_hybrid_with_filters_the_vector_plane_cannot_apply_is_lexical_and_says_so(
    tmp_path: Path, corpus
) -> None:
    """The plane has no filter columns: a filter applied AFTER scoring is not a filter.

    So the vector channel does not run, and the response declares why instead of serving
    unfiltered geometry under a filtered request.
    """
    data = _data(tmp_path, corpus, with_plane=True)
    embedder = QueryEmbedder(_pick(data, contains_query=False)[1])
    filters = SearchFilters(created_from=datetime(2000, 1, 1, tzinfo=timezone.utc))

    response = search(QUERY, _context(data, corpus, embedder), strategy="hybrid", filters=filters)

    assert response.strategy == "lexical"
    assert "vector_filters_unsupported" in response.index.degraded
    assert not [match for match in _matches(response) if "vector" in match.matched_by]
    assert embedder.calls == []


def test_hybrid_is_deterministic(tmp_path: Path, corpus) -> None:
    """Spec §3.7.8: the same query over the same state is the same envelope, byte for byte."""
    data = _data(tmp_path, corpus, with_plane=True)
    context = _context(data, corpus, QueryEmbedder(_pick(data, contains_query=False)[1]))

    first = search(QUERY, context, strategy="hybrid", limit=50)
    second = search(QUERY, context, strategy="hybrid", limit=50)

    assert first.model_dump_json() == second.model_dump_json()
