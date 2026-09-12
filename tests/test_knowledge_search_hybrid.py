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
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build, search_service
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


# --------------------------------------------------------------------------------------------
# Regressions for the three defects reproduced on PR #184
# --------------------------------------------------------------------------------------------


def _walk_pages(context: QueryContext, strategy: str) -> list[str]:
    """Every item id a `limit=1` cursor walk serves, in order."""
    served: list[str] = []
    cursor = None
    for _ in range(500):
        response = search(QUERY, context, strategy=strategy, limit=1, cursor=cursor)
        served += [result.item_id for result in response.results]
        cursor = response.cursor
        if cursor is None:
            return served
    raise AssertionError("the cursor walk did not terminate")


def test_hybrid_pages_of_one_reassemble_the_unpaged_ranking(tmp_path: Path, corpus) -> None:
    """Defect 1: at `limit=1` the walk served one item twice and never served another.

    Each page had materialised its OWN window, and RRF over a deeper window re-orders the head,
    so the pages were slices of different rankings. Pages must be disjoint and reassemble one.
    """
    data = _data(tmp_path, corpus, with_plane=True)
    context = _context(data, corpus, QueryEmbedder(_pick(data, contains_query=False)[1]))

    whole = search(QUERY, context, strategy="hybrid", limit=500)
    walked = _walk_pages(context, "hybrid")

    assert not whole.truncated
    assert len(whole.results) > 2, "paging a ranking of two pages or fewer proves nothing"
    assert len(walked) == len(set(walked)), walked
    assert walked == [result.item_id for result in whole.results]


MANY = "Marrowgate"


def _many_chunk_corpus(corpus):  # noqa: ANN001, ANN202 - the fixture tuple, passed through
    """The fixture with `k08`'s transcript cut into many windows, every one naming `Marrowgate`.

    The corpus `test_a_long_transcript_yields_one_result_with_at_most_three_matches` builds, so
    ONE item holds more lexical matches than any per-item cap — the only shape in which the two
    channels' per-item heads can disagree about a chunk both of them found.
    """
    store, vocab, pages = corpus
    item = store["k08"]
    assert item.content is not None
    sources = list(item.content.sources)
    long_text = " ".join(f"{MANY} segment {n} of the talk." for n in range(400))
    sources[0] = sources[0].model_copy(update={"text": long_text})
    content = item.content.model_copy(update={"sources": sources})
    return {**store, "k08": item.model_copy(update={"content": content})}, vocab, pages


def _all_chunks(data: Path) -> dict[str, tuple[str, str, str]]:
    """`{chunk_id: (owner_type, owner_id, text)}` for every chunk the lexical plane holds."""
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        rows = connection.execute("SELECT chunk_id, owner_type, owner_id, text FROM chunks")
        return {str(row[0]): (str(row[1]), str(row[2]), str(row[3])) for row in rows}
    finally:
        connection.close()


def test_a_chunk_both_channels_found_is_explained_by_both(tmp_path: Path, corpus) -> None:
    """Defect 2: a chunk both channels found was served naming only one of them.

    The loss came from asking each channel for only the item's best `cap` chunks: a chunk
    inside one channel's cap and outside the other's was explained by that channel alone. So
    the test makes the two heads DIFFER on purpose: `head` is bm25's best chunk of `k08`
    (read through the public `lexical` strategy) and `target` is another chunk of `k08` that
    also names the word, embedded as the query so the vector channel ranks it first. Both
    channels found both chunks — premises asserted, not assumed: every served chunk names the
    word, and the corpus is smaller than the vector window — so whichever one is served must
    say `("lexical", "vector")`. On the defective code the served chunk named one channel.
    """
    many = _many_chunk_corpus(corpus)
    data = _data(tmp_path, many, with_plane=True)
    chunks = _all_chunks(data)
    matching = {
        chunk_id: text
        for chunk_id, (owner_type, owner_id, text) in chunks.items()
        if (owner_type, owner_id) == ("item", "k08") and MANY.lower() in text.lower()
    }
    assert len(matching) > 3, "premise: more lexical matches in one item than the default cap"
    assert len(chunks) < search_service.FUSED_CHUNK_WINDOW, "premise: the window holds them all"

    lexical = search(MANY, replace(_context(data, many), max_matches_per_item=1))
    head = next(r for r in lexical.results if r.item_id == "k08").matches[0].chunk_id
    target = min(chunk_id for chunk_id in matching if chunk_id != head)
    context = replace(_context(data, many, QueryEmbedder(matching[target])), max_matches_per_item=1)

    response = search(MANY, context, strategy="hybrid", limit=500)
    k08 = next(result for result in response.results if result.item_id == "k08")

    assert k08.matches, "k08 must be served with evidence"
    for match in k08.matches:
        assert match.chunk_id in matching, "premise: the served chunk names the word"
        assert match.matched_by == ("lexical", "vector"), match
        assert match.lexical_rank is not None and match.vector_rank is not None, match


def _edit_summary(item: Item, text: str) -> Item:
    """The change `enrich` makes: a new summary and a new `enriched_at`."""
    assert item.enriched is not None
    return item.model_copy(
        update={
            "enriched": item.enriched.model_copy(
                update={
                    "summary": text,
                    "enriched_at": item.enriched.enriched_at + timedelta(hours=1),
                }
            )
        }
    )


def _summary_chunks(data: Path) -> dict[str, tuple[str, str]]:
    """`{item_id: (chunk_id, text)}` of each item's first summary chunk."""
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        rows = connection.execute(
            "SELECT owner_id, chunk_id, text FROM chunks WHERE owner_type = 'item' "
            "AND surface_type = 'summary' AND chunk_index = 0"
        )
        return {str(row[0]): (str(row[1]), str(row[2])) for row in rows}
    finally:
        connection.close()


def test_a_vector_left_stale_by_update_is_not_served_and_the_response_says_so(
    tmp_path: Path, corpus
) -> None:
    """Defect 3: after `index update` without re-embedding, the old geometry was served.

    The summary is rewritten and the index updated WITHOUT an embedder, so the chunk id
    survives (it is positional) and its row still holds the vector of the OLD text. A query
    embedded as that old text found it at cosine 1 and served it with a `vector_rank` and an
    empty `degraded`.
    """
    store, vocab, pages = corpus
    data = _data(tmp_path, corpus, with_plane=True)
    before = _summary_chunks(data)
    item_id = next(
        owner for owner in sorted(before) if QUERY.lower() not in before[owner][1].lower()
    )
    chunk_id, old_text = before[item_id]
    edited = {**store, item_id: _edit_summary(store[item_id], "Resumen reescrito tras enrich.")}
    save_store(edited, data / "items.json")
    inputs = index_build.load_index_inputs(
        data / "items.json", data / "vocab.yaml", data / "topics.json"
    )
    index_build.update(data / "index", inputs)
    assert _summary_chunks(data)[item_id][0] == chunk_id, "premise: the id survived the edit"

    context = _context(data, (edited, vocab, pages), QueryEmbedder(old_text))
    response = search(QUERY, context, strategy="hybrid", limit=500)

    assert search_service.VECTOR_PLANE_BEHIND in response.index.degraded
    assert not [
        match
        for match in _matches(response)
        if match.chunk_id == chunk_id and "vector" in match.matched_by
    ]
