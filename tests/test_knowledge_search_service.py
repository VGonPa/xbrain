# tests/test_knowledge_search_service.py
"""`search` end to end (Plan 02 §4, steps 10, 10b, 11, 13, 14, 15, 18, 19, 20, 20b, 29, 30).

RUN THROUGH THE PUBLIC SERVICE, not against its internals. CLAUDE.md rule 3: the judge must
EXECUTE. Every assertion here goes through `search_service.search`, which is the same
function the CLI adapter and Plan 04's MCP tool will call — so a defect that only shows up
once the pieces are wired is visible here rather than in the second adapter.

The fixtures build a REAL index on disk with the real writer and then query it read-only,
because the failures this file is about — a corrupt row excluded, an index behind the store,
a revoked verdict — are all properties of the round trip and none of them can be staged in
memory.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build, search_service
from xbrain.knowledge.contracts import SearchFilters, SearchResponse
from xbrain.knowledge.index_schema import (
    IndexIncompatibleError,
    IndexMissingError,
    db_path,
    manifest_path,
    open_index,
)
from xbrain.knowledge.search_service import QueryContext, search
from xbrain.knowledge.surfaces import knowledge_item
from xbrain.models import Item, Topic, TopicPage, VerificationVerdict

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


def _write_store(path: Path, store: dict[str, Item]) -> None:
    path.write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )


@pytest.fixture()
def context(tmp_path: Path, corpus) -> QueryContext:
    """A built index plus the live store — the two halves every query needs."""
    store, vocab, pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    _write_store(data / "items.json", store)
    index_build.build(data / "index", store, vocab, pages, data / "items.json")
    return QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_search_returns_a_valid_frozen_response(context: QueryContext) -> None:
    """Acceptance 5: a `SearchResponse` valid against the Plan 01 schema, with the explanation.

    `extra="forbid"` on the frozen models means a field the service invents fails
    construction, so building the response IS the schema check — which is why this asserts
    the round trip through `model_validate` rather than eyeballing keys.
    """
    response = search("Quillfeather", context)
    assert isinstance(response, SearchResponse)
    assert SearchResponse.model_validate(response.model_dump()) == response
    assert response.schema_version == "1" and response.strategy == "lexical"
    assert response.results
    match = response.results[0].matches[0]
    assert match.matched_by == ("lexical",)
    assert match.lexical_rank == 1
    assert match.locator.char_start is not None
    assert response.results[0].available_surfaces


def test_the_response_declares_no_embeddings(context: QueryContext) -> None:
    """Spec §9.3 / Plan 02 §16: without a vector backend the response DECLARES the degradation.

    Plan 02 is literally that state, and saying so is what stops a consumer reading a lexical
    answer as a hybrid one. Seen red by returning an empty `degraded`.
    """
    response = search("Quillfeather", context)
    assert "no_embeddings" in response.index.degraded


def test_a_strategy_with_no_backend_is_answered_lexically_and_says_so(
    context: QueryContext,
) -> None:
    """F-2 / spec §9.3: *lexical sigue operativo y el response declara estrategia degradada;
    no finge resultados vectoriales.*

    `SearchResponse.strategy` used to be the strategy REQUESTED, echoed back without a check:
    `search(..., strategy="hybrid")` returned `strategy: "hybrid"` over results produced
    entirely by bm25. `degraded: ["no_embeddings"]` was there, but the field that NAMES the
    retriever is the one a consumer reads to know what ran, and it said the wrong name — the
    "finge resultados vectoriales" the spec forbids, in the one field that could commit it.

    The response now names the strategy that EXECUTED, and the degradation names the one that
    could not, so both halves of the spec bullet are readable from the JSON alone. This is
    the shape Plan 04's MCP adapter consumes; the CLI never passes `--strategy` today, which
    is why the library API is where it had to be fixed.

    Seen red before the fix: `strategy` came back `'vector'` and `'hybrid'`, and no
    `*_not_implemented` flag existed at all.
    """
    for requested in ("vector", "hybrid", "hybrid_graph"):
        response = search("Quillfeather", context, strategy=requested)
        assert response.strategy == "lexical", requested
        assert f"{requested}_not_implemented" in response.index.degraded, requested
        assert response.results, "lexical stays operational — the spec's first clause"


def test_the_implemented_strategy_is_declared_without_a_degradation(
    context: QueryContext,
) -> None:
    """The other side of the same predicate, so the flag is not raised unconditionally.

    Without this, a `degraded` that always carried a `*_not_implemented` marker would satisfy
    the test above for the wrong reason (rule 1).
    """
    response = search("Quillfeather", context, strategy="lexical")
    assert response.strategy == "lexical"
    assert not [flag for flag in response.index.degraded if flag.endswith("_not_implemented")]


def test_a_strategy_that_is_not_in_the_contract_is_refused_before_any_work(
    context: QueryContext,
) -> None:
    """A DECLARED strategy with no backend degrades; a strategy that does not exist refuses.

    The two are different failures and the spec treats them differently: §9.3 asks for
    degradation when the embeddings backend is missing, and for a *stable validation error*
    when the arguments are invalid. Silently answering `strategy="banana"` with lexical
    results would turn a typo into a measurement.

    THE INDEX DIRECTORY IS DELIBERATELY GONE, and that is what makes this assertion able to
    fail. Without it the test passes today for the wrong reason (rule 1): `SearchResponse` is
    a pydantic model over a `Literal`, so an unknown strategy raises a `ValidationError` —
    itself a `ValueError` — at the very END, after the whole query has run. Pointing at a
    missing index separates the two: refusing FIRST raises our error, refusing last raises
    `IndexMissingError`, and only one of those can happen.

    Seen red before the fix: `IndexMissingError: No hay índice en ...`.
    """
    nowhere = replace(context, index_dir=context.index_dir.parent / "does-not-exist")
    with pytest.raises(ValueError, match="Estrategia desconocida") as caught:
        search("Quillfeather", nowhere, strategy="banana")
    assert "lexical" in str(caught.value), "names what would have been valid"


def test_search_is_deterministic(context: QueryContext) -> None:
    """Spec §3.7.8: the same query over the same index answers identically, twice."""
    assert search("agents", context).model_dump() == search("agents", context).model_dump()


# ---------------------------------------------------------------------------
# 13, 14 — validation refuses rather than guessing
# ---------------------------------------------------------------------------


def test_an_empty_query_is_a_validation_error(context: QueryContext) -> None:
    """Step 14 / spec §9.3. An empty result set would claim something about the corpus."""
    with pytest.raises(ValueError, match="vacía"):
        search("   ", context)


def test_a_non_positive_limit_is_a_validation_error(context: QueryContext) -> None:
    """Plan 02 §11: `--limit 0` or negative is a validation error, not an empty answer."""
    with pytest.raises(ValueError, match="limit"):
        search("agents", context, limit=0)


def test_an_unknown_topic_lists_the_valid_ones(context: QueryContext) -> None:
    """Step 13 / spec §3.7.5: *topics y filtros no se inventan desde el texto del query.*

    Answering a typo with zero results IS that invention, silently — and it is
    indistinguishable from a topic that genuinely has no matches. Seen red by accepting the
    slug: the query returns nothing and looks like a fact about the corpus.
    """
    with pytest.raises(ValueError, match="agent-evaluation"):
        search("agents", context, filters=SearchFilters(topics=("no-such-topic",)))


# ---------------------------------------------------------------------------
# 15 — grouping (spec §5.4)
# ---------------------------------------------------------------------------


def test_a_long_transcript_yields_one_result_with_at_most_three_matches(
    tmp_path: Path, corpus
) -> None:
    """Step 15 / acceptance 7: ten adjacent windows must not take ten of the top ten.

    Built from a transcript long enough to produce more than three windows, so the cap is
    exercised rather than asserted against a body that could never reach it (rule 2). Seen
    red by removing the per-item cap: the same item returns every window it has.
    """
    store, vocab, pages = corpus
    item = store["k08"]
    long_text = " ".join(f"Marrowgate segment {n} of the talk." for n in range(400))
    sources = list(item.content.sources)
    sources[0] = sources[0].model_copy(update={"text": long_text})
    store = dict(store)
    store["k08"] = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )
    data = tmp_path / "data"
    data.mkdir()
    _write_store(data / "items.json", store)
    index_build.build(data / "index", store, vocab, pages, data / "items.json")
    context = QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )

    windows = _rows(data, "SELECT COUNT(*) FROM chunks WHERE surface_type = 'video_transcript'")
    assert windows > 3, "the fixture must produce more windows than the cap, or nothing is tested"

    response = search("Marrowgate", context)
    k08 = [r for r in response.results if r.item_id == "k08"]
    assert len(k08) == 1, "one item, one result"
    assert len(k08[0].matches) <= 3


# ---------------------------------------------------------------------------
# 18, 19 — the derived-source rule (spec §3.5)
# ---------------------------------------------------------------------------


def test_a_summary_match_points_at_the_underlying_article(context: QueryContext) -> None:
    """Step 18 / acceptance 8: a match on a derived surface leads to the SOURCE.

    `k03` has an `external_article`, so a summary match on it must name that article as what
    to ask `get` for. Seen red by emptying `verify_with`.
    """
    item = context.store["k03"]
    term = item.enriched.summary.split()[0]
    response = search(term, context)
    result = next(r for r in response.results if r.item_id == "k03")
    assert "external_article" in result.verify_with


def test_a_derived_match_with_no_primary_source_says_so(tmp_path: Path, corpus) -> None:
    """Step 19 / acceptance 8: `verify_with: []` plus the `no_underlying_source` predicate.

    Spec §3.5: *si una superficie derivada no permite llegar a material sustentante, el
    resultado debe decirlo.*

    HOW OFTEN THIS HAPPENS ON THE REAL CORPUS: never (F-5). Measured 2026-09-01 over
    `data/items.json` (2,404 items, sha256 `f76341a3…`), items with NO evidence-class
    surface: **0 of 2,404**. Every item has a `post` — the tweet text — and a `post` is a
    `primary_source`. The docstring used to say *961 of 2,404 (40 %)*, which is a different
    population: items with no `content` BLOCK (960 of 2,404 on the same measurement, F-14).
    Having no `content` and having no primary surface are not the same fact, and the second
    one is empty. Rule 2, inside a docstring, in the repository that wrote rule 2.

    THE BRANCH IS STILL WORTH GUARDING because it is a CONTRACT guarantee, not a frequency
    claim: `verify_with == ()` is the structural form of `no_underlying_source`, so if the
    emitter ever stops emitting `post` for some item shape, this is what says so instead of
    silently returning a derived match with nowhere to check it.

    The item here is stripped of its post as well as its content, because a post IS a primary
    surface: leaving it would make `verify_with` non-empty for a perfectly good reason and
    the test would pass without testing the branch (rule 1). That stripping is asserted to be
    load-bearing below, so the constructed item cannot quietly stop being the reason.
    """
    store, vocab, pages = corpus
    store = dict(store)
    naked = store["k02"].model_copy(update={"text": "", "content": None, "media": []})
    store["k02"] = naked.model_copy(
        update={"enriched": naked.enriched.model_copy(update={"summary": "Snapdragon sin fuente"})}
    )
    data = tmp_path / "data"
    data.mkdir()
    _write_store(data / "items.json", store)
    index_build.build(data / "index", store, vocab, pages, data / "items.json")
    context = QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )

    response = search("Snapdragon", context)
    result = next(r for r in response.results if r.item_id == "k02")
    assert result.matches and all(m.derived for m in result.matches)
    assert result.verify_with == ()
    assert search_service.no_underlying_source(result) is True

    # And the stripping is what produced it: with its post back, the very same item has a
    # primary surface to verify against. This is the corpus-wide case (0 of 2,404 items lack
    # one), so without this line the test could not tell "the branch fired" from "the emitter
    # stopped emitting `post` for everything".
    dressed = knowledge_item(store["k02"].model_copy(update={"text": "Snapdragon original"}))
    assert "post" in dressed.available_surfaces


def test_a_primary_match_verifies_against_itself(context: QueryContext) -> None:
    """The complement: a match on a post or an article names that surface, not a detour."""
    response = search("Quillfeather", context)
    result = next(r for r in response.results if r.matches[0].derived is False)
    assert result.matches[0].surface_type in result.verify_with
    assert search_service.no_underlying_source(result) is False


# ---------------------------------------------------------------------------
# 20 — a topic match reaches real items
# ---------------------------------------------------------------------------


def test_a_topic_note_match_returns_the_topics_supporting_items(context: QueryContext) -> None:
    """Step 20: the consumer can jump to real items with `get`.

    `SearchResult` is item-shaped — it requires an `item_id`, a `url`, an `author` and a
    `created_at`, none of which a `topic_note` has — so the match is attached to the topic's
    supporting items. That is stronger than a list of ids: the items arrive hydrated.

    Seen red by dropping the topic expansion: the note matches, and the answer is empty.
    """
    slug = context.vocab[0].slug
    note = context.topic_pages[slug].notes[0]
    term = max(note.split(), key=len)

    response = search(term, context)

    topic_results = [
        r
        for r in response.results
        if any(m.surface_type in {"topic_note", "topic_overview"} for m in r.matches)
    ]
    assert topic_results, f"no result carried a topic surface for {term!r}"
    expected = search_service.supporting_item_ids(slug, context)
    assert {r.item_id for r in topic_results} <= set(expected)
    assert all(r.item_id in context.store for r in topic_results)


# ---------------------------------------------------------------------------
# 10 — a corrupt chunk fails closed
# ---------------------------------------------------------------------------


def test_a_chunk_with_a_manipulated_fingerprint_is_excluded_and_counted(
    context: QueryContext,
) -> None:
    """Step 10 / acceptance 4 / invariant 6 of spec §3.7.

    The row is edited BY HAND, which is the situation the check exists for: a row written by
    another chunker version, or a database someone poked at. It is excluded and COUNTED —
    excluded silently would make the corpus look smaller than it is with nothing saying why,
    and repairing it is forbidden (spec §5.6).

    Seen red by returning the row anyway: it reappears in the results and the counter is 0.
    """
    connection = sqlite3.connect(db_path(context.index_dir))
    victim = connection.execute(
        "SELECT chunk_id FROM chunks WHERE text LIKE '%Quillfeather%' LIMIT 1"
    ).fetchone()[0]
    connection.execute("UPDATE chunks SET fingerprint = ? WHERE chunk_id = ?", ("0" * 64, victim))
    connection.commit()
    connection.close()

    response = search("Quillfeather", context)

    assert response.index.corrupt_chunks_excluded >= 1
    returned = {m.chunk_id for r in response.results for m in r.matches}
    assert victim not in returned


def test_search_never_repairs_the_index(context: QueryContext) -> None:
    """Step 11 / spec §5.6: the query connection is read-only, so a repair CANNOT happen.

    Asserted by content rather than by intent: the corrupted row is still corrupted after the
    query. A service that "does not write" is a claim about the code; a row that is unchanged
    is a fact about the database.
    """
    connection = sqlite3.connect(db_path(context.index_dir))
    victim = connection.execute("SELECT chunk_id FROM chunks LIMIT 1").fetchone()[0]
    connection.execute("UPDATE chunks SET fingerprint = ? WHERE chunk_id = ?", ("0" * 64, victim))
    connection.commit()
    connection.close()

    search("Quillfeather", context)

    connection = sqlite3.connect(db_path(context.index_dir))
    after = connection.execute(
        "SELECT fingerprint FROM chunks WHERE chunk_id = ?", (victim,)
    ).fetchone()[0]
    connection.close()
    assert after == "0" * 64, "search repaired a row it was supposed to exclude"


def test_the_query_connection_refuses_a_write(context: QueryContext) -> None:
    """And the mechanism behind it, pinned directly."""
    from xbrain.knowledge.index_store import open_for_query

    index = open_for_query(context.index_dir, context.items_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            index.lexical.connection.execute("DELETE FROM chunks")
    finally:
        index.close()


# ---------------------------------------------------------------------------
# 10b — the failure that will actually happen (B3)
# ---------------------------------------------------------------------------


def test_editing_the_store_without_reindexing_declares_the_index_behind(
    context: QueryContext,
) -> None:
    """Step 10b / acceptance 4b: the failure indexing-by-decision makes inevitable.

    Spec §1.1 wants a consumer to *detect that an index is incomplete or out of date INSTEAD
    of receiving stale evidence*. The internal fingerprint check cannot see this — every row
    is consistent with itself — so the cheap store signal is what fires. The response is
    still ANSWERED, because spec §9.3 calls this possibly-stale evidence rather than an
    error; what it must not do is stay quiet.

    Seen red by comparing only the chunk fingerprint against itself: nothing changes and the
    answer looks fresh.
    """
    assert "index_behind_store" not in search("agents", context).index.degraded

    store = dict(context.store)
    victim = store["k02"]
    store["k02"] = victim.model_copy(
        update={
            "enriched": victim.enriched.model_copy(
                update={
                    "summary": "un resumen nuevo que el índice no tiene",
                    "enriched_at": victim.enriched.enriched_at + timedelta(hours=1),
                }
            )
        }
    )
    _write_store(context.items_path, store)

    response = search("agents", QueryContext(**{**context.__dict__, "store": store}))

    assert "index_behind_store" in response.index.degraded
    assert response.results, "a behind index still ANSWERS; it just says so"


# ---------------------------------------------------------------------------
# 20b — verification comes from the live store (M5)
# ---------------------------------------------------------------------------


def test_a_verdict_revoked_without_reindexing_is_not_shown(tmp_path: Path, corpus) -> None:
    """Step 20b / acceptance 8b: `verification_status` is hydrated from the LIVE store.

    A verdict copied into the index could never be invalidated when the verdict changed —
    `surface_fingerprint` hashes (version, type, origin, text) and not the verdict — so a
    `FAIL` revoked by `verify --audit` would keep being served as the `PASS` it used to be.
    CLAUDE.md rule 6, run backwards.

    The index is built while the item carries a PASS and is NOT rebuilt after the verdict is
    removed. Seen red by reading verification from a column: the PASS survives the revocation.
    """
    from xbrain.verification import contract_fingerprint, fingerprint_output

    store, vocab, pages = corpus
    store = dict(store)
    item = store["k03"]
    passing = item.model_copy(
        update={
            "verification": {
                "summary": VerificationVerdict(
                    target="summary",
                    verdict="PASS",
                    faithfulness="PASS",
                    adherence="PASS",
                    output_fingerprint=fingerprint_output(item, "summary"),
                    contract_fingerprint=contract_fingerprint(item, "summary", "English"),
                    verified_at=datetime.now(timezone.utc),
                )
            }
        }
    )
    store["k03"] = passing
    data = tmp_path / "data"
    data.mkdir()
    _write_store(data / "items.json", store)
    index_build.build(data / "index", store, vocab, pages, data / "items.json")
    context = QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )
    term = passing.enriched.summary.split()[0]
    before = next(r for r in search(term, context).results if r.item_id == "k03")
    assert before.summary is not None and before.summary.verification_status == "PASS", (
        "the fixture must actually show a PASS first, or the revocation below proves nothing"
    )

    revoked = dict(store)
    revoked["k03"] = passing.model_copy(update={"verification": {}})
    after_context = QueryContext(**{**context.__dict__, "store": revoked})

    after = next(r for r in search(term, after_context).results if r.item_id == "k03")
    assert after.summary is not None
    assert after.summary.verification_status is None, "a revoked verdict was still shown"


# ---------------------------------------------------------------------------
# 29, 30 — the two ways an index cannot answer
# ---------------------------------------------------------------------------


def test_a_missing_index_names_the_build_command(tmp_path: Path, corpus) -> None:
    """Step 30: an actionable error, never a raw traceback (spec §9.3)."""
    store, vocab, pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    _write_store(data / "items.json", store)
    context = QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
    )
    with pytest.raises(IndexMissingError, match="xbrain index build"):
        search("agents", context)


def test_an_incompatible_manifest_refuses_the_whole_query(context: QueryContext) -> None:
    """Step 29 / spec §9.3: *manifest incompatible: no se consulta parcialmente.*

    Seen red by logging a warning and answering anyway — which is worse than the error,
    because the answer looks the same as a correct one.
    """
    path = manifest_path(context.index_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chunker_version"] = "xbrain-knowledge-chunker/v9"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        search("agents", context)


# ---------------------------------------------------------------------------
# 28 — read-only with respect to the store
# ---------------------------------------------------------------------------


def test_search_does_not_touch_items_json(context: QueryContext) -> None:
    """Acceptance 13, hashed before and after — a fact about the file, not about the code."""
    import hashlib

    before = hashlib.sha256(context.items_path.read_bytes()).hexdigest()
    search("Quillfeather", context)
    assert hashlib.sha256(context.items_path.read_bytes()).hexdigest() == before


def test_filters_reach_the_service(context: QueryContext) -> None:
    """The eight filters are tested at the index; here they are proved to be WIRED.

    A service that accepted `SearchFilters` and dropped it on the floor would pass every
    index-level test in `test_knowledge_lexical.py` and answer every query unfiltered.
    """
    unfiltered = {r.item_id for r in search("agents", context, limit=20).results}
    mine = {
        r.item_id
        for r in search(
            "agents", context, limit=20, filters=SearchFilters(source="own_tweet")
        ).results
    }
    assert mine < unfiltered
    assert all(context.store[item_id].source == "own_tweet" for item_id in mine)


def _rows(data: Path, sql: str) -> int:
    connection = open_index(db_path(data / "index"), read_only=True)
    value = connection.execute(sql).fetchone()[0]
    connection.close()
    return int(value)
