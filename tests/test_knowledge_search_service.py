# tests/test_knowledge_search_service.py
"""`search` end to end (Plan 02 §4, steps 10, 10b, 11, 13, 14, 15, 18, 19, 20, 20b, 29, 30).

RUN THROUGH THE PUBLIC SERVICE, not against its internals. CLAUDE.md rule 3: the judge must
EXECUTE. Every assertion here goes through `search_service.search`, which is the same function
the CLI adapter and Plan 04's MCP tool will call — so a defect that only shows up once the
pieces are wired is visible here rather than in the second adapter.

The fixtures build a REAL index on disk with the real writer and then query it read-only,
because the failures this file is about — a corrupt row excluded, an index behind the store, a
revoked verdict — are all properties of the round trip and none of them can be staged in memory.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import contracts, index_build, index_store, search_service
from xbrain.knowledge.contracts import SearchFilters, SearchResponse
from xbrain.knowledge.index_schema import (
    IndexIncompatibleError,
    IndexMissingError,
    db_path,
    manifest_path,
    open_index,
)
from xbrain.knowledge.index_store import open_for_query
from xbrain.knowledge.lexical import LexicalHit
from xbrain.knowledge.search_service import QueryContext, search
from xbrain.knowledge.surfaces import item_surfaces, knowledge_item
from xbrain.models import (
    Author,
    Content,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Topic,
    TopicPage,
    VerificationVerdict,
)
from xbrain.rubrics import save_vocab
from xbrain.store import save_store, save_topic_pages

FIXTURES = Path(__file__).parent / "fixtures"
UTC = timezone.utc


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


def _paths(data: Path) -> tuple[Path, Path, Path]:
    """The three inputs, in the order every door of this package takes them."""
    return data / "items.json", data / "vocab.yaml", data / "topics.json"


def _persist(data: Path, store, vocab, pages) -> None:
    """All THREE inputs on disk, through the store's own writers.

    The cheap signal covers `vocab.yaml` and `topics.json` as well as `items.json` (P1a), so a
    fixture with only the first would pass the signal tests for a reason unrelated to their
    name (rule 1).
    """
    save_store(store, data / "items.json")
    save_vocab(vocab, data / "vocab.yaml")
    save_topic_pages(pages, data / "topics.json")


def _build(data: Path) -> None:
    """Build from ONE snapshot of the three inputs — the only supported call (P1b)."""
    index_build.build(data / "index", index_build.load_index_inputs(*_paths(data)))


def _context(data: Path, store, vocab, pages) -> QueryContext:
    return QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
    )


@pytest.fixture()
def context(tmp_path: Path, corpus) -> QueryContext:
    """A built index plus the live store — the two halves every query needs."""
    store, vocab, pages = corpus
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    return _context(data, store, vocab, pages)


def _rows(data: Path, sql: str) -> int:
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        return int(connection.execute(sql).fetchone()[0])
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 15 / acceptance 7 — grouping
# ---------------------------------------------------------------------------


def test_a_long_transcript_yields_one_result_with_at_most_three_matches(
    tmp_path: Path, corpus
) -> None:
    """Step 15 / §15.7: ten adjacent windows must not take ten of the top ten.

    Built from a transcript long enough to produce more than three windows, so the cap is
    exercised rather than asserted against a body that could never reach it (rule 2). The
    window count is asserted as a PRECONDITION for exactly that reason.

    Seen red by removing the per-item cap: the same item returns every window it has.
    """
    store, vocab, pages = corpus
    item = store["k08"]
    assert item.content is not None
    long_text = " ".join(f"Marrowgate segment {n} of the talk." for n in range(400))
    sources = list(item.content.sources)
    sources[0] = sources[0].model_copy(update={"text": long_text})
    store = {
        **store,
        "k08": item.model_copy(
            update={"content": item.content.model_copy(update={"sources": sources})}
        ),
    }
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)

    windows = _rows(data, "SELECT COUNT(*) FROM chunks WHERE surface_type = 'video_transcript'")
    assert windows > 3, "the fixture must produce more windows than the cap, or nothing is tested"

    response = search("Marrowgate", context)

    k08 = [r for r in response.results if r.item_id == "k08"]
    assert len(k08) == 1, "one item, one result"
    assert len(k08[0].matches) <= 3


# ---------------------------------------------------------------------------
# 10 / acceptance 4 — a row this code cannot serve honestly
# ---------------------------------------------------------------------------


def test_a_chunk_with_a_manipulated_fingerprint_is_excluded_and_counted(
    context: QueryContext,
) -> None:
    """Step 10 / §15.4 / invariant 6 of spec §3.7.

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


def test_search_returns_a_valid_frozen_response(context: QueryContext) -> None:
    """Acceptance 5: a `SearchResponse` valid against the Plan 01 schema, with the explanation.

    `extra="forbid"` on the frozen models means a field the service invents fails
    construction, so building the response IS the schema check — which is why this asserts
    the round trip through `model_validate` rather than eyeballing keys.

    THE ENVELOPE VERSION IS `"2"`, AND THE LITERAL IS WRITTEN OUT ON PURPOSE. #155 versioned
    the three response envelopes independently, so `SearchResponse` moved to `"2"` while
    `EvidenceBundle` stayed at `"1"`. Asserting it against `SEARCH_SCHEMA_VERSION` would be a
    tautology — that constant is READ OFF this field's default (`contracts.py`) — so the pin
    on the constant lives once, in `tests/test_knowledge_contracts.py`, and what is checked
    here is the different fact that a RESPONSE the service built carries it.

    AND `title` IS ASSERTED POPULATED, WHICH IS THE ONLY WAY TO ASSERT IT AT ALL. It was the
    one field of the envelope that was declared, hydrated and then dropped: `chunks` stores it,
    `LexicalIndex` reads it back into `LexicalHit.title`, and the single production
    `SearchMatch` constructor did not pass it — so every article fragment reached a consumer as
    an orphan paragraph. Spec §4 requires the title to accompany its chunk, and the envelope
    was bumped to `"2"` FOR this field, so the response advertised a capability the service did
    not deliver.

    Nothing caught it because a null is indistinguishable from a never-set until something
    asserts the POPULATED case (rule 2) — this test pinned five other fields of the same match
    and not this one, which is exactly how it reached a fourth review. The expected value is
    the fixture's own article title rather than «is not None»: a constructor that passed some
    other string would satisfy the weaker form.

    Seen red before `title=hit.title` was passed: `None != 'On Controls and Thresholds'`.
    """
    response = search("Quillfeather", context)
    assert isinstance(response, SearchResponse)
    assert SearchResponse.model_validate(response.model_dump()) == response
    assert response.schema_version == "2" and response.strategy == "lexical"
    assert response.results
    match = response.results[0].matches[0]
    assert match.matched_by == ("lexical",)
    assert match.lexical_rank == 1
    assert match.locator.char_start is not None
    assert response.results[0].available_surfaces

    matches = [m for result in response.results for m in result.matches]
    article = next(m for m in matches if m.surface_type == "external_article")
    assert article.title == "On Controls and Thresholds", article.title

    # BOTH DIRECTIONS, because «carry the title» and «invent one» fail the same assertion set
    # otherwise: a constructor emitting a placeholder for every surface satisfies the line
    # above and was measured surviving it. A user note has no title and `None` is the honest
    # value — the item's URL or a neighbour's title in its place is the fabrication A-1
    # removed from the locator, one field over.
    untitled = next(m for m in matches if m.surface_type == "user_note")
    assert untitled.title is None, untitled.title


def test_no_embeddings_is_read_off_the_manifest_not_hard_coded(context: QueryContext) -> None:
    """B-i (gate round 04, M-2 of gate 03): `no_embeddings` was a CONSTANT in `_degraded`.

    Spec §9.3: the response declares a degraded strategy when there is no vector backend —
    and the manifest is where the backend is recorded (`embeddings: null` until Plan 03
    fills `{model, dimension, normalized, command_version}`). A flag that is always on says
    nothing about this index; the day Plan 03 writes the embeddings block, the test beside
    this one would stay green while the response lied — the exact shape of F-2, fixed one
    round ago on the `strategy` field.

    Staged through the manifest on disk, not through a fake index: write an embeddings block,
    query, and the flag must be gone; write `null` back and it must return.

    Seen red before the fix: `"no_embeddings"` was declared with the block present.
    """
    path = manifest_path(context.index_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["embeddings"] = {
        "model": "test-embedder",
        "dimension": 8,
        "normalized": True,
        "command_version": "0",
    }
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert "no_embeddings" not in search("Quillfeather", context).index.degraded

    raw["embeddings"] = None
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert "no_embeddings" in search("Quillfeather", context).index.degraded


def test_the_response_declares_no_embeddings(context: QueryContext) -> None:
    """Spec §9.3 / Plan 02 §16: without a vector backend the response DECLARES the degradation.

    Plan 02 is literally that state, and saying so is what stops a consumer reading a lexical
    answer as a hybrid one. Seen red by returning an empty `degraded`.
    """
    response = search("Quillfeather", context)
    assert "no_embeddings" in response.index.degraded


def test_a_strategy_with_no_backend_is_answered_lexically_and_says_so(
    context: QueryContext, monkeypatch
) -> None:
    """F-2 / spec §9.3: *lexical sigue operativo y el response declara estrategia degradada;
    no finge resultados vectoriales.*

    THE PREMISE IS PINNED, NOT INHERITED (M-2, round 02): this test is ABOUT the degradation,
    so it needs a strategy with no backend — and it used to borrow that from production by
    assuming `vector` stays unimplemented, the coupling F-2 removed from the guardrail one
    door further in. Simulated with `vector` added to `IMPLEMENTED_STRATEGIES`: red before
    the pin (`'vector' == 'lexical'` fails), green with it, because the test now declares
    the world it tests instead of depending on Plan 03 not having landed.

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
    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical"}))
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


@pytest.mark.parametrize("table", ["chunks_fts", "surfaces", "profiles_fts"])
def test_an_index_missing_a_table_is_refused_naming_the_rebuild(
    context: QueryContext, table: str
) -> None:
    """C-2 (round 02, both gates): an INCOMPLETE schema was answered as a valid search.

    `open_index(read_only=True)` only proved `sqlite_master` was readable, and
    `LexicalIndex._fetch` swallowed EVERY `sqlite3.OperationalError` but a read-only one, so
    `DROP TABLE chunks_fts` produced a normal response — profile-plane results, or none —
    with `degraded: ["no_embeddings"]` and no word about the missing table. Measured on the
    real corpus (2,404 items, 2026-09-01): `query_returned_normally=True`, 10 results, and
    nothing declared. A schema that cannot answer the question is the corrupt-base case of
    F-3 in another costume, and it ends with the same sentence.

    Seen red before the fix: `search` returned a `SearchResponse` for all three tables.
    """
    connection = sqlite3.connect(db_path(context.index_dir))
    connection.execute(f"DROP TABLE {table}")  # nosec B608 — a test fixture, closed set
    connection.commit()
    connection.close()

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        search("Quillfeather", context)
    assert table in str(caught.value), "names WHAT is missing, not only that something is"


def test_a_corrupt_fts_structure_is_an_actionable_error_not_a_traceback(
    context: QueryContext,
) -> None:
    """G-4 (gate round 04): a `sqlite3.DatabaseError` at QUERY time escaped as a traceback.

    C-2 guards the DECLARED tables; F-3 proves the first page is readable. Neither sees an
    FTS5 SHADOW table (`chunks_fts_data`, outside `TABLES | FTS_TABLES`) that is gone, nor a
    file corrupt beyond page 1: on the real corpus `DROP TABLE chunks_fts_data` made `search`
    print a 68-line Rich traceback ending in `DatabaseError: fts5: corruption found reading
    blob 10 from table "chunks_fts"` — loud, not silent, but not the actionable error spec
    §9.3 requires and Plan 02 §11 tabulates as *base corrupta -> `index build --force`*.

    Two layers close it: a positive PROBE at the open door (a trivial `MATCH` on each FTS
    plane, 0.01 ms each — it sees exactly what `search` sees), and `LexicalIndex._fetch`
    turning any remaining `DatabaseError` into the same sentence.

    Seen red before the fix: `sqlite3.DatabaseError` propagated out of `search`.
    """
    connection = sqlite3.connect(db_path(context.index_dir))
    connection.execute("DROP TABLE chunks_fts_data")
    connection.commit()
    connection.close()

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force"):
        search("Quillfeather", context)


def test_search_refuses_a_base_that_disagrees_with_its_manifest(context: QueryContext) -> None:
    """G-2's third closure, and B-c (gate round 04): `search` compared VERSIONS and schema
    against the manifest, never `counts`, so over a base whose topic plane had been deleted
    behind the manifest's back it answered normally and declared nothing — while `update`
    and `status` (C-3) refused the same base naming the plane. The gate called this *the gap
    G-2 enters through*: a base of zero rows under a standing manifest is exactly a base
    that does not hold what its manifest declares.

    Five `COUNT(*)` per query (measured 0.04 ms in total on the 52 MB real index), and the
    same sentence `update` and `status` already use, so the three instruments agree.

    Seen red before the fix: `search` returned a `SearchResponse` with `degraded ==
    ("no_embeddings",)` over the amputated base.
    """
    connection = open_index(db_path(context.index_dir))
    try:
        with connection:
            index_build._clear_topics(connection)
    finally:
        connection.close()

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        search("Quillfeather", context)
    assert "topics" in str(caught.value), "names WHICH plane disagrees"


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
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)

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

    index = open_for_query(
        context.index_dir, context.items_path, context.vocab_path, context.topics_path
    )
    try:
        with pytest.raises(sqlite3.OperationalError):
            index.lexical.connection.execute("DELETE FROM chunks")
    finally:
        index.close()


# ---------------------------------------------------------------------------
# A-1 (round 05) — the profile plane is a route, and it has to be seen serving
# ---------------------------------------------------------------------------


def test_a_query_by_handle_is_answered_by_the_profile_plane_with_no_citable_match(
    context: QueryContext,
) -> None:
    """A-1 (gate Fable, round 05): the profile plane of `search` could DISAPPEAR entirely
    with the suite green. With `profile_ids = []` in `search_service.search` the nine Plan 02
    suites stayed at 274 passed, and on this fixture a query by author went from 10 results
    to 0 with nothing red — CLAUDE.md rule 11's fail-open shape (the answer shrinks and
    nothing says so) on the plane G-1 had just touched. The retriever is tested (the
    totality tests of `lexical.py` go red on it); what was missing was ONE positive test at
    the SERVICE level, so here it is.

    The premise is asserted first (rule 1): the handle lives in NO chunk and DOES live in
    the profile plane, so the profile plane is the only route by which these results can
    arrive. Then the shape spec §5.1.A requires — a profile is a retrieval representation,
    never a citation: `matches == ()`, a `verify_with` that still leads to a real surface —
    and the human line that says why there is no excerpt.

    Seen red under `profile_ids = []` in an isolated copy: `response.results == ()`.
    """

    opened = open_for_query(
        context.index_dir,
        context.items_path,
        context.vocab_path,
        context.topics_path,
    )
    try:
        assert opened.lexical.search("vgonpa", 50) == (), (
            "the handle must live in no chunk, or the chunk plane could be what answers"
        )
        assert opened.lexical.search_profiles("vgonpa", 50), "and the profile plane holds it"
    finally:
        opened.close()

    response = search("vgonpa", context)

    assert response.results, "a query by handle is served by the profile plane, or by nothing"
    assert all(result.matches == () for result in response.results), "a profile is never cited"
    assert all(result.verify_with for result in response.results), "and still leads to a source"
    assert all(context.store[r.item_id].author.handle == "vgonpa" for r in response.results)
    # The HUMAN line that says why there is no excerpt is `render`'s, and `render` is 02.11's.
    # The service-side facts it renders are all asserted above; 02.11 restores the last line.


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
    save_store(store, context.items_path)

    response = search("agents", QueryContext(**{**context.__dict__, "store": store}))

    assert "index_behind_store" in response.index.degraded
    assert response.results, "a behind index still ANSWERS; it just says so"


@pytest.mark.parametrize("moved", ["topics.json", "vocab.yaml"])
def test_editing_topics_or_vocab_without_reindexing_declares_the_index_behind(
    context: QueryContext, moved: str
) -> None:
    """P1a (gate Codex, round 05 — probes B and C, reproduced verbatim on HEAD `0312634`).

    The index derives from THREE inputs. `topic_overview`, `topic_note` and
    `topic_description` are chunks it serves, and every topic description enters the
    profile of each item assigned to it (spec §5.1.A). The manifest recorded all three
    fingerprints, but the query door compared `StoreSignal.of(items_path)` and nothing
    else — so `xbrain topics`, which writes `topics.json` and never touches `items.json`
    (`cli.py`), and a `vocab.yaml` edit both left every later `search` answering over the
    old plane with `degraded: ("no_embeddings",)`. The gate's probes: a term added to a
    topic note or a topic description, byte-identical `items.json`, `search` -> 0 results,
    no `index_behind_store`, while `status` reported `topics_changed=1`. Spec §5.6 / §9.3:
    *nunca se sirve evidencia obsoleta en silencio*; `docs/tutorial.md` promised that
    forgetting the update after `topics` is something `search` tells you.

    The file is rewritten the way the CLI rewrites it (`save_topic_pages` / `save_vocab`),
    and the assertion is on the FLAG: the answer is still given (possibly-stale evidence is
    usable as long as it says so), it just cannot stay quiet. Seen red before the fix:
    `"index_behind_store" not in ("no_embeddings",)` on both parametrisations.
    """
    assert "index_behind_store" not in search("agents", context).index.degraded

    if moved == "topics.json":
        slug = sorted(context.topic_pages)[0]
        page = context.topic_pages[slug]
        pages = {
            **context.topic_pages,
            slug: page.model_copy(update={"notes": [*page.notes, "topicfreshonlytoken"]}),
        }
        save_topic_pages(pages, context.topics_path)
        live = QueryContext(**{**context.__dict__, "topic_pages": pages})
        term = "topicfreshonlytoken"
    else:
        vocab = list(context.vocab)
        vocab[0] = vocab[0].model_copy(
            update={"description": vocab[0].description + " vocabfreshonlytoken"}
        )
        save_vocab(vocab, context.vocab_path)
        live = QueryContext(**{**context.__dict__, "vocab": vocab})
        term = "vocabfreshonlytoken"

    response = search(term, live)
    assert "index_behind_store" in response.index.degraded, response.index.degraded
    assert "index_behind_store" in search("agents", live).index.degraded
    # And the deep instrument agrees with the cheap one on the same state.
    assert (
        index_build.status(
            live.index_dir,
            index_build.load_index_inputs(live.items_path, live.vocab_path, live.topics_path),
        ).behind
        is True
    )


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
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)
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
    _persist(data, store, vocab, pages)
    context = _context(data, store, vocab, pages)
    with pytest.raises(IndexMissingError, match="xbrain index build"):
        search("agents", context)


def test_chunker_parameters_that_moved_refuse_the_query_as_update_and_status_do(
    context: QueryContext,
) -> None:
    """M-1 (gate Fable, round 05): `search` never checked `chunker_params`.

    `load_compatible_manifest` compares the parameters only when handed `params`, and
    `search` called `open_for_query(index_dir, items_path)` without them — the one production
    path, and the docstring of `open_for_query` said the argument existed for exactly this
    case (a sweep that lands on new parameters without bumping `CHUNKER_VERSION` cuts chunks
    differently under IDENTICAL ids). So `update --dry-run` refused, `status` said «ninguna
    consulta lo usará», and `search` answered: two instruments, opposite answers (rule 9),
    and a guard only the suite exercised (rule 1). `docs/troubleshooting.md` promised the
    refusal on «the chunker or its parameters» and it was true of two commands out of three.

    The parameters now travel in `QueryContext` (the CLI fills them from the same
    `IndexOptions` the build uses) and the query is refused entire, naming the rebuild. The
    manifest is restored afterwards and the same query is shown to ANSWER, so the refusal is
    proved to come from the parameters and not from a door that always shuts (rule 1).

    Seen red before the fix: `search` returned a `SearchResponse` with 2 results.
    """
    path = manifest_path(context.index_dir)
    original = path.read_text(encoding="utf-8")
    raw = json.loads(original)
    raw["chunker_params"]["target"] = raw["chunker_params"]["target"] + 400
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="index build --force") as caught:
        search("Quillfeather", context)
    assert "chunker_params" in str(caught.value), "names WHAT moved"

    path.write_text(original, encoding="utf-8")
    assert search("Quillfeather", context).results, "with the parameters back, it answers"


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


@pytest.mark.parametrize("origin", ["vlm", "asr"])
def test_an_origin_filter_never_admits_a_profile_only_candidate(
    context: QueryContext, origin: str
) -> None:
    """G-1 (gate round 04): `--origin` reached the chunk plane and NOT the profile plane.

    `search_profiles` applied only the item-scoped clauses, `origins` lives on `chunks.origin`,
    and `_append_profile_candidates` appended whatever the profile plane returned — so
    `search "Forecasting" --origin asr` on the real corpus (2,404 items, 2026-09-01) answered
    8 items, 7 with no match and 6 with no ASR surface anywhere. Spec §7.2 puts `--origin vlm`
    as its literal example; acceptance 6 says *the eight* filters, and this was the eighth.

    Through the public service, on the fixture corpus: every result under an origin filter
    carries at least one citable match, and every match is of that origin. A profile has no
    origin, so it cannot satisfy the filter and contributes no candidate.

    Seen red before the fix: `k02` (no VLM or ASR surface at all) and `k08` came back with
    zero matches under both origins, for the queries `agents` and `evaluation`.
    """
    for query in ("agents", "evaluation", "Quillfeather", "audit"):
        response = search(query, context, filters=SearchFilters(origins=(origin,)), limit=10)
        for result in response.results:
            assert result.matches, f"{query!r} --origin {origin}: {result.item_id} has no match"
            assert {m.origin for m in result.matches} == {origin}, (query, result.item_id)


def _rows(data: Path, sql: str) -> int:
    connection = open_index(db_path(data / "index"), read_only=True)
    value = connection.execute(sql).fetchone()[0]
    connection.close()
    return int(value)


# ---------------------------------------------------------------------------
# A-1 — the poster is not the author of what they quote, in `search` too
# ---------------------------------------------------------------------------


def test_a_quoted_post_match_carries_the_quoted_author_not_the_poster(
    context: QueryContext,
) -> None:
    """A-1 (round 02, both gates): `SearchMatch.attribution` exists in the frozen contract for
    exactly this and `search` never filled it. The index STORES the surface's attribution
    (`surfaces.attribution_*`) and `search` threw it away on the way out: k07's quoted post
    belongs to @othervoice and came back with `attribution: null` under the poster's name,
    with a locator pointing at the poster's tweet. On the real corpus (2026-09-01) the same
    shape: item 1875646438350450928, quoted post by @mdancho84, served under @miguelgfierro.

    CLAUDE.md lists *a quoted post rendered as if the poster had written it* among the
    defects that cost blood, and every other LLM surface enforces the rule through one shared
    label; `search` is a NEW surface that did not (rule 5, spec §3.7 invariant 3).

    Seen red before the fix: `match.attribution is None`.
    """
    response = search("audit", context)
    result = next(r for r in response.results if r.item_id == "k07")
    match = next(m for m in result.matches if m.surface_type == "quoted_post")
    quoted = next(s for s in item_surfaces(context.store["k07"]) if s.surface_type == "quoted_post")

    assert match.attribution == quoted.attribution
    assert match.attribution is not None and match.attribution.handle == "othervoice"
    assert result.author.handle == "vgonpa", "the RESULT is still the poster's item"
    assert match.attribution != result.author, "and the match says who wrote the quote"


def test_a_match_locator_is_the_surface_locator_plus_the_character_range(
    context: QueryContext,
) -> None:
    """The second half of A-1: `_match` fabricated a generic locator from the chunk's own
    columns — `source_index: null`, `content_kind: null`, `url` = the ITEM's — so a consumer
    could not resolve the match back to the source it came from (spec §3.8: *superficie,
    propietario y localizador son resolubles*). The surface's locator was in
    `surfaces.locator_json` the whole time.

    The match locator is now the SURFACE's locator with the chunk's character range on top:
    `source_index`, `content_kind` and the source's own URL for a content source;
    `media_index` for an image description.

    Seen red before the fix: `source_index is None` and `url` was the poster's tweet.
    """
    response = search("audit", context)
    match = next(
        m
        for r in response.results
        if r.item_id == "k07"
        for m in r.matches
        if m.surface_type == "quoted_post"
    )
    quoted = next(s for s in item_surfaces(context.store["k07"]) if s.surface_type == "quoted_post")
    expected = quoted.locator.model_dump(exclude_none=True)
    got = match.locator.model_dump(exclude_none=True)
    assert {k: got[k] for k in expected} == expected, got
    assert got["url"] == "https://x.com/othervoice/status/k07q"
    assert match.locator.char_start == 0 and match.locator.char_end == len(quoted.text)

    described = next(
        (iid, s)
        for iid, item in context.store.items()
        for s in item_surfaces(item)
        if s.surface_type == "image_description"
    )
    term = next(w for w in described[1].text.split() if len(w) > 6).strip(".,")
    response = search(term, context, limit=20)
    image = next(
        m
        for r in response.results
        if r.item_id == described[0]
        for m in r.matches
        if m.surface_type == "image_description"
    )
    assert image.locator.kind == "media"
    assert image.locator.media_index == described[1].locator.media_index is not None


# ---------------------------------------------------------------------------
# The four guards this child SHIPS that the atomic split left without a test
#
# The matrix moves `test_knowledge_search_service.py`'s L999-1397 block to 02.14 wholesale,
# and only two of those tests actually need `get_service`. The rest cover code that lands
# HERE, so following the move literally ships four guards nothing can hold: measured, six
# adversarial mutations of this child's own code survived the whole suite. The matrix's own
# §2.3(b) disqualifies a split that «ship código sin prueba exigible», so these are the
# SMALLEST tests that close each survivor, written here rather than ported wholesale — the
# block's richer versions still travel to 02.14 with the file that names them.
# ---------------------------------------------------------------------------


def test_a_forged_chunk_id_is_excluded_and_counted(context: QueryContext) -> None:
    """M-2: `chunk_id` is the one field a `--json`/MCP consumer walks back to the surface.

    It is NOT an arm of the evidence — it is DERIVED from two arms that are (`surface_id`,
    `chunk_index`) plus the chunker version the door already proved — so hashing it again
    would prove nothing, and it is RECOMPUTED and compared instead. Forging it alone leaves
    the fingerprint recomputing perfectly: the right text is served under a wrong name, with
    `corrupt_chunks_excluded: 0`, and the id a consumer follows resolves to nothing.

    Seen red with the `hit.chunk_id != chunk_id(...)` arm removed: the row comes back and the
    counter stays 0.
    """
    connection = sqlite3.connect(db_path(context.index_dir))
    victim = connection.execute(
        "SELECT chunk_id FROM chunks WHERE text LIKE '%Quillfeather%' LIMIT 1"
    ).fetchone()[0]
    forged = f"{victim}-forged"
    connection.execute("UPDATE chunks SET chunk_id = ? WHERE chunk_id = ?", (forged, victim))
    connection.commit()
    connection.close()

    response = search("Quillfeather", context)

    assert response.index.corrupt_chunks_excluded >= 1
    returned = {m.chunk_id for r in response.results for m in r.matches}
    assert forged not in returned, "a forged id was served under a name nothing resolves"
    assert victim not in returned


def test_a_chunk_whose_surface_holds_no_locator_is_excluded_and_counted(
    context: QueryContext,
) -> None:
    """B-k / invariant 1 of spec §3.7: every chunk resolves to a surface.

    A chunk whose surface row holds no locator cannot be served with one, and `_match` used
    to FABRICATE it from the chunk's own columns — the ITEM's url, no source index — which
    points a reader at the wrong bytes with confidence. It is excluded and counted in the
    SAME counter as a fingerprint that does not recompute, because it is the same operator
    situation: a row this code cannot serve honestly, repaired by a rebuild.

    Seen red with `resolvable_hits` returning every hit: `_match` raises instead, which is
    the guard behind the guard — the exclusion is what keeps it unreachable.
    """
    connection = sqlite3.connect(db_path(context.index_dir))
    surfaces = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT surface_id FROM chunks WHERE text LIKE '%Quillfeather%'"
        )
    ]
    assert surfaces, "the fixture must hold a matching chunk, or nothing is tested"
    # `locator_json` is NOT NULL, so the reachable state is a row holding a document a
    # `Locator` CANNOT hold — which is what `_surface_locator` answers None for, and what a
    # different emitter version or a hand edit actually leaves behind.
    connection.executemany(
        "UPDATE surfaces SET locator_json = ? WHERE surface_id = ?",
        [('{"kind": "not-a-locator-kind"}', s) for s in surfaces],
    )
    connection.commit()
    connection.close()

    response = search("Quillfeather", context)

    assert response.index.corrupt_chunks_excluded >= len(surfaces)
    served = {m.chunk_id for r in response.results for m in r.matches}
    assert served.isdisjoint(surfaces)


def _locatorless_hit() -> LexicalHit:
    """A hit whose surface row holds no locator a `Locator` can carry — B-k's shape."""
    return LexicalHit(
        chunk_id="s1:0:x",
        surface_id="s1",
        owner_type="item",
        owner_id="k01",
        surface_type="post",
        origin="user",
        trust_class="user_text",
        derived=False,
        chunk_index=0,
        char_start=0,
        char_end=3,
        title=None,
        url=None,
        language=None,
        fingerprint="0" * 64,
        text="abc",
        excerpt="abc",
        score=-1.0,
        attribution=None,
        surface_locator=None,
    )


def test_resolvable_hits_drops_and_counts_a_hit_with_no_locator() -> None:
    """The function's OWN contract, which `search` cannot pin.

    `resolvable_hits` and `verify_fingerprints` overlap on purpose — the second re-checks the
    locator so a caller that skipped the first is still closed — and that overlap makes the
    first one's exclusion INVISIBLE through `search`: neuter it and the totals are identical,
    because the second arm drops the same hits into the same counter. Measured: that mutation
    survives every end-to-end assertion in this file.

    So the belt is pinned where it is decided. `resolvable_hits` is public — 02.14's seams
    test enumerates it by name, and Plan 03's retriever is the second caller — so «returns
    (kept, excluded)» is a contract, not an implementation detail of `search`.
    """
    kept, excluded = index_store.resolvable_hits([_locatorless_hit()])

    assert kept == () and excluded == 1


def test_verify_fingerprints_closes_on_a_locatorless_hit_a_caller_forgot_to_drop() -> None:
    """The BELT its docstring promises, which no path through `search` can reach.

    `search` runs `resolvable_hits` first, so by the time `verify_fingerprints` sees a hit it
    has a locator — and that makes the claim «a caller that skipped that step is still
    closed» an untested sentence about a second caller that does not exist yet. Asked of the
    function directly, because a guard whose only cover is the guard in front of it is one
    guard (rule 11's fail-open cell), and 02.9 is where the second caller becomes possible.
    """
    kept, excluded = index_store.verify_fingerprints([_locatorless_hit()])

    assert kept == () and excluded == 1


def test_a_get_cursor_is_refused_by_name_rather_than_restarted_from_zero(
    context: QueryContext,
) -> None:
    """M-4: three cursor shapes live in this package and each decoder refuses the others.

    Restarting from zero on a foreign cursor would loop a paginating consumer forever while
    looking like progress — every page identical to the first, `truncated` true, no error.
    The refusal names WHICH sequence the cursor belongs to.

    Seen red with the prefix check removed: the query answers page one again.
    """
    with pytest.raises(ValueError, match="no es un cursor de `search`"):
        search("Quillfeather", context, cursor="q:3")
    with pytest.raises(ValueError, match="no es un cursor de `search`"):
        search("Quillfeather", context, cursor="post:0")


def test_a_malformed_search_cursor_is_refused_rather_than_restarted_from_zero(
    context: QueryContext,
) -> None:
    """The same loop by the other route: a cursor with the right prefix and no offset.

    Seen red with the `int(tail)` failure swallowed into `return 0`: page one, forever.
    """
    with pytest.raises(ValueError, match="Cursor inválido"):
        search("Quillfeather", context, cursor="s:abc")
    with pytest.raises(ValueError, match="Cursor inválido"):
        search("Quillfeather", context, cursor="s:-1")


def test_a_page_shorter_than_the_ranking_is_declared_and_its_cursor_continues(
    context: QueryContext,
) -> None:
    """`truncated` is a MEASUREMENT of the ranking against the page, not a decoration.

    Both fields were declared in the frozen envelope and never set: `--limit 2` cut a long
    ranking to two with `truncated: false` — the silent cut spec §9.3 forbids, on a field
    that could not come out any other way (rule 2). The cursor is asserted to CONTINUE rather
    than merely to exist: pages must be disjoint and reassemble the ranking in order, which
    is the only thing that makes paging different from re-querying.

    The complement is asserted in the same test, because `truncated` being always-true is the
    same defect wearing the other sign.

    Seen red with `truncated = False`: the first assertion; and with the cursor dropped, the
    second page cannot be asked for.
    """
    whole = search("Quillfeather vgonpa", context, limit=50)
    assert len(whole.results) > 2, "the ranking must exceed the page, or nothing is tested"
    assert whole.truncated is False and whole.cursor is None

    first = search("Quillfeather vgonpa", context, limit=1)
    assert first.truncated is True
    assert first.cursor is not None

    second = search("Quillfeather vgonpa", context, limit=1, cursor=first.cursor)
    assert [r.item_id for r in first.results] != [r.item_id for r in second.results]
    assert [r.item_id for r in first.results] + [r.item_id for r in second.results] == [
        r.item_id for r in whole.results[:2]
    ], "the pages must reassemble the ranking, not re-rank it"


def test_profile_only_candidates_follow_every_chunk_matched_result(
    context: QueryContext,
) -> None:
    """The two planes' bm25 scores are computed over different corpora (spec §5.1).

    There is no scale on which to compare them, so appending is a DECLARED ordering and
    interleaving would be an invented one. Chunk matches lead because they carry a citable
    excerpt — the thing a consumer can actually verify — while a profile match only says
    *this item is about that* and the profile itself may never be quoted.

    Asserted as a partition rather than on an exact order, which is what the property says:
    every result carrying matches precedes every result carrying none.

    Seen red with the profile candidates seeded before the grouping: the first result carries
    no matches.
    """
    response = search("Quillfeather vgonpa", context, limit=50)
    shape = [bool(result.matches) for result in response.results]

    assert any(shape) and not all(shape), "the query must produce BOTH kinds, or nothing is tested"
    assert shape == sorted(shape, reverse=True), (
        f"a profile-only candidate was interleaved among the chunk-matched ones: {shape}"
    )


def test_an_empty_query_is_refused_by_the_service_before_the_index_is_opened(
    context: QueryContext,
) -> None:
    """Step 14 / spec §9.3 — and the PORTED test for this passed for the wrong reason.

    It asserted `pytest.raises(ValueError, match="vacía")`, and that is satisfied by
    `lexical._expression`'s own «la consulta está vacía», raised deep inside the retriever
    AFTER the index has been opened. Measured: with `_validate`'s refusal deleted the ported
    assertion stayed GREEN, because a different layer raised a different sentence that happens
    to share a word (rule 1).

    Two things separate the layers, and both are asserted. The SENTENCE: `_validate` tells the
    operator what to do next («Escribe algo que buscar»), which the retriever's message does
    not. And the MOMENT: validation refuses before any file is touched, so an empty query
    against an index that does not exist is still an empty-query error — under the mutation it
    becomes `IndexMissingError`, i.e. the operator is told to build an index when what they
    actually did was ask nothing.

    Seen red with `_validate`'s first clause removed: the second assertion raises
    `IndexMissingError`.
    """
    with pytest.raises(ValueError, match="Escribe algo que buscar"):
        search("   ", context)

    nowhere = replace(context, index_dir=context.index_dir.parent / "no-index-here")
    with pytest.raises(ValueError, match="Escribe algo que buscar"):
        search("", nowhere)


def test_the_two_degradations_are_ordered_by_the_declared_constant(
    context: QueryContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spec §3.7.8 applies to the ENVELOPE, not only to the ranking.

    Two responses over the same state must be byte-identical, and `_degraded` builds its
    answer from a SET — so `return tuple(flags)` would order the flags by set iteration. One
    degradation can never show that; this is the only state where both apply at once.

    AND ASSERTING THE PAIR AS A LITERAL IS NOT ENOUGH, which is the second thing measured
    here. With `PYTHONHASHSEED` pinned (as `check.sh` pins it) the set happens to iterate
    these two strings in exactly `DEGRADED_ORDER`, so `tuple(flags)` SURVIVED an assertion on
    the literal pair — green by coincidence, on a value that could not come out any other way
    (rules 1 and 2). The claim is that the order comes FROM the declared constant, so the
    constant is reversed and the output must follow: a mutant that ignores it cannot.

    Seen red with `return tuple(flags)`: the reversed run still answers in set order.
    """
    items = context.items_path
    items.write_text(items.read_text(encoding="utf-8") + " ", encoding="utf-8")

    degraded = search("Quillfeather", context).index.degraded
    assert set(degraded) == {"index_behind_store", "no_embeddings"}, degraded
    assert degraded == index_store.DEGRADED_ORDER, degraded

    monkeypatch.setattr(index_store, "DEGRADED_ORDER", ("no_embeddings", "index_behind_store"))
    reversed_run = search("Quillfeather", context).index.degraded

    assert reversed_run == ("no_embeddings", "index_behind_store"), reversed_run


def test_the_requested_strategy_that_did_not_run_leads_the_degradation_tuple(
    context: QueryContext,
) -> None:
    """*What you asked for did not run* outranks *this index has no embeddings*.

    The ported test asserted MEMBERSHIP — `f"{requested}_not_implemented" in degraded` — which
    a tuple in any order satisfies. Order is the claim here: it is what a consumer reads first,
    and putting it first keeps the envelope deterministic without a set operation.

    Seen red with `self.degraded + strategy_degradation`: the index's own flag leads.
    """
    degraded = search("Quillfeather", context, strategy="hybrid").index.degraded

    assert degraded[0] == "hybrid_not_implemented", degraded
    assert degraded == ("hybrid_not_implemented", "no_embeddings")


def test_a_primary_match_names_ITSELF_not_every_primary_surface_the_item_has(
    context: QueryContext,
) -> None:
    """The sharp half of §15.8, and the ported test could not see it.

    «A primary match names its own surface» and «a derived match names the item's primary
    surfaces» are two branches, and on an item with ONE primary surface they return the same
    thing — so `matches[0].surface_type in result.verify_with` is satisfied by the fallthrough
    as well. Measured: deleting the primary branch left the ported assertion green.

    k03 carries TWO primary surfaces (`post` and `external_article`), so the branches diverge:
    verifying an article match against the tweet that linked it is the detour spec §3.5 says a
    derived result needs and a primary one does not.

    Seen red with the `if primary_matched` branch removed: `verify_with` comes back holding
    both surfaces.
    """
    response = search("Quillfeather", context)
    result = next(r for r in response.results if r.item_id == "k03")
    matched = [m.surface_type for m in result.matches if m.derived is False]

    assert "external_article" in matched, "the fixture must match k03's article"
    assert "post" in result.available_surfaces, "and k03 must carry a second primary surface"
    assert result.verify_with == tuple(dict.fromkeys(matched)), result.verify_with
    assert "post" not in result.verify_with, "a primary match does not detour through the post"


def test_a_topic_match_lists_primary_members_before_secondary_ones(tmp_path: Path, corpus) -> None:
    """Primary first is a CLAIM about strength, not an incidental order.

    A primary assignment says the item IS about the topic; a secondary says it touches it. The
    fixture's two topics have no secondary members at all, so the ported topic test cannot
    tell the orders apart — measured: swapping them left it green. One item is reassigned here
    to make the two groups non-empty, which is the only state where the claim is visible.

    Seen red with `(secondary + primary)`: k03 leads.
    """
    store, vocab, pages = corpus
    slug = "agent-evaluation"
    guest = store["k03"]
    assert guest.enriched is not None and guest.enriched.primary_topic != slug, (
        "the guest must belong to ANOTHER topic primarily, or it is not a secondary member"
    )
    store = {
        **store,
        "k03": guest.model_copy(
            update={
                "enriched": guest.enriched.model_copy(
                    update={"topics": [*guest.enriched.topics, slug]}
                )
            }
        ),
    }
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    ctx = _context(data, store, vocab, pages)

    members = search_service.supporting_item_ids(slug, ctx)

    assert set(members) == {"k02", "k03", "k08"}, members
    assert members == ("k02", "k08", "k03"), "primary members lead, each group sorted"


def test_match_refuses_a_locatorless_hit_rather_than_fabricating_a_locator() -> None:
    """The backstop behind B-k, which no path through `search` can reach.

    `resolvable_hits` drops these upstream, so this raise is unreachable in production — and
    that is exactly why it needs a test: its own docstring calls it «what keeps a fabrication
    from ever being reachable again, not a path a caller takes», and a guard whose only cover
    is the guard in front of it is one guard (rule 11's fail-open cell). The fabrication it
    replaced invented `content_source` with the ITEM's url and no source index, pointing a
    reader at the wrong bytes with confidence.

    Seen red with the raise removed: `SearchMatch` is constructed with `locator=None`, which
    the frozen contract then rejects for a different reason entirely — a traceback naming no
    command instead of the actionable sentence.
    """
    with pytest.raises(IndexIncompatibleError, match="no resuelve a su superficie"):
        search_service._match(1, _locatorless_hit())


# ---------------------------------------------------------------------------
# Review round 1 — the candidate window is not the answer set
# ---------------------------------------------------------------------------


def _seed_one_source(item: Item, body: str) -> Item:
    """Replace the first content source's body, leaving every other field alone."""
    assert item.content is not None
    first, *rest = item.content.sources
    return item.model_copy(
        update={
            "content": item.content.model_copy(
                update={"sources": [first.model_copy(update={"text": body}), *rest]}
            )
        }
    )


def _long_and_short(
    store: dict[str, Item], token: str
) -> tuple[dict[str, Item], list[str], list[str]]:
    """Two documents that monopolise the candidate window, and two that sit outside it.

    A token in a fetched BODY reaches `chunks` and not `profile_text` — which carries the
    item's own text, titles, summary, digests, topics and author, never a fetched body — so
    the chunk plane alone decides this ranking and the profile plane cannot refill the page.
    Sources carrying `blocks` are skipped: `ContentSourceSuccess` validates that `text` is the
    concatenation of its blocks, so rewriting the body alone would not construct.
    """
    usable = [
        k
        for k, i in store.items()
        if i.content
        and i.content.sources
        and getattr(i.content.sources[0], "text", None)
        and not getattr(i.content.sources[0], "blocks", None)
    ]
    hogs, tail = usable[:2], usable[2:4]
    long_body = " ".join(f"{token} parrafo {n} del documento largo." for n in range(300))
    seeded = dict(store)
    for k in hogs:
        seeded[k] = _seed_one_source(store[k], long_body)
    for k in tail:
        seeded[k] = _seed_one_source(store[k], f"Una nota breve que menciona {token} una vez.")
    return seeded, hogs, tail


def test_a_page_refills_past_excluded_candidates_instead_of_reporting_the_corpus_empty(
    tmp_path: Path, corpus
) -> None:
    """The candidate window is sized in CANDIDATES; the page is measured in ANSWERS.

    `search_owners(q, N)` materialises `N * 4` chunks and stops as soon as they hold N distinct
    owners. Everything after that drops rows — `resolvable_hits`, `verify_fingerprints`, and
    `_group_by_item` for an owner no longer in the store — so a window sized for two owners can
    survive into ZERO, and nothing re-deepened it.

    Staged so the window is a genuine PREFIX: two long documents monopolise the first eight
    chunks, two short ones sit outside. Corrupt the long pair and, before the fix, `limit=1`
    answered `results=[]`, `truncated=False`, `cursor=None` — a COMPLETE "the corpus has
    nothing" over a corpus that still matches twice. That is worse than a short page: spec
    §9.3 forbids the silent cut, and this one claims something about the corpus that is false.

    The exclusion count is asserted to GROW with the deepening rather than to stay at the old
    window's 8, because `corrupt_chunks_excluded` counts what THIS response considered, and a
    response that looked deeper considered more.

    Seen red before the fix: `results == []` with `truncated is False`.
    """
    store, vocab, pages = corpus
    token = "Thornwillow"
    store, hogs, tail = _long_and_short(store, token)
    data = tmp_path / "data"
    _persist(data, store, vocab, pages)
    _build(data)
    context = _context(data, store, vocab, pages)

    clean = search(token, context, limit=50)
    ranking = [r.item_id for r in clean.results]
    # The pair leads; WHICH of the two leads is bm25's call and is not this test's claim.
    assert set(ranking[:2]) == set(hogs), f"the long pair must lead the ranking: {ranking}"
    assert set(tail) <= set(ranking), "and the short pair must be findable"

    connection = sqlite3.connect(db_path(context.index_dir))
    corrupted = sum(
        connection.execute(
            "UPDATE chunks SET fingerprint = ? WHERE owner_type = 'item' AND owner_id = ?",
            ("0" * 64, victim),
        ).rowcount
        for victim in hogs
    )
    connection.commit()
    connection.close()
    assert corrupted > 8, "the corrupted set must exceed one window, or the refill is untested"

    page = search(token, context, limit=1)
    served = [r.item_id for r in page.results]

    assert served and served[0] in tail, (
        f"the page must refill past the excluded head, not come back empty: {served}"
    )
    assert page.truncated is True and page.cursor is not None
    # BOUNDED BY THE ROWS THAT EXIST, which is what stops the count being accumulated across
    # the doublings: every window is a PREFIX of the same ranking, so the deepest already holds
    # every exclusion the shallower ones saw, and summing them counts rows once per doubling.
    # Measured: accumulating puts the total ABOVE the number of rows actually corrupted.
    assert 8 <= page.index.corrupt_chunks_excluded <= corrupted, (
        f"excluded {page.index.corrupt_chunks_excluded} of {corrupted} corrupted rows"
    )

    rest = search(token, context, limit=1, cursor=page.cursor)
    later = [r.item_id for r in rest.results]
    assert later and later[0] in tail and later != served, (
        "the next page continues the ranking rather than repeating it"
    )
    assert set(served + later) == set(tail), "the two pages reassemble the survivors"


def test_an_unknown_topic_is_refused_even_when_the_vocabulary_is_empty(
    context: QueryContext,
) -> None:
    """Spec §3.7.5: *los topics no se inventan desde el texto del query.*

    The guard read `if filters.topics and context.vocab`, so an EMPTY vocabulary disabled it
    entirely — and an empty vocabulary is the state in which EVERY topic is unknown, which
    makes it the one case the check exists for. Measured before the fix: a filter naming a
    topic that exists nowhere returned `0 results` with exit 0, indistinguishable from a topic
    that genuinely has no matches. A typo answered as a fact about the corpus.

    The two states are asserted apart, because «refuses when the vocabulary is empty» is only
    half a rule: a KNOWN topic must still be accepted when the vocabulary holds it.

    Seen red before the fix: the first call returned a `SearchResponse`.
    """
    bare = replace(context, vocab=[])

    with pytest.raises(ValueError, match="El vocabulario está vacío") as empty:
        search("Quillfeather", bare, filters=SearchFilters(topics=("no-such-topic",)))
    assert "no-such-topic" in str(empty.value)
    assert "Los válidos son" not in str(empty.value), (
        "an empty list of valid slugs reads as «none matched», which is the wrong diagnosis"
    )

    with pytest.raises(ValueError, match="no-such-topic"):
        search("Quillfeather", context, filters=SearchFilters(topics=("no-such-topic",)))

    known = context.vocab[0].slug
    assert search("Quillfeather", context, filters=SearchFilters(topics=(known,))) is not None


# ---------------------------------------------------------------------------
# Review round 2 — the refill deepened ONE plane and terminated on it
# ---------------------------------------------------------------------------


def _without(store: dict[str, Item], gone) -> dict[str, Item]:
    """The store after items were deleted and nobody reindexed — the index still holds them."""
    return {k: v for k, v in store.items() if k not in set(gone)}


def test_a_profile_only_page_refills_past_owners_the_store_no_longer_holds(
    context: QueryContext,
) -> None:
    """A handle lives in every profile and in no chunk, so this query is answered by ONE plane.

    Round 1 taught the refill to deepen the CHUNK window. It never taught it to deepen the
    PROFILE one: `search_profiles(query, needed)` asked for a fixed depth, and the loop's
    termination read `len(candidates) == previous` — the chunk window, which for a profile-only
    query is empty at every depth and therefore never grows. So the FIRST shortfall the profile
    plane produced ended the search.

    Delete the three top-ranked owners without reindexing — `_append_profile_candidates` skips
    an id the store no longer holds — and, before the fix, `limit=1` answered `results: []`,
    `truncated: false`, `cursor: null` over nine owners that were still there. Same shape as
    round 1's finding, reached through the plane round 1 did not touch.

    Seen red before the fix: `results == []`.
    """
    store = context.store
    whole = search("vgonpa", context, limit=50)
    order = [r.item_id for r in whole.results]
    assert len(order) > 4, "the profile plane must rank several owners, or nothing is tested"
    assert all(not r.matches for r in whole.results), "a handle is never a citable chunk match"

    shrunk = replace(context, store=_without(store, order[:3]))

    page = search("vgonpa", shrunk, limit=1)

    assert [r.item_id for r in page.results] == [order[3]], (
        "the page must reach past the deleted head instead of reporting the corpus empty"
    )
    assert page.truncated is True and page.cursor is not None


def test_paging_reassembles_the_ranking_when_both_planes_lose_candidates(
    context: QueryContext,
) -> None:
    """ONE ordering, whatever page you slice it at — the cursor's whole promise.

    `needed = offset + limit + 1`, so every page materialises its own candidate set, and both
    planes drop rows between that set and the answer. With the refill keyed on one plane the
    WALK stopped early: measured on this fixture with `k03` and `k07` deleted, paging `vgonpa`
    at `limit=1` yielded seven owners and then handed back no cursor, while a single `limit=50`
    call returned ten. Three valid items were unreachable by paging — omitted, not merely
    reordered.

    Asserted as the round trip a consumer actually performs: walk the cursor to exhaustion and
    require the concatenation to EQUAL the whole ranking. Equality catches an omission, a
    duplicate and a reorder in one assertion, which three separate counts would not.

    Seen red before the fix: seven of ten, no cursor.
    """
    shrunk = replace(context, store=_without(context.store, ("k03", "k07")))
    whole = [r.item_id for r in search("vgonpa", shrunk, limit=50).results]
    assert len(whole) > 6, "the ranking must outlast a few pages, or nothing is tested"

    walked: list[str] = []
    cursor: str | None = None
    for _ in range(len(whole) + 5):
        page = search("vgonpa", shrunk, limit=1, cursor=cursor)
        walked += [r.item_id for r in page.results]
        cursor = page.cursor
        if cursor is None:
            break

    assert cursor is None, "the walk must terminate rather than page forever"
    assert walked == whole, f"paging must reassemble the ranking: {walked} != {whole}"


# ---------------------------------------------------------------------------
# Review round 3 — the walk invariant, on a corpus that populates BOTH planes
# ---------------------------------------------------------------------------

_MIXED_HANDLE = "pauthor"
_MIXED_TOKEN = "Zephyrbloom"


def _mixed_plane_corpus(size: int, chunked: int) -> dict[str, Item]:
    """`size` items sharing one handle; the first `chunked` also carry a fetched ARTICLE BODY.

    The handle reaches `profile_text` and never a chunk; the body reaches `chunks` and never
    the profile. So one query lights BOTH candidate planes, which is the state a single-plane
    fixture cannot produce and the state every pagination defect in this file has lived in.
    """
    store: dict[str, Item] = {}
    for index in range(size):
        ident = f"p{index:03d}"
        content = None
        if index < chunked:
            body = " ".join(f"{_MIXED_TOKEN} parrafo {n} de {ident}." for n in range(40))
            content = Content(
                fetched_at=datetime(2026, 1, 2, tzinfo=UTC),
                sources=[
                    ContentSourceSuccess(
                        kind="external_article",
                        url=f"https://example.test/{ident}",
                        title=f"Articulo {ident}",
                        text=body,
                    )
                ],
            )
        store[ident] = Item(
            id=ident,
            source="bookmark",
            url=f"https://x.com/{_MIXED_HANDLE}/status/{ident}",
            author=Author(handle=_MIXED_HANDLE, name="P Author"),
            text=f"Nota {ident} del autor.",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
            content=content,
            enriched=Enrichment(
                summary=f"Resumen de {ident}.",
                topics=[],
                primary_topic=None,
                enriched_at=datetime(2026, 1, 3, tzinfo=UTC),
                model="m",
                executor="manual",
            ),
        )
    return store


def _walk(query: str, context: QueryContext, *, limit: int) -> list[str]:
    """Every page the cursor reaches, concatenated — the round trip a consumer performs."""
    walked: list[str] = []
    cursor: str | None = None
    for _ in range(200):
        page = search(query, context, limit=limit, cursor=cursor)
        walked += [result.item_id for result in page.results]
        cursor = page.cursor
        if cursor is None:
            return walked
    raise AssertionError("the walk did not terminate")


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_a_walk_across_both_candidate_planes_equals_the_unpaged_ranking(
    tmp_path: Path, limit: int
) -> None:
    """THE INVARIANT, on a shape that DISCRIMINATES — which the first draft of this test did not.

    Both planes are lit at once: `p000`/`p001` carry a fetched article body and reach `chunks`,
    every item shares a handle and reaches `profile_text`, and one query hits both. Five owners
    are then deleted from the store without reindexing, so the index still offers ids the
    answer cannot use — the state in which the candidate window and the answer set diverge.

    THE SHAPE WAS SEARCHED FOR, NOT GUESSED. A first version of this test seeded four chunked
    items and no deletions; it passed against the PRE-FIX implementation, which makes it no
    test at all (rule 1). A sweep over corpus shapes, run against `9da71f9`'s `_materialise`,
    produced this one — and it fails there exactly as the review reported, `p008` among the
    omitted:

        whole (11): p000 p003 p004 p005 p007 p008 p011 p012 p013 p014 p015
        walked (4): p000 p003 p004 p005

    NO CORRUPTION IS STAGED, and that omission is deliberate. The shape was found with three
    ids corrupted; removing them changes nothing — measured, identical `whole` and identical
    `walked` — because those ids own no chunk row, so the `UPDATE` touches nothing. A step that
    looks like it exercises the fingerprint path while exercising nothing is worse than its
    absence; corruption has its own regressions above.

    ASSERTED AS EQUALITY against one unpaged call. Equality catches an omission, a duplicate
    and a reorder together; «no duplicates» and «same length» would each pass on the other two.
    Parametrised over three page sizes because `needed = offset + limit + 1` — a defect visible
    at one size only is one a single-size test would call fixed.
    """
    store = _mixed_plane_corpus(16, chunked=2)
    data = tmp_path / "data"
    _persist(data, store, [], {})
    _build(data)

    live = {k: v for k, v in store.items() if k not in {"p001", "p002", "p006", "p009", "p010"}}
    context = _context(data, live, [], {})
    query = f"{_MIXED_TOKEN} {_MIXED_HANDLE}"

    results = search(query, context, limit=200).results
    whole = [result.item_id for result in results]
    assert any(r.matches for r in results), "the chunk plane must be lit"
    assert any(not r.matches for r in results), "and the profile plane too"
    assert len(whole) > 3 * limit, "the ranking must outlast several pages, or nothing is tested"

    assert _walk(query, context, limit=limit) == whole


# ---------------------------------------------------------------------------
# Review round 4 — profile candidates must not end the chunk refill
# ---------------------------------------------------------------------------


def _one_marker_corpus(seed: Item, size: int, marker: str) -> dict[str, Item]:
    """`size` copies of one item whose TEXT is the marker and which has nothing else.

    The text reaches the `post` surface (the chunk plane) AND `profile_text` (the profile
    plane), so a single query lights both planes over the SAME owners — the population the
    review named, and the one every earlier fixture in this file missed by lighting the two
    planes over DISJOINT owners.
    """
    return {
        f"p{index:03d}": seed.model_copy(
            update={"id": f"p{index:03d}", "text": marker, "content": None, "enriched": None}
        )
        for index in range(size)
    }


def _leading_chunk_hits(context: QueryContext, query: str, owners: int):
    """The raw window `search_owners` materialises for a page of `owners` — before grouping."""
    opened = index_store.open_for_query(
        context.index_dir, context.items_path, context.vocab_path, context.topics_path
    )
    try:
        return opened.lexical.search_owners(query, owners, filters=None)[0]
    finally:
        opened.close()


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_profile_candidates_do_not_end_the_chunk_refill_before_deeper_owners(
    tmp_path: Path, corpus, limit: int
) -> None:
    """The population the previous three rounds all missed (review round 4, HIGH).

    `_append_profile_candidates` filled `grouped` BEFORE `len(grouped) >= needed` ended the
    refill. So when the leading chunk candidates fail verification, matching profiles can
    satisfy the quota while deeper VALID chunk owners are still unreached — and because a
    larger offset materialises a deeper chunk window, each page slices a DIFFERENT ranking.
    The append-only claim the loop rested on is false for exactly this population.

    Staged as the review staged it: thirty items whose text is one marker, so the chunk plane
    and the profile plane rank the SAME owners; then the eight raw hits that `search_owners`
    returns for a `limit=1` page are corrupted, so the whole leading window is excluded.

    Measured before the fix, and this is the shape no earlier fixture here could produce:

        whole : p008 p009 … p029 p000 … p007
        paged : p000 p009 … p029 p000 … p007
        p008 unreachable, p000 served twice

    Note what the first page got WRONG: it served `p000`, a profile-only candidate, when the
    ranking's first result is `p008`, a real chunk match. A page that answers with the weaker
    plane while the stronger one had an unexamined answer is not a pagination detail.

    Asserted as equality against one unpaged call, plus the first page against the ranking's
    head — the second is what names WHICH defect this is when it fails.
    """
    store, _vocab, _pages = corpus
    marker = "Unicornmarker"
    live = _one_marker_corpus(store["k01"], 30, marker)
    data = tmp_path / "data"
    _persist(data, live, [], {})
    _build(data)
    context = _context(data, live, [], {})

    leading = _leading_chunk_hits(context, marker, 2)
    assert leading, "the chunk plane must answer, or nothing is tested"
    connection = open_index(db_path(context.index_dir))
    with connection:
        for hit in leading:
            connection.execute(
                "UPDATE chunks SET fingerprint = ? WHERE chunk_id = ?", ("bad", hit.chunk_id)
            )
    connection.close()

    whole = [result.item_id for result in search(marker, context, limit=200).results]
    assert len(whole) == 30, whole
    assert whole[0] not in {hit.owner_id for hit in leading}, "the head must be a SURVIVING owner"

    first = search(marker, context, limit=limit)
    assert [r.item_id for r in first.results] == whole[:limit], (
        "the first page served the profile plane while a verified chunk owner was unexamined"
    )
    assert first.results[0].matches, "and the ranking's head is a real chunk match"

    assert _walk(marker, context, limit=limit) == whole
