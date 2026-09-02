# tests/test_knowledge_lexical.py
"""The lexical retriever over the PERSISTED schema (Plan 02 §1, §4.2, steps 12, 12b, 16, 17).

THIS FILE IS WHERE `tests/test_knowledge_lexical_memory.py` WENT (m-vii). Plan 02 §9 deletes
`lexical_memory.py` — the `:memory:` wrapper — and the plan's first draft said "and its
test", which would have left `tests/fixtures/knowledge_ranking.json` with nobody reading it:
the chunker sweep of §7 could then change the ranking and nothing would go red, which is the
exact hole the fixture exists to close. Every assertion of the old file is here, unchanged in
substance, now aimed at `LexicalIndex`.

WHAT MAKES THE MOVE LEGITIMATE, rather than a fixture quietly re-pointed at a new
implementation: the scorer did not change. `lexical_fts.py` owns the tokenizer, the column
set, the connective and the tie-break, and both the in-memory baseline and the persisted
index compose their DDL from it. What changed is where the database lives — `:memory:` ->
`data/index/knowledge.db` — which is precisely what Plan 01 §5.3 promised would be the only
difference. So the fixture still pins ONE scorer across time, and it is NOT regenerated.

FILTERS ARE `WHERE` CLAUSES, NEVER QUERY TEXT (spec §5.3). Appending a topic or a date to the
query string would let the filter's own words compete for bm25 weight against the user's
terms — a filter that changes the RANKING is not a filter. The eight filters of spec §7.2 are
therefore asserted three ways each where it matters: that they change the result set, that
`EXPLAIN QUERY PLAN` shows the backend using an index rather than scanning, and that the
number of rows the scorer sees actually falls (m3 — asserting the SQL string only proves we
wrote that string).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge.chunking import ChunkerParams, chunk_surfaces
from xbrain.knowledge.contracts import SearchFilters
from xbrain.knowledge.index_schema import open_index, open_memory_index
from xbrain.knowledge import lexical
from xbrain.knowledge.lexical import LexicalIndex
from xbrain.knowledge.lexical_fts import FTS_CONNECTIVE, FTS_TOKENIZE
from xbrain.knowledge.models import KnowledgeChunk, KnowledgeSurface, Locator
from xbrain.knowledge.surfaces import item_surfaces
from xbrain.models import Item

FIXTURES = Path(__file__).parent / "fixtures"

# The parameters the ranking fixture was built with, PINNED HERE and passed explicitly (M7).
# Plan 02 §7 sweeps `target x overlap` and bumps CHUNKER_VERSION; if this read the module
# constant, that sweep would break the fixture it exists to protect, and the comfortable fix
# would be to regenerate it — at which point it pins nothing.
PINNED_CHUNKER_PARAMS = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)

# AND THE VERSION, for the same reason and through the same door — the gap Plan 02 §7's M7
# note left open. It made the PARAMS an argument so the sweep could not move the fixture, and
# then mandated a `CHUNKER_VERSION` bump when the sweep changed them. But `chunk_id` ENDS in
# the chunker version, so the bump renames every id the fixture pins and breaks it just as
# surely — from the very section that promised it would stay green without regenerating.
#
# Pinning the version loses nothing the fixture exists to protect: the version is a label in
# the id and contributes no character to the ranking, while the SPANS come from the params,
# which are pinned beside it. Change the chunking and this still goes red.
PINNED_CHUNKER_VERSION = "xbrain-knowledge-chunker/v1"


# Every item genuinely has a creation and a capture instant, so `items` declares both NOT
# NULL and the helper below supplies them where the test does not care which they are.
_STAMP = {
    "created_at": datetime(2025, 1, 1, tzinfo=timezone.utc),
    "captured_at": datetime(2025, 1, 2, tzinfo=timezone.utc),
}


def _index() -> LexicalIndex:
    return LexicalIndex(open_memory_index())


def _corpus_chunks(
    *,
    params: ChunkerParams = PINNED_CHUNKER_PARAMS,
    chunker_version: str = PINNED_CHUNKER_VERSION,
) -> list[KnowledgeChunk]:
    """The fixture corpus, chunked with parameters the CALLER names.

    Both are arguments defaulting to the PINNED values, so the characterization test gets the
    Plan 01 provisional whatever the module defaults become, and the winner's test asks for
    the module defaults explicitly. Neither reads a constant by accident.
    """
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    chunks: list[KnowledgeChunk] = []
    for item_raw in raw["items"].values():
        item = Item.model_validate(item_raw)
        chunks += list(
            chunk_surfaces(
                item_surfaces(item),
                params=params,
                url=item.url,
                chunker_version=chunker_version,
            )
        )
    return chunks


def _chunk(chunk_id: str, text: str, surface_id: str = "item:x:post:0", **kwargs) -> KnowledgeChunk:
    fields: dict = {
        "chunk_id": chunk_id,
        "surface_id": surface_id,
        "owner_type": "item",
        "owner_id": "x",
        "surface_type": "post",
        "text": text,
        "chunk_index": 0,
        "char_start": 0,
        "char_end": len(text),
        "origin": "source",
        "trust_class": "primary_source",
        "derived": False,
        "fingerprint": "f" * 64,
    }
    fields.update(kwargs)
    return KnowledgeChunk(**fields)


# ---------------------------------------------------------------------------
# U-4 (round 07) — a late SQLite failure at query time is the rebuild advice, never raw
# ---------------------------------------------------------------------------


def test_a_column_that_vanishes_after_the_door_is_the_rebuild_advice_not_a_traceback() -> None:
    """Gate Codex F3: `_fetch` re-raised every `OperationalError` that was not an FTS parse
    error, so a schema the door had not verified (columns, until U-4) reached the operator
    as `OperationalError: no such column: surfaces.attribution_name` inside a 60-line
    traceback. The door verifies columns now; this pins the OTHER half — whatever SQLite
    raises past the door that is not the parser saying «I could not parse your query» is
    a base this code cannot read, and it ends in the one sentence that names the rebuild.

    Staged AFTER the door: the column is dropped on the open connection, which is the only
    way a query can meet it. Seen red on `9dfa34e`: a raw `sqlite3.OperationalError`.
    """
    from xbrain.knowledge.index_schema import REBUILD_ADVICE, IndexIncompatibleError

    index = _index()
    index.add([_chunk("c1", "marrowgate body")])
    index.connection.execute("ALTER TABLE surfaces DROP COLUMN attribution_name")

    with pytest.raises(IndexIncompatibleError, match="no such column") as refused:
        index.search("marrowgate", 5)
    assert REBUILD_ADVICE in str(refused.value)
    # The parser's own family still degrades to «no results» on a sound base — a query the
    # parser refuses is data that held nothing, not a base this code cannot read (C-2).
    sound = _index()
    sound.add([_chunk("c1", "marrowgate body")])
    assert sound.search("NEAR(a b", 5) == ()


# ---------------------------------------------------------------------------
# The engine — moved verbatim in substance from test_knowledge_lexical_memory.py
# ---------------------------------------------------------------------------


def test_fts5_and_bm25_are_available_in_this_interpreter() -> None:
    """A precondition, asserted rather than assumed.

    If FTS5 were compiled out of the local sqlite, every ranking test below would fail with
    an opaque `sqlite3.OperationalError` from deep inside the index. Failing here, by name,
    is the difference between "your Python has no FTS5" and "the ranking changed".
    """
    connection = open_memory_index()
    connection.execute("INSERT INTO chunks_fts (rowid, text, title) VALUES (1, 'hola mundo', '')")
    rows = connection.execute(
        "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?", ("hola",)
    ).fetchall()
    assert rows and isinstance(rows[0][1], float)


def test_the_tokenizer_folds_diacritics() -> None:
    """`unicode61 remove_diacritics 2` — a Spanish query must reach an unaccented body.

    The corpus is bilingual and the queries are written in Spanish; without folding,
    "evaluacion" and "evaluación" are two different terms and half the golden set's own
    queries would miss. ONE constant, so the persisted index and the fixture cannot drift.
    """
    assert "remove_diacritics 2" in FTS_TOKENIZE
    index = _index()
    index.add([_chunk("c1", "La evaluación automática de agentes")])
    assert [hit.chunk_id for hit in index.search("evaluacion", limit=5)] == ["c1"]


def test_there_is_no_stemming_and_the_limit_is_documented_not_hidden() -> None:
    """FTS5 has no multilingual stemmer, and an English one would wreck the Spanish half.

    So "agent" does not match "agents" — a REAL limitation of the lexical baseline, and one
    the vector layer of Plan 03 has to beat. Pinning it as a test is what keeps the published
    baseline honest: a limit nobody wrote down gets quietly attributed to the corpus instead
    of to the tokenizer.
    """
    index = _index()
    index.add([_chunk("c1", "evaluating agents in production")])
    assert index.search("agent", limit=5) == ()
    assert index.search("agents", limit=5)


# ---------------------------------------------------------------------------
# 16 — stable tie-break
# ---------------------------------------------------------------------------


def test_ties_break_on_chunk_id_ascending() -> None:
    """Step 16 / spec §3.7.8: the order is STABLE under ties.

    Two chunks with identical text get an identical bm25 score, and sqlite's row order for a
    tie is an implementation detail — so without an explicit second key the ranking would
    differ between rebuilds of the same data. Seen red by dropping `, chunk_id ASC` from
    `rank_order`: the pair comes back in insertion order and reverses when inserted the other
    way round.
    """
    text = "identical body text for both chunks"
    forward = _index()
    forward.add([_chunk("b:2", text), _chunk("a:1", text)])
    backward = _index()
    backward.add([_chunk("a:1", text), _chunk("b:2", text)])
    assert [h.chunk_id for h in forward.search("identical", limit=5)] == ["a:1", "b:2"]
    assert [h.chunk_id for h in backward.search("identical", limit=5)] == ["a:1", "b:2"]


def test_the_query_is_escaped_not_interpolated() -> None:
    """A quote or an FTS operator in a query must not become syntax.

    Spec §9.3 asks for a stable validation error rather than a crash, and the golden set
    contains literal queries like `@simonw` and `11.37%` whose punctuation FTS5 would
    otherwise read as operators.
    """
    index = _index()
    index.add([_chunk("c1", "the handle is @simonw here")])
    assert [h.chunk_id for h in index.search("@simonw", limit=5)] == ["c1"]
    assert index.search('"unbalanced', limit=5) == ()
    assert index.search("NEAR(", limit=5) == ()


def test_an_empty_query_is_a_validation_error() -> None:
    """Step 14 / spec §9.3: an empty query is a stable error, not an empty result set.

    Empty results would say "nothing in your corpus matches", which is a claim about the
    corpus; the truth is that nothing was asked.
    """
    index = _index()
    with pytest.raises(ValueError, match="vacía"):
        index.search("   ", limit=5)


def test_a_non_positive_limit_is_a_validation_error() -> None:
    """Spec §9.3 / Plan 02 §11: `--limit 0` or negative is a validation error.

    `LIMIT 0` returns nothing, which would read as "your corpus has no match" — a claim about
    the corpus made by a malformed request.
    """
    index = _index()
    with pytest.raises(ValueError, match="limit"):
        index.search("anything", limit=0)


# ---------------------------------------------------------------------------
# 17 — the characterization fixture, MOVED here from the deleted module's test
# ---------------------------------------------------------------------------

RANKING_QUERIES = (
    "Quillfeather",
    "Marrowgate protocol",
    "Bramblewick",
    "Cindervale checklist",
    "Thistledown",
    "Pelicanine",
    # Two distinctive terms that never co-occur in one chunk (M3). The six above CANNOT
    # discriminate the connective — their rankings are byte-identical under a conjunction and
    # a disjunction — so a change to the query semantics, which moves every recall number
    # downstream, would pass this fixture in silence.
    "Marrowgate Zephyrine",
)


def test_ranking_matches_the_characterization_fixture() -> None:
    """Step 17: the SAME pinned top-10 the Plan 01 baseline produced, on the persisted schema.

    THIS IS THE ASSERTION FROM PLAN 01 §7 STEP 25, MOVED (m-vii) — not a new one, and the
    fixture is NOT regenerated. It is green because the scorer is unchanged: same tokenizer,
    same indexed columns, same `bm25()`, same explicit tie-break, all composed from
    `lexical_fts.py`. Only the connection string moved.

    Built with `PINNED_CHUNKER_PARAMS` passed EXPLICITLY (M7), so Plan 02 §7's sweep — which
    changes the module constant — cannot move this fixture. The sweep winner is pinned
    separately under its own `CHUNKER_VERSION`, and the report compares the two.

    Red whenever chunking, tokenization, the indexed column set, the ORDER BY or the fixture
    corpus change. Those are exactly the five things that silently alter every downstream
    recall number.
    """
    expected = json.loads((FIXTURES / "knowledge_ranking.json").read_text(encoding="utf-8"))
    index = _index()
    index.add(_corpus_chunks())
    actual = {
        query: [hit.chunk_id for hit in index.search(query, limit=10)] for query in RANKING_QUERIES
    }
    assert actual == expected["rankings"]


def test_the_ranking_fixture_records_the_parameters_it_was_built_with() -> None:
    """A pinned ranking with no record of its parameters is unreproducible.

    Without them, the day the test goes red nobody can tell whether the ranking changed or
    the parameters did — and the fixture would be regenerated to make it green, losing the
    signal entirely.
    """
    fixture = json.loads((FIXTURES / "knowledge_ranking.json").read_text(encoding="utf-8"))
    assert fixture["chunker_params"] == {
        "target": PINNED_CHUNKER_PARAMS.target,
        "max_chars": PINNED_CHUNKER_PARAMS.max_chars,
        "overlap": PINNED_CHUNKER_PARAMS.overlap,
        "min_chars": PINNED_CHUNKER_PARAMS.min_chars,
    }
    assert fixture["tokenize"] == FTS_TOKENIZE
    assert fixture["connective"] == FTS_CONNECTIVE


def test_a_hit_resolves_back_to_its_item_and_surface() -> None:
    """Spec §3.3: reverse resolution from a chunk to its surface and owner.

    A ranked id nobody can resolve is not evidence, it is a number.
    """
    index = _index()
    index.add(_corpus_chunks())
    (hit,) = index.search("Quillfeather", limit=1)
    assert hit.owner_id in {"k03", "k12"}
    assert hit.surface_type in {"external_article", "user_note"}
    assert "Quillfeather" in hit.excerpt


def test_a_surface_with_no_chunks_contributes_nothing() -> None:
    """An empty index answers a query with no results, not with an error."""
    empty_surface = KnowledgeSurface(
        surface_id="item:z:post:0",
        owner_type="item",
        owner_id="z",
        surface_type="post",
        text="",
        origin="source",
        trust_class="primary_source",
        derived=False,
        locator=Locator(kind="item_text"),
        fingerprint="a" * 64,
    )
    index = _index()
    index.add(chunk_surfaces((empty_surface,)))
    assert index.search("anything", limit=5) == ()


# ---------------------------------------------------------------------------
# M3 — the query semantics: bm25 ranks, the connective does not pre-empty
# ---------------------------------------------------------------------------


def test_a_question_shaped_query_is_not_emptied_by_its_own_length() -> None:
    """M3: ANDing every term makes a long question demand that ALL of it sit in ONE chunk.

    Measured on the real corpus BEFORE the connective changed: 18 of the 21 scorable cases
    received not a single row. WHAT IS ASSERTED IS WHAT THE CHANGE GUARANTEES: the chunk
    holding the distinctive term is RETRIEVED — not that it ranks first, which is bm25's job
    and depends on corpus statistics.
    """
    chunks = _corpus_chunks()
    index = _index()
    index.add(chunks)
    wanted = {c.chunk_id for c in chunks if "Marrowgate" in c.text}
    assert len(wanted) == 1, "the fixture must hold the term in exactly one chunk"

    hits = index.search("¿Qué dice el protocolo Marrowgate sobre los umbrales?", limit=10)

    assert hits, "a question with one matching term must not come back empty"
    assert wanted <= {hit.chunk_id for hit in hits}, (
        f"the chunk holding the distinctive term was not retrieved: {[h.chunk_id for h in hits]}"
    )


def test_a_rare_term_outranks_a_term_that_matches_almost_everything() -> None:
    """The property that makes disjunction safe, pinned: IDF does the discriminating.

    The objection to OR is that any one common word drags the whole corpus in. It does bring
    it in as CANDIDATES — and bm25 then ranks it last, because a term present in nearly every
    chunk carries almost no inverse document frequency.
    """
    index = _index()
    index.add(_corpus_chunks())

    hits = index.search("Zephyrine quality", limit=10)

    assert len(hits) > 1, "the common term must genuinely widen the candidate set"
    assert "Zephyrine" in hits[0].excerpt, (
        f"a term matching almost everything outranked a term matching one chunk: {hits[0]}"
    )


def test_terms_are_still_data_and_never_syntax() -> None:
    """The connective changed; the quoting did not, and it is the security property."""
    from xbrain.knowledge.lexical_fts import match_expression

    assert match_expression("@simonw 11.37%") == '"@simonw" OR "11.37%"'
    assert match_expression('NEAR( a"b') == '"NEAR" OR "a" OR "b"'
    assert match_expression("   ") is None

    index = _index()
    index.add([_chunk("c1", "Contacta con @simonw sobre el 11.37% de mejora")])
    assert index.search("@simonw 11.37%", limit=5)


def test_idf_is_relative_to_THIS_corpus_and_that_limit_is_declared() -> None:
    """The declared limit of the disjunction, pinned as a measurement rather than a warning.

    IDF is a property of the INDEXED CORPUS, not of the language, so a word that is a
    function word to a reader can still be rare to the index. Measured 2026-09-01 with the
    SHIPPED chunker (v2, `800/0`) on `data/items.json` sha256 `f76341a3…`: `el` sits in 1 of
    the 49 fixture chunks (2.0 %) and in **6,070 of the 22,286** real-corpus chunks
    (**27.2 %**), so the fixture ranks it high and the real corpus does not — both bm25
    behaving correctly on the corpus it was given. This is what a vector layer is for
    (Plan 03).

    THE ASSERTION BELOW CHUNKS AT v1 (`_corpus_chunks()` defaults to `PINNED_CHUNKER_PARAMS`),
    which is why it says 43 and not 49; the ratio is 1-in-something either way and the point
    is the CONTRAST with the real corpus. The prose used to quote `5,748 of 18,319 (31.4 %)`
    — the v1 figure, left standing after this branch changed the chunker to v2 (F-4).
    """
    index = _index()
    index.add(_corpus_chunks())

    def document_frequency(term: str) -> int:
        return index.connection.execute(
            "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{term}"',)
        ).fetchone()[0]

    assert document_frequency("el") == 1, (
        "the fixture corpus is English-dominated, which is WHY a Spanish function word is "
        "distinctive here; if this moved, the limit above is being measured on a different "
        "population (CLAUDE.md rule 2)"
    )
    assert document_frequency("quality") > document_frequency("Marrowgate")


# ---------------------------------------------------------------------------
# 12 / 12b — the EIGHT filters of spec §7.2, pushed into SQL before scoring
# ---------------------------------------------------------------------------


@pytest.fixture()
def filtered_index() -> LexicalIndex:
    """Two items that differ on EVERY filter axis and share the query term.

    Sharing the term is the point: with both matching the text, a filter that does nothing
    leaves both in the result, so every assertion below can only pass because the filter ran.
    """
    index = _index()
    index.add(
        [
            _chunk(
                "c-old",
                "the shared marrowgate term appears here",
                surface_id="item:old:post:0",
                owner_id="old",
                origin="source",
                surface_type="post",
            )
        ],
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        source="bookmark",
    )
    index.add(
        [
            _chunk(
                "c-new",
                "the shared marrowgate term appears here too",
                surface_id="item:new:image_description:0",
                owner_id="new",
                origin="vlm",
                surface_type="image_description",
                derived=True,
                trust_class="machine_extracted",
            )
        ],
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        source="own_tweet",
    )
    index.set_item_metadata(
        "old",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        source="bookmark",
        author_handle="karpathy",
        author_name="Andrej",
        url="https://x.com/karpathy/1",
        captured_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        topics=("agent-evaluation",),
        content_kinds=("external_article",),
        surface_types=("post", "external_article"),
    )
    index.set_item_metadata(
        "new",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        source="own_tweet",
        author_handle="vgonpa",
        author_name="Victor",
        url="https://x.com/vgonpa/2",
        captured_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        topics=("ai-policy",),
        content_kinds=("x_video",),
        surface_types=("post", "image_description"),
    )
    return index


def _owners(index: LexicalIndex, **filter_kwargs) -> set[str]:
    filters = SearchFilters(**filter_kwargs)
    return {hit.owner_id for hit in index.search("marrowgate", limit=10, filters=filters)}


def test_filter_created_from_and_to(filtered_index: LexicalIndex) -> None:
    """Filter 1 & 2: the date window, on the DENORMALISED column, before the MATCH."""
    assert _owners(filtered_index) == {"old", "new"}
    assert _owners(filtered_index, created_from=datetime(2026, 1, 1, tzinfo=timezone.utc)) == {
        "new"
    }
    assert _owners(filtered_index, created_to=datetime(2025, 6, 1, tzinfo=timezone.utc)) == {"old"}


def test_filter_source(filtered_index: LexicalIndex) -> None:
    """Filter 3: `--mine` maps to `source=own_tweet` (spec §7.2)."""
    assert _owners(filtered_index, source="own_tweet") == {"new"}
    assert _owners(filtered_index, source="bookmark") == {"old"}


def test_filter_author(filtered_index: LexicalIndex) -> None:
    """Filter 4: a JOIN onto `items`, not a substring of the query."""
    assert _owners(filtered_index, author="karpathy") == {"old"}
    assert _owners(filtered_index, author="nobody") == set()


def test_filter_topics(filtered_index: LexicalIndex) -> None:
    """Filter 5: `EXISTS` over `item_topics` — the assignment, never the query text.

    Spec §3.7.5: *topics and filters are not invented from the query text; they come from the
    store/vocabulary.*
    """
    assert _owners(filtered_index, topics=("agent-evaluation",)) == {"old"}
    assert _owners(filtered_index, topics=("ai-policy",)) == {"new"}


def test_filter_content_kinds(filtered_index: LexicalIndex) -> None:
    """Filter 6 (m2): `item_content_kinds` is a table this plan introduces.

    It was declared in `SearchFilters` by Plan 01 with no column behind it, which is why the
    evaluation harness had to report cases using it as UNMEASURED rather than 0.0. Seen red
    by dropping the table: the filter then matches nothing and both assertions fail.
    """
    assert _owners(filtered_index, content_kinds=("external_article",)) == {"old"}
    assert _owners(filtered_index, content_kinds=("x_video",)) == {"new"}


def test_filter_origins(filtered_index: LexicalIndex) -> None:
    """Filter 7: `chunks.origin` — a chunk-level property, so no join at all."""
    assert _owners(filtered_index, origins=("vlm",)) == {"new"}
    assert _owners(filtered_index, origins=("source",)) == {"old"}


def test_filter_has_surfaces(filtered_index: LexicalIndex) -> None:
    """Filter 8 (m2): an aggregate `EXISTS` over `surfaces`, the other one with no column."""
    assert _owners(filtered_index, has_surfaces=("external_article",)) == {"old"}
    assert _owners(filtered_index, has_surfaces=("image_description",)) == {"new"}
    assert _owners(filtered_index, has_surfaces=("post",)) == {"old", "new"}


def test_every_declared_filter_is_actually_pushed_to_sql(filtered_index: LexicalIndex) -> None:
    """The TOTALITY half: no field of `SearchFilters` may be silently ignored.

    A filter the backend drops does not error — it returns MORE results, which looks like a
    permissive query rather than a broken one. So the eight declared fields are enumerated
    from the model and each is asserted to narrow the result set. Adding a ninth field to the
    frozen contract without wiring it here goes red.
    """
    narrowing = {
        "created_from": {"created_from": datetime(2026, 1, 1, tzinfo=timezone.utc)},
        "created_to": {"created_to": datetime(2025, 6, 1, tzinfo=timezone.utc)},
        "source": {"source": "own_tweet"},
        "author": {"author": "karpathy"},
        "topics": {"topics": ("ai-policy",)},
        "content_kinds": {"content_kinds": ("x_video",)},
        "origins": {"origins": ("vlm",)},
        "has_surfaces": {"has_surfaces": ("external_article",)},
    }
    assert set(narrowing) == set(SearchFilters.model_fields), (
        "a filter was added to the frozen contract without a test that it reaches SQL"
    )
    for name, kwargs in narrowing.items():
        assert _owners(filtered_index, **kwargs) == {"old"} or _owners(
            filtered_index, **kwargs
        ) == {"new"}, f"filter {name} did not narrow the result set"


def test_the_filter_is_applied_before_scoring_not_after(filtered_index: LexicalIndex) -> None:
    """Step 12 (m3): the plan asks for THREE pieces of evidence, and asserting the SQL string
    is none of them — that only proves we wrote that string.

    1. `EXPLAIN QUERY PLAN` shows sqlite using an index on the filtered column rather than
       scanning every chunk;
    2. the number of rows the scorer is handed actually FALLS;
    3. the filter changes the TOP-1, which is the user-visible consequence.

    PIECE 1 NAMES THE INDEX, and the previous version of it could not fail (F-1, rule 1). It
    asked for `any("chunks" in step and "SCAN" not in step)`, which is satisfied by
    `SEARCH chunks USING INTEGER PRIMARY KEY (rowid=?)` — the step an external-content FTS5
    join emits UNCONDITIONALLY, filter or no filter. Measured on this very fixture, the
    predicate was `True` with `SearchFilters()`, with `SearchFilters(source=...)` and with
    `SearchFilters(origins=...)` alike, so it distinguished nothing. So piece 1 must name an
    index that the unfiltered plan does NOT contain — and it must name one the planner cannot
    talk itself out of.

    IT NAMES `sqlite_autoindex_items_1`, VIA `author`, AND NOT `chunks_source` VIA `source`,
    BECAUSE A QUERY PLAN IS A COST DECISION AND COST DECISIONS ARE NOT PORTABLE. Measured on
    this same fixture across the two interpreters this project supports:

        filter      SQLite 3.51.2 (py3.13)              SQLite 3.50.4 (py3.12, CI)
        source      SEARCH chunks USING INDEX           (no such step — the plan is
                    chunks_source                        BYTE-IDENTICAL to unfiltered)
        origins     SEARCH chunks USING INDEX           (idem)
                    chunks_origin
        author      SEARCH items ... USING INDEX        SEARCH items USING INDEX
                    sqlite_autoindex_items_1            sqlite_autoindex_items_1

    On 3.50.4 the planner drives the MATCH and applies `source` as a RESIDUAL filter on the
    rowid lookup: the predicate still reaches the `WHERE`, it simply does not drive an index,
    and the plan it produces is the one MUTATING THIS MODULE TO FILTER AFTER SCORING produces
    on 3.51.2. An assertion that cannot separate the correct implementation from that mutant
    on the very interpreter CI runs is not a guard, and `ANALYZE` does not rescue it — it
    degrades both versions to a plain `SCAN chunks`.

    `author` is stable across both because its clause is an `EXISTS` subquery keyed on
    `items.item_id`, so reaching `items` by its primary key is STRUCTURAL rather than a
    cost-model preference: whatever strategy the planner picks, it must look the row up.

    Falsifiability is asserted rather than asserted-about: the unfiltered plan is required NOT
    to name that index, so the pair goes red if the `author` clause stops reaching the `WHERE`
    (seen red by mutation, on both interpreters). Pieces 2 and 3 stay on `source` — they read
    row counts and results, never a plan, so they are planner-agnostic by construction, and
    piece 2 is what catches the filter-moved-after-scoring mutant (scored rows 2 -> 2 under
    the mutant against 2 -> 1 correct, with the user-visible answer identical in both).
    """
    plan = filtered_index.explain("marrowgate", SearchFilters(author="karpathy"))
    assert any("sqlite_autoindex_items_1" in step for step in plan), plan
    unfiltered_plan = filtered_index.explain("marrowgate", SearchFilters())
    assert not any("sqlite_autoindex_items_1" in step for step in unfiltered_plan), unfiltered_plan

    unfiltered_rows = filtered_index.scored_row_count("marrowgate", SearchFilters())
    filtered_rows = filtered_index.scored_row_count("marrowgate", SearchFilters(source="own_tweet"))
    assert filtered_rows < unfiltered_rows

    top_unfiltered = filtered_index.search("marrowgate", limit=1)[0].owner_id
    top_filtered = filtered_index.search(
        "marrowgate", limit=1, filters=SearchFilters(source="own_tweet")
    )[0].owner_id
    assert top_unfiltered != top_filtered


def test_an_unknown_topic_is_rejected_by_the_index_not_silently_empty() -> None:
    """Step 13 belongs to the service, but the index must not INVENT a match either.

    A slug that is in no assignment simply narrows to nothing here; the service is what turns
    that into an error listing the valid slugs, because only the service knows the vocabulary.
    """
    index = _index()
    index.add([_chunk("c1", "marrowgate")])
    index.set_item_metadata("x", topics=("agent-evaluation",), **_STAMP)
    assert index.search("marrowgate", limit=5, filters=SearchFilters(topics=("nope",))) == ()


def test_a_topic_owned_chunk_matches_its_own_slug(filtered_index: LexicalIndex) -> None:
    """A `topic_note` of topic X satisfies `--topic X`, though it has no `item_topics` row.

    Without this branch the topic filter would exclude exactly the surfaces that ARE the
    topic, which reads as "the topic has no notes" — a claim about the corpus produced by the
    filter's own shape.
    """
    filtered_index.add(
        [
            _chunk(
                "t1",
                "the marrowgate note of the topic itself",
                surface_id="topic:ai-policy:topic_note:0",
                owner_type="topic",
                owner_id="ai-policy",
                surface_type="topic_note",
                origin="llm",
                trust_class="llm_synthesis",
                derived=True,
            )
        ]
    )
    owners = _owners(filtered_index, topics=("ai-policy",))
    assert owners == {"new", "ai-policy"}


def test_item_scoped_filters_exclude_topic_chunks(filtered_index: LexicalIndex) -> None:
    """A topic surface has no author, no date and no source, so it FAILS CLOSED on those.

    Including it would answer "posts by @karpathy" with a topic overview nobody wrote as
    @karpathy — the derived/primary confusion provenance exists to prevent.
    """
    filtered_index.add(
        [
            _chunk(
                "t1",
                "the marrowgate note",
                surface_id="topic:ai-policy:topic_note:0",
                owner_type="topic",
                owner_id="ai-policy",
                surface_type="topic_note",
                origin="llm",
                trust_class="llm_synthesis",
                derived=True,
            )
        ]
    )
    assert "ai-policy" not in _owners(filtered_index, author="karpathy")
    assert "ai-policy" not in _owners(filtered_index, source="own_tweet")
    assert "ai-policy" not in _owners(
        filtered_index, created_from=datetime(2020, 1, 1, tzinfo=timezone.utc)
    )


# ---------------------------------------------------------------------------
# Security — the query builder concatenates constants, never values
# ---------------------------------------------------------------------------


def test_placeholders_can_only_ever_be_commas_and_question_marks() -> None:
    """The one dynamically built SQL fragment is derived from an INTEGER.

    That is what makes the `# nosec B608` honest: the variadic `IN` clause is the shape
    sqlite3 cannot parameterise wholesale, so the placeholder GROUP has to be built — but it
    is built from a count, so no caller string can reach the statement.
    """
    from xbrain.knowledge.lexical import _placeholders

    for n in range(0, 8):
        assert set(_placeholders(n)) <= {"?", ","}
    assert _placeholders(3) == "?,?,?"


def test_a_filter_value_containing_sql_is_bound_not_interpolated() -> None:
    """A hostile filter value is DATA. Defence in depth, and cheap."""
    index = _index()
    index.add([_chunk("c1", "ordinary body text")])
    hits = index.search(
        "ordinary", limit=5, filters=SearchFilters(author="x'; DROP TABLE chunks; --")
    )
    assert hits == ()
    assert len(index) == 1, "the table is still there"


def test_the_index_never_writes_when_opened_read_only(tmp_path: Path) -> None:
    """Step 11: `search` opens the database read-only, so a write RAISES (spec §5.6).

    Not "we checked and it does not write" — the connection cannot. Seen red by opening it
    read-write: the insert succeeds and the query has repaired an index nobody asked it to.
    """
    path = tmp_path / "knowledge.db"
    writer = LexicalIndex(open_index(path, create=True))
    writer.add([_chunk("c1", "marrowgate body")])
    writer.connection.commit()
    writer.connection.close()

    reader = LexicalIndex(open_index(path, read_only=True))
    assert [h.chunk_id for h in reader.search("marrowgate", limit=5)] == ["c1"]
    with pytest.raises(sqlite3.OperationalError):
        reader.add([_chunk("c2", "another body")])


# ---------------------------------------------------------------------------
# The profile plane (spec §5.1.A)
# ---------------------------------------------------------------------------


def test_the_profile_plane_returns_items_not_citable_chunks() -> None:
    """Spec §5.1: the profile finds the item as a conceptual UNIT.

    It returns an item id and a score and NOTHING ELSE — no excerpt, no chunk id, no surface.
    A profile is a string nobody wrote (a tweet, a summary and three topic descriptions glued
    together), so returning it as evidence would present a machine's collage as something
    someone said. The shape of `ProfileHit` is what makes that structurally impossible.
    """
    from xbrain.knowledge.lexical import ProfileHit

    index = _index()
    index.add_profile("i1", "agent evaluation harnesses and their pitfalls", "a" * 64)
    index.add_profile("i2", "unrelated cooking notes", "b" * 64)

    hits = index.search_profiles("evaluation", limit=5)
    assert [h.item_id for h in hits] == ["i1"]
    assert set(ProfileHit.__dataclass_fields__) == {"item_id", "score"}


def test_the_profile_plane_honours_the_same_filters() -> None:
    """A filter that applies to items must apply on both planes, or the two disagree.

    `origins` is the one that did NOT (G-1): it lives on `chunks.origin`, the profile plane
    applied only the item-scoped clauses, and `search "Forecasting" --origin asr` on the real
    corpus answered 8 items of which 6 had no ASR surface at all. A profile is a string
    nobody wrote and has no origin, so under an origin filter it contributes NOTHING — the
    same fail-closed shape as a topic chunk under an author filter.

    Seen red before the fix: `origins=("vlm",)` returned both profiles.
    """
    index = _index()
    index.add_profile("i1", "agent evaluation harnesses", "a" * 64)
    index.add_profile("i2", "agent evaluation notes", "b" * 64)
    index.set_item_metadata("i1", source="bookmark", author_handle="karpathy", **_STAMP)
    index.set_item_metadata("i2", source="own_tweet", author_handle="vgonpa", **_STAMP)

    hits = index.search_profiles("evaluation", limit=5, filters=SearchFilters(source="own_tweet"))
    assert [h.item_id for h in hits] == ["i2"]
    assert (
        index.search_profiles("evaluation", limit=5, filters=SearchFilters(origins=("vlm",))) == ()
    )


@pytest.fixture()
def profiled_index() -> LexicalIndex:
    """Two PROFILES whose items differ on every filter axis and share the query term.

    The profile-plane twin of `filtered_index`: with both profiles matching the text, a
    filter the profile plane ignores leaves both in the result, so every assertion below can
    only pass because the filter reached `search_profiles`.
    """
    index = _index()
    index.add_profile("old", "the shared marrowgate profile", "a" * 64)
    index.add_profile("new", "the shared marrowgate profile too", "b" * 64)
    index.set_item_metadata(
        "old",
        created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        captured_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        source="bookmark",
        author_handle="karpathy",
        topics=("agent-evaluation",),
        content_kinds=("external_article",),
        surface_types=("post", "external_article"),
    )
    index.set_item_metadata(
        "new",
        created_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        captured_at=datetime(2026, 6, 2, tzinfo=timezone.utc),
        source="own_tweet",
        author_handle="vgonpa",
        topics=("ai-policy",),
        content_kinds=("x_video",),
        surface_types=("post", "image_description"),
    )
    return index


def test_every_declared_filter_is_pushed_on_the_profile_plane_too(
    profiled_index: LexicalIndex,
) -> None:
    """The TOTALITY half for the SECOND plane (G-1, spec §7.2, acceptance 6: *the eight*).

    `test_every_declared_filter_is_actually_pushed_to_sql` enumerates the eight fields of
    `SearchFilters` against `index.search` only, and `origins` slipped through the profile
    plane for exactly that reason: seven filters were asserted on both planes, one on one.
    The expected set is spelled out per filter rather than "old or new", because the honest
    answer for `origins` is the EMPTY set — a profile has no origin — and "narrowed to
    nothing" is a different claim from "narrowed to the other item".

    Seen red before the fix: `origins` returned `{"old", "new"}`.
    """
    expected = {
        "created_from": ({"created_from": datetime(2026, 1, 1, tzinfo=timezone.utc)}, {"new"}),
        "created_to": ({"created_to": datetime(2025, 6, 1, tzinfo=timezone.utc)}, {"old"}),
        "source": ({"source": "own_tweet"}, {"new"}),
        "author": ({"author": "karpathy"}, {"old"}),
        "topics": ({"topics": ("ai-policy",)}, {"new"}),
        "content_kinds": ({"content_kinds": ("x_video",)}, {"new"}),
        "origins": ({"origins": ("vlm",)}, set()),
        "has_surfaces": ({"has_surfaces": ("external_article",)}, {"old"}),
    }
    assert set(expected) == set(SearchFilters.model_fields), (
        "a filter was added to the frozen contract without a test that it reaches the profile plane"
    )

    def owners(**kwargs) -> set[str]:
        hits = profiled_index.search_profiles(
            "marrowgate", limit=10, filters=SearchFilters(**kwargs)
        )
        return {hit.item_id for hit in hits}

    assert owners() == {"old", "new"}, "the fixture must match both before any filter"
    for name, (kwargs, wanted) in expected.items():
        assert owners(**kwargs) == wanted, f"filter {name} did not narrow the profile plane"


def test_a_missing_table_is_an_error_not_an_empty_result() -> None:
    """C-2: `_fetch` used to absorb EVERY `OperationalError` except a read-only one and
    return `[]`, so `no such table`, `database is locked` and `disk I/O error` all came back
    as "the corpus holds nothing" — and the decision was made by substring of the message,
    which is CLAUDE.md rule 9 in miniature.

    The catch now absorbs exactly what it was written for — an expression FTS5's parser
    rejects — and everything else is an ERROR. Seen red before the fix: `search` returned
    `()` over an index whose `chunks_fts` had been dropped. Since U-4 (round 07) the error
    is the actionable one rather than the raw `OperationalError` this test used to pin: a
    base missing a table past the door is the same operator situation as a corrupt one,
    and the sentence names the rebuild — while still carrying SQLite's own words.
    """
    from xbrain.knowledge.index_schema import IndexIncompatibleError

    index = _index()
    index.add(_corpus_chunks())
    index.connection.execute("DROP TABLE chunks_fts")
    with pytest.raises(IndexIncompatibleError, match="no such table"):
        index.search("marrowgate", 5)


def test_a_database_error_at_query_time_is_the_rebuild_advice_not_a_traceback() -> None:
    """G-4 in `_fetch`: what the open-door probe does not catch, the query turns into the
    same sentence. `sqlite3.DatabaseError` (the parent of `OperationalError`) is what FTS5
    raises on a corrupt or missing shadow table — `fts5: corruption found reading blob…` —
    and it is neither an `IndexError_` nor an `OSError`, so it reached the operator raw.

    The parser's own errors keep degrading to `[]` (the test below), an `OperationalError`
    that is not a parse error keeps propagating as itself (C-2), and everything else in the
    `DatabaseError` family becomes `IndexIncompatibleError` with the rebuild advice.

    Seen red before the fix: `sqlite3.DatabaseError` propagated out of `search`.
    """
    from xbrain.knowledge.index_schema import IndexIncompatibleError

    index = _index()
    index.add(_corpus_chunks())
    index.connection.execute("DROP TABLE chunks_fts_data")
    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force"):
        index.search("marrowgate", 5)


def test_only_an_fts_syntax_error_degrades_to_no_results() -> None:
    """The other half of C-2: what the catch was FOR still degrades, so narrowing it did not
    turn a hostile query into a traceback.

    `match_expression` quotes every term, so no expression built by `search` can reach the
    parser malformed; the branch is exercised through `_fetch` directly with the three
    error families FTS5's parser emits (measured on sqlite 3.51.2: `fts5: syntax error`,
    `unterminated string`, `unknown special query`).
    """
    index = _index()
    index.add(_corpus_chunks())
    sql = f"{lexical._SELECT_CHUNKS} WHERE chunks_fts MATCH ? {lexical._CHUNK_RANK_ORDER} LIMIT ?"
    for malformed in ("NEAR(", '"unterminated', "*"):
        assert index._fetch(sql, (malformed, 5)) == [], malformed
