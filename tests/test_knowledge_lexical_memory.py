# tests/test_knowledge_lexical_memory.py
"""The deterministic lexical baseline: FTS5 on `sqlite3(":memory:")` (Plan 01 §5.3, m16).

WHY NOT A HAND-WRITTEN BM25. An earlier draft of the plan asked for an Okapi BM25 with its
own tokenizer. That is code written to be deleted one PR later, and worse: it makes the
characterization fixture a bridge between TWO DIFFERENT SCORERS, so when Plan 02 replaces it
the fixture breaks, and the comfortable fix is to regenerate it — at which point it silently
stops pinning anything.

Standing up the SAME FTS5 on an in-memory database costs no dependency (`sqlite3` is stdlib)
and removes the problem at the root: what dies in Plan 02 is WHERE the database lives
(`:memory:` -> `data/index/knowledge.db`), not HOW it scores. The fixture then pins one
scorer against itself across time, which is what a characterization is for.

Verified in this interpreter: sqlite 3.51.2, FTS5 and `bm25()` operational.

`rg --files-with-matches` is NOT a baseline (spec §8.5): its order comes from the filesystem
walk and it defines no relevance score, so it cannot produce a valid precision@k or MRR.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from xbrain.knowledge.chunking import ChunkerParams, chunk_surfaces
from xbrain.knowledge.lexical_fts import FTS_TOKENIZE, create_schema
from xbrain.knowledge.lexical_memory import InMemoryLexicalIndex
from xbrain.knowledge.models import KnowledgeChunk, KnowledgeSurface, Locator
from xbrain.knowledge.surfaces import item_surfaces
from xbrain.models import Item

FIXTURES = Path(__file__).parent / "fixtures"

# The parameters the ranking fixture was built with, PINNED HERE and passed explicitly (M7).
# Plan 02 sweeps `target x overlap` and bumps CHUNKER_VERSION; if this read the module
# constant, that sweep would break the fixture it exists to protect.
PINNED_CHUNKER_PARAMS = ChunkerParams(target=1200, max_chars=2000, overlap=150, min_chars=40)


def _corpus_chunks() -> list[KnowledgeChunk]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    chunks: list[KnowledgeChunk] = []
    for item_raw in raw["items"].values():
        item = Item.model_validate(item_raw)
        chunks += list(
            chunk_surfaces(item_surfaces(item), params=PINNED_CHUNKER_PARAMS, url=item.url)
        )
    return chunks


def _chunk(chunk_id: str, text: str, surface_id: str = "item:x:post:0") -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        surface_id=surface_id,
        owner_type="item",
        owner_id="x",
        surface_type="post",
        text=text,
        chunk_index=0,
        char_start=0,
        char_end=len(text),
        origin="source",
        trust_class="primary_source",
        derived=False,
        fingerprint="f" * 64,
    )


# ---------------------------------------------------------------------------
# The engine is the one Plan 02 will persist
# ---------------------------------------------------------------------------


def test_fts5_and_bm25_are_available_in_this_interpreter() -> None:
    """A precondition, asserted rather than assumed.

    If FTS5 were compiled out of the local sqlite, every ranking test below would fail with
    an opaque `sqlite3.OperationalError` from deep inside the index. Failing here, by name,
    is the difference between "your Python has no FTS5" and "the ranking changed".
    """
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    connection.execute("INSERT INTO chunk_fts (rowid, text) VALUES (1, 'hola mundo')")
    rows = connection.execute(
        "SELECT rowid, bm25(chunk_fts) FROM chunk_fts WHERE chunk_fts MATCH ?", ("hola",)
    ).fetchall()
    assert rows and isinstance(rows[0][1], float)


def test_the_tokenizer_folds_diacritics() -> None:
    """`unicode61 remove_diacritics 2` — a Spanish query must reach an unaccented body.

    The corpus is bilingual and the queries are written in Spanish; without folding,
    "evaluacion" and "evaluación" are two different terms and half the golden set's own
    queries would miss. This is the SAME tokenizer string Plan 02 persists, exported as one
    constant so the two cannot drift.
    """
    assert "remove_diacritics 2" in FTS_TOKENIZE
    index = InMemoryLexicalIndex()
    index.add([_chunk("c1", "La evaluación automática de agentes")])
    assert [hit.chunk_id for hit in index.search("evaluacion", limit=5)] == ["c1"]


def test_there_is_no_stemming_and_the_limit_is_documented_not_hidden() -> None:
    """FTS5 has no multilingual stemmer, and an English one would wreck the Spanish half.

    So "agent" does not match "agents" — a REAL limitation of the lexical baseline, and one
    the vector layer of Plan 03 has to beat. Pinning it as a test is what keeps the published
    baseline honest: a limit nobody wrote down gets quietly attributed to the corpus instead
    of to the tokenizer.
    """
    index = InMemoryLexicalIndex()
    index.add([_chunk("c1", "evaluating agents in production")])
    assert index.search("agent", limit=5) == ()
    assert index.search("agents", limit=5)


# ---------------------------------------------------------------------------
# 24 — stable tie-break
# ---------------------------------------------------------------------------


def test_ties_break_on_chunk_id_ascending() -> None:
    """Spec §3.7.8: the order is STABLE under ties.

    Two chunks with identical text get an identical bm25 score, and sqlite's row order for
    a tie is an implementation detail — so without an explicit second key the ranking would
    differ between rebuilds of the same data. Seen red by dropping `, chunk_id ASC` from the
    ORDER BY: the pair comes back in insertion order and reverses when inserted the other
    way round.
    """
    text = "identical body text for both chunks"
    forward = InMemoryLexicalIndex()
    forward.add([_chunk("b:2", text), _chunk("a:1", text)])
    backward = InMemoryLexicalIndex()
    backward.add([_chunk("a:1", text), _chunk("b:2", text)])
    assert [h.chunk_id for h in forward.search("identical", limit=5)] == ["a:1", "b:2"]
    assert [h.chunk_id for h in backward.search("identical", limit=5)] == ["a:1", "b:2"]


def test_the_query_is_escaped_not_interpolated() -> None:
    """A quote or an FTS operator in a query must not become syntax.

    Spec §9.3 asks for a stable validation error rather than a crash, and the golden set
    contains literal queries like `@simonw` and `11.37%` whose punctuation FTS5 would
    otherwise read as operators. It also means a query can never be an injection vector,
    though the database here holds only the user's own corpus.
    """
    index = InMemoryLexicalIndex()
    index.add([_chunk("c1", "the handle is @simonw here")])
    assert [h.chunk_id for h in index.search("@simonw", limit=5)] == ["c1"]
    assert index.search('"unbalanced', limit=5) == ()
    assert index.search("NEAR(", limit=5) == ()


def test_an_empty_query_is_a_validation_error() -> None:
    """Spec §9.3: an empty query is a stable error, not an empty result set.

    Empty results would say "nothing in your corpus matches", which is a claim about the
    corpus; the truth is that nothing was asked.
    """
    index = InMemoryLexicalIndex()
    with pytest.raises(ValueError, match="vacía"):
        index.search("   ", limit=5)


# ---------------------------------------------------------------------------
# 25 — the characterization fixture
# ---------------------------------------------------------------------------

RANKING_QUERIES = (
    "Quillfeather",
    "Marrowgate protocol",
    "Bramblewick",
    "Cindervale checklist",
    "Thistledown",
    "Pelicanine",
)


def test_ranking_matches_the_characterization_fixture() -> None:
    """The pinned top-10 for six queries over the 12-item fixture corpus.

    Built with `PINNED_CHUNKER_PARAMS` passed EXPLICITLY (M7). Plan 02's sweep changes the
    module constant, not this argument, so the sweep cannot move this fixture; the sweep
    winner is pinned separately, under its own CHUNKER_VERSION, and the report compares the
    two — which is the comparison that actually matters.

    Red whenever chunking, tokenization, the ORDER BY or the fixture corpus change. That is
    the point: those are exactly the four things that silently alter every downstream recall
    number.
    """
    expected = json.loads((FIXTURES / "knowledge_ranking.json").read_text(encoding="utf-8"))
    index = InMemoryLexicalIndex()
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


def test_a_hit_resolves_back_to_its_item_and_surface() -> None:
    """Spec §3.3: reverse resolution from a chunk to its surface and owner.

    A ranked id nobody can resolve is not evidence, it is a number. The hit therefore
    carries the surface, the owner and the excerpt, not just a score.
    """
    index = InMemoryLexicalIndex()
    index.add(_corpus_chunks())
    (hit,) = index.search("Quillfeather", limit=1)
    assert hit.owner_id in {"k03", "k12"}
    assert hit.surface_type in {"external_article", "user_note"}
    assert "Quillfeather" in hit.excerpt


def test_filters_are_applied_before_scoring_not_as_query_text() -> None:
    """Spec §5.3: *filters are applied before scoring whenever the backend allows it.*

    Appending a topic to the query string would let the topic's own words compete for bm25
    weight against the user's terms — a filter that changes the RANKING is not a filter.
    Here the surface-type restriction is a `WHERE` clause on the metadata table.
    """
    index = InMemoryLexicalIndex()
    index.add(_corpus_chunks())
    unfiltered = index.search("Quillfeather", limit=10)
    filtered = index.search("Quillfeather", limit=10, surface_types=("user_note",))
    assert {h.surface_type for h in filtered} == {"user_note"}
    assert len(filtered) < len(unfiltered)


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
    index = InMemoryLexicalIndex()
    index.add(chunk_surfaces((empty_surface,)))
    assert index.search("anything", limit=5) == ()
