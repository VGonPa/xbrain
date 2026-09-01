"""The deterministic lexical baseline — the SAME FTS5, on `sqlite3(":memory:")` (m16).

Spec §8.5 requires a deterministic lexical baseline to compare against. This is it, and it
deliberately shares every scoring decision with the index Plan 02 will persist: same DDL,
same tokenizer, same `bm25()`, same explicit tie-break (`lexical_fts`). The only difference
is where the database lives, so the baseline published today is measured by the same
instrument as the one that replaces it.

`rg --files-with-matches` is NOT a baseline (spec §8.5): its order comes from the filesystem
walk and it defines no relevance score, so it yields no valid precision@k or MRR. It stays
useful as a literal-match diagnostic, and nothing more.

Read-only with respect to the store: it is handed chunks and never opens `items.json`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Sequence

from xbrain.knowledge.lexical_fts import RANK_ORDER, create_schema, match_expression
from xbrain.knowledge.models import KnowledgeChunk, SurfaceType

# How much of a matching chunk is shown back. Long enough to recognise the hit, short enough
# that a log or a report never carries an article (spec §10.8).
EXCERPT_CHARS = 300


# The fixed SELECT skeleton. A module constant rather than an inline literal so the only
# thing the query builder concatenates at call time is a WHERE clause made of bound `?`
# placeholders and the `RANK_ORDER` constant — which is what makes the `# nosec B608` below
# a statement of fact rather than a request to look away.
_SELECT = (
    "SELECT chunk.*, bm25(chunk_fts) AS score FROM chunk_fts "
    "JOIN chunk ON chunk.rowid = chunk_fts.rowid"
)


@dataclass(frozen=True)
class LexicalHit:
    """One ranked chunk, resolvable back to its surface and owner.

    Carries the owner and the surface, not just a score: spec §3.3 requires reverse
    resolution from a chunk to its surface and item, and a ranked id nobody can resolve is a
    number rather than evidence.
    """

    chunk_id: str
    surface_id: str
    owner_type: str
    owner_id: str
    surface_type: SurfaceType
    origin: str
    trust_class: str
    derived: bool
    title: str | None
    url: str | None
    excerpt: str
    score: float


class InMemoryLexicalIndex:
    """An FTS5 index over a set of chunks, held in memory for one run.

    Not a persistence layer and not trying to be: Plan 02 owns `data/index/`, incremental
    updates and staleness. This exists so the evaluation harness has a real, deterministic
    retriever to measure BEFORE any of that is built — which is the ordering the spec
    requires, since the embedding model may not be chosen without an evaluation to choose it
    with (spec §5.5, §14).
    """

    def __init__(self) -> None:
        self._connection = sqlite3.connect(":memory:")
        self._connection.row_factory = sqlite3.Row
        create_schema(self._connection)

    def add(self, chunks: Sequence[KnowledgeChunk]) -> int:
        """Index a batch of chunks. Returns how many were stored.

        A chunk already present (same `chunk_id`) is skipped rather than duplicated: the
        emitter can legitimately produce the same chunk twice across two calls, and a
        duplicate would let one body occupy two ranks.
        """
        stored = 0
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            cursor = self._connection.execute(
                "INSERT OR IGNORE INTO chunk (chunk_id, surface_id, owner_type, owner_id, "
                "surface_type, origin, trust_class, derived, title, url, text, fingerprint) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    chunk.chunk_id,
                    chunk.surface_id,
                    chunk.owner_type,
                    chunk.owner_id,
                    chunk.surface_type,
                    chunk.origin,
                    chunk.trust_class,
                    int(chunk.derived),
                    chunk.title,
                    chunk.url,
                    chunk.text,
                    chunk.fingerprint,
                ),
            )
            if cursor.rowcount:
                # The FTS row is written with the SAME rowid as its metadata row, which is
                # what lets the join below be a plain equality rather than a second lookup —
                # and is why `rowid INTEGER PRIMARY KEY` is declared explicitly (VACUUM
                # renumbers an implicit one, silently repointing every FTS row).
                self._connection.execute(
                    "INSERT INTO chunk_fts (rowid, text, title) VALUES (?,?,?)",
                    (cursor.lastrowid, chunk.text, chunk.title or ""),
                )
                stored += 1
        self._connection.commit()
        return stored

    def search(
        self,
        query: str,
        limit: int,
        *,
        surface_types: tuple[SurfaceType, ...] = (),
        origins: tuple[str, ...] = (),
    ) -> tuple[LexicalHit, ...]:
        """The top `limit` chunks for `query`, best first, deterministic under ties.

        FILTERS ARE `WHERE` CLAUSES, not query text (spec §5.3). Appending a topic or a kind
        to the query string would let the filter's own words compete for bm25 weight against
        the user's terms — a filter that changes the ranking is not a filter.

        An empty query is a validation error, not an empty result (spec §9.3): an empty
        result set claims something about the corpus, when the truth is that nothing was
        asked.
        """
        if not query.strip():
            raise ValueError("la consulta está vacía")
        expression = match_expression(query)
        if expression is None:
            return ()
        clauses, params = ["chunk_fts MATCH ?"], [expression]
        if surface_types:
            clauses.append(f"chunk.surface_type IN ({_placeholders(len(surface_types))})")
            params += list(surface_types)
        if origins:
            clauses.append(f"chunk.origin IN ({_placeholders(len(origins))})")
            params += list(origins)
        # nosec B608 - no value is interpolated. Every filter VALUE is bound through a `?`
        # placeholder; the only computed fragments are `_placeholders(n)`, which can return
        # nothing but commas and question marks by construction, and `RANK_ORDER`, a module
        # constant. The variadic `IN` clause is the one shape sqlite3 cannot parameterise
        # wholesale, so building the placeholder GROUP is unavoidable — but the group is
        # derived from a COUNT, never from the caller's strings.
        where = " AND ".join(clauses)
        sql = f"{_SELECT} WHERE {where} {RANK_ORDER} LIMIT ?"  # nosec B608
        try:
            rows = self._connection.execute(sql, (*params, limit)).fetchall()
        except sqlite3.OperationalError:
            # A malformed MATCH expression that survived quoting. Degrading to "no results"
            # is right here and only here: the query was understood as data, it simply
            # matched nothing FTS5 could parse. It is NOT the same as an empty query, which
            # is rejected above.
            return ()
        return tuple(_hit(row) for row in rows)

    def __len__(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM chunk").fetchone()[0])


def _placeholders(count: int) -> str:
    """`?,?,?` for a variadic `IN` clause — derived from a COUNT, never from a caller string.

    The one SQL fragment this module builds dynamically, and it is built from an integer, so
    it cannot carry a caller's text into the statement no matter what was passed. Extracted
    as a function so that claim is structurally true and checkable in one line, rather than a
    promise attached to an f-string.
    """
    return ",".join("?" * count)


def _hit(row: sqlite3.Row) -> LexicalHit:
    return LexicalHit(
        chunk_id=row["chunk_id"],
        surface_id=row["surface_id"],
        owner_type=row["owner_type"],
        owner_id=row["owner_id"],
        surface_type=row["surface_type"],
        origin=row["origin"],
        trust_class=row["trust_class"],
        derived=bool(row["derived"]),
        title=row["title"],
        url=row["url"],
        excerpt=row["text"][:EXCERPT_CHARS],
        score=float(row["score"]),
    )
