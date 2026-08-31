"""The FTS5 schema and scorer, SHARED by the in-memory baseline and Plan 02's index.

ONE definition of how text is tokenized and scored, in one module, because the alternative
is two: a throwaway baseline here and the real thing in Plan 02. Two scorers make the
characterization fixture a bridge between them, so the day Plan 02 lands the fixture breaks,
and the comfortable fix is to regenerate it — at which point it stops pinning anything.

What changes in Plan 02 is WHERE the database lives (`:memory:` -> `data/index/knowledge.db`),
not how it scores. This module is what makes that true.

THE TOKENIZER, AND ITS DOCUMENTED LIMIT. `unicode61 remove_diacritics 2` folds accents, which
a bilingual corpus queried in Spanish needs — without it, "evaluacion" and "evaluación" are
different terms. There is NO STEMMING: FTS5 has no multilingual stemmer, and the English one
would wreck the Spanish half of the corpus. So "agent" does not match "agents". That is a
real limitation of the lexical baseline, it is what the vector layer of Plan 03 has to beat,
and it is pinned by a test — because a limit nobody writes down gets quietly attributed to
the corpus instead of to the tokenizer.
"""

from __future__ import annotations

import re
import sqlite3

# The ONE tokenizer string. Exported so the baseline, Plan 02's persisted index and the
# ranking fixture all record the same value and a change to it is visible in a diff.
FTS_TOKENIZE = "unicode61 remove_diacritics 2"

# `rowid INTEGER PRIMARY KEY` explicitly (m1): an implicit rowid is renumbered by `VACUUM`,
# which would silently repoint every FTS row at a different chunk.
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS chunk (
    rowid        INTEGER PRIMARY KEY,
    chunk_id     TEXT NOT NULL UNIQUE,
    surface_id   TEXT NOT NULL,
    owner_type   TEXT NOT NULL,
    owner_id     TEXT NOT NULL,
    surface_type TEXT NOT NULL,
    origin       TEXT NOT NULL,
    trust_class  TEXT NOT NULL,
    derived      INTEGER NOT NULL,
    title        TEXT,
    url          TEXT,
    text         TEXT NOT NULL,
    fingerprint  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunk_owner ON chunk (owner_type, owner_id);
CREATE INDEX IF NOT EXISTS chunk_surface_type ON chunk (surface_type);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5 (
    text,
    title,
    tokenize = '{FTS_TOKENIZE}'
);
"""

# The tie-break is EXPLICIT (spec §3.7.8). Two chunks with identical text score identically,
# and sqlite's row order for a tie is an implementation detail — so without the second key
# the ranking would differ between two builds of the same data.
RANK_ORDER = "ORDER BY bm25(chunk_fts) ASC, chunk.chunk_id ASC"

# FTS5 reads punctuation as syntax: `@`, `"`, `*`, `(`, `NEAR`, `-`. The golden set has
# literal queries — `@simonw`, `11.37%` — whose characters would otherwise become operators
# or a syntax error. Every term is therefore quoted as an FTS5 string literal, so the query
# is DATA and never syntax.
_TERM_SPLIT = re.compile(r"[^\w@#.%/-]+", re.UNICODE)


def create_schema(connection: sqlite3.Connection) -> None:
    """Create the chunk table and its FTS5 companion. Idempotent."""
    connection.executescript(_SCHEMA)


def match_expression(query: str) -> str | None:
    """`query` as a safe FTS5 MATCH expression, or None when it holds no usable term.

    Every term is wrapped in double quotes with internal quotes doubled, which makes it an
    FTS5 *string* — punctuation inside is literal, so `@simonw` searches for the handle
    instead of erroring, and `NEAR(` cannot start an operator. Terms are ANDed, the FTS5
    default.

    Returning None rather than raising lets the caller distinguish "you asked nothing"
    (a validation error, spec §9.3) from "your terms were all punctuation" (an honest empty
    result). Answering the second with results from a mangled query would be worse than
    either.
    """
    terms = [term for term in _TERM_SPLIT.split(query.strip()) if term]
    if not terms:
        return None
    return " ".join('"' + term.replace('"', '""') + '"' for term in terms)
