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

THE CONNECTIVE IS A DISJUNCTION, AND THAT IS A RETRIEVAL DECISION, NOT A DEFAULT (M3). Terms
were ANDed — the FTS5 default — which turns a twelve-word question into a demand that all
twelve words sit inside ONE chunk. Measured on the real corpus (2,404 items, 18,319 chunks,
the 21 scorable cases of `eval/golden-set.yaml`): the conjunction returned NOT ONE ROW for 18
of the 21, and the only three that retrieved anything were single-term `exacto` queries. The
published baseline — `semantico` 0.0, `cruzado_idioma` 0.0, `topic` 0.0 — was therefore a
measurement of the query builder, and it was being read as an absence of vocabulary overlap.
The counterfactual, on the SAME index, SAME bm25, SAME chunks, SAME tie-break, changing only
the connective: empty result sets 18/21 -> 0/21, mean `recall@10` 0.1429 -> 0.8099, and the
`exacto` stratum did not degrade (0.60 -> 0.80).

The deeper reason is that a conjunction and `bm25()` are two different retrieval models
stacked on one another. bm25 is a RANKING function over a bag of words: it expects a wide
candidate set and discriminates inside it, weighting each term by inverse document frequency
so that a word present in nearly every chunk contributes almost nothing. Requiring every term
first does that discrimination by brute force BEFORE the scorer runs, and in 18 of 21 cases
it left the scorer nothing to rank. Plan 01 §5.3 justifies sharing this module with Plan 02's
persisted index on the grounds that "what dies in Plan 02 is where the database lives, not how
it scores" — which is only true if the score is what decides the ranking. Under the
conjunction it mostly was not.

The cost is real and is declared: the candidate set is much wider, so latency went from p50
0.23 ms / p95 0.74 ms to p50 8.6 ms / p95 27.4 ms over the same corpus and the same 21 cases.

WHAT WAS DELIBERATELY NOT DONE. No stopword list (bm25's IDF already discounts a term that
matches everything, and a bilingual list would be a judgement nobody measured), and no
"AND first, OR if empty" fallback — that would score some cases under boolean retrieval and
others under bm25 while publishing both in one column, which is the two-definitions defect of
CLAUDE.md rule 5. Choosing between `OR`, a minimum-should-match or a per-term weighting is
Plan 02's sweep, with the golden set in front of it. What this module fixes is that the
number published today measures the retriever.
"""

from __future__ import annotations

import re
import sqlite3

# The ONE tokenizer string. Exported so the baseline, Plan 02's persisted index and the
# ranking fixture all record the same value and a change to it is visible in a diff.
FTS_TOKENIZE = "unicode61 remove_diacritics 2"

# The ONE connective, exported for the same reason as the tokenizer: it decides the candidate
# set, so it decides every recall number downstream, and it belongs in the characterization
# fixture where a change to it shows up in a diff instead of in a rewritten baseline.
FTS_CONNECTIVE = "OR"

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
    instead of erroring, and `NEAR(` cannot start an operator. That quoting is the security
    property and it is unchanged by the connective: a term that is literally `OR` becomes the
    string `"OR"`, never the operator.

    Terms are joined by `FTS_CONNECTIVE`, a DISJUNCTION (M3). The FTS5 default is a
    conjunction, and it required every word of a question to appear in one chunk: 18 of the
    21 scorable golden-set cases came back with zero rows against the real corpus, so the
    published `semantico: 0.0` measured the connective rather than the corpus. See the module
    docstring for the measurement and for what a disjunction costs.

    Returning None rather than raising lets the caller distinguish "you asked nothing"
    (a validation error, spec §9.3) from "your terms were all punctuation" (an honest empty
    result). Answering the second with results from a mangled query would be worse than
    either.
    """
    terms = [term for term in _TERM_SPLIT.split(query.strip()) if term]
    if not terms:
        return None
    quoted = ('"' + term.replace('"', '""') + '"' for term in terms)
    return f" {FTS_CONNECTIVE} ".join(quoted)
