"""The persisted index: its DDL, how it is opened, and the two deletion orders (Plan 02 §2).

The index is DERIVED and RECONSTRUCTIBLE (spec §5.6). It lives under `data/index/`, it is
never versioned, and nothing in it is a second source of truth: `get` reads the store, and
`search` hydrates verification from the live store. What lives here is a retrieval structure
and the metadata needed to filter BEFORE scoring.

THREE DECISIONS IN THIS FILE FAIL SILENTLY WHEN THEY ARE WRONG, which is why each is pinned
by behaviour in `tests/test_knowledge_index_schema.py`:

**1. `rowid` is an explicit `INTEGER PRIMARY KEY` on `chunks` and on `profiles` (m1).** Both
FTS5 tables are EXTERNAL CONTENT tables: they store no text and read it back from their
content table by rowid. SQLite documents that `VACUUM` may renumber the rowids of a table
that lacks an `INTEGER PRIMARY KEY` — so an implicit rowid would leave every FTS entry
pointing at a DIFFERENT row, and the index would return wrong text without raising anything.
`chunk_id` stays the contract's identity (`TEXT UNIQUE NOT NULL`); it simply stops being the
physical key.

**2. A delete retracts the OLD text FIRST.** With external content FTS5 cannot retract a
row's tokens after the row is gone, because it can no longer read them. Deleting the content
row first leaves orphan entries in the index — the same silent wrongness as the rowid. The
order lives in exactly two functions (`delete_chunk_rows`, `delete_profile_rows`) and nowhere
else.

**3. `profiles_fts` is external content over `profiles`, which stores `profile_text`.** Plan
02 §2 sketched it as `content=''`. A contentless FTS5 table cannot be deleted from without
supplying the ORIGINAL text, which a contentless table by definition does not keep — so the
incremental update path (`index update` removes an item) would have had no way to retract a
profile. Storing the text makes deletion possible at all, and makes it the SAME operation as
on the chunk plane rather than a second, subtly different one. The cost is one copy of the
profile composition on disk; the alternative was `contentless_delete=1`, which pins a minimum
SQLite version for no gain here.

**WHAT IS NOT HERE, ON PURPOSE.** There is no `verification` column on `surfaces` (M5): a
stored verdict cannot be invalidated when the verdict changes, so a `FAIL` revoked by
`verify --audit` would keep being served as the `PASS` it used to be. `search` hydrates it
from the live store with the same freshness check `generate._verdict_badge` applies.

And there is no `FOREIGN KEY` on `chunks.surface_id`. SQLite does not enforce foreign keys
unless `PRAGMA foreign_keys=ON`, so a `REFERENCES` clause with the pragma off is a claim in
the DDL that nothing checks — prose in the column where a guard belongs. The property it
would assert (no chunk without its surface) is asserted where it can actually be checked:
over the output of `index build`, by a test.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path

from xbrain.knowledge.lexical_fts import FTS_TOKENIZE, fts5_table_sql
from xbrain.models import _reject_local_path_traversal

# Bumped when the physical layout changes in a way an existing database cannot answer. The
# manifest records it and a mismatch refuses the query ENTIRELY (spec §9.3) — never a partial
# answer over a schema the code no longer understands.
SCHEMA_VERSION = "1"

DB_FILENAME = "knowledge.db"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_INDEX_DIR_NAME = "index"

# The plain tables, declared so the DDL and the suite cannot drift: a table created without
# being declared, or declared without being created, goes red.
TABLES: frozenset[str] = frozenset(
    {
        "items",
        "item_topics",
        "item_content_kinds",
        "surfaces",
        "chunks",
        "profiles",
        "topics",
        "source_failures",
        "unfetched_links",
    }
)

# The two RETRIEVAL PLANES (spec §5.1). Separate tables on purpose: the profile finds the
# item as a conceptual unit, the chunks find the fragment where a fact lives. One shared
# table would let a profile — a string nobody wrote — surface as a citable `SearchMatch`,
# which Plan 01 forbids.
FTS_TABLES: frozenset[str] = frozenset({"chunks_fts", "profiles_fts"})

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS items (
    item_id           TEXT PRIMARY KEY,
    source            TEXT NOT NULL,
    url               TEXT NOT NULL,
    author_handle     TEXT NOT NULL,
    author_name       TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    captured_at       TEXT NOT NULL,
    primary_topic     TEXT,
    note_path         TEXT,
    bookmark_folder   TEXT,
    store_fingerprint TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_author ON items (author_handle);
CREATE INDEX IF NOT EXISTS items_source ON items (source);

CREATE TABLE IF NOT EXISTS item_topics (
    item_id    TEXT NOT NULL,
    slug       TEXT NOT NULL,
    is_primary INTEGER NOT NULL,
    PRIMARY KEY (item_id, slug)
);
CREATE INDEX IF NOT EXISTS item_topics_slug ON item_topics (slug);

CREATE TABLE IF NOT EXISTS item_content_kinds (
    item_id TEXT NOT NULL,
    kind    TEXT NOT NULL,
    PRIMARY KEY (item_id, kind)
);
CREATE INDEX IF NOT EXISTS item_content_kinds_kind ON item_content_kinds (kind);

CREATE TABLE IF NOT EXISTS surfaces (
    surface_id         TEXT PRIMARY KEY,
    owner_type         TEXT NOT NULL,
    owner_id           TEXT NOT NULL,
    surface_type       TEXT NOT NULL,
    origin             TEXT NOT NULL,
    trust_class        TEXT NOT NULL,
    derived            INTEGER NOT NULL,
    attribution_handle TEXT,
    attribution_name   TEXT,
    title              TEXT,
    url                TEXT,
    locator_json       TEXT NOT NULL,
    language           TEXT,
    fingerprint        TEXT NOT NULL,
    char_length        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS surfaces_owner ON surfaces (owner_type, owner_id);
CREATE INDEX IF NOT EXISTS surfaces_type ON surfaces (owner_id, surface_type);

CREATE TABLE IF NOT EXISTS chunks (
    rowid         INTEGER PRIMARY KEY,
    chunk_id      TEXT NOT NULL UNIQUE,
    surface_id    TEXT NOT NULL,
    owner_type    TEXT NOT NULL DEFAULT 'item',
    owner_id      TEXT NOT NULL DEFAULT '',
    surface_type  TEXT NOT NULL DEFAULT 'post',
    origin        TEXT NOT NULL DEFAULT 'source',
    trust_class   TEXT NOT NULL DEFAULT 'primary_source',
    derived       INTEGER NOT NULL DEFAULT 0,
    chunk_index   INTEGER NOT NULL DEFAULT 0,
    char_start    INTEGER NOT NULL DEFAULT 0,
    char_end      INTEGER NOT NULL DEFAULT 0,
    text          TEXT NOT NULL,
    title         TEXT,
    url           TEXT,
    language      TEXT,
    fingerprint   TEXT NOT NULL DEFAULT '',
    -- Denormalised ON PURPOSE: these are the filters spec §7.2 asks for most often, and a
    -- copy here lets the `WHERE` run BEFORE the `MATCH` without a JOIN per query. The cost
    -- is one copy per chunk, paid in `build`, which is explicit and manual.
    created_at    TEXT,
    source        TEXT,
    primary_topic TEXT
);
CREATE INDEX IF NOT EXISTS chunks_owner ON chunks (owner_type, owner_id);
CREATE INDEX IF NOT EXISTS chunks_surface_type ON chunks (surface_type);
CREATE INDEX IF NOT EXISTS chunks_created_at ON chunks (created_at);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks (source);
CREATE INDEX IF NOT EXISTS chunks_origin ON chunks (origin);
CREATE INDEX IF NOT EXISTS chunks_surface ON chunks (surface_id);

CREATE TABLE IF NOT EXISTS profiles (
    rowid        INTEGER PRIMARY KEY,
    item_id      TEXT NOT NULL UNIQUE,
    profile_text TEXT NOT NULL,
    fingerprint  TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS topics (
    slug                  TEXT PRIMARY KEY,
    description           TEXT NOT NULL,
    overview              TEXT,
    notes_json            TEXT NOT NULL DEFAULT '[]',
    synthesized_at        TEXT,
    post_count_at_synth   INTEGER,
    stale                 INTEGER NOT NULL DEFAULT 1,
    primary_item_ids_json TEXT NOT NULL DEFAULT '[]',
    secondary_item_ids_json TEXT NOT NULL DEFAULT '[]',
    vocab_fingerprint     TEXT NOT NULL,
    synthesis_fingerprint  TEXT
);

CREATE TABLE IF NOT EXISTS source_failures (
    item_id        TEXT NOT NULL,
    kind           TEXT NOT NULL,
    url            TEXT NOT NULL,
    failure_reason TEXT NOT NULL,
    error          TEXT,
    http_status    INTEGER,
    attempts       INTEGER
);
CREATE INDEX IF NOT EXISTS source_failures_item ON source_failures (item_id);

-- m7: links NEVER attempted, or downloaded with no extractable body. NOT failures, and the
-- two facts are kept apart because a link nobody tried is not a link that returned a 404.
CREATE TABLE IF NOT EXISTS unfetched_links (
    item_id TEXT NOT NULL,
    url     TEXT NOT NULL,
    reason  TEXT NOT NULL,
    detail  TEXT
);
CREATE INDEX IF NOT EXISTS unfetched_links_item ON unfetched_links (item_id);

{fts5_table_sql("chunks_fts", content="chunks")};
{fts5_table_sql("profiles_fts", columns=("profile_text",), content="profiles")};
"""


class IndexError_(Exception):
    """Base class for the index's actionable errors — never a raw traceback (spec §9.3)."""


class IndexMissingError(IndexError_):
    """The index has not been built yet. Names the command that builds it (step 30)."""


class IndexIncompatibleError(IndexError_):
    """The manifest was written by a different schema, emitter or chunker (step 29).

    Raised INSTEAD of answering. Spec §9.3: an incompatible manifest is never queried
    partially — a partial answer over a schema the code no longer understands is a wrong
    answer wearing a right one's shape.
    """


def db_path(index_dir: Path) -> Path:
    """Where the SQLite database lives inside the index directory."""
    return index_dir / DB_FILENAME


def manifest_path(index_dir: Path) -> Path:
    """Where the manifest lives inside the index directory."""
    return index_dir / MANIFEST_FILENAME


def resolve_index_dir(data_dir: Path, name: str) -> Path:
    """`data_dir / name`, resolved and PROVEN to stay inside `data_dir` (§12.6, m8).

    Two checks, and they are not the same check. `_reject_local_path_traversal` — reused
    rather than reimplemented, so the repo has one definition of "this path is not allowed to
    climb" — rejects an absolute path and any literal `..`. It does NOT establish containment:
    a symlink under `data/` pointing anywhere on the filesystem passes it untouched. The
    `is_relative_to` on the RESOLVED paths is what actually contains it.
    """
    _reject_local_path_traversal(name)
    root = data_dir.resolve()
    resolved = (data_dir / name).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            f"El directorio del índice {name!r} resuelve a {resolved}, fuera de {root}. "
            "Ajusta `[index].dir` en config.toml a una ruta contenida en data/."
        )
    return resolved


def create_schema(connection: sqlite3.Connection) -> None:
    """Create every table, both FTS planes and the indexes. Idempotent."""
    connection.executescript(_SCHEMA)


def open_index(path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the index database, creating the schema when opening for writing.

    `read_only=True` opens `file:…?mode=ro`, so a stray write is an `OperationalError`
    instead of a silent repair — spec §5.6 is explicit that a query never modifies or repairs
    the index. A promise not to write is not the same property as being unable to.
    """
    if read_only:
        if not path.exists():
            raise IndexMissingError(
                f"No hay índice en {path}. Constrúyelo con `xbrain index build`."
            )
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    create_schema(connection)
    return connection


def open_memory_index() -> sqlite3.Connection:
    """The SAME schema on `sqlite3(":memory:")` — what the evaluation harness measures on.

    Plan 01 §5.3 justifies sharing the scorer with the persisted index on the grounds that
    *what dies in Plan 02 is where the database lives, not how it scores*. This function is
    what keeps that literally true: one DDL, one tokenizer, one `bm25()`, one tie-break, and
    the only difference is the connection string.
    """
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    create_schema(connection)
    return connection


def delete_chunk_rows(connection: sqlite3.Connection, chunk_ids: Sequence[str]) -> int:
    """Remove chunks from BOTH planes, in the only order that works. Returns how many.

    THE ORDER IS THE WHOLE FUNCTION. `chunks_fts` is external content, so FTS5 must re-read
    the OLD text to retract its tokens; once the row in `chunks` is gone it cannot, and the
    tokens stay behind pointing at a rowid that no longer exists. Reversing these two
    statements produces phantom results and raises nothing — which is why the deletion lives
    in one function instead of at each call site.
    """
    deleted = 0
    for chunk_id in chunk_ids:
        row = connection.execute(
            "SELECT rowid, text, title FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row is None:
            continue
        connection.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, text, title) VALUES('delete', ?, ?, ?)",
            (row["rowid"], row["text"], row["title"] or ""),
        )
        connection.execute("DELETE FROM chunks WHERE rowid = ?", (row["rowid"],))
        deleted += 1
    return deleted


def delete_profile_rows(connection: sqlite3.Connection, item_ids: Sequence[str]) -> int:
    """Remove item profiles from both planes, in the same order and for the same reason."""
    deleted = 0
    for item_id in item_ids:
        row = connection.execute(
            "SELECT rowid, profile_text FROM profiles WHERE item_id = ?", (item_id,)
        ).fetchone()
        if row is None:
            continue
        connection.execute(
            "INSERT INTO profiles_fts(profiles_fts, rowid, profile_text) VALUES('delete', ?, ?)",
            (row["rowid"], row["profile_text"]),
        )
        connection.execute("DELETE FROM profiles WHERE rowid = ?", (row["rowid"],))
        deleted += 1
    return deleted


def delete_item_rows(connection: sqlite3.Connection, item_ids: Iterable[str]) -> None:
    """Every row an item owns, across the metadata tables. Chunks and profiles go separately.

    Split from the two functions above because those two carry the FTS ordering constraint
    and these do not: a metadata table is an ordinary delete. Keeping them apart means the
    constrained path stays small enough to read in one screen.
    """
    for item_id in item_ids:
        for table in ("items", "item_topics", "item_content_kinds", "source_failures"):
            connection.execute(f"DELETE FROM {table} WHERE item_id = ?", (item_id,))  # noqa: S608
        connection.execute("DELETE FROM unfetched_links WHERE item_id = ?", (item_id,))
        connection.execute(
            "DELETE FROM surfaces WHERE owner_type = 'item' AND owner_id = ?", (item_id,)
        )


def tokenizer() -> str:
    """The tokenizer string this schema was created with — recorded in the manifest."""
    return FTS_TOKENIZE
