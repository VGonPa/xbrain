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

**2. A delete RETRACTS the old text — and what matters is that it happens at all, not the
order (F-7).** With external content, `chunks_fts` stores no text: a `'delete'` command has
to be handed the original values, which is why both deletion functions SELECT the row before
touching anything. Because the values are captured first, running the two statements in the
other order is harmless — measured on sqlite 3.51.2, retract-then-delete and
delete-then-retract both leave zero rows behind. This file used to claim the opposite
(*"reversing these two statements produces phantom results"*), and the claim was the wrong
guard to give the next maintainer: the defect that actually exists is omitting the retraction,
which leaves an orphan FTS entry that an `INNER JOIN` HIDES until the rowid is reused — and
then the query returns a chunk whose body is a completely different one. That is what
`tests/test_knowledge_index_schema.py` pins, by reusing the rowid. The retraction lives in
exactly two functions (`delete_chunk_rows`, `delete_profile_rows`) and nowhere else.

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
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import NamedTuple

from xbrain.knowledge.lexical_fts import FTS_TOKENIZE, fts5_table_sql
from xbrain.models import _reject_local_path_traversal

# Bumped when the physical layout changes in a way an existing database cannot answer. The
# manifest records it and a mismatch refuses the query ENTIRELY (spec §9.3) — never a partial
# answer over a schema the code no longer understands.
# "2" since C-3/A-3: `items` carries the per-item omission counters, so the manifest's
# `skipped` is a SUM over rows the incremental path already maintains, rather than a figure
# carried over from the previous manifest and wrong after the first `update`.
# "3" since U-5 (round 07): `chunks.fingerprint` hashes the whole evidence projection
# (`chunking.chunk_evidence` — provenance, owner, position, attribution, narrowed locator),
# and a v2 base's fingerprints were computed over the text alone, so every one of its rows
# would fail verification. The door refuses it by name, with the rebuild, rather than
# answering «22,286 chunks excluded».
# "4" since the 02.6a2a review: `source_failures.attempts` is GONE. It was the one column
# across both failure planes with no field on the knowledge `SourceFailure`, so the versioned
# public projection and the DDL disagreed and `item_fingerprint`, which hashes the projection,
# could not see it — the shape review #161 exists to close. Of the two doors, ADDING the field
# would put fetch bookkeeping inside the hash, and `fetch._source_signature` excludes
# `attempts` for exactly that reason: a persistently-failing link re-fetched every run would
# re-index an item whose evidence did not move. It would also bump `EvidenceBundle`'s public
# version for a number no consumer asked for. So the column goes instead: the only writer that
# ever existed bound it to a literal `None` (`b61e04b:index_build.py:838`) and nothing has ever
# read it back. `tests/test_knowledge_index_build.py` binds the two DDLs to their projections
# by NAME, so neither side can move again without the other.
SCHEMA_VERSION = "4"

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
    store_fingerprint TEXT NOT NULL,
    -- What the emitter DECLINED for this item, counted where the item is (A-3). The
    -- manifest's `skipped` is a SUM over these, so an incremental update that deletes and
    -- rewrites the row keeps the total exact without re-walking the corpus.
    skipped_empty_text INTEGER NOT NULL DEFAULT 0,
    skipped_decorative INTEGER NOT NULL DEFAULT 0,
    skipped_no_speech  INTEGER NOT NULL DEFAULT 0
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
    http_status    INTEGER
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

    It is ALSO what a corrupt database raises (F-3). Both are the same operator situation —
    *what is on disk is not what this code can read, and the fix is to rebuild* — so they end
    with the same sentence rather than with two that have to be kept in step.
    """


# The advice every incompatibility ends with. ONE string, so the CLI, the services and the
# tests all name the same command; it lives here, beside the error that carries it, rather
# than in `index_build`, which cannot be imported from this module.
REBUILD_ADVICE = "Reconstruye el índice con `xbrain index build --force`."


def corrupt_base_error(path: Path, error: sqlite3.DatabaseError) -> IndexIncompatibleError:
    """The ONE sentence for a base this code cannot read — the open door and every later read.

    Built here, beside `REBUILD_ADVICE`, so `_prove_readable` and `reading_base` cannot
    drift into two wordings of one fact (rule 5). SQLite's own text rides in parentheses:
    it is what distinguishes a torn page from a file that is not a database at all.
    """
    return IndexIncompatibleError(
        f"La base del índice en {path} no se puede leer ({error}). {REBUILD_ADVICE}"
    )


@contextmanager
def reading_base(path: Path) -> Iterator[None]:
    """Turn a `sqlite3.DatabaseError` raised inside the block into the rebuild advice (D-1).

    `sqlite3.connect` is lazy and the open door reads page 1, `sqlite_master` and one `MATCH`
    per FTS plane — so a page damaged under `items`, `chunks` or `topics` surfaced at the
    first MAINTENANCE read that touched it: `count_rows`, `_stored_fingerprints`,
    `stored_topic_rows`, each a raw `DatabaseError` out of `status`, `search` and `update`,
    a 61-line traceback naming no command, while CLAUDE.md declared G-4 closed on the three
    (the round-06 gate, D-1). `LexicalIndex._fetch` already converts at QUERY time; this is
    the same conversion for the reads that are not queries, in one place, wrapped around
    them rather than remembered at each.

    `DatabaseError` is the whole family on purpose — `OperationalError` included — because
    the door already treats it so (`_fetch`, C-2): the index has no concurrent writer by
    design (spec §9.2), so what the family means here is *the file is not what this code
    can read*, and the honest remedy is the rebuild the sentence names.
    """
    try:
        yield
    except sqlite3.DatabaseError as error:
        raise corrupt_base_error(path, error) from error


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


def require_database(index_dir: Path) -> Path:
    """The database file, or the ONE actionable error for its absence (G-2).

    The advice depends on what is left beside the missing file. With no manifest the index
    was simply never built, and plain `xbrain index build` builds it. With a manifest still
    standing — the operator deleted the 52 MB database by hand and kept the 1 KB document —
    plain `build` REFUSES (*Ya existe un índice*), so naming it would send the operator into
    a dead end in two hops; the honest command is the forced rebuild. One function, called by
    every door — `open_index`, `update`, `open_for_query` and, since round 07 (U-2),
    `status` (`index_build._index_contents`), which used to test `exists()` by itself and
    read a standing manifest over a missing base as an index never built — so the doors
    cannot disagree on which command that is. `tests/test_knowledge_seams.py` enumerates the
    doors structurally and `test_knowledge_index_invalidation.py` swaps this function for a
    sentinel that every door must repeat.
    """
    database = db_path(index_dir)
    if database.exists():
        return database
    if manifest_path(index_dir).exists():
        raise IndexMissingError(
            f"No hay base de datos en {database} pero su manifest sigue en pie: el índice "
            f"quedó incompleto. {REBUILD_ADVICE}"
        )
    raise IndexMissingError(f"No hay índice en {index_dir}. Constrúyelo con `xbrain index build`.")


def open_index(path: Path, *, read_only: bool = False, create: bool = False) -> sqlite3.Connection:
    """Open the index database. Creates it ONLY when `create=True`, which only `build` passes.

    `read_only=True` opens `file:…?mode=ro`, so a stray write is an `OperationalError`
    instead of a silent repair — spec §5.6 is explicit that a query never modifies or repairs
    the index. A promise not to write is not the same property as being unable to.

    THE WRITE DOOR DOES NOT CREATE A FILE EITHER (G-2). It used to `mkdir + connect +
    create_schema` whenever the file was absent, which is how `index update --dry-run` left
    an EMPTY database under a manifest that was still standing: the consistency check then
    raised the right error with the file already on disk, and the next `search` — which
    checked `exists()` and a compatible manifest — answered «Sin resultados» with exit 0 over
    zero rows while `status` said incomplete. Measured on the real corpus through the CLI
    (2026-09-01): a 167,936-byte `knowledge.db` from a dry run. Creation is now opt-in and
    named, so the instrument that says "let me see what would happen" cannot change what
    happens next.
    """
    if read_only:
        require_database(path.parent)
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return _probe_fts(_verify_schema(_prove_readable(connection, path), path), path)
    existed = path.exists()
    if not existed and not create:
        require_database(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    _prove_readable(connection, path)
    if existed:
        # An EXISTING database is verified, never repaired: `create_schema` is idempotent,
        # so a dropped table would be silently re-created EMPTY and `index update` would
        # then carry on as if nothing had happened (C-2). A fresh file is the only one
        # that gets its schema created here.
        _probe_fts(_verify_schema(connection, path), path)
    create_schema(connection)
    return connection


def _prove_readable(connection: sqlite3.Connection, path: Path) -> sqlite3.Connection:
    """Force the first page read HERE, and turn a corrupt file into actionable advice (F-3).

    `sqlite3.connect` is LAZY: it opens a corrupt file without complaint and the
    `DatabaseError` surfaces later, at whatever statement happens to touch the disk first —
    inside `search`, inside `index status`, inside `index update`. `sqlite3.DatabaseError` is
    not an `IndexError_` and not an `OSError`, so it escaped the CLI's `_OPERATOR_ERRORS` and
    all three commands printed a raw traceback naming no command, which is precisely the row
    Plan 02 §11 tabulates as *error accionable con `index build --force`*.

    One probe, at the one door every path already goes through, so no caller has to remember.
    """
    try:
        connection.execute("SELECT count(*) FROM sqlite_master")
    except sqlite3.DatabaseError as error:
        connection.close()
        raise corrupt_base_error(path, error) from error
    return connection


class ColumnSpec(NamedTuple):
    """What `PRAGMA table_info` says about one column: the parts a query depends on."""

    type: str
    notnull: int
    pk: int


@cache
def declared_columns() -> dict[str, dict[str, ColumnSpec]]:
    """`{table: {column: spec}}` of the DDL THIS code ships — read, never restated (U-4).

    The reference is `create_schema` run on `:memory:` and read back through `PRAGMA
    table_info`, so the door verifies exactly the schema the code creates and the two cannot
    drift (rule 5): a column added to `_SCHEMA` is required of every base without anybody
    remembering to list it here. `pk` is in the spec because the m1 guard — `chunks.rowid`
    an explicit `INTEGER PRIMARY KEY`, or `VACUUM` may repoint every FTS entry — was «the
    DDL assertion and not a behavioural test», and a base whose `chunks` was recreated
    without it is one this code must not read. Cached: it costs one in-memory schema per
    process and the answer is a constant of the code.
    """
    connection = open_memory_index()
    try:
        return {
            table: {
                row["name"]: ColumnSpec(str(row["type"]), int(row["notnull"]), int(row["pk"]))
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            for table in sorted(TABLES | FTS_TABLES)
        }
    finally:
        connection.close()


def _verify_schema(connection: sqlite3.Connection, path: Path) -> sqlite3.Connection:
    """Refuse a database that lacks a table — or a COLUMN — this code queries (C-2, U-4).

    `_prove_readable` proves the FILE is SQLite; it says nothing about what is inside. Both
    round-03 gates dropped `chunks_fts` behind a healthy manifest and got a normal answer:
    `LexicalIndex._fetch` absorbed the `no such table` as "no rows", so `search` returned
    the profile-plane candidates — or nothing — with `degraded: ["no_embeddings"]` and not
    one word about the missing table, indistinguishable from a corpus that holds nothing.
    Spec §9.3 asks for an actionable error; Plan 02 §11 tabulates a base the code cannot
    read as *error accionable con `index build --force`*. Same operator situation as a
    corrupt file, same sentence.

    AND THE COLUMNS (U-4, round 07 — gate Codex F3). Table names were the whole check, so
    `ALTER TABLE surfaces DROP COLUMN attribution_name` — `quick_check: ok`, every table
    present — passed every door: `status` certified the base healthy, `update` re-sealed the
    manifest over it with exit 0, and `search` failed late in a raw `OperationalError: no
    such column` naming no command. The effective schema is the columns the queries read;
    they are compared against `declared_columns` — name, type, `NOT NULL` and the
    primary-key flag — so a missing or redefined column is refused here, with its name.
    Extra columns are tolerated: a base that holds more than this code reads is readable.

    Checked against `TABLES | FTS_TABLES`, the declared set, so a table added to the DDL is
    verified without anybody remembering to add it here.
    """
    present = {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    missing = sorted((TABLES | FTS_TABLES) - present)
    if missing:
        connection.close()
        raise IndexIncompatibleError(
            f"La base del índice en {path} está incompleta: faltan las tablas "
            f"{', '.join(missing)}. {REBUILD_ADVICE}"
        )
    absent, redefined = _column_drift(connection)
    if absent or redefined:
        connection.close()
        raise IndexIncompatibleError(
            f"La base del índice en {path} está incompleta: "
            + (f"faltan las columnas {', '.join(absent)}" if absent else "")
            + ("; " if absent and redefined else "")
            + (f"columnas con otra definición {', '.join(redefined)}" if redefined else "")
            + f". {REBUILD_ADVICE}"
        )
    return connection


def _column_drift(connection: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """`(absent, redefined)` — `table.column` names, sorted — against `declared_columns`."""
    absent: list[str] = []
    redefined: list[str] = []
    for table, columns in declared_columns().items():
        stored = {
            row["name"]: ColumnSpec(str(row["type"]), int(row["notnull"]), int(row["pk"]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for column, spec in columns.items():
            if column not in stored:
                absent.append(f"{table}.{column}")
            elif stored[column] != spec:
                redefined.append(f"{table}.{column}")
    return sorted(absent), sorted(redefined)


# One trivial `MATCH` per FTS plane — LITERALS keyed by table, like `_COUNT_STATEMENTS`, and
# a test asserts the key set equals `FTS_TABLES` so a plane added to the DDL is probed too.
_FTS_PROBES: dict[str, str] = {
    "chunks_fts": "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
    "profiles_fts": "SELECT count(*) FROM profiles_fts WHERE profiles_fts MATCH ?",
}
_PROBE_TERM = '"xbrain-index-probe"'


def _probe_fts(connection: sqlite3.Connection, path: Path) -> sqlite3.Connection:
    """Run the query `search` runs, so the OPEN door sees what the query would see (G-4).

    `_prove_readable` reads page 1 and `_verify_schema` reads `sqlite_master`; neither
    touches an FTS5 SHADOW table (`chunks_fts_data`, `…_idx`, `…_config`) or a page beyond
    the first. With `chunks_fts_data` dropped on the real corpus, `search` died with a
    68-line traceback ending in `DatabaseError: fts5: corruption found reading blob 10 from
    table "chunks_fts"`, while `status` — which never issues a `MATCH` — exited 0 and called
    the index healthy, and `update --dry-run` returned normally. The diagnostic instrument
    was the one that lied (rule 9).

    A `MATCH` on a term nobody indexed costs 0.01 ms per plane (measured on the 52 MB real
    index) and raises exactly where `search` would. `PRAGMA quick_check` would see more
    (152 ms) but reports the fts5 corruption as a ROW, not an exception, and a probe that
    has to parse its answer is a second guard to keep in step. What the probe still cannot
    reach — a corrupt segment a real query hits later — `LexicalIndex._fetch` turns into the
    same sentence.
    """
    try:
        for statement in _FTS_PROBES.values():
            connection.execute(statement, (_PROBE_TERM,)).fetchone()
    except sqlite3.DatabaseError as error:
        connection.close()
        raise IndexIncompatibleError(
            f"La base del índice en {path} no se puede consultar ({error}). {REBUILD_ADVICE}"
        ) from error
    return connection


def quick_check(connection: sqlite3.Connection) -> str:
    """`PRAGMA quick_check` as one sentence: `` when the base is sound, else the first finding.

    THE DIAGNOSTIC INSTRUMENT SEES MORE THAN THE OPEN DOOR (B-1, round 05). `_prove_readable`
    reads page 1, `_verify_schema` reads `sqlite_master`, `_probe_fts` runs one `MATCH` per
    FTS plane — and 16 KB of `0xff` written over pages 17–20 of the real 52 MB index sat where
    none of them looks: `quick_check` reported «btreeInitPage() returns error code 11» while
    `status` said `incomplete: false`. A query that touches the page still fails closed
    (`_fetch`, G-4), so no instrument lied; but `status` is the EXPLICIT command an operator
    runs to find out, and it can pay what this costs on the 52 MB real index where a query
    cannot: 155–167 ms measured by the gate; 362–850 ms, median 425, re-measured in round
    05 at load average 8.6 (5 runs, `quick_check(1)`, read-only connection). `status` runs it; the open door does not, on purpose — `quick_check` reports
    fts5 corruption as a ROW, not an exception, so the door keeps its positive `MATCH` probe
    and this reader parses the answer here, once.

    A `DatabaseError` raised by the pragma itself (a header too damaged to read) is the same
    fact and is returned as the sentence.
    """
    try:
        rows = connection.execute("PRAGMA quick_check(1)").fetchall()
    except sqlite3.DatabaseError as error:
        return str(error)
    first = str(rows[0][0]) if rows else ""
    return "" if first == "ok" else " ".join(first.split())


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

    THE RETRACTION IS THE WHOLE FUNCTION — not the order of its two statements (F-7).
    `chunks_fts` is external content and stores no text, so the `'delete'` command must be
    HANDED the original values; that is what the `SELECT` above is for, and it is why running
    the `DELETE` first is harmless (measured: both orders leave zero rows behind).

    What is NOT harmless is omitting the `'delete'`. The tokens then stay behind pointing at
    a rowid that no longer exists, the `INNER JOIN` hides the orphan for as long as nothing
    reuses that rowid, and the moment `index update` does, the query returns a chunk whose
    body is a different one entirely. Silent, and invisible to any test that only checks that
    the deleted word stops matching.

    Which is why the deletion lives in one function instead of at each call site: a call site
    that forgets the retraction is the failure, and there is nowhere to forget it from here.
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
    """Remove item profiles from both planes, retracting first and for the same reason."""
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

    The six statements are LITERALS rather than an f-string over a table-name tuple. The
    f-string version was safe — the names came from a module constant — but it made `bandit`
    report B608 and needed a suppression, and a suppression is a request to stop looking. Six
    literals need no argument from anybody.
    """
    statements = (
        "DELETE FROM items WHERE item_id = ?",
        "DELETE FROM item_topics WHERE item_id = ?",
        "DELETE FROM item_content_kinds WHERE item_id = ?",
        "DELETE FROM source_failures WHERE item_id = ?",
        "DELETE FROM unfetched_links WHERE item_id = ?",
        "DELETE FROM surfaces WHERE owner_type = 'item' AND owner_id = ?",
    )
    for item_id in item_ids:
        for statement in statements:
            connection.execute(statement, (item_id,))


def tokenizer() -> str:
    """The tokenizer string this schema was created with — recorded in the manifest."""
    return FTS_TOKENIZE
