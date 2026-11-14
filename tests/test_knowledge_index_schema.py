# tests/test_knowledge_index_schema.py
"""The persisted index schema (Plan 02 §2, steps 1, 1b, 2).

THREE PROPERTIES CARRY THE WEIGHT HERE, and each one fails SILENTLY when it is wrong —
which is why they are pinned by behaviour rather than by reading the DDL back.

* **`chunks.rowid` is an explicit `INTEGER PRIMARY KEY` (m1).** `chunks_fts` is an FTS5
  table of EXTERNAL content: it stores no text and reads it from `chunks` by rowid. SQLite
  documents that `VACUUM` may renumber the rowids of a table that has no `INTEGER PRIMARY
  KEY` — so an implicit rowid would leave every FTS entry pointing at a different chunk,
  returning wrong text with no error at all. Guarded STRUCTURALLY, because *may* is not
  *does*: measured on this interpreter, `VACUUM` did not renumber either way, so a
  behavioural test would pass for a reason unrelated to the property.
* **The read-only connection really is read-only.** Spec §3.7.12: `search`, `get` and `index
  status` are read operations, and §5.6 says a query never repairs the index silently. A
  connection opened read-write "but we promise not to write" is a promise, not a property.
* **`data/index/` is git-ignored**, asserted through `git check-ignore` rather than by
  reading `.gitignore` — the file has patterns and negations, and what matters is git's own
  answer.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from xbrain.knowledge.index_schema import (
    FTS_TABLES,
    SCHEMA_VERSION,
    TABLES,
    create_schema,
    db_path,
    manifest_path,
    open_index,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    create_schema(connection)
    return connection


# ---------------------------------------------------------------------------
# 1 — the DDL creates every table and both FTS planes
# ---------------------------------------------------------------------------


def test_the_ddl_creates_every_declared_table() -> None:
    """Step 1: every table of Plan 02 §2 exists after `create_schema`.

    Asserted against the DECLARED set (`TABLES`), not against a literal list written twice:
    a table added to the DDL without being declared, or declared without being created, goes
    red. Seen red by deleting `profiles_fts` from the DDL.
    """
    connection = _connection()
    present = {
        row["name"]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert TABLES <= present
    assert FTS_TABLES <= present


def test_every_fts_plane_has_an_open_door_probe() -> None:
    """G-4's totality half: a plane added to the DDL without a probe goes red here.

    The probes are literals keyed by table (bandit reads an f-string over a table name as
    B608, and a suppression is a request to stop looking), so the key set is asserted equal
    to `FTS_TABLES` the way `_COUNT_STATEMENTS` is asserted against the counted planes.
    """
    from xbrain.knowledge.index_schema import _FTS_PROBES

    assert set(_FTS_PROBES) == FTS_TABLES


def test_the_two_fts_planes_are_separate_tables() -> None:
    """Spec §5.1: the profile finds the ITEM, the chunks find the FRAGMENT.

    They are separate tables on purpose — Plan 01 forbids the profile from ever being a
    citation, and one shared table would let a profile row surface as a `SearchMatch`.
    """
    assert FTS_TABLES == frozenset({"chunks_fts", "profiles_fts"})


def test_the_schema_version_is_declared() -> None:
    """The manifest records it, and a mismatch refuses the query entirely (spec §9.3).

    "2" since C-3/A-3: `items` gained the three per-item omission columns the manifest's
    `skipped` is summed from, and a v1 base has no such columns. "3" since U-5 (round 07):
    `chunks.fingerprint` hashes the whole evidence projection — provenance, ownership,
    position, attribution, narrowed locator — and a v2 base's fingerprints were computed
    over the text alone, so every row of it would fail verification: the door must refuse
    it by name rather than answer «22,286 chunks excluded». "4" since the 02.6a2a review
    dropped the dead `source_failures.attempts` column so the DDL and the versioned public
    `SourceFailure` hold the same fields. The pin exists so a layout change cannot ship
    without the bump that makes an existing index refuse the query.
    """
    assert SCHEMA_VERSION == "4"


# ---------------------------------------------------------------------------
# 1b — the rowid cannot be implicit
# ---------------------------------------------------------------------------


def test_chunks_rowid_is_an_explicit_integer_primary_key() -> None:
    """m1, and THIS is the test that guards it — not the VACUUM round-trip below.

    SQLite documents that `VACUUM` **may** renumber the rowids of a table that is not
    declared `INTEGER PRIMARY KEY`. *May*. Measured on this interpreter (sqlite 3.51.2, 20
    rows, half deleted, then `VACUUM`): it did **not** renumber, with either declaration. So
    a behavioural test that waits for the renumbering to happen is a test that passes for a
    reason unrelated to the property — CLAUDE.md rule 1 — and would keep passing on the day a
    different build, a different page layout or a bigger table did renumber.

    The structural assertion is reliable and it is the guard. Seen red by declaring
    `chunk_id TEXT PRIMARY KEY`: `rowid` is then not a declared column at all.
    """
    connection = _connection()
    columns = {row["name"]: row for row in connection.execute("PRAGMA table_info(chunks)")}
    assert "rowid" in columns, (
        "`chunks.rowid` must be DECLARED: an implicit rowid may be renumbered by VACUUM, "
        "silently repointing every external-content FTS entry at a different chunk"
    )
    assert columns["rowid"]["pk"] == 1
    assert columns["rowid"]["type"] == "INTEGER"
    assert columns["chunk_id"]["pk"] == 0
    profile_columns = {
        row["name"]: row for row in connection.execute("PRAGMA table_info(profiles)")
    }
    assert profile_columns["rowid"]["pk"] == 1, "the profile plane has the same trap"


def test_the_index_still_answers_the_same_after_a_vacuum(tmp_path: Path) -> None:
    """A regression guard, and its docstring says exactly what it does and does not prove.

    It proves the index survives a `VACUUM` on a FILE database with reclaimed pages: the same
    query returns the same `chunk_id`s. It does NOT prove the declaration above is what makes
    that true — see that test for the measurement. Kept because the failure it would catch
    (an index that stops answering after a compaction) is real and cheap to check.
    """
    path = tmp_path / "knowledge.db"
    connection = open_index(path, create=True)
    for index in range(1, 21):
        _insert_chunk(connection, f"c{index:02d}", f"quillfeather body number {index}")
    connection.executescript(
        "INSERT INTO chunks_fts(chunks_fts, rowid, text, title) "
        "SELECT 'delete', rowid, text, title FROM chunks WHERE chunk_id < 'c10';"
        "DELETE FROM chunks WHERE chunk_id < 'c10';"
    )
    connection.commit()

    def ranked() -> list[str]:
        return [
            row[0]
            for row in connection.execute(
                "SELECT chunks.chunk_id FROM chunks_fts JOIN chunks "
                "ON chunks.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ? "
                "ORDER BY chunks.chunk_id",
                ('"quillfeather"',),
            )
        ]

    before = ranked()
    assert before, "the fixture must retrieve something before the vacuum"
    connection.execute("VACUUM")
    assert ranked() == before


def test_a_deleted_chunk_leaves_no_tokens_behind_for_a_reused_rowid(tmp_path: Path) -> None:
    """Step 7b, ASSERTING THE FAILURE THAT ACTUALLY HAPPENS — not the one the plan sketched.

    Plan 02 §3 says the `'delete'` command must be issued BEFORE the row is removed, and
    tests it by deleting a chunk and querying for a word that lived only in it. Written that
    way the test passes with the two statements in EITHER order (verified), because
    `delete_chunk_rows` reads the old values BEFORE deleting and hands them to FTS5 as
    parameters — the `'delete'` command never re-reads the content table, so once you have
    captured the values the ordering constraint is gone. A test that cannot come out any
    other way is not a test (rule 2).

    The failure that IS real is omitting the retraction. FTS5 keeps the tokens under the old
    rowid; an `INNER JOIN` hides that while the rowid is unused, and then a LATER chunk lands
    on the reused rowid and starts matching a word it never contained. Measured: with the
    retraction dropped, a query for `marrowgate` returns `c2`, whose body is *"a totally
    different body"*.

    Seen red by deleting the `'delete'` statement from `delete_chunk_rows`.
    """
    from xbrain.knowledge.index_schema import delete_chunk_rows

    connection = open_index(tmp_path / "knowledge.db", create=True)
    _insert_chunk(connection, "c1", "the marrowgate protocol is documented here")
    connection.commit()
    (freed,) = connection.execute("SELECT rowid FROM chunks WHERE chunk_id = 'c1'").fetchone()

    delete_chunk_rows(connection, ["c1"])
    connection.execute(
        "INSERT INTO chunks (rowid, chunk_id, surface_id, text) VALUES (?,?,?,?)",
        (freed, "c2", "item:x:post:c2", "a totally different body"),
    )
    connection.execute(
        "INSERT INTO chunks_fts (rowid, text, title) VALUES (?,?,?)",
        (freed, "a totally different body", ""),
    )
    connection.commit()

    rows = connection.execute(
        "SELECT chunks.chunk_id FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ?",
        ('"marrowgate"',),
    ).fetchall()
    assert rows == [], f"a chunk that never held the term matched it: {[r[0] for r in rows]}"
    assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1


def test_deleting_a_profile_removes_its_terms_from_the_index(tmp_path: Path) -> None:
    """The same order, the same trap, on the OTHER plane.

    `profiles_fts` is external content over `profiles` for exactly this reason: Plan 02 §2
    sketched it as `content=''`, and a contentless FTS5 table cannot be deleted from without
    the ORIGINAL text, which a contentless table by definition does not keep. Storing the
    profile text in `profiles` makes incremental deletion possible at all, and makes it the
    same operation as on the chunk plane instead of a second, subtly different one.

    Asserted on `profiles_fts` DIRECTLY rather than through a join, so an orphan entry is
    visible here instead of being hidden until a rowid is reused. Seen red by dropping the
    `'delete'` statement from `delete_profile_rows`.
    """
    from xbrain.knowledge.index_schema import delete_profile_rows

    connection = open_index(tmp_path / "knowledge.db", create=True)
    _insert_profile(connection, "i1", "zephyrine appears only in this profile")
    _insert_profile(connection, "i2", "an ordinary profile")
    connection.commit()

    delete_profile_rows(connection, ["i1"])
    connection.commit()

    rows = connection.execute(
        "SELECT rowid FROM profiles_fts WHERE profiles_fts MATCH ?", ('"zephyrine"',)
    ).fetchall()
    assert rows == []
    assert connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Read-only is a property, not a promise (spec §3.7.12, §5.6)
# ---------------------------------------------------------------------------


def test_a_read_only_connection_refuses_to_write(tmp_path: Path) -> None:
    """Spec §5.6: *the query operation does not modify or repair the index silently.*

    Opened with `file:…?mode=ro`, so an accidental write is an `OperationalError` rather
    than a repair nobody asked for. Seen red by opening it read-write.
    """
    path = tmp_path / "knowledge.db"
    open_index(path, create=True).close()
    connection = open_index(path, read_only=True)
    with pytest.raises(sqlite3.OperationalError):
        connection.execute("INSERT INTO chunks (chunk_id, surface_id, text) VALUES ('x','y','z')")


def test_opening_a_missing_index_read_only_is_an_actionable_error(tmp_path: Path) -> None:
    """Step 30: an absent index names `xbrain index build`, never a raw traceback."""
    from xbrain.knowledge.index_schema import IndexMissingError

    with pytest.raises(IndexMissingError, match="xbrain index build"):
        open_index(tmp_path / "nope.db", read_only=True)


def test_opening_a_missing_database_for_writing_does_not_create_it(tmp_path: Path) -> None:
    """G-2's second closure: the WRITE door does not create a file outside `build`.

    `open_index(path)` used to `mkdir + connect + create_schema` whenever the file was
    absent, which is what let `index update --dry-run` leave an empty base under a standing
    manifest. Creation is now an explicit `create=True`, which only `build` passes; every
    other writer finds the absence and names the command. And the advice depends on what
    is left: with a manifest beside the missing file, plain `xbrain index build` REFUSES
    (a manifest exists), so the sentence has to name `--force`.

    Seen red before the fix: the file existed after the first call.
    """
    from xbrain.knowledge.index_schema import IndexMissingError, manifest_path

    path = tmp_path / "index" / "knowledge.db"
    with pytest.raises(IndexMissingError, match="xbrain index build"):
        open_index(path)
    assert not path.exists() and not path.parent.exists()

    path.parent.mkdir()
    manifest_path(path.parent).write_text("{}", encoding="utf-8")
    with pytest.raises(IndexMissingError, match="xbrain index build --force"):
        open_index(path)
    assert not path.exists()

    open_index(path, create=True).close()
    assert path.exists()


# ---------------------------------------------------------------------------
# U-4 (round 07) — the schema the door verifies is the EFFECTIVE one: columns too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "table, column",
    # `chunks.trust_class` rather than `chunks.origin`: SQLite refuses to drop a column an
    # index depends on, and `chunks_origin` is one — the staging must survive, or nothing
    # is tested.
    [("surfaces", "attribution_name"), ("chunks", "trust_class"), ("items", "store_fingerprint")],
)
def test_a_base_missing_a_column_this_code_reads_is_refused_at_the_door(
    tmp_path: Path, table: str, column: str
) -> None:
    """Gate Codex F3 (round 07): `_verify_schema` compared TABLE names, so a base with a
    column dropped (`ALTER TABLE surfaces DROP COLUMN attribution_name`, `quick_check: ok`)
    passed the door — `status` certified it healthy, `update` re-sealed the manifest over
    it with exit 0, and `search` died late in a raw `OperationalError: no such column`.
    The schema this code needs is the DDL it ships, column by column; the reference is
    read from that DDL on a `:memory:` connection, never restated, and a column missing
    from the base names the table, the column and the rebuild.

    Seen red on `9dfa34e`: the door opened.
    """
    from xbrain.knowledge.index_schema import REBUILD_ADVICE, IndexIncompatibleError

    path = tmp_path / "knowledge.db"
    open_index(path, create=True).close()
    connection = sqlite3.connect(path)
    connection.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    connection.commit()
    connection.close()

    for open_ in (lambda: open_index(path, read_only=True), lambda: open_index(path)):
        with pytest.raises(IndexIncompatibleError) as refused:
            open_()
        assert f"{table}.{column}" in str(refused.value)
        assert REBUILD_ADVICE in str(refused.value)


def test_a_chunks_table_without_its_explicit_rowid_is_refused_at_the_door(tmp_path: Path) -> None:
    """The m1 guard, MECHANISED. CLAUDE.md says the explicit `INTEGER PRIMARY KEY` on
    `chunks` is «the DDL assertion and not a behavioural test that would pass for the wrong
    reason» — a `VACUUM` MAY renumber an implicit rowid and repoint every FTS entry. The
    column comparison sees the `pk` flag, so a base whose `chunks` lost the explicit key
    (recreated by hand, or by a tool that rewrote the table) is refused before a query
    could read text under the wrong rowid.
    """
    from xbrain.knowledge.index_schema import IndexIncompatibleError

    path = tmp_path / "knowledge.db"
    open_index(path, create=True).close()
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE chunks_implicit AS SELECT * FROM chunks WHERE 0;
        DROP TABLE chunks;
        ALTER TABLE chunks_implicit RENAME TO chunks;
        """
    )
    connection.commit()
    connection.close()

    with pytest.raises(IndexIncompatibleError, match="chunks.rowid"):
        open_index(path, read_only=True)


def test_the_declared_columns_are_read_from_the_ddl_this_code_ships() -> None:
    """The reference has ONE definition — `create_schema` on `:memory:` — so the door and
    the DDL cannot drift (rule 5): every declared table is in it, the two FTS planes have
    their columns, and `chunks.rowid` carries the primary-key flag the m1 guard depends on.
    """
    from xbrain.knowledge.index_schema import declared_columns

    reference = declared_columns()
    assert set(reference) == TABLES | FTS_TABLES
    assert set(reference["chunks_fts"]) == {"text", "title"}
    assert reference["chunks"]["rowid"].pk == 1
    assert reference["profiles"]["rowid"].pk == 1
    assert reference["surfaces"]["attribution_name"].notnull == 0


# ---------------------------------------------------------------------------
# 2 — the index is derived data and lives outside Git
# ---------------------------------------------------------------------------


def test_the_index_directory_is_git_ignored() -> None:
    """Step 2: `data/index/` is derived and reconstructible, and never versioned (spec §5.6).

    Asked of GIT, not of `.gitignore`: the file has patterns and a negation (`!data/.gitkeep`)
    and what matters is git's own verdict. Seen red by adding `!data/index/`.
    """
    result = subprocess.run(  # noqa: S603
        ["git", "check-ignore", "-q", "data/index/manifest.json"],  # noqa: S607
        cwd=REPO_ROOT,
        check=False,
    )
    assert result.returncode == 0, "data/index/ must be ignored by git"


def test_the_index_paths_are_derived_from_the_index_directory(tmp_path: Path) -> None:
    """One place decides where the database and the manifest live."""
    assert db_path(tmp_path) == tmp_path / "knowledge.db"
    assert manifest_path(tmp_path) == tmp_path / "manifest.json"


def test_an_index_directory_outside_the_data_root_is_rejected(tmp_path: Path) -> None:
    """§12.6 (m8): rejecting `..` is NOT the same as checking containment.

    `_reject_local_path_traversal` catches a literal `..`; a SYMLINK pointing outside passes
    it and fails only a real containment check. So `resolve_index_dir` resolves the path and
    asserts `is_relative_to(data_dir.resolve())`. Seen red by dropping the `is_relative_to`
    call: the symlink case then builds a database outside `data/`.
    """
    from xbrain.knowledge.index_schema import resolve_index_dir

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (data_dir / "escape").symlink_to(outside)

    assert resolve_index_dir(data_dir, "index") == (data_dir / "index").resolve()
    with pytest.raises(ValueError, match="fuera de"):
        resolve_index_dir(data_dir, "escape")
    with pytest.raises(ValueError, match="relativ"):
        resolve_index_dir(data_dir, "/etc")
    with pytest.raises(ValueError, match=r"\.\."):
        resolve_index_dir(data_dir, "../elsewhere")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _insert_chunk(connection: sqlite3.Connection, chunk_id: str, text: str) -> None:
    cursor = connection.execute(
        "INSERT INTO chunks (chunk_id, surface_id, text) VALUES (?,?,?)",
        (chunk_id, f"item:x:post:{chunk_id}", text),
    )
    connection.execute(
        "INSERT INTO chunks_fts (rowid, text, title) VALUES (?,?,?)",
        (cursor.lastrowid, text, ""),
    )


def _insert_profile(connection: sqlite3.Connection, item_id: str, text: str) -> None:
    cursor = connection.execute(
        "INSERT INTO profiles (item_id, profile_text) VALUES (?,?)", (item_id, text)
    )
    connection.execute(
        "INSERT INTO profiles_fts (rowid, profile_text) VALUES (?,?)", (cursor.lastrowid, text)
    )
