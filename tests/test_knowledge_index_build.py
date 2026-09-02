# tests/test_knowledge_index_build.py
"""`index build` and its manifest (Plan 02 §2, §3, steps 3, 4, 7c, 9, 10c).

THE MANIFEST IS THE CONTRACT BETWEEN A BUILD AND EVERY LATER QUERY. Spec §5.6 enumerates
what it must record, and the enumeration is asserted as a SET rather than field by field, so
a field dropped from the writer goes red instead of quietly disappearing from a JSON document
nobody reads until a query answers under the wrong chunker.

TWO SIGNALS, TWO COSTS, TWO PLACES (B3). `store_signal` is an `os.stat` — mtime and size of
`data/items.json` — and it is what a QUERY can afford on every call. `store_fingerprint` is a
sha256 per item and costs loading the store, so it is paid only by `build`, `update` and
`status`. The cheap one says *the store moved*; the expensive one says *which items changed,
and how many*. A false positive on the cheap one costs one warning; a false negative costs
serving stale evidence as fresh, so it fails towards the warning.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS
from xbrain.knowledge.index_schema import (
    SCHEMA_VERSION,
    IndexIncompatibleError,
    IndexMissingError,
    db_path,
    manifest_path,
    open_index,
)
from xbrain.knowledge.lexical import LexicalIndex
from xbrain.models import Item, Topic, TopicPage

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    return store, vocab, pages


@pytest.fixture()
def workspace(tmp_path: Path, corpus) -> Path:
    """A data/ directory holding the fixture store, so `store_signal` has a real file."""
    store, _vocab, _pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )
    return data


def _build(workspace: Path, corpus, **kwargs) -> index_build.BuildReport:
    store, vocab, pages = corpus
    return index_build.build(
        workspace / "index", store, vocab, pages, workspace / "items.json", **kwargs
    )


def _status(workspace: Path, store, corpus) -> index_build.StatusReport:
    """`status` takes the vocabulary and the pages like `build`/`update` do (H1)."""
    _store, vocab, pages = corpus
    return index_build.status(workspace / "index", store, vocab, pages, workspace / "items.json")


# ---------------------------------------------------------------------------
# 3 — the manifest records everything spec §5.6 asks for
# ---------------------------------------------------------------------------


def test_build_writes_a_manifest_with_every_field_the_spec_requires(workspace, corpus) -> None:
    """Step 3: asserted as a SET against `MANIFEST_FIELDS`, not one `in` per key.

    A key-by-key list is a second copy of the schema that drifts; the set assertion makes
    "the writer emits exactly what the contract declares" a single fact. Seen red by removing
    `chunker_version` from the writer, or from the declared set — either way the two stop
    agreeing.
    """
    report = _build(workspace, corpus)
    raw = json.loads(manifest_path(workspace / "index").read_text(encoding="utf-8"))
    assert set(raw) == index_build.MANIFEST_FIELDS
    assert raw["schema_version"] == SCHEMA_VERSION
    assert raw["surface_version"] and raw["chunker_version"]
    # The MEASURED parameters (Plan 02 §7's sweep), read from the module rather than written
    # here a second time: two literals that must agree are two definitions (rule 5), and the
    # point of the assertion is that the manifest records what the code USED.
    assert raw["chunker_params"] == {
        "target": DEFAULT_CHUNKER_PARAMS.target,
        "max_chars": DEFAULT_CHUNKER_PARAMS.max_chars,
        "overlap": DEFAULT_CHUNKER_PARAMS.overlap,
        "min_chars": DEFAULT_CHUNKER_PARAMS.min_chars,
    }
    # The hole Plan 03 fills. Declared now so its arrival is not a manifest migration.
    assert raw["embeddings"] is None
    assert raw["counts"]["items"] == len(corpus[0])
    assert raw["counts"]["chunks"] == report.chunks_written > 0
    assert raw["counts"]["profiles"] == report.profiles_written > 0
    assert raw["counts"]["topics"] == len(corpus[1])


def test_the_manifest_records_the_tokenizer_and_the_connective(workspace, corpus) -> None:
    """Beyond spec §5.6, and deliberately so: these two DECIDE every recall number.

    The connective change of Plan 01 M3 moved mean `recall@10` from 0.1429 to 0.8099 without
    touching one line of the chunker. A manifest that records the chunker version but not the
    query semantics would let two incomparable baselines look like the same measurement.
    """
    _build(workspace, corpus)
    raw = json.loads(manifest_path(workspace / "index").read_text(encoding="utf-8"))
    assert raw["tokenize"] == "unicode61 remove_diacritics 2"
    assert raw["connective"] == "OR"


def test_the_manifest_records_the_cheap_store_signal(workspace, corpus) -> None:
    """Step 7c (B3): mtime and size of `data/items.json` at build time.

    This is what lets a QUERY notice the store moved without loading 17 MB. Seen red by
    omitting it: `search` then has nothing to compare and `index_behind_store` can never fire.
    """
    _build(workspace, corpus)
    raw = json.loads(manifest_path(workspace / "index").read_text(encoding="utf-8"))
    stat = (workspace / "items.json").stat()
    assert raw["store_signal"] == {
        "items_json_mtime_ns": stat.st_mtime_ns,
        "items_json_size": stat.st_size,
    }


def test_the_skipped_counters_say_what_they_counted(workspace, corpus) -> None:
    """Spec §5.6 asks for *chunks omitted or failed*, and the honest answer names each cause.

    `decorative` and `no_speech` CAN be non-zero and are, on this fixture — an avatar and a
    silent video. `empty_text` is structurally 0 today because the emitter drops a blank
    surface at `_blank` before the index ever sees it, and the counter is kept, and said to
    be 0 for that reason, rather than quoted as a measurement of the corpus (rule 2).
    """
    _build(workspace, corpus)
    raw = json.loads(manifest_path(workspace / "index").read_text(encoding="utf-8"))
    assert set(raw["skipped"]) == {"empty_text", "decorative", "no_speech", "failed_sources"}
    assert raw["skipped"]["failed_sources"] >= 1, "the fixture has a failed fetch (k11)"
    assert raw["skipped"]["no_speech"] >= 1, "the fixture has a silent video (k09)"


# ---------------------------------------------------------------------------
# 4 — dry run
# ---------------------------------------------------------------------------


def test_build_dry_run_creates_no_database_and_no_manifest(workspace, corpus) -> None:
    """Step 4: `--dry-run` reports counts and writes nothing at all.

    Both files, not just the database: a manifest without a database is worse than neither,
    because a query would find it and trust it.
    """
    report = _build(workspace, corpus, dry_run=True)
    assert report.chunks_written > 0, "a dry run still reports what it WOULD write"
    assert not db_path(workspace / "index").exists()
    assert not manifest_path(workspace / "index").exists()


def test_a_dry_run_does_not_destroy_an_existing_index(workspace, corpus) -> None:
    """A `--dry-run` that DELETED the index is worse than one that rebuilt it.

    Found by measuring, not by reading: the first implementation opened the real database
    (creating it if absent), rolled the transaction back, and then removed the file it
    believed it had created — so running `index build --dry-run` against a working index
    destroyed it, silently, from the flag whose entire promise is that it changes nothing.

    The bytes are compared, not just the existence: a dry run that recreated an empty
    database would satisfy `exists()` and still have thrown the index away.
    """
    _build(workspace, corpus)
    before = db_path(workspace / "index").read_bytes()
    manifest_before = manifest_path(workspace / "index").read_text(encoding="utf-8")

    report = _build(workspace, corpus, dry_run=True)

    assert report.chunks_written > 0
    assert db_path(workspace / "index").read_bytes() == before
    assert manifest_path(workspace / "index").read_text(encoding="utf-8") == manifest_before


def test_a_forced_rebuild_produces_a_file_the_same_size_as_a_fresh_one(workspace, corpus) -> None:
    """`--force` starts from a NEW file, so the artefact is deterministic.

    Measured on the real corpus first: clearing the rows in place left SQLite's freelist
    behind, so a fresh build was 51.2 MB and the same index after five `--force` rebuilds was
    66.5 MB — and a `VACUUM` only recovered it to 60.6 MB. A derived artefact whose size
    depends on how many times it has been rebuilt is a derived artefact nobody can reason
    about, so `--force` unlinks first.
    """
    _build(workspace, corpus)
    fresh = db_path(workspace / "index").stat().st_size
    for _ in range(3):
        _build(workspace, corpus, force=True)
    assert db_path(workspace / "index").stat().st_size == fresh


def test_the_dry_run_prints_counts_not_text(workspace, corpus) -> None:
    """§12.7 (spec §10.8): logs never carry an article body.

    Asserted on the report OBJECT rather than on captured output, because the object is what
    the CLI renders and a test on the rendering would move with the wording.
    """
    report = _build(workspace, corpus, dry_run=True)
    for value in report.__dict__.values():
        assert not isinstance(value, str) or len(value) < 200


# ---------------------------------------------------------------------------
# 9 — a half-written index is not a green one
# ---------------------------------------------------------------------------


def test_an_interrupted_build_leaves_no_manifest_and_no_partial_rows(workspace, corpus) -> None:
    """Step 9: `Ctrl-C` mid-build aborts the transaction; `status` then says "incomplete".

    Simulated with a `KeyboardInterrupt` raised from the item writer, which is where a real
    interrupt lands. Seen red by dropping the explicit transaction: the rows written so far
    survive and the database looks like a small but valid index.
    """
    store, vocab, pages = corpus
    calls = {"n": 0}
    real = index_build.write_item

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    original = index_build.write_item
    index_build.write_item = explode  # type: ignore[assignment]
    try:
        with pytest.raises(KeyboardInterrupt):
            index_build.build(workspace / "index", store, vocab, pages, workspace / "items.json")
    finally:
        index_build.write_item = original  # type: ignore[assignment]

    assert not manifest_path(workspace / "index").exists(), (
        "the manifest must not exist: a query would trust it and answer from a partial index"
    )
    connection = open_index(db_path(workspace / "index"))
    assert connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_an_interrupted_forced_rebuild_leaves_no_manifest_behind(workspace, corpus) -> None:
    """C-1 (round 02, both gates): `--force` over an EXISTING index is the documented way to
    rebuild — every rebuild error names it — and it is the one shape the test above never
    exercised, because a fresh build has no previous manifest that could survive.

    HEAD unlinked the database and left the OLD manifest standing until the new one was
    written. Interrupt the rebuild and the transaction rolls the rows back, but the old
    manifest — same versions, same cheap signal — is still there, so `load_compatible_manifest`
    accepts it, `status` says nothing is wrong, and `search` answers "no results" over an
    EMPTY database: indistinguishable from a corpus with no matches. Measured on the real
    corpus (2,404 items, 2026-09-01): `old_manifest_survives=True`, `chunks_after_interrupt=0`,
    `status_incomplete=False`, `query_returned_normally=True` with 0 results.

    The manifest is what carries the safety property, so a forced rebuild removes it FIRST.
    Seen red on HEAD: the manifest survived, `status` said `incomplete=False`, and the
    `update` below "recovered" by inserting every item while dropping the whole topic plane.
    """
    store, vocab, pages = corpus
    _build(workspace, corpus)
    assert manifest_path(workspace / "index").exists()

    calls = {"n": 0}
    real = index_build.write_item

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    index_build.write_item = explode  # type: ignore[assignment]
    try:
        with pytest.raises(KeyboardInterrupt):
            _build(workspace, corpus, force=True)
    finally:
        index_build.write_item = real  # type: ignore[assignment]

    assert not manifest_path(workspace / "index").exists(), (
        "the OLD manifest survived a forced rebuild: a query would trust it over an empty base"
    )
    report = _status(workspace, store, corpus)
    assert report.incomplete is True
    assert "xbrain index build" in report.advice

    # The recovery path: `update` REFUSES (there is no manifest to be incremental against),
    # and a fresh `build` restores everything — including the topic plane C-3 lost.
    with pytest.raises(IndexMissingError):
        index_build.update(workspace / "index", store, vocab, pages, workspace / "items.json")
    _build(workspace, corpus)
    connection = open_index(db_path(workspace / "index"), read_only=True)
    try:
        assert connection.execute("SELECT COUNT(*) FROM topics").fetchone()[0] == len(vocab)
        assert (
            connection.execute("SELECT COUNT(*) FROM chunks WHERE owner_type = 'topic'").fetchone()[
                0
            ]
            > 0
        )
    finally:
        connection.close()


def test_status_calls_an_index_without_a_manifest_incomplete(workspace, corpus) -> None:
    """The other half of step 9: the state is REPORTED, not merely absent."""
    _build(workspace, corpus)
    manifest_path(workspace / "index").unlink()
    report = _status(workspace, corpus[0], corpus)
    assert report.incomplete is True
    assert "xbrain index build" in report.advice


# ---------------------------------------------------------------------------
# Rebuilding, and the guard against doing it by accident
# ---------------------------------------------------------------------------


def test_building_over_an_existing_index_requires_force(workspace, corpus) -> None:
    """A rebuild throws away an index that may have taken minutes; it is asked for once.

    The error names BOTH ways forward — `index update` for the incremental path and
    `--force` for the deliberate rebuild — because an error that names no command is a
    traceback with better manners.
    """
    _build(workspace, corpus)
    with pytest.raises(ValueError, match="--force"):
        _build(workspace, corpus)
    report = _build(workspace, corpus, force=True)
    assert report.chunks_written > 0


def test_build_is_deterministic(workspace, corpus) -> None:
    """Spec §3.7.8: two builds of the same store produce the same ids in the same order.

    Otherwise the `chunk_id` tie-break would silently reorder results between rebuilds, and
    the characterization fixture would be measuring the build rather than the scorer.
    """
    _build(workspace, corpus)
    first = _chunk_ids(workspace)
    _build(workspace, corpus, force=True)
    assert _chunk_ids(workspace) == first


def test_no_chunk_survives_without_its_surface(workspace, corpus) -> None:
    """The property a `FOREIGN KEY` would only have CLAIMED (see `index_schema`).

    SQLite ignores a `REFERENCES` clause unless `PRAGMA foreign_keys=ON`, so the constraint
    would have been decoration. Asserted over the real output of `build`, where it can be.
    """
    _build(workspace, corpus)
    connection = open_index(db_path(workspace / "index"), read_only=True)
    orphans = connection.execute(
        "SELECT COUNT(*) FROM chunks LEFT JOIN surfaces "
        "ON surfaces.surface_id = chunks.surface_id WHERE surfaces.surface_id IS NULL"
    ).fetchone()[0]
    assert orphans == 0


def test_the_profile_plane_holds_one_row_per_item_with_a_profile(workspace, corpus) -> None:
    """Spec §5.1: every item participates at BOTH levels.

    A profile per item, and the profile never becomes a chunk: asserted by checking that no
    chunk's `surface_id` mentions a profile, since the profile has no `surface_id` at all
    (Plan 01 forbids it having one).
    """
    _build(workspace, corpus)
    connection = open_index(db_path(workspace / "index"), read_only=True)
    profiles = connection.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    assert profiles == len(corpus[0])
    assert (
        connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE surface_id LIKE '%profile%'"
        ).fetchone()[0]
        == 0
    )


def test_the_metadata_tables_are_populated_for_every_filter(workspace, corpus) -> None:
    """The six filters that need a table have one, filled, after a real build (m2).

    Without this the filters pass their unit tests against hand-written rows and fail against
    the actual writer — the gap between "the query is right" and "the data is there".
    """
    _build(workspace, corpus)
    connection = open_index(db_path(workspace / "index"), read_only=True)
    for table in ("items", "item_topics", "item_content_kinds", "surfaces"):
        assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] > 0, table  # noqa: S608
    assert connection.execute("SELECT COUNT(*) FROM source_failures").fetchone()[0] >= 1
    assert connection.execute("SELECT COUNT(*) FROM unfetched_links").fetchone()[0] >= 1


def test_a_topic_surface_is_indexed_with_its_own_owner(workspace, corpus) -> None:
    """Topics are owners too (spec §3.6), and their chunks say so."""
    _build(workspace, corpus)
    connection = open_index(db_path(workspace / "index"), read_only=True)
    rows = connection.execute(
        "SELECT DISTINCT surface_type FROM chunks WHERE owner_type = 'topic'"
    ).fetchall()
    assert {row[0] for row in rows} & {"topic_description", "topic_overview", "topic_note"}


# ---------------------------------------------------------------------------
# 10c — status says HOW MANY items changed, not merely that something did
# ---------------------------------------------------------------------------


def test_status_reports_how_many_items_changed(workspace, corpus) -> None:
    """Step 10c: `status` is an explicit command, so it CAN afford to load the store.

    "Something changed" is not actionable — it does not distinguish a touched file from a
    hundred re-enriched items. So the fixture changes TWO items, removes TWO and adds TWO,
    and asserts `== 2` on each: the previous version changed one and removed one and
    asserted `== 1` / `== 0`, which a BOOLEAN satisfies — and its docstring claimed *seen red
    by reporting a boolean*, which was false (G-3, gate round 04: with `len(delta.x)` mutated
    to `int(bool(delta.x))` in `status` and `_update_report`, every count test stayed green).

    Seen red, for real this time, under that same mutation in an isolated copy: `2 == 1`.
    """
    store, vocab, pages = corpus
    _build(workspace, corpus)
    clean = _status(workspace, store, corpus)
    assert clean.items_changed == 0 and clean.items_added == 0 and clean.items_removed == 0

    def reenriched(item: Item) -> Item:
        return item.model_copy(
            update={
                "enriched": item.enriched.model_copy(
                    update={
                        "summary": f"un resumen completamente distinto para {item.id}",
                        "enriched_at": item.enriched.enriched_at + timedelta(hours=1),
                    }
                )
            }
        )

    changed = dict(store)
    changed["k02"] = reenriched(store["k02"])
    changed["k03"] = reenriched(store["k03"])
    del changed["k01"]
    del changed["k04"]
    changed["k98"] = store["k05"].model_copy(update={"id": "k98"})
    changed["k99"] = store["k06"].model_copy(update={"id": "k99"})
    after = _status(workspace, changed, corpus)
    assert after.items_changed == 2
    assert after.items_removed == 2
    assert after.items_added == 2
    assert "xbrain index update" in after.advice


def test_status_declares_topic_rows_that_do_not_hold_what_the_store_implies(
    workspace, corpus
) -> None:
    """H1's other half (gate Codex, round 04): the diagnostic instrument said HEALTHY.

    Two states, both of which `status` must name. First, the store moved an item's topic
    and nobody reindexed: `status` already counted the item, and it now also counts the
    TOPICS whose stored membership differs from what the store implies — two of them, since
    k02 leaves one and joins the other. Second, and the one that proves `status` reads the
    BASE rather than re-deriving everything from item fingerprints: every item fingerprint
    matches and a topic row is behind anyway — the state the pre-fix `update` produced on
    every topic move, and the state any index updated by that code is in today. The base is
    edited by hand to reach it, the way C-3 amputates the topic plane behind the manifest.
    The `update` it names repairs exactly that row and nothing else.

    Seen red before the fix: `advice == ''` and no `topics_changed` in the report.
    """
    from xbrain.knowledge.index_schema import db_path, open_index

    store, _vocab, _pages = corpus
    _build(workspace, corpus)
    clean = _status(workspace, store, corpus)
    assert clean.advice == ""

    changed = dict(store)
    changed["k02"] = store["k02"].model_copy(
        update={
            "enriched": store["k02"].enriched.model_copy(
                update={"primary_topic": "ai-policy", "topics": ["ai-policy"]}
            )
        }
    )
    moved = _status(workspace, changed, corpus)
    assert "xbrain index update" in moved.advice

    connection = open_index(db_path(workspace / "index"))
    with connection:
        connection.execute(
            "UPDATE topics SET primary_item_ids_json = '[]', stale = 1 WHERE slug = 'ai-policy'"
        )
    connection.close()
    behind = _status(workspace, store, corpus)
    assert "xbrain index update" in behind.advice, "status read the base and found it behind"
    assert behind.items_changed == 0, "every item fingerprint still matches"

    assert clean.topics_changed == 0
    assert (moved.items_changed, moved.topics_changed) == (1, 2)
    assert behind.topics_changed == 1

    report = index_build.update(
        workspace / "index", store, corpus[1], corpus[2], workspace / "items.json"
    )
    assert (report.items_changed, report.topics_refreshed) == (0, 1)
    repaired = _status(workspace, store, corpus)
    assert repaired.advice == "" and repaired.topics_changed == 0


def test_status_and_update_see_the_corruption_search_would_hit(workspace, corpus) -> None:
    """G-4's other half: the diagnostic instrument said HEALTHY over a base `search` crashed on.

    `_prove_readable` touched only `sqlite_master`, so with `chunks_fts_data` dropped `status`
    exited 0 with every count in place and `update --dry-run` returned normally — while
    `search` died with a `DatabaseError`. Two instruments, opposite answers on one state
    (rule 9), and the one that lies is the one an operator runs to find out. The probe runs
    at the open door, so the three see the same thing and name the same command.

    Seen red before the fix: `status` returned `incomplete=False` and `update` an
    `UpdateReport`.
    """
    store, vocab, pages = corpus
    _build(workspace, corpus)
    connection = sqlite3.connect(db_path(workspace / "index"))
    connection.execute("DROP TABLE chunks_fts_data")
    connection.commit()
    connection.close()

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force"):
        _status(workspace, store, corpus)
    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force"):
        index_build.update(
            workspace / "index", store, vocab, pages, workspace / "items.json", dry_run=True
        )


def test_status_reports_the_index_behind_the_store_from_the_cheap_signal(workspace, corpus) -> None:
    """B3: the mtime/size signal, read with an `os.stat`, on an explicit command too.

    A `touch` with no edit is a FALSE POSITIVE and that is accepted: the cost of one extra
    warning is a warning, and the cost of a false negative is serving stale evidence as
    fresh. It fails towards the warning, like `origin: unknown -> llm_synthesis`.
    """
    _build(workspace, corpus)
    assert _status(workspace, corpus[0], corpus).behind is False
    path = workspace / "items.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert _status(workspace, corpus[0], corpus).behind is True


# ---------------------------------------------------------------------------
# 29 — an incompatible manifest is never queried partially
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        # DERIVED from the constant, not a literal: the literal `"2"` stopped being foreign
        # the day C-3 bumped the schema to "2", and the test went red for a reason that had
        # nothing to do with what it pins — the shape of defect 5 in the execution report.
        ("schema_version", f"{SCHEMA_VERSION}-foreign"),
        ("surface_version", "xbrain-knowledge-surface/v9"),
        ("chunker_version", "xbrain-knowledge-chunker/v9"),
    ],
)
def test_an_incompatible_manifest_refuses_the_query_entirely(
    workspace, corpus, field: str, value: str
) -> None:
    """Step 29 / spec §9.3: *manifest incompatible: no se consulta parcialmente.*

    All three versions, because each invalidates a different thing and any one of them makes
    the stored rows mean something other than what the code would compute. Seen red by
    logging a warning and answering anyway.
    """
    _build(workspace, corpus)
    path = manifest_path(workspace / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        index_build.load_compatible_manifest(workspace / "index")


def test_a_corrupt_manifest_is_an_actionable_error_not_a_json_traceback(workspace) -> None:
    """Plan 02 §11: a corrupt base or manifest names `index build --force`; nothing is repaired."""
    (workspace / "index").mkdir(parents=True)
    manifest_path(workspace / "index").write_text("{not json", encoding="utf-8")
    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        index_build.load_compatible_manifest(workspace / "index")


# ---------------------------------------------------------------------------
# The fingerprint, and the population it has to reach (rule 6)
# ---------------------------------------------------------------------------


def test_the_item_fingerprint_changes_when_indexable_text_changes(corpus) -> None:
    """It hashes the SURFACES, not `(fetched_at, enriched_at)` as Plan 01 §10 sketched.

    Two reasons, and CLAUDE.md rule 6 is both of them. First, `content.fetched_at` cannot
    reach an item whose `content` is `None` — 960 of 2,404 in the real store (measured
    2026-09-01 on sha256 `f76341a3…`) — because there
    is nothing to stamp. Second, a timestamp is a PROXY: a summary edited by hand, or any
    repair that rewrites text without touching a clock, changes the indexable corpus and
    leaves the proxy unmoved, so the index would keep serving the old body under a
    fingerprint that says it is current.

    Hashing the emitted surface fingerprints answers the question directly — *did the text
    this index holds change?* — and it reaches every item, with or without `content`.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    before = index_build.item_fingerprint(item)
    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": "otro resumen"})}
    )
    assert index_build.item_fingerprint(edited) != before
    assert index_build.item_fingerprint(item) == before, "and it is stable"


def test_the_item_fingerprint_reaches_an_item_with_no_content(corpus) -> None:
    """Step 6 / rule 6: the invalidation signal must reach the population it has to reach.

    `k02` carries no `content` at all, so a fingerprint over `content.fetched_at` would be
    constant for it forever. Seen red by hashing only the timestamps.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    assert item.content is None, "this test is only meaningful on an item with no content"
    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": "otro resumen"})}
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


def test_the_item_fingerprint_also_covers_the_filterable_metadata(corpus) -> None:
    """The index stores more than text: a changed author or topic changes what a filter returns.

    Ignoring them would leave `--author` answering from an author the store no longer records.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    moved = item.model_copy(update={"source": "own_tweet"})
    assert index_build.item_fingerprint(moved) != index_build.item_fingerprint(item)


@pytest.mark.parametrize(
    "field, value",
    [
        ("author", {"handle": "someoneelse", "name": "Someone Else"}),
        ("title", "A different title"),
        ("language", "fr"),
        ("url", "https://x.com/othervoice/status/moved"),
    ],
)
def test_the_item_fingerprint_covers_what_the_index_stores_about_a_surface(
    corpus, field: str, value: object
) -> None:
    """G-5: every column the index STORES about a surface moves the fingerprint.

    `surface_fingerprint` is `(version, type, origin, text)` by design — two surfaces with
    the same text under different provenance must differ, and nothing more. But the index
    persists more than that in `surfaces`: attribution, title, url, locator and language,
    which `search` serves on every match (A-1) and `--has-surface` filters on. A change to
    any of them with the text untouched left `update` seeing nothing to do.

    Four axes, one parametrised test, each changing ONE field of k07's quoted post and
    nothing else. The url moves the locator too (`locator.url`), which is the point: the
    locator is what the consumer resolves the evidence through. `producer` is deliberately
    NOT here — the index has no producer column and `get` reads it from the configured
    command at read time, so hashing it would rewrite every ASR/VLM item on a binary
    rename for a field no query serves from the index.

    Seen red before the fix on all four: the fingerprint did not move.
    """
    from xbrain.models import Author

    store, _vocab, _pages = corpus
    item = store["k07"]
    position = next(i for i, s in enumerate(item.content.sources) if s.kind == "quoted_tweet")
    sources = list(item.content.sources)
    patch = {field: Author(**value) if field == "author" else value}
    sources[position] = sources[position].model_copy(update=patch)
    edited = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item), field


def test_the_fingerprint_hashes_the_same_row_the_writer_inserts(workspace, corpus) -> None:
    """G-5's binding half (rule 5): ONE projection of a surface, shared by the writer and the
    fingerprint, so a column added to `surfaces` cannot be stored without being hashed.

    Asserted by reading the rows back: for every surface of every item, the tuple the
    fingerprint hashes IS the row `_write_surfaces` wrote, column for column. Two lists that
    "should" match would be the divergence; one function is the binding.
    """
    store, _vocab, _pages = corpus
    _build(workspace, corpus)
    connection = open_index(db_path(workspace / "index"), read_only=True)
    try:
        for item in store.values():
            for surface in index_build.item_surfaces(item):
                row = connection.execute(
                    "SELECT surface_id, owner_type, owner_id, surface_type, origin, trust_class, "
                    "derived, attribution_handle, attribution_name, title, url, locator_json, "
                    "language, fingerprint, char_length FROM surfaces WHERE surface_id = ?",
                    (surface.surface_id,),
                ).fetchone()
                assert row is not None, surface.surface_id
                assert tuple(row) == index_build.surface_row(surface), surface.surface_id
    finally:
        connection.close()


def test_the_store_fingerprint_is_order_independent(corpus) -> None:
    """Two loads of the same store must agree, whatever order the dict happens to iterate in."""
    store, _vocab, _pages = corpus
    reversed_store = dict(reversed(list(store.items())))
    assert index_build.store_fingerprint(reversed_store) == index_build.store_fingerprint(store)


def test_the_store_signal_is_one_stat_and_nothing_else(workspace) -> None:
    """The cheap signal must stay cheap, or it is the expensive one with a different name."""
    signal = index_build.StoreSignal.of(workspace / "items.json")
    stat = (workspace / "items.json").stat()
    assert signal.items_json_mtime_ns == stat.st_mtime_ns
    assert signal.items_json_size == stat.st_size


def test_the_store_signal_of_a_missing_store_is_zeroed_not_an_exception(tmp_path: Path) -> None:
    """A query must still be able to say "the index is behind" when the store is gone.

    Raising here would turn a missing store into a crash inside `search`, which is the wrong
    place to learn it: the response declares the degradation instead.
    """
    signal = index_build.StoreSignal.of(tmp_path / "nope.json")
    assert signal == index_build.StoreSignal(items_json_mtime_ns=0, items_json_size=0)


# ---------------------------------------------------------------------------
# 28 — nothing here writes to the store
# ---------------------------------------------------------------------------


def test_build_does_not_touch_items_json(workspace, corpus) -> None:
    """Step 28 / acceptance 13, at the module level: a byte-for-byte comparison.

    `index build` reads the store and writes a derived artefact. A hash before and after is
    the only claim worth making, because "we do not call save_store" is a claim about the
    code and this is a claim about the file.
    """
    import hashlib

    path = workspace / "items.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    _build(workspace, corpus)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def _chunk_ids(workspace: Path) -> list[str]:
    connection = open_index(db_path(workspace / "index"), read_only=True)
    return [row[0] for row in connection.execute("SELECT chunk_id FROM chunks ORDER BY rowid")]


def test_the_index_answers_a_query_after_a_real_build(workspace, corpus) -> None:
    """End to end: build, then retrieve. The cheapest proof the writer and reader agree."""
    _build(workspace, corpus)
    index = LexicalIndex(open_index(db_path(workspace / "index"), read_only=True))
    hits = index.search("Quillfeather", limit=5)
    assert hits and all(hit.owner_id for hit in hits)


def test_the_manifest_built_at_is_an_instant_not_a_date(workspace, corpus) -> None:
    """A rebuild the same day must be distinguishable from the previous one."""
    _build(workspace, corpus)
    raw = json.loads(manifest_path(workspace / "index").read_text(encoding="utf-8"))
    built = datetime.fromisoformat(raw["built_at"])
    assert built.tzinfo is not None, "an instant with no timezone is a local guess"
    assert abs((datetime.now(timezone.utc) - built).total_seconds()) < 120
