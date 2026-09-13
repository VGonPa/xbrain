# tests/test_knowledge_index_invalidation.py
"""`index update` — the incremental path (Plan 02 §3, steps 5, 6, 7, 7b, 8).

THIS IS THE HEART OF THE PLAN, and the reason is that indexing is MANUAL BY DECISION (spec
§9.2). The expected failure is therefore not corruption: it is *you ran `enrich` and did not
reindex*. Everything here exists to make that state visible instead of silently answered.

FOUR SITUATIONS, FOUR ACTIONS (spec §5.6):

| a chunk is new              | inserted                                        |
| its fingerprint changed     | delete + insert, one transaction, one order     |
| its owner disappeared       | deleted, same order                             |
| the chunker version moved   | `update` REFUSES and names `index build --force`|

The last one is the least obvious and the most important. A chunker bump means every stored
chunk was cut differently from how the code would cut it now; an incremental update would
leave the corpus half in one version and half in the other, and every id would still
resolve — so nothing would raise, and the ranking would be a blend of two chunkers.

EVERY DOOR HERE TAKES AN `IndexInputs`, NEVER LOOSE OBJECTS AND A PATH (P1b). The three files
are written to disk first and read back through `load_index_inputs`, which is what the command
does, so the cheap signal a test observes is the signal of the bytes the delta was computed
over. A helper that passed an in-memory store beside a path would be testing a call shape this
tree does not have.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from xbrain.knowledge import index_build, index_schema, index_store
from xbrain.knowledge.chunking import ChunkerParams
from xbrain.knowledge.get_service import get
from xbrain.knowledge.index_schema import (
    FTS_TABLES,
    REBUILD_ADVICE,
    TABLES,
    IndexIncompatibleError,
    IndexMissingError,
    db_path,
    manifest_path,
    open_index,
)
from xbrain.knowledge.lexical import LexicalIndex
from xbrain.knowledge.profile import profile_text
from xbrain.knowledge.search_service import QueryContext
from xbrain.knowledge.surfaces import item_surfaces, knowledge_item
from xbrain.models import Author, Content, ContentSourceSuccess, Item, Topic, TopicPage
from xbrain.rubrics import save_vocab
from xbrain.store import save_store, save_topic_pages

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


@pytest.fixture()
def built(tmp_path: Path, corpus) -> Path:
    """A data/ with ALL THREE inputs on disk and a freshly built index over them."""
    store, vocab, pages = corpus
    data = tmp_path / "data"
    _persist(data, store=store, vocab=vocab, pages=pages)
    index_build.build(data / "index", _inputs(data))
    return data


def _paths(data: Path) -> tuple[Path, Path, Path]:
    """The three inputs, in the order every door of this module takes them."""
    return data / "items.json", data / "vocab.yaml", data / "topics.json"


def _persist(data: Path, *, store=None, vocab=None, pages=None) -> None:
    """Write the planes a test moved, through the store's OWN writers."""
    items_path, vocab_path, topics_path = _paths(data)
    if store is not None:
        save_store(store, items_path)
    if vocab is not None:
        save_vocab(vocab, vocab_path)
    if pages is not None:
        save_topic_pages(pages, topics_path)


def _inputs(data: Path) -> index_build.IndexInputs:
    return index_build.load_index_inputs(*_paths(data))


def _update(data: Path, **kwargs) -> index_build.UpdateReport:
    return index_build.update(data / "index", _inputs(data), **kwargs)


def _status(data: Path, **kwargs) -> index_build.StatusReport:
    """`status` takes the same snapshot `build`/`update` do (H1)."""
    return index_build.status(data / "index", _inputs(data), **kwargs)


def _edit_summary(item: Item, text: str) -> Item:
    """The change `enrich` actually makes: a new summary and a new `enriched_at`."""
    assert item.enriched is not None
    return item.model_copy(
        update={
            "enriched": item.enriched.model_copy(
                update={
                    "summary": text,
                    "enriched_at": item.enriched.enriched_at + timedelta(hours=1),
                }
            )
        }
    )


def _reassign(item: Item, slug: str) -> Item:
    """The OTHER change `enrich` makes: a new topic assignment, text untouched (H1)."""
    assert item.enriched is not None
    return item.model_copy(
        update={
            "enriched": item.enriched.model_copy(update={"primary_topic": slug, "topics": [slug]})
        }
    )


def _topic_rows(data: Path) -> dict[str, tuple[list[str], list[str], int]]:
    """`{slug: (primary ids, secondary ids, stale)}` as the BASE holds them."""
    return {
        slug: (json.loads(primary), json.loads(secondary), stale)
        for slug, primary, secondary, stale in _rows(
            data,
            "SELECT slug, primary_item_ids_json, secondary_item_ids_json, stale FROM topics "
            "ORDER BY slug",
        )
    }


# The one profile-plane query these tests ask, written once.
_FTS = "SELECT COUNT(*) FROM profiles_fts WHERE profiles_fts MATCH ?"


def _rows(data: Path, sql: str, *params) -> list:
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        return connection.execute(sql, params).fetchall()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 5 — no changes, no writes
# ---------------------------------------------------------------------------


def test_update_with_no_changes_writes_nothing(built: Path, corpus) -> None:
    """Step 5 / acceptance 3: zero inserts, zero deletes.

    Asserted on the REPORT and on the row identity: a rebuild that happened to produce the
    same ids would satisfy a count-only assertion while having rewritten everything. Seen red
    by dropping the fingerprint comparison — every item then re-indexes on every run.

    THE TOPIC PLANE IS COMPARED BY ROWID, AND `SELECT *` WAS NOT ENOUGH. `_write_topic_row`
    issues `INSERT OR REPLACE`, which on a `slug TEXT PRIMARY KEY` table DELETES and
    re-inserts — the values come back identical and the rowid MOVES (measured: 1 -> 2). So a
    refresh that rewrote every topic row on a no-op run was invisible to a `SELECT *`
    comparison, and `topics_refreshed` stayed 0 because it counts `len(behind)`, not writes.
    The rowid is the cheap witness; `test_a_no_op_update_issues_no_write_against_the_topic_plane`
    is the direct one.
    """
    before = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")
    topics_before = _rows(built, "SELECT rowid, * FROM topics ORDER BY slug")

    report = _update(built)

    assert (report.items_added, report.items_changed, report.items_removed) == (0, 0, 0)
    assert (report.chunks_inserted, report.chunks_deleted) == (0, 0)
    assert _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid") == before
    # The topic ROWS too (H1): the membership refresh compares before it writes, so a
    # store that did not move rewrites no topic row either.
    assert _rows(built, "SELECT rowid, * FROM topics ORDER BY slug") == topics_before
    assert report.topics_refreshed == 0


# ---------------------------------------------------------------------------
# 8 — a chunker bump is not an incremental update
# ---------------------------------------------------------------------------


def test_a_different_chunker_version_refuses_the_update(built: Path, corpus) -> None:
    """Step 8: `update` refuses and names `index build --force`.

    An incremental update under a new chunker leaves the corpus half-cut one way and half the
    other. Every id still resolves, so nothing raises — the ranking simply becomes a blend of
    two chunkers, which is unrecoverable after the fact because nothing records which half is
    which. Seen red by letting it continue.
    """
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chunker_version"] = "xbrain-knowledge-chunker/v99"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        _update(built)


def test_a_base_stamped_with_the_pre_graph_schema_refuses_the_update(built: Path, corpus) -> None:
    """Plan 04.2: `graph_edges` is a layout change, so a base sealed as "4" is not incremental.

    A v4 base has no graph plane, and updating over it would re-seal a manifest certifying a
    graph that was never written. Seen red with `SCHEMA_VERSION` still "4": the same stamp
    compared equal and the update ran.
    """
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "4"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        _update(built)


# ---------------------------------------------------------------------------
# 6 — the item with no `content` (rule 6)
# ---------------------------------------------------------------------------


def test_update_detects_a_summary_change_on_an_item_with_no_content(built: Path, corpus) -> None:
    """Step 6: 960 of 2,404 real items have NO `content`, so `fetched_at` reaches none of them.

    (Measured 2026-09-01 on `data/items.json`, sha256 `f76341a3…`. The claim holds with either
    number; 961 was one item stale.)

    CLAUDE.md rule 6 in its exact form: *check the invalidation signal actually reaches the
    population being repaired*. Seen red by fingerprinting `content.fetched_at` alone — this
    item then never changes, whatever is done to its summary.
    """
    store, _vocab, _pages = corpus
    assert store["k02"].content is None
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "resumen nuevo para un item sin content")
    _persist(built, store=changed)

    report = _update(built)

    assert report.items_changed == 1
    texts = [row[0] for row in _rows(built, "SELECT text FROM chunks WHERE owner_id = 'k02'")]
    assert "resumen nuevo para un item sin content" in texts


def test_update_touches_only_the_changed_item(built: Path, corpus) -> None:
    """Acceptance 3: one changed item touches ONLY that item's chunks.

    The others keep their rowids, which is a stronger claim than keeping their ids: a rowid
    that moved means the row was rewritten, and with external-content FTS5 that is exactly
    where an index goes quietly wrong.
    """
    store, _vocab, _pages = corpus
    untouched_before = _rows(
        built, "SELECT chunk_id, rowid FROM chunks WHERE owner_id != 'k02' ORDER BY rowid"
    )
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "un resumen completamente distinto")
    _persist(built, store=changed)

    report = _update(built)

    assert report.items_changed == 1
    assert report.chunks_inserted > 0 and report.chunks_deleted > 0
    assert (
        _rows(built, "SELECT chunk_id, rowid FROM chunks WHERE owner_id != 'k02' ORDER BY rowid")
        == untouched_before
    )
    (summary_text,) = _rows(
        built, "SELECT text FROM chunks WHERE owner_id = 'k02' AND surface_type = 'summary'"
    )
    assert summary_text[0] == "un resumen completamente distinto"


# ---------------------------------------------------------------------------
# 7 / 7b — removal, and the FTS retraction that goes with it
# ---------------------------------------------------------------------------


def test_update_removes_every_row_of_an_item_deleted_from_the_store(built: Path, corpus) -> None:
    """Step 7: chunks, profile, surfaces, topics, kinds, failures and links — all of them.

    Enumerated per table rather than checking `chunks` alone: a leftover `items` row keeps
    answering `--author`, and a leftover `surfaces` row keeps answering `has_surfaces`, for an
    item that no longer exists.
    """
    store, _vocab, _pages = corpus
    _persist(built, store={k: v for k, v in store.items() if k != "k03"})

    report = _update(built)

    assert report.items_removed == 1 and report.chunks_deleted > 0
    for table, column in (
        ("chunks", "owner_id"),
        ("items", "item_id"),
        ("item_topics", "item_id"),
        ("item_content_kinds", "item_id"),
        ("profiles", "item_id"),
        ("source_failures", "item_id"),
        ("unfetched_links", "item_id"),
    ):
        assert _rows(built, f"SELECT COUNT(*) FROM {table} WHERE {column} = 'k03'")[0][0] == 0, (  # noqa: S608
            table
        )
    assert _rows(built, "SELECT COUNT(*) FROM surfaces WHERE owner_id = 'k03'")[0][0] == 0


def test_a_removed_items_terms_are_retracted_from_the_index(built: Path, corpus) -> None:
    """Step 7b / Plan 02 §10.7b: a word that lived only in the removed item returns ZERO rows.

    Asserted against `chunks_fts` DIRECTLY, not through the join. An orphan FTS entry is
    invisible to an inner join until its rowid is reused — and then it makes an unrelated
    chunk match a word it never held, which is the failure that actually reaches a user. The
    order `'delete'`-then-`DELETE` is what makes it hold, and it lives in `delete_chunk_rows`.
    """
    store, _vocab, _pages = corpus
    assert (
        _rows(built, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", '"Quillfeather"')[
            0
        ][0]
        > 0
    )

    _persist(built, store={k: v for k, v in store.items() if k not in {"k03", "k12"}})
    _update(built)

    remaining = _rows(
        built, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", '"Quillfeather"'
    )[0][0]
    assert remaining == 0, "the term survived in the index after its only owners were removed"


def test_a_new_item_is_added_without_rewriting_the_rest(built: Path, corpus) -> None:
    """The first row of spec §5.6's table: a new chunk is inserted, and only that."""
    store, _vocab, _pages = corpus
    before = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")
    newcomer = store["k01"].model_copy(
        update={"id": "k99", "text": "un post nuevo sobre Bramblewick"}
    )
    _persist(built, store={**store, "k99": newcomer})

    report = _update(built)

    assert report.items_added == 1 and report.items_changed == 0 and report.items_removed == 0
    after = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")
    assert after[: len(before)] == before, "existing rows were rewritten by an ADD"


def test_update_counts_how_many_items_were_added_and_removed(built: Path, corpus) -> None:
    """The half of "how many" that a boolean-shaped counter satisfies.

    The monolith's round 03 fixed the boolean-shaped counters and pinned `status` on all
    three counts and `update` on `items_changed` — but the only assertions on `update`'s
    `items_added` and `items_removed` were `== 1` and `== 0`, which `int(bool(...))`
    satisfies: the gate mutated both in `_update_report` and 274 tests stayed green. "How
    many" asserted on one is a boolean with a number's name (rule 1).

    Two added, two removed, asserted `== 2` on each, and the base is checked to hold exactly
    that population so the report cannot be right by accident.

    Seen red under `items_added=int(bool(delta.added))` / `items_removed=int(bool(...))`:
    `(1, 0, 1) == (2, 0, 2)`.
    """
    store, _vocab, _pages = corpus
    changed = {k: v for k, v in store.items() if k not in {"k01", "k04"}}
    changed["k98"] = store["k05"].model_copy(update={"id": "k98"})
    changed["k99"] = store["k06"].model_copy(update={"id": "k99"})
    _persist(built, store=changed)

    report = _update(built)

    assert (report.items_added, report.items_changed, report.items_removed) == (2, 0, 2)
    ids = {row[0] for row in _rows(built, "SELECT item_id FROM items")}
    assert {"k98", "k99"} <= ids and not {"k01", "k04"} & ids


def test_a_different_surface_version_refuses_the_update(built: Path, corpus) -> None:
    """The same reasoning one layer up: a new emitter means different surfaces, hence ids."""
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["surface_version"] = "xbrain-knowledge-surface/v2"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        _update(built)


def test_different_chunker_parameters_also_refuse_the_update(built: Path, corpus) -> None:
    """Plan 02 §7 sweeps target × overlap WITHOUT necessarily bumping `CHUNKER_VERSION`.

    A sweep that lands on new parameters and keeps the version would produce chunks cut
    differently under IDENTICAL ids — the worst case, because the id resolves and the text
    behind it is not what it was. The manifest records the parameters precisely so this can be
    caught; the plan's step 8 only names the version.
    """
    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        # A target the default is NOT: `ChunkerParams(target=800)` was this line until Plan 02
        # §7's sweep made 800 the default, at which point the test compared the default with
        # itself and could only pass by accident (rule 1).
        _update(built, options=index_build.IndexOptions(params=ChunkerParams(target=1234)))


# ---------------------------------------------------------------------------
# The vocabulary, and what it drags with it
# ---------------------------------------------------------------------------


def test_a_changed_vocabulary_rebuilds_topics_and_profiles(built: Path, corpus) -> None:
    """Plan 02 §11: *update con `vocab.yaml` cambiado ⇒ topics y perfiles se recalculan.*

    The profile composes each assigned topic's DESCRIPTION (spec §5.1.A), so a vocabulary edit
    changes text that lives inside every affected item's profile. Rebuilding topics but not
    profiles would leave the item plane quoting a description the vocabulary no longer holds.
    Seen red by rebuilding only the `topics` table.
    """
    _store, vocab, _pages = corpus
    edited = [
        t.model_copy(update={"description": "una descripcion Snapdragon completamente nueva"})
        if t.slug == vocab[0].slug
        else t
        for t in vocab
    ]
    _persist(built, vocab=edited)

    report = _update(built)

    assert report.topics_rebuilt is True
    assert report.profiles_inserted > 0, "the profiles carry the topic descriptions"
    hits = _rows(
        built, "SELECT COUNT(*) FROM profiles_fts WHERE profiles_fts MATCH ?", '"Snapdragon"'
    )
    assert hits[0][0] > 0


def test_update_dry_run_reports_the_work_and_writes_nothing(built: Path, corpus) -> None:
    """`--dry-run` on the incremental path: the plan, never the change."""
    store, _vocab, _pages = corpus
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "otro resumen")
    _persist(built, store=changed)
    before = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")

    report = _update(built, dry_run=True)

    assert report.items_changed == 1 and report.dry_run is True
    assert _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid") == before


def test_update_refreshes_the_manifest_signals(built: Path, corpus) -> None:
    """After a successful update the index is NO LONGER behind, and says so.

    An update that repaired the rows but left the old signals would keep every later query
    declaring `index_behind_store` — a warning that is always on is a warning nobody reads.
    """
    store, _vocab, _pages = corpus
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "otro resumen")
    _persist(built, store=changed)
    assert _status(built).behind is True

    _update(built)

    after = _status(built)
    assert after.behind is False and after.items_changed == 0


def test_a_dry_run_update_does_not_refresh_the_manifest(built: Path, corpus) -> None:
    """The complement, and the sharper half: a dry run that stamped the manifest would make
    the index CLAIM to be current while holding the old rows — worse than doing nothing."""
    store, _vocab, _pages = corpus
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "otro resumen")
    _persist(built, store=changed)

    _update(built, dry_run=True)

    assert _status(built).behind is True


def test_update_does_not_touch_the_three_inputs(built: Path, corpus) -> None:
    """Acceptance 13, on the incremental path and on ALL THREE inputs, not only the store.

    The plan's criterion names `items.json`, `vocab.yaml` and `topics.json` together, and the
    three are read by one loader now — so a door that wrote back to any of them would be the
    same defect wherever it landed.
    """
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in _paths(built)}

    _update(built)

    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in _paths(built)} == before


def test_an_update_that_raises_mid_way_leaves_the_index_unchanged(built: Path, corpus) -> None:
    """Plan 02 §11: *disco lleno a mitad del build ⇒ la transacción revierte.*

    Same property, exercised on `update` because that is the one that runs unattended after
    `enrich`. Seen red by committing per item: the index is then a partial application of a
    change nobody can name.
    """
    store, _vocab, _pages = corpus
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "otro resumen")
    changed["k04"] = _edit_summary(store["k04"], "y otro mas")
    _persist(built, store=changed)
    before = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")

    calls = {"n": 0}
    real = index_build.write_item

    def explode(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("disk full")
        return real(*args, **kwargs)

    index_build.write_item = explode  # type: ignore[assignment]
    try:
        with pytest.raises(OSError, match="disk full"):
            _update(built)
    finally:
        index_build.write_item = real  # type: ignore[assignment]

    assert _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid") == before


# ---------------------------------------------------------------------------
# C-3 / A-3 — the manifest describes the database, or the update refuses
# ---------------------------------------------------------------------------


def _db_counts(data: Path) -> dict[str, int]:
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        return index_build.count_rows(connection)
    finally:
        connection.close()


def _manifest(data: Path) -> dict:
    return json.loads(manifest_path(data / "index").read_text(encoding="utf-8"))


def test_update_recomputes_surfaces_and_omissions_instead_of_carrying_them(
    built: Path, corpus
) -> None:
    """A-3: `_next_manifest` updated `items`, `topics`, `chunks` and `profiles` and CARRIED
    `counts["surfaces"]`, every `skipped` counter and `failed` over from the previous
    manifest. Remove an item with two surfaces and a failed fetch and the base was right while
    the manifest kept publishing the old population — `surfaces 43` against 41 rows,
    `failed_sources 1` against 0 — and `index status --json` exposed both as they were. Spec
    §5.6 asks for counts and omissions that are VALID, not merely present (§15.2).

    Every count and every omission is now DERIVED FROM THE DATABASE, by the same function a
    fresh build uses, so the two writers cannot disagree (rule 5). Seen red before the fix:
    `43 == 41` and `1 == 0`.
    """
    store, _vocab, _pages = corpus
    victim = "k11"  # two surfaces and a failed fetch, on this fixture
    assert len(item_surfaces(store[victim])) >= 2
    assert knowledge_item(store[victim]).failed_sources
    before = _manifest(built)

    _persist(built, store={k: v for k, v in store.items() if k != victim})
    _update(built)

    after = _manifest(built)
    counts = _db_counts(built)
    connection = open_index(db_path(built / "index"), read_only=True)
    try:
        failures = connection.execute("SELECT COUNT(*) FROM source_failures").fetchone()[0]
    finally:
        connection.close()
    assert after["counts"]["surfaces"] == counts["surfaces"] < before["counts"]["surfaces"]
    assert after["skipped"]["failed_sources"] == failures < before["skipped"]["failed_sources"]
    assert after["counts"] == counts, "every plane, not only the four that used to move"


def test_the_manifest_counts_match_the_database_after_every_update(built: Path, corpus) -> None:
    """The cumulative drift: with topics rebuilt, `_clear_topics` discarded the return of
    `delete_chunk_rows`, so each update with a changed vocabulary ADDED the topic chunks to
    `counts["chunks"]` without ever subtracting them — 84 declared against 56 real after four
    updates. Deriving the counts from the database makes the invariant hold by construction;
    this test pins it across three updates that each force a topic rebuild.
    """
    _store, _vocab, pages = corpus
    for round_ in range(3):
        slug = next(iter(pages))
        pages = {
            **pages,
            slug: pages[slug].model_copy(update={"overview": f"round {round_} overview"}),
        }
        _persist(built, pages=pages)
        report = _update(built)
        assert report.topics_rebuilt is True
        assert _manifest(built)["counts"] == _db_counts(built), f"drifted on round {round_}"


def test_update_refuses_a_database_that_disagrees_with_its_manifest(built: Path, corpus) -> None:
    """C-3: `update` decided `topics_rebuilt` by comparing the vocabulary and topic-page
    fingerprints against the MANIFEST, never against what the database contains. On the real
    corpus, after an interrupted forced rebuild (C-1), the old manifest declared 45 topics over
    a base that held 0: `update` inserted every item, never rebuilt the topic plane — 45
    topics, 616 surfaces, 703 chunks gone FOR GOOD, since no later update would find a
    fingerprint to disagree with — and `status` answered `incomplete=False` with an empty
    advice.

    C-1 closes that route (no manifest survives a forced rebuild), so the state is staged
    directly: the topic plane is deleted behind the manifest's back. An incremental update
    over a base that does not contain what its manifest declares has no honest baseline, so it
    REFUSES and names the rebuild, and `status` says INCOMPLETE and names it too.

    Seen red before the fix: `update` returned an `UpdateReport` with `topics_rebuilt=False`
    and `status` reported `incomplete=False`.
    """
    connection = open_index(db_path(built / "index"))
    try:
        with connection:
            index_build._clear_topics(connection)
    finally:
        connection.close()
    assert _db_counts(built)["topics"] == 0 < _manifest(built)["counts"]["topics"]

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        _update(built)
    assert "topics" in str(caught.value), "names WHICH plane disagrees"

    report = _status(built)
    assert report.incomplete is True
    assert "xbrain index build --force" in report.advice
    assert "topics" in report.advice


def test_update_over_a_missing_database_creates_nothing_and_leaves_search_closed(
    built: Path, corpus
) -> None:
    """G-2: `index update` — even `--dry-run` — CREATED an empty database under a manifest
    that was still standing, and the query door went from failing closed to answering open.

    `update()` opened the database for writing, and `open_index` without `read_only` did
    `mkdir + connect + create_schema` whenever the file did not exist; the consistency check
    then detected `chunks 0 != …` and raised — with the file already created. The query door
    only checked `exists()` plus a compatible manifest, so the next `search` answered "no
    results" with exit 0 over a base of zero rows while `status` said `incomplete=True`: two
    instruments, opposite answers (rule 9), and the exact shape of C-1 reached by the command
    an operator runs "to see what happens". Reproduced on the real corpus through the CLI: `rm
    knowledge.db` (52 MB, a natural clean-up target) → `update --dry-run` exit 1 with the
    right message AND a 167,936-byte `knowledge.db` → `search` exit 0, «Sin resultados».

    THREE CLOSURES, and the third arrived with 02.9: `update` refuses BEFORE touching the disk
    and names `--force` (plain `build` refuses while the manifest exists — the dead end); no
    file appears on either path; and the query door keeps refusing rather than answering an
    empty corpus. The query half is what the defect was ABOUT, so it is asserted here now that
    there is a door to assert it through.

    Seen red before the fix: `db_path(...).exists()` was True after the dry run, and the query
    door returned a response over a base of zero rows.
    """
    db_path(built / "index").unlink()
    assert manifest_path(built / "index").exists(), "the manifest is what makes this state"

    with pytest.raises(IndexMissingError, match="xbrain index build --force"):
        _update(built, dry_run=True)
    assert not db_path(built / "index").exists(), "update must not create the base"

    with pytest.raises(IndexMissingError, match="xbrain index build --force"):
        _update(built)
    assert not db_path(built / "index").exists()

    with pytest.raises(IndexMissingError, match="xbrain index build --force"):
        index_store.open_for_query(built / "index", *_paths(built))
    assert not db_path(built / "index").exists(), "the query door may not create it either"


def test_status_declares_a_standing_manifest_over_a_missing_database_as_every_door_does(
    built: Path, corpus
) -> None:
    """U-2: the ONE state the seam's docstring lists first — a manifest standing over a base
    that is not there (C-1's interruption, G-2's clean-up) — and the door that did not ask the
    question. `update` asks `require_database`, whose sentence names `build --force` because
    plain `build` refuses while a manifest exists; `status` checked `exists()` by itself, read
    "no base" as "never built", and answered `incomplete: False`, `+2404 nuevos`, «actualiza
    con `xbrain index update`» — the advice `update` then refused. Two instruments, one state,
    opposite answers (rule 9), on the diagnostic instrument.

    Asserted BY VALUE across all three doors: `status` reports incomplete and publishes,
    VERBATIM, the sentence `update` and the query door raise. `search` could not be the door
    that introduced the property — the defect was in `status` — but it is the door whose
    silence the defect was measured against, so it belongs in the comparison.

    `get` reads the live store and keeps answering with no index at all (spec §3.7 invariant
    7); it is `get_service`'s, and child 02.10 adds it to this list — the FOURTH door, and
    the only one of the four whose correct answer is a bundle. It is asserted in the same
    breath as the other three because the interesting property is the CONTRAST: in one state
    of the index, three doors refuse with one sentence and the fourth is untouched by it. A
    `get` that reached for the base — for a title, a topic, anything — would raise the same
    `IndexMissingError` here and the contrast would collapse into a fourth refusal.

    Seen red before the fix: `incomplete is False` and the advice named `index update`.
    """
    db_path(built / "index").unlink()
    assert manifest_path(built / "index").exists(), "the manifest is what makes this state"

    with pytest.raises(IndexMissingError, match="xbrain index build --force") as refused:
        _update(built, dry_run=True)
    sentence = str(refused.value)

    with pytest.raises(IndexMissingError) as refused_query:
        index_store.open_for_query(built / "index", *_paths(built))
    assert str(refused_query.value) == sentence, "one sentence, every door"

    report = _status(built)
    assert report.incomplete is True
    assert report.advice == sentence
    assert "xbrain index build --force" in report.advice

    store, vocab, pages = corpus
    bundle = get(
        "k03",
        QueryContext(
            store=store,
            vocab=vocab,
            topic_pages=pages,
            index_dir=built / "index",
            items_path=built / "items.json",
            vocab_path=built / "vocab.yaml",
            topics_path=built / "topics.json",
        ),
        surfaces=("external_article",),
    )
    assert bundle.surfaces and bundle.surfaces[0].surface_type == "external_article"

    assert not db_path(built / "index").exists(), "no door may create the base but `build`"


def test_update_sees_a_quoted_author_repaired_without_touching_the_body(
    built: Path, corpus
) -> None:
    """G-5: rule 6 on the attribution rule this repo says it paid for in blood.

    `item_fingerprint` hashed the item's metadata plus each surface's `surface_fingerprint` —
    `(version, type, origin, TEXT)` — and nothing else about the surface. So a repair that
    fills in the AUTHOR of a quoted post without touching its body (`refresh-quoted` on a
    quoted fetch that came back authorless; 826 `quoted_tweet` sources on the real store, 29 of
    them failures with no author yet) left `update` reporting `0 cambiados` and the index
    serving the old attribution from `surfaces.attribution_handle` — the repaired evidence with
    the derivative standing. Title, language and locator had the same blind spot, and the index
    stores and serves all four (A-1).

    Seen red before the fix: `items_changed == 0` and the stored `attribution_handle` was still
    `othervoice`.
    """
    store, _vocab, _pages = corpus
    item = store["k07"]
    assert item.content is not None
    position = next(i for i, s in enumerate(item.content.sources) if s.kind == "quoted_tweet")
    sources = list(item.content.sources)
    sources[position] = sources[position].model_copy(
        update={"author": Author(handle="repairedvoice", name="Repaired Voice")}
    )
    repaired = dict(store)
    repaired["k07"] = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )
    assert repaired["k07"].content.sources[position].text == item.content.sources[position].text
    _persist(built, store=repaired)

    report = _update(built)

    assert report.items_changed == 1
    (row,) = _rows(
        built,
        "SELECT attribution_handle FROM surfaces WHERE owner_id = 'k07' "
        "AND surface_type = 'quoted_post'",
    )
    assert row[0] == "repairedvoice"


@pytest.mark.parametrize("table", sorted(TABLES | FTS_TABLES))
def test_update_refuses_an_index_missing_a_table_instead_of_recreating_it_empty(
    built: Path, corpus, table: str
) -> None:
    """G-8: the WRITE door of the C-2 schema guard had no test.

    `open_index` in write mode verifies the schema of an EXISTING database so `update` does not
    carry on over a dropped table — `create_schema` is idempotent and would re-create it EMPTY.
    The read door had its test; this one did not, and the monolith's gate measured what
    removing the call costs: 119 tests still green, and then `DROP TABLE chunks_fts` → `update`
    returns normally (an empty `chunks_fts` under 56 full `chunks` rows) and
    `status.incomplete=False` — the silent mode of C-2, re-entered through `update`.

    Parametrised over the DECLARED set (`TABLES | FTS_TABLES`), like the DDL test, so a table
    added to the schema is covered without anyone remembering. `update` and `status` are both
    asserted, and on the same two facts: the error names the TABLE and the rebuild command.

    Seen red by removing `_verify_schema` from the write door of `open_index`: ALL
    parametrisations fail, through two different doors. For the tables no `COUNT(*)` watches
    (`chunks_fts`, `profiles_fts`, `item_topics`, `item_content_kinds`, `source_failures`,
    `unfetched_links`) `update` simply returns an `UpdateReport`. For the five counted planes
    `update` still raises — the consistency check sees `items 0 != 12` — but `create_schema`
    runs through `executescript`, which COMMITS, so the refused update has already re-created
    the table EMPTY on disk, and the `status` that follows finds a complete schema and merely
    declares a count mismatch instead of raising. A guard that repairs the schema on its way to
    refusing is the silent mode with one extra step, which is why both instruments are here.
    """
    connection = sqlite3.connect(db_path(built / "index"))
    connection.execute(f"DROP TABLE {table}")  # nosec B608 — a test fixture, closed set
    connection.commit()
    connection.close()

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        _update(built)
    assert table in str(caught.value), "names WHAT is missing, not only that something is"
    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        _status(built)
    assert table in str(caught.value)


def test_status_declares_a_manifest_the_code_cannot_use(built: Path, corpus) -> None:
    """The neighbour of C-3's declaration: a manifest from another chunker version.

    `update` refuses it with the rebuild advice; `status` read it without checking the versions
    and answered `incomplete=False`, `advice=''` — two instruments, opposite answers on the
    same state (rule 9). And its "incomplete" advice named plain `xbrain index build`, which
    REFUSES while a manifest exists: a dead end in two hops. `status` now applies the same
    compatibility check and names `--force`.

    Seen red before the fix: `incomplete is False`.
    """
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chunker_version"] = "xbrain-knowledge-chunker/v0"
    path.write_text(json.dumps(raw), encoding="utf-8")

    report = _status(built)
    assert report.incomplete is True
    assert "xbrain index build --force" in report.advice


# ---------------------------------------------------------------------------
# H1 — a changed topic ASSIGNMENT reaches the topic plane, not only `item_topics`
# ---------------------------------------------------------------------------


def test_update_refreshes_the_topic_rows_when_an_items_topics_move(built: Path, corpus) -> None:
    """H1 — CLAUDE.md rule 6 on the topic plane.

    `topics` stores `primary_item_ids_json`, `secondary_item_ids_json` and `stale`, but
    `topics_rebuilt` looked ONLY at the vocabulary and topic-page fingerprints, so a change of
    `Item.enriched.primary_topic` / `topics` — exactly what `enrich` writes — rewrote the item
    and `item_topics` and left the topic rows holding the OLD members and the OLD `stale` bit,
    while the manifest and `status` declared the index healthy. Reproduced on the fixture: k02
    moved from `agent-evaluation` to `ai-policy`, `item_topics` said `ai-policy`, `topics`
    still listed k02 under `agent-evaluation`, `stale=0` on both.

    The fix is NARROW on purpose: the vocabulary and the pages did not move, so the topic
    SURFACES and CHUNKS are untouched (their rowids prove it) and only the rows whose
    membership or staleness differs are rewritten — through the same row projection the full
    writer uses. `stale` flips on both topics because the live primary count no longer equals
    `post_count_at_synth`.

    Seen red before the fix: the `topics` rows after the update were identical to before.
    """
    store, _vocab, _pages = corpus
    before = _topic_rows(built)
    assert before["agent-evaluation"] == (["k02", "k08"], [], 0)
    assert before["ai-policy"] == (["k03", "k04", "k07", "k11"], [], 0)
    topic_chunks_before = _rows(
        built, "SELECT chunk_id, rowid FROM chunks WHERE owner_type = 'topic' ORDER BY rowid"
    )

    changed = dict(store)
    changed["k02"] = _reassign(store["k02"], "ai-policy")
    _persist(built, store=changed)
    report = _update(built)

    after = _topic_rows(built)
    assert after["ai-policy"] == (["k02", "k03", "k04", "k07", "k11"], [], 1)
    assert after["agent-evaluation"] == (["k08"], [], 1)
    assert (
        _rows(built, "SELECT chunk_id, rowid FROM chunks WHERE owner_type = 'topic' ORDER BY rowid")
        == topic_chunks_before
    ), "the topic surfaces and chunks did not change, so they must not be rewritten"
    assert report.topics_rebuilt is False, "vocabulary and pages did not move"
    assert report.topics_refreshed == 2

    status = _status(built)
    assert status.items_changed == 0 and status.topics_changed == 0
    assert status.advice == ""


# ---------------------------------------------------------------------------
# ONE answer to «does this manifest describe this base?» (B1, D-1, seam a)
# ---------------------------------------------------------------------------


def test_a_manifest_with_empty_counts_is_refused_by_every_door_not_sealed_as_healthy(
    built: Path, corpus
) -> None:
    """B1, the monolith gate's reproduction verbatim: `counts: {}` in the manifest and the rows
    of `chunks` and `profiles` deleted, tables kept, file readable. Before: `status`
    `incomplete=False`, `advice=''` publishing `chunks=0`; `search` zero results with
    `degraded: ["no_embeddings"]`, indistinguishable from a corpus with no matches; and
    `update` — zero changes, zero writes — wrote a manifest declaring `chunks=0`,
    `profiles=0`, sealing the amputation as sound. Three instruments converging on an
    incomplete index presented as healthy: the FAIL-OPEN family, sixth route.

    ALL THREE DOORS NOW, and the last line still matters most: nothing RE-SEALS the amputated
    base. A door that refuses but re-seals on the way out has only moved the failure one run
    later. The query door was the instrument that made this dangerous — zero results with
    `degraded: ["no_embeddings"]` is indistinguishable from a corpus with no matches — so its
    refusal is the one the criterion is really about.
    """
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["counts"] = {}
    path.write_text(json.dumps(raw), encoding="utf-8")
    connection = open_index(db_path(built / "index"))
    try:
        with connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM profiles")
    finally:
        connection.close()

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force"):
        _update(built)
    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force"):
        index_store.open_for_query(built / "index", *_paths(built))
    report = _status(built)
    assert report.incomplete is True
    assert "xbrain index build --force" in report.advice, report.advice
    assert json.loads(path.read_text(encoding="utf-8"))["counts"] == {}, (
        "a door re-sealed the amputated base as healthy"
    )


def test_status_search_and_update_ask_one_function_whether_the_manifest_describes_the_base(
    built: Path, corpus, monkeypatch
) -> None:
    """The seam, asserted by IDENTITY across ALL THREE of its consumers (CLAUDE.md rule 5).

    Six rounds of the monolith closed the fail-open family one route at a time — an interrupted
    forced rebuild, a missing table, a dry run creating an empty base, a signal covering one
    input of three, a manifest with empty counts — because each door carried its own reading of
    "the base is what the manifest says". There is now ONE function (`describe_base`), and this
    test replaces its answer with a sentinel: every door must repeat the sentinel VERBATIM. A
    door that re-derives the question stays silent under the sentinel and goes red here; a door
    that stops asking goes red here.

    The query door is the third consumer and it runs the same function with `whole_file=False`
    — a different ARGUMENT, the same question — which is precisely what a per-door
    re-derivation would have looked like from the outside until the sentinel was applied.

    Seen red by having `update` compare `manifest.counts` against `count_rows` itself instead of
    asking: the sentinel never appears and `update` returns a normal report.
    """
    sentinel = (
        "SENTINEL: esta base no es la que el manifest describe. "
        "Reconstruye el índice con `xbrain index build --force`."
    )
    monkeypatch.setattr(
        index_build,
        "describe_base",
        lambda *args, **kwargs: index_build.BaseVerdict(counts={}, sentence=sentinel),
    )

    with pytest.raises(IndexIncompatibleError) as caught:
        _update(built)
    assert str(caught.value) == sentinel
    with pytest.raises(IndexIncompatibleError) as caught:
        index_store.open_for_query(built / "index", *_paths(built))
    assert str(caught.value) == sentinel
    assert _status(built).advice == sentinel


def test_status_search_and_update_ask_one_function_whether_the_base_exists(
    built: Path, corpus, monkeypatch
) -> None:
    """The OTHER half of the seam's question — «is there a base at all?» — by identity (U-2).

    `describe_base` answers «does this manifest describe this base?» and presupposes a base;
    `require_database` answers whether there is one, and its docstring promised «one function,
    called by every door» while `status` was not among them, which is exactly how F7-1/F2 stayed
    open with the seam in place. The sentinel is raised from `require_database` in EVERY MODULE
    THAT BINDS THE NAME — `index_schema`, `index_build` and now `index_store` — because each
    imported it into its own namespace, so patching one module would leave the others calling
    the real function and the test would pass while proving nothing about them. `update` and the
    query door must raise it verbatim and `status` must publish it as its advice. A door that
    tests `exists()` by itself stays silent under the sentinel and goes red here.

    Seen red before the fix: `status` answered `advice == ''` under the sentinel, because it
    never asked.
    """
    sentinel = "SENTINEL: no hay base. Reconstruye el índice con `xbrain index build --force`."

    def refuse(index_dir):
        raise IndexMissingError(sentinel)

    for module in (index_schema, index_build, index_store):
        monkeypatch.setattr(module, "require_database", refuse)

    with pytest.raises(IndexMissingError) as caught:
        _update(built)
    assert str(caught.value) == sentinel
    with pytest.raises(IndexMissingError) as caught:
        index_store.open_for_query(built / "index", *_paths(built))
    assert str(caught.value) == sentinel
    report = _status(built)
    assert report.incomplete is True and report.advice == sentinel


@pytest.mark.parametrize("read", ["_stored_fingerprints", "stored_topic_rows"])
def test_a_database_error_past_quick_check_is_the_rebuild_advice_on_status_and_update(
    built: Path, corpus, monkeypatch, read: str
) -> None:
    """F7-5: the conversion `reading_base` performs — a `DatabaseError` raised by a MAINTENANCE
    read after a clean `quick_check` becomes the rebuild sentence — had no test of its own: with
    its body reduced to a bare `yield`, 495 tests stayed green, because what was pinned was the
    conversion INSIDE `describe_base`, never the one around `_stored_fingerprints`,
    `stored_topic_rows` and the update transaction. A guard that can be hollowed out with
    nothing red is the fail-open cell of rule 11.

    Staged as the read itself raising — the page it touches is not one `quick_check` or the five
    counts read, which is exactly D-1's case — and asserted on both doors that make the read.
    Both RAISE the converted sentence: `status` names the rebuild as an error rather than as a
    report, so the CLI prints it with exit 1.

    Seen red under the bare-`yield` mutation: a raw `sqlite3.DatabaseError` out of both doors.
    """

    def torn(connection):
        raise sqlite3.DatabaseError("database disk image is malformed (staged past page 1)")

    monkeypatch.setattr(index_build, read, torn)

    with pytest.raises(IndexIncompatibleError, match="staged past page 1") as refused:
        _status(built)
    assert REBUILD_ADVICE in str(refused.value)

    if read == "_stored_fingerprints":
        with pytest.raises(IndexIncompatibleError, match="staged past page 1") as refused:
            _update(built, dry_run=True)
        assert REBUILD_ADVICE in str(refused.value)


def test_a_database_error_inside_the_update_transaction_is_the_rebuild_advice(
    built: Path, corpus, monkeypatch
) -> None:
    """F7-5, the third read `reading_base` wraps: the transaction itself. A `DatabaseError`
    raised while rewriting rows — a torn page under a table the deltas touch — is the rebuild
    sentence, and the base is left as it was (the transaction rolls back).
    Seen red under the bare-`yield` mutation: a raw `sqlite3.DatabaseError`.
    """
    before = _db_counts(built)

    def torn(*args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed (staged in the transaction)")

    monkeypatch.setattr(index_build, "_apply_update", torn)

    with pytest.raises(IndexIncompatibleError, match="staged in the transaction") as refused:
        _update(built)
    assert REBUILD_ADVICE in str(refused.value)
    assert _db_counts(built) == before


def test_the_update_report_counts_the_topic_chunks_it_deletes(built: Path, corpus) -> None:
    """N-1: `chunks_deleted` omitted what `_clear_topics` removed, so after a
    `topics.json`-only update the report read `+22,287 / -21,583` (net +704) while the base
    moved 22,286 -> 22,287 (net +1). A counter that does not count what its name says (rule 2).
    Asserted against `COUNT(*)` before and after, so the net is the base's.

    Seen red before the fix: the net disagreed by the number of topic chunks.
    """
    _store, _vocab, pages = corpus
    before = _db_counts(built)["chunks"]
    slug = next(iter(pages))
    _persist(
        built,
        pages={
            **pages,
            slug: pages[slug].model_copy(
                update={"overview": pages[slug].overview + " nuevaobservacion"}
            ),
        },
    )

    report = _update(built)

    assert report.topics_rebuilt is True
    assert report.chunks_inserted - report.chunks_deleted == _db_counts(built)["chunks"] - before


def test_a_database_error_from_the_counts_themselves_is_the_rebuild_advice_not_a_traceback(
    built: Path, corpus, monkeypatch
) -> None:
    """D-1's THIRD arm, the one inside `describe_base`, and the ASYMMETRY between the doors.

    `describe_base` promises that any `DatabaseError` raised by `quick_check` or by the five
    `COUNT(*)` IS the answer. The two arms above it are covered by staged damage; this one is
    not reachable that way in THIS tree, and the reason is worth writing down rather than
    working around: `quick_check` swallows its own `DatabaseError` and returns it as a
    sentence, so with `whole_file=True` it always answers first. The arm belongs to the door
    that passes `whole_file=False` — 02.9's query door — and the honest staging until then is
    to make the count itself raise. Left untested it would be a guard covered only by the
    guard in front of it (rule 11's fail-open cell), which is how `count_rows` escaped as a
    61-line traceback naming no command while CLAUDE.md declared G-4 closed on all three.

    And the two doors must NOT answer alike: `status` is a report and publishes the sentence
    as its advice, `update` re-seals the manifest and must refuse. That asymmetry is stated in
    `describe_base`'s docstring and is asserted here, on the same sentence, so a door that
    quietly adopted the other's behaviour goes red.

    Seen red under `except sqlite3.DatabaseError` removed from `describe_base`: a raw
    `sqlite3.DatabaseError` out of `update` and out of `status`.
    """

    def torn(connection):
        raise sqlite3.DatabaseError("database disk image is malformed (staged in count_rows)")

    monkeypatch.setattr(index_build, "count_rows", torn)

    report = _status(built)
    assert report.incomplete is True
    assert "staged in count_rows" in report.advice and REBUILD_ADVICE in report.advice

    with pytest.raises(IndexIncompatibleError, match="staged in count_rows") as refused:
        _update(built, dry_run=True)
    assert str(refused.value) == report.advice, "one sentence, two behaviours"


def test_the_narrow_topic_refresh_refuses_a_row_it_cannot_rewrite_instead_of_counting_it(
    built: Path, corpus
) -> None:
    """The guard the comparator's new direction made necessary, asked DIRECTLY.

    `_topics_behind` now also names a row the base holds for a topic the vocabulary no longer
    declares. The narrow refresh has no record to rewrite that from, so it has exactly two
    honest options and took neither by default: writing the rest and returning `len(behind)`
    reports a row as refreshed that nothing touched (rule 2), and skipping it silently is the
    fail-open one layer down from the one just closed.

    IT IS ASKED DIRECTLY BECAUSE THERE IS NO HONEST WAY TO STAGE IT THROUGH `update`, and that
    is a property worth pinning rather than a limitation to apologise for: `update` reaches
    this function only when `topics_rebuilt` is False, i.e. when the current vocabulary's
    fingerprint equals the manifest's — and a base written under that same vocabulary has no
    orphan. Dropping a topic from `vocab.yaml` moves the fingerprint and takes the rebuild
    path, where `_clear_topics` removes the row. Staging it through `update` would mean
    forging a manifest, which tests a state the writer cannot produce. So the guard is a
    backstop for a base edited behind the manifest's back, and it is exercised where it can be
    exercised truthfully.

    The vocabulary is emptied to make every stored row an orphan, so the refusal cannot come
    from a row that merely differs.

    Seen red before the guard: `KeyError: 'agent-evaluation'` out of `_write_topic_row`.
    """
    _store, vocab, _pages = corpus
    inputs = _inputs(built)
    emptied = index_build.IndexInputs(
        store=inputs.store, vocab=[], topic_pages=inputs.topic_pages, signal=inputs.signal
    )
    connection = open_index(db_path(built / "index"))
    try:
        stored = index_build.stored_topic_rows(connection)
        assert stored, "the fixture base must hold topic rows, or nothing is tested"

        with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as refused:
            index_build._refresh_topic_rows(LexicalIndex(connection), emptied)
    finally:
        connection.close()

    for slug in stored:
        assert slug in str(refused.value), "the refusal names WHICH rows, not merely that some"
    assert _rows(built, "SELECT COUNT(*) FROM topics")[0][0] == len(stored), (
        "a refusal must not have rewritten anything on its way out"
    )


# A write against the topic plane, matched by the statement's LEADING VERB rather than by its
# effect. Anchoring on the verb is what separates a write from a read: the first version keyed
# on `FROM topics` and flagged `SELECT COUNT(*) FROM topics`, which `manifest_tallies` runs on
# every update. `item_topics` cannot match either — the pattern needs whitespace immediately
# before `topics`, and there an underscore sits in its place. Both halves are asserted below,
# because a detector nobody tested is the assertion passing for the wrong reason one level up.
_TOPIC_WRITE = re.compile(
    r"^\s*(?:INSERT|UPDATE|DELETE|REPLACE)\b.*\s+topics\b", re.IGNORECASE | re.DOTALL
)


def test_a_no_op_update_issues_no_write_against_the_topic_plane(
    built: Path, corpus, monkeypatch
) -> None:
    """Zero writes asserted as ZERO STATEMENTS, because every assertion on the EFFECT is blind.

    `_write_topic_row` issues `INSERT OR REPLACE`. Rewriting a row with the values it already
    holds changes nothing a query can see — same columns, same count, same content — and
    `topics_refreshed` counts `len(behind)`, so it stays 0 however many rows were written.
    Measured: mutating `for slug in behind` to `for slug in records`, which writes the WHOLE
    plane on every run including a no-op, left the entire suite green.

    That is rule 1's shape — an assertion satisfied for the wrong reason — and the escape is to
    stop asserting on the effect. `sqlite3.Connection.set_trace_callback` reports every
    statement the connection executes, so the claim becomes "no write was ISSUED", which no
    idempotent statement can satisfy quietly. The callback is attached by wrapping the door
    `update` opens its connection through, so what is traced is the real run and not a
    connection this test made.

    The rowid comparison in `test_update_with_no_changes_writes_nothing` is the cheap witness
    for the same fact and is kept: it needs no monkeypatch and catches the same mutation
    through a completely different mechanism, so neither is the only thing standing between
    this plane and a silent rewrite.

    Seen red under that mutation: 2 `INSERT OR REPLACE INTO topics` statements on a run whose
    report says nothing changed.
    """
    statements: list[str] = []
    real_open = index_build.open_index

    def tracing(*args, **kwargs):
        connection = real_open(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(index_build, "open_index", tracing)

    report = _update(built)

    assert statements, "nothing was traced: the wrapper never ran, so this proves nothing"
    assert (report.items_changed, report.topics_refreshed) == (0, 0)
    assert [sql for sql in statements if _TOPIC_WRITE.search(sql)] == []
    # An empty result means nothing unless the detector can SEE the statements it forbids and
    # ignore the ones it must not flag. Both directions, on the real SQL of this module.
    for writes in (
        "INSERT OR REPLACE INTO topics (slug, description) VALUES (?,?)",
        "DELETE FROM topics",
        "UPDATE topics SET stale = 1 WHERE slug = ?",
    ):
        assert _TOPIC_WRITE.search(writes), writes
    for reads in (
        "SELECT COUNT(*) FROM topics",
        "SELECT slug, description FROM topics",
        "INSERT OR REPLACE INTO item_topics (item_id, slug, is_primary) VALUES (?,?,?)",
    ):
        assert not _TOPIC_WRITE.search(reads), reads
    # And the trace really did carry topic reads, so the empty write list is a live result and
    # not a connection that executed nothing against this plane.
    assert [sql for sql in statements if "topics" in sql.lower()], "no topic statement traced"


def _titled_blank_source(item: Item, title: str) -> Item:
    """Add a source that carries a TITLE and an EMPTY body — the shape the two sides split on.

    `profile.py:_titles` gates on `if source.title`, so this reaches `profiles_fts`.
    `item_surfaces` drops a source whose body is blank, so it emits nothing. Everything in
    this test hangs on those two gates disagreeing, and both are asserted before the act.
    """
    blank = ContentSourceSuccess(
        kind="external_article", url="https://example.test/sin-cuerpo", title=title, text=""
    )
    content = (
        Content(sources=[blank], fetched_at=item.captured_at)
        if item.content is None
        else item.content.model_copy(update={"sources": [*item.content.sources, blank]})
    )
    return item.model_copy(update={"content": content})


def test_a_title_only_change_on_a_blank_bodied_source_invalidates_the_stale_profile(
    built: Path, corpus
) -> None:
    """The profile plane, through the PUBLIC path, on the one shape no surface can carry.

    THE TWO SIDES DISAGREE ABOUT WHAT IS EMPTY, and that is the whole defect. `_titles` gates
    on `if source.title` while `item_surfaces` drops a source whose BODY is blank, so a source
    with a title and no body reaches `profiles`/`profiles_fts` and emits no surface row —
    leaving it outside every atom `item_fingerprint` hashed. `index_build` had this filed as a
    known debt against 02.7, which shipped the writer and did not discharge it; 02.8 is where
    it becomes observable, because 02.8 is what reports the index current.

    Measured on this exact shape before the fix: `item_fingerprint` did not move, `update`
    reported `items_changed=0` and `profiles_inserted=0`, `status` answered `advice=''`, and
    `profiles_fts` went on matching the OLD title (1 row) while never matching the new one
    (0 rows). A searchable string nobody can reach any more, under an index declaring itself
    current — rule 6, failing open.

    The stale term is asserted GONE and the fresh one PRESENT, both against `profiles_fts`
    directly rather than through a count: a profile rewritten to the same bytes would satisfy
    `profiles_inserted > 0` while the old term still matched.
    """
    store, vocab, _pages = corpus
    victim = "k02"
    old_title, new_title = "El titulo Quillfeatherbis ORIGINAL", "Un titulo Snapdragonbis nuevo"

    _persist(built, store={**store, victim: _titled_blank_source(store[victim], old_title)})
    _update(built)

    seeded = _inputs(built).store[victim]
    assert [s for s in item_surfaces(seeded) if (s.title or "") == old_title] == [], (
        "the blank-bodied source must emit NO surface, or the defect is not staged"
    )
    assert old_title in profile_text(seeded, vocab), "but its title must reach the profile"
    assert _rows(built, _FTS, '"Quillfeatherbis"')[0][0] == 1

    _persist(built, store={**store, victim: _titled_blank_source(store[victim], new_title)})
    report = _update(built)

    assert report.items_changed == 1, "the item moved: only its title did, and that is enough"
    assert report.profiles_inserted > 0
    assert _rows(built, _FTS, '"Quillfeatherbis"')[0][0] == 0, "the stale term is still searchable"
    assert _rows(built, _FTS, '"Snapdragonbis"')[0][0] == 1, "the new title never became findable"
    assert _status(built).items_changed == 0, "and the index is current afterwards"


def test_a_whitespace_only_summary_is_the_same_split_and_is_hashed_too(built: Path, corpus) -> None:
    """The second of the three shapes the one atom closes, so the fix is not title-shaped.

    `profile_text` appends the summary on truthiness (`if item.enriched.summary`), while the
    emitter drops it through `_blank()`, which STRIPS. A summary of `"   "` is therefore
    profile-bearing and surface-less, exactly as a titled blank-bodied source is. Asked of the
    fingerprint directly, because the point is that the ATOM covers the family rather than the
    one member a regression test happened to stage.

    Seen red before the atom: the two fingerprints were equal.
    """
    store, _vocab, _pages = corpus
    base = store["k02"]
    assert base.enriched is not None

    def with_summary(text: str) -> Item:
        return base.model_copy(
            update={"enriched": base.enriched.model_copy(update={"summary": text})}
        )

    one, two = with_summary("   "), with_summary(" \t ")
    assert [s.text for s in item_surfaces(one)] == [s.text for s in item_surfaces(two)], (
        "neither whitespace summary may reach a surface, or this stages nothing"
    )
    assert profile_text(one, []) != profile_text(two, []), "but both reach the profile"
    assert index_build.item_fingerprint(one) != index_build.item_fingerprint(two)
