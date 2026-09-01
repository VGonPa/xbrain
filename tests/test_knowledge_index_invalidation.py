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
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.chunking import ChunkerParams
from xbrain.knowledge.index_schema import IndexIncompatibleError, db_path, manifest_path, open_index
from xbrain.models import Item, Topic, TopicPage

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
    """A data/ with the fixture store on disk and a freshly built index over it."""
    store, vocab, pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    _write_store(data / "items.json", store)
    index_build.build(data / "index", store, vocab, pages, data / "items.json")
    return data


def _write_store(path: Path, store: dict[str, Item]) -> None:
    path.write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )


def _update(data: Path, store, corpus, **kwargs) -> index_build.UpdateReport:
    _vocab_store, vocab, pages = corpus
    return index_build.update(data / "index", store, vocab, pages, data / "items.json", **kwargs)


def _edit_summary(item: Item, text: str) -> Item:
    """The change `enrich` actually makes: a new summary and a new `enriched_at`."""
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


def _rows(data: Path, sql: str, *params) -> list:
    connection = open_index(db_path(data / "index"), read_only=True)
    return connection.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# 5 — no changes, no writes
# ---------------------------------------------------------------------------


def test_update_with_no_changes_writes_nothing(built: Path, corpus) -> None:
    """Step 5 / acceptance 3: zero inserts, zero deletes.

    Asserted on the REPORT and on the row identity: a rebuild that happened to produce the
    same ids would satisfy a count-only assertion while having rewritten everything. Seen red
    by dropping the fingerprint comparison — every item then re-indexes on every run.
    """
    store, _vocab, _pages = corpus
    before = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")
    report = _update(built, store, corpus)
    assert (report.items_added, report.items_changed, report.items_removed) == (0, 0, 0)
    assert (report.chunks_inserted, report.chunks_deleted) == (0, 0)
    assert _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid") == before


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
    _write_store(built / "items.json", changed)

    report = _update(built, changed, corpus)

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
# 6 — the item with no `content` (rule 6)
# ---------------------------------------------------------------------------


def test_update_detects_a_summary_change_on_an_item_with_no_content(built: Path, corpus) -> None:
    """Step 6: 961 of 2,404 real items have NO `content`, so `fetched_at` reaches none of them.

    CLAUDE.md rule 6 in its exact form: *check the invalidation signal actually reaches the
    population being repaired*. Seen red by fingerprinting `content.fetched_at` alone — this
    item then never changes, whatever is done to its summary.
    """
    store, _vocab, _pages = corpus
    assert store["k02"].content is None
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "resumen nuevo para un item sin content")
    _write_store(built / "items.json", changed)

    report = _update(built, changed, corpus)

    assert report.items_changed == 1
    texts = [row[0] for row in _rows(built, "SELECT text FROM chunks WHERE owner_id = 'k02'")]
    assert "resumen nuevo para un item sin content" in texts


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
    changed = {k: v for k, v in store.items() if k != "k03"}
    _write_store(built / "items.json", changed)

    report = _update(built, changed, corpus)

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
        assert _rows(built, f"SELECT COUNT(*) FROM {table} WHERE {column} = 'k03'")[0][0] == 0, (
            table
        )  # noqa: S608
    assert _rows(built, "SELECT COUNT(*) FROM surfaces WHERE owner_id = 'k03'")[0][0] == 0


def test_a_removed_items_terms_are_retracted_from_the_index(built: Path, corpus) -> None:
    """Step 7b: a word that lived only in the removed item returns ZERO rows afterwards.

    Asserted against `chunks_fts` DIRECTLY, not through the join. An orphan FTS entry is
    invisible to an inner join until its rowid is reused — and then it makes an unrelated
    chunk match a word it never held, which is the failure that actually reaches a user.
    """
    store, _vocab, _pages = corpus
    assert _rows(
        built, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", '"Quillfeather"'
    )

    changed = {k: v for k, v in store.items() if k not in {"k03", "k12"}}
    _write_store(built / "items.json", changed)
    _update(built, changed, corpus)

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
    changed = {**store, "k99": newcomer}
    _write_store(built / "items.json", changed)

    report = _update(built, changed, corpus)

    assert report.items_added == 1 and report.items_changed == 0 and report.items_removed == 0
    after = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")
    assert after[: len(before)] == before, "existing rows were rewritten by an ADD"


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
    store, _vocab, _pages = corpus
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chunker_version"] = "xbrain-knowledge-chunker/v2"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        _update(built, store, corpus)


def test_a_different_surface_version_refuses_the_update(built: Path, corpus) -> None:
    """The same reasoning one layer up: a new emitter means different surfaces, hence ids."""
    store, _vocab, _pages = corpus
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["surface_version"] = "xbrain-knowledge-surface/v2"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        _update(built, store, corpus)


def test_different_chunker_parameters_also_refuse_the_update(built: Path, corpus) -> None:
    """Plan 02 §7 sweeps target × overlap WITHOUT necessarily bumping `CHUNKER_VERSION`.

    A sweep that lands on new parameters and keeps the version would produce chunks cut
    differently under IDENTICAL ids — the worst case, because the id resolves and the text
    behind it is not what it was. The manifest records the parameters precisely so this can
    be caught; the plan's step 8 only names the version.
    """
    store, vocab, pages = corpus
    with pytest.raises(IndexIncompatibleError, match="index build --force"):
        index_build.update(
            built / "index",
            store,
            vocab,
            pages,
            built / "items.json",
            options=index_build.IndexOptions(params=ChunkerParams(target=800)),
        )


# ---------------------------------------------------------------------------
# The vocabulary, and what it drags with it
# ---------------------------------------------------------------------------


def test_a_changed_vocabulary_rebuilds_topics_and_profiles(built: Path, corpus) -> None:
    """Plan 02 §11: *update con `vocab.yaml` cambiado ⇒ topics y perfiles se recalculan.*

    The profile composes each assigned topic's DESCRIPTION (spec §5.1.A), so a vocabulary
    edit changes text that lives inside every affected item's profile. Rebuilding topics but
    not profiles would leave the item plane quoting a description the vocabulary no longer
    holds. Seen red by rebuilding only the `topics` table.
    """
    store, vocab, pages = corpus
    edited = [
        t.model_copy(update={"description": "una descripcion Snapdragon completamente nueva"})
        if t.slug == vocab[0].slug
        else t
        for t in vocab
    ]

    report = index_build.update(built / "index", store, edited, pages, built / "items.json")

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
    _write_store(built / "items.json", changed)
    before = _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid")

    report = _update(built, changed, corpus, dry_run=True)

    assert report.items_changed == 1
    assert _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid") == before


def test_update_refreshes_the_manifest_signals(built: Path, corpus) -> None:
    """After a successful update the index is NO LONGER behind, and says so.

    An update that repaired the rows but left the old signals would keep every later query
    declaring `index_behind_store` — a warning that is always on is a warning nobody reads.
    """
    store, _vocab, _pages = corpus
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "otro resumen")
    _write_store(built / "items.json", changed)
    assert index_build.status(built / "index", changed, built / "items.json").behind is True

    _update(built, changed, corpus)

    after = index_build.status(built / "index", changed, built / "items.json")
    assert after.behind is False and after.items_changed == 0


def test_a_dry_run_update_does_not_refresh_the_manifest(built: Path, corpus) -> None:
    """The complement, and the sharper half: a dry run that stamped the manifest would make
    the index CLAIM to be current while holding the old rows — worse than doing nothing."""
    store, _vocab, _pages = corpus
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "otro resumen")
    _write_store(built / "items.json", changed)

    _update(built, changed, corpus, dry_run=True)

    assert index_build.status(built / "index", changed, built / "items.json").behind is True


def test_update_does_not_touch_items_json(built: Path, corpus) -> None:
    """Acceptance 13, on the incremental path too."""
    import hashlib

    store, _vocab, _pages = corpus
    path = built / "items.json"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    _update(built, store, corpus)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


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
    _write_store(built / "items.json", changed)
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
            _update(built, changed, corpus)
    finally:
        index_build.write_item = real  # type: ignore[assignment]

    assert _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid") == before
