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
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.chunking import ChunkerParams
from xbrain.knowledge.index_schema import (
    FTS_TABLES,
    TABLES,
    IndexIncompatibleError,
    db_path,
    manifest_path,
    open_index,
)
from xbrain.knowledge.surfaces import item_surfaces, knowledge_item
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


def _status(data: Path, store, corpus) -> index_build.StatusReport:
    """`status` takes the vocabulary and the pages like `build`/`update` do (H1)."""
    _vocab_store, vocab, pages = corpus
    return index_build.status(data / "index", store, vocab, pages, data / "items.json")


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


def _reassign(item: Item, slug: str) -> Item:
    """The OTHER change `enrich` makes: a new topic assignment, text untouched (H1)."""
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
    topics_before = _rows(built, "SELECT * FROM topics ORDER BY slug")
    report = _update(built, store, corpus)
    assert (report.items_added, report.items_changed, report.items_removed) == (0, 0, 0)
    assert (report.chunks_inserted, report.chunks_deleted) == (0, 0)
    assert _rows(built, "SELECT chunk_id, rowid FROM chunks ORDER BY rowid") == before
    # The topic ROWS too (H1): the membership refresh compares before it writes, so a
    # store that did not move rewrites no topic row either.
    assert _rows(built, "SELECT * FROM topics ORDER BY slug") == topics_before
    assert report.topics_refreshed == 0


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
    """Step 6: 960 of 2,404 real items have NO `content`, so `fetched_at` reaches none of them.

    (Measured 2026-09-01 on `data/items.json`, sha256 `f76341a3…`. The claim holds with
    either number; 961 was one item stale, F-14.)

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
    raw["chunker_version"] = "xbrain-knowledge-chunker/v99"
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
            # A target the default is NOT: `ChunkerParams(target=800)` was this line until
            # Plan 02 §7's sweep made 800 the default, at which point the test compared the
            # default with itself and could only pass by accident (rule 1).
            options=index_build.IndexOptions(params=ChunkerParams(target=1234)),
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
    assert _status(built, changed, corpus).behind is True

    _update(built, changed, corpus)

    after = _status(built, changed, corpus)
    assert after.behind is False and after.items_changed == 0


def test_a_dry_run_update_does_not_refresh_the_manifest(built: Path, corpus) -> None:
    """The complement, and the sharper half: a dry run that stamped the manifest would make
    the index CLAIM to be current while holding the old rows — worse than doing nothing."""
    store, _vocab, _pages = corpus
    changed = dict(store)
    changed["k02"] = _edit_summary(store["k02"], "otro resumen")
    _write_store(built / "items.json", changed)

    _update(built, changed, corpus, dry_run=True)

    assert _status(built, changed, corpus).behind is True


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
    """A-3 (round 02, Codex F-05): `_next_manifest` updated `items`, `topics`, `chunks` and
    `profiles` and CARRIED `counts["surfaces"]`, every `skipped` counter and `failed` over
    from the previous manifest. Remove an item with two surfaces and a failed fetch and the
    base was right while the manifest kept publishing the old population — `surfaces 43`
    against 41 rows, `failed_sources 1` against 0 — and `index status --json` exposed both
    as they were. Spec §5.6 asks for counts and omissions that are VALID, not merely present
    (acceptance 2).

    Every count and every omission is now DERIVED FROM THE DATABASE, by the same function a
    fresh build uses, so the two writers cannot disagree (rule 5). Seen red before the fix:
    `43 == 41` and `1 == 0`.
    """
    store, _vocab, _pages = corpus
    victim = "k11"  # two surfaces and a failed fetch, on this fixture
    assert len(item_surfaces(store[victim])) >= 2
    assert knowledge_item(store[victim]).failed_sources
    before = _manifest(built)

    smaller = {k: v for k, v in store.items() if k != victim}
    _write_store(built / "items.json", smaller)
    _update(built, smaller, corpus)

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
    """The cumulative drift the Claude gate measured (M-4 there, A-3 here): with topics
    rebuilt, `_clear_topics` discarded the return of `delete_chunk_rows`, so each update
    with a changed vocabulary ADDED the topic chunks to `counts["chunks"]` without ever
    subtracting them — 84 declared against 56 real after four updates. Deriving the counts
    from the database makes the invariant hold by construction; this test pins it across
    three updates that each force a topic rebuild.
    """
    store, vocab, pages = corpus
    for round_ in range(3):
        slug = next(iter(pages))
        pages = {
            **pages,
            slug: pages[slug].model_copy(update={"overview": f"round {round_} overview"}),
        }
        report = index_build.update(built / "index", store, vocab, pages, built / "items.json")
        assert report.topics_rebuilt is True
        assert _manifest(built)["counts"] == _db_counts(built), f"drifted on round {round_}"


def test_update_refuses_a_database_that_disagrees_with_its_manifest(built: Path, corpus) -> None:
    """C-3 (round 02, Claude gate H-3): `update` decided `topics_rebuilt` by comparing the
    vocabulary and topic-page fingerprints against the MANIFEST, never against what the
    database contains. On the real corpus, after an interrupted forced rebuild (C-1), the
    old manifest declared 45 topics over a base that held 0: `update` inserted every item,
    never rebuilt the topic plane — 45 topics, 616 surfaces, 703 chunks gone FOR GOOD,
    since no later update would find a fingerprint to disagree with — and `status` answered
    `incomplete=False` with an empty advice.

    C-1 closes that route (no manifest survives), so the state is staged directly: the
    topic plane is deleted behind the manifest's back. An incremental update over a base
    that does not contain what its manifest declares has no honest baseline, so it REFUSES
    and names the rebuild, and `status` says INCOMPLETE and names it too — the declaration
    the gate found missing.

    Seen red before the fix: `update` returned an `UpdateReport` with `topics_rebuilt=False`
    and `status` reported `incomplete=False`.
    """
    store, vocab, pages = corpus
    connection = open_index(db_path(built / "index"))
    try:
        with connection:
            index_build._clear_topics(connection)
    finally:
        connection.close()
    assert _db_counts(built)["topics"] == 0 < _manifest(built)["counts"]["topics"]

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        _update(built, store, corpus)
    assert "topics" in str(caught.value), "names WHICH plane disagrees"

    report = _status(built, store, corpus)
    assert report.incomplete is True
    assert "xbrain index build --force" in report.advice
    assert "topics" in report.advice


def test_update_over_a_missing_database_creates_nothing_and_leaves_search_closed(
    built: Path, corpus
) -> None:
    """G-2 (gate round 04): `index update` — even `--dry-run` — CREATED an empty database
    under a manifest that was still standing, and `search` went from failing closed to
    answering open.

    `update()` opened the database for writing, and `open_index` without `read_only` did
    `mkdir + connect + create_schema` whenever the file did not exist; `_require_consistent`
    then detected `chunks 0 != …` and raised — with the file already created. `open_for_query`
    only checked `exists()` plus a compatible manifest, so the next `search` answered "no
    results" with exit 0 over a base of zero rows while `status` said `incomplete=True`: two
    instruments, opposite answers (rule 9), and the exact shape of CRITICAL C-1 reached by
    the command an operator runs "to see what happens". Reproduced on the real corpus through
    the CLI: `rm knowledge.db` (52 MB, a natural clean-up target) → `update --dry-run` exit 1
    with the right message AND a 167,936-byte `knowledge.db` → `search` exit 0, «Sin
    resultados».

    Three closures, asserted through the public functions: `update` refuses BEFORE touching
    the disk and names `--force` (plain `build` refuses while the manifest exists — the dead
    end B-r is about); no file appears; `search` keeps refusing.

    Seen red before the fix: `db_path(...).exists()` was True after the dry run and `search`
    returned a `SearchResponse`.
    """
    from xbrain.knowledge.index_schema import IndexError_, IndexMissingError
    from xbrain.knowledge.search_service import QueryContext, search

    store, vocab, pages = corpus
    db_path(built / "index").unlink()
    assert manifest_path(built / "index").exists(), "the manifest is what makes this state"

    with pytest.raises(IndexMissingError, match="xbrain index build --force"):
        _update(built, store, corpus, dry_run=True)
    assert not db_path(built / "index").exists(), "update must not create the base"

    with pytest.raises(IndexMissingError, match="xbrain index build --force"):
        _update(built, store, corpus)
    assert not db_path(built / "index").exists()

    context = QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=built / "index",
        items_path=built / "items.json",
    )
    with pytest.raises(IndexError_, match="xbrain index build --force"):
        search("Quillfeather", context)


def test_update_sees_a_quoted_author_repaired_without_touching_the_body(
    built: Path, corpus
) -> None:
    """G-5 (gate round 04): rule 6 on the attribution rule this repo says it paid for in blood.

    `item_fingerprint` hashed the item's metadata plus each surface's `surface_fingerprint`
    — `(version, type, origin, TEXT)` — and nothing else about the surface. So a repair that
    fills in the AUTHOR of a quoted post without touching its body (`refresh-quoted` on a
    quoted fetch that came back authorless; 826 `quoted_tweet` sources on the real store, 29
    of them failures with no author yet) left `update` reporting `0 cambiados` and `search`
    serving the old attribution from `surfaces.attribution_handle` — the repaired evidence
    with the derivative standing. Title, language and locator had the same blind spot, and
    the index stores and serves all four (A-1).

    Seen red before the fix: `items_changed == 0` and the stored `attribution_handle` was
    still `othervoice`.
    """
    from xbrain.models import Author

    store, _vocab, _pages = corpus
    item = store["k07"]
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
    _write_store(built / "items.json", repaired)

    report = _update(built, repaired, corpus)

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
    """G-8 (gate round 04): the WRITE door of the C-2 schema guard had no test.

    `open_index` in write mode verifies the schema of an EXISTING database so `update` does
    not carry on over a dropped table — `create_schema` is idempotent and would re-create it
    EMPTY. The read door had its test (`test_an_index_missing_a_table_is_refused_naming_the_
    rebuild`); this one did not, and the gate measured what removing the call costs: 119
    tests still green, and then `DROP TABLE chunks_fts` → `update` returns normally (an empty
    `chunks_fts` under 56 full `chunks` rows), `status.incomplete=False`, and `search`
    answers with 0 matches and `degraded: ["no_embeddings"]` — the silent mode of C-2,
    re-entered through `update`.

    Parametrised over the DECLARED set (`TABLES | FTS_TABLES`), like the DDL test, so a table
    added to the schema is covered without anyone remembering. `update` and `status` are both
    asserted, and on the same two facts: the error names the TABLE and the rebuild command.

    Seen red by removing `_verify_schema` from the write door of `open_index` (in an isolated
    copy of the tree): ALL ELEVEN parametrisations fail, through two different doors. For
    the six tables no `COUNT(*)` watches (`chunks_fts`, `profiles_fts`, `item_topics`,
    `item_content_kinds`, `source_failures`, `unfetched_links`) `update` simply returns an
    `UpdateReport`. For the five counted planes `update` still raises — `require_consistent`
    sees `items 0 != 12` — but `create_schema` runs through `executescript`, which COMMITS,
    so the refused update has already re-created the table EMPTY on disk, and the `status`
    that follows finds a complete schema and merely declares a count mismatch instead of
    raising. A guard that repairs the schema on its way to refusing is the silent mode with
    one extra step, which is why both instruments are asserted here.
    """
    store, _vocab, _pages = corpus
    connection = sqlite3.connect(db_path(built / "index"))
    connection.execute(f"DROP TABLE {table}")  # nosec B608 — a test fixture, closed set
    connection.commit()
    connection.close()

    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        _update(built, store, corpus)
    assert table in str(caught.value), "names WHAT is missing, not only that something is"
    with pytest.raises(IndexIncompatibleError, match="xbrain index build --force") as caught:
        _status(built, store, corpus)
    assert table in str(caught.value)


def test_status_declares_a_manifest_the_code_cannot_use(built: Path, corpus) -> None:
    """The neighbour of C-3's declaration: a manifest from another chunker version.

    `search` and `update` refuse it with exit 1 and the rebuild advice; `status` read it
    without checking the versions and answered `incomplete=False`, `advice=''` — two
    instruments, opposite answers on the same state (rule 9). And its "incomplete" advice
    named plain `xbrain index build`, which REFUSES while a manifest exists: a dead end in
    two hops. `status` now applies the same compatibility check and names `--force`.

    Seen red before the fix: `incomplete is False`.
    """
    store, _vocab, _pages = corpus
    path = manifest_path(built / "index")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["chunker_version"] = "xbrain-knowledge-chunker/v0"
    path.write_text(json.dumps(raw), encoding="utf-8")

    report = _status(built, store, corpus)
    assert report.incomplete is True
    assert "xbrain index build --force" in report.advice


# ---------------------------------------------------------------------------
# H1 — a changed topic ASSIGNMENT reaches the topic plane, not only `item_topics`
# ---------------------------------------------------------------------------


def test_update_refreshes_the_topic_rows_when_an_items_topics_move(built: Path, corpus) -> None:
    """H1 (gate Codex, round 04 — CLAUDE.md rule 6 on the topic plane).

    `topics` stores `primary_item_ids_json`, `secondary_item_ids_json` and `stale`, but
    `topics_rebuilt` looked ONLY at the vocabulary and topic-page fingerprints, so a change
    of `Item.enriched.primary_topic` / `topics` — exactly what `enrich` writes — rewrote the
    item and `item_topics` and left the topic rows holding the OLD members and the OLD
    `stale` bit, while the manifest and `status` declared the index healthy. Reproduced on
    the fixture: k02 moved from `agent-evaluation` to `ai-policy`, `item_topics` said
    `ai-policy`, `topics` still listed k02 under `agent-evaluation`, `stale=0` on both.

    The fix is NARROW on purpose: the vocabulary and the pages did not move, so the topic
    SURFACES and CHUNKS are untouched (their rowids prove it) and only the rows whose
    membership or staleness differs are rewritten — through the same row projection the
    full writer uses. `stale` flips on both topics because the live primary count no longer
    equals `post_count_at_synth` (2 and 4 in the fixture pages).

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
    _write_store(built / "items.json", changed)
    report = _update(built, changed, corpus)

    after = _topic_rows(built)
    assert after["ai-policy"] == (["k02", "k03", "k04", "k07", "k11"], [], 1)
    assert after["agent-evaluation"] == (["k08"], [], 1)
    assert (
        _rows(built, "SELECT chunk_id, rowid FROM chunks WHERE owner_type = 'topic' ORDER BY rowid")
        == topic_chunks_before
    ), "the topic surfaces and chunks did not change, so they must not be rewritten"
    assert report.topics_rebuilt is False, "vocabulary and pages did not move"
    assert report.topics_refreshed == 2

    status = _status(built, changed, corpus)
    assert status.items_changed == 0 and status.topics_changed == 0
    assert status.advice == ""
