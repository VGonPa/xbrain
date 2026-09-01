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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS
from xbrain.knowledge.index_schema import (
    IndexIncompatibleError,
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
    assert raw["schema_version"] == "1"
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


def test_a_forced_rebuild_produces_a_file_the_same_size_as_a_fresh_one(
    workspace, corpus
) -> None:
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


def test_status_calls_an_index_without_a_manifest_incomplete(workspace, corpus) -> None:
    """The other half of step 9: the state is REPORTED, not merely absent."""
    _build(workspace, corpus)
    manifest_path(workspace / "index").unlink()
    report = index_build.status(workspace / "index", corpus[0], workspace / "items.json")
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
    hundred re-enriched items. Seen red by reporting a boolean.
    """
    store, vocab, pages = corpus
    _build(workspace, corpus)
    clean = index_build.status(workspace / "index", store, workspace / "items.json")
    assert clean.items_changed == 0 and clean.items_added == 0 and clean.items_removed == 0

    changed = dict(store)
    victim = changed["k02"]
    changed["k02"] = victim.model_copy(
        update={
            "enriched": victim.enriched.model_copy(
                update={
                    "summary": "un resumen completamente distinto",
                    "enriched_at": victim.enriched.enriched_at + timedelta(hours=1),
                }
            )
        }
    )
    del changed["k01"]
    after = index_build.status(workspace / "index", changed, workspace / "items.json")
    assert after.items_changed == 1
    assert after.items_removed == 1
    assert "xbrain index update" in after.advice


def test_status_reports_the_index_behind_the_store_from_the_cheap_signal(workspace, corpus) -> None:
    """B3: the mtime/size signal, read with an `os.stat`, on an explicit command too.

    A `touch` with no edit is a FALSE POSITIVE and that is accepted: the cost of one extra
    warning is a warning, and the cost of a false negative is serving stale evidence as
    fresh. It fails towards the warning, like `origin: unknown -> llm_synthesis`.
    """
    _build(workspace, corpus)
    assert (
        index_build.status(workspace / "index", corpus[0], workspace / "items.json").behind is False
    )
    path = workspace / "items.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert (
        index_build.status(workspace / "index", corpus[0], workspace / "items.json").behind is True
    )


# ---------------------------------------------------------------------------
# 29 — an incompatible manifest is never queried partially
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", "2"),
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
    reach an item whose `content` is `None` — 961 of 2,404 in the real store — because there
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
