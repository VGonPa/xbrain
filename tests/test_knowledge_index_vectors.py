# tests/test_knowledge_index_vectors.py
"""The vector plane WIRED INTO `index_build` — manifest, build, update, status (Plan 03.4).

03.3 shipped the plane's storage and its arithmetic; nothing built one, nothing declared one,
and nothing knew when one had gone stale. This file pins the three seams that close that gap,
and each is here because losing it fails SILENTLY rather than loudly.

1. **THE MANIFEST DECLARES THE SPEC, TOTAL AND CLOSED.** The `embeddings` slot shipped in 02.6
   with its SHAPE checked and its INSIDE unvalidated — `null` or an object, and any object.
   A block that drops `passage_prefix` therefore loaded fine, and a query embedded with no
   prefix against a corpus embedded with one is well-formed, unit-length and simply answers
   another question. The block is the `VectorSpec`, read BOTH ways like every other nested
   schema in that reader.

2. **A SPEC CHANGE COSTS THE MATRIX AND NOTHING ELSE** (spec §5.5, criterion §13.5). This is
   the one property the whole plane is arranged around, and the trap is that the cheapest
   implementation — routing the refusal through `IndexIncompatibleError` — is INVISIBLY wrong:
   the sentence would order `xbrain index build --force`, which throws away a lexical index
   that took minutes and is not stale at all. The test therefore asserts what SURVIVES the
   refusal, not only what raises.

3. **AN UPDATE LEAVES THE PLANE BEHIND, AND SAYS SO.** `update` rewrites the chunks of every
   item it touched, and a chunk id is content-derived — so the new ids have no row and the old
   rows have no chunk. Both halves are silent by construction: a missing row is a fragment no
   vector query can reach, and an orphaned row is a slot spent on nothing. Nothing is repaired
   here (re-embedding is a subprocess, and `update` has no embedder); it is DECLARED.

WHY THIS FILE CARRIES NO `pytest.importorskip("numpy")`: the same reason
`test_knowledge_vector_index.py` carries none. CI syncs `--extra embeddings`, so an absent
`numpy` is a broken environment and not a supported configuration, and a skip would hand back
a green `quality` check that exercised none of this.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields as dataclass_fields
from datetime import timedelta
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.index_schema import (
    REBUILD_ADVICE,
    IndexIncompatibleError,
    db_path,
    manifest_path,
    open_index,
)
from xbrain.knowledge.vector_index import (
    VECTORS_FILENAME,
    VECTOR_REBUILD_ADVICE,
    VectorPlaneIncompatible,
    VectorSpec,
    load_vector_plane,
    text_fingerprint,
    vector_plane_exists,
)
from xbrain.models import Item, Topic, TopicPage
from xbrain.rubrics import save_vocab
from xbrain.store import save_store, save_topic_pages

FIXTURES = Path(__file__).parent / "fixtures"

# One spec for the file, so a test about a spec CHANGE has to name the change itself.
SPEC = VectorSpec(
    model="intfloat/multilingual-e5-base",
    dimension=2,
    normalized=True,
    query_prefix="query: ",
    passage_prefix="passage: ",
)

# The five names spec §5.6's `embeddings` block carries, WRITTEN OUT BY HAND. This is the one
# side of the comparison that does not come out of `vector_index`, which is what stops the
# assertion from being `VectorSpec` compared against itself (rule 1).
SPEC_FIELD_NAMES = frozenset({"model", "dimension", "normalized", "query_prefix", "passage_prefix"})


# --------------------------------------------------------------------------- the fake embedder


class Embedder:
    """A deterministic stand-in for the external subprocess, that COUNTS what it was asked.

    Every call is recorded, because the properties under test are about WHAT reaches the
    embedder — the base's own text, each distinct body exactly once, nothing at all under a
    dry run — and a fake that only returned numbers could not tell any of them apart.

    The vectors are a unit circle position derived from the text's hash: distinct texts get
    distinct directions, identical texts get identical numbers, and every row is exactly unit
    length, which `write_vector_plane` verifies rather than assumes.
    """

    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(text for batch in self.batches for text in batch)

    def __call__(self, texts):  # noqa: ANN001, ANN204 - the seam's own signature
        batch = tuple(texts)
        self.batches.append(batch)
        return [self._vector(text) for text in batch]

    @staticmethod
    def _vector(text: str) -> tuple[float, ...]:
        angle = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        angle *= 2 * math.pi
        return (math.cos(angle), math.sin(angle))


class ShortChanging:
    """An embedder that answers one vector fewer than it was asked for."""

    def __call__(self, texts):  # noqa: ANN001, ANN204
        return [Embedder._vector(text) for text in list(texts)[:-1]]


class Exploding:
    """An embedder that dies mid-corpus, like a backend that goes away on chunk 9.000."""

    def __call__(self, texts):  # noqa: ANN001, ANN204
        raise RuntimeError("the embedder went away")


# --------------------------------------------------------------------------------- the corpus


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


def _paths(data: Path) -> tuple[Path, Path, Path]:
    return data / "items.json", data / "vocab.yaml", data / "topics.json"


def _persist(data: Path, *, store=None, vocab=None, pages=None) -> None:
    items_path, vocab_path, topics_path = _paths(data)
    if store is not None:
        save_store(store, items_path)
    if vocab is not None:
        save_vocab(vocab, vocab_path)
    if pages is not None:
        save_topic_pages(pages, topics_path)


def _inputs(data: Path) -> index_build.IndexInputs:
    return index_build.load_index_inputs(*_paths(data))


@pytest.fixture()
def data(tmp_path: Path, corpus) -> Path:
    """A `data/` with all three inputs on disk, and NO index yet."""
    store, vocab, pages = corpus
    root = tmp_path / "data"
    _persist(root, store=store, vocab=vocab, pages=pages)
    return root


def _build(data: Path, **kwargs) -> index_build.BuildReport:
    return index_build.build(data / "index", _inputs(data), **kwargs)


def _update(data: Path, **kwargs) -> index_build.UpdateReport:
    return index_build.update(data / "index", _inputs(data), **kwargs)


def _status(data: Path, **kwargs) -> index_build.StatusReport:
    return index_build.status(data / "index", _inputs(data), **kwargs)


def _built(data: Path, embedder: Embedder | None = None) -> Embedder:
    """Build the index WITH a vector plane and hand back the embedder that fed it."""
    embedder = embedder or Embedder()
    _build(data, vectors=index_build.VectorBuild(spec=SPEC, embed=embedder))
    return embedder


def _chunk_text(data: Path) -> dict[str, str]:
    """`{chunk_id: text}` as the LEXICAL plane holds it — the corpus a query actually serves."""
    connection = open_index(db_path(data / "index"), read_only=True)
    try:
        return {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT chunk_id, text FROM chunks")
        }
    finally:
        connection.close()


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


# ------------------------------------------------------- 1. the manifest declares the spec


def test_the_embeddings_block_carries_exactly_the_spec_fields() -> None:
    """The block's schema is READ OFF `VectorSpec`, and the names come from the spec by hand.

    `EMBEDDINGS_FIELDS` is derived in `index_build` so a sixth field added to `VectorSpec`
    cannot leave the manifest declaring five; the literal set here is the half written out
    from spec §5.6, so the comparison is not the dataclass against itself.
    """
    assert index_build.EMBEDDINGS_FIELDS == SPEC_FIELD_NAMES
    assert index_build.EMBEDDINGS_FIELDS == {f.name for f in dataclass_fields(VectorSpec)}


def test_an_index_built_without_an_embedder_declares_no_plane(data: Path) -> None:
    """The plane is OPT-IN end to end: no `vectors=`, no files, `embeddings: null`.

    Green before 03.4 and kept anyway: it is what stops a later default from quietly making
    every build pay for a subprocess, and it is the state `manifest_spec` must answer `None`
    for rather than raising.
    """
    _build(data)
    manifest = index_build.load_manifest(data / "index")
    assert manifest.embeddings is None
    assert index_build.manifest_spec(manifest) is None
    assert not vector_plane_exists(data / "index")


def test_the_manifest_records_the_spec_the_plane_was_written_under(data: Path) -> None:
    """A build with an embedder seals the spec, and it round-trips back as a `VectorSpec`."""
    _built(data)
    manifest = index_build.load_manifest(data / "index")
    assert index_build.manifest_spec(manifest) == SPEC


def test_a_manifest_whose_block_drops_a_field_is_refused(data: Path) -> None:
    """A block missing `passage_prefix` is NOT a block with a default (rule: total).

    This is the defect the unvalidated slot allowed: the prefix is a property of the model,
    a query embedded without it is still unit-length, and the only thing that would ever have
    said so is this reader.
    """
    _built(data)
    path = manifest_path(data / "index")
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["embeddings"]["passage_prefix"]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IndexIncompatibleError) as error:
        index_build.load_manifest(data / "index")
    assert "passage_prefix" in str(error.value)


def test_a_manifest_whose_block_declares_more_is_refused(data: Path) -> None:
    """And CLOSED: a block from a writer watching something this code cannot see is refused.

    Dropping the extra key would certify the plane current over a property nobody looked at —
    a quantization, a pooling strategy — which is the fail-open half `Manifest.from_dict`
    already closes for every other nested schema.
    """
    _built(data)
    path = manifest_path(data / "index")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["embeddings"]["quantization"] = "int8"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IndexIncompatibleError) as error:
        index_build.load_manifest(data / "index")
    assert "quantization" in str(error.value)


def test_a_block_whose_dimension_is_not_a_number_is_refused(data: Path) -> None:
    """`dimension` is the width of the matrix: a string there is a hand edit, not a spec."""
    _built(data)
    path = manifest_path(data / "index")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["embeddings"]["dimension"] = "768"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IndexIncompatibleError):
        index_build.load_manifest(data / "index")


# ---------------------------------------------------------------- 2. `build` writes the plane


def test_build_writes_a_plane_covering_every_chunk_the_base_holds(data: Path) -> None:
    """The plane's chunk ids are the base's chunk ids — no fragment is unreachable by vector.

    Asserted as a SET EQUALITY against the `chunks` table rather than as a count, because a
    count matches while the two populations differ, and the difference is exactly the silent
    half: a chunk with no row never comes back, and a row with no chunk spends a slot.
    """
    _built(data)
    plane = load_vector_plane(data / "index")
    try:
        stored = set(_chunk_text(data))
        assert stored
        assert {cid for row in range(plane.row_count) for cid in plane.chunk_ids_for_row(row)} == (
            stored
        )
    finally:
        plane.close()


def test_the_embedder_is_handed_the_text_the_lexical_plane_serves(data: Path) -> None:
    """ONE corpus, read back from the base (rule 5), never re-derived from the store.

    A second walk of the emitters would embed a body the `chunks` table does not hold, and the
    two descriptions would stay internally consistent while ranking different text.
    """
    embedder = _built(data)
    assert set(embedder.texts) == set(_chunk_text(data).values())


def test_identical_text_is_embedded_once_and_shares_one_row(data: Path, corpus) -> None:
    """Dedupe happens BEFORE the subprocess, which is the only place it saves anything.

    `write_vector_plane` collapses identical text onto one row either way, so a build that
    embedded first and deduped after would be correct and would still pay the model for every
    duplicate — invisible in the output, visible only here.
    """
    store, vocab, pages = corpus
    twin = next(iter(store.values()))
    store[twin.id + "0"] = twin.model_copy(update={"id": twin.id + "0"})
    _persist(data, store=store, vocab=vocab, pages=pages)

    embedder = _built(data)
    texts = _chunk_text(data).values()
    distinct = {text_fingerprint(text) for text in texts}
    assert len(embedder.texts) == len(distinct) < len(list(texts))
    assert len(embedder.texts) == len(set(embedder.texts))


def test_the_build_report_counts_the_chunks_and_the_rows(data: Path) -> None:
    """`chunks - rows` is how much of the corpus was duplicate prose: reported, not derived."""
    store_report = index_build.build(
        data / "index", _inputs(data), vectors=index_build.VectorBuild(spec=SPEC, embed=Embedder())
    )
    plane = load_vector_plane(data / "index")
    try:
        assert store_report.vector_chunks == plane.chunk_count
        assert store_report.vector_rows == plane.row_count
    finally:
        plane.close()


def test_a_build_without_an_embedder_reports_no_vector_numbers(data: Path) -> None:
    """`None`, never `0`: a plane that was never asked for is not a plane of zero rows."""
    report = _build(data)
    assert report.vector_chunks is None
    assert report.vector_rows is None


def test_an_empty_corpus_writes_an_empty_plane_and_never_calls_the_embedder(
    tmp_path: Path,
) -> None:
    """`embeddings.embed_texts` RAISES on an empty batch and names this caller as the guard.

    Its docstring: «`xbrain index build` checks for an empty corpus before it gets here» —
    there is nothing to embed and no dimension a response could be validated against. An empty
    corpus is not an error, it is an index of zero chunks, so the honest result is a plane of
    zero rows rather than a crash out of the subprocess contract.
    """
    empty = tmp_path / "data"
    _persist(empty, store={}, vocab=[], pages={})
    embedder = Embedder()

    report = _build(empty, vectors=index_build.VectorBuild(spec=SPEC, embed=embedder))

    assert embedder.batches == []
    assert (report.vector_chunks, report.vector_rows) == (0, 0)
    plane = load_vector_plane(empty / "index")
    try:
        assert plane.row_count == 0
    finally:
        plane.close()


def test_a_dry_run_never_embeds_and_writes_no_plane(data: Path) -> None:
    """The flag's whole promise is that it changes nothing — and costs nothing.

    The embedder is the expensive half of this build, so a dry run that invoked it would spend
    the minutes it exists to avoid, on a run whose output is thrown away.
    """
    embedder = Embedder()
    report = _build(data, dry_run=True, vectors=index_build.VectorBuild(spec=SPEC, embed=embedder))
    assert embedder.batches == []
    assert not vector_plane_exists(data / "index")
    assert report.dry_run


def test_a_dry_run_still_reports_the_rows_a_real_build_would_write(data: Path) -> None:
    """And the numbers are MEASURED, not estimated: the dedupe key is a hash, not a model.

    `text_fingerprint` is free, so the row count a real build would produce is knowable
    without one subprocess call — which is what lets a dry run answer the question an
    operator actually asks before paying for an embedding run.
    """
    dry = _build(data, dry_run=True, vectors=index_build.VectorBuild(spec=SPEC, embed=Embedder()))
    wet = _build(data, vectors=index_build.VectorBuild(spec=SPEC, embed=Embedder()))
    assert (dry.vector_chunks, dry.vector_rows) == (wet.vector_chunks, wet.vector_rows)
    assert dry.vector_rows and dry.vector_rows > 0


def test_a_forced_rebuild_without_an_embedder_removes_the_previous_plane(data: Path) -> None:
    """Otherwise a manifest declaring NO plane stands over a matrix from another corpus.

    The files are on disk and nothing in the manifest mentions them, so `vector_plane_exists`
    answers True to anyone who asks the filesystem — and the rows it holds are keyed by chunk
    ids the new base may not even contain.
    """
    _built(data)
    assert vector_plane_exists(data / "index")
    _build(data, force=True)
    assert not vector_plane_exists(data / "index")
    assert index_build.load_manifest(data / "index").embeddings is None


def test_an_embedder_that_answers_fewer_vectors_is_refused_before_a_byte_lands(
    data: Path,
) -> None:
    """Vectors are paired with texts BY POSITION, so a short answer shifts every later one.

    `zip` would truncate silently and hand each fragment after the gap the vector of its
    neighbour: well-formed, unit-length, and attributed to the wrong text. The count is checked
    before anything is paired, and the refusal leaves no plane and no manifest behind.
    """
    with pytest.raises(VectorPlaneIncompatible, match="vectores para"):
        _build(data, vectors=index_build.VectorBuild(spec=SPEC, embed=ShortChanging()))
    assert not vector_plane_exists(data / "index")
    assert not manifest_path(data / "index").exists()


def test_an_embedder_that_dies_leaves_no_manifest(data: Path) -> None:
    """A half-built index is REFUSED, never answered partially (spec §9.3).

    The manifest is what every door trusts, so it must not be sealed over a plane that was
    never finished. The repair is `xbrain index build` again, exactly as for an interrupted
    lexical build.
    """
    with pytest.raises(RuntimeError):
        _build(data, vectors=index_build.VectorBuild(spec=SPEC, embed=Exploding()))
    assert not manifest_path(data / "index").exists()


# ------------------------------------------------- 3. selective invalidation (§13.5)


def test_a_spec_change_invalidates_the_plane_and_nothing_else(data: Path) -> None:
    """THE criterion of §13.5, asserted on what SURVIVES and not only on what refuses.

    A model change leaves a matrix answering with another geometry, so the plane is unusable —
    and it leaves the SQLite base cut by the same chunker, sealed under the same versions and
    correct in every column. Routing this through `IndexIncompatibleError` would be invisibly
    wrong: the operator would be told to throw away minutes of lexical work that is not stale.
    """
    _built(data)
    manifest = index_build.load_manifest(data / "index")
    other = VectorSpec(
        **{
            **{f.name: getattr(SPEC, f.name) for f in dataclass_fields(SPEC)},
            "model": "BAAI/bge-m3",
        }
    )

    verdict = index_build.vector_verdict(data / "index", manifest, expected=other)

    assert verdict.state == "spec_changed"
    assert not verdict.usable
    # what survives: the lexical index is still compatible, still populated, still there.
    assert index_build.load_compatible_manifest(data / "index") == manifest
    assert db_path(data / "index").exists()
    assert len(_chunk_text(data)) == manifest.counts["chunks"]


def test_the_spec_change_sentence_never_orders_a_full_rebuild(data: Path) -> None:
    """It names `--embeddings`, and it is NOT `index_schema.REBUILD_ADVICE`.

    Two constants, two costs. Asserting the identity of the one it uses — rather than that the
    word «rebuild» appears somewhere — is what stops a later edit from swapping in the
    expensive sentence with the test still green (rule 1).
    """
    _built(data)
    manifest = index_build.load_manifest(data / "index")
    other = VectorSpec(
        **{**{f.name: getattr(SPEC, f.name) for f in dataclass_fields(SPEC)}, "dimension": 3}
    )

    sentence = index_build.vector_verdict(data / "index", manifest, expected=other).sentence

    assert VECTOR_REBUILD_ADVICE in sentence
    assert REBUILD_ADVICE not in sentence


def test_a_spec_change_is_decided_without_reading_the_matrix(data: Path) -> None:
    """Two dataclasses answer it, so a truncated matrix does not get to answer it instead.

    Deciding this from a FAILED LOAD would label a corrupt digest `spec_changed` whenever the
    config happened to differ, and it would need `numpy` to compare two specs — so `index
    status` on a machine without the `[embeddings]` extra could not report the one thing spec
    §5.5 is about. Staged by truncating the matrix: under the manifest's own spec that plane
    is `unreadable`, and under a different one it is still `spec_changed`.
    """
    _built(data)
    manifest = index_build.load_manifest(data / "index")
    (data / "index" / VECTORS_FILENAME).write_bytes(b"\x00\x00\x00\x00")
    other = VectorSpec(
        **{**{f.name: getattr(SPEC, f.name) for f in dataclass_fields(SPEC)}, "model": "otro"}
    )

    assert index_build.vector_verdict(data / "index", manifest).state == "unreadable"
    assert (
        index_build.vector_verdict(data / "index", manifest, expected=other).state == "spec_changed"
    )


def test_a_plane_read_under_its_own_spec_is_current(data: Path) -> None:
    _built(data)
    manifest = index_build.load_manifest(data / "index")
    verdict = index_build.vector_verdict(data / "index", manifest, expected=SPEC)
    assert verdict.state == "current"
    assert verdict.usable
    assert verdict.sentence == ""
    assert verdict.spec == SPEC


def test_no_plane_and_no_declaration_is_absent_not_broken(data: Path) -> None:
    """The supported opt-out. `absent` is not `usable`, and it is not an error either."""
    _build(data)
    verdict = index_build.vector_verdict(data / "index", index_build.load_manifest(data / "index"))
    assert verdict.state == "absent"
    assert not verdict.usable
    assert verdict.sentence == ""


def test_a_manifest_declaring_a_plane_whose_matrix_is_gone_is_refused(data: Path) -> None:
    """Half a plane is not a plane: a meta with no matrix describes nothing."""
    _built(data)
    (data / "index" / VECTORS_FILENAME).unlink()
    verdict = index_build.vector_verdict(
        data / "index", index_build.load_manifest(data / "index"), expected=SPEC
    )
    assert verdict.state == "missing"
    assert not verdict.usable
    assert VECTOR_REBUILD_ADVICE in verdict.sentence


def test_files_on_disk_the_manifest_does_not_declare_are_refused(data: Path) -> None:
    """The other direction, and it is the one a `--force` without an embedder used to leave.

    A matrix nothing declares is a matrix nobody can check the spec of, so serving from it is
    serving another model's geometry with the manifest saying there are no embeddings at all.
    """
    _built(data)
    path = manifest_path(data / "index")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["embeddings"] = None
    path.write_text(json.dumps(document), encoding="utf-8")
    verdict = index_build.vector_verdict(data / "index", index_build.load_manifest(data / "index"))
    assert verdict.state == "undeclared"
    assert not verdict.usable
    assert VECTOR_REBUILD_ADVICE in verdict.sentence


# ------------------------------------------- 3b. an update leaves the plane behind, and says so


def test_an_update_that_changes_nothing_leaves_the_plane_complete(data: Path) -> None:
    """The baseline the next test is only meaningful against."""
    _built(data)
    report = _update(data)
    assert report.vector_missing == 0
    assert report.vector_orphaned == 0


def test_an_edit_leaves_the_vector_stale_while_every_id_still_resolves(data: Path, corpus) -> None:
    """THE SILENT HALF, and the reason coverage is not a set of ids.

    A `chunk_id` is POSITIONAL — `<surface_id>:<chunk_index>:<chunker_version>` — so `enrich`
    rewriting a summary changes the chunk's prose and leaves its id untouched. Measured on
    this corpus: the update inserts and deletes 2 chunks and ORPHANS NOTHING, because the ids
    that came back are the ids that left. An id-only coverage check therefore reports a
    complete plane over a row that answers with the geometry of the previous summary.

    `orphaned == 0` is asserted, not omitted: it is what pins the id-only check as the wrong
    instrument, and a test that only looked at `missing` would pass under it too.
    """
    store, vocab, pages = corpus
    _built(data)
    edited = next(i for i in store.values() if i.enriched is not None)
    store[edited.id] = _edit_summary(edited, "un resumen completamente distinto del anterior")
    _persist(data, store=store)

    report = _update(data)

    assert report.items_changed == 1
    assert report.vector_missing > 0
    assert report.vector_orphaned == 0


def test_a_removed_item_orphans_the_rows_its_chunks_read(data: Path, corpus) -> None:
    """The OTHER half, and the one that does move ids: an owner that is gone takes its ids.

    The rows stay in the matrix — Plan 03 §2.3 makes an orphan a designed state, compacted by
    the next rebuild — and counting them is what tells an operator how much of the plane is
    now answering for nothing.
    """
    store, vocab, pages = corpus
    _built(data)
    removed = next(iter(store))
    del store[removed]
    _persist(data, store=store)

    report = _update(data)

    assert report.items_removed == 1
    assert report.vector_orphaned > 0
    assert report.vector_missing == 0


def test_an_update_carries_the_spec_forward_verbatim(data: Path, corpus) -> None:
    """The versions an update INHERITS now include the spec: it did not re-embed, so it did
    not re-decide what produced the numbers."""
    store, vocab, pages = corpus
    _built(data)
    edited = next(i for i in store.values() if i.enriched is not None)
    store[edited.id] = _edit_summary(edited, "otro resumen")
    _persist(data, store=store)
    _update(data)
    assert index_build.manifest_spec(index_build.load_manifest(data / "index")) == SPEC


def test_an_update_over_an_index_with_no_plane_reports_no_vector_numbers(data: Path) -> None:
    """`None`, never `0` — the same distinction `BuildReport` makes, for the same reason."""
    _build(data)
    report = _update(data)
    assert report.vector_missing is None
    assert report.vector_orphaned is None
    assert report.vector_state is None


@pytest.mark.parametrize(
    ("break_matrix", "state"),
    [
        (lambda path: path.unlink(), "missing"),
        (lambda path: path.write_bytes(b"not a matrix"), "unreadable"),
    ],
    ids=["matrix-deleted", "matrix-unreadable"],
)
def test_an_update_over_a_plane_it_cannot_read_reports_no_debt_it_never_measured(
    data: Path, break_matrix, state: str
) -> None:
    """A plane nobody could open owes an UNKNOWN amount, and `0` is a claim that it owes none.

    `VectorVerdict` defaults both counts to `0` on every state that never reached `_coverage`,
    so copying them into the report published `0 / 0` — the exact numbers of a complete plane —
    for a matrix that is not on disk. The counts come out `None` and the state says why, so an
    unreadable plane cannot be confused with the opt-out either.
    """
    _built(data)
    break_matrix(data / "index" / VECTORS_FILENAME)

    report = _update(data)

    assert report.vector_missing is None
    assert report.vector_orphaned is None
    assert report.vector_state == state


# --------------------------------------------------------------------- 3c. `status` agrees


def test_status_reports_the_plane_as_current_after_a_build(data: Path) -> None:
    _built(data)
    report = _status(data)
    assert report.vector is not None
    assert report.vector.state == "current"


def test_status_declares_a_plane_the_store_has_moved_past(data: Path, corpus) -> None:
    """`status` is the instrument an operator runs to find out, so it must not be the one
    that stays quiet (rule 9): `update` counted the gap, `status` names it in its advice."""
    store, vocab, pages = corpus
    _built(data)
    edited = next(i for i in store.values() if i.enriched is not None)
    store[edited.id] = _edit_summary(edited, "un resumen que mueve los chunks")
    _persist(data, store=store)
    _update(data)

    report = _status(data)

    assert report.vector is not None
    assert report.vector.state == "behind"
    assert not report.vector.usable
    assert VECTOR_REBUILD_ADVICE in report.advice


def test_status_over_an_index_with_no_plane_says_absent_and_advises_nothing(data: Path) -> None:
    """An opt-out is not a defect: nothing about the vector plane reaches the advice."""
    _build(data)
    report = _status(data)
    assert report.vector is not None
    assert report.vector.state == "absent"
    assert VECTOR_REBUILD_ADVICE not in report.advice
