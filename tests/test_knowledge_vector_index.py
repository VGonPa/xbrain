# tests/test_knowledge_vector_index.py
"""The vector plane's STORAGE and its cosine top-k (Plan 03 §2.1, §2.2, steps 9-12, 14, 26).

WHY THIS FILE CARRIES NO `pytest.importorskip("numpy")` (delivery §7.3c). The `[embeddings]`
extra is installed by CI (`uv sync --extra dev --extra embeddings --locked`), so `numpy`
missing is a BROKEN ENVIRONMENT, not a supported configuration. `importorskip` would turn
that into a silent skip and hand back a GREEN `quality` check that exercised none of the
vector plane; a plain top-level import turns it into a collection error, which is red. The
module under test still defers its own import — that is a different property, asserted below
by reading the source rather than by skipping.

THE THREE PROPERTIES THAT ARE NOT ARITHMETIC, and each is here because losing it fails
silently rather than loudly:

1. **Two chunks with identical text share a row and BOTH survive** (§13.6). The dedupe key is
   `sha256(text)`, so the natural-looking implementation keys the plane by that fingerprint —
   at which point the second chunk has no id in the index, never comes back from a search, and
   its owner, author and URL are unreachable. The vectors are identical either way, so nothing
   downstream can tell.
2. **The top-k is deterministic under ties** (spec §8.6 gate 2). Float scores tie constantly
   once the corpus holds near-duplicate prose, and `argpartition` breaks ties by whatever the
   partition happened to do — reproducible today, different tomorrow.
3. **The plane refuses to be read under a spec it was not written under** (§13.5). A model,
   dimension or prefix change leaves a matrix that is still well-formed, still unit-length and
   simply answers with another model's geometry.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xbrain.knowledge.vector_index import (
    META_SCHEMA_VERSION,
    VECTORS_FILENAME,
    VECTORS_META_FILENAME,
    ChunkVector,
    VectorBackendUnavailable,
    VectorPlaneIncompatible,
    VectorSpec,
    load_vector_plane,
    text_fingerprint,
    vector_plane_exists,
    write_vector_plane,
)

# One spec for the whole file, so a test that cares about a spec CHANGE has to say so.
SPEC = VectorSpec(
    model="intfloat/multilingual-e5-base",
    dimension=2,
    normalized=True,
    query_prefix="query: ",
    passage_prefix="passage: ",
)

# Unit vectors in two dimensions, exactly representable in float32 so an assertion on a score
# is an assertion about the ranking and not about rounding.
EAST = (1.0, 0.0)
NORTH = (0.0, 1.0)
DIAGONAL = (0.6, 0.8)


def chunk(chunk_id: str, text: str, vector: tuple[float, ...] = EAST) -> ChunkVector:
    """One chunk's row, named so a test reads as its property and not as its plumbing."""
    return ChunkVector(chunk_id=chunk_id, text=text, vector=vector)


def plane(tmp_path: Path, *chunks: ChunkVector, spec: VectorSpec = SPEC):
    """Write `chunks` and hand back the loaded plane."""
    write_vector_plane(tmp_path, spec, chunks)
    return load_vector_plane(tmp_path)


# ---------------------------------------------------------------- the extra is optional (26)


def test_the_package_imports_with_the_embeddings_extra_absent() -> None:
    """`import xbrain` must work for someone who never installed the extra (§13.12, m11).

    Run in a SUBPROCESS with `numpy` marked unimportable, because that is the property —
    asserted on the source instead (no top-level `import numpy` in this file) it would be
    satisfied by an import nested under any `if`, and satisfied without ever proving that the
    rest of the import chain stays clean either.
    """
    blocked = (
        "import sys; sys.modules['numpy'] = None; "
        "import xbrain; "
        "from xbrain.knowledge import vector_index; "
        "print(vector_index.VECTORS_FILENAME)"
    )
    run = subprocess.run(  # noqa: S603 - the interpreter running this very test
        [sys.executable, "-c", blocked],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert run.returncode == 0, run.stderr
    assert VECTORS_FILENAME in run.stdout


def test_a_missing_numpy_names_the_install_command_instead_of_an_import_error(monkeypatch) -> None:
    """Without the extra the operator gets an actionable error, never a raw `ImportError`.

    `None` in `sys.modules` is what CPython itself uses to mark a module as unimportable, so
    the deferred `import numpy` raises exactly the `ImportError` an un-synced extra raises —
    no stand-in for the import machinery, which would be asserting on a fake.
    """
    from xbrain.knowledge import vector_index

    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(VectorBackendUnavailable) as excinfo:
        vector_index._numpy()
    assert "xbrain[embeddings]" in str(excinfo.value)
    assert isinstance(excinfo.value, RuntimeError)


def test_a_search_without_numpy_names_the_install_command(tmp_path: Path, monkeypatch) -> None:
    """The error reaches the operator through a COMMAND, not only through the helper."""
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(VectorBackendUnavailable) as excinfo:
        loaded.search(EAST, limit=1)
    assert "xbrain[embeddings]" in str(excinfo.value)


# ------------------------------------------------------------------------ dedupe (§13.6, 9)


def test_the_dedupe_key_is_sha256_of_the_text() -> None:
    """The key Plan 03 §2.2 names, pinned against `hashlib` and not against itself."""
    assert text_fingerprint("un párrafo") == hashlib.sha256("un párrafo".encode()).hexdigest()


def test_two_chunks_with_identical_text_share_one_row(tmp_path: Path) -> None:
    """One row on disk for two chunks — the vector is shared (Plan 03 §2.2)."""
    quote = "el mismo párrafo citado por dos items"
    report = write_vector_plane(
        tmp_path, SPEC, [chunk("a:0:v3", quote, EAST), chunk("b:0:v3", quote, EAST)]
    )
    assert (report.chunks, report.rows, report.shared_rows) == (2, 1, 1)
    assert (tmp_path / VECTORS_FILENAME).stat().st_size == 1 * SPEC.dimension * 4


def test_both_owners_of_a_shared_row_keep_their_own_author_and_url(tmp_path: Path) -> None:
    """§13.6: the vector is shared, the ASSOCIATIONS are not.

    The plane stores no owner, author or URL — that metadata has one home, the lexical
    `chunks` table, and a second copy is the divergence rule 5 exists to stop. What §13.6
    requires of THIS module is that both chunk ids survive the dedupe, because a chunk id that
    does not come back is a chunk whose owner, author and URL nobody can reach. The dict below
    stands in for that table, and the assertion is that the join lands on TWO different
    owners: key the plane by the text fingerprint instead of by the chunk id and only one
    chunk id exists, so this collapses to one owner.
    """
    quote = "an identical paragraph, quoted by two different people"
    owners = {
        "1111:tweet_text:0:v3": ("item", "1111", "@ada", "https://x.com/ada/status/1111"),
        "2222:tweet_text:0:v3": ("item", "2222", "@grace", "https://x.com/grace/status/2222"),
    }
    loaded = plane(tmp_path, *(chunk(cid, quote, EAST) for cid in owners))

    hits = loaded.search(EAST, limit=10)

    assert [hit.chunk_id for hit in hits] == sorted(owners)
    assert len({hit.row for hit in hits}) == 1, "both chunks must read the same vector row"
    resolved = [owners[hit.chunk_id] for hit in hits]
    assert resolved == [
        ("item", "1111", "@ada", "https://x.com/ada/status/1111"),
        ("item", "2222", "@grace", "https://x.com/grace/status/2222"),
    ]


def test_the_same_text_under_a_different_chunk_id_is_not_a_new_row(tmp_path: Path) -> None:
    """The fingerprint map is what `index update` reads to skip re-embedding (Plan 03 §2.2)."""
    loaded = plane(tmp_path, chunk("a:0:v3", "x", EAST), chunk("b:0:v3", "x", EAST))
    assert loaded.row_of("a:0:v3") == loaded.row_of("b:0:v3") == 0
    assert loaded.chunk_ids_for_row(0) == ("a:0:v3", "b:0:v3")
    assert loaded.row_count == 1
    assert loaded.chunk_count == 2


def test_different_texts_never_share_a_row(tmp_path: Path) -> None:
    loaded = plane(tmp_path, chunk("a:0:v3", "uno", EAST), chunk("b:0:v3", "dos", NORTH))
    assert loaded.row_of("a:0:v3") != loaded.row_of("b:0:v3")
    assert loaded.row_count == 2


# ------------------------------------------------------------------ cosine top-k (14, §2.1)


def test_cosine_similarity_is_the_dot_product_of_unit_rows(tmp_path: Path) -> None:
    """L2-normalized on write ⇒ one `vectors @ q` IS the cosine (Plan 03 §2.1)."""
    loaded = plane(tmp_path, chunk("east:0:v3", "e", EAST), chunk("north:0:v3", "n", NORTH))
    hits = loaded.search(EAST, limit=2)
    assert [hit.chunk_id for hit in hits] == ["east:0:v3", "north:0:v3"]
    assert hits[0].score == pytest.approx(1.0)
    assert hits[1].score == pytest.approx(0.0)


def test_the_ranking_is_by_descending_score(tmp_path: Path) -> None:
    loaded = plane(
        tmp_path,
        chunk("far:0:v3", "f", NORTH),
        chunk("near:0:v3", "n", DIAGONAL),
        chunk("exact:0:v3", "e", EAST),
    )
    hits = loaded.search(EAST, limit=3)
    assert [hit.chunk_id for hit in hits] == ["exact:0:v3", "near:0:v3", "far:0:v3"]
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_a_tie_is_broken_by_chunk_id_not_by_insertion_order(tmp_path: Path) -> None:
    """Step 14. Written in DESCENDING id order so insertion order cannot pass for the rule."""
    loaded = plane(
        tmp_path,
        chunk("zzz:0:v3", "z", EAST),
        chunk("mmm:0:v3", "m", EAST),
        chunk("aaa:0:v3", "a", EAST),
    )
    hits = loaded.search(EAST, limit=3)
    assert [hit.chunk_id for hit in hits] == ["aaa:0:v3", "mmm:0:v3", "zzz:0:v3"]


def test_a_tie_at_the_top_k_boundary_is_broken_by_chunk_id(tmp_path: Path) -> None:
    """The cell `argpartition` alone gets wrong: two rows tie ACROSS the cut.

    `argpartition` returns *some* k largest, and which of two equal scores it keeps is an
    implementation detail of the partition. With `limit=2` and scores (1.0, 0.0, 0.0) the
    answer is fixed by the id, so a partition that happened to keep `zzz` fails here.
    """
    loaded = plane(
        tmp_path,
        chunk("exact:0:v3", "e", EAST),
        chunk("zzz:0:v3", "z", NORTH),
        chunk("aaa:0:v3", "a", NORTH),
    )
    assert [hit.chunk_id for hit in loaded.search(EAST, limit=2)] == ["exact:0:v3", "aaa:0:v3"]


def test_the_ranking_is_the_one_written_down_here(tmp_path: Path) -> None:
    """Spec §8.6 gate 2, pinned as a LITERAL instead of against a second call.

    Comparing `search(q)` to `search(q)` in one process asserts that a deterministic sort is
    deterministic: it cannot fail, and it would stay green through any re-ranking this module
    ever grows. What the gate is about is the ranking being the same TOMORROW, so the expected
    order and the expected scores are written out by hand — including the 0.6 tie, which is
    the only part a partition could reorder.
    """
    loaded = plane(
        tmp_path,
        chunk("west:0:v3", "w", (-1.0, 0.0)),
        chunk("east-b:0:v3", "eb", EAST),
        chunk("north:0:v3", "n", NORTH),
        chunk("east-a:0:v3", "ea", EAST),
        chunk("diag:0:v3", "d", DIAGONAL),
    )
    hits = loaded.search(DIAGONAL, limit=5)
    assert [hit.chunk_id for hit in hits] == [
        "diag:0:v3",
        "north:0:v3",
        "east-a:0:v3",
        "east-b:0:v3",
        "west:0:v3",
    ]
    assert [round(hit.score, 4) for hit in hits] == [1.0, 0.8, 0.6, 0.6, -0.6]


def test_the_limit_caps_the_chunks_returned_not_the_rows(tmp_path: Path) -> None:
    """A shared row expands to every chunk on it, and THEN the limit applies."""
    shared = "compartido"
    loaded = plane(tmp_path, *(chunk(f"{n}:0:v3", shared, EAST) for n in range(5)))
    assert loaded.row_count == 1
    assert len(loaded.search(EAST, limit=3)) == 3
    assert len(loaded.search(EAST, limit=99)) == 5


def test_a_non_positive_limit_is_refused(tmp_path: Path) -> None:
    """An empty result would claim something about the corpus; nothing was asked."""
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    with pytest.raises(ValueError):
        loaded.search(EAST, limit=0)


def test_a_query_of_the_wrong_dimension_is_refused(tmp_path: Path) -> None:
    """Mixing geometries is the failure the manifest's `dimension` exists to stop."""
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        loaded.search((1.0, 0.0, 0.0), limit=1)
    assert "3" in str(excinfo.value) and "2" in str(excinfo.value)


def test_an_empty_plane_answers_nothing_instead_of_raising(tmp_path: Path) -> None:
    """Plan 03 §9: a corpus of 0 chunks is a legal state, not a crash."""
    report = write_vector_plane(tmp_path, SPEC, [])
    assert (report.chunks, report.rows) == (0, 0)
    loaded = load_vector_plane(tmp_path)
    assert loaded.search(EAST, limit=5) == ()
    assert loaded.row_count == 0


# ------------------------------------------------------ filtering happens BEFORE the scoring


def test_a_chunk_outside_the_allowed_set_never_appears(tmp_path: Path) -> None:
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST), chunk("b:0:v3", "b", NORTH))
    hits = loaded.search(EAST, limit=5, allowed_chunk_ids={"b:0:v3"})
    assert [hit.chunk_id for hit in hits] == ["b:0:v3"]


def test_an_allowed_chunk_below_the_global_top_k_is_still_returned(tmp_path: Path) -> None:
    """Filter-then-score, not score-then-filter: the difference is recall, and it is silent.

    `wanted` sits 10th by score. Post-filtering a top-1 would return NOTHING and look like an
    empty corpus.
    """
    chunks = [chunk(f"{n:02d}:0:v3", f"t{n}", EAST) for n in range(10)]
    chunks.append(chunk("wanted:0:v3", "w", NORTH))
    loaded = plane(tmp_path, *chunks)
    hits = loaded.search(EAST, limit=1, allowed_chunk_ids={"wanted:0:v3"})
    assert [hit.chunk_id for hit in hits] == ["wanted:0:v3"]


def test_an_empty_allowed_set_returns_nothing(tmp_path: Path) -> None:
    """`frozenset()` means "no candidate passed the filters", which is not "no filter"."""
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    assert loaded.search(EAST, limit=5, allowed_chunk_ids=frozenset()) == ()


def test_an_unknown_allowed_chunk_id_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    """The lexical plane and the vector plane can disagree; a query is not the place to fail."""
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    hits = loaded.search(EAST, limit=5, allowed_chunk_ids={"a:0:v3", "ghost:0:v3"})
    assert [hit.chunk_id for hit in hits] == ["a:0:v3"]


# --------------------------------------------------------------- what the write refuses (12)


def test_a_vector_of_the_wrong_dimension_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", (1.0, 0.0, 0.0))])
    assert "a:0:v3" in str(excinfo.value)


def test_a_vector_that_is_not_unit_length_is_refused(tmp_path: Path) -> None:
    """The whole top-k rests on cosine == dot product, which rests on this (rule 9)."""
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", (3.0, 4.0))])
    assert "a:0:v3" in str(excinfo.value)


def test_a_non_finite_vector_is_refused(tmp_path: Path) -> None:
    """A NaN poisons every dot product it touches, ranking unpredictably instead of failing."""
    with pytest.raises(VectorPlaneIncompatible):
        write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", (float("nan"), 0.0))])


def test_a_repeated_chunk_id_is_refused(tmp_path: Path) -> None:
    """Two rows claiming one id make `row_of` answer with whichever was written last."""
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        write_vector_plane(
            tmp_path, SPEC, [chunk("a:0:v3", "uno", EAST), chunk("a:0:v3", "dos", NORTH)]
        )
    assert "a:0:v3" in str(excinfo.value)


def test_a_refused_write_leaves_the_previous_plane_intact(tmp_path: Path) -> None:
    """Validation runs BEFORE a byte is written — an interrupted build keeps the old plane."""
    write_vector_plane(tmp_path, SPEC, [chunk("good:0:v3", "g", EAST)])
    with pytest.raises(VectorPlaneIncompatible):
        write_vector_plane(tmp_path, SPEC, [chunk("bad:0:v3", "b", (9.0, 9.0))])
    loaded = load_vector_plane(tmp_path)
    assert loaded.chunk_ids_for_row(0) == ("good:0:v3",)


def test_a_spec_that_declares_unnormalized_vectors_cannot_be_written(tmp_path: Path) -> None:
    """`normalized: false` would make the dot product something other than the cosine."""
    with pytest.raises(VectorPlaneIncompatible):
        write_vector_plane(
            tmp_path,
            VectorSpec("m", 2, normalized=False, query_prefix="", passage_prefix=""),
            [chunk("a:0:v3", "a", EAST)],
        )


# ------------------------------------------------------------- what is actually on disk (2.1)


def test_the_matrix_is_contiguous_float32_in_row_order(tmp_path: Path) -> None:
    """Plan 03 §2.1: a C-order `float32` matrix, mapped with `np.memmap`."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST), chunk("b:0:v3", "b", DIAGONAL)])
    raw = np.fromfile(tmp_path / VECTORS_FILENAME, dtype=np.float32)
    assert raw.shape == (2 * SPEC.dimension,)
    assert raw.reshape(2, SPEC.dimension).tolist() == [
        [pytest.approx(1.0), pytest.approx(0.0)],
        [pytest.approx(0.6), pytest.approx(0.8)],
    ]


def test_the_meta_records_model_dimension_normalization_and_both_prefixes(tmp_path: Path) -> None:
    """Step 11: the prefixes are part of the plane's identity, not of its caller's memory."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    meta = json.loads((tmp_path / VECTORS_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["schema_version"] == META_SCHEMA_VERSION
    assert meta["model"] == "intfloat/multilingual-e5-base"
    assert meta["dimension"] == 2
    assert meta["normalized"] is True
    assert meta["query_prefix"] == "query: "
    assert meta["passage_prefix"] == "passage: "
    assert meta["chunk_rows"] == {"a:0:v3": 0}
    assert meta["text_fingerprint_to_row"] == {text_fingerprint("a"): 0}


def test_two_identical_writes_produce_byte_identical_files(tmp_path: Path) -> None:
    """Deterministic output, so a diff of the index is a diff of the corpus."""
    chunks = [chunk("b:0:v3", "b", NORTH), chunk("a:0:v3", "a", EAST)]
    write_vector_plane(tmp_path, SPEC, chunks)
    first = [(tmp_path / name).read_bytes() for name in (VECTORS_FILENAME, VECTORS_META_FILENAME)]
    write_vector_plane(tmp_path, SPEC, chunks)
    second = [(tmp_path / name).read_bytes() for name in (VECTORS_FILENAME, VECTORS_META_FILENAME)]
    assert first == second


def test_a_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        VECTORS_FILENAME,
        VECTORS_META_FILENAME,
    ]


def test_vector_plane_exists_is_false_until_both_files_are_there(tmp_path: Path) -> None:
    assert vector_plane_exists(tmp_path) is False
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    assert vector_plane_exists(tmp_path) is True
    (tmp_path / VECTORS_META_FILENAME).unlink()
    assert vector_plane_exists(tmp_path) is False


# ------------------------------------------------------------------ what the load refuses (10-12)


def test_a_missing_plane_is_an_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "index build" in str(excinfo.value)


def test_a_changed_model_refuses_the_vector_plane_and_names_the_rebuild(tmp_path: Path) -> None:
    """§13.5, the vector half: another model's geometry is not this index's."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    other = VectorSpec("BAAI/bge-m3", 2, True, "query: ", "passage: ")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path, expected=other)
    message = str(excinfo.value)
    assert "BAAI/bge-m3" in message and "intfloat/multilingual-e5-base" in message
    assert "--embeddings" in message


def test_a_changed_query_prefix_refuses_the_vector_plane(tmp_path: Path) -> None:
    """Step 11. The prefix is a property of the model, and it moves every stored vector."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    other = VectorSpec("intfloat/multilingual-e5-base", 2, True, "", "passage: ")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path, expected=other)
    assert "query_prefix" in str(excinfo.value)


def test_a_changed_passage_prefix_refuses_the_vector_plane(tmp_path: Path) -> None:
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    other = VectorSpec("intfloat/multilingual-e5-base", 2, True, "query: ", "")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path, expected=other)
    assert "passage_prefix" in str(excinfo.value)


def test_a_changed_dimension_refuses_the_vector_plane(tmp_path: Path) -> None:
    """Step 12: a hard error, never one matrix holding two models' geometry."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    other = VectorSpec("intfloat/multilingual-e5-base", 768, True, "query: ", "passage: ")
    with pytest.raises(VectorPlaneIncompatible):
        load_vector_plane(tmp_path, expected=other)


def test_refusing_the_vector_plane_touches_nothing_else_in_the_index(tmp_path: Path) -> None:
    """§13.5: changing model invalidates the VECTOR plane; the lexical one survives.

    Asserted where this module can honestly assert it — it does not delete, truncate or
    rewrite a thing on refusal, so the sibling files of `data/index/` come out untouched.
    """
    lexical = tmp_path / "knowledge.db"
    lexical.write_bytes(b"SQLite format 3\x00 not really, but nobody may touch it")
    before = lexical.read_bytes()
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    with pytest.raises(VectorPlaneIncompatible):
        load_vector_plane(tmp_path, expected=VectorSpec("other", 2, True, "query: ", "passage: "))
    assert lexical.read_bytes() == before
    assert lexical.exists()


def test_a_truncated_matrix_is_refused_instead_of_read_short(tmp_path: Path) -> None:
    """Plan 03 §9: an interrupted write leaves «vectores incompletos», not a shorter corpus."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST), chunk("b:0:v3", "b", NORTH)])
    matrix = tmp_path / VECTORS_FILENAME
    matrix.write_bytes(matrix.read_bytes()[:-4])
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "vectors.f32" in str(excinfo.value)


def test_a_meta_declaring_unnormalized_vectors_is_refused(tmp_path: Path) -> None:
    """A plane written by something that did not normalize ranks by dot product, not cosine."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    path = tmp_path / VECTORS_META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["normalized"] = False
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "normalized" in str(excinfo.value)


def test_a_meta_of_an_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    path = tmp_path / VECTORS_META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["schema_version"] = "99"
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "99" in str(excinfo.value)


def test_a_meta_missing_a_field_is_refused(tmp_path: Path) -> None:
    """Total: a field this code needs and the document does not carry."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    path = tmp_path / VECTORS_META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    del meta["passage_prefix"]
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "passage_prefix" in str(excinfo.value)


def test_a_meta_declaring_a_field_this_code_does_not_know_is_refused(tmp_path: Path) -> None:
    """Closed: a document written by a writer watching a plane this code cannot see."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    path = tmp_path / VECTORS_META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["quantization"] = "int8"
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "quantization" in str(excinfo.value)


def test_a_meta_that_is_not_a_json_object_is_refused(tmp_path: Path) -> None:
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    (tmp_path / VECTORS_META_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(VectorPlaneIncompatible):
        load_vector_plane(tmp_path)


def test_a_meta_pointing_a_chunk_at_a_row_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    """A row index out of range would read another chunk's geometry, or crash mid-query."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    path = tmp_path / VECTORS_META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["chunk_rows"]["a:0:v3"] = 7
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(VectorPlaneIncompatible):
        load_vector_plane(tmp_path)


def edited_meta(tmp_path: Path, **changes: object) -> None:
    """Rewrite the meta on disk with `changes` applied — a hand-edit, or another writer."""
    path = tmp_path / VECTORS_META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update(changes)
    path.write_text(json.dumps(meta), encoding="utf-8")


def test_a_meta_that_is_not_valid_json_is_refused(tmp_path: Path) -> None:
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    (tmp_path / VECTORS_META_FILENAME).write_text("{ no cierro", encoding="utf-8")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert VECTORS_META_FILENAME in str(excinfo.value)


def test_a_meta_whose_row_count_its_fingerprints_do_not_cover_is_refused(tmp_path: Path) -> None:
    """`rows` is what the size check compares against, so it may not be taken on trust."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    edited_meta(tmp_path, rows=2)
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "2" in str(excinfo.value)


def test_a_dimension_that_is_not_an_integer_is_refused(tmp_path: Path) -> None:
    """`int("2")` would take it, and the matrix would disagree with the meta in silence."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    edited_meta(tmp_path, dimension="2")
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "dimension" in str(excinfo.value)


def test_a_chunk_rows_block_that_is_not_a_map_of_text_to_integer_is_refused(
    tmp_path: Path,
) -> None:
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    edited_meta(tmp_path, chunk_rows={"a:0:v3": "0"})
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "chunk_rows" in str(excinfo.value)


def test_a_matrix_that_vanishes_after_the_meta_is_read_is_refused(tmp_path: Path) -> None:
    """The window `vector_plane_exists` cannot close: the file was there, and then was not.

    Exercised on the mapping step directly, because reaching it through `load_vector_plane`
    would mean deleting the file between two statements of the same call.
    """
    from xbrain.knowledge import vector_index

    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        vector_index._mapped_matrix(tmp_path / VECTORS_FILENAME, 1, 2, "0" * 64)
    assert VECTORS_FILENAME in str(excinfo.value)


def test_a_matching_spec_loads(tmp_path: Path) -> None:
    """The other half of the refusals: the expected spec is a gate, not a wall."""
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    loaded = load_vector_plane(tmp_path, expected=SPEC)
    assert loaded.spec == SPEC
    loaded.close()


# ------------------------------------------------------- the four blockers of the review round


class _TornNumpy:
    """A `numpy` stand-in whose `tofile` writes some bytes and then dies.

    An interrupted rebuild is otherwise unreachable from a test: the failure has to land in
    the middle of the write, not before it (nothing is written) and not after it (it
    succeeded).
    """

    float32 = np.float32

    def __init__(self, prefix: bytes) -> None:
        self._prefix = prefix

    def asarray(self, values, dtype=None):  # noqa: ANN001 - a stand-in, not an API
        return self

    def reshape(self, *shape):  # noqa: ANN002
        return self

    def tofile(self, path) -> None:  # noqa: ANN001
        Path(path).write_bytes(self._prefix)
        raise OSError("no space left on device")


def test_an_interrupted_rebuild_does_not_corrupt_the_previous_matrix(
    tmp_path: Path, monkeypatch
) -> None:
    """Blocker 1a: the matrix must not be written in place.

    `tofile` opens its target for truncation, so rebuilding a plane of the SAME shape over
    itself and dying halfway leaves a file of exactly the right size holding a mix of the new
    head and the old tail — which every size check in the world calls healthy, and which pairs
    each chunk with somebody else's geometry.
    """
    from xbrain.knowledge import vector_index

    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST), chunk("b:0:v3", "b", NORTH)])
    before = (tmp_path / VECTORS_FILENAME).read_bytes()

    monkeypatch.setattr(vector_index, "_numpy", lambda: _TornNumpy(b"\x00" * 8))
    with pytest.raises(OSError):
        write_vector_plane(
            tmp_path, SPEC, [chunk("a:0:v3", "x", NORTH), chunk("b:0:v3", "y", EAST)]
        )

    assert (tmp_path / VECTORS_FILENAME).read_bytes() == before


def test_a_same_size_rewrite_of_the_matrix_is_refused(tmp_path: Path) -> None:
    """Blocker 1b: the size check cannot see a replacement of the right length.

    Same bytes-on-disk, different numbers: every chunk id still resolves, and each one now
    reads another text's geometry. The meta records a digest of the matrix precisely because
    «the file is the right size» is not «the file is the right file».
    """
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST), chunk("b:0:v3", "b", DIAGONAL)])
    matrix = tmp_path / VECTORS_FILENAME
    # Reversed, NOT mirrored: `[1, 0, 0, 1]` reversed is `[1, 0, 0, 1]`, so a plane built from
    # EAST and NORTH would have the same digest after the swap and this test would pass
    # against a module that checks nothing at all.
    swapped = np.fromfile(matrix, dtype=np.float32)[::-1].copy()
    assert swapped.tolist() != np.fromfile(matrix, dtype=np.float32).tolist()
    swapped.tofile(matrix)
    assert matrix.stat().st_size == 2 * SPEC.dimension * 4

    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert VECTORS_FILENAME in str(excinfo.value)


def test_an_orphan_row_does_not_consume_a_top_k_slot(tmp_path: Path) -> None:
    """Blocker 2: a row nobody points at must not hide a result that does have an owner.

    Plan 03 §2.3 makes orphan rows a DESIGNED state — a deleted chunk leaves its row behind
    until the next `build --force` compacts it. Taking the best `limit` ROWS then expanding
    them onto chunks spends the slot on a row that expands to nothing, and the caller gets
    fewer results than exist, with nothing saying so.
    """
    write_vector_plane(
        tmp_path, SPEC, [chunk("orphan:0:v3", "huérfano", EAST), chunk("kept:0:v3", "vivo", NORTH)]
    )
    path = tmp_path / VECTORS_META_FILENAME
    meta = json.loads(path.read_text(encoding="utf-8"))
    orphan_row = meta["chunk_rows"].pop("orphan:0:v3")
    path.write_text(json.dumps(meta), encoding="utf-8")

    loaded = load_vector_plane(tmp_path)
    assert loaded.chunk_ids_for_row(orphan_row) == ()

    hits = loaded.search(EAST, limit=1)
    assert [hit.chunk_id for hit in hits] == ["kept:0:v3"]


def test_a_query_vector_that_is_not_unit_length_is_refused(tmp_path: Path) -> None:
    """Blocker 3: the scores this module returns are declared to be cosines.

    A query of norm 5 scales every one of them by five. The ORDER survives, so nothing looks
    wrong — and then 03.5 fuses those numbers with a lexical channel, where a score that is
    not on the scale it claims is a silent re-weighting of the whole result set.
    """
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        loaded.search((3.0, 4.0), limit=1)
    assert "norma" in str(excinfo.value)


def test_a_non_finite_query_vector_is_refused(tmp_path: Path) -> None:
    """Blocker 3: a NaN makes every comparison false, so the top-k comes back EMPTY.

    Which reads exactly like «the corpus has nothing for you», and is the most expensive way
    for a retrieval layer to be wrong.
    """
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    with pytest.raises(VectorPlaneIncompatible):
        loaded.search((float("nan"), 0.0), limit=1)


def test_a_meta_that_is_not_utf8_is_refused_without_leaking_the_decode_error(
    tmp_path: Path,
) -> None:
    """Blocker 4: `UnicodeDecodeError` is a `ValueError`, so neither `except` caught it.

    It came out of the CLI as a raw traceback — the one thing spec §9.3 asks this layer never
    to do — and its message carries a slice of the offending bytes, which in a file that
    indexes a personal corpus is not something to print by accident.
    """
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    (tmp_path / VECTORS_META_FILENAME).write_bytes(b"\xff\xfe{not utf-8}")

    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    message = str(excinfo.value)
    assert VECTORS_META_FILENAME in message and "UTF-8" in message
    assert "0xff" not in message and "byte" not in message


def test_a_row_outside_the_matrix_is_refused_by_chunk_ids_for_row(tmp_path: Path) -> None:
    """Blocker 5: `-1` is a VALID index in Python, and it answers with the LAST row.

    So a caller that computed a row wrong does not crash — it receives some other text's
    chunk ids and attributes them to a fragment nobody retrieved.
    """
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST), chunk("b:0:v3", "b", NORTH))
    with pytest.raises(VectorPlaneIncompatible):
        loaded.chunk_ids_for_row(-1)
    with pytest.raises(VectorPlaneIncompatible):
        loaded.chunk_ids_for_row(2)


def test_searching_a_closed_plane_raises_instead_of_answering_nothing(tmp_path: Path) -> None:
    """Blocker 6: `()` from a closed plane is indistinguishable from an empty corpus.

    One is a bug in the caller and the other is a fact about the data, and a retrieval layer
    that reports them the same way makes the first one invisible.
    """
    loaded = plane(tmp_path, chunk("a:0:v3", "a", EAST))
    assert loaded.search(EAST, limit=1) != ()
    loaded.close()
    with pytest.raises(ValueError):
        loaded.search(EAST, limit=1)


def test_a_spec_of_dimension_zero_cannot_be_written(tmp_path: Path) -> None:
    """Blocker 2: dimension 0 makes the matrix 0 bytes and every cosine undefined."""
    with pytest.raises(VectorPlaneIncompatible):
        write_vector_plane(tmp_path, VectorSpec("m", 0, True, "", ""), [chunk("a:0:v3", "a", ())])


def test_a_meta_declaring_dimension_zero_is_refused(tmp_path: Path) -> None:
    """Blocker 2, the read half: a 0-dimension meta makes the size check vacuously true.

    `rows * 0 * 4 == 0` for ANY number of rows, so the one guard standing between a truncated
    matrix and a short corpus stops being able to fail.
    """
    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    edited_meta(tmp_path, dimension=0)
    with pytest.raises(VectorPlaneIncompatible) as excinfo:
        load_vector_plane(tmp_path)
    assert "dimension" in str(excinfo.value)


# ------------------------------- verification and mapping must see the SAME file (race)


class _SwapOnMemmap:
    """Real `numpy`, except that the first `memmap` is preceded by a rebuild landing.

    The race needs the swap to happen in ONE exact window — after the digest has been
    checked and before the bytes are mapped — and a thread cannot be aimed at a window that
    narrow without being flaky. Driving it from inside `memmap` puts it there every time.
    """

    def __init__(self, matrix: Path, replacement: Path) -> None:
        self._matrix = matrix
        self._replacement = replacement
        self.swapped = False

    def __getattr__(self, name: str):  # noqa: ANN204 - everything else is real numpy
        return getattr(np, name)

    def memmap(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        if not self.swapped:
            self.swapped = True
            os.replace(self._replacement, self._matrix)
        return np.memmap(*args, **kwargs)


def test_a_rebuild_landing_mid_load_cannot_serve_bytes_nobody_verified(
    tmp_path: Path, monkeypatch
) -> None:
    """The matrix is hashed and then mapped, and `os.replace` swaps the NAME, not the file.

    `write_vector_plane` finishes with exactly that call, so a build running while a query
    opens the index is not hypothetical — it is this module's own writer. Resolving the path
    once to hash it and again to map it makes those two lookups answer with two different
    inodes: the digest passes over the old matrix and the query is served the new one, at the
    old meta's `chunk_rows`. Every id still resolves, and every one of them reads a vector
    belonging to another text.

    Same size and same digest length on both sides, so nothing but the identity of the file
    distinguishes them.
    """
    from xbrain.knowledge import vector_index

    write_vector_plane(tmp_path, SPEC, [chunk("a:0:v3", "a", EAST)])
    matrix = tmp_path / VECTORS_FILENAME
    landing = tmp_path / "rebuilt.f32"
    np.asarray([NORTH], dtype=np.float32).tofile(landing)
    assert landing.stat().st_size == matrix.stat().st_size

    swapper = _SwapOnMemmap(matrix, landing)
    monkeypatch.setattr(vector_index, "_numpy", lambda: swapper)

    loaded = load_vector_plane(tmp_path)
    assert swapper.swapped, "the probe never reached the window it exists to test"

    # EAST is what was verified; NORTH is what landed. A cosine of 1.0 says the plane served
    # the bytes it checked, and 0.0 says it served the ones it never looked at.
    hits = loaded.search(EAST, limit=1)
    assert [hit.chunk_id for hit in hits] == ["a:0:v3"]
    assert hits[0].score == pytest.approx(1.0)
