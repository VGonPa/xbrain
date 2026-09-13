"""The vector plane: a float32 matrix on disk, deduplicated by text, searched by cosine.

Plan 03 §2. The lexical plane answers with the words a chunk contains; this one answers with
the geometry of what it means, and the two are fused in 03.5. What lives here is ONLY the
storage and the arithmetic — no embedder is invoked (`xbrain.embeddings` owns that), no
manifest is written (`knowledge.index_build` owns that), and no query is planned
(`knowledge.search_service` owns that).

**The matrix is `data/index/vectors.f32`**: `rows x dimension` float32 values, C-order, with
nothing else in the file. Its companion `data/index/vectors.meta.json` says what the numbers
mean — the model, the dimension, whether the rows are normalized, both prefixes, and the two
maps that turn a row into chunks and a text into a row. The bytes are DERIVED: deleting both
files costs one `xbrain index build --embeddings`, which is what licenses every refusal below.

**Cosine similarity is the dot product, because the rows are unit vectors.** That is the whole
reason `search` is one `matrix @ query`, and it is VERIFIED on write rather than assumed:
`embeddings.embed_texts` already normalizes, but this writer takes a plain sequence of floats
from any caller, and a row of norm 5 would multiply its own similarity by five against every
query, silently and forever. Checking costs one pass over data already in memory (CLAUDE.md
rule 9: assert on the source, never on the reported conclusion).

**Dedupe shares the VECTOR, never the ASSOCIATIONS** (spec §5.6, Plan 03 §2.2, criterion
§13.6). Two chunks whose text is byte-identical embed to the same numbers, so they share one
row keyed by `sha256(text)`. Each keeps its own `chunk_id`, and therefore its own surface,
owner, author and URL. The map that makes this work runs `chunk_id -> row`, MANY-TO-ONE — not
`row -> chunk_id`, which the plan's first sketch wrote and which cannot express the criterion
it belongs to: a row that names one chunk leaves the second chunk with no id in the plane, so
no search returns it and nothing can reach its owner. The vectors are identical either way,
which is exactly why the loss would be invisible.

**This plane stores no owner, author, URL or text.** They have one home, the lexical plane's
`chunks` table, and a second copy is the divergence rule 5 exists to stop: a re-chunked corpus
would leave two descriptions of the same fragment, both internally consistent. A hit here
carries the `chunk_id` and nothing else the index already knows how to resolve.

**`numpy` arrives through the `[embeddings]` extra, and the import is DEFERRED** (Plan 03
§2.1, m11). `import xbrain` must keep working for someone who only runs the CLI, so the import
lives inside `_numpy()` and its absence raises `VectorBackendUnavailable` naming
`uv pip install 'xbrain[embeddings]'` — never a raw `ImportError` out of a command.

**Two error families, because they are two situations.** `VectorBackendUnavailable` is an
ENVIRONMENT problem and a sibling of `embeddings.EmbeddingError` (a `RuntimeError`, which the
CLI turns into a clean exit-1). `VectorPlaneIncompatible` is an INDEX problem and a sibling of
`index_schema.IndexIncompatibleError` — but deliberately NOT that class, because its advice is
*rebuild the index* and spec §5.5 says changing the embedding model invalidates the vector
part of the index and NOT the store or the lexical plane. The advice here names
`--embeddings`, and a refusal deletes, truncates and rewrites nothing.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, BinaryIO

from xbrain.knowledge.index_schema import IndexError_

if TYPE_CHECKING:  # pragma: no cover - the extra is optional, the annotations are not
    # The ONE place `numpy` is named at module level, and it costs nothing at runtime: a
    # `TYPE_CHECKING` block never executes, so `import xbrain` still works without the
    # `[embeddings]` extra while mypy still types the boundary between the matrix and the
    # plain floats this module hands back (Plan 03 §12: nothing `Any` reaching a model).
    import numpy as _np
    from numpy.typing import NDArray

    Matrix = NDArray[_np.float32]

# The two files of the plane. Both, or neither: a matrix without its meta is a block of
# numbers nobody can interpret, and a meta without its matrix describes nothing.
VECTORS_FILENAME = "vectors.f32"
VECTORS_META_FILENAME = "vectors.meta.json"

# The meta document's version, bumped when its SHAPE changes. It is not the same thing as the
# model or the chunker version: those invalidate the numbers, this invalidates the reader.
META_SCHEMA_VERSION = "1"

# Every field the document must carry, and the only ones it may. Read BOTH ways on load
# (`_closure_gap`), because a meta written by a plane this code cannot see — a quantized one,
# say — is not a compatible document just because it happens to carry the keys we ask for.
_META_FIELDS = frozenset(
    {
        "schema_version",
        "model",
        "dimension",
        "normalized",
        "query_prefix",
        "passage_prefix",
        "rows",
        "matrix_sha256",
        "chunk_rows",
        "text_fingerprint_to_row",
    }
)

# The sentence every refusal of this plane ends with. ONE string, so the loader, the writer
# and the tests name the same command. It is NOT `index_schema.REBUILD_ADVICE`, which omits
# `--embeddings` and would seal an index with no plane at all.
#
# IT SAYS WHAT THE COMMAND DOES, NOT WHAT spec §5.5 WANTS IT TO DO. §5.5 promises that a model
# change costs the vector plane alone; there is no vector-only rebuild, and `--embeddings` is
# wired to the FULL build, which unlinks the lexical base first and re-derives it. An advice
# promising «el plano léxico no se toca» sent the operator into that window blind: if the
# embedder then fails, no index answers — not even lexically — until `xbrain index build`.
# A vector-only rebuild that preserves the lexical plane is a declared follow-up (PR #185).
VECTOR_REBUILD_ADVICE = (
    "Reconstruye el índice con su plano vectorial: `xbrain index build --embeddings --force`. "
    "Reconstruye también el plano léxico desde el store (no hay reconstrucción solo "
    "vectorial), y si el embedder falla a mitad no queda índice que responder —ni léxico— "
    "hasta un `xbrain index build`."
)

# How far a row's L2 norm may sit from 1.0 before the writer calls it unnormalized. A unit
# float64 vector stored as float32 comes back a few ulps off, and over a 768-dimensional row
# those accumulate; `embeddings.py` keeps its own tolerance for a different boundary (a
# backend printing floats as decimal text), so these are two numbers with two reasons rather
# than one constant copied twice.
_UNIT_TOLERANCE = 1e-6

# A float32 value is four bytes. Named because it appears in the size check, where a bare 4
# reads as a magic number and the check is the only thing standing between an interrupted
# write and a corpus silently read short.
_FLOAT32_BYTES = 4


class VectorBackendUnavailable(RuntimeError):
    """`numpy` is not installed — the `[embeddings]` extra was never synced.

    A `RuntimeError` like `embeddings.EmbeddingError`, and for the same reason: it is an
    environment problem the operator fixes with one install command, and the CLI already
    turns this family into a clean exit-1 instead of a traceback.
    """


class VectorPlaneIncompatible(IndexError_):
    """What is on disk is not what this code can read, or what the caller asked for.

    Under `IndexError_` so the CLI's index door handles it, and NOT under
    `IndexIncompatibleError` so nobody reads it as *rebuild the whole index*: a model, a
    dimension or a prefix change costs the vector plane and leaves the lexical one standing
    (spec §5.5). Nothing is deleted, truncated or rewritten when this is raised.
    """


@dataclass(frozen=True)
class VectorSpec:
    """What produced the numbers — and therefore what invalidates them (Plan 03 §2.3).

    All five fields, not just the model. `dimension` because two models' geometry must never
    share a matrix; `normalized` because the dot product is the cosine only for unit rows; and
    BOTH prefixes because the E5 and BGE families embed `"query: …"` and `"passage: …"` into
    different regions of the same space — change one and every stored row is an answer to a
    question nobody asked any more, while staying well-formed and unit-length.

    This is the value `index_build` records in the manifest's `embeddings` slot (03.4) and the
    value `search` checks a query against before scoring it.
    """

    model: str
    dimension: int
    normalized: bool
    query_prefix: str
    passage_prefix: str


@dataclass(frozen=True)
class ChunkVector:
    """One chunk's embedding, as the builder hands it over.

    `text` is here for its FINGERPRINT, not to be stored: it is the dedupe key, and the plane
    keeps the hash rather than the prose (the corpus is personal, and a second copy of it in
    `data/index/` is a second thing to keep in step with the store).
    """

    chunk_id: str
    text: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class VectorHit:
    """One ranked chunk: which chunk, which row it read, and the cosine it scored.

    `row` travels with the hit because a shared row is the one thing about this plane a
    consumer cannot re-derive — two hits on one row are two chunks quoting the same text, and
    that is worth being able to see (rule 7).
    """

    chunk_id: str
    row: int
    score: float


@dataclass(frozen=True)
class VectorWriteReport:
    """What a write did, in the three numbers that differ when the dedupe works.

    `chunks - rows == shared_rows`, always: it is reported rather than derived so the caller's
    log says how much of the corpus was duplicate prose without recomputing anything.
    """

    chunks: int
    rows: int
    shared_rows: int


def text_fingerprint(text: str) -> str:
    """`sha256(text)` — the dedupe key of Plan 03 §2.2.

    The TEXT alone, not the chunk's own `fingerprint`: that one hashes the chunker version,
    the owner, the position and the provenance beside the prose precisely so two identical
    paragraphs under different owners stay distinguishable — which is the opposite of what is
    wanted here, where they must collapse onto one embedding.

    The passage prefix is not part of the key either. It is plane-wide, it is recorded in
    `VectorSpec`, and changing it invalidates every row at once, so folding it in would add a
    constant to every hash and distinguish nothing.
    """
    return sha256(text.encode("utf-8")).hexdigest()


def vector_plane_exists(index_dir: Path) -> bool:
    """Whether both files of the plane are present. Neither one alone is a plane."""
    return (index_dir / VECTORS_FILENAME).is_file() and (
        index_dir / VECTORS_META_FILENAME
    ).is_file()


def _numpy() -> ModuleType:
    """The `numpy` module, or an error that names the install command (Plan 03 §2.1).

    THE IMPORT IS HERE AND NOT AT MODULE LEVEL, and that placement is the whole feature:
    `numpy` ships in the optional `[embeddings]` extra, so importing it at the top of this
    file would make `import xbrain` fail for everyone who only runs the CLI. A test reads this
    module's source and asserts no top-level import of it.
    """
    try:
        import numpy
    except ImportError as exc:
        raise VectorBackendUnavailable(
            "el plano vectorial necesita `numpy`, que viaja en el extra opcional "
            "`[embeddings]`: instálalo con `uv pip install 'xbrain[embeddings]'` "
            f"(o `uv sync --extra embeddings`) y repite el comando ({exc})"
        ) from exc
    return numpy


def _validated_unit_vector(
    vector: Sequence[float], dimension: int, subject: str
) -> tuple[float, ...]:
    """A finite unit vector of the plane's dimension, or a refusal naming `subject`.

    ONE function for BOTH sides of the arithmetic — the rows going in and the query coming
    against them — because the property is one property: `matrix @ q` is the cosine only when
    both operands are unit vectors, and checking the stored half alone leaves a query of norm
    5 multiplying every similarity by five. The order survives, so nothing looks wrong; 03.5
    then fuses those numbers with a lexical channel, where a score off its declared scale is a
    silent re-weighting of the whole result set. Two checks in two places would be two
    definitions of one invariant, which is the drift rule 5 exists to stop.

    A non-finite value is refused rather than stored or scored: `NaN` makes every comparison
    against it false, so it does not rank badly — it empties the top-k, which reads exactly
    like a corpus with nothing in it.
    """
    if len(vector) != dimension:
        raise VectorPlaneIncompatible(
            f"{subject} trae un vector de dimensión {len(vector)} y este plano guarda "
            f"vectores de dimensión {dimension}: no se pueden comparar"
        )
    if not all(math.isfinite(value) for value in vector):
        raise VectorPlaneIncompatible(
            f"el vector de {subject} tiene algún valor no finito (NaN o inf): envenenaría "
            "todos los productos escalares que tocase en vez de fallar"
        )
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if abs(norm - 1.0) > _UNIT_TOLERANCE:
        raise VectorPlaneIncompatible(
            f"el vector de {subject} tiene norma {norm:.6f} y no 1.0: el coseno de este "
            "plano es el producto escalar, y sólo lo es para vectores unitarios"
        )
    return tuple(vector)


def _assign_rows(
    entries: Sequence[ChunkVector], dimension: int
) -> tuple[list[tuple[float, ...]], dict[str, int], dict[str, int]]:
    """Fold the chunks onto deduplicated rows: `(matrix, chunk_rows, fingerprint_rows)`.

    Which of two identical texts' vectors ends up stored cannot matter — if they differ, the
    backend contradicted itself about the same input, and that is a property of the backend
    this module cannot adjudicate — so the FIRST is kept and the rest join its row.

    A repeated `chunk_id` is refused instead: unlike a repeated text it is not a fact about
    the corpus, it is two rows claiming one identity, and `chunk_rows` would answer with
    whichever was written last.
    """
    matrix: list[tuple[float, ...]] = []
    chunk_rows: dict[str, int] = {}
    fingerprint_rows: dict[str, int] = {}
    for entry in entries:
        if entry.chunk_id in chunk_rows:
            raise VectorPlaneIncompatible(
                f"el chunk {entry.chunk_id} aparece dos veces en el mismo lote: un id "
                "repetido haría que el plano respondiese con la fila escrita en último lugar"
            )
        vector = _validated_unit_vector(entry.vector, dimension, f"el chunk {entry.chunk_id}")
        fingerprint = text_fingerprint(entry.text)
        row = fingerprint_rows.get(fingerprint)
        if row is None:
            row = len(matrix)
            fingerprint_rows[fingerprint] = row
            matrix.append(vector)
        chunk_rows[entry.chunk_id] = row
    return matrix, chunk_rows, fingerprint_rows


def _sha256_stream(handle: BinaryIO) -> str:
    """The digest of what is left in `handle`, a megabyte at a time.

    It takes the OPEN FILE and not a path, which is the whole of the race fix below: a digest
    computed by reopening the name proves something about whatever answered that name at that
    moment, and nothing about the bytes anybody later maps.
    """
    digest = sha256()
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """The digest of the file at `path`, for the write side, where nothing is racing yet.

    The matrix is hashed while it still lives under its temporary name, before the rename
    that publishes it, so no other process can be holding or replacing it.
    """
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _write_matrix(path: Path, matrix: Sequence[tuple[float, ...]], dimension: int) -> str:
    """The float32 bytes, C-order, nothing else in the file — and never written IN PLACE.

    `tofile` opens its target for truncation and writes forward, so a rebuild of a plane with
    the SAME shape that dies halfway would leave a file of exactly the right length holding
    the new head and the old tail. Every size check calls that healthy, and every chunk on the
    surviving half now reads another text's geometry. It goes to a temporary name and is
    renamed over the top, which is atomic within a filesystem.

    Written with `tofile` rather than through a `w+` memmap: mapping is what makes READING
    25k x 768 floats cheap, and on the write side it would only add a mapping to tear down.
    An empty corpus writes an empty file — a legal state (Plan 03 §9), and one `np.memmap`
    cannot represent, which is why the reader special-cases it too.

    Returns the digest of what was written, which the meta records.
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        if matrix:
            numpy = _numpy()
            array = numpy.asarray(matrix, dtype=numpy.float32).reshape(len(matrix), dimension)
            array.tofile(temporary)
        else:
            temporary.write_bytes(b"")
        digest = _sha256_file(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return digest


def _meta_document(
    spec: VectorSpec,
    rows: int,
    matrix_sha256: str,
    chunk_rows: Mapping[str, int],
    fingerprints: Mapping[str, int],
) -> dict[str, object]:
    """The meta as it is written — one JSON object, sorted, so two equal writes are equal."""
    return {
        "schema_version": META_SCHEMA_VERSION,
        "model": spec.model,
        "dimension": spec.dimension,
        "normalized": spec.normalized,
        "query_prefix": spec.query_prefix,
        "passage_prefix": spec.passage_prefix,
        "rows": rows,
        "matrix_sha256": matrix_sha256,
        "chunk_rows": dict(sorted(chunk_rows.items())),
        "text_fingerprint_to_row": dict(sorted(fingerprints.items())),
    }


def _replace_atomically(destination: Path, payload: bytes) -> None:
    """Write `payload` beside `destination` and rename it over the top.

    A half-written matrix that still carried its old meta would be read as a complete corpus
    of the wrong size; `os.replace` is atomic within a filesystem, so a reader sees the old
    file or the new one. The window that remains is BETWEEN the two files — a crash after the
    matrix lands and before the meta does — and it is exactly what the size check on load
    catches, which is why that check is not optional.
    """
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def write_vector_plane(
    index_dir: Path, spec: VectorSpec, chunk_vectors: Iterable[ChunkVector]
) -> VectorWriteReport:
    """Write the whole plane: deduplicated matrix first, then the meta that explains it.

    EVERYTHING IS VALIDATED BEFORE A BYTE IS WRITTEN. A build that dies on chunk 9,000 of
    23,000 must leave the previous plane readable rather than a prefix of the new one wearing
    the old plane's meta (Plan 03 §9: *transacción revertida; manifest sin actualizar*).

    The matrix is written before the meta, for the same reason `index_build` writes the
    manifest last: the document that says «this plane is complete» is the last thing to land.
    """
    if spec.dimension < 1:
        raise VectorPlaneIncompatible(
            f"un plano vectorial de dimensión {spec.dimension} no tiene geometría: el coseno "
            "no está definido y su matriz ocuparía cero octetos con cualquier número de filas"
        )
    if not spec.normalized:
        raise VectorPlaneIncompatible(
            "este plano declara `normalized: false`, y su top-k es un producto escalar: "
            "guardar vectores sin normalizar haría que la similitud dejase de ser el coseno "
            "sin que nada lo dijese"
        )
    entries = tuple(chunk_vectors)
    matrix, chunk_rows, fingerprints = _assign_rows(entries, spec.dimension)
    index_dir.mkdir(parents=True, exist_ok=True)
    digest = _write_matrix(index_dir / VECTORS_FILENAME, matrix, spec.dimension)
    document = _meta_document(spec, len(matrix), digest, chunk_rows, fingerprints)
    _replace_atomically(
        index_dir / VECTORS_META_FILENAME,
        (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )
    return VectorWriteReport(
        chunks=len(entries), rows=len(matrix), shared_rows=len(entries) - len(matrix)
    )


def _closure_gap(present: set[str], expected: frozenset[str]) -> tuple[set[str], set[str]]:
    """`(missing, unknown)` — the document read in BOTH directions.

    The same shape `index_build.Manifest.from_dict` uses, and for the same reason: refusing
    only the missing half lets a document written by a newer writer load as compatible, with
    whatever it declares that this code cannot honour silently dropped.
    """
    return set(expected) - present, present - set(expected)


def _meta_mapping(path: Path) -> Mapping[str, object]:
    """The meta file as a JSON object, refusing anything else before a field is read."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        # `UnicodeDecodeError` is a `ValueError`, so neither of the two clauses below ever
        # saw it and it left this layer as a raw traceback (spec §9.3 asks for the opposite).
        # Its own message quotes the offending input, and this file indexes a personal
        # corpus, so what is relayed is the file and the codec — never the content.
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} no está codificado en UTF-8 y no se puede leer "
            f"(no se reproduce su contenido). {VECTOR_REBUILD_ADVICE}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorPlaneIncompatible(
            f"no se puede leer {VECTORS_META_FILENAME} ({exc}). {VECTOR_REBUILD_ADVICE}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} no es un objeto JSON, es {type(raw).__name__}. "
            f"{VECTOR_REBUILD_ADVICE}"
        )
    missing, unknown = _closure_gap(set(raw), _META_FIELDS)
    if missing:
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} no declara {sorted(missing)}. {VECTOR_REBUILD_ADVICE}"
        )
    if unknown:
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} declara {sorted(unknown)}, que este código no conoce. "
            f"{VECTOR_REBUILD_ADVICE}"
        )
    if str(raw["schema_version"]) != META_SCHEMA_VERSION:
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} habla la versión {raw['schema_version']!r} y este "
            f"código habla la {META_SCHEMA_VERSION!r}. {VECTOR_REBUILD_ADVICE}"
        )
    return raw


def _meta_spec(raw: Mapping[str, object]) -> VectorSpec:
    """The stored spec, with the one claim this module cannot work under refused.

    `normalized: false` is not a configuration, it is a plane whose dot product is not a
    cosine. Reading it and ranking anyway would report similarities scaled by each row's own
    norm — a number in the right range, in the right field, meaning something else.
    """
    if raw["normalized"] is not True:
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} declara `normalized: {raw['normalized']!r}`, y este "
            "plano sólo puede buscar por coseno sobre vectores unitarios. "
            f"{VECTOR_REBUILD_ADVICE}"
        )
    return VectorSpec(
        model=str(raw["model"]),
        dimension=_bounded_int(raw["dimension"], "dimension", 1),
        normalized=True,
        query_prefix=str(raw["query_prefix"]),
        passage_prefix=str(raw["passage_prefix"]),
    )


def _require_same_spec(stored: VectorSpec, expected: VectorSpec) -> None:
    """Refuse a plane built under another model, dimension or prefix (Plan 03 §2.3).

    THE FIELD IS NAMED, not just the fact of the mismatch: «el plano no coincide» sends the
    operator to re-read a config they already believe is right, while «query_prefix» tells
    them which line moved. Every one of these is a total invalidation of the vector plane and
    of nothing else — the lexical plane and the store are untouched (spec §5.5).
    """
    for name in ("model", "dimension", "query_prefix", "passage_prefix"):
        mine, theirs = getattr(stored, name), getattr(expected, name)
        if mine != theirs:
            raise VectorPlaneIncompatible(
                f"el plano vectorial se construyó con {name}={mine!r} y se ha pedido "
                f"{name}={theirs!r}: un cambio de modelo, dimensión o prefijo invalida "
                f"todos los vectores guardados. {VECTOR_REBUILD_ADVICE}"
            )


def _checked_row_maps(
    raw: Mapping[str, object],
) -> tuple[int, dict[str, int], tuple[tuple[str, ...], ...], dict[str, int]]:
    """`(rows, chunk_rows, row_chunks, text_rows)`, with every index proved to exist.

    An out-of-range row is the failure this catches: it either reads another chunk's geometry
    or dies inside a query, and both happen long after whatever wrote the document.
    """
    rows = _bounded_int(raw["rows"], "rows", 0)
    fingerprints = _int_map(raw["text_fingerprint_to_row"], "text_fingerprint_to_row")
    chunk_rows = _int_map(raw["chunk_rows"], "chunk_rows")
    if sorted(fingerprints.values()) != list(range(rows)):
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} declara {rows} filas que sus huellas de texto no "
            f"cubren exactamente una vez. {VECTOR_REBUILD_ADVICE}"
        )
    grouped: list[list[str]] = [[] for _ in range(rows)]
    for chunk_id, row in sorted(chunk_rows.items()):
        if not 0 <= row < rows:
            raise VectorPlaneIncompatible(
                f"{VECTORS_META_FILENAME} apunta el chunk {chunk_id} a la fila {row}, que no "
                f"existe en una matriz de {rows} filas. {VECTOR_REBUILD_ADVICE}"
            )
        grouped[row].append(chunk_id)
    return rows, chunk_rows, tuple(tuple(ids) for ids in grouped), fingerprints


def _bounded_int(raw: object, field: str, minimum: int) -> int:
    """One integer field of the meta, at or above `minimum`, or a refusal naming it.

    `int(raw)` on its own would take `"768"`, `True` and `7.9` — three documents this code
    cannot honour, arriving as a dimension that silently disagrees with the matrix.

    `dimension` takes a minimum of 1 and `rows` of 0, and the difference is not pedantry: the
    size check is `rows * dimension * 4`, which for dimension 0 is 0 bytes for ANY number of
    rows. A zero dimension therefore does not merely describe an impossible plane, it disarms
    the one guard standing between a truncated matrix and a corpus silently read short.
    """
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < minimum:
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} trae un {field} que no es un entero >= {minimum}: "
            f"{raw!r}. {VECTOR_REBUILD_ADVICE}"
        )
    return raw


def _int_map(raw: object, field: str) -> dict[str, int]:
    """One `{str: int}` block of the meta, or a refusal naming the field."""
    if not isinstance(raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in raw.items()
    ):
        raise VectorPlaneIncompatible(
            f"{VECTORS_META_FILENAME} trae un {field} que no es un mapa de texto a entero. "
            f"{VECTOR_REBUILD_ADVICE}"
        )
    return {str(key): int(value) for key, value in raw.items()}


def _mapped_matrix(path: Path, rows: int, dimension: int, digest: str) -> Matrix | None:
    """The matrix as a read-only `np.memmap`, or `None` for an empty plane.

    THE SIZE IS CHECKED FIRST, and this is the check the whole atomic-write story rests on: a
    matrix truncated by a full disk or a killed build is still a valid float32 file, just
    shorter, and `np.memmap` would map whatever fits and answer queries over a corpus missing
    its tail with no sign that anything was lost.

    AND ALL THREE STEPS GO THROUGH ONE OPEN FILE, which is not tidiness — it is the only
    thing that makes the digest mean anything. `os.replace` swaps a directory ENTRY, not an
    inode, so measuring the name, hashing the name and mapping the name are three independent
    lookups that a rebuild landing between any two of them answers differently.
    `write_vector_plane` ends with exactly that call, so the racing writer is this module's
    own: the verification would pass over the old matrix and the query be served the new one,
    under the old meta's `chunk_rows`, with every id still resolving and every one of them
    reading another text's vector. An open descriptor pins the inode, and a POSIX mapping
    outlives the descriptor that made it, so what is mapped is exactly what was hashed.
    """
    expected_bytes = rows * dimension * _FLOAT32_BYTES
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise VectorPlaneIncompatible(
            f"falta {VECTORS_FILENAME} en {path.parent} ({exc}). {VECTOR_REBUILD_ADVICE}"
        ) from exc
    with handle:
        actual_bytes = os.fstat(handle.fileno()).st_size
        if actual_bytes != expected_bytes:
            raise VectorPlaneIncompatible(
                f"{VECTORS_FILENAME} ocupa {actual_bytes} bytes y su meta describe {rows} "
                f"filas de dimensión {dimension} ({expected_bytes} bytes): los vectores "
                f"están incompletos. {VECTOR_REBUILD_ADVICE}"
            )
        actual_digest = _sha256_stream(handle)
        if actual_digest != digest:
            raise VectorPlaneIncompatible(
                f"{VECTORS_FILENAME} no es la matriz que describe su meta (sha256 "
                f"{actual_digest[:12]}… frente a {digest[:12]}…): el tamaño coincide y el "
                f"contenido no, así que cada chunk leería la geometría de otro texto. "
                f"{VECTOR_REBUILD_ADVICE}"
            )
        if rows == 0:
            return None
        handle.seek(0)
        numpy = _numpy()
        return numpy.memmap(handle, dtype=numpy.float32, mode="r", shape=(rows, dimension))


def load_vector_plane(index_dir: Path, *, expected: VectorSpec | None = None) -> VectorPlane:
    """Open the plane read-only, refusing anything it cannot answer honestly.

    `expected` is the spec the caller intends to query with — the manifest's, or the config's.
    Given, it must match what built the plane; omitted, the plane is opened under its own
    stored spec, which is what `index status` and a rebuild need in order to say what is
    there.
    """
    if not vector_plane_exists(index_dir):
        raise VectorPlaneIncompatible(
            f"no hay plano vectorial en {index_dir} ({VECTORS_FILENAME} y "
            f"{VECTORS_META_FILENAME}). {VECTOR_REBUILD_ADVICE}"
        )
    raw = _meta_mapping(index_dir / VECTORS_META_FILENAME)
    spec = _meta_spec(raw)
    if expected is not None:
        _require_same_spec(spec, expected)
    rows, chunk_rows, row_chunks, text_rows = _checked_row_maps(raw)
    matrix = _mapped_matrix(
        index_dir / VECTORS_FILENAME, rows, spec.dimension, str(raw["matrix_sha256"])
    )
    return VectorPlane(
        spec=spec,
        _matrix=matrix,
        _chunk_rows=chunk_rows,
        _row_chunks=row_chunks,
        _text_rows=text_rows,
    )


@dataclass(frozen=True)
class VectorPlane:
    """A loaded, read-only vector plane — constructed only by `load_vector_plane`.

    Holding one IS the proof that the meta was total, closed, current, unit-normalized and
    consistent with the matrix on disk, so nothing downstream has to remember to check.
    """

    spec: VectorSpec
    _matrix: Matrix | None
    _chunk_rows: Mapping[str, int]
    _row_chunks: tuple[tuple[str, ...], ...]
    _text_rows: Mapping[str, int]
    _closed: bool = False

    @property
    def row_count(self) -> int:
        """Distinct vectors stored — fewer than `chunk_count` wherever text repeats."""
        return len(self._row_chunks)

    @property
    def chunk_count(self) -> int:
        """Chunks the plane can answer for, duplicates included."""
        return len(self._chunk_rows)

    def row_of(self, chunk_id: str) -> int | None:
        """The row a chunk reads, or `None` if this plane never saw it."""
        return self._chunk_rows.get(chunk_id)

    def chunk_ids_for_row(self, row: int) -> tuple[str, ...]:
        """Every chunk sharing that row, in `chunk_id` order.

        More than one means the corpus quotes the same text twice, which is a fact worth being
        able to see rather than a duplicate to hide.

        A row outside the matrix is REFUSED rather than indexed: `-1` is a perfectly valid
        Python index and answers with the LAST row, so a caller that computed a row wrong
        would receive another text's chunk ids and attribute them to a fragment nobody
        retrieved.
        """
        if not 0 <= row < len(self._row_chunks):
            raise VectorPlaneIncompatible(
                f"la fila {row} no existe en una matriz de {len(self._row_chunks)} filas"
            )
        return self._row_chunks[row]

    def covers(self, chunk_id: str, text: str) -> bool:
        """Whether this plane holds the vector OF THIS TEXT for this chunk (Plan 03.4).

        THE TEXT IS PART OF THE QUESTION, and that is the whole reason this is not
        `row_of(chunk_id) is not None`. A `chunk_id` is POSITIONAL —
        `<surface_id>:<chunk_index>:<chunker_version>` — so it survives an edit of the prose
        behind it: after `enrich` rewrites a summary the id still resolves, still reads a row,
        and that row still answers with the geometry of what used to be there. An id-only
        check calls that plane complete. Measured on the fixture corpus: an edited summary
        changed two chunks and left both ids untouched.
        """
        row = self._chunk_rows.get(chunk_id)
        return row is not None and self._text_rows.get(text_fingerprint(text)) == row

    def close(self) -> None:
        """Release the mapping. A memmap holds a file handle until it is dropped.

        The flag is kept SEPARATE from `_matrix` being `None`, which already means something
        else — an empty corpus — and conflating the two is what let a query on a closed plane
        answer `()` instead of raising. Both are set through `object.__setattr__` because the
        dataclass is frozen and these two are the one piece of state that legitimately moves.
        """
        object.__setattr__(self, "_closed", True)
        object.__setattr__(self, "_matrix", None)

    def search(
        self,
        query: Sequence[float],
        limit: int,
        *,
        allowed_chunk_ids: Collection[str] | None = None,
    ) -> tuple[VectorHit, ...]:
        """The top `limit` chunks by cosine similarity, best first, deterministic under ties.

        `allowed_chunk_ids` narrows the candidates BEFORE anything is scored, because a filter
        applied afterwards is not a filter: it would score the whole corpus, cut at `limit`,
        and then discard — returning nothing at all whenever what the caller asked for ranks
        below the cut, which looks exactly like an empty corpus. An id this plane does not
        know is ignored rather than fatal: the lexical and vector planes can legitimately
        disagree about a chunk (one was updated, the other not), and a query is not where that
        gets adjudicated.

        Ties are broken by `chunk_id`, and that is not cosmetic. Near-duplicate prose ties
        constantly once the corpus is real, and `argpartition` resolves a tie by whatever the
        partition happened to do — stable within a run, free to differ across runs, which
        would break spec §8.6's reproducibility gate without ever looking wrong.
        """
        if self._closed:
            raise ValueError(
                "este plano vectorial está cerrado: devolver un top-k vacío haría que un "
                "error del llamante se leyese como un corpus sin resultados"
            )
        if limit <= 0:
            raise ValueError("search requiere un limit positivo")
        _validated_unit_vector(query, self.spec.dimension, "la consulta")
        rows = self._candidate_row_indices(allowed_chunk_ids)
        matrix = self._matrix
        if matrix is None or not rows:
            return ()
        return self._ranked(matrix, rows, query, limit, allowed_chunk_ids)

    def _candidate_row_indices(self, allowed: Collection[str] | None) -> list[int]:
        """The rows worth scoring: all of them, or just those the allowed chunks read."""
        if allowed is None:
            # ROWS WITH NO CHUNK ARE NOT CANDIDATES. Plan 03 §2.3 makes orphans a designed
            # state — a deleted chunk leaves its row behind until the next `build --force`
            # compacts it — and `_top_positions` takes the best `limit` ROWS. An orphan among
            # them expands to nothing, so the slot is spent and a result that does have an
            # owner never reaches the caller, with nothing saying it was dropped.
            return [row for row in range(self.row_count) if self._row_chunks[row]]
        rows = {row for row in (self._chunk_rows.get(cid) for cid in allowed) if row is not None}
        return sorted(rows)

    def _ranked(
        self,
        matrix: Matrix,
        rows: list[int],
        query: Sequence[float],
        limit: int,
        allowed: Collection[str] | None,
    ) -> tuple[VectorHit, ...]:
        """Score the candidate rows, expand each onto its chunks, and cut at `limit`."""
        numpy = _numpy()
        vector = numpy.asarray(query, dtype=numpy.float32)
        # Fancy-indexing COPIES the candidate rows, which is the point: when a filter has
        # narrowed the corpus, scoring the copy is cheaper than scoring the whole matrix and
        # masking. Unnarrowed, the mapping is used in place and nothing is copied at all.
        scores = (matrix[rows] if len(rows) < self.row_count else matrix) @ vector
        scored = [
            (float(scores[position]), chunk_id, rows[position])
            for position in _top_positions(numpy, scores, limit)
            for chunk_id in self._row_chunks[rows[position]]
            if allowed is None or chunk_id in allowed
        ]
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return tuple(
            VectorHit(chunk_id=chunk_id, row=row, score=score)
            for score, chunk_id, row in scored[:limit]
        )


def _top_positions(numpy: ModuleType, scores: Matrix, limit: int) -> list[int]:
    """The positions worth expanding — the `limit` best, plus everything tied with the last.

    Two steps, and the second is what makes the first safe. `argpartition` finds `limit` of
    the largest scores in linear time but says nothing about WHICH of several equal scores it
    kept, so every row scoring at least as much as the worst of them is taken as a candidate
    and the exact order is settled afterwards by `(-score, chunk_id)`.

    Taking `limit` ROWS is enough to guarantee the top `limit` CHUNKS, even though a row can
    hold many chunks: a chunk scores exactly what its row scores, so any row excluded here has
    at least `limit` rows strictly ahead of it, each contributing at least one chunk that
    outranks every chunk on the excluded row.
    """
    total = int(scores.shape[0])
    if total <= limit:
        return list(range(total))
    partitioned = numpy.argpartition(-scores, limit - 1)[:limit]
    threshold = float(scores[partitioned].min())
    return [int(position) for position in numpy.flatnonzero(scores >= threshold)]
