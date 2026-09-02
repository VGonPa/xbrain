"""Opening the index FOR A QUERY: compatibility, degradation, and the fail-closed chunk.

Three things stand between a query and the rows, and each one exists because the alternative
fails silently.

**The connection is read-only.** `file:…?mode=ro`, so a write RAISES. Spec §5.6: *the query
operation does not modify or repair the index silently*, and §3.7.12 makes `search`, `get`
and `index status` read operations. "We checked and it does not write" is a claim about the
code; being unable to write is a property of the object.

**An incompatible manifest refuses the whole query.** Spec §9.3 — never a partial answer over
a schema, an emitter or a chunker the code no longer matches, because a partial answer over
the wrong version is a wrong answer wearing a right one's shape.

**A chunk whose fingerprint does not recompute is NOT RETURNED, and is counted.** Invariant 6
of spec §3.7. The check is cheap because the fingerprint is recomputed over the text already
in the row; what it detects is a row written by a different chunker or edited by hand. It is
counted in `corrupt_chunks_excluded`, whose name was `stale_chunks_excluded` until B3 pointed
out that it sounded like the OTHER signal and measured this one.

TWO SIGNALS, AND THE SECOND IS THE ONE THAT WILL ACTUALLY FIRE. Indexing is manual by
decision (spec §9.2), so the failure that happens is *you ran `enrich` and did not reindex* —
which the fingerprint check cannot see, because every row is internally consistent with
itself. `index_behind_store` compares the manifest's cheap `StoreSignal` against
`data/items.json` right now. It is DECLARED, not repaired and not raised: spec §9.3 calls it
*evidencia posiblemente obsoleta*, which is answerable as long as the answer says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from xbrain.knowledge.chunking import ChunkerParams
from xbrain.knowledge.contracts import IndexStatusRef
from xbrain.knowledge.ids import chunk_fingerprint
from xbrain.knowledge.index_build import (
    Manifest,
    StoreSignal,
    load_compatible_manifest,
    require_consistent,
)
from xbrain.knowledge.index_schema import open_index, require_database
from xbrain.knowledge.lexical import LexicalHit, LexicalIndex

# The degradations this plan can declare, in a FIXED order so two responses over the same
# state are byte-identical (spec §3.7.8 applies to the envelope too, not only to the ranking).
#
# `no_embeddings` is what every Plan 02 index declares, because every Plan 02 manifest carries
# `embeddings: null`: spec §9.3 requires that without a vector backend *lexical sigue operativo
# y el response declara estrategia degradada*, and declaring it is what stops a consumer
# reading a lexical answer as a hybrid one. It is derived from the manifest, not permanent
# (B-i) — see `_degraded`.
DEGRADED_ORDER: tuple[str, ...] = ("index_behind_store", "no_embeddings")


@dataclass(frozen=True)
class OpenIndex:
    """A read-only index that has already proved it can be queried.

    Constructed only by `open_for_query`, so nothing downstream has to remember to check the
    manifest: holding one of these IS the proof that the check passed.
    """

    lexical: LexicalIndex
    manifest: Manifest
    degraded: tuple[str, ...]

    def close(self) -> None:
        self.lexical.connection.close()

    def status_ref(
        self,
        corrupt_chunks_excluded: int = 0,
        *,
        strategy_degradation: tuple[str, ...] = (),
    ) -> IndexStatusRef:
        """The `index` block of a response (spec §7.2).

        `strategy_degradation` comes from the CALLER, not from the index: whether the
        requested retriever exists is a property of the build (`IMPLEMENTED_STRATEGIES`),
        not of the database that was opened. It LEADS the tuple because *what you asked for
        did not run* outranks *this index has no embeddings*, and putting it first keeps the
        order deterministic without a set operation (spec §3.7.8 applies to the envelope).
        """
        return IndexStatusRef(
            manifest_version=self.manifest.schema_version,
            built_at=self.manifest.built_at,
            corrupt_chunks_excluded=corrupt_chunks_excluded,
            degraded=strategy_degradation + self.degraded,
        )


def open_for_query(
    index_dir: Path,
    items_path: Path,
    *,
    vocab_path: Path | None = None,
    topics_path: Path | None = None,
    params: ChunkerParams | None = None,
) -> OpenIndex:
    """Open the index read-only, refusing anything the code cannot answer honestly.

    `params` is threaded through so a chunker sweep that changed the parameters without
    bumping `CHUNKER_VERSION` is caught here too — the case where the id resolves and the
    text behind it is not what it was.

    AND THE BASE MUST HOLD WHAT THE MANIFEST DECLARES (G-2, B-c). Versions and schema were
    checked; `counts` were not, so a base amputated behind the manifest's back — or the
    empty one a write door used to leave behind (`index update --dry-run` over a deleted
    database) — was answered as a corpus with no matches, `degraded: ["no_embeddings"]` and
    nothing else, while `update` and `status` refused it naming the plane. The check is the
    SAME function they run (`require_consistent`), five `COUNT(*)` measured at 0.04 ms on the
    52 MB real index, so a query and the two maintenance commands say one thing.
    """
    database = require_database(index_dir)
    manifest = load_compatible_manifest(index_dir, params=params)
    degraded = _degraded(manifest, items_path, vocab_path, topics_path)
    connection = open_index(database, read_only=True)
    # The SAME question the two maintenance doors ask, through the same function (round 06,
    # `index_build.describe_base`); the query door does not pay `quick_check` (B-1) and a
    # page it never read fails closed the moment a query touches it (`_fetch`, G-4).
    require_consistent(connection, manifest, database, whole_file=False)
    return OpenIndex(lexical=LexicalIndex(connection), manifest=manifest, degraded=degraded)


def _degraded(
    manifest: Manifest, items_path: Path, vocab_path: Path | None, topics_path: Path | None
) -> tuple[str, ...]:
    """Which degradations apply right now, in `DEGRADED_ORDER`.

    `index_behind_store` is ONE `os.stat` (B3). A `touch` with no edit is a false positive
    and that is accepted: a false positive costs one warning, a false negative costs serving
    stale evidence as fresh. It fails towards the warning.

    `no_embeddings` is READ OFF THE MANIFEST (B-i), never hard-coded: the manifest's
    `embeddings` block is where a vector backend is recorded, and a flag that was always on
    said nothing about this index — the day Plan 03 writes the block, a constant would keep
    declaring a degradation that no longer applies, with the test beside it green (the F-2
    shape, one field over).
    """
    flags = set()
    if manifest.embeddings is None:
        flags.add("no_embeddings")
    if manifest.store_signal != StoreSignal.of(items_path, vocab_path, topics_path):
        flags.add("index_behind_store")
    return tuple(flag for flag in DEGRADED_ORDER if flag in flags)


def verify_fingerprints(hits: Sequence[LexicalHit]) -> tuple[tuple[LexicalHit, ...], int]:
    """Drop every hit whose fingerprint does not recompute. Returns `(kept, excluded)`.

    Invariant 6 of spec §3.7: *un índice obsoleto falla cerrado para el chunk afectado y
    reporta la exclusión*. Recomputed over the text ALREADY IN THE ROW, so this is an
    internal consistency check: it catches a row written by a different chunker version or
    edited by hand, and it cannot catch a store that moved — that is the other signal.

    Excluded rather than repaired, and counted rather than logged: spec §5.6 forbids the
    query from repairing the index, and a silent exclusion would make the corpus look smaller
    than it is with nothing saying why.
    """
    kept: list[LexicalHit] = []
    excluded = 0
    for hit in hits:
        expected = chunk_fingerprint(hit.surface_id, hit.chunk_index, hit.text)
        if expected == hit.fingerprint:
            kept.append(hit)
        else:
            excluded += 1
    return tuple(kept), excluded
