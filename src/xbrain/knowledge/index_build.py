"""`index build`, `index update`, `index status` and the manifest (Plan 02 §2, §3).

THE MANIFEST IS THE CONTRACT BETWEEN A BUILD AND EVERY LATER QUERY. Spec §5.6 enumerates what
it must record; `MANIFEST_FIELDS` is that enumeration in one place, asserted as a set by the
suite so a field dropped from the writer cannot quietly disappear from a document nobody
reads until a query answers under the wrong chunker.

TWO SIGNALS, TWO COSTS, TWO PLACES (B3). Indexing is MANUAL BY DECISION (spec §9.2), so the
failure that actually happens is not corruption — it is *you ran `enrich` and did not
reindex*. Two different instruments answer two different questions:

| `StoreSignal` — mtime + size | one `os.stat` | EVERY query | "the store moved" |
| `store_fingerprint` — sha256 per item | loads the store | build/update/status | "WHICH items changed, and how many" |

The cheap one can give false positives (a `touch` with no edit) and that is accepted: a false
positive costs one warning, a false negative costs serving stale evidence as fresh. It fails
towards the warning, the same direction `origin: unknown -> llm_synthesis` fails.

THE PER-ITEM FINGERPRINT IS OVER THE SURFACES, NOT OVER `(fetched_at, enriched_at)` as Plan
01 §10 sketched, and CLAUDE.md rule 6 is both reasons. First, `content.fetched_at` cannot
reach an item whose `content` is `None` — 960 of 2,404 in the real store, measured
2026-09-01 on sha256 `f76341a3…` — because there is
nothing to stamp. Second, a timestamp is a PROXY: a summary edited by hand, or any repair
that rewrites text without touching a clock, changes the indexable corpus and leaves the
proxy unmoved, so the index would keep serving the old body under a fingerprint asserting it
is current. Hashing the emitted surface fingerprints asks the question directly — *did the
text this index holds change?* — and it reaches every item. The cost is one emitter pass over
the store per `update`/`status`, which is measured and published in the execution report.

NOTHING HERE WRITES TO THE STORE. `items.json`, `vocab.yaml` and `topics.json` are read and
never touched, and no command of this plan takes a snapshot, because none of them is
destructive: `data/index/` is derived and reconstructible by definition (spec §5.6).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from xbrain.executors.api import iter_content_sources
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams, chunk_surfaces
from xbrain.knowledge.ids import CHUNKER_VERSION, SURFACE_VERSION, surface_fingerprint
from xbrain.knowledge.index_schema import (
    REBUILD_ADVICE,
    SCHEMA_VERSION,
    IndexIncompatibleError,
    IndexMissingError,
    db_path,
    delete_chunk_rows,
    delete_item_rows,
    delete_profile_rows,
    corrupt_base_error,
    manifest_path,
    open_index,
    open_memory_index,
    quick_check,
    reading_base,
    require_database,
)
from xbrain.knowledge.lexical import LexicalIndex
from xbrain.knowledge.lexical_fts import FTS_CONNECTIVE, FTS_TOKENIZE
from xbrain.knowledge.models import KnowledgeSurface, TopicRecord
from xbrain.knowledge.profile import profile_text
from xbrain.knowledge.surfaces import (
    CONTENT_KIND_TO_SURFACE_TYPES,
    article_block_texts,
    item_surfaces,
    item_topics,
    knowledge_item,
    topic_record,
    topic_surfaces,
)
from xbrain.models import Item, MediaPhotoDescribed, Topic, TopicPage
from xbrain.rubrics import parse_vocab
from xbrain.store import parse_store, parse_topic_pages

# Spec §5.6, field by field, in one place. The suite asserts the written document's key set
# equals this, so writer and contract cannot drift apart.
MANIFEST_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "built_at",
        "store_fingerprint",
        "store_signal",
        "vocab_fingerprint",
        "topics_fingerprint",
        "surface_version",
        "chunker_version",
        "chunker_params",
        # Beyond spec §5.6 ON PURPOSE: these two DECIDE every recall number. The connective
        # change of Plan 01 M3 moved mean recall@10 from 0.1429 to 0.8099 without touching
        # the chunker, and a manifest that recorded the chunker but not the query semantics
        # would let two incomparable baselines look like one measurement.
        "tokenize",
        "connective",
        "embeddings",
        "counts",
        "skipped",
        "failed",
    }
)

# `REBUILD_ADVICE` is imported from `index_schema`, beside the error that carries it: a
# corrupt database raises the same error with the same advice, and two copies of the sentence
# would be two things that have to be kept in step (rule 5).
UPDATE_ADVICE = "Actualiza el índice con `xbrain index update`."

# The NESTED schema of the manifest, each set defined ONCE and read by the writer, the reader
# and the consistency check (B1, round 06). `counts` holds exactly the five planes
# `count_rows` counts; `skipped` exactly the four causes spec §5.6 names; `chunker_params`
# exactly the fields of `ChunkerParams`. A plane added to `_COUNT_STATEMENTS` is therefore
# required of every manifest, compared by `manifest_mismatch` and refused when absent, with
# nobody remembering to add it in three places.
COUNT_PLANES: frozenset[str] = frozenset({"items", "topics", "surfaces", "chunks", "profiles"})
SKIPPED_CAUSES: frozenset[str] = frozenset(
    {"empty_text", "decorative", "no_speech", "failed_sources"}
)
CHUNKER_PARAM_NAMES: frozenset[str] = frozenset(
    field_.name for field_ in dataclass_fields(ChunkerParams)
)


@dataclass(frozen=True)
class StoreSignal:
    """The CHEAP change signal: `mtime_ns` and size of the THREE inputs (spec §5.6, P1a).

    Three `os.stat`, so a query can afford it on every call. A missing file yields zeros
    rather than raising: a query must still be able to say *the index is behind* when the
    store has been moved away, and raising from inside `search` is the wrong place to learn
    it.

    THREE FILES, NOT ONE (P1a, gate Codex round 05). Spec §5.6 names `data/items.json` as
    the file the cheap signal watches, and that is what the first version stat'ed — but the
    index derives from `vocab.yaml` and `topics.json` too: a topic description enters every
    assigned item's PROFILE (spec §5.1.A), and overviews and notes are chunks the index
    serves. The manifest already recorded their deep fingerprints; the query door never
    compared them, so `xbrain topics` — which writes `topics.json` and never `items.json` —
    left every later `search` answering over the old topic plane with nothing declared, the
    silent staleness spec §9.3 forbids, on two of the three inputs. One signal over the
    three files, one comparison, three `stat` calls.

    A manifest written before round 05 carries no vocab/topics entries: they read back as
    zeros, compare unequal to the live files, and the index is declared behind until the
    next `update` re-seals the manifest. That is the direction this signal is meant to
    fail in — towards the warning — and it costs one `index update`.
    """

    items_json_mtime_ns: int
    items_json_size: int
    vocab_yaml_mtime_ns: int = 0
    vocab_yaml_size: int = 0
    topics_json_mtime_ns: int = 0
    topics_json_size: int = 0

    @classmethod
    def of(
        cls, items_path: Path, vocab_path: Path | None = None, topics_path: Path | None = None
    ) -> StoreSignal:
        """The signal of the three inputs AS THEY ARE ON DISK NOW — the query-time side."""
        items_mtime, items_size = _stat_signal(items_path)
        vocab_mtime, vocab_size = _stat_signal(vocab_path)
        topics_mtime, topics_size = _stat_signal(topics_path)
        return cls(
            items_json_mtime_ns=items_mtime,
            items_json_size=items_size,
            vocab_yaml_mtime_ns=vocab_mtime,
            vocab_yaml_size=vocab_size,
            topics_json_mtime_ns=topics_mtime,
            topics_json_size=topics_size,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "items_json_mtime_ns": self.items_json_mtime_ns,
            "items_json_size": self.items_json_size,
            "vocab_yaml_mtime_ns": self.vocab_yaml_mtime_ns,
            "vocab_yaml_size": self.vocab_yaml_size,
            "topics_json_mtime_ns": self.topics_json_mtime_ns,
            "topics_json_size": self.topics_json_size,
        }

    @classmethod
    def from_dict(cls, raw: object) -> StoreSignal:
        """The signal as a manifest recorded it — validated, never cast (B1).

        The two `items.json` entries are REQUIRED; the four vocab/topics entries are the
        round-05 additions and default to zero, which is the documented compatibility
        promise: a pre-round-05 manifest compares unequal to the live files and is declared
        behind until its first `update`. Required and optional are read off THIS dataclass
        — a field with a default is optional — so the schema has one definition.
        """
        return cls(**_closed_int_mapping(raw, "store_signal", *_signal_keys()))


def _stat_signal(path: Path | None) -> tuple[int, int]:
    """`(mtime_ns, size)` of one input file, or `(0, 0)` when it is absent or not given."""
    if path is None:
        return 0, 0
    try:
        stat = path.stat()
    except OSError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


@dataclass(frozen=True)
class IndexInputs:
    """The three inputs of the index AND the cheap signal of the snapshot they were read from.

    The signal travels WITH the objects because it describes them (P1b): a signal taken from
    the path at any other moment describes whatever file is there at that moment, which is
    what let the manifest certify an `items.json` the base had never seen.
    """

    store: dict[str, Item]
    vocab: list[Topic]
    topic_pages: dict[str, TopicPage]
    signal: StoreSignal


def load_index_inputs(
    items_path: Path, vocab_path: Path | None = None, topics_path: Path | None = None
) -> IndexInputs:
    """Read the three inputs and return them WITH the signal of the bytes that were read.

    THE SIGNAL IS BOUND TO THE SNAPSHOT, NOT TO THE PATH (P1b, gate Codex round 05). `build`
    and `update` used to seal the manifest with `StoreSignal.of(items_path)` taken AFTER the
    rows were committed — a `stat` of whatever file the path pointed at by then. The caller
    had loaded the store minutes earlier (the CLI loads it before calling in), so a save that
    landed in that window put the base under the OLD objects and the manifest under the NEW
    file's mtime and size: `search` then compared equal signals and answered over stale rows
    with nothing declared, while `status` — which loads the store — saw the changed item.
    The gate's probe A: `raceonlytoken` in the file, not in the base, `degraded:
    ("no_embeddings",)`, `items_changed=1`, `behind=False`.

    Every file is read through ITS OWN HANDLE and the signal is `os.fstat` of that handle,
    taken BEFORE the read. The store's writers replace files atomically (`os.replace`), so an
    open handle keeps the inode it opened and the bytes parsed are the bytes that inode holds:
    the signal describes exactly what was parsed, by construction, and a replacement that
    lands during the read leaves the path pointing at a NEWER inode, which the query-time
    `StoreSignal.of` then reports as different — the index declares itself behind. For a
    writer that rewrites in place instead (`save_vocab` uses `write_text`), taking the stat
    before the read means a write that lands mid-read produces an older signal than the
    content, so the index is again declared behind rather than certified fresh: the same
    direction, the warning.

    A missing file reads as its empty value and a zero signal, exactly as `load_store`,
    `load_vocab`, `load_topic_pages` and `StoreSignal.of` treat it.
    """
    items_text, items_mtime, items_size = _read_bound(items_path)
    vocab_text, vocab_mtime, vocab_size = _read_bound(vocab_path)
    topics_text, topics_mtime, topics_size = _read_bound(topics_path)
    return IndexInputs(
        store=parse_store(items_text) if items_text is not None else {},
        vocab=parse_vocab(vocab_text) if vocab_text is not None else [],
        topic_pages=parse_topic_pages(topics_text) if topics_text is not None else {},
        signal=StoreSignal(
            items_json_mtime_ns=items_mtime,
            items_json_size=items_size,
            vocab_yaml_mtime_ns=vocab_mtime,
            vocab_yaml_size=vocab_size,
            topics_json_mtime_ns=topics_mtime,
            topics_json_size=topics_size,
        ),
    )


def _bound_signal(
    signal: StoreSignal | None, items_path: Path, vocab_path: Path | None, topics_path: Path | None
) -> StoreSignal:
    """The signal `build`/`update` seal into the manifest: the caller's, or a stat taken NOW.

    The caller who LOADED the objects is the only one who can say which snapshot they are —
    `load_index_inputs` hands the signal over with them, and the CLI passes it through. A
    caller that passes none gets the three paths stat'ed HERE, before the first row is
    written — never after the commit, which is where the first version took it and how the
    manifest came to certify a file the base had never seen (P1b). That fallback binds
    nothing to the objects; it is honest only for a caller that wrote the files itself a
    moment ago (the suite's fixtures), and the docstring of `build` says so.
    """
    if signal is not None:
        return signal
    return StoreSignal.of(items_path, vocab_path, topics_path)


def _read_bound(path: Path | None) -> tuple[str | None, int, int]:
    """`(text, mtime_ns, size)` of one input, the stat taken on the handle the text came from.

    `(None, 0, 0)` for an absent or unnamed file, matching `_stat_signal`.
    """
    if path is None:
        return None, 0, 0
    try:
        handle = path.open("rb")
    except OSError:
        return None, 0, 0
    with handle:
        stat = os.fstat(handle.fileno())
        data = handle.read()
    return data.decode("utf-8"), stat.st_mtime_ns, stat.st_size


@dataclass(frozen=True)
class IndexOptions:
    """Everything a build needs that is not the corpus itself.

    `transcribe_command` and `vision_command` travel here for the reason `item_surfaces`
    takes them: they are the only PRODUCERS that do not live in the store, and CLAUDE.md
    records why that matters — parakeet does not fail on Spanish audio, it invents, so a
    reader must be able to recover what wrote the words they are reading.
    """

    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS
    vault_dir: Path | None = None
    transcribe_command: str | None = None
    vision_command: str | None = None


@dataclass(frozen=True)
class Manifest:
    """The index's self-description. Written last, so its presence means the build finished."""

    schema_version: str
    built_at: datetime
    store_fingerprint: str
    store_signal: StoreSignal
    vocab_fingerprint: str
    topics_fingerprint: str
    surface_version: str
    chunker_version: str
    chunker_params: dict[str, int]
    tokenize: str
    connective: str
    counts: dict[str, int]
    skipped: dict[str, int]
    failed: list[dict[str, str]] = field(default_factory=list)
    # The hole Plan 03 fills with `{model, dimension, normalized, command_version}`. Declared
    # now so its arrival is not a manifest migration in the next plan.
    embeddings: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "built_at": self.built_at.isoformat(),
            "store_fingerprint": self.store_fingerprint,
            "store_signal": self.store_signal.to_dict(),
            "vocab_fingerprint": self.vocab_fingerprint,
            "topics_fingerprint": self.topics_fingerprint,
            "surface_version": self.surface_version,
            "chunker_version": self.chunker_version,
            "chunker_params": dict(self.chunker_params),
            "tokenize": self.tokenize,
            "connective": self.connective,
            "embeddings": self.embeddings,
            "counts": dict(self.counts),
            "skipped": dict(self.skipped),
            "failed": list(self.failed),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Manifest:
        """The document, TOTALLY validated — every nested key, every value's type (B1).

        The first version checked the top-level key set and cast what sat under it, so a
        manifest whose `counts` was `{}` loaded as compatible, and `manifest_mismatch`,
        iterating whatever `counts` offered, compared nothing: `status` called an amputated
        base healthy, `search` answered zero results over it with `no_embeddings` and nothing
        else, and `update` sealed the state as sound (the round-06 gate, reproduced). Spec
        §9.3: an incompatible manifest is never queried partially — and a manifest that
        declares less than the schema is incompatible, not lenient. Closed as well as total:
        an undeclared plane or cause is refused, because nothing could compare it.
        """
        missing = MANIFEST_FIELDS - set(raw)
        if missing:
            raise IndexIncompatibleError(
                f"El manifest no declara {sorted(missing)}. {REBUILD_ADVICE}"
            )
        return cls(
            schema_version=str(raw["schema_version"]),
            built_at=_instant(raw["built_at"]),
            store_fingerprint=str(raw["store_fingerprint"]),
            store_signal=StoreSignal.from_dict(raw["store_signal"]),
            vocab_fingerprint=str(raw["vocab_fingerprint"]),
            topics_fingerprint=str(raw["topics_fingerprint"]),
            surface_version=str(raw["surface_version"]),
            chunker_version=str(raw["chunker_version"]),
            chunker_params=_closed_int_mapping(
                raw["chunker_params"], "chunker_params", CHUNKER_PARAM_NAMES
            ),
            tokenize=str(raw["tokenize"]),
            connective=str(raw["connective"]),
            embeddings=_optional_mapping(raw["embeddings"], "embeddings"),
            counts=_closed_int_mapping(raw["counts"], "counts", COUNT_PLANES),
            skipped=_closed_int_mapping(raw["skipped"], "skipped", SKIPPED_CAUSES),
            failed=_failures(raw["failed"]),
        )


def _malformed(field_name: str, detail: str) -> IndexIncompatibleError:
    """One actionable sentence for every malformed field, naming the field and the reason.

    A hand-edited manifest is exactly the input this reader has to survive, and spec §9.3
    asks for a stable error rather than a traceback from inside a query.
    """
    return IndexIncompatibleError(
        f"El manifest tiene el campo {field_name!r} malformado: {detail}. {REBUILD_ADVICE}"
    )


def _closed_int_mapping(
    value: object,
    field_name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, int]:
    """A mapping with EXACTLY the declared keys, each a non-negative integer.

    `type(count) is int` rather than `isinstance`: `True` is an `int` to `isinstance`, and a
    JSON `true` where a count belongs is a malformed document, not a count of one. A string
    `"56"` is refused for the same reason — `int("56")` accepted it silently before.
    """
    if not isinstance(value, dict):
        raise _malformed(field_name, "no es un objeto")
    keys = set(value)
    if absent := sorted(required - keys):
        raise _malformed(field_name, f"faltan {absent}")
    if unknown := sorted(keys - required - optional):
        raise _malformed(field_name, f"claves no declaradas {unknown}")
    for key, count in value.items():
        if type(count) is not int or count < 0:
            raise _malformed(field_name, f"{key!r} debe ser un entero no negativo, es {count!r}")
    return {str(key): int(count) for key, count in value.items()}


def _signal_keys() -> tuple[frozenset[str], frozenset[str]]:
    """`(required, optional)` entries of `store_signal`, read off `StoreSignal` itself."""
    required = frozenset(f.name for f in dataclass_fields(StoreSignal) if f.default is MISSING)
    optional = frozenset(f.name for f in dataclass_fields(StoreSignal)) - required
    return required, optional


def _optional_mapping(value: object, field_name: str) -> dict[str, object] | None:
    """`null` or an object — the `embeddings` slot Plan 03 fills — never anything else."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _malformed(field_name, "no es null ni un objeto")
    return {str(key): item for key, item in value.items()}


def _failures(value: object) -> list[dict[str, str]]:
    """The `failed` list: every entry a mapping of strings, or the document is refused."""
    if not isinstance(value, list) or not all(
        isinstance(entry, dict) and all(isinstance(v, str) for v in entry.values())
        for entry in value
    ):
        raise _malformed("failed", "no es una lista de objetos de texto")
    return [{str(k): str(v) for k, v in entry.items()} for entry in value]


def _instant(value: object) -> datetime:
    """`built_at` as an instant, or the malformed-field sentence instead of a `ValueError`."""
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise _malformed("built_at", f"{value!r} no es un instante ISO") from error


@dataclass(frozen=True)
class BuildReport:
    """What a build did — or, under `--dry-run`, what it WOULD have done."""

    items_written: int
    topics_written: int
    surfaces_written: int
    chunks_written: int
    profiles_written: int
    skipped: dict[str, int]
    failed: list[dict[str, str]]
    duration_seconds: float
    dry_run: bool


@dataclass(frozen=True)
class UpdateReport:
    """What an incremental update changed, counted per cause."""

    items_added: int
    items_changed: int
    items_removed: int
    chunks_inserted: int
    chunks_deleted: int
    profiles_inserted: int
    profiles_deleted: int
    topics_rebuilt: bool
    # H1: the topic ROWS rewritten because their members or `stale` bit moved while the
    # vocabulary and the pages did not. 0 when `topics_rebuilt` took the whole plane.
    topics_refreshed: int
    duration_seconds: float
    dry_run: bool


@dataclass(frozen=True)
class StatusReport:
    """What `index status` reports (acceptance 2, step 10c).

    `items_changed` is a NUMBER, not a flag: "something changed" does not distinguish a
    touched file from a hundred re-enriched items, and the difference decides whether a
    rebuild is worth its minutes.
    """

    manifest: Manifest | None
    counts: dict[str, int]
    items_added: int
    items_changed: int
    items_removed: int
    # H1: topics whose stored row (members, `stale`, description, synthesis) is not the row
    # the store implies now — read from the BASE, so an index whose item fingerprints all
    # match and whose topic plane is behind anyway is still declared.
    topics_changed: int
    behind: bool
    incomplete: bool
    advice: str


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------


def item_fingerprint(item: Item, *, options: IndexOptions | None = None) -> str:
    """sha256 over everything about this item that the INDEX holds.

    Two halves, and both are needed. The SURFACE ROWS answer *did what the index holds about
    each surface change?* — `surface_row` is the exact tuple `_write_surfaces` inserts, so
    it covers the surface fingerprint (emitter version, type, origin, body) AND the
    attribution, title, url, locator and language the index stores and `search` serves on
    every match (A-1). The filterable METADATA of the item (source, author, date, topics,
    content kinds) answers the other half: a changed author changes what `--author` returns
    even though not one character of text moved.

    THE ROW, NOT THE SURFACE FINGERPRINT ALONE (G-5). `surface_fingerprint` is
    `(version, type, origin, text)` by design and must stay so; but hashing only that here
    meant a `refresh-quoted` that filled in the author of a quoted post without touching
    its body left `update` reporting `0 cambiados` and `search` serving the old attribution
    — the evidence repaired, the derivative standing (CLAUDE.md rule 6), on the attribution
    rule this repo paid for in blood. Sharing the writer's projection is what makes "every
    stored column is hashed" structural rather than a list kept in step by hand.

    What is NOT hashed, and why: `producer`. The index has no producer column — `get` reads
    it from the configured transcribe/vision command at read time — so hashing a config
    value here would rewrite every ASR/VLM item on a binary rename for a field no query
    serves from the index.

    Deliberately NOT `(item_id, content.fetched_at, enriched.enriched_at)`: see the module
    docstring for why a timestamp proxy both misses hand edits and cannot reach the 40 % of
    the corpus with no `content` at all.
    """
    options = options or IndexOptions()
    surfaces = item_surfaces(
        item,
        transcribe_command=options.transcribe_command,
        vision_command=options.vision_command,
    )
    parts = [
        SURFACE_VERSION,
        item.id,
        item.source,
        item.url,
        item.author.handle,
        item.author.name,
        item.created_at.isoformat(),
        item.captured_at.isoformat(),
        item.bookmark_folder or "",
        *item_topics(item),
        *sorted(
            source.kind
            for _index, source in iter_content_sources(item, set(CONTENT_KIND_TO_SURFACE_TYPES))
        ),
        *(json.dumps(surface_row(surface), ensure_ascii=False) for surface in surfaces),
    ]
    return _sha256("\0".join(parts))


# The column order of `surfaces`, as ONE tuple type shared by the writer and the fingerprint.
SurfaceRow = tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    int,
    str | None,
    str | None,
    str | None,
    str | None,
    str,
    str | None,
    str,
    int,
]


def surface_row(surface: KnowledgeSurface) -> SurfaceRow:
    """What the index STORES about a surface — the row `_write_surfaces` inserts, verbatim.

    One projection, two readers: the writer binds it to the `INSERT`, `item_fingerprint`
    hashes it. A column added to `surfaces` therefore cannot be stored without being hashed,
    and `test_the_fingerprint_hashes_the_same_row_the_writer_inserts` reads the rows back to
    prove the two never drifted. The last column is a LENGTH, never the body (spec §10.8);
    the body is covered by `fingerprint`, which hashes it.
    """
    return (
        surface.surface_id,
        surface.owner_type,
        surface.owner_id,
        surface.surface_type,
        surface.origin,
        surface.trust_class,
        int(surface.derived),
        surface.attribution.handle if surface.attribution else None,
        surface.attribution.name if surface.attribution else None,
        surface.title,
        surface.locator.url,
        surface.locator.model_dump_json(),
        surface.language,
        surface.fingerprint,
        len(surface.text),
    )


# The column order of `topics`, as ONE tuple type shared by the writer and the comparator (H1).
TopicRow = tuple[
    str,
    str,
    str | None,
    str,
    str | None,
    int | None,
    int,
    str,
    str,
    str,
    str | None,
]


def topic_row(record: TopicRecord) -> TopicRow:
    """What the index STORES about a topic — the row `_write_topic_row` inserts, verbatim.

    The same pattern as `surface_row` (G-5), for the same reason: one projection, two
    readers. The writer binds it to the `INSERT`; `topic_rows_behind` compares it against
    what the base holds. A membership-derived column — `primary_item_ids_json`,
    `secondary_item_ids_json`, `stale` — therefore cannot be stored without being compared,
    which is exactly what H1 lacked: `update` decided the whole topic plane from the
    vocabulary and page fingerprints and never asked whether the members it had written were
    still the members the store implies, so a topic move done by `enrich` rewrote
    `item_topics` and left `topics` holding the old members under a healthy manifest.
    """
    return (
        record.slug,
        record.description.text,
        record.overview.text if record.overview else None,
        json.dumps([note.text for note in record.notes], ensure_ascii=False),
        record.synthesized_at.isoformat() if record.synthesized_at else None,
        record.post_count_at_synth,
        int(record.stale),
        json.dumps(list(record.primary_item_ids)),
        json.dumps(list(record.secondary_item_ids)),
        record.vocab_fingerprint,
        record.synthesis_fingerprint,
    )


def store_fingerprint(store: Mapping[str, Item], *, options: IndexOptions | None = None) -> str:
    """The DEEP store signal: one sha256 over every item's fingerprint, in id order.

    Order-independent by construction — the ids are sorted — because a dict's iteration order
    is a property of how the store was loaded, not of what it contains.
    """
    parts = [
        f"{item_id}={item_fingerprint(store[item_id], options=options)}"
        for item_id in sorted(store)
    ]
    return _sha256("\0".join(parts))


def vocab_fingerprint(vocab: Sequence[Topic]) -> str:
    """sha256 over the vocabulary's slugs AND descriptions.

    The descriptions are in because they enter every assigned item's PROFILE (spec §5.1.A):
    editing one changes indexed text on the item plane, not only on the topic plane.
    """
    parts = [f"{topic.slug}={topic.description}" for topic in sorted(vocab, key=lambda t: t.slug)]
    return _sha256("\0".join(parts))


def topics_fingerprint(pages: Mapping[str, TopicPage]) -> str:
    """sha256 over each topic page's overview and notes — the synthesised text."""
    parts = []
    for slug in sorted(pages):
        page = pages[slug]
        parts.append(surface_fingerprint("topic_overview", "llm", page.overview))
        parts += [surface_fingerprint("topic_note", "llm", note) for note in page.notes]
        parts.append(slug)
    return _sha256("\0".join(parts))


def _sha256(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# the writer — ONE definition, used by build, update and the evaluation harness
# ---------------------------------------------------------------------------


@dataclass
class WriteCounters:
    """Mutable tally shared by the writers. Not frozen: it is a running total, not a result."""

    surfaces: int = 0
    chunks: int = 0
    profiles: int = 0
    empty_text: int = 0
    decorative: int = 0
    no_speech: int = 0
    failed_sources: int = 0


def write_item(
    index: LexicalIndex,
    item: Item,
    vocab: Sequence[Topic],
    counters: WriteCounters,
    *,
    options: IndexOptions,
) -> None:
    """Everything one item contributes to the index: metadata, surfaces, chunks and profile.

    THE ONE WRITER. `build`, `update` and the evaluation harness all come through here, so
    there is no second walk that could emit a slightly different corpus and make the measured
    baseline describe something other than what `search` queries (rule 5).
    """
    surfaces = item_surfaces(
        item,
        transcribe_command=options.transcribe_command,
        vision_command=options.vision_command,
    )
    fingerprint = item_fingerprint(item, options=options)
    _write_item_metadata(index, item, fingerprint, options=options, counters=counters)
    _write_surfaces(index, surfaces, counters)

    chunks = chunk_surfaces(
        surfaces,
        params=options.params,
        topics=item_topics(item),
        url=item.url,
        blocks_by_surface_id=article_block_texts(item),
    )
    stored = index.add(chunks, created_at=item.created_at, source=item.source)
    counters.chunks += stored
    empty_text = len(chunks) - stored
    decorative, no_speech = _count_omissions(item)
    counters.empty_text += empty_text
    counters.decorative += decorative
    counters.no_speech += no_speech
    # Recorded ON THE ITEM'S ROW, so the manifest's `skipped` can be summed from the base
    # after an incremental update instead of carried over from the previous manifest (A-3).
    index.connection.execute(
        "UPDATE items SET skipped_empty_text = ?, skipped_decorative = ?, "
        "skipped_no_speech = ? WHERE item_id = ?",
        (empty_text, decorative, no_speech, item.id),
    )
    if index.add_profile(item.id, profile_text(item, list(vocab)), fingerprint):
        counters.profiles += 1


def _write_item_metadata(
    index: LexicalIndex,
    item: Item,
    fingerprint: str,
    *,
    options: IndexOptions,
    counters: WriteCounters,
) -> None:
    """The five filterable tables, from the SAME projection a consumer would receive.

    Built from `knowledge_item` rather than re-derived from the store, so `--author`,
    `--topic`, `--kind` and `has_surfaces` answer with exactly what `get` would show. A
    second derivation here is the divergence rule 5 is about, and the field that would go
    wrong first is `content_kinds`: a FAILED fetch has a kind, and listing it would tell a
    consumer to ask `get` for a body that does not exist.
    """
    projection = knowledge_item(item, vault_dir=options.vault_dir)
    index.connection.execute(
        "INSERT OR REPLACE INTO items (item_id, source, url, author_handle, author_name, "
        "created_at, captured_at, primary_topic, note_path, bookmark_folder, store_fingerprint) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            item.id,
            item.source,
            item.url,
            item.author.handle,
            item.author.name,
            item.created_at.isoformat(),
            item.captured_at.isoformat(),
            projection.primary_topic,
            projection.note_path,
            item.bookmark_folder,
            fingerprint,
        ),
    )
    for slug in projection.topics:
        index.connection.execute(
            "INSERT OR REPLACE INTO item_topics (item_id, slug, is_primary) VALUES (?,?,?)",
            (item.id, slug, int(slug == projection.primary_topic)),
        )
    for kind in projection.content_kinds:
        index.connection.execute(
            "INSERT OR REPLACE INTO item_content_kinds (item_id, kind) VALUES (?,?)",
            (item.id, kind),
        )
    for failure in projection.failed_sources:
        counters.failed_sources += 1
        index.connection.execute(
            "INSERT INTO source_failures (item_id, kind, url, failure_reason, error, "
            "http_status, attempts) VALUES (?,?,?,?,?,?,?)",
            (
                item.id,
                failure.kind,
                failure.url,
                failure.failure_reason,
                failure.error,
                failure.http_status,
                None,
            ),
        )
    for link in projection.unfetched_links:
        index.connection.execute(
            "INSERT INTO unfetched_links (item_id, url, reason, detail) VALUES (?,?,?,?)",
            (item.id, link.url, link.reason, link.detail),
        )


def _count_omissions(item: Item) -> tuple[int, int]:
    """`(decorative, no_speech)` — the two omissions that CAN be non-zero (spec §5.6).

    A decorative photo and a silent video are surfaces the emitter deliberately does not
    produce. Counting them here, rather than inferring them from a missing chunk, is what
    lets the manifest name the CAUSE instead of reporting a gap.
    """
    decorative = sum(
        1
        for entry in item.media
        if isinstance(entry, MediaPhotoDescribed) and (entry.is_decorative or not entry.description)
    )
    no_speech = sum(
        1
        for _index, source in iter_content_sources(item, {"x_video"})
        if source.has_speech is False
    )
    return decorative, no_speech


def write_topic(
    index: LexicalIndex,
    topic: Topic,
    page: TopicPage | None,
    primary_ids: tuple[str, ...],
    secondary_ids: tuple[str, ...],
    counters: WriteCounters,
    *,
    options: IndexOptions,
) -> None:
    """One topic's record and its three surfaces (spec §3.6)."""
    _write_topic_row(index, topic_record(topic, page, primary_ids, secondary_ids))
    surfaces = topic_surfaces(topic, page)
    _write_surfaces(index, surfaces, counters)
    chunks = chunk_surfaces(surfaces, params=options.params)
    stored = index.add(chunks)
    counters.chunks += stored
    counters.empty_text += len(chunks) - stored


def _write_surfaces(
    index: LexicalIndex, surfaces: Sequence[KnowledgeSurface], counters: WriteCounters
) -> None:
    for surface in surfaces:
        counters.surfaces += 1
        # The SAME projection `item_fingerprint` hashes (G-5): what is stored is what is
        # fingerprinted, by construction. LENGTH, never the body, in the last column: spec
        # §10.8 keeps articles out of derived stores, and `chunks.text` is where text lives.
        index.connection.execute(
            "INSERT OR REPLACE INTO surfaces (surface_id, owner_type, owner_id, surface_type, "
            "origin, trust_class, derived, attribution_handle, attribution_name, title, url, "
            "locator_json, language, fingerprint, char_length) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            surface_row(surface),
        )


def topic_membership(
    store: Mapping[str, Item], slug: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(primary_item_ids, secondary_item_ids)` for a topic, both sorted.

    Sorted because `TopicRecord` is fingerprinted downstream and an order following dict
    iteration would make two builds of the same store differ. A PRIMARY item is excluded from
    the secondary list rather than appearing in both: the two lists answer different
    questions, and double counting would inflate any membership figure taken from them.
    """
    primary = tuple(
        sorted(i.id for i in store.values() if i.enriched and i.enriched.primary_topic == slug)
    )
    secondary = tuple(
        sorted(
            i.id
            for i in store.values()
            if i.enriched and slug in i.enriched.topics and i.id not in primary
        )
    )
    return primary, secondary


def _write_topic_row(index: LexicalIndex, record: TopicRecord) -> None:
    """The one `INSERT` into `topics` — the full writer and the membership refresh share it."""
    index.connection.execute(
        "INSERT OR REPLACE INTO topics (slug, description, overview, notes_json, "
        "synthesized_at, post_count_at_synth, stale, primary_item_ids_json, "
        "secondary_item_ids_json, vocab_fingerprint, synthesis_fingerprint) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        topic_row(record),
    )


def stored_topic_rows(connection: sqlite3.Connection) -> dict[str, TopicRow]:
    """`{slug: the row the base holds}`, in `topic_row` column order — the comparison's left side."""
    return {
        row[0]: cast("TopicRow", tuple(row))
        for row in connection.execute(
            "SELECT slug, description, overview, notes_json, synthesized_at, "
            "post_count_at_synth, stale, primary_item_ids_json, secondary_item_ids_json, "
            "vocab_fingerprint, synthesis_fingerprint FROM topics"
        )
    }


def expected_topic_records(
    store: Mapping[str, Item], vocab: Sequence[Topic], topic_pages: Mapping[str, TopicPage]
) -> dict[str, TopicRecord]:
    """`{slug: the record the writer would insert NOW}` — the comparison's right side.

    Built exactly as `build` builds it: the page from `topics.json`, the members from the
    store, `stale` derived by `topic_record` from the live primary count. One derivation,
    consumed by the build, the refresh and `status`.
    """
    return {
        topic.slug: topic_record(
            topic, topic_pages.get(topic.slug), *topic_membership(store, topic.slug)
        )
        for topic in vocab
    }


def topic_rows_behind(
    connection: sqlite3.Connection,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
) -> list[str]:
    """The topics whose stored row is not the row the store implies, sorted (H1).

    Members, `stale`, description, synthesis — the whole row, compared with `topic_row`. A
    row that is missing from the base counts as behind. This is what `status` reports as
    `topics_changed` and what `update` rewrites when the vocabulary and pages did not move.
    """
    return _topics_behind(
        stored_topic_rows(connection), expected_topic_records(store, vocab, topic_pages)
    )


def _topics_behind(stored: Mapping[str, TopicRow], records: Mapping[str, TopicRecord]) -> list[str]:
    return sorted(slug for slug, record in records.items() if stored.get(slug) != topic_row(record))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build(
    index_dir: Path,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    items_path: Path,
    *,
    vocab_path: Path | None = None,
    topics_path: Path | None = None,
    signal: StoreSignal | None = None,
    options: IndexOptions | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> BuildReport:
    """Build `data/index/` from scratch. Read-only with respect to the store.

    `signal` IS THE CHEAP SIGNAL OF THE SNAPSHOT `store`/`vocab`/`topic_pages` WERE READ FROM
    (P1b), and the manifest is sealed with it: pass the one `load_index_inputs` returned
    beside the objects. Without it the three paths are stat'ed before the first write —
    which describes the files at that moment, not the objects — so omit it only when you
    wrote those files yourself a moment ago. The first version stat'ed `items_path` AFTER
    the commit, and a save landing between the caller's load and that stat produced a
    manifest certifying a store the base had never seen: `search` compared equal signals
    and answered over stale rows with nothing declared (the round-05 gate's probe A).

    ONE TRANSACTION, AND THE MANIFEST LAST — AND, ON A FORCED REBUILD, THE OLD MANIFEST
    REMOVED FIRST. A `Ctrl-C` or a full disk mid-build rolls the rows back and leaves no
    manifest — and an index with no manifest is REFUSED by every query rather than answered
    partially, so an interruption cannot produce a small index that looks valid (spec §9.3,
    Plan 02 §11). That sentence was only true for a FRESH build until C-1: `--force` kept the
    previous manifest standing while the new database was written, so an interrupted forced
    rebuild left a manifest that every query accepted over an empty base.

    A forced rebuild therefore does NOT preserve the previous index: the manifest and the
    database are both gone before the first row is written, and recovering from an
    interruption is `xbrain index build` again (which `status` names). Rebuilding over an
    existing index requires `force`, because a rebuild throws away something that may have
    taken minutes and the incremental path usually wants `index update` instead. The error
    names both.

    TWO THINGS FOUND BY MEASURING, NOT BY READING, and both are here:

    * `--dry-run` builds into `sqlite3(":memory:")` and touches NO FILE AT ALL. The first
      version opened the real database (creating it when absent), rolled back, and then
      removed the file it believed it had created — so a dry run against a working index
      DESTROYED it, from the flag whose whole promise is that it changes nothing;
    * `--force` UNLINKS the database before rebuilding instead of clearing the rows. Clearing
      in place left SQLite's freelist behind: on the real corpus a fresh build was 51.2 MB and
      the same index after five forced rebuilds was 66.5 MB, with a `VACUUM` recovering it
      only to 60.6 MB. A derived artefact whose size depends on how many times it has been
      rebuilt is one nobody can reason about.
    """
    options = options or IndexOptions()
    if manifest_path(index_dir).exists() and not force and not dry_run:
        raise ValueError(
            f"Ya existe un índice en {index_dir}. {UPDATE_ADVICE} "
            "Si de verdad quieres reconstruirlo desde cero, usa `xbrain index build --force`."
        )
    signal = _bound_signal(signal, items_path, vocab_path, topics_path)
    started = time.perf_counter()
    counters = WriteCounters()
    failed: list[dict[str, str]] = []

    if dry_run:
        connection = open_memory_index()
    else:
        # THE MANIFEST GOES FIRST (C-1). It is what every query trusts, so it must not
        # outlive the database it describes: with the old manifest standing while the new
        # database was being written, an interrupted `--force` rolled the rows back and left
        # a manifest whose versions and cheap signal still matched — `status` reported
        # nothing wrong and `search` answered "no results" over an EMPTY base, which is
        # indistinguishable from a corpus with no matches. Measured on the real corpus
        # (2,404 items, 2026-09-01). With the manifest gone first, the interrupted state is
        # the one this docstring promises: refused by every query, named by `status`.
        manifest_path(index_dir).unlink(missing_ok=True)
        db_path(index_dir).unlink(missing_ok=True)
        # The ONE caller allowed to create the file (G-2): every other door finds the
        # absence and names the command instead of leaving an empty base behind.
        connection = open_index(db_path(index_dir), create=True)
    try:
        with connection:  # a single transaction: commit on success, rollback on any exception
            _write_everything(
                LexicalIndex(connection), store, vocab, topic_pages, counters, options=options
            )
            tallies = manifest_tallies(connection)
            if dry_run:
                # A dry run does the whole walk and then throws it away, so the counts it
                # reports are the counts a real build WOULD produce — not an estimate.
                raise _DryRun
    except _DryRun:
        connection.close()
        return _build_report(
            counters, failed, started, items=len(store), topics=len(vocab), dry_run=True
        )
    finally:
        if not connection_closed(connection):
            connection.close()

    write_manifest(
        index_dir,
        _fresh_manifest(store, vocab, topic_pages, signal, tallies, failed, options=options),
    )
    return _build_report(
        counters, failed, started, items=len(store), topics=len(vocab), dry_run=False
    )


def _write_everything(
    index: LexicalIndex,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    counters: WriteCounters,
    *,
    options: IndexOptions,
) -> None:
    """Every item and every topic, in SORTED order, inside the caller's transaction.

    Sorted so two builds of the same store write the same rows in the same sequence — spec
    §3.7.8 needs that for the `chunk_id` tie-break to mean anything, since an order following
    dict iteration would reorder results between rebuilds of identical data.
    """
    for item_id in sorted(store):
        write_item(index, store[item_id], vocab, counters, options=options)
    for topic in sorted(vocab, key=lambda t: t.slug):
        primary, secondary = topic_membership(store, topic.slug)
        write_topic(
            index, topic, topic_pages.get(topic.slug), primary, secondary, counters, options=options
        )


@dataclass(frozen=True)
class ManifestTallies:
    """`counts` and `skipped` as the DATABASE holds them — the one source for both writers.

    A fresh build and an incremental update used to compute these differently: the build
    from its run counters, the update by carrying the previous manifest's `surfaces`,
    `skipped` and `failed` over and adjusting four of the five counts by hand — and the
    hand adjustment drifted (topic chunks were added on every rebuild and never subtracted).
    Reading them back from the rows means the manifest describes the base by construction,
    and a base that disagrees with its manifest can be DETECTED (C-3).
    """

    counts: dict[str, int]
    skipped: dict[str, int]


def manifest_tallies(connection: sqlite3.Connection) -> ManifestTallies:
    """What the manifest reports about the base, read from the base itself."""
    counts = count_rows(connection)
    summed = connection.execute(
        "SELECT COALESCE(SUM(skipped_empty_text), 0), COALESCE(SUM(skipped_decorative), 0), "
        "COALESCE(SUM(skipped_no_speech), 0) FROM items"
    ).fetchone()
    failed_sources = connection.execute("SELECT COUNT(*) FROM source_failures").fetchone()[0]
    return ManifestTallies(
        counts=counts,
        skipped={
            "empty_text": int(summed[0]),
            "decorative": int(summed[1]),
            "no_speech": int(summed[2]),
            "failed_sources": int(failed_sources),
        },
    )


def manifest_mismatch(manifest: Manifest, counts: Mapping[str, int]) -> str:
    """The planes on which the base disagrees with its manifest, as one sentence, or ``.

    Empty means consistent. Compared plane by plane rather than as one boolean so the
    error names WHAT is missing — `topics 0 != 45` is a diagnosis, `incomplete` is not.

    The planes iterated are the REQUIRED ones (`COUNT_PLANES`), never `manifest.counts`
    (B1): a `Manifest` holding `counts={}` used to compare nothing and agree with any base.
    The reader refuses that document now; the comparison fails closed on its own as well.
    """
    differing = [
        f"{plane} {counts.get(plane, 0)} != {manifest.counts.get(plane, '—')}"
        for plane in sorted(COUNT_PLANES)
        if counts.get(plane, 0) != manifest.counts.get(plane)
    ]
    return ", ".join(differing)


def _fresh_manifest(
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    signal: StoreSignal,
    tallies: ManifestTallies,
    failed: list[dict[str, str]],
    *,
    options: IndexOptions,
) -> Manifest:
    """The manifest a full build writes — every version taken from the CODE, not carried over.

    The opposite of `_next_manifest`, and deliberately so: a build is what DEFINES the
    versions the index was written under, while an update has already proved they match and
    must copy them rather than silently "fix" a mismatch that should have refused the run.
    """
    return Manifest(
        schema_version=SCHEMA_VERSION,
        built_at=datetime.now(timezone.utc),
        store_fingerprint=store_fingerprint(store, options=options),
        store_signal=signal,
        vocab_fingerprint=vocab_fingerprint(vocab),
        topics_fingerprint=topics_fingerprint(topic_pages),
        surface_version=SURFACE_VERSION,
        chunker_version=CHUNKER_VERSION,
        chunker_params=_params_dict(options.params),
        tokenize=FTS_TOKENIZE,
        connective=FTS_CONNECTIVE,
        counts=dict(tallies.counts),
        skipped=dict(tallies.skipped),
        failed=failed,
    )


class _DryRun(Exception):
    """Internal: unwinds the transaction after a dry run has counted the work."""


def connection_closed(connection: sqlite3.Connection) -> bool:
    """Whether the connection is already closed, without raising on the happy path."""
    try:
        connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def _params_dict(params: ChunkerParams) -> dict[str, int]:
    return {
        "target": params.target,
        "max_chars": params.max_chars,
        "overlap": params.overlap,
        "min_chars": params.min_chars,
    }


def _skipped(counters: WriteCounters) -> dict[str, int]:
    """The four causes spec §5.6 asks about, each counting what its name says.

    `empty_text` is structurally 0 today: the emitters drop a blank surface at `_blank`
    before the index ever sees it, so the counter can only move if a surface with a
    whitespace-only body ever reaches here. It is kept because the manifest shape is
    specified and because the path is real — and it is documented as 0-by-construction so
    nobody quotes it as a measurement of the corpus (rule 2).
    """
    return {
        "empty_text": counters.empty_text,
        "decorative": counters.decorative,
        "no_speech": counters.no_speech,
        "failed_sources": counters.failed_sources,
    }


def _build_report(
    counters: WriteCounters,
    failed: list[dict[str, str]],
    started: float,
    *,
    items: int,
    topics: int,
    dry_run: bool,
) -> BuildReport:
    return BuildReport(
        items_written=items,
        topics_written=topics,
        surfaces_written=counters.surfaces,
        chunks_written=counters.chunks,
        profiles_written=counters.profiles,
        skipped=_skipped(counters),
        failed=failed,
        duration_seconds=time.perf_counter() - started,
        dry_run=dry_run,
    )


def write_manifest(index_dir: Path, manifest: Manifest) -> None:
    """Write the manifest LAST. Its presence is what says a build completed.

    THE WRITER ROUND-TRIPS THROUGH THE READER (B1). The document is serialised, parsed and
    validated by `Manifest.from_dict` before one byte lands, so a build or an update cannot
    seal a manifest every later door would refuse — and, the other direction, a writer whose
    shape drifted from the reader's schema fails HERE, loudly, instead of producing a
    document the reader happens to accept while comparing less than it should.
    """
    document = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)
    Manifest.from_dict(json.loads(document))
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(index_dir).write_text(document, encoding="utf-8")


def load_manifest(index_dir: Path) -> Manifest:
    """Read the manifest, turning any malformed document into an ACTIONABLE error.

    A `JSONDecodeError` from inside a query is a traceback; spec §9.3 asks for an error that
    names the command that fixes it. Nothing is repaired here — Plan 02 §11: a corrupt base
    is rebuilt, never patched.
    """
    path = manifest_path(index_dir)
    if not path.exists():
        raise IndexMissingError(
            f"No hay manifest en {path}: el índice no está construido o quedó incompleto. "
            "Constrúyelo con `xbrain index build`."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IndexIncompatibleError(
            f"El manifest está corrupto ({error}). {REBUILD_ADVICE}"
        ) from error
    return Manifest.from_dict(raw)


def load_compatible_manifest(index_dir: Path, *, params: ChunkerParams | None = None) -> Manifest:
    """The manifest, REFUSED unless every version the code depends on matches (step 29).

    Spec §9.3: *manifest incompatible: no se consulta parcialmente.* Four checks, not three.
    The plan names the schema, the emitter and the chunker; the fourth is the chunker
    PARAMETERS, because Plan 02 §7 sweeps `target x overlap` and a sweep that lands on new
    parameters without bumping `CHUNKER_VERSION` produces chunks cut differently under
    IDENTICAL ids — the worst case, since the id resolves and the text behind it is not what
    it was.
    """
    manifest = load_manifest(index_dir)
    mismatches = []
    if manifest.schema_version != SCHEMA_VERSION:
        mismatches.append(f"schema_version {manifest.schema_version} != {SCHEMA_VERSION}")
    if manifest.surface_version != SURFACE_VERSION:
        mismatches.append(f"surface_version {manifest.surface_version} != {SURFACE_VERSION}")
    if manifest.chunker_version != CHUNKER_VERSION:
        mismatches.append(f"chunker_version {manifest.chunker_version} != {CHUNKER_VERSION}")
    if params is not None and manifest.chunker_params != _params_dict(params):
        mismatches.append(f"chunker_params {manifest.chunker_params} != {_params_dict(params)}")
    if mismatches:
        raise IndexIncompatibleError(
            "El índice fue construido con otra versión: "
            + "; ".join(mismatches)
            + f". {REBUILD_ADVICE}"
        )
    return manifest


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


# One literal per table. The f-string version was safe — the names came from a tuple in this
# module — but it made `bandit` report B608 and needed a suppression, and a suppression is a
# request to stop looking.
_COUNT_STATEMENTS: dict[str, str] = {
    "items": "SELECT COUNT(*) FROM items",
    "topics": "SELECT COUNT(*) FROM topics",
    "surfaces": "SELECT COUNT(*) FROM surfaces",
    "chunks": "SELECT COUNT(*) FROM chunks",
    "profiles": "SELECT COUNT(*) FROM profiles",
}


def count_rows(connection: sqlite3.Connection) -> dict[str, int]:
    """How many rows each plane holds — what `index status --json` reports (acceptance 2)."""
    return {
        table: int(connection.execute(statement).fetchone()[0])
        for table, statement in _COUNT_STATEMENTS.items()
    }


def _stored_fingerprints(connection: sqlite3.Connection) -> dict[str, str]:
    """`{item_id: the fingerprint the index was built from}` — the comparison's left side."""
    return {
        row["item_id"]: row["store_fingerprint"]
        for row in connection.execute("SELECT item_id, store_fingerprint FROM items")
    }


@dataclass(frozen=True)
class _Delta:
    """Which items are new, gone or different — the whole decision an update makes."""

    added: list[str]
    removed: list[str]
    changed: list[str]


def _classify(current: Mapping[str, str], stored: Mapping[str, str]) -> _Delta:
    """Compare the two fingerprint maps. Sorted, so an update is deterministic."""
    return _Delta(
        added=sorted(set(current) - set(stored)),
        removed=sorted(set(stored) - set(current)),
        changed=sorted(k for k in set(current) & set(stored) if current[k] != stored[k]),
    )


def _apply_update(
    connection: sqlite3.Connection,
    index: LexicalIndex,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    delta: _Delta,
    counters: WriteCounters,
    *,
    topics_rebuilt: bool,
    options: IndexOptions,
) -> tuple[int, int, int]:
    """Delete then rewrite, inside the CALLER'S transaction.

    Returns `(chunks gone, profiles gone, topic rows refreshed)`.

    THE VOCABULARY DRAGS THE ITEM PLANE WITH IT. The profile composes each assigned topic's
    DESCRIPTION (spec §5.1.A), so a `vocab.yaml` edit rewrites indexed text on every affected
    item — not only on the topic plane. Rebuilding the topic tables alone would leave the
    profiles quoting a description the vocabulary no longer holds, and nothing would say so.

    AND THE ITEMS DRAG THE TOPIC ROWS WITH THEM (H1). `topics` stores who the members are
    and whether the page is stale, and both are functions of the items' assignments — which
    `enrich` rewrites. When the plane is not rebuilt, the rows whose members or `stale` bit
    moved are rewritten through the same projection the full writer uses; the topic
    surfaces and chunks are left alone, because nothing they hold depends on membership.

    THE PAGES DRAG THE ITEM PLANE TOO, AND THAT IS A COST, NOT A NECESSITY (S-1, round 06).
    `topics_rebuilt` fuses the vocabulary and the page fingerprints, so a `topics.json`-only
    change rewrites every item as a `vocab.yaml` change must — measured on the real corpus,
    an update for one added topic note costs what a `build --force` costs (11.6–31.8 s
    depending on load) — although `profile_text` reads no `TopicPage`. What is served
    afterwards is correct and byte-identical to a rebuild (the gate checked six queries), so
    this is declared here rather than fixed here; separating the two triggers is the natural
    follow-up and touches no contract.
    """
    rewrite = sorted(store) if topics_rebuilt else delta.added + delta.changed
    deleted_chunks = 0
    deleted_profiles = 0
    for item_id in delta.removed + [i for i in rewrite if i not in delta.added]:
        deleted_chunks += _delete_item(connection, item_id)
        deleted_profiles += delete_profile_rows(connection, [item_id])
    for item_id in rewrite:
        write_item(index, store[item_id], vocab, counters, options=options)
    if topics_rebuilt:
        # Counted (N-1): the report's `chunks_deleted` omitted the topic plane, so after a
        # `topics.json`-only update it read `+22,287 / -21,583` while the base moved by one.
        deleted_chunks += _clear_topics(connection)
        for topic in sorted(vocab, key=lambda t: t.slug):
            primary, secondary = topic_membership(store, topic.slug)
            write_topic(
                index,
                topic,
                topic_pages.get(topic.slug),
                primary,
                secondary,
                counters,
                options=options,
            )
        return deleted_chunks, deleted_profiles, 0
    return deleted_chunks, deleted_profiles, _refresh_topic_rows(index, store, vocab, topic_pages)


def _refresh_topic_rows(
    index: LexicalIndex,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
) -> int:
    """Rewrite ONLY the topic rows the store no longer agrees with (H1). Returns how many.

    Compared before written — the same comparison `status` reports — so a store that did
    not move rewrites no row, and `update` with no changes stays at zero writes.
    """
    records = expected_topic_records(store, vocab, topic_pages)
    behind = _topics_behind(stored_topic_rows(index.connection), records)
    for slug in behind:
        _write_topic_row(index, records[slug])
    return len(behind)


def _update_report(
    delta: _Delta,
    counters: WriteCounters,
    deleted_chunks: int,
    deleted_profiles: int,
    topics_rebuilt: bool,
    topics_refreshed: int,
    started: float,
    *,
    dry_run: bool,
) -> UpdateReport:
    return UpdateReport(
        items_added=len(delta.added),
        items_changed=len(delta.changed),
        items_removed=len(delta.removed),
        chunks_inserted=counters.chunks,
        chunks_deleted=deleted_chunks,
        profiles_inserted=counters.profiles,
        profiles_deleted=deleted_profiles,
        topics_rebuilt=topics_rebuilt,
        topics_refreshed=topics_refreshed,
        duration_seconds=time.perf_counter() - started,
        dry_run=dry_run,
    )


def _next_manifest(
    previous: Manifest,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    signal: StoreSignal,
    tallies: ManifestTallies,
    *,
    options: IndexOptions,
) -> Manifest:
    """The manifest after an update: NEW signals and tallies, the versions carried over.

    The versions are copied rather than recomputed because `load_compatible_manifest` has
    already proved they match — recomputing them here would silently "fix" a mismatch that
    was supposed to have refused the run. The COUNTS AND OMISSIONS ARE NOT COPIED (A-3):
    the first version carried `surfaces`, `skipped` and `failed` over and adjusted the other
    four by hand, so after one update the manifest published the previous population and
    `index status --json` exposed it as current. They are read from the base now, through
    the same function a fresh build uses.
    """
    return Manifest(
        schema_version=previous.schema_version,
        built_at=datetime.now(timezone.utc),
        store_fingerprint=store_fingerprint(store, options=options),
        store_signal=signal,
        vocab_fingerprint=vocab_fingerprint(vocab),
        topics_fingerprint=topics_fingerprint(topic_pages),
        surface_version=previous.surface_version,
        chunker_version=previous.chunker_version,
        chunker_params=previous.chunker_params,
        tokenize=previous.tokenize,
        connective=previous.connective,
        embeddings=previous.embeddings,
        counts=dict(tallies.counts),
        skipped=dict(tallies.skipped),
        failed=previous.failed,
    )


def _status_advice(
    incomplete: bool,
    delta: _Delta,
    *,
    behind: bool,
    unusable: str = "",
    topics_changed: int = 0,
) -> str:
    """The command that fixes what `status` just found — never a bare diagnosis.

    `unusable` is the sentence for a manifest that EXISTS but cannot be used — another
    version, a malformed document, or a base that does not hold what it declares (C-3). It
    already names `index build --force`, and it must, because plain `index build` refuses
    while a manifest exists: the previous advice sent the operator into a dead end.
    """
    if unusable:
        return unusable
    if incomplete:
        return (
            "El índice está incompleto o no tiene manifest: ninguna consulta lo usará. "
            "Constrúyelo con `xbrain index build`."
        )
    if delta.added or delta.removed or delta.changed or behind or topics_changed:
        return UPDATE_ADVICE
    return ""


def update(
    index_dir: Path,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    items_path: Path,
    *,
    vocab_path: Path | None = None,
    topics_path: Path | None = None,
    signal: StoreSignal | None = None,
    options: IndexOptions | None = None,
    dry_run: bool = False,
) -> UpdateReport:
    """Bring the index up to date, touching only what changed (spec §5.6).

    `signal` is the cheap signal of the snapshot the objects were read from, exactly as in
    `build` (P1b): the manifest an update writes is sealed with it, never with a stat taken
    after the commit.

    ONE TRANSACTION for the whole run. Committing per item would leave a partial application
    of a change nobody can name after a failure — and the index would look fine, because
    every id it holds still resolves.
    """
    options = options or IndexOptions()
    manifest = load_compatible_manifest(index_dir, params=options.params)
    signal = _bound_signal(signal, items_path, vocab_path, topics_path)
    started = time.perf_counter()

    # BEFORE the write door (G-2): an update over a database that is not there has nothing
    # to be incremental over, and opening for writing used to create it — an empty base
    # under a standing manifest, which the next `search` answered as an empty corpus.
    database = require_database(index_dir)
    connection = open_index(database)
    counters = WriteCounters()
    deleted_chunks = 0
    deleted_profiles = 0
    topics_refreshed = 0
    try:
        # The maintenance door pays the whole-file check (D-1): an update re-seals the
        # manifest, and it must not seal it over a torn page its `COUNT(*)` never read.
        require_consistent(connection, manifest, database, whole_file=True)
        index = LexicalIndex(connection)
        with reading_base(database):
            stored = _stored_fingerprints(connection)
        current = {
            item_id: item_fingerprint(item, options=options) for item_id, item in store.items()
        }
        delta = _classify(current, stored)

        topics_rebuilt = (
            vocab_fingerprint(vocab) != manifest.vocab_fingerprint
            or topics_fingerprint(topic_pages) != manifest.topics_fingerprint
        )
        try:
            with reading_base(database), connection:
                deleted_chunks, deleted_profiles, topics_refreshed = _apply_update(
                    connection,
                    index,
                    store,
                    vocab,
                    topic_pages,
                    delta,
                    counters,
                    topics_rebuilt=topics_rebuilt,
                    options=options,
                )
                tallies = manifest_tallies(connection)
                if dry_run:
                    raise _DryRun
        except _DryRun:
            return _update_report(
                delta,
                counters,
                deleted_chunks,
                deleted_profiles,
                topics_rebuilt,
                topics_refreshed,
                started,
                dry_run=True,
            )
    finally:
        if not connection_closed(connection):
            connection.close()

    write_manifest(
        index_dir,
        _next_manifest(manifest, store, vocab, topic_pages, signal, tallies, options=options),
    )
    return _update_report(
        delta,
        counters,
        deleted_chunks,
        deleted_profiles,
        topics_rebuilt,
        topics_refreshed,
        started,
        dry_run=False,
    )


@dataclass(frozen=True)
class BaseVerdict:
    """The seam's answer: what the base holds, and why it is not what the manifest says (or ``).

    `counts` is what was read before the answer was reached — `{}` when the base could not
    be read at all — so `status` can publish it next to the sentence without a second read.
    """

    counts: dict[str, int]
    sentence: str


def describe_base(
    connection: sqlite3.Connection, manifest: Manifest, database: Path, *, whole_file: bool
) -> BaseVerdict:
    """«Does this manifest describe this base?» — ONE function, asked by every door (round 06).

    Six rounds closed the fail-open family one route at a time: an interrupted forced
    rebuild (C-1), a missing table (C-2), a manifest that counted less than the base held
    (C-3), a dry run that created an empty base (G-2), a cheap signal covering one input of
    three (P1a), a manifest whose `counts` was `{}` (B1) — and a damaged page under the
    maintenance reads was a raw traceback on the three commands (D-1). Each fix was right
    and each door kept its own reading of the question, which is CLAUDE.md rule 5: one
    definition, or five that silently diverge. This is the one definition. `status` reports
    its sentence as advice, `search` and `update` raise it (`require_consistent`), and
    `tests/test_knowledge_index_invalidation.py` replaces it with a sentinel and asserts the
    three doors repeat it verbatim, so a door that re-derives the question goes red.

    Three things, in order, and any `DatabaseError` raised by any of them IS the answer:

    1. `whole_file` — `PRAGMA quick_check`, the reader that sees a page none of the open
       door's probes touch (B-1). `status` and `update` pay it (155–850 ms on the 52 MB real
       index depending on load, §8 of `docs/knowledge-index.md`): one is the instrument an
       operator runs to find out, the other re-seals the manifest and must not seal it over
       a torn page — measured in round 06, `update --dry-run` returned a normal report with
       the root page of `chunks` overwritten, because `COUNT(*)` was answered from an index.
       `search` does not pay it, by the B-1 decision: a query fails closed the moment it
       reaches the page (`LexicalIndex._fetch`, G-4).
    2. The five `COUNT(*)` against the five planes the manifest is REQUIRED to declare
       (`manifest_mismatch`, 0.04 ms).
    3. The conversion: `count_rows` on a damaged root page raises `sqlite3.DatabaseError`,
       and before round 06 it escaped `status`, `search` and `update` as a traceback naming
       no command while CLAUDE.md said G-4 had closed exactly that. The sentence is
       `corrupt_base_error`'s — the same one the open door uses.
    """
    try:
        if whole_file:
            damage = quick_check(connection)
            if damage:
                return BaseVerdict(counts={}, sentence=_damage_advice(database, damage))
        counts = count_rows(connection)
    except sqlite3.DatabaseError as error:
        return BaseVerdict(counts={}, sentence=str(corrupt_base_error(database, error)))
    mismatch = manifest_mismatch(manifest, counts)
    if mismatch:
        return BaseVerdict(
            counts=counts,
            sentence=(
                f"El índice no contiene lo que su manifest declara ({mismatch}): quedó "
                f"incompleto. {REBUILD_ADVICE}"
            ),
        )
    return BaseVerdict(counts=counts, sentence="")


def require_consistent(
    connection: sqlite3.Connection, manifest: Manifest, database: Path, *, whole_file: bool
) -> dict[str, int]:
    """`describe_base` as a refusal: raise its sentence, closing the connection first.

    C-3: `update` decided `topics_rebuilt` against the MANIFEST's fingerprints and never
    looked at the base, so a manifest declaring 45 topics over a base holding 0 produced an
    update that rewrote every item and silently dropped the topic plane for good — no later
    update would find a fingerprint to disagree with. The manifest is the baseline an
    incremental update reasons from; when the base contradicts it there is no baseline, and
    the honest answer is the rebuild.

    Public since G-2 because `search` runs it too (`index_store.open_for_query`): it used to
    compare versions and schema only, so a base amputated behind the manifest's back — or
    the empty one a stray write door left behind — was answered as a corpus with no matches
    while `update` and `status` refused it. Returns the counts it read, so a caller does not
    count twice.
    """
    verdict = describe_base(connection, manifest, database, whole_file=whole_file)
    if verdict.sentence:
        connection.close()
        raise IndexIncompatibleError(verdict.sentence)
    return verdict.counts


def _delete_item(connection: sqlite3.Connection, item_id: str) -> int:
    """Every chunk and every metadata row of one item. Returns the chunks removed."""
    chunk_ids = [
        row["chunk_id"]
        for row in connection.execute(
            "SELECT chunk_id FROM chunks WHERE owner_type = 'item' AND owner_id = ?", (item_id,)
        )
    ]
    removed = delete_chunk_rows(connection, chunk_ids)
    delete_item_rows(connection, [item_id])
    return removed


def _clear_topics(connection: sqlite3.Connection) -> int:
    """Remove the whole topic plane. Returns the chunks removed, so the report can count them."""
    chunk_ids = [
        row["chunk_id"]
        for row in connection.execute("SELECT chunk_id FROM chunks WHERE owner_type = 'topic'")
    ]
    removed = delete_chunk_rows(connection, chunk_ids)
    connection.execute("DELETE FROM surfaces WHERE owner_type = 'topic'")
    connection.execute("DELETE FROM topics")
    return removed


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _status_manifest(index_dir: Path, options: IndexOptions) -> tuple[Manifest | None, str]:
    """The manifest as `status` sees it: the document, and why the code cannot use it (or ``).

    Applies the SAME compatibility check `search` and `update` apply. Without it `status`
    answered `incomplete=False` over a manifest both of them refuse — two instruments,
    opposite answers on one state (rule 9) — and its advice named plain `index build`, which
    refuses while a manifest exists.
    """
    try:
        manifest = load_manifest(index_dir)
    except IndexMissingError:
        return None, ""
    except IndexIncompatibleError as error:
        return None, str(error)
    try:
        load_compatible_manifest(index_dir, params=options.params)
    except IndexIncompatibleError as error:
        return manifest, str(error)
    return manifest, ""


def _index_contents(
    index_dir: Path, manifest: Manifest | None, unusable: str
) -> tuple[dict[str, int], dict[str, str], dict[str, TopicRow], str]:
    """`(row counts per plane, {item_id: stored fingerprint}, {slug: stored topic row}, unusable)`.

    Three empties when there is no base: with nothing stored, every item is "added" and every
    topic is "behind", which is the truthful reading of an index that does not exist.

    With a usable manifest the base is judged by `describe_base` FIRST — `quick_check`, then
    the five counts, any `DatabaseError` converted (D-1) — and a base the manifest does not
    describe is read no further: the deep reads below would raise on the same damage, and
    the delta they feed is meaningless against a base that has to be rebuilt. A manifest the
    code cannot use (another version) still gets its counts and its delta, because the base
    itself is readable and the operator may want to know how far it moved.
    """
    database = db_path(index_dir)
    if not database.exists():
        return {}, {}, {}, unusable
    connection = open_index(database, read_only=True)
    try:
        with reading_base(database):
            if manifest is not None and not unusable:
                verdict = describe_base(connection, manifest, database, whole_file=True)
                if verdict.sentence:
                    return verdict.counts, {}, {}, verdict.sentence
                counts = verdict.counts
            else:
                counts = count_rows(connection)
            return (
                counts,
                _stored_fingerprints(connection),
                stored_topic_rows(connection),
                unusable,
            )
    finally:
        connection.close()


def _damage_advice(database: Path, damage: str) -> str:
    """The B-1 sentence for `quick_check`'s first finding."""
    return f"La base del índice en {database} está dañada (quick_check: {damage}). {REBUILD_ADVICE}"


def status(
    index_dir: Path,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    items_path: Path,
    *,
    vocab_path: Path | None = None,
    topics_path: Path | None = None,
    options: IndexOptions | None = None,
) -> StatusReport:
    """What the index holds, and how far behind the store it is (acceptance 2, step 10c).

    `status` is an EXPLICIT command, so it can afford what a query cannot: loading the store
    and computing a fingerprint per item. That is what lets it answer *how many* items
    changed instead of merely *something did* — and the difference decides whether a rebuild
    is worth its minutes.

    And it runs `PRAGMA quick_check` over the whole file (B-1, round 05): the open door's
    probes read page 1, `sqlite_master` and one `MATCH` per FTS plane, so damage on a page
    none of them touches — measured: 16 KB of `0xff` over pages 17–20 of the real index — was
    reported by `quick_check` and not by `status`. A query still fails closed the moment it
    reaches the page (G-4); this is the explicit command paying the whole-file check —
    155–167 ms (gate) to a 425 ms median at load 8.6 (round 05) on the 52 MB real index —
    so the operator hears it first.

    It takes the vocabulary and the pages like `build` and `update` do (H1), because the
    topic plane is derived from all three, and it reads the TOPIC ROWS back from the base:
    `topics_changed` counts the topics whose stored members, `stale` bit, description or
    synthesis are not what the store implies now. An index whose item fingerprints all
    match can still be behind on that plane — the pre-H1 `update` left it that way on every
    topic move — and an instrument that only compared item fingerprints called it healthy.
    """
    options = options or IndexOptions()
    manifest, unusable = _status_manifest(index_dir, options)
    counts, stored, stored_topics, unusable = _index_contents(index_dir, manifest, unusable)

    current = {item_id: item_fingerprint(item, options=options) for item_id, item in store.items()}
    delta = _classify(current, stored)
    topics_changed = len(
        _topics_behind(stored_topics, expected_topic_records(store, vocab, topic_pages))
    )
    behind = manifest is not None and manifest.store_signal != StoreSignal.of(
        items_path, vocab_path, topics_path
    )
    incomplete = manifest is None or bool(unusable)
    return StatusReport(
        manifest=manifest,
        counts=counts,
        items_added=len(delta.added),
        items_changed=len(delta.changed),
        items_removed=len(delta.removed),
        topics_changed=topics_changed,
        behind=behind,
        incomplete=incomplete,
        advice=_status_advice(
            incomplete, delta, behind=behind, unusable=unusable, topics_changed=topics_changed
        ),
    )
