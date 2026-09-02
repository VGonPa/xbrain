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
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path

from xbrain.executors.api import iter_content_sources
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams
from xbrain.knowledge.ids import CHUNKER_VERSION, SURFACE_VERSION, surface_fingerprint
from xbrain.knowledge.index_schema import (
    REBUILD_ADVICE,
    SCHEMA_VERSION,
    IndexIncompatibleError,
    IndexMissingError,
    manifest_path,
)
from xbrain.knowledge.lexical_fts import FTS_CONNECTIVE, FTS_TOKENIZE
from xbrain.knowledge.models import KnowledgeSurface, TopicRecord
from xbrain.knowledge.surfaces import (
    CONTENT_KIND_TO_SURFACE_TYPES,
    item_surfaces,
    item_topics,
)
from xbrain.models import Item, Topic, TopicPage
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

    A MISSING file reads as its empty value and a zero signal, exactly as `load_store`,
    `load_vocab`, `load_topic_pages` and `StoreSignal.of` treat it. A file that exists and
    cannot be read RAISES (A-2): an unreadable store is not an empty one, and every door
    that loads through here closes on the error instead of answering over nothing.
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

    `(None, 0, 0)` for an ABSENT or unnamed file — and for nothing else (A-2, round 08).
    The first version caught every `OSError`, so a file that EXISTS and cannot be read — a
    `chmod 000`, a directory standing in its place, an `EIO` from a failing mount — loaded
    as the empty store reserved for a missing one: the cheap signal still stat'ed fine, so
    no door saw anything wrong, and on the real index `status` reported `items_removed
    2404` as healthy, `search` answered zero results with exit 0, `update` planned the
    deletion of every item and `build --force` replaced 22,286 chunks with the topic plane's
    703, sealed consistent, exit 0. `load_store` (`store.py`) only ever read an absent file
    as `{}`; this loader now agrees, and any other `OSError` propagates — the CLI prints it
    as `Error: …` with exit 1, and the index on disk is not touched.
    """
    if path is None:
        return None, 0, 0
    try:
        handle = path.open("rb")
    except FileNotFoundError:
        return None, 0, 0
    with handle:
        stat = os.fstat(handle.fileno())
        data = handle.read()
    return data.decode("utf-8"), stat.st_mtime_ns, stat.st_size


@dataclass(frozen=True)
class IndexOptions:
    """Everything a build needs that is not the corpus itself.

    The configured transcribe/vision commands no longer travel here (F7-7, round 08): they
    were stamped on the ASR/VLM surfaces as `producer`, a provenance claim the store cannot
    back, and the emitter no longer takes them. See `surfaces.item_surfaces`.
    """

    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS
    vault_dir: Path | None = None


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
    def from_dict(cls, raw: object) -> Manifest:
        """The document, TOTALLY validated — every nested key, every value's type (B1).

        The first version checked the top-level key set and cast what sat under it, so a
        manifest whose `counts` was `{}` loaded as compatible, and `manifest_mismatch`,
        iterating whatever `counts` offered, compared nothing: `status` called an amputated
        base healthy, `search` answered zero results over it with `no_embeddings` and nothing
        else, and `update` sealed the state as sound (the round-06 gate, reproduced). Spec
        §9.3: an incompatible manifest is never queried partially — and a manifest that
        declares less than the schema is incompatible, not lenient. Closed as well as total:
        an undeclared plane or cause is refused, because nothing could compare it.

        THE DOCUMENT IS A JSON OBJECT BEFORE IT IS ANYTHING ELSE. The totality check was
        `MANIFEST_FIELDS - set(raw)`, and `set()` of a non-mapping is not an error — it is
        the ELEMENTS. A hand-edited `manifest.json` holding the JSON LIST
        `["schema_version", "built_at", …]` therefore passed the guard with `missing`
        empty, because a list of the field NAMES has exactly the field names as its
        elements, and the first subscript below raised `TypeError: list indices must be
        integers` — out of `load_manifest`, out of `load_compatible_manifest`, out of every
        door, as a traceback. Spec §9.3 asks for an actionable error; a `TypeError` from
        inside a query is the shape this reader exists to remove. A top-level `3`, `null`
        or `true` was worse still: `set()` of them raises `TypeError: not iterable` from
        the guard LINE ITSELF, before one field was read. The type is checked first, so
        every non-object document leaves by the same door as every other malformed one.
        """
        if not isinstance(raw, Mapping):
            raise IndexIncompatibleError(
                f"El manifest no es un objeto JSON, es {type(raw).__name__}. {REBUILD_ADVICE}"
            )
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

    What is NOT hashed, and why: `producer`. The index has no producer column, and the
    producers the store records (`enriched.executor`, `description_version`) travel with
    the surface `get` re-emits; the ASR/VLM surfaces carry none since round 08 (F7-7),
    because the store records no transcriber and a configured command is not evidence.

    Deliberately NOT `(item_id, content.fetched_at, enriched.enriched_at)`: see the module
    docstring for why a timestamp proxy both misses hand edits and cannot reach the 40 % of
    the corpus with no `content` at all.
    """
    options = options or IndexOptions()
    surfaces = item_surfaces(item)
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


def _params_dict(params: ChunkerParams) -> dict[str, int]:
    return {
        "target": params.target,
        "max_chars": params.max_chars,
        "overlap": params.overlap,
        "min_chars": params.min_chars,
    }


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

    Spec §9.3: *manifest incompatible: no se consulta parcialmente.* Six checks, not three.
    The plan names the schema, the emitter and the chunker; the fourth is the chunker
    PARAMETERS, because Plan 02 §7 sweeps `target x overlap` and a sweep that lands on new
    parameters without bumping `CHUNKER_VERSION` produces chunks cut differently under
    IDENTICAL ids — the worst case, since the id resolves and the text behind it is not what
    it was. The fifth and sixth are the TOKENIZER and the CONNECTIVE (F7-8, round 07): the
    manifest recorded them «because they decide every recall number» and no door compared
    them — `tokenize: porter` / `connective: AND` were accepted by all three on the real
    index. The tokenizer is baked into the FTS DDL, so a base built under one and queried
    under another is the M-1 shape with no guard; the connective moved recall@10 from
    0.1429 to 0.8099 in Plan 01 M3.
    """
    manifest = load_manifest(index_dir)
    mismatches = []
    # `!r` on every manifest string (T-1, round 08): the manifest is hand-editable, and the
    # three version strings were interpolated raw, so a newline in `schema_version` stood at
    # column 0 of `index status` as a forged header and an ESC reached the TTY through the
    # sentence `search` and `update` raise. A repr carries neither.
    if manifest.schema_version != SCHEMA_VERSION:
        mismatches.append(f"schema_version {manifest.schema_version!r} != {SCHEMA_VERSION!r}")
    if manifest.surface_version != SURFACE_VERSION:
        mismatches.append(f"surface_version {manifest.surface_version!r} != {SURFACE_VERSION!r}")
    if manifest.chunker_version != CHUNKER_VERSION:
        mismatches.append(f"chunker_version {manifest.chunker_version!r} != {CHUNKER_VERSION!r}")
    if params is not None and manifest.chunker_params != _params_dict(params):
        mismatches.append(f"chunker_params {manifest.chunker_params} != {_params_dict(params)}")
    if manifest.tokenize != FTS_TOKENIZE:
        mismatches.append(f"tokenize {manifest.tokenize!r} != {FTS_TOKENIZE!r}")
    if manifest.connective != FTS_CONNECTIVE:
        mismatches.append(f"connective {manifest.connective!r} != {FTS_CONNECTIVE!r}")
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
