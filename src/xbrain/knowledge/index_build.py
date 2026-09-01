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
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

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
    manifest_path,
    open_index,
    open_memory_index,
    require_database,
)
from xbrain.knowledge.lexical import LexicalIndex
from xbrain.knowledge.lexical_fts import FTS_CONNECTIVE, FTS_TOKENIZE
from xbrain.knowledge.models import KnowledgeSurface
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


@dataclass(frozen=True)
class StoreSignal:
    """The CHEAP change signal: `mtime_ns` and size of `data/items.json` (spec §5.6).

    One `os.stat`, so a query can afford it on every call. A missing store yields zeros
    rather than raising: a query must still be able to say *the index is behind* when the
    store has been moved away, and raising from inside `search` is the wrong place to learn
    it.
    """

    items_json_mtime_ns: int
    items_json_size: int

    @classmethod
    def of(cls, items_path: Path) -> StoreSignal:
        try:
            stat = items_path.stat()
        except OSError:
            return cls(items_json_mtime_ns=0, items_json_size=0)
        return cls(items_json_mtime_ns=stat.st_mtime_ns, items_json_size=stat.st_size)

    def to_dict(self) -> dict[str, int]:
        return {
            "items_json_mtime_ns": self.items_json_mtime_ns,
            "items_json_size": self.items_json_size,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, int]) -> StoreSignal:
        return cls(
            items_json_mtime_ns=int(raw.get("items_json_mtime_ns", 0)),
            items_json_size=int(raw.get("items_json_size", 0)),
        )


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
        missing = MANIFEST_FIELDS - set(raw)
        if missing:
            raise IndexIncompatibleError(
                f"El manifest no declara {sorted(missing)}. {REBUILD_ADVICE}"
            )
        return cls(
            schema_version=str(raw["schema_version"]),
            built_at=datetime.fromisoformat(str(raw["built_at"])),
            store_fingerprint=str(raw["store_fingerprint"]),
            store_signal=StoreSignal.from_dict(_mapping(raw["store_signal"])),
            vocab_fingerprint=str(raw["vocab_fingerprint"]),
            topics_fingerprint=str(raw["topics_fingerprint"]),
            surface_version=str(raw["surface_version"]),
            chunker_version=str(raw["chunker_version"]),
            chunker_params={k: int(v) for k, v in _mapping(raw["chunker_params"]).items()},
            tokenize=str(raw["tokenize"]),
            connective=str(raw["connective"]),
            embeddings=cast("dict[str, object] | None", raw["embeddings"]),
            counts={k: int(v) for k, v in _mapping(raw["counts"]).items()},
            skipped={k: int(v) for k, v in _mapping(raw["skipped"]).items()},
            failed=cast("list[dict[str, str]]", raw["failed"]),
        )


def _mapping(value: object) -> dict[str, Any]:
    """A manifest sub-object, or an ACTIONABLE error instead of an `AttributeError`.

    A hand-edited manifest is exactly the input this reader has to survive, and spec §9.3
    asks for a stable error rather than a traceback from inside a query.
    """
    if not isinstance(value, dict):
        raise IndexIncompatibleError(f"El manifest tiene un campo malformado. {REBUILD_ADVICE}")
    return value


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
    record = topic_record(topic, page, primary_ids, secondary_ids)
    index.connection.execute(
        "INSERT OR REPLACE INTO topics (slug, description, overview, notes_json, "
        "synthesized_at, post_count_at_synth, stale, primary_item_ids_json, "
        "secondary_item_ids_json, vocab_fingerprint, synthesis_fingerprint) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            topic.slug,
            record.description.text,
            record.overview.text if record.overview else None,
            json.dumps([note.text for note in record.notes], ensure_ascii=False),
            record.synthesized_at.isoformat() if record.synthesized_at else None,
            record.post_count_at_synth,
            int(record.stale),
            json.dumps(list(primary_ids)),
            json.dumps(list(secondary_ids)),
            record.vocab_fingerprint,
            record.synthesis_fingerprint,
        ),
    )
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
    options: IndexOptions | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> BuildReport:
    """Build `data/index/` from scratch. Read-only with respect to the store.

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
        _fresh_manifest(store, vocab, topic_pages, items_path, tallies, failed, options=options),
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
    """
    differing = [
        f"{plane} {counts.get(plane, 0)} != {declared}"
        for plane, declared in sorted(manifest.counts.items())
        if counts.get(plane, 0) != declared
    ]
    return ", ".join(differing)


def _fresh_manifest(
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    items_path: Path,
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
        store_signal=StoreSignal.of(items_path),
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
    """Write the manifest LAST. Its presence is what says a build completed."""
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(index_dir).write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
) -> tuple[int, int]:
    """Delete then rewrite, inside the CALLER'S transaction. Returns `(chunks, profiles)` gone.

    THE VOCABULARY DRAGS THE ITEM PLANE WITH IT. The profile composes each assigned topic's
    DESCRIPTION (spec §5.1.A), so a `vocab.yaml` edit rewrites indexed text on every affected
    item — not only on the topic plane. Rebuilding the topic tables alone would leave the
    profiles quoting a description the vocabulary no longer holds, and nothing would say so.
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
        _clear_topics(connection)
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
    return deleted_chunks, deleted_profiles


def _update_report(
    delta: _Delta,
    counters: WriteCounters,
    deleted_chunks: int,
    deleted_profiles: int,
    topics_rebuilt: bool,
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
        duration_seconds=time.perf_counter() - started,
        dry_run=dry_run,
    )


def _next_manifest(
    previous: Manifest,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    items_path: Path,
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
        store_signal=StoreSignal.of(items_path),
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


def _status_advice(incomplete: bool, delta: _Delta, *, behind: bool, unusable: str = "") -> str:
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
    if delta.added or delta.removed or delta.changed or behind:
        return UPDATE_ADVICE
    return ""


def update(
    index_dir: Path,
    store: Mapping[str, Item],
    vocab: Sequence[Topic],
    topic_pages: Mapping[str, TopicPage],
    items_path: Path,
    *,
    options: IndexOptions | None = None,
    dry_run: bool = False,
) -> UpdateReport:
    """Bring the index up to date, touching only what changed (spec §5.6).

    ONE TRANSACTION for the whole run. Committing per item would leave a partial application
    of a change nobody can name after a failure — and the index would look fine, because
    every id it holds still resolves.
    """
    options = options or IndexOptions()
    manifest = load_compatible_manifest(index_dir, params=options.params)
    started = time.perf_counter()

    # BEFORE the write door (G-2): an update over a database that is not there has nothing
    # to be incremental over, and opening for writing used to create it — an empty base
    # under a standing manifest, which the next `search` answered as an empty corpus.
    connection = open_index(require_database(index_dir))
    require_consistent(connection, manifest)
    index = LexicalIndex(connection)
    stored = _stored_fingerprints(connection)
    current = {item_id: item_fingerprint(item, options=options) for item_id, item in store.items()}
    delta = _classify(current, stored)

    topics_rebuilt = (
        vocab_fingerprint(vocab) != manifest.vocab_fingerprint
        or topics_fingerprint(topic_pages) != manifest.topics_fingerprint
    )
    counters = WriteCounters()
    deleted_chunks = 0
    deleted_profiles = 0

    try:
        with connection:
            deleted_chunks, deleted_profiles = _apply_update(
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
        connection.close()
        return _update_report(
            delta, counters, deleted_chunks, deleted_profiles, topics_rebuilt, started, dry_run=True
        )
    finally:
        if not connection_closed(connection):
            connection.close()

    write_manifest(
        index_dir,
        _next_manifest(manifest, store, vocab, topic_pages, items_path, tallies, options=options),
    )
    return _update_report(
        delta, counters, deleted_chunks, deleted_profiles, topics_rebuilt, started, dry_run=False
    )


def require_consistent(connection: sqlite3.Connection, manifest: Manifest) -> None:
    """Refuse a base that does not hold what its manifest declares. Closes the connection.

    C-3: `update` decided `topics_rebuilt` against the MANIFEST's fingerprints and never
    looked at the base, so a manifest declaring 45 topics over a base holding 0 produced an
    update that rewrote every item and silently dropped the topic plane for good — no later
    update would find a fingerprint to disagree with. The manifest is the baseline an
    incremental update reasons from; when the base contradicts it there is no baseline, and
    the honest answer is the rebuild.

    Public since G-2 because `search` runs it too (`index_store.open_for_query`): it used to
    compare versions and schema only, so a base amputated behind the manifest's back — or
    the empty one a stray write door left behind — was answered as a corpus with no matches
    while `update` and `status` refused it. Five `COUNT(*)`, 0.04 ms on the 52 MB real
    index, and the three instruments say one thing.
    """
    mismatch = manifest_mismatch(manifest, count_rows(connection))
    if mismatch:
        connection.close()
        raise IndexIncompatibleError(
            f"El índice no contiene lo que su manifest declara ({mismatch}): quedó "
            f"incompleto. {REBUILD_ADVICE}"
        )


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


def _clear_topics(connection: sqlite3.Connection) -> None:
    chunk_ids = [
        row["chunk_id"]
        for row in connection.execute("SELECT chunk_id FROM chunks WHERE owner_type = 'topic'")
    ]
    delete_chunk_rows(connection, chunk_ids)
    connection.execute("DELETE FROM surfaces WHERE owner_type = 'topic'")
    connection.execute("DELETE FROM topics")


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


def _index_contents(index_dir: Path) -> tuple[dict[str, int], dict[str, str]]:
    """`(row counts per plane, {item_id: stored fingerprint})`, or two empties with no base."""
    if not db_path(index_dir).exists():
        return {}, {}
    connection = open_index(db_path(index_dir), read_only=True)
    try:
        return count_rows(connection), _stored_fingerprints(connection)
    finally:
        connection.close()


def _mismatch_advice(manifest: Manifest, counts: Mapping[str, int]) -> str:
    """The C-3 sentence, or `` when the base holds what the manifest declares."""
    mismatch = manifest_mismatch(manifest, counts)
    if not mismatch:
        return ""
    return (
        f"El índice está incompleto: la base no contiene lo que el manifest declara "
        f"({mismatch}). {REBUILD_ADVICE}"
    )


def status(
    index_dir: Path,
    store: Mapping[str, Item],
    items_path: Path,
    *,
    options: IndexOptions | None = None,
) -> StatusReport:
    """What the index holds, and how far behind the store it is (acceptance 2, step 10c).

    `status` is an EXPLICIT command, so it can afford what a query cannot: loading the store
    and computing a fingerprint per item. That is what lets it answer *how many* items
    changed instead of merely *something did* — and the difference decides whether a rebuild
    is worth its minutes.
    """
    options = options or IndexOptions()
    manifest, unusable = _status_manifest(index_dir, options)
    counts, stored = _index_contents(index_dir)
    if manifest is not None and not unusable:
        unusable = _mismatch_advice(manifest, counts)

    current = {item_id: item_fingerprint(item, options=options) for item_id, item in store.items()}
    delta = _classify(current, stored)
    behind = manifest is not None and manifest.store_signal != StoreSignal.of(items_path)
    incomplete = manifest is None or bool(unusable)
    return StatusReport(
        manifest=manifest,
        counts=counts,
        items_added=len(delta.added),
        items_changed=len(delta.changed),
        items_removed=len(delta.removed),
        behind=behind,
        incomplete=incomplete,
        advice=_status_advice(incomplete, delta, behind=behind, unusable=unusable),
    )
