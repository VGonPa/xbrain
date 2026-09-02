"""The three inputs of the index, the signal bound to them, and their fingerprints
(Plan 02 §2, §3).

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
from dataclasses import dataclass
from pathlib import Path

from xbrain.executors.api import iter_content_sources
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams
from xbrain.knowledge.ids import SURFACE_VERSION, surface_fingerprint
from xbrain.knowledge.models import KnowledgeSurface
from xbrain.knowledge.surfaces import (
    CONTENT_KIND_TO_SURFACE_TYPES,
    item_surfaces,
    item_topics,
)
from xbrain.models import Item, Topic, TopicPage
from xbrain.rubrics import parse_vocab
from xbrain.store import parse_store, parse_topic_pages


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


def item_fingerprint(item: Item, *, options: IndexOptions | None = None) -> str:
    """sha256 over everything about this item that the INDEX holds.

    Two halves, and both are needed. The SURFACE ROWS answer *did what the index holds about
    each surface change?* — `surface_row` is the projection of every column the `surfaces`
    table holds, so it covers the surface fingerprint (emitter version, type, origin, body)
    AND the attribution, title, url, locator and language the index stores and `search`
    serves on every match (A-1). The filterable METADATA of the item (source, author, date,
    topics, content kinds) answers the other half: a changed author changes what `--author`
    returns even though not one character of text moved.

    THE ROW, NOT THE SURFACE FINGERPRINT ALONE (G-5). `surface_fingerprint` is
    `(version, type, origin, text)` by design and must stay so; but hashing only that here
    meant a `refresh-quoted` that filled in the author of a quoted post without touching
    its body left `update` reporting `0 cambiados` and `search` serving the old attribution
    — the evidence repaired, the derivative standing (CLAUDE.md rule 6), on the attribution
    rule this repo paid for in blood. Hashing the projection itself is what keeps "every
    stored column is hashed" ONE list instead of two — see `surface_row` for what still
    binds that list to the persisted schema by hand today, and what 02.7 owes it.

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


# The column order of `surfaces`, as ONE tuple type: hashed here, and bound to the `INSERT`
# by the writer 02.7 lands.
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
    """What the index STORES about a surface — the projection of one `surfaces` row.

    ONE PROJECTION, AND 02.7's WRITER HAS TO CONSUME IT. `item_fingerprint` hashes this
    tuple today; the writer that binds it to the `INSERT` into `surfaces` lands in 02.7 and
    does not exist in this tree. Until it does, the correspondence with the persisted DDL
    (`index_schema`: fifteen columns, this tuple's order and types) is kept BY HAND —
    nothing here forces a column added to `surfaces` to be hashed, and this docstring is the
    contract, not the guard.

    Making it structural is 02.7's job, in two parts: its writer binds THIS function instead
    of assembling a second tuple, and it writes the readback test — red first — that proves
    the stored row and the hashed row cannot drift. Read any claim of that guard here as a
    statement of 02.7's obligation, never as one already discharged.

    The last column is a LENGTH, never the body (spec §10.8); the body is covered by
    `fingerprint`, which hashes it.
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
