"""The three inputs of the index read as ONE snapshot, and the DEEP fingerprints of all
FOUR planes — item, store, vocabulary and topics (Plan 02 §2, §3; spec §5.6).

TWO SIGNALS, AND THE WHOLE DESIGN IS THAT THEY COST DIFFERENT THINGS. Indexing is MANUAL BY
DECISION (spec §9.2), so the failure that actually happens is not corruption — it is *you ran
`enrich` and did not reindex*.

- The CHEAP one, `StoreSignal`, is `mtime_ns` and size of the THREE inputs: three `os.stat`,
  cheap enough for EVERY query, answering *an input moved*. It cannot say WHICH items.
- The DEEP ones, `item_fingerprint` and `store_fingerprint`, walk the corpus and emit every
  surface to answer *which items changed*, and `vocab_fingerprint` and `topics_fingerprint`
  answer the same of the other two inputs. All four are paid ONLY by `build`/`update`/`status`.
  No path here makes a cheap reader pay a deep one, and none may — 02.6a1's contract.

THE FOUR PLANES ARE FOUR BECAUSE THEY MOVE APART, and the vocabulary proves it: a description
edit rewrites rows that `item_fingerprint`, which takes no vocabulary, cannot see (the rows and
the measurement are under `vocab_fingerprint`). One fused signal would rebuild everything for a
topic-note typo or miss that, so `Manifest` seals the four in four separate fields (02.6b).

AND THE MANIFEST IS THE CONTRACT, NOT A WRITER. `Manifest` / `write_manifest` / `load_manifest`
/ `load_compatible_manifest` define the document those values are sealed into and refused by;
nothing in this tree CALLS them, because the caller is `index build`, which is 02.7's. That
ordering is deliberate and is the argument #161 made: the coverage gaps that review named were
closed BEFORE anything could seal a manifest — free in this tree, expensive once one exists.

The cheap signal can give false positives (a `touch` with no edit) and that is accepted: a
false positive costs one warning, a false negative costs serving stale evidence as fresh. It
fails towards the warning, the same direction `origin: unknown -> llm_synthesis` fails.

THE SIGNAL DESCRIBES THE BYTES THAT WERE PARSED, NOT THE PATH. `load_index_inputs` reads each
input through its OWN handle and takes the signal from `os.fstat` of that handle, BEFORE the
read; see its docstring for the failure that shape exists to close.

NOTHING HERE WRITES TO THE STORE. The three inputs are read and never touched, and THIS module
ships no command and takes no snapshot — it only reads (`data/index/` is derived and
reconstructible, spec §5.6). Said of this module, not of the plan: `build --force` DOES destroy a
good index, and round 08 caught it doing exactly that at exit 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from xbrain.executors.api import iter_content_sources
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams, chunk_surfaces
from xbrain.knowledge.ids import CHUNKER_VERSION, SURFACE_VERSION
from xbrain.knowledge.index_schema import (
    REBUILD_ADVICE,
    SCHEMA_VERSION,
    IndexIncompatibleError,
    IndexMissingError,
    db_path,
    manifest_path,
    open_index,
    open_memory_index,
)
from xbrain.knowledge.lexical import LexicalIndex
from xbrain.knowledge.models import KnowledgeSurface, TopicRecord
from xbrain.knowledge.profile import profile_text
from xbrain.knowledge.surfaces import (
    article_block_texts,
    failed_sources,
    item_content_kinds,
    item_surfaces,
    item_topics,
    knowledge_item,
    topic_record,
    topic_surfaces,
    unfetched_links,
)
from xbrain.models import Item, MediaPhotoDescribed, Topic, TopicPage
from xbrain.rubrics import parse_vocab
from xbrain.store import parse_store, parse_topic_pages

UNSTATTABLE = (-1, -1)
"""What a query reads for an input it cannot stat, IMPOSSIBLE ON A REAL `os.stat`, and that is
the whole contract. `st_size` is a byte count and never negative, while a pre-epoch `st_mtime_ns`
IS negative (measured) — so the SIZE is the half that makes the pair unforgeable. And
`_read_bound` RAISES on every obstruction producing it, so no snapshot can ever SEAL it."""


@dataclass(frozen=True)
class StoreSignal:
    """The CHEAP change signal: `mtime_ns` and size of the THREE inputs (spec §5.6, P1a).

    Three `os.stat`, so a query can afford it on every call. A missing file yields zeros
    rather than raising: a query must still say *the index is behind* when the store has been
    moved away, and `search` is the wrong place to learn it by exception. An input obstructed
    for ANY OTHER reason reads as `UNSTATTABLE` and never as those zeros — `_stat_signal`.

    THREE FILES, NOT ONE (P1a, gate Codex round 05). Spec §5.6 names `data/items.json`, and
    that is what the first version stat'ed — but the index derives from `vocab.yaml` and
    `topics.json` too: a topic description enters every assigned item's PROFILE (spec §5.1.A),
    and overviews and notes are chunks the index serves. The query door compared `items.json`
    alone, so `xbrain topics` — which writes `topics.json` and never `items.json` — left every
    later `search` answering over the old topic plane with nothing declared, the silent
    staleness spec §9.3 forbids, on two of the three inputs.

    ALL SIX FIELDS ARE REQUIRED, AND THAT IS THE FIX, NOT AN OVERSIGHT. They carried
    `= 0` defaults so a caller could build a two-input signal; zeros are also what an ABSENT
    file reads as, so a signal that omitted the vocabulary was byte-identical to one taken
    over a vocabulary that is not there. Two such signals compare EQUAL forever, however
    `vocab.yaml` changes — the round-05 defect above, reinstated by one missing argument, in
    the false-negative direction this signal is built never to fail in. A reader that must
    supply a legacy zero (a manifest written before the vocabulary and the topic pages were
    watched) states it at ITS OWN seam, where it knows the entry was absent from the
    persisted record rather than absent from the caller's mind.
    """

    items_json_mtime_ns: int
    items_json_size: int
    vocab_yaml_mtime_ns: int
    vocab_yaml_size: int
    topics_json_mtime_ns: int
    topics_json_size: int

    @classmethod
    def of(cls, items_path: Path, vocab_path: Path, topics_path: Path) -> StoreSignal:
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
        """The six fields, each under its own key — the half a manifest records (02.6b)."""
        return {name: getattr(self, name) for name in SIGNAL_FIELDS}

    @classmethod
    def from_dict(cls, raw: object) -> StoreSignal:
        """The signal as a manifest recorded it — VALIDATED, never cast, never rehydrated.

        ALL SIX ARE REQUIRED AND THERE IS NO LEGACY RECORD TO BE LENIENT TOWARDS. Filling an
        omitted entry with `0` declares that input ABSENT, and two manifests filled that way
        compare EQUAL forever however the input moves — the round-05 defect reinstated by
        omission, in the false-negative direction this signal exists never to fail in. The
        implementation this replaces did exactly that: its four vocabulary/topic fields carried
        `= 0` and `from_dict` read required-vs-optional off those defaults. Nothing in this tree
        has ever SEALED a manifest, so tolerating a short one buys compatibility with nothing;
        Plan 02 §2 already says what a document of another version gets, and it is a refusal.

        THE SIZE AND THE MTIME ARE NOT VALIDATED ALIKE, AND THE ASYMMETRY IS THE CONTRACT.
        `st_mtime_ns` IS negative for a pre-epoch file (measured), so refusing every negative
        would make a legitimate corpus unbuildable. `st_size` never is, which is the half that
        makes `UNSTATTABLE` unforgeable — and a hand-edited `manifest.json` is the ONE path
        that bypasses `_read_bound`'s raise, so a sealed `-1` size would meet the live `-1` of
        an obstructed input and certify the index current over a file nobody can stat. This is
        where that is closed, because this is the only boundary a hand edit crosses.

        `type(v) is int` and not `isinstance`: a JSON `true` is an `int` to `isinstance` and is
        not a size, `int("17599352")` accepts a string, and a float compares unequal to what a
        live stat returns — so a coerced value would declare the index behind for a reason no
        operator could see.
        """
        values = _closed_keys(raw, "store_signal", frozenset(SIGNAL_FIELDS))
        checked: dict[str, int] = {}
        for name in SIGNAL_FIELDS:
            value = values[name]
            if type(value) is not int:
                raise _malformed("store_signal", f"{name!r} debe ser un entero, es {value!r}")
            if name in SIGNAL_SIZE_FIELDS and value < 0:
                raise _malformed(
                    "store_signal", f"{name!r} debe ser un tamaño no negativo, es {value!r}"
                )
            checked[name] = value
        return cls(**checked)


SIGNAL_FIELDS: tuple[str, ...] = tuple(f.name for f in dataclass_fields(StoreSignal))
"""The six field names IN ORDER, read off the dataclass so the schema has ONE definition."""

SIGNAL_SIZE_FIELDS: frozenset[str] = frozenset(n for n in SIGNAL_FIELDS if n.endswith("_size"))
"""The three that a real `os.stat` can never report negative — see `StoreSignal.from_dict`."""


def _stat_signal(path: Path) -> tuple[int, int]:
    """`(mtime_ns, size)` of one input: zeros when ABSENT, `UNSTATTABLE` when OBSTRUCTED.

    EVERY `OSError` IS SWALLOWED HERE, and the breadth is the contract, not laziness: this is
    the function a query pays on every call, and its whole promise is that a query can always
    ANSWER — declaring the index behind — instead of learning about the filesystem by
    exception from inside `search`. Measured, the reachable ones are a path standing INSIDE a
    regular file (`NotADirectoryError`, `ENOTDIR`), a symlink loop (`ELOOP`) and a PARENT
    directory whose permissions were dropped (`PermissionError`, `EACCES`); an `EIO` from a
    failing mount is the same shape with no way to stage it here.

    BUT ABSENCE IS NOT ONE OF THEM, AND SPLITTING IT OFF IS THE ROUND-09 FIX (Codex, HIGH). Both
    read `(0, 0)` before. `_read_bound` SEALS zeros for an ABSENT input, so an index built while
    `items.json` was missing compared EQUAL to a query taken once that same path could no longer
    be stat'ed: it certifies itself current over an input it never opened, and *I could not stat
    it* is never evidence that nothing moved. Absence keeps the zeros — a manifest's legacy zero
    still reads as absence — and every other `OSError` answers `UNSTATTABLE`. The warning again.

    THE ZEROS STILL COLLIDE WITH ONE REAL STATE, the harmless one: a file that EXISTS, is EMPTY
    and carries an `mtime_ns` of exactly 0 stats as `(0, 0)` (measured). It takes a deliberate
    `os.utime(path, ns=(0, 0))` — no writer here emits a zero-byte input (2, 11 and 2 bytes,
    measured) — and both readings mean the same downstream: `parse_vocab("")` is `[]` either way,
    and `parse_store("")` RAISES rather than passing as an empty store. Left alone: separating it
    would need a sentinel a real stat CAN produce, which is what the one above is not.

    WHAT DOES NOT REACH THIS `except`, and it is worth knowing which: `stat` needs neither
    read permission on the file nor the file to be a file, so a `chmod 000` FILE and a
    directory standing in its place both stat FINE. They are exactly the obstacles that reach
    `_read_bound` instead, which is why the test that holds this breadth had to be built on
    `ENOTDIR` and not on either of those. Nor does a path carrying an embedded NUL: it raises
    `ValueError` before any syscall, and it is the one shape that escapes the promise above.

    `_read_bound` is the DELIBERATE opposite and the pair is the design: what the loader
    cannot read is an error, because an unreadable store is not an empty one.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return 0, 0
    except OSError:
        return UNSTATTABLE
    return stat.st_mtime_ns, stat.st_size


@dataclass(frozen=True)
class IndexInputs:
    """The three inputs of the index AND the cheap signal of the snapshot they were read from.

    The signal travels WITH the objects because it describes them (P1b): taken from the path
    at any other moment it describes whatever file is there then, which is what let a manifest
    certify an `items.json` its own base had never seen.
    """

    store: dict[str, Item]
    vocab: list[Topic]
    topic_pages: dict[str, TopicPage]
    signal: StoreSignal


def load_index_inputs(items_path: Path, vocab_path: Path, topics_path: Path) -> IndexInputs:
    """Read the three inputs and return them WITH the signal of the bytes that were read.

    THE SIGNAL IS BOUND TO THE SNAPSHOT, NOT TO THE PATH (P1b, gate Codex round 05). The shape
    this closes: a caller loads the store, commits rows from it, and only then seals
    `StoreSignal.of(items_path)` — a `stat` of whatever file the path points at by then. A
    save landing in that window puts the rows under the OLD objects and the signal under the
    NEW file's mtime and size, so a later query compares EQUAL and answers over stale rows
    with nothing declared. The gate's probe A: `raceonlytoken` in the file, not in the rows,
    `degraded: ("no_embeddings",)`, `items_changed=1`, `behind=False`.

    Every file is read through ITS OWN HANDLE and the signal is `os.fstat` of that handle,
    taken BEFORE the read. Both halves matter and they fail differently. The HANDLE closes the
    atomic case: the store's writers replace files (`os.replace`), so an open handle keeps the
    inode it opened, the bytes parsed are that inode's, and the replacement leaves the PATH on
    a newer inode that query-time `StoreSignal.of` reports as different — the index declares
    itself behind. BEFORE closes the in-place case: `save_vocab` rewrites through
    `write_text`, which truncates the inode the reader is holding, so a stat taken after the
    read would describe bytes this loader never parsed and seal them as the snapshot; taken
    before, it is older than the content, the comparison is unequal, and the index is again
    declared behind. Same direction, the warning.

    A MISSING file reads as its empty value and a zero signal, exactly as `load_store`,
    `load_vocab`, `load_topic_pages` and `StoreSignal.of` treat it. A file that EXISTS and
    cannot be read RAISES (A-2) — see `_read_bound`.
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


def _read_bound(path: Path) -> tuple[str | None, int, int]:
    """`(text, mtime_ns, size)` of one input, the stat taken on the handle the text came from.

    `(None, 0, 0)` FOR AN ABSENT TARGET, AND FOR NOTHING ELSE (A-2, round 08) — a dangling
    symlink is that same case and not a further one, since `open` resolves it and raises the very
    `ENOENT` an absent path raises, exactly as all three doors read it. The first version
    caught every `OSError` and answered `(None, 0, 0)`, so a file that EXISTS and cannot be
    read — a `chmod 000`, a directory standing in its place, an `EIO` from a failing mount —
    loaded as the empty store reserved for a missing one: the cheap signal still stat'ed fine,
    so no door saw anything wrong, and on the real index `status` reported `items_removed
    2404` as healthy, `search` answered zero results with exit 0, `update` planned the deletion
    of every item and `build --force` replaced 22,286 chunks with the topic plane's 703, sealed
    consistent, exit 0. Every other `OSError` propagates to the caller unswallowed. Turning it
    into `Error: <file>` and exit 1 is the door's job and no door in this tree loads through
    here yet: read that as the obligation the first consumer owes, never as behaviour shipped.

    THIS LOADER IS STRICTER THAN THE THREE DOORS, ON EXACTLY TWO `OSError` SHAPES, DELIBERATELY.
    `load_store` / `load_vocab` / `load_topic_pages` gate on `Path.exists()`, which answers False
    for more than a missing file: it swallows exactly `pathlib`'s `_IGNORED_ERRNOS` — `ENOENT`,
    `ENOTDIR`, `EBADF`, `ELOOP` — and RE-RAISES the rest, `PermissionError` (`EACCES`) included
    (measured on CPython 3.12.11, the version CI pins). Two of those four are where this loader
    parts company: a path inside a regular file (`ENOTDIR`) and a symlink loop (`ELOOP`) read back
    as `{}` / `[]` / `{}` through those doors, while opening either RAISES here. Under `EACCES`
    there is nothing to argue — the door raises too. That is the A-2 direction — an unreadable
    input is not an empty one — so the divergence is the feature. It is also not hypothetical:
    `ENOTDIR` is the obstruction this module's own `_stat_signal` test builds, and it asserts
    this raise. A THIRD shape parts them and is NOT an `OSError`, which is why the count above
    is scoped: `Path.exists()` also answers False on a `ValueError`, so a path with an embedded
    NUL reads as `{}` / `[]` / `{}` through the doors while both halves here raise (measured).

    A file that exists and is not UTF-8 raises `UnicodeDecodeError`, which is a `ValueError`
    and not an `OSError` at all, so it is outside the `FileNotFoundError` guard by type as
    well as by intent — and there the three doors agree, because they decode too. It is a
    REFUSAL, not a repair: decoding with `errors="replace"` would turn undecodable bytes into
    U+FFFD and index them as if they were the corpus, which is the fail-open family this whole
    module exists to close, one layer lower.
    """
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

    CARRIED INERT IN THIS CHILD, AND SAYING SO IS THE POINT. Neither field is read ANYWHERE in
    this tree, so `item_fingerprint(item, options=X)` silently discards `X` and a caller who
    passed the chunker's parameters expecting them to be covered would be wrong. The dataclass
    travels so the signature 02.7 consumes is already the ported one, never because anything
    reads it; two persisted columns are unreachable for exactly this reason and are named in
    `item_fingerprint` (`items.note_path` needs `vault_dir`, `items.skipped_empty_text` needs
    `params`). The tests pin the inertness by BEHAVIOUR — two different `IndexOptions` hashing
    alike — so 02.7's first consumer cannot land without reddening that test on purpose.
    """

    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS
    vault_dir: Path | None = None


# The column order of `surfaces`, as ONE tuple type: hashed by `item_fingerprint`, and the
# tuple 02.7's writer is meant to bind its `INSERT` to.
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

    ONE PROJECTION, AND 02.7's WRITER HAS TO CONSUME IT — WHICH IT DOES NOT YET.
    `item_fingerprint` hashes this tuple today; the writer that binds it to the `INSERT` lands
    in 02.7, and no writer exists in this tree at all, so the correspondence with the persisted
    DDL (`index_schema._SCHEMA`: fifteen columns, this order and these types) is kept BY HAND.
    What ships instead of that force is a totality test reading the column names straight out
    of the DDL: it catches a column ADDED, and cannot catch one REORDERED into a same-typed
    neighbour. Making the binding structural is 02.7's — its writer binds THIS function rather
    than assembling a second tuple, and writes the readback test, red first. Read any claim of
    that guard here as 02.7's obligation, never as one discharged.

    The last column is a LENGTH, never the body (spec §10.8), which `fingerprint` hashes.
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


def declined_media(item: Item) -> tuple[int, int]:
    """`(decorative, no_speech)` — the two omissions the `items` row COUNTS (spec §5.6).

    A decorative photo and a silent video are surfaces the emitter deliberately does not
    produce, and `items.skipped_decorative` / `items.skipped_no_speech` are where the index
    records that it declined them rather than that it found nothing.

    HASHED, BECAUSE OTHERWISE THEY MOVE UNSEEN. Both are persisted columns and pure functions
    of the item, so leaving them out reproduces HIGH-1's shape one plane over: `xbrain describe`
    classifying a photo as decorative flips `skipped_decorative` 0 -> 1 while emitting NO
    surface and changing no other hashed atom — the row on disk moves and `update` reports the
    item unchanged (rule 6). DEFINED HERE ONCE SO 02.7 CONSUMES IT: deriving the pair a second
    time next to the `UPDATE` is rule 5's five-hands divergence.

    The third counter, `skipped_empty_text`, is NOT here and cannot be: it is
    `len(chunks) - stored`, under the chunker parameters that reach this module only through
    the inert `IndexOptions`. See `item_fingerprint`.
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


def _model_atoms(model: BaseModel) -> list[list[object]]:
    """One pydantic projection as `[[field_name, value], ...]`, in field-definition order.

    STRUCTURAL ON PURPOSE, and the difference between this child and the review that named
    HIGH-1. A hand-written field list is what let `source_failures` and `unfetched_links` change
    on disk while every fingerprint stood still; walking `model_fields` means a field ADDED to
    `SourceFailure` or `UnfetchedLink` enters the hash the moment it exists, with nobody having
    to remember, and the NAME is hashed beside the value so a rename moves it too. It fails
    CLOSED on a field this encoder cannot represent: `json.dumps` raises `TypeError` on a
    `datetime` or a `set`, loudly, at the first build, rather than dropping it.
    """
    return [[name, getattr(model, name)] for name in type(model).model_fields]


def _persisted_atoms(model: BaseModel, *, mode: Literal["python", "json"]) -> list[list[object]]:
    """One pydantic projection as `[[field_name, value], ...]`, READ OUT OF THE WRITER'S DUMP.

    THE KEY SET COMES FROM THE DUMP, NOT FROM A HUMAN, so the hashed atom set is EQUAL to the
    persisted one BY CONSTRUCTION: a field added to a model enters the hash the moment it is
    persisted. The hand lists this replaces were HIGH-1 of review #161 reintroduced.

    THE MODE IS THE WRITER'S, AND REQUIRED, BECAUSE THE TWO WRITERS DISAGREE (B3):
    `save_topic_pages` dumps json, `save_vocab` PYTHON, and hashing one projection while the
    writer persists the other is that same false negative one level down. Measured on a
    `datetime | str` field: json renders BOTH sides to `2026-01-20T00:00:00Z` while `save_vocab`
    writes a bare timestamp for one and a quoted string for the other — two `vocab.yaml` files
    under one digest. Python mode keeps the type, so `_canonical` REFUSES what its declared
    domain cannot hold rather than hash a rendering nobody stored. ORDER is the MODEL's, not the
    file's: `save_topic_pages` sorts keys on disk, `save_vocab` keeps declaration order.
    """
    return [[name, value] for name, value in model.model_dump(mode=mode).items()]


def item_fingerprint(item: Item, *, options: IndexOptions | None = None) -> str:
    """sha256 over everything about this item that the INDEX PERSISTS.

    Five planes, because five tables carry a row keyed by this item and each can move alone:

    - **`surfaces`** — the SURFACE ROWS. `surface_row` is every column that table holds, so
      this covers the surface fingerprint AND the attribution, title, url, locator and
      language `search` serves on every match (A-1).
    - **`items`** / **`item_topics`** / **`item_content_kinds`** — the filterable METADATA. A
      changed author changes what `--author` returns with no text moved.
    - **`source_failures`** and **`unfetched_links`** — HIGH-1 of review #161, and the reason
      this child exists. Both are written per item and read back by `get`, and before this
      neither moved any fingerprint: a link that started returning 404, or a fetch that began
      failing, rewrote the row on disk while `update` reported the item unchanged.
    - **`chunks`** — the BLOCK PARTITION of every X Article, as the ordered LENGTHS of its
      `ArticleTextBlock` bodies, keyed by `surface_id` (never by position: `fetch` rewrites
      `content.sources`). `ContentSourceSuccess` validates `text == "".join(blocks)`, so the
      flattened body is a function of the partition and NOT the reverse: two block lists that
      concatenate alike leave `surface_row` and `surface.fingerprint` identical while
      `chunk_surfaces(..., blocks_by_surface_id=article_block_texts(item))` — called today by
      `cli.py` and `evaluation.py` — cuts different `chunk_id`s, offsets, bodies and
      `chunks_fts` rows. Measured 2026-09-03 on the live store: **41** items carry usable
      blocks and **41 of 41** carry more than one. LENGTHS, never the bodies: the
      concatenation is already hashed by `surface.fingerprint`, so only the cuts were
      missing, and a second copy of the text is what spec §10.8 forbids.

    THE ROW, NOT THE SURFACE FINGERPRINT ALONE (G-5). `surface_fingerprint` is
    `(version, type, origin, text)` by design and must stay so; hashing only that here meant a
    `refresh-quoted` that filled in a quoted post's author without touching its body left
    `update` reporting `0 cambiados` and `search` serving the old attribution — rule 6, on the
    attribution rule this repo paid for in blood. `producer` is NOT hashed (no producer column;
    the producers travel with the surface `get` re-emits, F7-7), and neither is any timestamp:
    a timestamp claims when something was written, never what it says.

    `primary_topic` IS ITS OWN ATOM, beside the topics tuple and not folded into it: it is a
    persisted `items` column AND a persisted `item_topics.is_primary` flag, and `item_topics()`
    puts the primary first and then DEDUPLICATES, so `primary_topic=None, topics=["a", "b"]`
    and `primary_topic="a", topics=["b"]` produce the identical tuple `("a", "b")` while the
    stored column reads `NULL` against `"a"`. Both states load from a real `items.json`;
    measured 2026-09-03 (2,404 items, sha256 `f76341a3...`), **0** sit in that class — what
    keeps them absent is `guardrails.yaml`, which the model does not enforce, so read the zero
    as today's corpus and never as an invariant. 02.7 inherits the second half: the two
    obsolete writers derive `item_topics.is_primary` differently (slug-vs-`primary_topic`, and
    position 0 with a `or topics[0]` fallback) and disagree on exactly the falsy-primary states
    this atom separates. A fingerprint that distinguishes two states the writer stores alike is
    worse than one that distinguishes neither, so collapsing them is 02.7's, not a detail.

    THREE KNOWN FALSE POSITIVES, ALL IN THE DIRECTION THIS MODULE FAILS IN ON PURPOSE — and
    this said ONE until someone reordered the other two and watched the hash move. The topics
    region keeps the order `Enrichment.topics` was written in while `item_topics` on disk is a
    SET keyed `(item_id, slug)`, so a re-enrichment returning the same topics in a different
    order re-hashes an item whose rows do not move (2,073 of 2,404 items carry more than one
    topic). The failures and links regions are order-sensitive the same way, and neither
    `source_failures` nor `unfetched_links` has a primary key or an order column either;
    `fetch` rewriting `content.sources` is how the first of those becomes reachable. All three
    cost one wasted rewrite and never a stale row — the trade the cheap signal already makes,
    and the reason the count being wrong was cheap and being unstated would not have been.
    `kinds` needs no such trade: `surfaces.item_content_kinds` — the ONE derivation
    `knowledge_item` also reads — deduplicates, so the region and the rows are the same object.

    THE VARIADIC REGIONS ARE NESTED, NEVER SPLICED. Flattening them into one delimited list is
    NOT injective: `topics=("thread",)` with no sources serialised exactly like no topics with
    one blank `thread` source, so two item states hashed alike and `update` called the item
    unchanged — rule 6, failing OPEN. Each region is its own JSON array, so the boundary is
    STRUCTURAL, and `_canonical` makes the atoms inside them unforgeable too.

    **WHAT THIS STILL CANNOT REACH, named rather than left to be discovered.** None is a defect
    of the encoding; all are inputs this child does not have.

    - `items.note_path` — `surfaces._note_path` stats the VAULT under `IndexOptions.vault_dir`,
      so generating a note moves that column while nothing about the item does. 02.6b / 02.7.
    - `items.skipped_empty_text` — `len(chunks) - stored`, the chunker's arithmetic under the
      inert `IndexOptions.params`; structurally 0 today, and "0 today" is not "covered". 02.7.
    - `CHUNKER_VERSION` (`ids.py:43`) and `ChunkerParams` — the first is stamped into every
      `chunk_id` and `chunks.fingerprint`, the second decides where every span falls (measured,
      `800/0` against `400/0`: 5 rows against 9 over one body). A bump of either rewrites the
      whole `chunks` plane with this fingerprint unmoved, and that is CORRECT: a manifest
      refusing the query outright beats per-item invalidation. `ids.py:42` names
      `load_compatible_manifest` as its home and 02.6b landed it, below in this file: it
      compares BOTH the version and the parameters. Still uncalled here — 02.7 wires it.
    - `profiles.profile_text` — a `vocab.yaml` edit splices each assigned topic's DESCRIPTION
      into it (spec §5.1.A) and rewrites `profiles`/`profiles_fts` for every assigned item while
      this fingerprint, which takes no vocabulary, cannot move: DISCHARGED by `vocab_fingerprint`
      below, via 02.7's rebuild. Its OTHER half is NOT, and filing it there would record a
      debt under an owner who cannot discharge it — `profile.py:_titles` gates on
      `if source.title`, so a title on a blank-bodied source reaches the profile while the
      emitter produces nothing (a whitespace-only `summary`/`digest` is the same shape: profile
      on truthiness, emitter on `_blank()`, which strips). Constructible, 0 of 2,404 today, and
      02.7's — the only writer that sees both sides.

    `items.store_fingerprint` IS THIS VALUE, which is why it appears in neither list above: a
    hash cannot be inside itself. The NAME is historical and is a trap for 02.7's writer — the
    only writer that ever existed put a PER-ITEM digest in that column
    (`b61e04b:index_build.py:760-761`), never a store-level one.

    Read this fingerprint as *what changed about the item*, never as *what changed about its
    indexed rows*.

    `source_failures.attempts` USED TO BE ON THAT LIST and is not any more: it was the only
    column across both failure planes with no field on the knowledge projection this hashes,
    and `index_schema.SCHEMA_VERSION` "4" drops it rather than adding the field. The reasoning,
    and the test that stops either side moving alone, are recorded there.

    `options` IS NOT READ HERE. It is accepted so the signature 02.7 consumes is already the
    ported one, and discarded — see `IndexOptions`.
    """
    options = options or IndexOptions()
    decorative, no_speech = declined_media(item)
    kinds = sorted(item_content_kinds(item))
    return _fingerprint(
        "item",
        [
            SURFACE_VERSION,
            item.id,
            item.source,
            item.url,
            item.author.handle,
            item.author.name,
            item.created_at.isoformat(),
            item.captured_at.isoformat(),
            item.bookmark_folder,
            item.enriched.primary_topic if item.enriched else None,
            list(item_topics(item)),
            kinds,
            [surface_row(surface) for surface in item_surfaces(item)],
            [[s, [len(t) for t in b]] for s, b in sorted(article_block_texts(item).items())],
            [_model_atoms(failure) for failure in failed_sources(item)],
            [_model_atoms(link) for link in unfetched_links(item)],
            [decorative, no_speech],
        ],
    )


def store_fingerprint(store: Mapping[str, Item], *, options: IndexOptions | None = None) -> str:
    """The DEEP store signal: one sha256 over every item's fingerprint, in id order.

    Order-independent by construction — the ids are sorted — because a dict's iteration order
    is a property of how the store was loaded, not of what it contains. Each `(id, hash)` is
    its own array, so an id cannot run into the hash beside it: that nesting is what stops an
    id ending in a hex run from re-cutting the boundary and presenting a different store as
    this one.

    NOT PAID BY A QUERY. This walks every item and emits every surface; the cheap `StoreSignal`
    above is what a query affords on every call, and keeping the two apart is 02.6a1's whole
    contract. The tempting shape is a fingerprint computed beside `IndexInputs.signal`, *since
    we have already parsed the store*: measured on the live store (2,404 items, sha256
    `f76341a3...`), this walk costs the same ORDER as the parse itself, so folding it in
    roughly doubles a door `status` calls. Read that as a ratio and not as a millisecond
    figure — three measurements on this machine gave three answers.
    """
    return _fingerprint(
        "store", [[k, item_fingerprint(store[k], options=options)] for k in sorted(store)]
    )


# The two PROJECTION versions of this child, one per plane, versioned APART from
# `SURFACE_VERSION` (how a surface is EMITTED) and from each other, because bumping one must not
# rebuild the other — a `Topic` gaining a field is not a `TopicPage` gaining one. Bump whenever
# the payload below changes SHAPE, or one sealed under the old keeps comparing EQUAL to one under
# the new; both read v2 from B-2's `[name, value]` re-cut (B3 moved a MODE, no digest).
VOCAB_VERSION = "xbrain-knowledge-vocab/v2"
TOPICS_VERSION = "xbrain-knowledge-topics/v2"

# Which live input each plane is a projection OF. ONE table (rule 5), so the refusal in
# `_fingerprint` can name the file an operator has to repair instead of making them guess which
# of the three inputs carried the byte. FILENAMES AND NOT PATHS: `Config.data_dir` is
# configurable and this module never sees a directory — `load_index_inputs` takes the three
# paths as arguments. A hardcoded `data/...` would name a file that need not exist, and under a
# test's `tmp_path` one that never did. The filename is the fixed half, so it is what is quoted.
_PLANE_INPUT = {
    "item": "items.json",
    "store": "items.json",
    "vocab": "vocab.yaml",
    "topics": "topics.json",
}


class FingerprintError(ValueError):
    """A plane could not be fingerprinted: an input holds text UTF-8 cannot encode.

    NAMED HERE BECAUSE THIS CHILD IS THE FIRST CONSUMER, which is the obligation `_canonical`
    records. `ensure_ascii=False` is an injectivity choice, and its cost is that a LONE
    SURROGATE reaches `_sha256`'s `.encode("utf-8")` and raises a bare `UnicodeEncodeError` —
    a `ValueError` naming a byte offset into a JSON blob nobody wrote, from a call stack that
    says nothing about which of the three inputs is at fault.

    REACHABLE FROM A REAL INPUT, measured on this tree for BOTH planes: the escape is pure ASCII
    on disk, so `_read_bound`'s decode succeeds and the parser hands the surrogate back intact.

    REFUSAL, NOT `surrogatepass`, AND THE MEASUREMENT IS WHAT DECIDES IT. `surrogatepass` would
    hash deterministically — but `sqlite3` raises the SAME error binding the value as `TEXT`
    (measured below), so the fingerprint would certify what the index can never store and the
    failure would resurface inside 02.7's writer, unnamed and far from the byte.
    """


def _fingerprint(domain: str, value: object) -> str:
    """`_sha256(_canonical(...))` for all four planes, with the encoding refusal named once.

    THE ONE PLACE THE TWO HELPERS ARE COMPOSED (rule 5), so *what happens when a payload cannot
    be encoded* has ONE definition rather than four that drift; the test asserts all four entry
    points raise the same error, each on an atom its OWN plane encodes. HASH-NEUTRAL: the success
    path is exactly the composition the item and store planes already performed.
    """
    try:
        return _sha256(_canonical(domain, value))
    except UnicodeEncodeError as exc:
        offender = ascii(exc.object[exc.start : exc.end])
        # `.get`, never `[...]`: a plane added later without a table entry would raise
        # `KeyError` from INSIDE this handler, chaining a lookup bug on top of the fault being
        # reported and destroying the only message that names the cause. The fallback names the
        # plane, which is the half of the answer the table is not needed for.
        source = _PLANE_INPUT.get(domain, f"the {domain} input")
        raise FingerprintError(
            f"{domain}: {source} (under the configured data directory) holds {offender}, "
            f"which UTF-8 cannot encode ({exc.reason}). Repair the input — the index cannot "
            f"store it either."
        ) from exc


def vocab_fingerprint(vocab: Sequence[Topic]) -> str:
    """sha256 over the WHOLE persisted vocabulary projection — slugs AND descriptions.

    THE DESCRIPTIONS ARE THE POINT, and they are why this plane needs a fingerprint of its own.
    `profile.profile_text` splices each assigned topic's DESCRIPTION into that item's
    `profiles.profile_text`, so editing one word of one description rewrites the `profiles` and
    `profiles_fts` rows of EVERY item carrying that slug — while `item_fingerprint`, taking no
    vocabulary, cannot move. Rule 6 exactly, and `item_fingerprint` names the gap as this one's.
    Two persisted `topics` columns are functions of the description alone: `topics.description`,
    and `topics.vocab_fingerprint`, which `surfaces.topic_record` derives through
    `surface_fingerprint` — stamping `SURFACE_VERSION`, which is why that version is an arm here.

    ORDER, AND THE DUPLICATE SLUG THAT MAKES IT SUBTLE. `sorted` by slug makes the hash
    INDEPENDENT of the order distinct topics happen to sit in — `save_vocab` writes with
    `sort_keys=False`, so the file order is the caller's. But `sorted` is STABLE, and that is
    LOAD-BEARING: `parse_vocab` accepts DUPLICATE slugs and `profile_text` resolves them through
    a dict comprehension, so the LAST entry wins (both measured). Two duplicate orderings
    therefore persist DIFFERENT profile text, and a sort that discarded input order among equal
    slugs would hash them ALIKE — fail-open. A set, a dict or a sort on `(slug, description)`
    reddens the guard that pins this. The accepted false positive is the mirror: an
    exactly-repeated entry persists as ONE row and hashes differently, costing one rebuild.

    NESTED, NEVER JOINED, and each entry is `_persisted_atoms` so no description can re-cut the
    boundary of the next atom — a NUL survives `save_vocab`/`parse_vocab` intact (measured), so
    the flat `"\\0".join(f"{slug}={description}")` this replaces was collidable from a real
    `vocab.yaml`, not only in theory.
    """
    return _fingerprint(
        "vocab",
        [
            VOCAB_VERSION,
            SURFACE_VERSION,
            [_persisted_atoms(t, mode="python") for t in sorted(vocab, key=lambda t: t.slug)],
        ],
    )


def topics_fingerprint(pages: Mapping[str, TopicPage]) -> str:
    """sha256 over the WHOLE persisted topic-page projection — every field, not the text alone.

    EVERY PERSISTED FIELD — now BY CONSTRUCTION (`_persisted_atoms`), never a list someone has
    to keep exhaustive. `topics.synthesized_at` and `topics.post_count_at_synth` are persisted
    columns, and `post_count_at_synth` is half the derivation of a THIRD: `surfaces.topic_record`
    computes `stale` as `len(primary_item_ids) != page.post_count_at_synth`. The implementation
    this replaces hashed overview and notes only, so re-synthesising to the SAME prose against a
    moved post count rewrote two columns and flipped a third with this unmoved. `stale`'s other
    half is an ITEM assignment riding in `item_fingerprint`; neither plane covers it alone.

    `SURFACE_VERSION` is an arm for the same reason as on the vocabulary plane:
    `topics.synthesis_fingerprint` is `surface_fingerprint("topic_overview", "llm", overview)`,
    so a bump rewrites the column with the overview unmoved. Overview and notes are hashed as
    THEMSELVES, so this plane keeps moving for a prose edit even if that derivation narrows.

    THE KEY IS THE CONTRACT; THE FIELD IS HASHED AS A DECLARED FALSE POSITIVE, and getting that
    round the wrong way is a false NEGATIVE. The two CAN diverge — `store.parse_topic_pages`
    takes the key from the JSON object while `TopicPage.slug` is a bare `str` — and the MAPPING
    KEY is the load-bearing one, the join being `topic_pages.get(topic.slug)`. The field rides in
    the projection anyway, costing one rebuild on a hand edit; keying off it would hash two joins
    alike the moment they diverge. The guard below pins both halves.

    The version this replaces hashed `synthesized_at.isoformat()` and CLAIMED it was what is on
    disk; measured it is not, for UTC: pydantic renders `2026-01-20T00:00:00Z` where
    `isoformat()` renders `...+00:00` (naive and `+02:00` DO agree, which kept the guard green).
    INJECTIVITY FOLLOWS from the atom BEING the persisted one; `_persisted_atoms` carries why.

    What is deliberately NOT normalised is the offset. `TopicPage` carries no UTC validator
    (measured: naive and `+02:00` are both accepted), so two pages at the same INSTANT under
    different offsets are two different files; collapsing them would be a false negative.

    WHAT THIS PLANE DOES NOT REACH: `CHUNKER_VERSION` and `ChunkerParams` decide every topic
    chunk's id, span, body and fingerprint and neither is an input here. Both are covered where
    they can be, which is the manifest: `load_compatible_manifest` refuses a base whose chunker
    version OR whose parameters differ from the caller's, so a sweep that re-cuts every chunk
    under unchanged ids cannot be queried. A plane's hash and a document's refusal are not the
    same instrument, and this one is deliberately the second.

    Order-independent by construction — `sorted(pages.items())` — because a dict's iteration
    order is how the file was parsed, not what it holds. Each page is its own nested array: the
    flat `"\\0".join([overview_fp, *note_fps, slug])` this replaces put the SLUG last among
    64-hex fingerprints, and a slug may BE 64 hex, so one could stand where a note's did.
    """
    rows = [[key, _persisted_atoms(page, mode="json")] for key, page in sorted(pages.items())]
    return _fingerprint("topics", [TOPICS_VERSION, SURFACE_VERSION, rows])


def _canonical(domain: str, value: object) -> str:
    """The ONE serialisation every fingerprint here hashes, and the only one (rule 5).

    INJECTIVE ON THE PAYLOAD DOMAIN THESE FINGERPRINTS ACTUALLY BUILD, and the scope of that
    claim is the claim. JSON is NOT injective over Python values in general — `(1, 2)` and
    `[1, 2]` encode identically, and so do `{1: "a"}` and `{"1": "a"}` — so an unqualified
    *injective by round trip* would be false the moment someone believed it. What holds is the
    domain these payloads are built out of: `str`, `int`, `None`, and SEQUENCES of those nested
    to any depth, over which `json.loads` recovers the exact structure. Three consequences a
    future payload must respect: `list` and `tuple` are the SAME value here (`surface_row`
    returns a tuple and is safe only because its position is what carries it); no `dict` may
    enter, its keys being coerced to strings, which is why `_model_atoms` emits
    `[[name, value], ...]`; and no `float` may enter, `allow_nan=False` making a stray
    `NaN`/`Infinity` RAISE rather than emit a literal no reader could parse back.

    This replaced a NUL-JOIN, which framed nothing below the region: both store writers persist
    a NUL, so a stored value could re-split the stream and move every later boundary, including
    the count tags meant to fix them. Measured before the change: two topic lists, two
    vocabularies and two topic planes, each pair distinct after `save`/`load` and hashing alike.

    `ensure_ascii=False` IS THE INJECTIVE SETTING, a correctness choice: with `True` a lone
    surrogate PAIR and the astral character it spells serialise to the same escape, which is a
    collision; with `False` they differ and a lone surrogate raises at `.encode("utf-8")`. That
    raise is REACHABLE from a real `items.json` — the escape is pure ASCII on disk, so
    `_read_bound`'s decode succeeds and `json.loads` hands back the surrogate intact — and it
    surfaces as a raw `UnicodeEncodeError`, a `ValueError` no `OSError` handler will catch;
    naming it is the first consumer's obligation, exactly as `_read_bound`'s is. `domain` is
    hashed IN so two planes cannot serialise alike: `store_fingerprint({"a": i})` and a
    one-entry vocabulary with slug `a` and description `i`'s fingerprint were both
    `[["a", <64 hex>]]`, measured EQUAL. (NUL is spelled in words here: as an escape in a
    non-raw docstring it puts a real one in `__doc__` — three, in the first version.)
    """
    return json.dumps([domain, value], ensure_ascii=False, allow_nan=False)


def _sha256(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# The manifest — the document that SEALS the four fingerprints and the cheap signal
#
# Spec §5.6 field by field, and Plan 02 §2's `manifest.json`. Its PRESENCE is what says a
# build finished; its CONTENT is what every later door compares itself against. Two properties
# decide whether it is worth anything, and both are enforced here rather than described:
#
# 1. IT IS TOTAL AND CLOSED. A document that declares LESS than the schema is incompatible,
#    not lenient (spec §9.3) — the first version of this reader checked the top-level key set
#    and cast what sat under it, so a manifest whose `counts` was `{}` loaded fine and the
#    consistency check, iterating whatever `counts` offered, compared NOTHING: an amputated
#    base reported healthy. And a document declaring MORE is a document from another version:
#    dropping the extra key certifies the index current over something this code never looked
#    at. Both directions refuse, with the command that repairs it.
# 2. IT IS VALIDATED AT THE ONE BOUNDARY A HAND EDIT CROSSES. `manifest.json` is a small text
#    file an operator can open, and everything the writer could never emit — a sealed
#    `UNSTATTABLE`, a negative count, a `built_at` that is not an instant, a top-level JSON
#    list — arrives HERE or nowhere. `write_manifest` round-trips through this same reader
#    before a byte lands, so the two halves cannot drift apart either.
#
# WHAT IS DELIBERATELY NOT HERE. No `build`, no `update`, no `status`, no query door: this
# child ships the contract those commands seal and read, and nothing that seals one. There is
# no `tokenize` / `connective` field either — they decide every recall number and belong with
# the query semantics, which this tree does not have yet; adding them later costs a
# `SCHEMA_VERSION` bump and nothing else, because no manifest exists on disk anywhere today.
# ---------------------------------------------------------------------------


# Spec §5.6, field by field, in ONE place. The reader requires exactly this set, the writer
# emits exactly this set, and `tests/test_knowledge_manifest.py` compares it against the names
# written out by hand from the spec — the one side of that comparison not out of this module.
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
        "embeddings",
        "counts",
        "skipped",
        "failed",
    }
)

# The NESTED schemas, each read off the thing it describes wherever one exists (rule 5), so a
# plane renamed in the DDL or a parameter added to the chunker cannot leave the manifest
# declaring a key nothing produces. `SKIPPED_CAUSES` is the exception and is enumerated: spec
# §5.6 names the four causes and no code object carries them yet — the counters are 02.7's —
# so the literal IS the contract until there is something to read it off.
COUNT_PLANES: frozenset[str] = frozenset({"items", "topics", "surfaces", "chunks", "profiles"})
SKIPPED_CAUSES: frozenset[str] = frozenset(
    {"empty_text", "decorative", "no_speech", "failed_sources"}
)
CHUNKER_PARAM_NAMES: frozenset[str] = frozenset(f.name for f in dataclass_fields(ChunkerParams))


@dataclass(frozen=True)
class Manifest:
    """The index's self-description. Written LAST, so its presence means the build finished."""

    schema_version: str
    built_at: datetime
    store_fingerprint: str
    store_signal: StoreSignal
    vocab_fingerprint: str
    topics_fingerprint: str
    surface_version: str
    chunker_version: str
    chunker_params: dict[str, int]
    counts: dict[str, int]
    skipped: dict[str, int]
    failed: list[dict[str, str]] = field(default_factory=list)
    # The hole Plan 03 fills with `{model, dimension, normalized, command_version}`. Declared
    # NOW so its arrival is not a manifest migration in the next plan. Its INSIDE is not
    # validated: this child ships no embeddings, and a schema for a payload it cannot produce
    # would be prose in the column where a guard belongs. The SLOT's shape is checked.
    embeddings: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """The JSON document. `built_at` as ISO text, the signal as its six integers."""
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
            "embeddings": self.embeddings,
            "counts": dict(self.counts),
            "skipped": dict(self.skipped),
            "failed": [dict(entry) for entry in self.failed],
        }

    @classmethod
    def from_dict(cls, raw: object) -> Manifest:
        """The document, validated TOTAL and CLOSED — and the second half was the bug.

        BOTH DIRECTIONS, AND ONLY ONE OF THEM USED TO BE CHECKED HERE. Missing fields were
        refused; an UNDECLARED one was silently dropped, so a manifest from a writer watching
        a plane this code cannot see loaded as compatible through every door. `_closure_gap`
        now answers both, and it is the same function the nested mappings ask — which is what
        makes «total and closed» one rule rather than two that already disagreed.

        THE DOCUMENT IS A JSON OBJECT BEFORE IT IS ANYTHING ELSE, and that ordering is not
        style. The totality check is a set difference, and `set()` of a non-mapping
        does not raise — it takes the ELEMENTS. A hand-edited `manifest.json` holding the JSON
        LIST of the field NAMES therefore passed the guard with `missing` empty, and the first
        subscript raised `TypeError: list indices must be integers` out of every door, as a
        traceback. A top-level `3`, `null` or `true` was worse still: `set()` of them raises
        from the guard LINE ITSELF, before one field is read. Spec §9.3 asks for an actionable
        error, so the type is checked first and every malformed document leaves by one door.
        """
        if not isinstance(raw, Mapping):
            raise IndexIncompatibleError(
                f"El manifest no es un objeto JSON, es {type(raw).__name__}. {REBUILD_ADVICE}"
            )
        missing, unknown = _closure_gap(set(raw), MANIFEST_FIELDS)
        if missing:
            raise IndexIncompatibleError(f"El manifest no declara {missing}. {REBUILD_ADVICE}")
        if unknown:
            raise IndexIncompatibleError(
                f"El manifest declara {unknown}, que este código no conoce. {REBUILD_ADVICE}"
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
            chunker_params=_counter_mapping(
                raw["chunker_params"], "chunker_params", CHUNKER_PARAM_NAMES
            ),
            embeddings=_optional_mapping(raw["embeddings"], "embeddings"),
            counts=_counter_mapping(raw["counts"], "counts", COUNT_PLANES),
            skipped=_counter_mapping(raw["skipped"], "skipped", SKIPPED_CAUSES),
            failed=_failures(raw["failed"]),
        )


def _malformed(field_name: str, detail: str) -> IndexIncompatibleError:
    """ONE actionable sentence for every malformed field, naming the field and the reason.

    A hand-edited manifest is exactly the input this reader has to survive, and spec §9.3 asks
    for an error that names the command that fixes it rather than a traceback from inside a
    query. `REBUILD_ADVICE` is imported from `index_schema`, beside the error that carries it:
    a corrupt database ends with the same sentence, and two copies would be two things to keep
    in step (rule 5).
    """
    return IndexIncompatibleError(
        f"El manifest tiene el campo {field_name!r} malformado: {detail}. {REBUILD_ADVICE}"
    )


def _closed_keys(value: object, field_name: str, declared: frozenset[str]) -> dict[str, object]:
    """A mapping carrying EXACTLY the declared keys — neither fewer nor more.

    TOTAL because a reader that tolerates an absent key compares less than it should and calls
    an amputated index healthy; CLOSED because a key this code does not know is a document from
    another version, and dropping it certifies the index current over something never looked
    at. Both are refusals for the same reason: nothing here could compare the difference.
    """
    if not isinstance(value, Mapping):
        raise _malformed(field_name, f"no es un objeto, es {type(value).__name__}")
    absent, unknown = _closure_gap(set(value), declared)
    if absent:
        raise _malformed(field_name, f"faltan {absent}")
    if unknown:
        raise _malformed(field_name, f"claves no declaradas {unknown}")
    return dict(value)


def _closure_gap(keys: set[str], declared: frozenset[str]) -> tuple[list[str], list[str]]:
    """`(absent, unknown)` — the ONE definition of «exactly the declared keys» (rule 5).

    THE DOCUMENT AND ITS MAPPINGS USED TO ANSWER THIS DIFFERENTLY, AND THAT IS THE BUG THIS
    EXISTS TO MAKE IMPOSSIBLE. `_closed_keys` refused an undeclared key in `counts`,
    `skipped`, `chunker_params` and `store_signal`, four tests pinned exactly that, and the
    phrase «total and closed» therefore read as covered — while the TOP-LEVEL document only
    ever checked what was MISSING. Measured: a valid manifest plus `"future_plane": {...}`
    loaded through `Manifest.from_dict` and through `load_compatible_manifest`, which
    returned it as COMPATIBLE with the key silently dropped, and a drifted `to_dict` carrying
    it passed `write_manifest`'s round-trip and reached disk.

    A key this code does not know was written by something watching a plane this code cannot
    see, so accepting it certifies the index current over exactly that plane. Both callers now
    compute the gap here and only the WORDING is theirs: the document says «no declara» /
    «declara … que este código no conoce», a mapping says «faltan» / «claves no declaradas»,
    because a caller reading `manifest.json` needs to be told which of the two it is.
    """
    return sorted(declared - keys), sorted(keys - declared)


def _counter_mapping(value: object, field_name: str, declared: frozenset[str]) -> dict[str, int]:
    """A closed mapping whose every value is a NON-NEGATIVE integer.

    The rule that is right for a counter and WRONG for the cheap signal, which is why the
    signal has its own (`StoreSignal.from_dict`): a plane holding minus one row, or a chunker
    whose `max_chars` is minus one, is a malformed document, while a negative `mtime_ns` is a
    pre-epoch file. `type(count) is int` for the reason spelled out there.
    """
    raw = _closed_keys(value, field_name, declared)
    checked: dict[str, int] = {}
    for key, count in raw.items():
        if type(count) is not int or count < 0:
            raise _malformed(field_name, f"{key!r} debe ser un entero no negativo, es {count!r}")
        checked[str(key)] = count
    return checked


def _optional_mapping(value: object, field_name: str) -> dict[str, object] | None:
    """`null` or an object — the `embeddings` slot Plan 03 fills — and nothing else."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _malformed(field_name, f"no es null ni un objeto, es {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def _failures(value: object) -> list[dict[str, str]]:
    """The `failed` list: every entry a mapping of TEXT, or the document is refused.

    Spec §5.6's *chunks omitidos o fallidos*, and it reaches an operator's screen — a nested
    structure where a string belongs prints as a Python repr from inside a command, so the
    shape is fixed where it is read rather than where it happens to be shown.
    """
    if not isinstance(value, list) or not all(
        isinstance(entry, Mapping) and all(isinstance(v, str) for v in entry.values())
        for entry in value
    ):
        raise _malformed("failed", "no es una lista de objetos de texto")
    return [{str(k): str(v) for k, v in entry.items()} for entry in value]


def _instant(value: object) -> datetime:
    """`built_at` as an instant, or the malformed-field sentence instead of a `ValueError`.

    `datetime.fromisoformat` raises a bare `ValueError` naming the string and no command, which
    is precisely the traceback-from-inside-a-query shape spec §9.3 rules out.
    """
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as error:
        raise _malformed("built_at", f"{value!r} no es un instante ISO") from error


def write_manifest(index_dir: Path, manifest: Manifest) -> None:
    """Write the manifest LAST. Its presence is what says a build completed.

    THE WRITER ROUND-TRIPS THROUGH THE READER, AND NOTHING LANDS UNTIL IT HAS. The document is
    serialised, parsed and validated by `Manifest.from_dict` BEFORE one byte is written, so a
    build cannot seal a manifest every later door would refuse — a state whose only exit is
    `build --force` — and, the other direction, a writer whose shape drifted from the reader's
    schema fails HERE, loudly, instead of producing a document the reader happens to accept
    while comparing less than it should.
    """
    document = json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2)
    Manifest.from_dict(json.loads(document))
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(index_dir).write_text(document, encoding="utf-8")


def load_manifest(index_dir: Path) -> Manifest:
    """Read the manifest, turning any malformed document into an ACTIONABLE error.

    An index that was never built is not a corrupt one, so the two get different advice:
    `build` for the absence, `build --force` for everything else. Nothing is repaired here —
    Plan 02 §11: a corrupt base is rebuilt, never patched.
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
    """The manifest, REFUSED unless every version the code depends on matches (Plan 02 §2).

    Spec §9.3: *manifest incompatible: no se consulta parcialmente.* Each of the three versions
    names a real failure. A base written under another `SCHEMA_VERSION` has columns this code
    cannot read; one under another `SURFACE_VERSION` holds surface fingerprints computed over a
    different projection; one under another `CHUNKER_VERSION` holds chunk fingerprints that
    would ALL fail verification, and the honest answer is the rebuild, never «22.286 chunks
    excluded».

    THE FOURTH CHECK IS THE CHUNKER'S PARAMETERS, AND IT IS NOT A DETAIL. Plan 02 §7 sweeps
    `target x overlap`, and a sweep does not bump `CHUNKER_VERSION` — so a base cut at one
    `target` and queried under another holds chunks whose ids RESOLVE and whose spans are not
    what they were: nothing raises, and the text behind a citable id is a different text. It is
    compared only when the caller SUPPLIES the parameters: a door that does not chunk has
    nothing to compare against and must not invent the defaults.

    `!r` ON EVERY MANIFEST STRING. The manifest is hand-editable and these strings are
    interpolated into a sentence a command prints, so a raw newline in `schema_version` stands
    at column 0 as a forged header and an ESC reaches the TTY. A repr carries neither.
    """
    manifest = load_manifest(index_dir)
    mismatches = []
    if manifest.schema_version != SCHEMA_VERSION:
        mismatches.append(f"schema_version {manifest.schema_version!r} != {SCHEMA_VERSION!r}")
    if manifest.surface_version != SURFACE_VERSION:
        mismatches.append(f"surface_version {manifest.surface_version!r} != {SURFACE_VERSION!r}")
    if manifest.chunker_version != CHUNKER_VERSION:
        mismatches.append(f"chunker_version {manifest.chunker_version!r} != {CHUNKER_VERSION!r}")
    if params is not None and manifest.chunker_params != asdict(params):
        mismatches.append(f"chunker_params {manifest.chunker_params!r} != {asdict(params)!r}")
    if mismatches:
        raise IndexIncompatibleError(
            "El índice fue construido con otra versión: "
            + "; ".join(mismatches)
            + f". {REBUILD_ADVICE}"
        )
    return manifest


# ---------------------------------------------------------------------------
# 02.7 — the WRITER and `build`: put the index on disk and seal it
#
# ONE WRITER, and that is the whole reason `write_item` / `write_topic` are public. `build`,
# 02.8's `update` and the evaluation harness all come through here, so there is no second walk
# that could emit a slightly different corpus and make a measured baseline describe something
# other than what a query answers over (rule 5).
#
# THE MANIFEST IS WRITTEN LAST, OUTSIDE THE TRANSACTION, AND THAT ORDERING IS THE WHOLE SAFETY
# PROPERTY. Rows go in one transaction; the manifest is sealed only after it commits. An
# interruption therefore leaves rows rolled back AND no manifest — and an index with no
# manifest is REFUSED by every door rather than answered partially (spec §9.3), so a `Ctrl-C`
# cannot produce a small index that looks valid.
#
# WHAT IS NOT HERE. No `update`, no `status`, no invalidation, no query door, no CLI: 02.8 and
# later. `count_rows` lands here because the manifest's `counts` are read back from the base.
# ---------------------------------------------------------------------------


UPDATE_ADVICE = "Actualiza el índice con `xbrain index update`."


# The column order of `topics`, as ONE tuple type — the writer binds it and 02.8's comparator
# will read it back. The same pattern as `SurfaceRow`, for the same reason: a membership-derived
# column cannot be stored without there being one place that says what it is.
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

# One literal per table. The f-string version was safe — the names come from a tuple in this
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
    """How many rows each plane holds, read from the base itself.

    Keyed by `COUNT_PLANES`, which is what the manifest declares — a test binds the two, so a
    plane added to one and forgotten in the other goes red instead of producing a manifest
    that counts four planes and claims five.
    """
    return {
        table: int(connection.execute(statement).fetchone()[0])
        for table, statement in _COUNT_STATEMENTS.items()
    }


@dataclass(frozen=True)
class BuildReport:
    """What a build did — or, under `dry_run`, what it WOULD have done."""

    items_written: int
    topics_written: int
    surfaces_written: int
    chunks_written: int
    profiles_written: int
    skipped: dict[str, int]
    failed: list[dict[str, str]]
    duration_seconds: float
    dry_run: bool


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


def topic_row(record: TopicRecord) -> TopicRow:
    """What the index STORES about a topic — the row `_write_topic_row` inserts, verbatim.

    The same pattern as `surface_row` (G-5), for the same reason: one projection, and 02.8's
    comparator reads it back. A membership-derived column — `primary_item_ids_json`,
    `secondary_item_ids_json`, `stale` — therefore cannot be stored without there being a
    single place that says what it is.
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


def write_item(
    index: LexicalIndex,
    item: Item,
    vocab: Sequence[Topic],
    counters: WriteCounters,
    *,
    options: IndexOptions,
) -> None:
    """Everything one item contributes to the index: metadata, surfaces, chunks and profile."""
    surfaces = item_surfaces(item)
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
    decorative, no_speech = declined_media(item)
    counters.empty_text += empty_text
    counters.decorative += decorative
    counters.no_speech += no_speech
    # Recorded ON THE ITEM'S ROW, so the manifest's `skipped` can be SUMMED from the base
    # rather than carried over from a previous manifest (A-3). 02.8's incremental path
    # deletes and rewrites the row, and the total stays exact without re-walking the corpus.
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

    Built from `knowledge_item` rather than re-derived from the store, so a later `--author`,
    `--topic` or `--kind` answers with exactly what `get` would show. A second derivation here
    is the divergence rule 5 is about, and the field that would go wrong first is
    `content_kinds`: a FAILED fetch has a kind, and listing it would tell a consumer to ask for
    a body that does not exist.

    `source_failures` takes SIX columns and not seven: `attempts` was dropped at
    `SCHEMA_VERSION` "4" because it is fetch bookkeeping with no field on the knowledge
    projection, and the only writer that ever existed bound it to a literal `None`.
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
            "http_status) VALUES (?,?,?,?,?,?)",
            (
                item.id,
                failure.kind,
                failure.url,
                failure.failure_reason,
                failure.error,
                failure.http_status,
            ),
        )
    for link in projection.unfetched_links:
        index.connection.execute(
            "INSERT INTO unfetched_links (item_id, url, reason, detail) VALUES (?,?,?,?)",
            (item.id, link.url, link.reason, link.detail),
        )


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


def _write_topic_row(index: LexicalIndex, record: TopicRecord) -> None:
    """The one `INSERT` into `topics`, binding `topic_row` in its declared column order."""
    index.connection.execute(
        "INSERT OR REPLACE INTO topics (slug, description, overview, notes_json, "
        "synthesized_at, post_count_at_synth, stale, primary_item_ids_json, "
        "secondary_item_ids_json, vocab_fingerprint, synthesis_fingerprint) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        topic_row(record),
    )


def _write_surfaces(
    index: LexicalIndex, surfaces: Sequence[KnowledgeSurface], counters: WriteCounters
) -> None:
    """Every surface row, from the SAME projection `item_fingerprint` hashes (G-5).

    What is stored is what is fingerprinted, by construction. LENGTH, never the body, in the
    last column: spec §10.8 keeps article text out of derived stores, and `chunks.text` is
    where text lives.
    """
    for surface in surfaces:
        counters.surfaces += 1
        index.connection.execute(
            "INSERT OR REPLACE INTO surfaces (surface_id, owner_type, owner_id, surface_type, "
            "origin, trust_class, derived, attribution_handle, attribution_name, title, url, "
            "locator_json, language, fingerprint, char_length) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            surface_row(surface),
        )


def topic_membership(
    store: Mapping[str, Item], slug: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(primary_item_ids, secondary_item_ids)` for a topic, both sorted.

    Sorted because `TopicRecord` is fingerprinted downstream and an order following dict
    iteration would make two builds of the same store differ. A PRIMARY item is EXCLUDED from
    the secondary list rather than appearing in both: the two answer different questions, and
    double counting would inflate any membership figure taken from them.
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


@dataclass(frozen=True)
class ManifestTallies:
    """`counts` and `skipped` as the DATABASE holds them — the one source for both writers.

    Reading them back from the rows means the manifest describes the base BY CONSTRUCTION, so
    a base that disagrees with its manifest can be DETECTED (C-3). Computing them from the run
    counters instead is what drifted before: an incremental update adjusted four of the five
    counts by hand and topic chunks were added on every rebuild and never subtracted.
    """

    counts: dict[str, int]
    skipped: dict[str, int]


def manifest_tallies(connection: sqlite3.Connection) -> ManifestTallies:
    """What the manifest reports about the base, read from the base itself."""
    summed = connection.execute(
        "SELECT COALESCE(SUM(skipped_empty_text), 0), COALESCE(SUM(skipped_decorative), 0), "
        "COALESCE(SUM(skipped_no_speech), 0) FROM items"
    ).fetchone()
    failed_source_rows = connection.execute("SELECT COUNT(*) FROM source_failures").fetchone()[0]
    return ManifestTallies(
        counts=count_rows(connection),
        skipped={
            "empty_text": int(summed[0]),
            "decorative": int(summed[1]),
            "no_speech": int(summed[2]),
            "failed_sources": int(failed_source_rows),
        },
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


def build(
    index_dir: Path,
    inputs: IndexInputs,
    *,
    options: IndexOptions | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> BuildReport:
    """Build `data/index/` from scratch and SEAL it. Read-only with respect to the store.

    IT TAKES AN `IndexInputs`, AND THAT IS P1b ENFORCED BY TYPE. The rows and the cheap signal
    come from ONE snapshot, bound together by `load_index_inputs` at the moment the bytes were
    parsed, so no caller can pair rows read at one instant with a `stat` taken at another. The
    shape this closes is not hypothetical: sealing `StoreSignal.of(paths)` AFTER the commit
    describes whatever file sits on that path by then, and a save landing in that window
    produced a manifest certifying a store the base had never seen — the next query compared
    EQUAL and answered over stale rows with nothing declared. A signal argument that could be
    omitted is the same defect one keyword away, which is why there is no such argument.

    ONE TRANSACTION, AND THE MANIFEST LAST — AND, ON A FORCED REBUILD, THE OLD MANIFEST REMOVED
    FIRST. A `Ctrl-C` or a full disk mid-build rolls the rows back and leaves no manifest, and
    an index with no manifest is REFUSED by every door rather than answered partially (spec
    §9.3), so an interruption cannot produce a small index that looks valid. That sentence was
    only true for a FRESH build until C-1: `--force` kept the previous manifest standing while
    the new database was written, so an interrupted forced rebuild left a manifest whose
    versions and cheap signal still matched — `status` reported nothing wrong and a query
    answered «no results» over an EMPTY base, indistinguishable from a corpus with no matches.

    A forced rebuild therefore does NOT preserve the previous index: manifest and database are
    both gone before the first row is written, and recovering from an interruption is
    `xbrain index build` again. Rebuilding over an existing index REQUIRES `force`, because a
    rebuild throws away something that may have taken minutes and the incremental path usually
    wants `index update` instead. The error names both commands.

    TWO THINGS FOUND BY MEASURING, NOT BY READING:

    * `dry_run` builds into `sqlite3(":memory:")` and touches NO FILE AT ALL. The first version
      opened the real database (creating it when absent), rolled back, then removed the file it
      believed it had created — so a dry run against a working index DESTROYED it, from the
      flag whose whole promise is that it changes nothing;
    * `force` UNLINKS the database instead of clearing the rows. Clearing in place left
      SQLite's freelist behind: on the real corpus a fresh build was 51.2 MB and the same index
      after five forced rebuilds was 66.5 MB, with `VACUUM` recovering it only to 60.6 MB. A
      derived artefact whose size depends on how many times it has been rebuilt is one nobody
      can reason about.
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
        # THE MANIFEST GOES FIRST (C-1). It is what every query trusts, so it must not outlive
        # the database it describes — see the docstring for what an interrupted `--force` left
        # standing before this line existed.
        manifest_path(index_dir).unlink(missing_ok=True)
        db_path(index_dir).unlink(missing_ok=True)
        # The ONE caller allowed to create the file (G-2): every other door finds the absence
        # and names the command instead of leaving an empty base behind.
        connection = open_index(db_path(index_dir), create=True)
    try:
        with connection:  # one transaction: commit on success, rollback on ANY exception
            _write_everything(
                LexicalIndex(connection),
                inputs.store,
                inputs.vocab,
                inputs.topic_pages,
                counters,
                options=options,
            )
            tallies = manifest_tallies(connection)
            if dry_run:
                # A dry run does the whole walk and then throws it away, so the counts it
                # reports are the counts a real build WOULD produce — not an estimate.
                raise _DryRun
    except _DryRun:
        connection.close()
        return _build_report(
            counters,
            failed,
            started,
            items=len(inputs.store),
            topics=len(inputs.vocab),
            dry_run=True,
        )
    finally:
        if not connection_closed(connection):
            connection.close()

    write_manifest(
        index_dir,
        _fresh_manifest(inputs, tallies, failed, options=options),
    )
    return _build_report(
        counters,
        failed,
        started,
        items=len(inputs.store),
        topics=len(inputs.vocab),
        dry_run=False,
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


def _fresh_manifest(
    inputs: IndexInputs,
    tallies: ManifestTallies,
    failed: list[dict[str, str]],
    *,
    options: IndexOptions,
) -> Manifest:
    """The manifest a full build writes — every version taken from the CODE, not carried over.

    A build is what DEFINES the versions the index was written under, so they come from the
    constants; 02.8's update has already proved they match and must COPY them instead, rather
    than silently "fixing" a mismatch that should have refused the run.

    The four deep fingerprints are computed HERE, from the same `IndexInputs` the rows were
    written from — never re-read from disk — so the manifest's answer to *what changed* is
    about the corpus that is actually in the base.
    """
    return Manifest(
        schema_version=SCHEMA_VERSION,
        built_at=datetime.now(timezone.utc),
        store_fingerprint=store_fingerprint(inputs.store, options=options),
        store_signal=inputs.signal,
        vocab_fingerprint=vocab_fingerprint(inputs.vocab),
        topics_fingerprint=topics_fingerprint(inputs.topic_pages),
        surface_version=SURFACE_VERSION,
        chunker_version=CHUNKER_VERSION,
        chunker_params=asdict(options.params),
        counts=dict(tallies.counts),
        skipped=dict(tallies.skipped),
        failed=failed,
    )


def _skipped(counters: WriteCounters) -> dict[str, int]:
    """The four causes spec §5.6 asks about, each counting what its name says.

    `empty_text` is structurally 0 today: the emitters drop a blank surface before the index
    ever sees it, so the counter can only move if a surface with a whitespace-only body ever
    reaches here. It is kept because the manifest shape is specified and because the path is
    real — and it is documented as 0-by-construction so nobody quotes it as a measurement of
    the corpus (rule 2).
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
