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

THE FOUR PLANES ARE FOUR BECAUSE THEY MOVE APART, and the vocabulary is the one that proves it:
a description edit rewrites the `profiles` and `profiles_fts` rows of every item carrying that
slug, and `item_fingerprint` — which takes no vocabulary — cannot see it. One fused signal
would either rebuild everything for a topic-note typo or miss that, and the manifest that seals
the four separately is 02.6b's.

WHAT IS NOT HERE. Nothing in this tree CONSUMES a fingerprint yet, which is why the coverage
gaps review #161 named were closed here — free before a manifest exists, expensive after.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from xbrain.executors.api import iter_content_sources
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams
from xbrain.knowledge.ids import SURFACE_VERSION
from xbrain.knowledge.models import KnowledgeSurface
from xbrain.knowledge.surfaces import (
    article_block_texts,
    failed_sources,
    item_content_kinds,
    item_surfaces,
    item_topics,
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
      refusing the query outright beats per-item invalidation. 02.6b's — and `ids.py:42` names
      `load_compatible_manifest` as its home, a function that does not exist in this tree yet.
    - `profiles.profile_text` — a `vocab.yaml` edit splices each assigned topic's DESCRIPTION
      into it (spec §5.1.A) and rewrites `profiles`/`profiles_fts` for every assigned item while
      this fingerprint, which takes no vocabulary, cannot move. DISCHARGED by
      `vocab_fingerprint` below, which is why that plane exists; the rebuild it obliges still
      reaches the item plane through 02.7. Its OTHER half is NOT, and filing it there would record a
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
# `SURFACE_VERSION` and from each other. `SURFACE_VERSION` says how a surface is emitted;
# these say what these two fingerprints HASH, which is a different thing that moves for
# different reasons. Separate constants because bumping one must not rebuild the other: a
# `Topic` gaining a field is not a `TopicPage` gaining one. Bump the relevant one whenever the
# corresponding payload below changes shape, or every fingerprint sealed under the old shape
# keeps comparing EQUAL to one taken under the new — the fail-open direction, and the whole
# reason a projection carries a version at all.
VOCAB_VERSION = "xbrain-knowledge-vocab/v1"
TOPICS_VERSION = "xbrain-knowledge-topics/v1"

# Which live input each plane is a projection OF. ONE table (rule 5), so the refusal in
# `_fingerprint` can name the file an operator has to repair instead of making them guess
# which of the three inputs carried the byte.
_PLANE_INPUT = {
    "item": "data/items.json",
    "store": "data/items.json",
    "vocab": "data/vocab.yaml",
    "topics": "data/topics.json",
}


class FingerprintError(ValueError):
    """A plane could not be fingerprinted: an input holds text UTF-8 cannot encode.

    NAMED HERE BECAUSE THIS CHILD IS THE FIRST CONSUMER, which is the obligation `_canonical`
    records. `ensure_ascii=False` is an injectivity choice, and its cost is that a LONE
    SURROGATE reaches `_sha256`'s `.encode("utf-8")` and raises a bare `UnicodeEncodeError` —
    a `ValueError` naming a byte offset into a JSON blob nobody wrote, from a call stack that
    says nothing about which of the three inputs is at fault.

    REACHABLE FROM A REAL INPUT, measured on this tree: the escape is pure ASCII on disk, so
    `_read_bound`'s decode succeeds and the parser hands the surrogate back intact — verified
    for BOTH of this child's planes, a `data/topics.json` of ASCII bytes containing `\\ud800`
    and the same escape in a `data/vocab.yaml`.

    REFUSAL, NOT `surrogatepass`, AND THE MEASUREMENT IS WHAT DECIDES IT. `surrogatepass` would
    hash the value deterministically — but `sqlite3` raises the SAME `UnicodeEncodeError`
    binding it as `TEXT` (measured), so the fingerprint would certify a value the index can
    never store, and the failure would resurface inside 02.7's writer, unnamed and far from the
    byte. Refusing at the fingerprint, naming the FILE, is strictly the more actionable of the
    two, and it fails in the direction `_read_bound` already fails in one layer down.
    """


def _fingerprint(domain: str, value: object) -> str:
    """`_sha256(_canonical(...))` for all four planes, with the encoding refusal named once.

    THE ONE PLACE THE TWO HELPERS ARE COMPOSED (rule 5). Every fingerprint in this module goes
    through here, so *what happens when a payload cannot be encoded* has ONE definition rather
    than four that drift; the test asserts all four entry points raise the same error, which is
    what stops the next plane from being added with a bare `UnicodeEncodeError` again.

    HASH-NEUTRAL. It changes no digest — the success path is exactly the composition the item
    and store planes already performed — and only turns a raise into a named one.
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
            f"{domain}: {source} holds {offender}, which UTF-8 cannot encode "
            f"({exc.reason}). Repair the input — the index cannot store it either."
        ) from exc


def vocab_fingerprint(vocab: Sequence[Topic]) -> str:
    """sha256 over the WHOLE persisted vocabulary projection — slugs AND descriptions.

    THE DESCRIPTIONS ARE THE POINT, and they are why this plane needs a fingerprint of its own.
    `profile.profile_text` splices each assigned topic's DESCRIPTION into that item's
    `profiles.profile_text`, so editing one word of one description rewrites the `profiles` and
    `profiles_fts` rows of EVERY item carrying that slug — while `item_fingerprint`, which takes
    no vocabulary at all, cannot move. That is rule 6 exactly: the evidence is repaired and the
    derivative stands. This fingerprint is what makes the profile rebuild obligation
    DETERMINISTIC rather than something an operator has to remember, and `item_fingerprint`
    names the gap as this child's for that reason.

    Two persisted `topics` columns are also functions of the description and no other input:
    `topics.description`, and `topics.vocab_fingerprint`, which `surfaces.topic_record` derives
    as `surface_fingerprint("topic_description", "unknown", description)`. That derivation
    stamps `SURFACE_VERSION`, which is why the emitter version is an arm here: a bump rewrites
    that column with every description byte unmoved.

    ORDER, AND THE DUPLICATE SLUG THAT MAKES IT SUBTLE. `sorted` by slug makes the hash
    INDEPENDENT of the order distinct topics happen to sit in — `save_vocab` writes with
    `sort_keys=False`, so the file order is the caller's, and a reordering that persists
    identically must not force a rebuild. But `sorted` is STABLE, and that stability is
    LOAD-BEARING, not incidental: `parse_vocab` accepts DUPLICATE slugs (measured — two entries
    with slug `a` survive the round trip in input order) and `profile_text` resolves them
    through a dict comprehension, so the LAST entry wins (measured: the same item profiles as
    `...a\\nSECOND...` under `[FIRST, SECOND]` and `...a\\nFIRST...` reversed). Two duplicate
    orderings therefore persist DIFFERENT profile text, and a sort that discarded input order
    among equal slugs would hash them ALIKE — fail-open, the direction this module never fails
    in. Replacing the stable sort with a set, a dict or a sort on `(slug, description)` reddens
    the guard that pins this.

    THE ACCEPTED FALSE POSITIVE, named rather than discovered: an exactly-repeated entry
    (`[(a, x), (a, x)]`) persists as the single row `[(a, x)]` and hashes differently, so it
    costs one wasted rebuild. That is the warning direction, and it is the same trade the cheap
    signal and the item plane's three variadic regions already make.

    NESTED, NEVER JOINED. Each entry is its own array, so no description can re-cut the
    boundary of the next atom — a NUL survives `save_vocab`/`parse_vocab` intact (measured), so
    the flat `"\\0".join(f"{slug}={description}")` this replaces was collidable from a real
    `vocab.yaml`, not only in theory.
    """
    return _fingerprint(
        "vocab",
        [
            VOCAB_VERSION,
            SURFACE_VERSION,
            [[topic.slug, topic.description] for topic in sorted(vocab, key=lambda t: t.slug)],
        ],
    )


def topics_fingerprint(pages: Mapping[str, TopicPage]) -> str:
    """sha256 over the WHOLE persisted topic-page projection — every field, not the text alone.

    EVERY PERSISTED FIELD, AND THE TWO THAT USED TO BE MISSING ARE THE REASON THIS IS NOT A
    REWRITE OF THE OLD ONE. `topics.synthesized_at` and `topics.post_count_at_synth` are
    persisted columns, and `post_count_at_synth` is half the derivation of a THIRD:
    `surfaces.topic_record` computes `stale` as `len(primary_item_ids) != page.post_count_at_synth`.
    The prior implementation hashed the overview and the notes only, so `xbrain topics`
    re-synthesising a page to the SAME prose against a moved post count rewrote two columns and
    flipped a third with this fingerprint unmoved — the row on disk moves and `update` reports
    the plane unchanged. The other half of `stale`, the live primary count, is an ITEM
    assignment and rides in `item_fingerprint` via `item_topics`; between the two planes the
    column is covered, and neither covers it alone.

    `SURFACE_VERSION` is an arm for the same reason as on the vocabulary plane:
    `topics.synthesis_fingerprint` is `surface_fingerprint("topic_overview", "llm", overview)`,
    so a bump rewrites the column with the overview unmoved. The overview and the notes are
    hashed as THEMSELVES rather than through `surface_fingerprint`, so this plane keeps moving
    for a prose edit even if that derivation is later narrowed.

    THE KEY IS THE CONTRACT; THE FIELD IS HASHED AS A DECLARED FALSE POSITIVE, and getting
    that round the wrong way is a false NEGATIVE. `store.parse_topic_pages` takes the mapping
    key from the JSON object and validates `TopicPage.slug` separately — `Topic.slug` is
    pattern-constrained but `TopicPage.slug` is a bare `str` — so the two CAN diverge in a
    hand-edited `topics.json`, and a round trip preserves the divergence. Measured: it is the
    MAPPING KEY that is load-bearing, because the join is `topic_pages.get(topic.slug)` and it
    decides which overview and which notes land on which topic row; `topics.slug` itself comes
    from the VOCABULARY, and `TopicPage.slug` is read by nothing on the index path. The field
    is hashed anyway because a hand edit to it costs one rebuild, while keying off it instead
    would hash two different joins alike the moment they diverge. A page under a key matching
    NO vocabulary slug is the same accepted trade: it contributes to no table (the walk is over
    the vocabulary) and still moves this hash. 45 of 45 keys agree with their field today, by
    the convention of the single writer that builds pages — not by construction.

    `synthesized_at` IS HASHED AS ITS `isoformat()`, WHICH IS WHAT IS ON DISK.
    `save_topic_pages` writes `model_dump(mode="json")`, i.e. that same string, and `TopicPage`
    carries NO UTC validator (measured: a naive datetime and a `+02:00` one are both accepted
    and their `isoformat()` differ). So two pages at the same INSTANT under different offsets
    are two different files and two different `topics.synthesized_at` values, and hashing the
    rendered string tracks the bytes rather than an instant nobody stored.

    WHAT THIS PLANE DOES NOT REACH, in the same boat as the item plane and for the same reason:
    `CHUNKER_VERSION` and `ChunkerParams` decide every topic chunk's id, span, body and
    fingerprint, and neither is an input here. A bump rewrites the topic chunk plane with this
    hash unmoved, which is CORRECT — a manifest refusing the query outright beats per-topic
    invalidation. 02.6b's, exactly as `item_fingerprint` records it for the item chunk plane.

    Order-independent by construction — `sorted(pages.items())` — because a dict's iteration
    order is a property of how the file was parsed, not of what it holds. Each page is its own
    nested array: the flat `"\\0".join([overview_fp, *note_fps, slug])` this replaces put the
    SLUG last among 64-hex fingerprints, and a slug may BE 64 hex characters under
    `models.Topic`'s pattern, so one page's slug could stand where another's note fingerprint
    did.
    """
    return _fingerprint(
        "topics",
        [
            TOPICS_VERSION,
            SURFACE_VERSION,
            [
                [
                    key,
                    page.slug,
                    page.overview,
                    list(page.notes),
                    page.synthesized_at.isoformat(),
                    page.post_count_at_synth,
                ]
                for key, page in sorted(pages.items())
            ],
        ],
    )


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
