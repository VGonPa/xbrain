"""The three inputs of the index, read as ONE snapshot, with the cheap signal bound to it
(Plan 02 §2, §3; spec §5.6).

ONE CHEAP SIGNAL, PAID ON EVERY QUERY. Indexing is MANUAL BY DECISION (spec §9.2), so the
failure that actually happens is not corruption — it is *you ran `enrich` and did not
reindex*. `StoreSignal` is `mtime_ns` and size of the THREE inputs: three `os.stat`, cheap
enough for every query, and it answers *an input moved*. It cannot answer *WHICH items
changed* — that is a per-item fingerprint over the emitted surfaces, a different cost paid
only by build/update/status, and it is the next child's. Nothing here computes one.

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

import os
from dataclasses import dataclass
from pathlib import Path

from xbrain.models import Item, Topic, TopicPage
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
