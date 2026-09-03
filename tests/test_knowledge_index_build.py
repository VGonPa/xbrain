# tests/test_knowledge_index_build.py
"""The three inputs of the index, read as one snapshot, with the cheap signal bound to it
(Plan 02 §2, §3, steps 3 and 4).

The argument for the cheap signal — what it can answer, what it cannot, and the direction it
fails in — is stated once, in `index_build.py`'s module docstring; it is not restated here.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build, index_schema
from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams
from xbrain.knowledge.ids import SURFACE_VERSION
from xbrain.knowledge.models import KnowledgeSurface, Locator, SourceFailure, UnfetchedLink
from xbrain.executors.api import iter_content_sources, iter_described_photos
from xbrain.knowledge.surfaces import (
    article_block_texts,
    failed_sources,
    item_content_kinds,
    item_surfaces,
    item_topics,
    unfetched_links,
)
from xbrain.models import (
    ArticleTextBlock,
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoDescribed,
    Topic,
    TopicPage,
)

FIXTURES = Path(__file__).parent / "fixtures"
UTC = timezone.utc

ZERO = index_build.StoreSignal(0, 0, 0, 0, 0, 0)


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    return store, vocab, pages


@pytest.fixture()
def three_inputs(tmp_path: Path, corpus) -> Path:
    """A data/ directory holding ALL THREE inputs, written by the store's own writers."""
    from xbrain.rubrics import save_vocab
    from xbrain.store import save_store, save_topic_pages

    store, vocab, pages = corpus
    data = tmp_path / "data"
    save_store(store, data / "items.json")
    save_vocab(vocab, data / "vocab.yaml")
    save_topic_pages(pages, data / "topics.json")
    return data


def _paths(data: Path) -> tuple[Path, Path, Path]:
    """The three inputs, in the order every door of this module takes them."""
    return data / "items.json", data / "vocab.yaml", data / "topics.json"


# ---------------------------------------------------------------------------
# P1b — the signal describes the bytes that were PARSED
#
# Two races, two different physical mechanisms, and only the second can catch a stat taken
# after the read. An ATOMIC REPLACE (`os.replace`, what `save_store` does) leaves the open
# handle on the old inode, whose stat never moves, so `fstat` before and after the read agree
# and the ordering is invisible. An IN-PLACE REWRITE (`write_text`, what `save_vocab` does)
# truncates the very inode the reader is holding, and there the ordering is the whole story.
# Both writers are real and in this repo, so both cases are reachable.
# ---------------------------------------------------------------------------


def test_load_index_inputs_binds_the_signal_to_the_handle_not_to_the_path(
    three_inputs: Path, corpus
) -> None:
    """The HANDLE half of P1b: an atomic replacement landing mid-read cannot be sealed.

    `save_store` writes through `os.replace`, so the handle keeps the inode it opened: the
    bytes parsed are that inode's, the signal is that inode's, and the PATH is left on a newer
    inode that query-time `StoreSignal.of` reports as different — the index declares itself
    behind rather than certifying rows it never read.

    STAGED BETWEEN THE `open` AND THE STAT, which is the only window where a stat of the PATH
    and a stat of the HANDLE can disagree: once the handle is open the two name different
    inodes. An earlier version of this test raced inside `read` instead and stayed GREEN under
    its own mutation — `path.stat()` ran before the replacement landed, so it read the same
    file either way and pinned nothing (rule 1, caught by executing the mutation).

    Seen red under the mutation `stat = os.fstat(handle.fileno())` -> `stat = path.stat()`:
    the signal then described the replacement while the parsed rows were the old ones, and
    `loaded.signal != before` — a manifest certifying an `items.json` it had never read.
    """
    from xbrain.store import save_store

    store, _vocab, _pages = corpus
    items, vocab, topics = _paths(three_inputs)
    before = index_build.StoreSignal.of(items, vocab, topics)
    newer = {**store, "k01": store["k01"].model_copy(update={"text": "replaced during the load"})}

    class RacingPath(type(items)):
        def open(self, *args, **kwargs):
            handle = super().open(*args, **kwargs)
            save_store(newer, items)  # the race: the replace lands with the handle already open
            return handle

    loaded = index_build.load_index_inputs(RacingPath(items), vocab, topics)
    after = index_build.StoreSignal.of(items, vocab, topics)

    assert after != before, "the replacement really moved the file the path names"
    assert loaded.store["k01"].text == store["k01"].text, "the parse saw the bytes it read"
    assert loaded.signal == before, "the signal describes the snapshot that was parsed"
    assert loaded.signal != after, "so the index declares itself behind, which is the warning"


def test_the_signal_is_stat_before_read_so_an_in_place_rewrite_cannot_reseal_it(
    three_inputs: Path, corpus
) -> None:
    """The BEFORE half of P1b, which the handle half above cannot reach.

    `save_vocab` rewrites through `write_text`: same inode, truncated under the reader. A stat
    taken AFTER the read therefore describes the REPLACEMENT while the text in hand is the
    original — the loader would seal a signal for bytes it never parsed, and the next query,
    comparing that sealed signal against the same file, finds them EQUAL and answers over the
    old vocabulary with nothing declared. That is the false negative the whole module exists
    to make impossible, and it is invisible to an atomic-replace race.

    Taken BEFORE, the signal is the older content's, so the comparison against disk is
    UNEQUAL and the index is declared behind — the direction this fails in, at one `update`.

    Seen red under the mutation `os.fstat(handle.fileno())` moved BELOW `data = handle.read()`
    in `_read_bound`: `loaded.signal != before`, the sealed size being the replacement's.
    """
    from xbrain.rubrics import save_vocab

    _store, vocab_topics, _pages = corpus
    items, vocab, topics = _paths(three_inputs)
    before = index_build.StoreSignal.of(items, vocab, topics)
    longer = [
        t.model_copy(update={"description": t.description + " — rewritten in place"})
        for t in vocab_topics
    ]

    class RewritingHandle:
        def __init__(self, handle):
            self._handle = handle

        def fileno(self):
            return self._handle.fileno()

        def read(self):
            data = self._handle.read()
            save_vocab(longer, vocab)  # in-place: truncates the inode still open here
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    class RewritingPath(type(vocab)):
        def open(self, *args, **kwargs):
            return RewritingHandle(super().open(*args, **kwargs))

    loaded = index_build.load_index_inputs(items, RewritingPath(vocab), topics)

    after = index_build.StoreSignal.of(items, vocab, topics)
    assert after.vocab_yaml_size != before.vocab_yaml_size, "the rewrite really moved the file"
    assert [t.description for t in loaded.vocab] == [t.description for t in vocab_topics], (
        "the parse saw the bytes it read, not the rewrite"
    )
    assert loaded.signal == before, "and the signal is those bytes', not the rewrite's"
    assert loaded.signal != after, "so the index declares itself behind, which is the warning"


# ---------------------------------------------------------------------------
# P1a — THREE INPUTS, NOT ONE, on both sides
# ---------------------------------------------------------------------------


def test_load_index_inputs_reads_all_three_inputs_each_bound_to_its_own_handle(
    three_inputs: Path, corpus
) -> None:
    """P1a at the loader: the index derives from THREE files, so the loader reads three.

    A topic description enters every assigned item's PROFILE (spec §5.1.A) and the overviews
    and notes are chunks the index serves, so a loader that reads `items.json` and defaults
    the other two to empty builds an index missing the whole topic plane while every signal it
    seals says the inputs were read.

    THE THREE SIZES MUST DIFFER, asserted first (rule 1): each entry is checked against its own
    file's `stat`, so an entry filled from a DIFFERENT input's handle would still pass wherever
    the two files happened to agree, and a zero would be indistinguishable from a real reading.

    Seen red under three mutations: `vocab=[], topic_pages={}` with the four vocab/topics
    entries left at zero (the pre-round-05 shape) — `AssertionError` on the vocabulary; and
    the vocabulary's and the topic pages' signal entries each filled from the OTHER input's
    handle — `AssertionError` naming the file whose entry was wrong.
    """
    _store, vocab_topics, pages = corpus
    items, vocab, topics = _paths(three_inputs)
    sizes = [p.stat().st_size for p in (items, vocab, topics)]
    assert len(set(sizes)) == 3, "distinct sizes, or one file's stat could satisfy another's entry"
    assert all(sizes), "and none empty, or a zeroed entry would be indistinguishable from truth"

    loaded = index_build.load_index_inputs(items, vocab, topics)

    assert {t.slug: t.description for t in loaded.vocab} == {
        t.slug: t.description for t in vocab_topics
    }, "descriptions included: they enter every assigned item's profile"
    assert set(loaded.topic_pages) == set(pages), "the topic pages were read"
    assert loaded.topic_pages["agent-evaluation"].overview == pages["agent-evaluation"].overview

    for path, mtime_field, size_field in [
        (items, "items_json_mtime_ns", "items_json_size"),
        (vocab, "vocab_yaml_mtime_ns", "vocab_yaml_size"),
        (topics, "topics_json_mtime_ns", "topics_json_size"),
    ]:
        stat = path.stat()
        assert getattr(loaded.signal, mtime_field) == stat.st_mtime_ns, path.name
        assert getattr(loaded.signal, size_field) == stat.st_size, path.name


@pytest.mark.parametrize(
    "filename, mtime_field, size_field",
    [
        ("items.json", "items_json_mtime_ns", "items_json_size"),
        ("vocab.yaml", "vocab_yaml_mtime_ns", "vocab_yaml_size"),
        ("topics.json", "topics_json_mtime_ns", "topics_json_size"),
    ],
)
def test_the_query_time_signal_watches_each_of_the_three_inputs(
    three_inputs: Path, filename: str, mtime_field: str, size_field: str
) -> None:
    """P1a's OTHER half: `StoreSignal.of` is the side a query pays, and it watched one file.

    This is the comparison every `search` makes against what a manifest sealed. Watching
    `items.json` alone is the round-05 defect: `xbrain topics` writes `topics.json`, never
    touches `items.json`, and every later query answers over the old topic plane with nothing
    declared — the silence spec §9.3 forbids. One case per input, each asserting only its own
    file's two entries, so zeroing ONE stat reddens exactly one case and names which input
    lost its watch. The distinct-and-non-empty precondition is the same rule-1 guard as above.

    Seen red under three mutants applied separately, one per line of `of`: `_stat_signal(…)`
    -> `(0, 0)` for the items, the vocabulary and the topic pages, each reddening its own case
    and leaving the other two green.
    """
    paths = _paths(three_inputs)
    sizes = [p.stat().st_size for p in paths]
    assert len(set(sizes)) == 3, "distinct sizes, or one file's stat could satisfy another's entry"
    assert all(sizes), "and none empty, or a zeroed entry would be indistinguishable from truth"

    signal = index_build.StoreSignal.of(*paths)
    stat = (three_inputs / filename).stat()
    assert getattr(signal, mtime_field) == stat.st_mtime_ns, filename
    assert getattr(signal, size_field) == stat.st_size, filename


def test_the_store_signal_is_one_stat_per_input_and_nothing_else(
    three_inputs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cheap signal must stay cheap, or it is the expensive one under a different name.

    Three stats, all distinct: a signal that grew a second stat per file and a signal that
    stat'ed one input twice in place of another are different regressions, and both move this
    count. The `set` is what catches the second — cross-wiring keeps the total at three.
    """
    real_stat = Path.stat
    calls: list[Path] = []

    def counted(self, *args, **kwargs):
        calls.append(self)
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", counted)
    index_build.StoreSignal.of(*_paths(three_inputs))

    assert len(set(calls)) == len(calls) == 3, f"one stat per input, all distinct: {calls}"


def test_neither_door_can_be_called_without_the_vocabulary_and_the_topic_pages() -> None:
    """THE FIX THIS CHILD EXISTS FOR, AND WITHOUT THIS TEST IT CAN BE REVERTED IN SILENCE.

    Measured on the tree before this test was written: restoring `Path | None = None` on both
    doors, or restoring the four `= 0` defaults on the dataclass, left **18 of 18 GREEN**.
    Every other test passes all three paths explicitly, so every one of them is satisfied just
    as happily by the OPTIONAL signature — and `mypy` runs on `src/` alone (`scripts/check.sh`),
    where a restored default is type-correct at every call site. A hundred per cent of
    statements were covered and not one assertion died when the defect came back. Rule 1 in
    its purest form: the fix could not be made to go red.

    WHAT IS BEING PREVENTED, once more, because it is the whole point: zeros are also what an
    ABSENT file reads as, so a signal that OMITTED the vocabulary was byte-identical to one
    taken over a vocabulary that is not there — and two such signals compare EQUAL forever,
    however `vocab.yaml` changes. The false-negative direction this module is built never to
    fail in, reachable by leaving one argument off the shortest call.

    Each case names the parameter it is missing, so a failure says WHICH half regressed: the
    two doors are the `Path | None` mutant, the constructor is the `= 0` mutant.

    THE CONSTRUCTOR IS SWEPT OVER EVERY ARITY, not probed at one. A first version asserted
    only `StoreSignal(1, 2)`, and a mutant that restored the defaults on the LAST THREE fields
    — leaving `vocab_yaml_mtime_ns` required — still raised there, naming that field, and the
    suite stayed GREEN at 20 of 20. A default on any TRAILING subset is the same defect, so
    the assertion has to be that a signal of fewer than six values does not exist at all.
    """
    path = Path("items.json")

    with pytest.raises(TypeError, match="vocab_path"):
        index_build.StoreSignal.of(path)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="topics_path"):
        index_build.StoreSignal.of(path, path)  # type: ignore[call-arg]

    with pytest.raises(TypeError, match="vocab_path"):
        index_build.load_index_inputs(path)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="topics_path"):
        index_build.load_index_inputs(path, path)  # type: ignore[call-arg]

    for arity in range(6):
        with pytest.raises(TypeError, match="required positional argument"):
            index_build.StoreSignal(*range(arity))  # type: ignore[call-arg]


def test_the_signal_has_six_fields_in_order_none_defaulted_and_is_frozen() -> None:
    """A DEFAULTED SEVENTH FIELD IS THE ROUND-05 DEFECT BACK, AND THE SWEEP ABOVE CANNOT SEE IT.

    Measured: `guardrails_yaml_mtime_ns: int = 0` left 20 of 20 GREEN, because a default changes
    no REQUIRED arity — so the sweep catches the SAFE variant (a required seventh field, red on
    collection) and misses the dangerous one, this child's own thesis failing on itself one input
    later. Zeros are what an ABSENT file reads as, so a defaulted field is a signal that compares
    EQUAL forever over an input nobody passed. ORDER is pinned in the same breath because 02.6b
    serialises this into a manifest, where a reorder that keeps the arity re-binds every value in
    silence; and FROZEN, because a signal re-bindable after the read is not sealed (also 20/20).
    """
    fields = dataclasses.fields(index_build.StoreSignal)
    inputs = ("items_json", "vocab_yaml", "topics_json")
    assert [f.name for f in fields] == [f"{i}_{k}" for i in inputs for k in ("mtime_ns", "size")]
    assert all(f.default is dataclasses.MISSING for f in fields), "no field may carry a default"
    assert all(f.default_factory is dataclasses.MISSING for f in fields)

    with pytest.raises(dataclasses.FrozenInstanceError):
        index_build.StoreSignal(*range(6)).items_json_size = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        index_build.IndexInputs({}, [], {}, ZERO).store = {}  # type: ignore[misc]


def test_an_input_that_is_not_utf8_is_refused_not_repaired(three_inputs: Path) -> None:
    """Undecodable bytes are a REFUSAL, and the refusal is the only thing standing there.

    Measured before this test existed: `data.decode("utf-8")` weakened to
    `errors="replace"` left **18 of 18 GREEN**. That mutation does not fail — it SUCCEEDS,
    turning undecodable bytes into U+FFFD and handing them on to be indexed as if they were
    the corpus. It is the fail-open family this module exists to close, one layer below the
    one the rest of the file guards: not an unreadable file read as empty, but a corrupt file
    read as CONTENT.

    The two doors are asserted TOGETHER because `_read_bound`'s docstring claims they agree
    here — unlike `ENOTDIR` and `ELOOP`, where it is deliberately stricter — and a claim of
    agreement is worth exactly what the assertion that both raise is worth.

    A third assertion stood here — `not isinstance(caught.value, OSError)` — and it is gone:
    `pytest.raises(UnicodeDecodeError)` already fixes the type, so it could not fail (rule 1).
    """
    from xbrain.rubrics import load_vocab

    items, vocab, topics = _paths(three_inputs)
    vocab.write_bytes(b"topics:\n- slug: a\n  description: \xff\xfe not utf-8\n")

    with pytest.raises(UnicodeDecodeError):
        index_build.load_index_inputs(items, vocab, topics)

    with pytest.raises(UnicodeDecodeError):
        load_vocab(vocab)


# ---------------------------------------------------------------------------
# A-2 — the asymmetry: a query always ANSWERS, a load never invents an empty store
# ---------------------------------------------------------------------------


def test_a_missing_input_is_zeroed_on_the_query_side_and_empty_on_the_load_side(
    tmp_path: Path,
) -> None:
    """ABSENT is the one reading that stays empty — `load_store`'s own semantics — so a fresh
    checkout with no `data/` still loads and still says the index is behind. Both doors are
    asserted here because they are the two halves of one promise: the query answers, the load
    yields the empty value, and only what EXISTS and cannot be read parts them (below)."""
    paths = _paths(tmp_path / "nothing")

    assert index_build.StoreSignal.of(*paths) == ZERO
    loaded = index_build.load_index_inputs(*paths)
    assert (loaded.store, loaded.vocab, loaded.topic_pages) == ({}, [], {})
    assert loaded.signal == ZERO


@pytest.mark.parametrize(
    "index, mtime_field, size_field",
    [
        (0, "items_json_mtime_ns", "items_json_size"),
        (1, "vocab_yaml_mtime_ns", "vocab_yaml_size"),
        (2, "topics_json_mtime_ns", "topics_json_size"),
    ],
)
def test_an_input_that_cannot_be_stat_ed_answers_for_a_query_and_raises_for_a_load(
    three_inputs: Path, index: int, mtime_field: str, size_field: str
) -> None:
    """The two directions of A-2, on ONE obstruction, which is the only way to see the pair.

    `_stat_signal` swallows every `OSError`, not only `FileNotFoundError`, and this is the
    test that holds it: a query must be able to ANSWER — declaring the index behind — instead
    of learning about the filesystem by exception from inside `search`. What it answers is
    `UNSTATTABLE`, NOT the zeros an absent input reads as — the round-09 split, whose
    consequence is the transition test below. The obstruction is a
    path standing INSIDE a regular file, which raises `NotADirectoryError` (`ENOTDIR`); the
    two obstacles the loader test below uses cannot reach this promise, because `stat` needs
    neither read permission on a file nor the file to be a file, so both of them stat fine and
    the breadth of the `except` stayed unguarded by every test in the snapshot this re-cuts.

    The OTHER TWO inputs must come back non-zero, asserted here (rule 1): a `StoreSignal.of`
    that gave up and returned zeros everywhere would satisfy the first half on its own, and a
    call that never ran at all would satisfy it too.

    Seen red under the mutation `except OSError` -> `except FileNotFoundError` in
    `_stat_signal`: `NotADirectoryError` out of `StoreSignal.of`, on each of the three cases.

    The SAME path is an error for `load_index_inputs`, and that is the design, not an
    inconsistency: an unreadable input is not an empty one, so the loader lets it through.
    """
    paths = list(_paths(three_inputs))
    paths[index] = paths[index] / "not-a-directory"

    signal = index_build.StoreSignal.of(*paths)
    obstructed = (getattr(signal, mtime_field), getattr(signal, size_field))
    assert obstructed == index_build.UNSTATTABLE, "obstructed, and not the ABSENT zeros"
    others = [
        (m, s)
        for k, (m, s) in enumerate(
            [
                (signal.items_json_mtime_ns, signal.items_json_size),
                (signal.vocab_yaml_mtime_ns, signal.vocab_yaml_size),
                (signal.topics_json_mtime_ns, signal.topics_json_size),
            ]
        )
        if k != index
    ]
    assert all(m > 0 and s > 0 for m, s in others), f"the other two still stat'ed: {others}"

    with pytest.raises(NotADirectoryError):
        index_build.load_index_inputs(*paths)


def test_a_signal_sealed_over_an_absent_input_is_unequal_once_it_cannot_be_stat_ed(
    tmp_path: Path,
) -> None:
    """ABSENT AND OBSTRUCTED MUST NOT BE THE SAME VALUE (Codex round 09, the blocking HIGH).

    Both read `(0, 0)` before the split, and `_read_bound` SEALS zeros for an absent input — so
    an index built while an input was missing compared EQUAL to a query taken once that same path
    could no longer be stat'ed, certifying itself current over an input it never opened. Seen red
    at `8febb37`: `sealed == current`, both all-zero, while the loader raised on the same path.

    The obstruction is a self-referential symlink AT THE SEALED PATH (`ELOOP`), so the only thing
    that moves between the two readings is whether that input can be stat'ed. The pair is one
    promise, so it is one test: the query ANSWERS (what `search` pays on every call must never
    raise from inside the filesystem) and the load RAISES (an unreadable input is not an empty).

    THE SENTINEL'S SAFETY IS ASSERTED ON THE SIZE, NEVER ON THE MTIME, and asserting only
    `== UNSTATTABLE` would be rule 1's row 4 — both sides out of the same module, satisfied by
    whatever the constant says. Measured: an empty file at `os.utime(p, ns=(-1, -1))` stats to
    exactly `(-1, 0)`, so a `-1` MTIME is forgeable by a real input and `UNSTATTABLE = (-1, 0)`
    left this file GREEN at 26 of 26. A negative SIZE is not forgeable, which is the contract.
    """
    items, vocab, topics = _paths(tmp_path)
    sealed = index_build.load_index_inputs(items, vocab, topics).signal
    assert sealed == ZERO, "an absent input still seals zeros, and that stays true"

    items.symlink_to(items)

    current = index_build.StoreSignal.of(items, vocab, topics)
    assert current != sealed, "absent at seal must not read as unchanged once obstructed"
    assert (current.items_json_mtime_ns, current.items_json_size) == index_build.UNSTATTABLE
    assert index_build.UNSTATTABLE[1] < 0, "the SIZE half is what a real `st_size` cannot be"
    with pytest.raises(OSError):
        index_build.load_index_inputs(items, vocab, topics)


def _obstruct(path: Path, obstacle: str) -> None:
    """Make `path` UNREADABLE without removing it — the two shapes an operator meets."""
    if obstacle == "chmod000":
        path.chmod(0)
    else:
        path.unlink()
        path.mkdir()


@pytest.mark.parametrize(
    "obstacle, error", [("chmod000", PermissionError), ("directory", IsADirectoryError)]
)
@pytest.mark.parametrize("filename", ["items.json", "vocab.yaml", "topics.json"])
def test_an_unreadable_input_is_an_error_never_an_empty_one(
    three_inputs: Path, filename: str, obstacle: str, error: type[OSError]
) -> None:
    """A-2 (gate Fable §5.2, round 08), on ALL THREE inputs and not only the one it probed.

    `_read_bound` caught EVERY `OSError` and answered `(None, 0, 0)`, the reading reserved for
    a file that is ABSENT, so an unreadable input loaded as an EMPTY one — the whole fail-open
    family with a destruction on top, enumerated in `_read_bound`'s own docstring and not
    restated here. The defect is a property of that function and all three inputs route
    through it, which is why this is parametrised over the three and not over the one the gate
    happened to probe: on `vocab.yaml` it is quieter and no less destructive, loading as «no
    topics» so every profile is built without its topic descriptions and `update` plans the
    deletion of the whole topic plane, exit 0.

    Seen red under the mutation `except FileNotFoundError` -> `except OSError` in
    `_read_bound`: `DID NOT RAISE` on all six cases.

    `chmod000` ASSUMES A NON-ROOT RUNNER — root ignores the permission bits and the open would
    succeed, so the case would report `DID NOT RAISE`. That is a FALSE RED, which is the safe
    direction, and it does not arise here: `quality.yml` runs on `ubuntu-latest` with no
    `container:`, so the job is the unprivileged `runner` user. The `directory` case needs no
    such caveat.
    """
    _obstruct(three_inputs / filename, obstacle)
    try:
        with pytest.raises(error):
            index_build.load_index_inputs(*_paths(three_inputs))
    finally:
        if obstacle == "chmod000":
            (three_inputs / filename).chmod(0o644)


# ---------------------------------------------------------------------------
# Rule 5 — one parser per input, shared by the two readers of the same bytes
# ---------------------------------------------------------------------------


def test_the_index_loader_and_the_store_doors_parse_through_one_parser_each(
    three_inputs: Path,
) -> None:
    """The seam `parse_store` / `parse_vocab` / `parse_topic_pages` exists FOR THIS (rule 5).

    `load_index_inputs` must bind what it parsed to the exact bytes it read, so it reads
    through its own handle — and the only honest way to do that without a second copy of every
    parse is to split the parse out of `load_store` / `load_vocab` / `load_topic_pages` and
    share it. The property that matters is not "the seam exists" (a tautology the moment it
    does) but that the TWO READERS OF THE SAME BYTES agree: the path-based door every other
    stage uses, and the handle-based one the index uses.

    WHAT IT CANNOT CATCH, SAID OUT LOUD (rule 1): once both doors delegate, a defect INSIDE a
    shared parser moves both sides of every assertion here identically and this test stays
    green. Measured — `parse_vocab` reduced to `data.get("topics")` (raw dicts instead of
    `Topic`) leaves it PASSING, and reddens the two tests above that compare against typed
    values instead. What this one pins is the property it names, DIVERGENCE: seen red under a
    loader that parsed the store with a bare `json.loads` (`Item` out of one door, dicts out
    of the other) and under a loader that defaulted the vocabulary and the topic pages away.

    AND IT COMPARES THE DOORS ONLY ON FILES THAT EXIST, which is the whole domain where they
    are meant to agree. On a path standing inside a regular file, or on a symlink loop, they
    DIVERGE BY DESIGN — `load_store` gates on `Path.exists()` and reads both as `{}`, while
    this loader raises — and `_read_bound`'s docstring is where that asymmetry is argued.
    """
    from xbrain.rubrics import load_vocab
    from xbrain.store import load_store, load_topic_pages

    items, vocab, topics = _paths(three_inputs)
    loaded = index_build.load_index_inputs(items, vocab, topics)

    assert loaded.store == load_store(items)
    assert loaded.vocab == load_vocab(vocab)
    assert loaded.topic_pages == load_topic_pages(topics)


@pytest.mark.parametrize(
    "name, door, key",
    [
        ("parse_store", "load_store", "store"),
        ("parse_vocab", "load_vocab", "vocab"),
        ("parse_topic_pages", "load_topic_pages", "topic_pages"),
    ],
)
def test_the_door_and_the_index_loader_share_one_parser_object(
    three_inputs: Path, monkeypatch: pytest.MonkeyPatch, name: str, door: str, key: str
) -> None:
    """Rule 5 BOUND IN CODE: re-inlining EITHER side left the test above at 20 of 20 GREEN.

    That test compares the two readers' OUTPUTS, so a byte-identical second copy satisfies it.
    Three legs here, each reddening a mutation the other two miss, all three measured. The DOOR
    resolves its parser in its own module at call time, so a re-inlined `load_store` fails leg 1.
    The LOADER is unreachable that way at all — `from xbrain.store import parse_store` binds the
    OBJECT into `index_build`'s namespace at IMPORT time — so leg 2 needs its own patch. And
    neither can tell ONE parser from TWO that behave alike, which is what rule 5 forbids, so leg
    3 is identity: NOT rule 1's banned tautology precisely because legs 1 and 2 patch two
    DIFFERENT names — measured, `index_build` shadowing `parse_store` with its own byte-identical
    copy passes both of them and reddens exactly here.
    """
    import xbrain.rubrics as rubrics
    import xbrain.store as store

    home = rubrics if name == "parse_vocab" else store
    path = dict(zip(("store", "vocab", "topic_pages"), _paths(three_inputs)))[key]
    marker = object()

    monkeypatch.setattr(home, name, lambda text: marker)
    assert getattr(home, door)(path) is marker, "the door re-inlined its parse"

    monkeypatch.setattr(index_build, name, lambda text: marker)
    loaded = index_build.load_index_inputs(*_paths(three_inputs))
    assert getattr(loaded, key) is marker, "the index loader re-inlined its parse"

    monkeypatch.undo()
    assert getattr(index_build, name) is getattr(home, name), "two parsers, not one"


def test_the_stat_guard_is_no_broader_than_os_error() -> None:
    """Widening `_stat_signal`'s `except OSError` to `except Exception` left 20 of 20 GREEN.

    The breadth is deliberate but BOUNDED: under `except Exception` a `None` path — #161's own
    signature defect — is swallowed into a signal instead of raising, the fail-open family this
    module exists to close. Reached through the public door, so it pins observable behaviour.
    """
    with pytest.raises(AttributeError):
        index_build.StoreSignal.of(None, Path("v"), Path("t"))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The DEEP plane (Plan 02 §2, child 02.6a2a). Every encoder guard below is asserted ON THE
# ENCODER, not through a fingerprint that happens to differ: two hashes differing tells you
# nothing about WHY, and rule 1's fourth row is the assertion whose two sides both come out
# of the thing under test.
# ---------------------------------------------------------------------------


def _item(**overrides) -> Item:
    """The smallest real item, so a guard can vary ONE field and nothing else."""
    defaults = dict(
        id="42",
        source="bookmark",
        url="https://x.com/a/status/42",
        author=Author(handle="a", name="A"),
        text="the post body",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        captured_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    return Item(**{**defaults, **overrides})


def _photo(**overrides) -> MediaPhotoDescribed:
    """A described photo, so a guard can vary only what makes it CONTENT-bearing."""
    defaults = dict(
        url="https://pbs.example/1.jpg",
        local_path="42/0.jpg",
        width=1,
        height=1,
        bytes_size=1,
        downloaded_at=datetime(2026, 1, 3, tzinfo=UTC),
        description="a described photo",
        description_lang="Spanish",
        is_decorative=False,
        description_version="v1",
        described_at=datetime(2026, 1, 3, tzinfo=UTC),
    )
    return MediaPhotoDescribed(**{**defaults, **overrides})


def _video_content(text: str = "", has_speech: bool | None = False) -> Content:
    """One `x_video` source, the only shape `no_speech` is derived from."""
    return Content(
        fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
        sources=[
            ContentSourceSuccess(
                kind="x_video", url="https://video.example/1", text=text, has_speech=has_speech
            )
        ],
    )


def _surface(**overrides) -> KnowledgeSurface:
    """A surface whose fifteen persisted values are all DISTINCT and self-naming."""
    defaults = dict(
        surface_id="item:42:post:0",
        owner_type="item",
        owner_id="42",
        surface_type="post",
        text="0123456789",
        title="THE-TITLE",
        origin="source",
        trust_class="primary_source",
        derived=True,
        attribution=Author(handle="THE-HANDLE", name="THE-NAME"),
        locator=Locator(kind="item_text", url="THE-LOCATOR-URL"),
        fingerprint="a" * 64,
        language="THE-LANGUAGE",
    )
    return KnowledgeSurface(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# `_canonical` — what the injectivity claim covers, and where it stops
# ---------------------------------------------------------------------------


def test_the_canonical_encoding_is_injective_where_ensure_ascii_would_not_be() -> None:
    """`ensure_ascii=True` collides a lone surrogate PAIR with the astral character it spells —
    `json.dumps` emits the same escape for both — so the setting is a correctness choice, and
    prose alone never held it: the mutant `False` -> `True` left the whole file GREEN. Asserted
    on the ENCODER because the surrogate side raises at `_sha256`'s `.encode("utf-8")`, which
    is fail-closed and not a hash to compare.
    """
    assert index_build._canonical("t", ["\ud83d\ude00"]) != index_build._canonical(
        "t", ["\U0001f600"]
    )


def test_the_domain_travels_at_the_head_so_two_planes_cannot_serialise_alike() -> None:
    """The collision the `domain` argument closed, reproduced on the ENCODER: a one-item store
    and a one-entry vocabulary both encoded as `[["a", <64 hex>]]`, measured EQUAL. Asserting
    two *fingerprints* differ would not pin it — they differ for many reasons and would keep
    differing if the domain moved somewhere it does not separate — so the payload is fixed, the
    domains vary, and the position of the domain is read back out of the blob. Which domain
    each FUNCTION passes is pinned separately, by reading it off a real call.
    """
    collided = [["a", "0" * 64]]
    assert index_build._canonical("store", collided) != index_build._canonical("vocab", collided)
    assert json.loads(index_build._canonical("store", collided))[0] == "store"


def test_the_variadic_regions_are_nested_so_a_splice_cannot_reproduce_them() -> None:
    """Two item states that a FLATTENED payload encodes identically: `topics=("thread",)` with
    no sources, against no topics with one blank `thread` source — spliced into one list both
    read `[..., "thread", ...]`, nested the boundary is structural. An ENCODER property: it
    shows why nesting is sufficient, not that `item_fingerprint` nests, which the atom-by-atom
    guard below is what reddens under the splice.
    """
    spliced_alike = (["thread"], []), ([], ["thread"])
    first, second = spliced_alike
    assert sum(first, []) == sum(second, [])
    assert index_build._canonical("item", list(first)) != index_build._canonical(
        "item", list(second)
    )


def test_a_nul_inside_a_hashed_value_cannot_forge_the_boundary_between_two_atoms() -> None:
    """The delimiter family, and why the join this encoder replaced had to go: it joined atoms
    with a NUL and framed nothing below the region, and both store writers persist a NUL, so a
    stored value could re-split the stream and move every later boundary. These two payloads
    collide under that join and separate under `_canonical`; the quote and the bracket are the
    same attack in the characters JSON itself uses.
    """
    forged = ["a\0b", "c"]
    honest = ["a", "b\0c"]
    assert "\0".join(forged) == "\0".join(honest)
    assert index_build._canonical("t", forged) != index_build._canonical("t", honest)
    assert index_build._canonical("t", ['a"],["b']) != index_build._canonical("t", ["a", "b"])


def test_the_injectivity_claim_is_scoped_to_the_domain_the_payloads_build() -> None:
    """The three limits the docstring names, pinned so nobody builds a payload assuming
    otherwise. JSON is NOT injective over Python values in general: a sequence's TYPE is not a
    distinguishing feature, a mapping's KEY TYPE is not either, and a float has values no
    reader could parse back. The first two are why the claim is SCOPED rather than stated flat;
    the third is why `allow_nan=False` is there.
    """
    assert index_build._canonical("t", (1, 2)) == index_build._canonical("t", [1, 2])
    assert index_build._canonical("t", {1: "a"}) == index_build._canonical("t", {"1": "a"})
    with pytest.raises(ValueError):
        index_build._canonical("t", [float("nan")])


def test_a_lone_surrogate_is_refused_by_the_hash_and_never_replaced() -> None:
    """Fail closed. `errors="replace"` would hash U+FFFD and call two different strings the
    same content — the fail-open family the whole module exists to close.
    """
    with pytest.raises(UnicodeEncodeError):
        index_build._sha256(index_build._canonical("t", ["\ud800"]))


def test_absent_and_empty_are_not_the_same_atom() -> None:
    """`None` is a column that holds NULL; `""` is a column that holds a string. The index
    persists both, `search` serves both, and an encoder that folded them would make a
    repaired-to-blank value read as never-set.
    """
    assert index_build._canonical("t", [None]) != index_build._canonical("t", [""])
    assert index_build._canonical("t", [None]) != index_build._canonical("t", [[]])


# ---------------------------------------------------------------------------
# `surface_row` — the fifteen columns, by value and by arity
# ---------------------------------------------------------------------------


def _schema_columns(table: str) -> list[str]:
    """The column NAMES the DDL declares for one table, read out of `_SCHEMA` itself.

    Read from the schema string rather than from a hand-copied list: a column added to
    `surfaces` has to reach this test without anyone remembering to edit it.
    """
    body = index_schema._SCHEMA.split(f"CREATE TABLE IF NOT EXISTS {table} (", 1)[1]
    body = body.split(");", 1)[0]
    return [
        line.strip().split()[0]
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]


def test_the_surface_row_is_the_fifteen_values_the_schema_persists_in_order() -> None:
    """Every column pinned BY VALUE, against a literal written by hand.

    Review #161 measured seven of the fifteen unpinned: a swap between two same-typed
    neighbours (title/url, handle/name, language/fingerprint) moved what the index STORES and
    reddened nothing. The expected side is a literal, not `surface_row` recomputed, so the two
    sides do not come out of one function (rule 1, fourth row). EVERY VALUE IS THE ONE A MUTANT
    WOULD NOT INSTALL — which is why `derived` is True and not the False a surface usually
    carries: pinned at False the column reads 0, a constant-0 mutant installs 0, and the guard
    passes. The count is a property of the mutant and not of the test: under column DELETION an
    arity assertion catches all fifteen and says nothing about any value, so "N unpinned"
    without the mutant beside it is the error rule 2 exists to stop. Nullability and cell TYPE
    are pinned separately below — `None` folded to `""`, and `True`/`1.0` for `1`, are what
    this literal cannot see.
    """
    surface = _surface()
    assert index_build.surface_row(surface) == (
        "item:42:post:0",
        "item",
        "42",
        "post",
        "source",
        "primary_source",
        1,
        "THE-HANDLE",
        "THE-NAME",
        "THE-TITLE",
        "THE-LOCATOR-URL",
        surface.locator.model_dump_json(),
        "THE-LANGUAGE",
        "a" * 64,
        10,
    )


def test_the_row_has_one_value_per_column_the_schema_declares() -> None:
    """A totality guard, and the only mechanical one this child can offer.

    It catches a column ADDED to `surfaces` without a value beside it. It does NOT catch a
    column REORDERED into a same-typed neighbour — that needs the readback test 02.7 owes,
    written against a writer that does not exist in this tree yet.
    """
    assert len(_schema_columns("surfaces")) == 15
    assert len(index_build.surface_row(_surface())) == len(_schema_columns("surfaces"))


def test_the_last_column_is_a_length_and_never_the_body() -> None:
    """Spec §10.8: the index stores what it needs to filter, not a second copy of the text.
    The body is covered by `fingerprint`, which hashes it — so two bodies of the same length
    still separate.
    """
    row = index_build.surface_row(_surface(text="x" * 10))
    assert row[-1] == 10
    assert "x" * 10 not in row
    assert index_build.surface_row(
        _surface(text="abcdefghij", fingerprint="b" * 64)
    ) != index_build.surface_row(_surface(text="0123456789", fingerprint="a" * 64))


def test_the_rows_region_is_a_set_because_one_item_cannot_emit_a_surface_id_twice(
    corpus,
) -> None:
    """What the rows region actually rests on — and it is NOT its order.

    `surface_id` is `<owner>:<id>:<type>:<source_key>` and unique per item, so the rows are a
    SET and their sequence carries nothing the set does not: the mutant `sorted(...)` around
    the region leaves this file GREEN, which is correct rather than a gap. What WOULD make the
    order load-bearing is a duplicate id, and that is what this pins across the whole corpus —
    two distinct surfaces on one row would quietly end the argument above.
    """
    store, _vocab, _pages = corpus
    for item in store.values():
        ids = [surface.surface_id for surface in item_surfaces(item)]
        assert len(ids) == len(set(ids)), item.id


# ---------------------------------------------------------------------------
# `item_fingerprint` — the four persisted planes it must reach (rule 6)
#
# Each guard varies ONE field and asserts what the projection does with it, so a green run
# says which plane is covered rather than that two hex strings differ.
# ---------------------------------------------------------------------------


def test_the_item_fingerprint_changes_when_indexable_text_changes(corpus) -> None:
    """The floor: the body the index serves is inside the hash."""
    store, _vocab, _pages = corpus
    item = next(iter(store.values()))
    assert index_build.item_fingerprint(item) != index_build.item_fingerprint(
        item.model_copy(update={"text": item.text + " edited"})
    )


def test_the_item_fingerprint_covers_the_source_failures_plane() -> None:
    """HIGH-1 of review #161, half one — and the reason this child exists.

    `source_failures` is written per item and read back by `get`, and before this nothing about
    it moved a fingerprint: a fetch whose recorded error changed rewrote the row on disk while
    `update` reported the item unchanged (rule 6). The two items differ in NOTHING else — a
    failure emits no surface and appears in no content kind — so the moved plane is the only
    thing that can move the hash.
    """

    def failed(error: str) -> Item:
        return _item(
            content=Content(
                fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
                sources=[
                    ContentSourceFailure(
                        kind="external_article",
                        url="https://e.example",
                        failure_reason="not_found",
                        error=error,
                    )
                ],
            )
        )

    failing, repaired = failed("one"), failed("two")
    assert item_surfaces(failing) == item_surfaces(repaired)
    assert failed_sources(failing) != failed_sources(repaired)
    assert index_build.item_fingerprint(failing) != index_build.item_fingerprint(repaired)


def test_the_item_fingerprint_covers_the_unfetched_links_plane() -> None:
    """HIGH-1 of review #161, half two.

    `item.links` reaches the index through `unfetched_links` and through NOTHING else — no
    surface, no content kind, no other hashed atom — so a link that changed and a
    fingerprint that did not was the whole defect, and this pair isolates it exactly.
    """
    before = _item(links=[Link(url="https://a.example", domain="a.example")])
    after = _item(links=[Link(url="https://b.example", domain="b.example")])
    assert item_surfaces(before) == item_surfaces(after)
    assert unfetched_links(before) != unfetched_links(after)
    assert index_build.item_fingerprint(before) != index_build.item_fingerprint(after)


def test_a_field_added_to_either_failure_projection_is_hashed_without_being_remembered() -> None:
    """The structural half of the HIGH-1 fix, asserted as a totality: a hand-written field list
    is what let those two planes drift, so `_model_atoms` walks `model_fields`. This pins that
    it walks ALL of them, in order, with the name beside the value — a rename is a schema
    change and has to move the hash too.
    """
    failure = SourceFailure(
        kind="external_article", url="u", failure_reason="not_found", http_status=404
    )
    assert [name for name, _value in index_build._model_atoms(failure)] == list(
        SourceFailure.model_fields
    )
    assert index_build._model_atoms(failure)[-1] == ["http_status", 404]
    link = UnfetchedLink(url="u", reason="not_attempted")
    assert [name for name, _value in index_build._model_atoms(link)] == list(
        UnfetchedLink.model_fields
    )


def test_each_failure_plane_persists_exactly_what_its_public_projection_carries() -> None:
    """The binding that makes the two failure DDLs and the ONE versioned projection agree.

    `item_fingerprint` hashes the projection, so a column the DDL declares and the projection
    does not carry is one no fingerprint can ever see: `source_failures` declared `attempts`,
    `SourceFailure` had no field for it, and an item whose attempt count moved certified as
    unchanged. The column is gone (`SCHEMA_VERSION` "4") and this stops the gap reopening from
    EITHER side. The DDL side is read out of `_SCHEMA` and the model side off `model_fields`,
    so neither is hand-copied and the two come from different modules; `item_id` is the join
    key, carried by `store_fingerprint`'s `(key, hash)` pair rather than by the projection.
    """
    for table, model in (("source_failures", SourceFailure), ("unfetched_links", UnfetchedLink)):
        assert set(_schema_columns(table)) == {"item_id", *model.model_fields}, table


def test_the_item_fingerprint_covers_the_primary_topic_the_topics_tuple_hides() -> None:
    """Two enrichments the persisted `items.primary_topic` column separates and
    `item_topics()` does not.

    `item_topics` puts the primary first and then DEDUPLICATES, so `(None, ["a", "b"])` and
    `("a", ["b"])` both collapse to `("a", "b")`. The equal side is asserted on purpose:
    without it a green run would not say the tuple hides anything, and the guard would read as
    a restatement of "different enrichment, different hash".
    """

    def enriched(primary: str | None, topics: list[str]) -> Item:
        return _item(
            enriched=Enrichment(
                enriched_at=datetime(2026, 1, 3, tzinfo=UTC),
                executor="manual",
                primary_topic=primary,
                topics=topics,
            )
        )

    hidden, declared = enriched(None, ["a", "b"]), enriched("a", ["b"])
    assert item_topics(hidden) == item_topics(declared) == ("a", "b")
    assert index_build.item_fingerprint(hidden) != index_build.item_fingerprint(declared)


def test_the_item_fingerprint_covers_the_media_the_emitter_declined_to_surface() -> None:
    """`items.skipped_decorative` moves with no surface behind it.

    `xbrain describe` classifying a photo as decorative flips the counter 0 -> 1 and emits
    NOTHING — no surface, no content kind — so before the counters were hashed the row on disk
    changed and the item read as unchanged. The equal `item_surfaces` is what makes this a
    measurement rather than a restatement. The mutant that DROPS `entry.is_decorative or` and
    keeps `not entry.description` survives, and no test can kill it: `MediaPhotoDescribed`
    validates `is_decorative => not description` at the TYPE boundary (`models.py:318`), so the
    input that would separate the two clauses cannot be constructed. An equivalent mutant, not
    a gap — the clause is kept because the emitter's own filter reads the flag.
    """
    bare = _item()
    decorated = _item(media=[_photo(description="", is_decorative=True)])
    assert item_surfaces(bare) == item_surfaces(decorated)
    assert index_build.declined_media(bare) == (0, 0)
    assert index_build.declined_media(decorated) == (1, 0)
    assert index_build.item_fingerprint(bare) != index_build.item_fingerprint(decorated)


def test_a_silent_video_is_counted_by_the_predicate_the_store_records() -> None:
    """`items.skipped_no_speech`, pinned against the two mutants that leave it silent.

    Asserting only that a silent video moves the hash passes under a swapped predicate —
    `has_speech is False` traded for "the transcript is blank" — because on an ordinary fixture
    the two agree. The `[music]` case is where they part: a transcriber that heard no speech
    and wrote a marker anyway is `has_speech=False` with non-blank text, and only the recorded
    flag gets it right. `has_speech=None` is where `is False` parts from truthiness: UNKNOWN is
    not KNOWN-SILENT, the emitter declines it and this counter must not claim it — the shape
    the corpus-complement guard below names, and 0 of 2,404 today. The counter is asserted
    NON-ZERO, because a slot only ever asserted at 0 is asserted against the constant a mutant
    installs.
    """
    silent = _item(content=_video_content(text="", has_speech=False))
    speaking = _item(content=_video_content(text="hello", has_speech=True))
    assert index_build.declined_media(silent) == (0, 1)
    assert index_build.declined_media(speaking) == (0, 0)
    assert index_build.declined_media(_item(content=_video_content("[music]", False))) == (0, 1)
    assert index_build.declined_media(_item(content=_video_content("", None))) == (0, 0)
    assert index_build.item_fingerprint(silent) != index_build.item_fingerprint(_item())


def test_a_photo_described_as_nothing_is_declined_like_a_decorative_one() -> None:
    """The second clause of the decorative sum, which makes it the COMPLEMENT of
    `iter_described_photos`'s seam rather than a reading of one flag: a photo the vision pass
    had nothing to say about emits no surface either, so dropping `or not entry.description`
    would leave it declined by the emitter and recorded by neither.
    """
    empty = _item(media=[_photo(description="", is_decorative=False)])
    assert list(iter_described_photos(empty)) == []
    assert index_build.declined_media(empty) == (1, 0)
    assert index_build.item_fingerprint(empty) != index_build.item_fingerprint(_item())


def test_the_declined_counters_are_the_emitters_complement_over_the_corpus(corpus) -> None:
    """The rule-5 guard: two hand-written readings of one fact, held to each other.

    `declined_media` decides a photo by `is_decorative or not description` and a video by
    `has_speech is False`; the EMITTER decides by `iter_described_photos`'s filter and by a
    blank transcript. Nothing in code binds the two, so this asserts the property that makes
    them one — declined plus emitted equals present — over every item. They agree on the live
    store too: measured 2026-09-03 over 2,404 items (sha256 `f76341a3...`), 14 declined photos,
    108 silent videos, 0 items where either sum fails. The agreement is CONTINGENT: an
    `x_video` with `has_speech=None` and no text (the HLS path, or a transcriber exiting clean
    with nothing) is declined by the emitter and not seen by this counter. Zero today — a fact
    about the corpus, not an invariant.
    """
    store, _vocab, _pages = corpus
    for item in store.values():
        decorative, no_speech = index_build.declined_media(item)
        described = sum(1 for entry in item.media if isinstance(entry, MediaPhotoDescribed))
        assert decorative + sum(1 for _i, _p in iter_described_photos(item)) == described, item.id
        videos = [source for _i, source in iter_content_sources(item, {"x_video"})]
        emitted = sum(1 for source in videos if source.text and source.text.strip())
        assert no_speech + emitted == len(videos), item.id


def test_a_bookmark_folder_that_is_absent_is_not_one_that_is_empty() -> None:
    """The mutation two reviewers measured reddening NOT ONE test on the snapshot:
    `item.bookmark_folder` -> `item.bookmark_folder or ""`. It is a nullable column and the two
    states are different rows — folding them means a bookmark moved OUT of every folder reads
    as one that was never in a folder, and `update` reports the item unchanged.
    """
    assert index_build.item_fingerprint(
        _item(bookmark_folder=None)
    ) != index_build.item_fingerprint(_item(bookmark_folder=""))


def test_the_item_fingerprint_covers_the_filterable_metadata_no_surface_carries() -> None:
    """A changed author changes what `--author` returns with no text moved (A-1). Handle and
    name are hashed APART: they are two columns, and one person renaming themselves is not
    the same row as a different account.
    """
    base = _item()
    for update in (
        {"author": Author(handle="b", name="A")},
        {"author": Author(handle="a", name="B")},
        {"source": "own_tweet"},
        {"url": "https://x.com/a/status/43"},
        {"created_at": datetime(2026, 1, 9, tzinfo=UTC)},
        {"captured_at": datetime(2026, 1, 9, tzinfo=UTC)},
    ):
        assert index_build.item_fingerprint(base) != index_build.item_fingerprint(
            base.model_copy(update=update)
        ), update


def test_the_item_fingerprint_moves_when_the_content_sources_are_permuted(corpus) -> None:
    """M4. Order is not decoration: `locator.source_index` points at a position in
    `content.sources`, so permuting them repoints every locator the index serves.

    The mechanism is asserted beside the hash, because the hash alone would also move if the
    rows had merely been reordered — and this property is about the locators, which is the
    half a reader has to be able to check by hand.
    """
    store, _vocab, _pages = corpus
    item = next(i for i in store.values() if i.content is not None and len(i.content.sources) > 1)
    flipped = item.model_copy(
        update={
            "content": item.content.model_copy(
                update={"sources": list(reversed(item.content.sources))}
            )
        }
    )
    assert [(s.surface_id, s.locator.source_index) for s in item_surfaces(item)] != [
        (s.surface_id, s.locator.source_index) for s in item_surfaces(flipped)
    ]
    assert index_build.item_fingerprint(item) != index_build.item_fingerprint(flipped)


def test_the_index_options_are_carried_inert_and_02_7_is_what_must_redden_this() -> None:
    """A CHARACTERIZATION of a deliberate hole, not a property anyone wants to keep.
    `IndexOptions` travels so the signature 02.7 consumes is already the ported one and is read
    NOWHERE — two options differing in both fields hash alike. When 02.7 gives them their first
    consumer this goes RED and the change is made on purpose, which is why it is written down.
    """
    item = _item()
    assert index_build.item_fingerprint(
        item, options=index_build.IndexOptions(params=ChunkerParams(target=1), vault_dir=Path("/x"))
    ) == index_build.item_fingerprint(item)
    assert index_build.store_fingerprint(
        {item.id: item}, options=index_build.IndexOptions(params=ChunkerParams(target=1))
    ) == index_build.store_fingerprint({item.id: item})
    assert [f.name for f in dataclasses.fields(index_build.IndexOptions)] == [
        "params",
        "vault_dir",
    ]
    assert index_build.IndexOptions().params == DEFAULT_CHUNKER_PARAMS
    assert index_build.IndexOptions().vault_dir is None


def test_the_emitter_version_is_the_belt_for_an_item_with_no_surfaces_at_all() -> None:
    """`SURFACE_VERSION` leads the payload so a bump invalidates even an item whose surface rows
    are empty — there is nothing else on such an item for a bump to move.

    The expected side is the payload ATOM BY ATOM, so this is the ARITY guard and the readable
    account of what a fingerprint is made of. It is NOT a positional guard, and the earlier
    claim that it was is WITHDRAWN: on a bare item the last seven atoms degenerate to
    `None, None, [], [], [], [], []`, and four same-shaped region swaps were measured leaving
    this file and the full suite green. Position is pinned by the populated literal below.
    """
    bare = _item(text="")
    assert item_surfaces(bare) == ()
    assert index_build.item_fingerprint(bare) == index_build._sha256(
        index_build._canonical(
            "item",
            [
                SURFACE_VERSION,
                "42",
                "bookmark",
                "https://x.com/a/status/42",
                "a",
                "A",
                "2026-01-01T00:00:00+00:00",
                "2026-01-02T00:00:00+00:00",
                None,
                None,
                [],
                [],
                [],
                [],
                [],
                [],
                [0, 0],
            ],
        )
    )


def _rich(*, parts: tuple[str, ...] = ("abc", "defg"), **overrides) -> Item:
    """An item that populates EVERY variadic region, each with a DIFFERENT value — which is
    what the bare item above cannot do. The sources are inserted in the order `sorted()` does
    NOT produce and the topics in the order `item_topics` does NOT return, so both deliberate
    normalisations are pinned; and every variadic region carries TWO entries, so truncating one
    to its first is visible.
    """
    defaults = dict(
        bookmark_folder="THE-FOLDER",
        links=[
            Link(url="https://one.example/", domain="one.example"),
            Link(url="https://two.example/", domain="two.example"),
        ],
        enriched=Enrichment(
            summary="",
            topics=["t-one", "t-two"],
            primary_topic="t-two",
            enriched_at=datetime(2026, 1, 4, tzinfo=UTC),
            executor="manual",
        ),
        content=Content(
            fetched_at=datetime(2026, 1, 3, tzinfo=UTC),
            sources=[
                ContentSourceSuccess(
                    kind="x_video", url="https://v.example/1", text="", has_speech=False
                ),
                ContentSourceSuccess(
                    kind="x_article",
                    url="https://x.com/i/article/1",
                    text="".join(parts),
                    blocks=[ArticleTextBlock(text=p) for p in parts],
                ),
                ContentSourceFailure(
                    kind="external_article",
                    url="https://one.example/",
                    failure_reason="not_found",
                    http_status=404,
                    attempts=1,
                ),
                ContentSourceFailure(
                    kind="quoted_tweet",
                    url="https://x.com/b/status/9",
                    failure_reason="forbidden",
                    attempts=2,
                ),
            ],
        ),
    )
    return _item(**{**defaults, **overrides})


def test_the_whole_payload_is_pinned_by_value_on_an_item_that_populates_every_region() -> None:
    """The POSITIONAL and CHARACTERIZATION guard the belt above is not, in one literal.

    Two holes close here. The belt's expected side runs the same `_canonical` and `_sha256` as
    its actual side, so an ENCODER change moves both together — rule 1's fourth row — and
    `separators=(",", ":")`, `hexdigest()[:16]` and `sha256 -> sha1` were each measured leaving
    the file green. And on a BARE item the last seven atoms are indistinguishable by position,
    so `kinds`<->`rows`, `failures`<->`links`, `topics`<->`kinds` and
    `bookmark_folder`<->`primary_topic` all survived. A digest written down as a LITERAL is on
    neither side of the encoder and sees both. Same instrument, and same reason, as
    `tests/test_evidence_characterization.py`'s pin on `contract_fingerprint`: one byte of
    drift retires every stored fingerprint, so a DELIBERATE change re-derives this literal in
    the commit that makes it. The belt above stays the readable account of the payload.
    """
    fingerprint = index_build.item_fingerprint(_rich())
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint), fingerprint
    assert (
        fingerprint
        == "83d6407b9c518db6fb72d12db66a755a16e8b9d5c3fab1662d99a56f9798a20e"  # pragma: allowlist secret
    )


def test_the_serialisation_itself_is_pinned_and_the_digest_is_a_real_sha256() -> None:
    """Both pinned against a value computed OUTSIDE this module — the only way an assertion on
    an encoder is not the encoder asserting itself. `_sha256("abc")` is the published SHA-256
    of `abc`, so `sha1` and a truncation both redden without this file owning the oracle.
    """
    assert index_build._canonical("t", ["a", 1, None]) == '["t", ["a", 1, null]]'
    assert index_build._sha256("abc") == hashlib.sha256(b"abc").hexdigest()


def test_an_article_block_repartition_moves_a_hash_no_surface_value_can_see() -> None:
    """The `chunks` plane. `ContentSourceSuccess` validates `text == "".join(blocks)`, so the
    flattened body is a function of the partition and NOT the reverse: these two items are one
    article cut in different places. Every persisted SURFACE value is identical — asserted
    beside the hash, because two hashes differing would not say the CUTS are what moved — while
    `chunk_surfaces(..., blocks_by_surface_id=...)` emits different `chunk_id`s, offsets and
    bodies. 41 of 2,404 items carry usable blocks, 41 of 41 more than one (2026-09-03), and the
    parser that sets them is the one CLAUDE.md records as unconfirmed against a real capture.
    """
    one, two = _rich(parts=("abcdefg",)), _rich(parts=("abc", "defg"))
    assert [index_build.surface_row(s) for s in item_surfaces(one)] == [
        index_build.surface_row(s) for s in item_surfaces(two)
    ]
    assert [len(t) for v in article_block_texts(one).values() for t in v] == [7]
    assert index_build.item_fingerprint(one) != index_build.item_fingerprint(two)


def test_the_item_fingerprint_covers_the_topics_plane_the_primary_atom_does_not() -> None:
    """`item_topics` is a persisted table, `chunks.topics` is copied from it, `--topic` filters
    on it, and 2,404 of 2,404 items populate it — yet the region could be a constant `[]` with
    the file green, because the only test that varied topics separated its items by the
    `primary_topic` atom. Here the primary is HELD EQUAL, so the tuple is all that can move.
    """
    assert index_build.item_fingerprint(
        _rich(
            enriched=Enrichment(
                summary="",
                topics=["t-one", "t-two"],
                primary_topic="t-two",
                enriched_at=datetime(2026, 1, 4, tzinfo=UTC),
                executor="manual",
            )
        )
    ) != index_build.item_fingerprint(
        _rich(
            enriched=Enrichment(
                summary="",
                topics=["t-one", "t-three"],
                primary_topic="t-two",
                enriched_at=datetime(2026, 1, 4, tzinfo=UTC),
                executor="manual",
            )
        )
    )


def test_the_item_fingerprint_covers_the_content_kinds_no_surface_and_no_counter_moves() -> None:
    """`item_content_kinds` is a persisted table with 1,387 of 2,404 items in it, and its atom
    could be a constant `[]` with nothing red. Isolated: the added source is an `x_video` with
    `has_speech=None` and no text, so it emits NO surface, counts in NEITHER declined counter
    (unknown is not known-silent), and moves nothing but the kinds region.

    The region is a SET, and that is a behaviour and not a comment: `item_content_kinds` is the
    ONE derivation `knowledge_item` also reads, and the plane on disk is keyed
    `(item_id, kind)`, so two sources of one kind are ONE row. A derivation that stopped
    deduplicating would re-hash every item that gains a second source of a kind it already had
    — 10 of 2,404 on the live store, measured 2026-09-03 — with no row moved.
    """
    without = _item()
    with_video = _item(content=_video_content("", None))
    doubled = _item(
        content=_video_content("", None).model_copy(
            update={
                "sources": [*_video_content("", None).sources, *_video_content("", None).sources]
            }
        )
    )
    assert item_content_kinds(doubled) == ("x_video",)
    assert item_surfaces(without) == item_surfaces(with_video)
    assert index_build.declined_media(without) == index_build.declined_media(with_video) == (0, 0)
    assert index_build.item_fingerprint(without) != index_build.item_fingerprint(with_video)


def test_a_nullable_surface_column_that_is_absent_is_not_one_that_is_empty() -> None:
    """The `bookmark_folder` argument, carried across to the five columns it was never applied
    to. `surfaces.title`, `.url`, `.language` and the two attribution cells are nullable exactly
    as `items.bookmark_folder` is, and NULL is a different row from `''`; the by-value literal
    pins every cell NON-null, so a `... or ""` mutant never meets a `None` and all five
    survived. The INTEGER cells are pinned by TYPE beside them: `True == 1` and `1.0 == 1`, so
    tuple equality cannot see `int(derived)` become a bool or a float — and `_canonical`, which
    states that no float may enter, hashes all three differently.
    """
    bare = _surface(title=None, language=None, attribution=None, locator=Locator(kind="item_text"))
    row = index_build.surface_row(bare)
    assert row[7:11] == (None, None, None, None)
    assert row[12] is None
    assert type(row[6]) is int and type(row[14]) is int


# ---------------------------------------------------------------------------
# `store_fingerprint` — the outer atom, and what an id may not forge
# ---------------------------------------------------------------------------


def test_the_store_fingerprint_is_independent_of_how_the_store_was_loaded(corpus) -> None:
    """A dict's iteration order is a property of the load, not of the contents. The ids are
    sorted, so two loads of the same corpus certify the same store.
    """
    store, _vocab, _pages = corpus
    assert index_build.store_fingerprint(store) == index_build.store_fingerprint(
        dict(reversed(list(store.items())))
    )


def test_the_outer_store_atom_is_a_pair_and_not_a_concatenation() -> None:
    """The boundary between an id and the hash beside it is STRUCTURAL, not arithmetic.

    Said exactly, because the overclaim is tempting: a flat `id + hex` join is injective
    TODAY, and only because the second half is always 64 characters, so a reader can cut it
    off the end. Measured — the mutant `[[k, hex]]` -> `[k + hex]` leaves the door's own
    guards GREEN. What the pair buys is that the separation stops depending on that width:
    the day an atom of variable length joins it, `["a", "b" + h]` and `["ab", h]` are the
    same stream and two different stores certify as one. Asserted on the ENCODER, which is
    where that forgery lives and the only place it can be shown.
    """
    forged, honest = ["a", "b" + "0" * 64], ["ab", "0" * 64]
    assert "".join(forged) == "".join(honest)
    assert index_build._canonical("store", [forged]) != index_build._canonical("store", [honest])


def test_the_store_plane_survives_a_nul_inside_an_id() -> None:
    """The pre-fix NUL join, on the plane where a real `items.json` can reach it. `Item.id` is a
    bare `str` with no pattern and no validator, and a NUL travels through `save_store` /
    `parse_store` as a plain ASCII escape, so the key `a=<hex>` + NUL + `b` round-trips and
    under the join produced exactly the stream a two-item store produces — two different
    stores, one certificate. The encoder guard above shows why the pair is safe; this shows the
    reversion that removes it.
    """
    item = _item(id="X")
    two = {"a": item, "b": item}
    forged = {f"a={index_build.item_fingerprint(item)}\0b": item}
    assert index_build.store_fingerprint(two) != index_build.store_fingerprint(forged)


def test_the_store_fingerprint_covers_the_key_an_item_is_filed_under() -> None:
    """The store is a MAPPING and the key is half of each entry. `item_fingerprint` hashes
    `item.id`, normally the same string, so this is what keeps the key in the atom for its own
    sake rather than by luck and reddens under dropping `k` from the pair. The DOMAIN is pinned
    beside it: `"store"` is a hand-written literal here, so the mutant that hashes this plane
    under `"item"` — which no other test could see — reddens.
    """
    item = _item(id="42")
    assert index_build.store_fingerprint({"42": item}) != index_build.store_fingerprint(
        {"filed-elsewhere": item}
    )
    assert index_build.store_fingerprint({}) == index_build._sha256(
        index_build._canonical("store", [])
    )


def test_the_store_fingerprint_moves_when_any_one_item_moves(corpus) -> None:
    """The deep signal's whole promise, and the one the cheap `StoreSignal` cannot make.

    BOTH ENDS OF THE ORDER, because "any" is the word this name makes load-bearing. Every
    store-plane guard varied `sorted(store)[0]`, and the mutant `[...][:1]` — hash the FIRST
    item and no other — left the whole file green: under it every item but one certifies as
    unchanged forever, the open-failing direction rule 6 exists for. The complement `[1:]` was
    already killed, and that asymmetry is the tell that the test measured position 0, not any.
    """
    store, _vocab, _pages = corpus
    for key in (sorted(store)[0], sorted(store)[-1]):
        edited = dict(store)
        edited[key] = store[key].model_copy(update={"text": store[key].text + " edited"})
        assert index_build.store_fingerprint(store) != index_build.store_fingerprint(edited), key


def test_the_deep_fingerprints_stay_out_of_the_cheap_signal(three_inputs: Path) -> None:
    """02.6a1's contract, inherited unchanged: a query pays three `os.stat` and nothing else. A
    deep read leaking into `StoreSignal.of` or `load_index_inputs` would put a full corpus walk
    behind every `search`; asserted by counting calls into the deep plane while both doors run.
    """
    calls: list[str] = []
    original = index_build.item_fingerprint
    index_build.item_fingerprint = lambda *a, **k: calls.append("deep") or original(*a, **k)
    try:
        index_build.StoreSignal.of(*_paths(three_inputs))
        index_build.load_index_inputs(*_paths(three_inputs))
    finally:
        index_build.item_fingerprint = original
    assert calls == []
