# tests/test_knowledge_index_build.py
"""The three inputs of the index, read as one snapshot, with the cheap signal bound to it
(Plan 02 §2, §3, steps 3 and 4).

The argument for the cheap signal — what it can answer, what it cannot, and the direction it
fails in — is stated once, in `index_build.py`'s module docstring; it is not restated here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.models import Item, Topic, TopicPage

FIXTURES = Path(__file__).parent / "fixtures"

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
