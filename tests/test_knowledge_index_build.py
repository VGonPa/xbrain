# tests/test_knowledge_index_build.py
"""The three inputs of the index, read as one snapshot, with the cheap signal bound to it
(Plan 02 §2, §3, steps 3 and 4).

The argument for the cheap signal — what it can answer, what it cannot, and the direction it
fails in — is stated once, in `index_build.py`'s module docstring; it is not restated here.
"""

from __future__ import annotations

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
def test_an_input_that_cannot_be_stat_ed_is_zeroed_for_a_query_and_raises_for_a_load(
    three_inputs: Path, index: int, mtime_field: str, size_field: str
) -> None:
    """The two directions of A-2, on ONE obstruction, which is the only way to see the pair.

    `_stat_signal` swallows every `OSError`, not only `FileNotFoundError`, and this is the
    test that holds it: a query must be able to ANSWER — declaring the index behind — instead
    of learning about the filesystem by exception from inside `search`. The obstruction is a
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
    assert (getattr(signal, mtime_field), getattr(signal, size_field)) == (0, 0), "zeroed"
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
    assert all(m and s for m, s in others), f"the other two inputs still stat'ed: {others}"

    with pytest.raises(NotADirectoryError):
        index_build.load_index_inputs(*paths)


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
    """
    from xbrain.rubrics import load_vocab
    from xbrain.store import load_store, load_topic_pages

    items, vocab, topics = _paths(three_inputs)
    loaded = index_build.load_index_inputs(items, vocab, topics)

    assert loaded.store == load_store(items)
    assert loaded.vocab == load_vocab(vocab)
    assert loaded.topic_pages == load_topic_pages(topics)
