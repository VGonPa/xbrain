# tests/test_knowledge_index_build.py
"""The three inputs of the index, the signal bound to them, and their fingerprints
(Plan 02 §2, §3, steps 3, 4, 9).

TWO SIGNALS, TWO COSTS, TWO PLACES (B3). `store_signal` is an `os.stat` — mtime and size of
`data/items.json` — and it is what a QUERY can afford on every call. `store_fingerprint` is a
sha256 per item and costs loading the store, so it is paid only by `build`, `update` and
`status`. The cheap one says *the store moved*; the expensive one says *which items changed,
and how many*. A false positive on the cheap one costs one warning; a false negative costs
serving stale evidence as fresh, so it fails towards the warning.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge import index_build
from xbrain.knowledge.surfaces import item_surfaces
from xbrain.models import Item, Topic, TopicPage

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    store = {k: Item.model_validate(v) for k, v in raw["items"].items()}
    vocab = [Topic.model_validate(v) for v in raw["vocab"].values()]
    pages = {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()}
    return store, vocab, pages


@pytest.fixture()
def workspace(tmp_path: Path, corpus) -> Path:
    """A data/ directory holding the fixture store, so `store_signal` has a real file."""
    store, _vocab, _pages = corpus
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in store.items()}), encoding="utf-8"
    )
    return data


# ---------------------------------------------------------------------------
# The fingerprint, and the population it has to reach (rule 6)
# ---------------------------------------------------------------------------


def test_the_item_fingerprint_changes_when_indexable_text_changes(corpus) -> None:
    """It hashes the SURFACES, not `(fetched_at, enriched_at)` as Plan 01 §10 sketched.

    Two reasons, and CLAUDE.md rule 6 is both of them. First, `content.fetched_at` cannot
    reach an item whose `content` is `None` — 960 of 2,404 in the real store (measured
    2026-09-01 on sha256 `f76341a3…`) — because there
    is nothing to stamp. Second, a timestamp is a PROXY: a summary edited by hand, or any
    repair that rewrites text without touching a clock, changes the indexable corpus and
    leaves the proxy unmoved, so the index would keep serving the old body under a
    fingerprint that says it is current.

    Hashing the emitted surface fingerprints answers the question directly — *did the text
    this index holds change?* — and it reaches every item, with or without `content`.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    before = index_build.item_fingerprint(item)
    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": "otro resumen"})}
    )
    assert index_build.item_fingerprint(edited) != before
    assert index_build.item_fingerprint(item) == before, "and it is stable"


def test_the_item_fingerprint_reaches_an_item_with_no_content(corpus) -> None:
    """Step 6 / rule 6: the invalidation signal must reach the population it has to reach.

    `k02` carries no `content` at all, so a fingerprint over `content.fetched_at` would be
    constant for it forever. Seen red by hashing only the timestamps.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    assert item.content is None, "this test is only meaningful on an item with no content"
    edited = item.model_copy(
        update={"enriched": item.enriched.model_copy(update={"summary": "otro resumen"})}
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


def test_the_item_fingerprint_also_covers_the_filterable_metadata(corpus) -> None:
    """The index stores more than text: a changed author or topic changes what a filter returns.

    Ignoring them would leave `--author` answering from an author the store no longer records.
    """
    store, _vocab, _pages = corpus
    item = store["k02"]
    moved = item.model_copy(update={"source": "own_tweet"})
    assert index_build.item_fingerprint(moved) != index_build.item_fingerprint(item)


@pytest.mark.parametrize(
    "field, value",
    [
        ("author", {"handle": "someoneelse", "name": "Someone Else"}),
        ("title", "A different title"),
        ("language", "fr"),
        ("url", "https://x.com/othervoice/status/moved"),
    ],
)
def test_the_item_fingerprint_covers_what_the_index_stores_about_a_surface(
    corpus, field: str, value: object
) -> None:
    """G-5: every column the index STORES about a surface moves the fingerprint.

    `surface_fingerprint` is `(version, type, origin, text)` by design — two surfaces with
    the same text under different provenance must differ, and nothing more. But the index
    persists more than that in `surfaces`: attribution, title, url, locator and language,
    which `search` serves on every match (A-1) and `--has-surface` filters on. A change to
    any of them with the text untouched left `update` seeing nothing to do — the evidence
    repaired and the derivative standing (rule 6), on the attribution rule this repo paid
    for in blood.

    Four axes, one parametrised test, each changing ONE field of k07's quoted post and
    nothing else. The `url` case moves the locator columns but does NOT pin them: it is
    satisfied through `surface_id`, which hashes the source's kind and url, and it stays
    green with both locator columns dropped from `surface_row` (round 09, B1). What pins
    the locator is the test below, which moves a locator field `surface_id` cannot reach.

    `producer` is deliberately NOT here — the index has no producer column, and since round
    08 (F7-7) the ASR/VLM surfaces carry none at all, because the store records no
    transcriber; the producers that ARE recorded (`enriched.executor`,
    `description_version`) travel with the surface and are not stored columns either.

    Seen red before the fix on all four: the fingerprint did not move. Seen red again here
    under the mutant `surface_row` → `(None, None, None, …)`, which the file passed 90/90
    before this test was restored — on three of the four, the `url` case being the exception
    named above: `surface_row` is the projection `item_fingerprint` hashes, so a column
    dropped from it is a column the index stores and never re-hashes.
    """
    from xbrain.models import Author

    store, _vocab, _pages = corpus
    item = store["k07"]
    position = next(i for i, s in enumerate(item.content.sources) if s.kind == "quoted_tweet")
    sources = list(item.content.sources)
    patch = {field: Author(**value) if field == "author" else value}
    sources[position] = sources[position].model_copy(update=patch)
    edited = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item), field


def test_the_item_fingerprint_covers_the_locator_not_only_the_url_the_id_hashes(
    corpus,
) -> None:
    """The LOCATOR column, which the `[url]` case above does not reach (round 09, B1).

    That case moves the fingerprint through `surface_id`, which hashes the source's kind
    and url — so it is satisfied with BOTH locator columns dropped from `surface_row`, and
    the mutant that returns `None` and `""` for `surface.locator.url` and
    `surface.locator.model_dump_json()` leaves the file green. That is rule 1's first row:
    an assertion satisfied for the wrong reason, on the projection this child owns.

    `Locator` carries eleven fields and ten of them reach the persisted `locator_json`
    column (`index_schema`) through no other hashed path. They are how a reader resolves a
    claim back to the bytes it came from — a frame timestamp says WHERE in the video the
    slide is, `char_start`/`char_end` say where in a body a quote is — which is the whole
    mechanism CLAUDE.md rule 7 rests on.

    So this moves exactly one of them: k08's first video frame keeps its description, its
    `frame_index`, its `source_key` and its source, and only its `frame_timestamp` changes.
    The two guards are what make it bite for the reason it names — exactly one surface row
    moves, in exactly one column, and that column deserialises to the new timestamp.

    The sibling column `surface.locator.url` is a DUPLICATE of a field `model_dump_json()`
    also serialises, so dropping it ALONE is unobservable in the hash by construction: no
    test can pin it, and one claiming to would be pinning the json.

    Seen red under the mutant that drops the two locator columns from `surface_row`.
    """
    store, _vocab, _pages = corpus
    item = store["k08"]
    position = next(i for i, s in enumerate(item.content.sources) if s.kind == "x_video")
    video = item.content.sources[position]
    assert video.frames, "this test needs a video the index locates frames inside"

    frames = list(video.frames)
    moved_to = frames[0].timestamp + 37.0
    frames[0] = frames[0].model_copy(update={"timestamp": moved_to})
    sources = list(item.content.sources)
    sources[position] = video.model_copy(update={"frames": frames})
    edited = item.model_copy(
        update={"content": item.content.model_copy(update={"sources": sources})}
    )

    before = [index_build.surface_row(s) for s in item_surfaces(item)]
    after = [index_build.surface_row(s) for s in item_surfaces(edited)]
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1, "exactly one surface row moves: the one whose timestamp changed"
    row_before, row_after = before[changed[0]], after[changed[0]]
    columns = [i for i, (a, b) in enumerate(zip(row_before, row_after, strict=True)) if a != b]
    assert len(columns) == 1, "and exactly one column of that row"
    assert json.loads(row_after[columns[0]])["frame_timestamp"] == moved_to, (
        "the column that moved is the serialised locator, carrying the new timestamp"
    )
    assert json.loads(row_before[columns[0]])["frame_timestamp"] == video.frames[0].timestamp

    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


@pytest.mark.parametrize(
    "axis, value",
    [
        ("author", {"handle": "someoneelse", "name": "Someone Else"}),
        ("created_at", datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)),
        ("captured_at", datetime(2026, 6, 2, 9, 0, tzinfo=timezone.utc)),
        ("bookmark_folder", "to-read"),
    ],
)
def test_the_item_fingerprint_covers_the_item_plane_no_surface_carries(
    corpus, axis: str, value: object
) -> None:
    """The other half of the hash: the `items` columns, pinned where no surface pins them.

    `items` persists `author_handle`, `author_name`, `created_at`, `captured_at` and
    `bookmark_folder` (`index_schema`), and `items_author` is the index `--author` answers
    from. The test above pins `source` and nothing else: measured under the mutant that
    deletes the rest from `item_fingerprint`, this file stayed green, so `item.author`
    could change and `update` report nothing to do — the evidence repaired and the
    derivative standing (rule 6), one plane up from G-5 and on the same attribution axis.

    THE BASE ITEM CARRIES NO POST SURFACE, AND THAT IS THE WHOLE CONSTRUCTION (rule 1).
    `item.author` is ALSO the attribution of the `post` surface and of a `user_note`
    (`surfaces.item_surfaces`), so on an ordinary item this assertion moves through
    `surface_row` and passes with the metadata half deleted — satisfied for the wrong
    reason, which is what the round-09 mutants found. k02's text is blanked so no `post`
    surface is emitted, and its `enriched` carries no `user_notes`: one `summary` surface
    remains, attributed to nobody.

    So every case asserts twice: that not one surface row moved — the only other input to
    this hash — and that the fingerprint moved anyway. Delete the axis from the metadata
    half and the second assertion fails; the first is what proves it could not have passed
    through the surfaces instead.
    """
    from xbrain.models import Author

    store, _vocab, _pages = corpus
    item = store["k02"].model_copy(update={"text": ""})
    surfaces = item_surfaces(item)
    assert [s.surface_type for s in surfaces] == ["summary"], (
        "the base item must carry a surface, and not one that repeats the item's author"
    )
    rows = [index_build.surface_row(s) for s in surfaces]

    edited = item.model_copy(update={axis: Author(**value) if axis == "author" else value})
    assert [index_build.surface_row(s) for s in item_surfaces(edited)] == rows, (
        "no surface row may move, or the fingerprint could differ for another reason"
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item), axis


def test_the_item_fingerprint_covers_a_content_kind_that_emits_no_surface(corpus) -> None:
    """`item_content_kinds` is a persisted plane, and a kind can arrive with no text at all.

    A no-speech `x_video` (107 of 259 in the corpus) fetched without `--frames` has an
    empty transcript, no frames and no digest, so it emits NOT ONE surface — and it is
    still a video the index records in `item_content_kinds` and a kind filter answers
    from. Measured under the mutant that deletes the sorted-kinds block from
    `item_fingerprint`: green, because every kind in the fixture arrives attached to a
    surface that moves the hash on its own.

    The hash is over the sorted MULTISET while the table's primary key is
    `(item_id, kind)` — a set — so a duplicated kind moves the fingerprint without
    changing the stored plane: the accepted false positive, failing towards the warning
    like everything else in this module. This case adds a kind the item did not have at
    all, so it moves both.

    Guarded like the item-plane cases above, and here the guard is also the evidence: the
    surface rows are identical precisely BECAUSE the silent video emitted nothing. The
    only other field the patch touches is `content.fetched_at`, which is hashed nowhere —
    deliberately, and that is the subject of this module's docstring.
    """
    from xbrain.models import Content, ContentSourceSuccess

    store, _vocab, _pages = corpus
    item = store["k02"]
    assert item.content is None, "the base item must start with no content source at all"
    rows = [index_build.surface_row(s) for s in item_surfaces(item)]

    silent = ContentSourceSuccess(
        kind="x_video", url="https://video.example/silent.mp4", text="", has_speech=False
    )
    edited = item.model_copy(
        update={"content": Content(fetched_at=item.captured_at, sources=[silent])}
    )
    assert [index_build.surface_row(s) for s in item_surfaces(edited)] == rows, (
        "a silent video with no frames and no digest emits no surface"
    )
    assert index_build.item_fingerprint(edited) != index_build.item_fingerprint(item)


def test_the_leading_surface_version_is_the_belt_for_an_item_with_no_surfaces(
    corpus, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An emitter-version bump has to reach an item that emits nothing.

    `SURFACE_VERSION` is hashed twice over: once at the head of `item_fingerprint`'s parts
    and once inside every `surface.fingerprint`. For any item that HAS a surface the second
    copy carries it, which is why deleting the first left this file green — no test held an
    item with zero surfaces. Such items exist: k01 has no content, no media and no
    `enriched`, so blanking its text leaves the emitter with nothing to emit, and the
    leading copy is the only place the version can still reach it. Without it, a bump would
    leave that item's fingerprint unchanged and `update` would report it current under an
    emitter version that no longer describes how it would be emitted.

    Patching this module's binding alone is an EXACT simulation here rather than a partial
    one: a real bump also travels through `ids.surface_fingerprint`, and this item has no
    surface for it to travel through.
    """
    store, _vocab, _pages = corpus
    item = store["k01"].model_copy(update={"text": ""})
    assert item_surfaces(item) == (), "the belt is load-bearing only with no surfaces"

    before = index_build.item_fingerprint(item)
    monkeypatch.setattr(index_build, "SURFACE_VERSION", index_build.SURFACE_VERSION + "-next")
    assert index_build.item_fingerprint(item) != before


def test_the_store_fingerprint_is_order_independent(corpus) -> None:
    """Two loads of the same store must agree, whatever order the dict happens to iterate in."""
    store, _vocab, _pages = corpus
    reversed_store = dict(reversed(list(store.items())))
    assert index_build.store_fingerprint(reversed_store) == index_build.store_fingerprint(store)


def test_load_index_inputs_binds_the_signal_to_the_bytes_it_read(workspace, corpus) -> None:
    """The mechanism behind P1b, at the loader: the signal is `fstat` of the handle the
    text came from, taken BEFORE the read — so a replacement that lands during the read
    leaves the objects and the signal describing the same snapshot (the handle keeps the
    inode it opened; the store's writers replace atomically), and leaves the path pointing
    at a newer inode the query-time `StoreSignal.of` reports as different.

    Staged INSIDE the read: the handle `load_index_inputs` opens replaces the file on disk
    the moment it is read from, which is the only place a stat of the path and a stat of
    the handle can disagree. Seen red under the mutation `stat = path.stat()` taken after
    the read: the signal then described the replacement, not the bytes parsed.
    """
    from xbrain.store import save_store

    store, _vocab, _pages = corpus
    items = workspace / "items.json"
    before = index_build.StoreSignal.of(items)
    newer = {**store, "k01": store["k01"].model_copy(update={"text": "replaced during the load"})}

    class RacingHandle:
        def __init__(self, handle):
            self._handle = handle

        def fileno(self):
            return self._handle.fileno()

        def read(self):
            save_store(newer, items)  # the race: a save lands while the loader is reading
            return self._handle.read()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

    class RacingPath(type(items)):
        def open(self, *args, **kwargs):
            return RacingHandle(super().open(*args, **kwargs))

    loaded = index_build.load_index_inputs(RacingPath(items))

    assert loaded.store["k01"].text == store["k01"].text, "the parse saw the bytes it read"
    assert loaded.signal == before, "the signal describes the snapshot that was parsed"
    assert index_build.StoreSignal.of(items) != loaded.signal, "the path now points elsewhere"


def test_the_store_signal_is_one_stat_and_nothing_else(workspace) -> None:
    """The cheap signal must stay cheap, or it is the expensive one with a different name."""
    signal = index_build.StoreSignal.of(workspace / "items.json")
    stat = (workspace / "items.json").stat()
    assert signal.items_json_mtime_ns == stat.st_mtime_ns
    assert signal.items_json_size == stat.st_size


def test_the_store_signal_of_a_missing_store_is_zeroed_not_an_exception(tmp_path: Path) -> None:
    """A query must still be able to say "the index is behind" when the store is gone.

    Raising here would turn a missing store into a crash inside `search`, which is the wrong
    place to learn it: the response declares the degradation instead.
    """
    signal = index_build.StoreSignal.of(tmp_path / "nope.json")
    assert signal == index_build.StoreSignal(items_json_mtime_ns=0, items_json_size=0)


def _obstruct(path: Path, obstacle: str) -> None:
    """Make `path` UNREADABLE without removing it — the two shapes an operator meets."""
    if obstacle == "chmod000":
        path.chmod(0)
    else:
        path.unlink()
        path.mkdir()


@pytest.mark.parametrize(
    ("obstacle", "error"), [("chmod000", PermissionError), ("directory", IsADirectoryError)]
)
def test_an_unreadable_store_is_an_error_never_an_empty_store(
    workspace: Path, obstacle: str, error: type[OSError]
) -> None:
    """A-2 (gate Fable §5.2, round 08 — the DIFF subagent's A-1, reproduced by the gate in
    two variants): `_read_bound` caught EVERY `OSError` and answered `(None, 0, 0)`, the
    reading reserved for a file that is ABSENT — so an `items.json` with `chmod 000`, or a
    directory standing where the file was, loaded as an EMPTY store. On the real index:
    `status` healthy with `items_removed 2404`, `search` «0 resultados» exit 0 with nothing
    declared, `update` planning `-2404 items`, and `build --force` replacing 22,286 chunks
    with the 703 of the topic plane, sealed consistent, exit 0. The loader is the route
    BEFORE seam (a), the one no door asks, and through it the whole fail-open family came
    back with a destruction on top.

    `load_store` (`store.py`) reads only an ABSENT file as `{}`; every other `OSError`
    propagates. The loader now agrees: only `FileNotFoundError` is the empty store, and
    `EACCES`/`EISDIR`/`EIO` raise, which the CLI prints as `Error: …` with exit 1.

    Seen red on `36f694b`: `loaded.store == {}` on both obstacles, no exception.
    """
    _obstruct(workspace / "items.json", obstacle)
    try:
        with pytest.raises(error):
            index_build.load_index_inputs(workspace / "items.json")
    finally:
        # Leave the temp dir removable whatever the outcome.
        if obstacle == "chmod000":
            (workspace / "items.json").chmod(0o644)


def test_an_absent_store_still_reads_as_an_empty_store(tmp_path: Path) -> None:
    """The positive control for the test above: ABSENT is the one reading that stays empty —
    `load_store`'s own semantics, and what `StoreSignal.of` reports as zeros — so a fresh
    checkout with no `data/` still loads, and only what EXISTS and cannot be read raises."""
    loaded = index_build.load_index_inputs(tmp_path / "nope.json")
    assert loaded.store == {}
    assert loaded.signal == index_build.StoreSignal(items_json_mtime_ns=0, items_json_size=0)


# ---------------------------------------------------------------------------
# THREE INPUTS, NOT ONE — and the two fingerprints over the other two
#
# The snapshot's suite reached `items.json` on every door and reached `vocab.yaml` /
# `topics.json` on none: `vocab_fingerprint` and `topics_fingerprint` had no direct test
# anywhere in it, and no test loaded a workspace that contained the other two files at all.
# Measured before writing these: the file was green at 14/14 with `load_index_inputs`
# reduced to `vocab=[], topic_pages={}` and the four vocab/topics signal entries left at
# zero — which is exactly the pre-round-05 defect P1a records, back with nothing to catch
# it. This child owns the three-input read and the three fingerprints over it, so the tests
# come with them (rule 1: each was seen red under the mutation named in its docstring).
# ---------------------------------------------------------------------------


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


def test_load_index_inputs_reads_the_vocabulary_and_the_topic_pages_too(
    three_inputs: Path, corpus
) -> None:
    """P1a at the loader: the index derives from THREE files, so the loader reads three.

    A topic description enters every assigned item's PROFILE (spec §5.1.A) and the overviews
    and notes are chunks the index serves, so a loader that reads `items.json` and defaults
    the other two to empty builds an index missing the whole topic plane while every signal
    it seals says the inputs were read. All six signal entries are the stat of the file that
    was actually parsed.

    Seen red under the mutation `vocab=[], topic_pages={}` with the four vocab/topics signal
    entries left at zero — the pre-round-05 shape: `AssertionError` on the vocabulary, and
    on `vocab_yaml_mtime_ns` once the parse was restored.
    """
    _store, vocab, pages = corpus
    loaded = index_build.load_index_inputs(
        three_inputs / "items.json", three_inputs / "vocab.yaml", three_inputs / "topics.json"
    )

    assert [t.slug for t in loaded.vocab] == [t.slug for t in vocab], "the vocabulary was read"
    assert {t.slug: t.description for t in loaded.vocab} == {
        t.slug: t.description for t in vocab
    }, "descriptions included: they enter every assigned item's profile"
    assert set(loaded.topic_pages) == set(pages), "the topic pages were read"
    assert loaded.topic_pages["agent-evaluation"].overview == pages["agent-evaluation"].overview

    for name, mtime_field, size_field in [
        ("items.json", "items_json_mtime_ns", "items_json_size"),
        ("vocab.yaml", "vocab_yaml_mtime_ns", "vocab_yaml_size"),
        ("topics.json", "topics_json_mtime_ns", "topics_json_size"),
    ]:
        stat = (three_inputs / name).stat()
        assert getattr(loaded.signal, mtime_field) == stat.st_mtime_ns, name
        assert getattr(loaded.signal, size_field) == stat.st_size, name


def test_an_unnamed_vocab_or_topics_path_reads_as_empty_with_a_zero_signal(
    three_inputs: Path,
) -> None:
    """The two optional inputs default to absent, and absent is empty — never an error.

    `load_index_inputs(items_path)` is the one-argument call every pre-round-05 caller made
    and the CLI still makes for a corpus with no vocabulary yet. It must load, not raise.

    Seen red under the mutation `_read_bound` without its `if path is None` guard:
    `AttributeError: 'NoneType' object has no attribute 'open'`.
    """
    loaded = index_build.load_index_inputs(three_inputs / "items.json")
    assert loaded.vocab == []
    assert loaded.topic_pages == {}
    assert (loaded.signal.vocab_yaml_mtime_ns, loaded.signal.vocab_yaml_size) == (0, 0)
    assert (loaded.signal.topics_json_mtime_ns, loaded.signal.topics_json_size) == (0, 0)


@pytest.mark.parametrize(
    "obstacle, error", [("chmod000", PermissionError), ("directory", IsADirectoryError)]
)
@pytest.mark.parametrize("filename", ["vocab.yaml", "topics.json"])
def test_an_unreadable_vocab_or_topics_is_an_error_never_an_empty_one(
    three_inputs: Path, filename: str, obstacle: str, error: type[OSError]
) -> None:
    """A-2 reaches all three inputs, not only the one the gate happened to probe.

    The fail-open family the gate found on `items.json` — an unreadable file read as the
    empty value reserved for an absent one — is a property of `_read_bound`, and the loader
    routes all three inputs through it. On `vocab.yaml` the same defect is quieter and no
    less destructive: a `chmod 000` vocabulary loads as «no topics», so every profile is
    built without its topic descriptions, `topic_rows_behind` sees an empty vocabulary and
    `update` plans the deletion of the whole topic plane — sealed consistent, exit 0.

    Seen red under the mutation `except OSError` in `_read_bound` (the pre-A-2 shape):
    `DID NOT RAISE` on all four parametrisations.
    """
    _obstruct(three_inputs / filename, obstacle)
    try:
        with pytest.raises(error):
            index_build.load_index_inputs(
                three_inputs / "items.json",
                three_inputs / "vocab.yaml",
                three_inputs / "topics.json",
            )
    finally:
        if obstacle == "chmod000":
            (three_inputs / filename).chmod(0o644)


def test_the_vocab_fingerprint_covers_the_descriptions_not_only_the_slugs(corpus) -> None:
    """The descriptions are hashed because they are INDEXED TEXT, not metadata (spec §5.1.A).

    A topic description enters the profile of every item assigned to it, so editing one
    changes indexable text on the ITEM plane, not only on the topic plane. A fingerprint
    over the slugs alone would leave `update` reporting nothing to do while every profile
    the index serves quotes the old description — rule 6, on the plane the vocabulary owns.

    And it is order-independent for the same reason `store_fingerprint` is: the order a
    vocabulary happens to be loaded in is a property of the load, not of what it contains.

    Seen red under three mutations: `f"{topic.slug}"` (the description dropped) on the
    first assertion, `f"{topic.description}"` (the slug dropped) on the second, and
    `sorted(...)` → `vocab` on the third. The rename PRESERVES SORT POSITION
    (`agent-evaluationz` sorts where `agent-evaluation` did) so that the second assertion
    cannot pass merely by reordering the concatenation — the way its first version did on
    the topics fingerprint below.
    """
    _store, vocab, _pages = corpus
    before = index_build.vocab_fingerprint(vocab)

    edited = [
        t.model_copy(update={"description": "otra descripción"}) if t.slug == vocab[0].slug else t
        for t in vocab
    ]
    assert index_build.vocab_fingerprint(edited) != before, "a description edit moves it"

    renamed = [t.model_copy(update={"slug": f"{t.slug}z"}) if t is vocab[0] else t for t in vocab]
    assert sorted(t.slug for t in renamed) == [t.slug for t in renamed], "sort position kept"
    assert index_build.vocab_fingerprint(renamed) != before, "and so does a slug"

    assert index_build.vocab_fingerprint(list(reversed(vocab))) == before, "order-independent"


def test_the_topics_fingerprint_covers_the_overview_every_note_and_the_slug(corpus) -> None:
    """The synthesised text of the topic plane, hashed through the SAME emitter the index uses.

    `topic_overview` and `topic_note` are surface types the index stores and `search`
    serves, so each is hashed through `surface_fingerprint` — the one definition — rather
    than through a second concatenation that could drift from it (rule 5). Three axes, and
    each is a chunk a consumer can be served: the overview, ANY note (not just the first),
    and the slug those chunks are filed under.

    THE RENAME PRESERVES SORT POSITION ON PURPOSE. The first version of this assertion
    refiled the page as `renombrado`, which sorts AFTER `ai-policy` — so the two pages'
    text swapped places in the concatenation and the hash moved whether or not the slug was
    hashed at all. Measured: under the mutant `parts.append(slug)` deleted, that version
    stayed GREEN and only `test_the_topics_fingerprint_agrees_with_the_emitter` went red.
    Rule 1, in the test written to pin the slug. `agent-evaluationz` sorts where
    `agent-evaluation` did, so the slug is the only thing that changed.

    Seen red under two mutations: dropping the notes comprehension (the second assertion)
    and `parts.append(slug)` deleted (the third). Hashing `page.overview` directly instead
    of through `surface_fingerprint` stays GREEN here and is caught by
    `test_the_topics_fingerprint_agrees_with_the_emitter` below — the reason that test
    exists.
    """
    _store, _vocab, pages = corpus
    before = index_build.topics_fingerprint(pages)

    page = pages["agent-evaluation"]
    assert len(page.notes) >= 2, "this test is only meaningful on a page with several notes"

    edited = {**pages, "agent-evaluation": page.model_copy(update={"overview": "otro resumen"})}
    assert index_build.topics_fingerprint(edited) != before, "the overview is hashed"

    last_note_changed = page.model_copy(update={"notes": [*page.notes[:-1], "otra nota"]})
    edited = {**pages, "agent-evaluation": last_note_changed}
    assert index_build.topics_fingerprint(edited) != before, "and EVERY note, not just the first"

    refiled = {("agent-evaluationz" if s == "agent-evaluation" else s): p for s, p in pages.items()}
    assert sorted(refiled) == ["agent-evaluationz", "ai-policy"], "the sort position is unchanged"
    assert index_build.topics_fingerprint(refiled) != before, "and the slug they are filed under"


def test_the_topics_fingerprint_agrees_with_the_emitter(corpus) -> None:
    """One definition of "what a topic-page surface is", shared with `knowledge.ids` (rule 5).

    The fingerprint is asserted against `surface_fingerprint` computed HERE from the page's
    own text, so a change to the emitter's `(version, type, origin, text)` shape moves this
    fingerprint too and the index re-derives. Hashing the raw text instead would leave a
    `SURFACE_VERSION` bump invisible to the topic plane while every item plane re-indexed.

    This is NOT the tautology `topics_fingerprint is topics_fingerprint`: the expected value
    is rebuilt from the page objects through the public emitter, so a version bump inside
    `surface_fingerprint` propagates to both sides but a change to the *composition* — the
    surface types used, the order, the slug — moves only one. Seen red under the mutation
    `surface_fingerprint("topic_note", ...)` → `surface_fingerprint("topic_overview", ...)`
    for the notes, which every other test in this file passes.
    """
    import hashlib

    from xbrain.knowledge.ids import surface_fingerprint

    _store, _vocab, pages = corpus
    parts: list[str] = []
    for slug in sorted(pages):
        page = pages[slug]
        parts.append(surface_fingerprint("topic_overview", "llm", page.overview))
        parts += [surface_fingerprint("topic_note", "llm", note) for note in page.notes]
        parts.append(slug)
    expected = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()

    assert index_build.topics_fingerprint(pages) == expected


def test_the_index_loader_and_the_store_doors_parse_through_one_parser_each(
    three_inputs: Path, corpus
) -> None:
    """The seam `parse_store` / `parse_vocab` / `parse_topic_pages` exists FOR THIS (rule 5).

    `load_index_inputs` must bind what it parsed to the exact bytes it read, so it reads
    through its own handle — and the only honest way to do that without a second copy of
    every parse is to split the parse out of `load_store` / `load_vocab` /
    `load_topic_pages` and share it. The property that matters is not "the seam exists" (a
    tautology the moment it does) but that the TWO READERS OF THE SAME BYTES agree: the
    path-based door every other stage uses, and the handle-based one the index uses.

    Seen red under the mutation `parse_vocab` → `data.get("topics")` without `Topic(**entry)`
    (raw dicts out of one door, `Topic` out of the other) and under a `load_index_inputs`
    that parsed the store with a bare `json.loads`.
    """
    from xbrain.rubrics import load_vocab
    from xbrain.store import load_store, load_topic_pages

    loaded = index_build.load_index_inputs(
        three_inputs / "items.json", three_inputs / "vocab.yaml", three_inputs / "topics.json"
    )

    assert loaded.store == load_store(three_inputs / "items.json")
    assert loaded.vocab == load_vocab(three_inputs / "vocab.yaml")
    assert loaded.topic_pages == load_topic_pages(three_inputs / "topics.json")
