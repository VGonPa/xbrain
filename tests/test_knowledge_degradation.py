# tests/test_knowledge_degradation.py
"""Observable degradation of the vector channel — Plan 03 §5, one test per row (Plan 03.6).

THROUGH THE PUBLIC SERVICE, over a REAL index (rule 3). Where a row is about a backend that is
absent, broken or slow, the query is embedded by the REAL adapter `bind_query_embedder` over the
REAL `xbrain.embeddings` contract, and what fails is the process boundary: a path that does not
exist, a file without the execute bit, or a `runner` that raises `TimeoutExpired`. No network,
no GPU, no model (criterion §13.11).

THE LINE THAT IS NOT CROSSED (spec §9.3, *no finge resultados vectoriales*): a response never
names `hybrid` or `vector` unless the vector channel RAN. Where the caller asked for `hybrid`
the answer is `lexical`, with the reason declared; where the caller asked for `vector`
explicitly, there is no lexical answer to give — it is an error that says how to get vectors.

The degradation names are written out BY HAND (`embeddings_not_configured`,
`embedder_unavailable`): they are the frozen contract a consumer reads, so a test that imported
them from the module under test would stay green while the module renamed them (rule 1).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import subprocess  # nosec B404 - only to raise TimeoutExpired from a fake runner, nothing runs
import sys
from pathlib import Path

import pytest

from xbrain.embeddings import SCHEMA_VERSION as EMBEDDINGS_SCHEMA_VERSION
from xbrain.embeddings import EmbedderFailed, EmbedderNotFound, embed_passages
from xbrain.knowledge import index_build
from xbrain.knowledge.contracts import SearchMatch, SearchResponse
from xbrain.knowledge.index_schema import IndexError_
from xbrain.knowledge.search_service import QueryContext, bind_query_embedder, search
from xbrain.knowledge.vector_index import (
    VECTOR_REBUILD_ADVICE,
    VECTORS_FILENAME,
    VectorPlaneIncompatible,
    VectorSpec,
    vector_plane_exists,
)
from xbrain.models import Item, Topic, TopicPage
from xbrain.rubrics import save_vocab
from xbrain.store import save_store, save_topic_pages

FIXTURES = Path(__file__).parent / "fixtures"
QUERY = "Quillfeather"

SPEC = VectorSpec(
    model="intfloat/multilingual-e5-base",
    dimension=2,
    normalized=True,
    query_prefix="query: ",
    passage_prefix="passage: ",
)


def _vector(text: str) -> tuple[float, ...]:
    angle = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    angle *= 2 * math.pi
    return (math.cos(angle), math.sin(angle))


class CountingEmbedder:
    """A query embedder that answers `vector` and records every query it was asked for."""

    def __init__(self, vector: tuple[float, ...] = (1.0, 0.0)) -> None:
        self.vector = vector
        self.calls: list[str] = []

    def __call__(self, query: str) -> tuple[float, ...]:
        self.calls.append(query)
        return self.vector


def _timeout_runner(argv, **kwargs):  # noqa: ANN001, ANN003, ANN202 - a subprocess.run stand-in
    raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 1))


def _answering_runner(vector: tuple[float, ...], *, model: str):  # noqa: ANN202
    """A `subprocess.run` stand-in that speaks the embedder protocol: `model`, one `vector`.

    Nothing runs. It records each call so a test can prove the query WAS embedded — i.e. that
    what failed afterwards was the geometry, not a backend that never answered.
    """
    calls: list[list[str]] = []

    def run(argv, **kwargs):  # noqa: ANN001, ANN003, ANN202
        calls.append(list(argv))
        body = {
            "schema_version": EMBEDDINGS_SCHEMA_VERSION,
            "model": model,
            "dimension": len(vector),
            "normalized": True,
            "vectors": [list(vector)],
        }
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(body), stderr="")

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture()
def corpus() -> tuple[dict[str, Item], list[Topic], dict[str, TopicPage]]:
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    return (
        {k: Item.model_validate(v) for k, v in raw["items"].items()},
        [Topic.model_validate(v) for v in raw["vocab"].values()],
        {k: TopicPage.model_validate(v) for k, v in raw["topics"].items()},
    )


def _inputs(tmp_path: Path, corpus) -> tuple[Path, index_build.IndexInputs]:  # noqa: ANN001
    store, vocab, pages = corpus
    data = tmp_path / "data"
    save_store(store, data / "items.json")
    save_vocab(vocab, data / "vocab.yaml")
    save_topic_pages(pages, data / "topics.json")
    inputs = index_build.load_index_inputs(
        data / "items.json", data / "vocab.yaml", data / "topics.json"
    )
    return data, inputs


def _data(tmp_path: Path, corpus, *, with_plane: bool) -> Path:  # noqa: ANN001
    data, inputs = _inputs(tmp_path, corpus)
    vectors = (
        index_build.VectorBuild(spec=SPEC, embed=lambda texts: [_vector(t) for t in texts])
        if with_plane
        else None
    )
    index_build.build(data / "index", inputs, vectors=vectors)
    return data


def _context(data: Path, corpus, embed_query=None) -> QueryContext:  # noqa: ANN001
    store, vocab, pages = corpus
    return QueryContext(
        store=store,
        vocab=vocab,
        topic_pages=pages,
        index_dir=data / "index",
        items_path=data / "items.json",
        vocab_path=data / "vocab.yaml",
        topics_path=data / "topics.json",
        embed_query=embed_query,
    )


def _matches(response: SearchResponse) -> list[SearchMatch]:
    return [match for result in response.results for match in result.matches]


def _assert_lexical_and_honest(response: SearchResponse) -> None:
    """Criterion §13.4: `lexical`, a populated `degraded`, and not ONE match claiming `vector`."""
    assert response.strategy == "lexical"
    assert response.index.degraded
    assert response.results, "lexical stays operational (spec §9.3)"
    assert not [match for match in _matches(response) if "vector" in match.matched_by]
    assert not [match for match in _matches(response) if match.vector_rank is not None]


# --------------------------------------------------------------------------------------------
# Row 1 — `[embeddings].command` empty
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["", "   "])
def test_row1_an_unconfigured_command_serves_lexical_and_declares_it(
    tmp_path: Path, corpus, command: str
) -> None:
    """Criterion §13.2. The plane exists; only the command is missing — so THAT is the reason.

    Before 03.6 this answered `hybrid_not_implemented`, which is false: `hybrid` IS implemented,
    and a consumer told so has no way to learn that one line of `config.toml` fixes it.
    """
    data = _data(tmp_path, corpus, with_plane=True)
    embed_query = bind_query_embedder(command, index_dir=data / "index", timeout_seconds=5)

    response = search(QUERY, _context(data, corpus, embed_query), strategy="hybrid")

    _assert_lexical_and_honest(response)
    assert response.index.degraded == ("embeddings_not_configured",)


def test_row1_a_plain_lexical_search_without_embeddings_declares_nothing_it_did_not_lose(
    tmp_path: Path, corpus
) -> None:
    """The default request asked for no vectors, so no vector degradation is invented for it."""
    data = _data(tmp_path, corpus, with_plane=True)

    response = search(QUERY, _context(data, corpus, None))

    assert response.strategy == "lexical"
    assert response.results
    assert response.index.degraded == ()


# --------------------------------------------------------------------------------------------
# Row 2 — binary absent or not executable
# --------------------------------------------------------------------------------------------


def _missing_binary(tmp_path: Path) -> str:
    return str(tmp_path / "bin" / "xbrain-embed-that-does-not-exist")


def _non_executable_binary(tmp_path: Path) -> str:
    path = tmp_path / "bin" / "xbrain-embed-not-executable"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o644)
    return str(path)


@pytest.mark.parametrize("binary", [_missing_binary, _non_executable_binary])
def test_row2_index_build_raises_embedder_not_found_and_seals_nothing(
    tmp_path: Path, corpus, binary
) -> None:
    """The build names the configured command and leaves no manifest a door could trust."""
    data, inputs = _inputs(tmp_path, corpus)
    command = binary(tmp_path)

    def embed(texts):  # noqa: ANN001, ANN202 - the `Embedder` shape
        return embed_passages(texts, command=command, model=SPEC.model).vectors

    with pytest.raises(EmbedderNotFound, match=r"\[embeddings\]\.command"):
        index_build.build(
            data / "index", inputs, vectors=index_build.VectorBuild(spec=SPEC, embed=embed)
        )
    assert not index_build.manifest_path(data / "index").exists()
    assert not vector_plane_exists(data / "index")


@pytest.mark.parametrize("binary", [_missing_binary, _non_executable_binary])
def test_row2_search_degrades_to_lexical_declaring_embedder_unavailable(
    tmp_path: Path, corpus, binary
) -> None:
    """Criterion §13.4 with the REAL adapter: the process cannot start, lexical answers."""
    data = _data(tmp_path, corpus, with_plane=True)
    embed_query = bind_query_embedder(binary(tmp_path), index_dir=data / "index", timeout_seconds=5)

    response = search(QUERY, _context(data, corpus, embed_query), strategy="hybrid")

    _assert_lexical_and_honest(response)
    assert response.index.degraded == ("embedder_unavailable",)


# --------------------------------------------------------------------------------------------
# Row 3 — the manifest announces a plane the disk does not hold
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["hybrid", "vector"])
def test_row3_a_declared_plane_that_is_gone_is_refused_not_half_queried(
    tmp_path: Path, corpus, strategy: str
) -> None:
    """The error names `xbrain index build --embeddings`, and no query vector is paid for."""
    data = _data(tmp_path, corpus, with_plane=True)
    (data / "index" / VECTORS_FILENAME).unlink()
    embedder = CountingEmbedder()

    with pytest.raises(VectorPlaneIncompatible, match=r"xbrain index build --embeddings"):
        search(QUERY, _context(data, corpus, embedder), strategy=strategy)
    assert embedder.calls == []


# --------------------------------------------------------------------------------------------
# Row 4 — the backend answers a dimension the manifest does not declare
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("through", ["counting_embedder", "real_adapter"])
@pytest.mark.parametrize("strategy", ["hybrid", "vector"])
def test_row4_a_query_vector_of_another_dimension_is_a_hard_error_never_a_degradation(
    tmp_path: Path, corpus, strategy: str, through: str
) -> None:
    """Two models' vectors are never mixed — and a `hybrid` request does NOT fall back to lexical.

    A dimension mismatch is not a backend that is down: it is a backend serving ANOTHER model.
    Degrading would hide exactly the misconfiguration that makes every later vector answer wrong.

    `real_adapter` is the case that guards `bind_query_embedder`'s one deliberate omission: it
    does NOT pass `expected_dimension`. Passed, the mismatch is raised inside `embeddings` as an
    `EmbedderFailed` — the class of a timeout — and `hybrid` degrades over it. A fake embedder
    that bypasses the adapter cannot see that; the adapter over a 3-dimensional backend can.
    """
    data = _data(tmp_path, corpus, with_plane=True)
    third = 1 / math.sqrt(3)
    if through == "counting_embedder":
        embedder = CountingEmbedder((third, third, third))
        embed_query, calls = embedder, lambda: len(embedder.calls)
    else:
        run = _answering_runner((third, third, third), model=SPEC.model)
        embed_query = bind_query_embedder(
            "xbrain-embed", index_dir=data / "index", timeout_seconds=5, runner=run
        )
        calls = lambda: len(run.calls)  # noqa: E731

    with pytest.raises(VectorPlaneIncompatible, match=r"dimensión 3.*dimensión 2"):
        search(QUERY, _context(data, corpus, embed_query), strategy=strategy)
    assert calls() == 1, "premise: the mismatch came from the query vector"


@pytest.mark.parametrize("strategy", ["hybrid", "vector"])
def test_row4_a_backend_serving_another_model_of_the_same_dimension_is_a_hard_error(
    tmp_path: Path, corpus, strategy: str
) -> None:
    """A matching dimension does not prove the model is the same one (Plan 03 §13.4).

    The backend DECLARES its model on every batch; the manifest says which model wrote the
    plane. Same width, other model: the cosine is computed between two unrelated geometries and
    looks exactly as healthy as a real one. So it is refused — under `hybrid` too, never
    degraded, because a backend serving another model is not a backend that is down.
    """
    data = _data(tmp_path, corpus, with_plane=True)
    run = _answering_runner((1.0, 0.0), model="another/model-of-dimension-2")
    embed_query = bind_query_embedder(
        "xbrain-embed", index_dir=data / "index", timeout_seconds=5, runner=run
    )

    with pytest.raises(
        VectorPlaneIncompatible,
        match=r"another/model-of-dimension-2.*intfloat/multilingual-e5-base",
    ):
        search(QUERY, _context(data, corpus, embed_query), strategy=strategy)
    assert len(run.calls) == 1, "premise: the query WAS embedded — the model is what differed"


# --------------------------------------------------------------------------------------------
# Row 5 — a timeout while embedding
# --------------------------------------------------------------------------------------------


_LEXICAL_TABLES = ("items", "topics", "surfaces", "chunks", "profiles")


def _timing_out_build(texts):  # noqa: ANN001, ANN202 - the `Embedder` shape
    return embed_passages(
        texts, command="xbrain-embed", model=SPEC.model, runner=_timeout_runner
    ).vectors


def _persisted_rows(index_dir: Path) -> dict[str, int]:
    """What a SEPARATE, read-only connection finds committed — not what the builder believes."""
    uri = f"file:{index_build.db_path(index_dir)}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # nosec B608
            for table in _LEXICAL_TABLES
        }


def test_row5_a_build_whose_embedder_times_out_raises_and_seals_no_manifest(
    tmp_path: Path, corpus
) -> None:
    """`EmbedderFailed`, no manifest, no plane — and the lexical rows ARE left committed.

    DECLARED DIVERGENCE from Plan 03 §5 row 5 («transacción revertida»): nothing is rolled
    back. The lexical rows commit when `with connection:` exits, and the plane is written
    AFTER that transaction, so a timeout lands on a base that is already durable. What keeps
    the directory from being served is only the ORDER — the manifest is the last thing written
    — and the door refuses a base with no manifest. This test pins that state as it is, read
    from an independent connection, instead of claiming a rollback that does not happen.
    """
    data, inputs = _inputs(tmp_path, corpus)
    lexical_only = tmp_path / "lexical-only"
    index_build.build(lexical_only, inputs)

    with pytest.raises(EmbedderFailed, match="timed out"):
        index_build.build(
            data / "index",
            inputs,
            vectors=index_build.VectorBuild(spec=SPEC, embed=_timing_out_build),
        )

    assert not index_build.manifest_path(data / "index").exists()
    assert not vector_plane_exists(data / "index")
    committed = _persisted_rows(data / "index")
    assert committed == _persisted_rows(lexical_only), "the WHOLE lexical base is durable"
    assert all(committed.values()), f"premise: a non-empty corpus — {committed}"
    with pytest.raises(IndexError_, match="xbrain index build"):
        search(QUERY, _context(data, corpus, CountingEmbedder()))


def test_row5_a_forced_rebuild_whose_embedder_times_out_leaves_lexical_search_down(
    tmp_path: Path, corpus
) -> None:
    """`index build --embeddings --force` over a GOOD index, and the embedder times out.

    DECLARED DIVERGENCE from spec §9.3 («lexical sigue operativo») and Plan 03 §9: `--force`
    discards the previous manifest, base and plane BEFORE the first row is written (C-1), so
    once the embedder fails there is no index left to answer from. A plain lexical search is
    REFUSED until `xbrain index build` runs again. Preserving the previous index across a failed
    forced rebuild is a declared follow-up, not something this test pretends already happens.
    """
    data = _data(tmp_path, corpus, with_plane=True)
    inputs = index_build.load_index_inputs(
        data / "items.json", data / "vocab.yaml", data / "topics.json"
    )
    assert search(QUERY, _context(data, corpus)).results, "premise: the index was good"

    with pytest.raises(EmbedderFailed, match="timed out"):
        index_build.build(
            data / "index",
            inputs,
            force=True,
            vectors=index_build.VectorBuild(spec=SPEC, embed=_timing_out_build),
        )

    assert not index_build.manifest_path(data / "index").exists()
    assert not vector_plane_exists(data / "index")
    with pytest.raises(IndexError_, match="xbrain index build"):
        search(QUERY, _context(data, corpus))


def _chunks_containing(index_dir: Path, needle: str) -> int:
    """How many committed chunks carry `needle`, read by a SEPARATE read-only connection."""
    uri = f"file:{index_build.db_path(index_dir)}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE instr(text, ?) > 0", (needle,)
        ).fetchone()[0]


def test_the_vector_rebuild_advice_does_not_promise_a_lexical_plane_its_command_rebuilds(
    tmp_path: Path, corpus
) -> None:
    """The command every plane refusal recommends REBUILDS the lexical plane, so it must say so.

    `--embeddings --force` is wired to the full build: the base is unlinked and re-derived from
    the store (a new file, carrying the store's CURRENT text). An advice promising «el plano
    léxico no se toca» sends an operator into exactly the window the forced-rebuild test above
    pins — no index at all if the embedder then fails.

    THE PREMISE IS PROVEN BY CONTENT, NOT BY THE FILESYSTEM. It used to compare `st_ino` before
    and after, and CI went red on a correct build: a freed inode may be handed to the very next
    file, so unlink + recreate can keep the number. That assertion measured the allocator, not
    the contract — it could fail, and pass, for a reason unrelated to what it claimed (rule 1);
    `mtime` has the same flaw at its granularity. What the advice must be honest about is that
    the lexical base is RE-DERIVED FROM THE STORE, so the store is edited between the two builds
    and the new base's chunks must carry the new text AND none of the old. A rebuild that kept
    the previous base — the vector-only rebuild the old advice promised — fails both.
    """
    data = _data(tmp_path, corpus, with_plane=True)
    old, new = "Zephyrine", "Marrowglade"
    assert _chunks_containing(data / "index", old) == 1, "premise: the old base holds the old text"
    assert _chunks_containing(data / "index", new) == 0, "premise: and not the new one"
    store, _, _ = corpus
    edited = store["k01"].model_copy(update={"text": store["k01"].text.replace(old, new)})
    save_store({**store, "k01": edited}, data / "items.json")
    inputs = index_build.load_index_inputs(
        data / "items.json", data / "vocab.yaml", data / "topics.json"
    )

    index_build.build(
        data / "index",
        inputs,
        force=True,
        vectors=index_build.VectorBuild(spec=SPEC, embed=lambda texts: [_vector(t) for t in texts]),
    )

    assert _chunks_containing(data / "index", new) == 1, (
        "the recommended command re-derived the lexical base from the store's CURRENT text"
    )
    assert _chunks_containing(data / "index", old) == 0, "and kept nothing of the previous base"
    assert "xbrain index build --embeddings --force" in VECTOR_REBUILD_ADVICE
    assert "no se toca" not in VECTOR_REBUILD_ADVICE
    assert re.search(r"también el plano léxico", VECTOR_REBUILD_ADVICE)


def test_row5_a_query_whose_embedder_times_out_degrades_hybrid(tmp_path: Path, corpus) -> None:
    """At query time a slow backend is a DOWN backend: lexical answers, and says so."""
    data = _data(tmp_path, corpus, with_plane=True)
    embed_query = bind_query_embedder(
        "xbrain-embed", index_dir=data / "index", timeout_seconds=5, runner=_timeout_runner
    )

    response = search(QUERY, _context(data, corpus, embed_query), strategy="hybrid")

    _assert_lexical_and_honest(response)
    assert response.index.degraded == ("embedder_unavailable",)


# --------------------------------------------------------------------------------------------
# Row 6 — `--strategy vector` with no vectors
# --------------------------------------------------------------------------------------------


def test_row6_vector_over_an_index_without_a_plane_is_an_error(tmp_path: Path, corpus) -> None:
    """Criterion §13.3: the caller asked for `vector`; a lexical answer would be a lie about it."""
    data = _data(tmp_path, corpus, with_plane=False)
    embedder = CountingEmbedder()

    with pytest.raises(ValueError, match=r"xbrain index build --embeddings"):
        search(QUERY, _context(data, corpus, embedder), strategy="vector")
    assert embedder.calls == []


def test_row6_vector_without_a_configured_command_is_an_error(tmp_path: Path, corpus) -> None:
    """Same request, other missing half: the error names the setting that supplies it."""
    data = _data(tmp_path, corpus, with_plane=True)
    embed_query = bind_query_embedder("", index_dir=data / "index", timeout_seconds=5)

    with pytest.raises(ValueError, match=r"\[embeddings\]\.command"):
        search(QUERY, _context(data, corpus, embed_query), strategy="vector")


def test_row6_vector_with_the_backend_down_is_the_backend_error(tmp_path: Path, corpus) -> None:
    """Not a degradation either: the embedder's own actionable error reaches the caller."""
    data = _data(tmp_path, corpus, with_plane=True)
    embed_query = bind_query_embedder(
        _missing_binary(tmp_path), index_dir=data / "index", timeout_seconds=5
    )

    with pytest.raises(EmbedderNotFound):
        search(QUERY, _context(data, corpus, embed_query), strategy="vector")


# --------------------------------------------------------------------------------------------
# The line that is not crossed — spec §9.3, criterion §13.4
# --------------------------------------------------------------------------------------------


def _unconfigured(tmp_path: Path, data: Path):  # noqa: ANN202
    return bind_query_embedder("", index_dir=data / "index", timeout_seconds=5)


def _absent(tmp_path: Path, data: Path):  # noqa: ANN202
    return bind_query_embedder(
        _missing_binary(tmp_path), index_dir=data / "index", timeout_seconds=5
    )


def _slow(tmp_path: Path, data: Path):  # noqa: ANN202
    return bind_query_embedder(
        "xbrain-embed", index_dir=data / "index", timeout_seconds=5, runner=_timeout_runner
    )


@pytest.mark.parametrize(
    "backend, reason",
    [
        (_unconfigured, "embeddings_not_configured"),
        (_absent, "embedder_unavailable"),
        (_slow, "embedder_unavailable"),
    ],
)
def test_hybrid_with_the_backend_down_never_says_hybrid(
    tmp_path: Path, corpus, backend, reason: str
) -> None:
    """THE test of this PR: every way the vector channel can fail to run, one honest answer.

    `strategy` is `lexical`, `degraded` names the cause and nothing pretends the request was
    unimplemented, and not a single match carries `vector` in `matched_by` or a `vector_rank`.
    """
    data = _data(tmp_path, corpus, with_plane=True)

    response = search(QUERY, _context(data, corpus, backend(tmp_path, data)), strategy="hybrid")

    _assert_lexical_and_honest(response)
    assert reason in response.index.degraded
    assert "hybrid_not_implemented" not in response.index.degraded


def test_hybrid_over_an_index_without_a_plane_never_says_hybrid(tmp_path: Path, corpus) -> None:
    """The fourth way: a configured embedder and no plane. The manifest's `no_embeddings` says
    why, and the embedder is not paid for a query that could not be scored."""
    data = _data(tmp_path, corpus, with_plane=False)
    embedder = CountingEmbedder()

    response = search(QUERY, _context(data, corpus, embedder), strategy="hybrid")

    _assert_lexical_and_honest(response)
    assert "no_embeddings" in response.index.degraded
    assert "hybrid_not_implemented" not in response.index.degraded
    assert embedder.calls == []


# --------------------------------------------------------------------------------------------
# Criterion §13.12, first half — `import xbrain` without the `[embeddings]` extra
# --------------------------------------------------------------------------------------------


def test_the_cli_and_the_search_service_import_without_numpy() -> None:
    """In a FRESH interpreter, so a `numpy` another test already imported cannot mask it."""
    probe = (
        "import sys; sys.modules['numpy'] = None; "
        "import xbrain.cli, xbrain.knowledge.search_service, xbrain.embeddings"
    )
    completed = subprocess.run(  # nosec B603 - fixed argv, this interpreter, no shell
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
