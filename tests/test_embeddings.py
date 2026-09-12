"""Tests for `xbrain.embeddings` — the external-embedder subprocess wrapper.

`embed_texts` shells out to an EXTERNAL embedding backend (config
`[embeddings].command`) and reads back one vector per input text. It is the third
sibling of the same locked shape as `transcribe.py` and `vision.py`: it imports NO
ML library — the model lives OUTSIDE xbrain core, invoked as a subprocess located
via config/PATH (Plan 03 §1.1).

The contract (Plan 03 §1.2): the request travels as JSON on **stdin** (argv cannot
carry thousands of chunk texts), the response as JSON on **stdout**. Every row of
the §1.2 validation table gets a test here, because a vector index that silently
accepts a wrong-shaped, non-finite or wrong-model batch is an index that answers
queries with numbers nobody can trace.

Every test injects a fake `runner` (a `subprocess.run` stand-in), so NO real
embedder, model, GPU or network is ever touched — Plan 03 §12, asserted at the end
of this file over the whole suite, not just this module.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
from pathlib import Path

import pytest

from xbrain.embeddings import (
    SCHEMA_VERSION,
    EmbedderFailed,
    EmbedderNotFound,
    EmbeddingBatch,
    embed_query,
    embed_texts,
    embed_passages,
)

# A distinctive string used as an input text wherever a test needs to prove that a
# failure message talks about SHAPES and never about CONTENT (Plan 03 §10.5). Named
# for what it is rather than `SECRET`, which detect-secrets flags as a credential
# keyword — widening the baseline to admit a decoy would blunt the real gate.
CORPUS_SENTINEL = "zzz-corpus-sentinel-sentence-zzz"


def _payload(
    vectors: list[list[float]],
    *,
    model: str = "test/model",
    dimension: int | None = None,
    normalized: bool = True,
    schema_version: str = SCHEMA_VERSION,
) -> str:
    """A well-formed backend response, with one knob per validation branch."""
    body: dict[str, object] = {
        "schema_version": schema_version,
        "model": model,
        "dimension": len(vectors[0]) if dimension is None and vectors else (dimension or 0),
        "normalized": normalized,
        "vectors": vectors,
    }
    if dimension is not None:
        body["dimension"] = dimension
    return json.dumps(body)


def _completed(stdout: str, *, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["e"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _runner(stdout: str, *, returncode: int = 0, stderr: str = ""):
    """A fake `subprocess.run` recording argv and the stdin payload of each call."""
    calls: list[dict] = []

    def _run(argv, **kwargs):
        calls.append({"argv": list(argv), "kwargs": kwargs})
        return _completed(stdout, returncode=returncode, stderr=stderr)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


# ---------------------------------------------------------------------------
# 1. The subprocess: argv, stdin, and never a shell
# ---------------------------------------------------------------------------


def test_command_is_shlex_split_so_a_wrapper_works():
    """A multi-token command (`python -m my_embedder`) is a supported backend."""
    runner = _runner(_payload([[1.0, 0.0]]))
    embed_texts(["hola"], command="python -m my_embedder", model=None, runner=runner)
    assert runner.calls[0]["argv"] == ["python", "-m", "my_embedder"]  # type: ignore[attr-defined]


def test_command_is_never_handed_to_a_shell():
    """`shlex` + no shell, per Plan 03 §10.2.

    Seen red by dropping the `shlex.split` and passing `shell=True`: the argv then
    arrives as ONE string and `;` would be a command separator rather than a
    literal token of the program name.
    """
    runner = _runner(_payload([[1.0, 0.0]]))
    embed_texts(["hola"], command="sh -c 'rm -rf /'", model=None, runner=runner)
    call = runner.calls[0]  # type: ignore[attr-defined]
    assert call["argv"] == ["sh", "-c", "rm -rf /"]
    assert call["kwargs"].get("shell") is not True


def test_model_is_passed_through_as_a_flag():
    runner = _runner(_payload([[1.0, 0.0]]))
    embed_texts(["hola"], command="xbrain-embed", model="intfloat/e5", runner=runner)
    argv = runner.calls[0]["argv"]  # type: ignore[attr-defined]
    assert argv == ["xbrain-embed", "--model", "intfloat/e5"]


def test_texts_travel_as_json_on_stdin_not_argv():
    """argv cannot carry thousands of chunk texts; stdin can (Plan 03 §1.2).

    Seen red by appending the texts to argv: the payload key disappears and the
    texts show up in the argv assertion below.
    """
    runner = _runner(_payload([[1.0, 0.0], [0.0, 1.0]]))
    embed_texts(["uno", "dos"], command="xbrain-embed", model=None, runner=runner)
    call = runner.calls[0]  # type: ignore[attr-defined]
    sent = json.loads(call["kwargs"]["input"])
    assert sent == {"schema_version": SCHEMA_VERSION, "model": None, "texts": ["uno", "dos"]}
    assert "uno" not in call["argv"]


def test_timeout_is_handed_to_the_runner():
    runner = _runner(_payload([[1.0, 0.0]]))
    embed_texts(["hola"], command="xbrain-embed", model=None, timeout_seconds=17, runner=runner)
    assert runner.calls[0]["kwargs"]["timeout"] == 17  # type: ignore[attr-defined]


def test_runner_defaults_to_subprocess_run_resolved_at_call_time(monkeypatch):
    """Resolved at CALL time, like transcribe/vision, so tests can substitute it."""
    seen: list[list[str]] = []

    def _fake_run(argv, **_kwargs):
        seen.append(list(argv))
        return _completed(_payload([[1.0, 0.0]]))

    monkeypatch.setattr(subprocess, "run", _fake_run)
    embed_texts(["hola"], command="xbrain-embed", model=None)
    assert seen == [["xbrain-embed"]]


# ---------------------------------------------------------------------------
# 2. Operator errors: the binary, the exit code, the clock
# ---------------------------------------------------------------------------


def test_unconfigured_command_is_an_actionable_operator_error():
    """`[embeddings].command` starts EMPTY — there is no bundled default, because a
    default would be choosing the model without evaluating it (Plan 03 §1.2)."""
    with pytest.raises(EmbedderNotFound) as excinfo:
        embed_texts(["hola"], command="   ", model=None, runner=_runner("{}"))
    assert "[embeddings].command" in str(excinfo.value)


def test_missing_binary_raises_embedder_not_found():
    def _run(_argv, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "xbrain-embed")

    with pytest.raises(EmbedderNotFound) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_run)
    assert "[embeddings].command" in str(excinfo.value)


def test_non_executable_binary_raises_embedder_not_found():
    def _run(_argv, **_kwargs):
        raise PermissionError(13, "Permission denied", "xbrain-embed")

    with pytest.raises(EmbedderNotFound):
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_run)


def test_non_zero_exit_reports_the_code_and_the_stderr():
    runner = _runner("", returncode=3, stderr="CUDA out of memory")
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    message = str(excinfo.value)
    assert "3" in message and "CUDA out of memory" in message


def test_timeout_raises_embedder_failed_naming_the_budget():
    def _run(_argv, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="xbrain-embed", timeout=600)

    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, timeout_seconds=600, runner=_run)
    assert "600" in str(excinfo.value)


def test_non_utf8_stdout_raises_embedder_failed():
    def _run(_argv, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_run)
    assert "UTF-8" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3. The §1.2 validation table — one test per row
# ---------------------------------------------------------------------------


def test_unparseable_stdout_raises_embedder_failed():
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_runner("not json"))
    assert "JSON" in str(excinfo.value)


def test_a_json_list_is_not_a_response():
    with pytest.raises(EmbedderFailed):
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_runner("[1, 2]"))


def test_unknown_schema_version_names_the_version_xbrain_expects():
    """Naming the EXPECTED version is what makes the message actionable: the
    operator has to know which side to upgrade."""
    runner = _runner(_payload([[1.0, 0.0]], schema_version="99"))
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    message = str(excinfo.value)
    assert "99" in message and SCHEMA_VERSION in message


def test_vector_count_mismatch_names_both_figures():
    """Two texts in, one vector out: the pairing is lost and every downstream
    chunk_id would point at the wrong row. Both counts go in the message."""
    runner = _runner(_payload([[1.0, 0.0]]))
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["uno", "dos"], command="xbrain-embed", model=None, runner=runner)
    message = str(excinfo.value)
    assert "2" in message and "1" in message


def test_inconsistent_dimension_names_the_culprit_index():
    """A ragged batch cannot become a matrix. The index of the offending vector is
    the only thing that makes the backend bug findable."""
    runner = _runner(_payload([[1.0, 0.0], [1.0, 0.0, 0.0]], dimension=2))
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["uno", "dos"], command="xbrain-embed", model=None, runner=runner)
    assert "1" in str(excinfo.value)


def test_declared_dimension_must_match_the_vectors_actually_sent():
    """The backend's own `dimension` field is a claim; the vectors are the source.

    A backend that declares 768 and emits 384 would write a manifest nothing can
    read back. Rule 9: assert on the source, never on the reported conclusion.
    """
    runner = _runner(_payload([[1.0, 0.0]], dimension=768))
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    message = str(excinfo.value)
    assert "768" in message and "2" in message


def test_dimension_other_than_the_expected_one_is_a_hard_error():
    """Plan 03 §1.2, last row: THE INDEX IS NEVER MIXED BETWEEN MODELS.

    `expected_dimension` is what the caller read from the manifest. A batch of a
    different width is refused here rather than appended to a matrix that would
    then hold two models' geometry and rank them against each other.
    """
    runner = _runner(_payload([[1.0, 0.0]]))
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(
            ["hola"], command="xbrain-embed", model=None, expected_dimension=768, runner=runner
        )
    message = str(excinfo.value)
    assert "768" in message and "2" in message


def test_expected_dimension_that_matches_is_accepted():
    runner = _runner(_payload([[1.0, 0.0]]))
    batch = embed_texts(
        ["hola"], command="xbrain-embed", model=None, expected_dimension=2, runner=runner
    )
    assert batch.dimension == 2


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_values_are_rejected(bad: str):
    """`json.loads` accepts `NaN`/`Infinity` happily. A NaN in the matrix poisons
    every dot product it touches and ranks unpredictably rather than failing."""
    body = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "model": "test/model",
            "dimension": 2,
            "normalized": True,
            "vectors": [[1.0, 0.0]],
        }
    ).replace("1.0", bad)
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_runner(body))
    assert "finite" in str(excinfo.value).lower()


def test_a_non_numeric_value_is_rejected():
    runner = _runner(_payload([["dos", 0.0]]))  # type: ignore[list-item]
    with pytest.raises(EmbedderFailed):
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)


def test_a_vectors_field_that_is_not_a_list_is_rejected():
    """`vectors` as an object, not an array. `len()` of a dict is its key count, so
    an unchecked implementation would compare the wrong number against the text
    count and then iterate the KEYS as if they were vectors."""
    body = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "model": "test/model",
            "dimension": 2,
            "normalized": True,
            "vectors": {"0": [1.0, 0.0]},
        }
    )
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_runner(body))
    assert "vectors" in str(excinfo.value)


def test_a_vector_that_is_not_a_list_is_rejected():
    """A scalar where a row should be: `vectors: [0.5]` is one number, not one
    one-dimensional point, and nothing later would notice the difference."""
    body = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "model": "test/model",
            "dimension": 2,
            "normalized": True,
            "vectors": [0.5],
        }
    )
    with pytest.raises(EmbedderFailed):
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_runner(body))


def test_an_empty_vector_is_rejected():
    """A zero-width vector is not a point in any space; it would make `dimension`
    0 and every similarity identical."""
    runner = _runner(_payload([[]], dimension=0))
    with pytest.raises(EmbedderFailed):
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)


def test_a_response_without_a_model_is_rejected():
    """The model name goes into the manifest and is what later invalidates the
    vector plane. A batch that cannot say which model produced it is unusable."""
    body = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "dimension": 2,
            "normalized": True,
            "vectors": [[1.0, 0.0]],
        }
    )
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=_runner(body))
    assert "model" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 4. Normalization — the post-condition, not the backend's claim
# ---------------------------------------------------------------------------


def test_unnormalized_vectors_are_l2_normalized_and_the_fact_is_recorded():
    """Plan 03 §1.2: `normalized: false` ⇒ xbrain normalizes, and RECORDS that it
    did — the manifest has to be able to say so."""
    runner = _runner(_payload([[3.0, 4.0]], normalized=False))
    batch = embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    assert batch.vectors == ((0.6, 0.8),)
    assert batch.normalized is True
    assert batch.renormalized is True


def test_already_unit_vectors_are_returned_untouched():
    runner = _runner(_payload([[1.0, 0.0]], normalized=True))
    batch = embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    assert batch.vectors == ((1.0, 0.0),)
    assert batch.renormalized is False


def test_a_false_normalized_claim_is_verified_not_trusted():
    """Rule 9 applied to a data contract: `normalized: true` is a CONCLUSION the
    backend reports, and the vectors are the SOURCE.

    Trusting the claim on a batch of norm 5 would make every stored cosine
    similarity five times its true value — silently, since 03.3 computes cosine as
    a plain dot product precisely BECAUSE the rows are unit-length. Verifying costs
    the same pass over the data that normalizing does.

    Seen red by returning the vectors unchanged whenever the flag says true.
    """
    runner = _runner(_payload([[3.0, 4.0]], normalized=True))
    batch = embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    assert batch.vectors == ((0.6, 0.8),)
    assert batch.renormalized is True


def test_a_zero_vector_cannot_be_normalized_and_says_so():
    """The branch that turns a `ZeroDivisionError` traceback into an operator
    error. A zero vector has no direction: it would sit at cosine 0 from every
    query forever and never be retrievable."""
    runner = _runner(_payload([[0.0, 0.0]], normalized=False))
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    assert "zero" in str(excinfo.value).lower()


def test_normalization_survives_a_json_float32_round_trip():
    """Real backends emit float32 printed as decimal, so an exactly-unit vector
    arrives a few ulps off. That must not be reported as a renormalization."""
    vector = _unit([0.31, 0.77, -0.55, 0.12])
    runner = _runner(_payload([vector], normalized=True))
    batch = embed_texts(["hola"], command="xbrain-embed", model=None, runner=runner)
    assert batch.renormalized is False
    assert math.isclose(sum(v * v for v in batch.vectors[0]), 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 5. Prefixes — a property of the MODEL, applied on the right side
# ---------------------------------------------------------------------------


def test_passage_prefix_is_applied_to_every_text():
    runner = _runner(_payload([[1.0, 0.0], [0.0, 1.0]]))
    embed_passages(
        ["uno", "dos"], command="xbrain-embed", model=None, prefix="passage: ", runner=runner
    )
    sent = json.loads(runner.calls[0]["kwargs"]["input"])  # type: ignore[attr-defined]
    assert sent["texts"] == ["passage: uno", "passage: dos"]


def test_query_prefix_is_applied_to_the_query():
    runner = _runner(_payload([[1.0, 0.0]]))
    embed_query(
        "qué es un transformer", command="xbrain-embed", model=None, prefix="query: ", runner=runner
    )
    sent = json.loads(runner.calls[0]["kwargs"]["input"])  # type: ignore[attr-defined]
    assert sent["texts"] == ["query: qué es un transformer"]


def test_swapping_the_two_prefixes_is_visible():
    """Plan 03 §7 step 8. The E5 family degrades NOTABLY when the two sides are
    swapped, and nothing else in the pipeline would ever report it — the vectors
    are still well-formed, still unit-length, just worse.

    Both calls here carry the wrong prefix; the assertion is that the payload shows
    which one was used, so a swap in `embed_query`/`embed_passages` cannot hide.
    """
    passages_runner = _runner(_payload([[1.0, 0.0]]))
    query_runner = _runner(_payload([[1.0, 0.0]]))
    embed_passages(["x"], command="e", model=None, prefix="passage: ", runner=passages_runner)
    embed_query("x", command="e", model=None, prefix="query: ", runner=query_runner)
    passage_sent = json.loads(passages_runner.calls[0]["kwargs"]["input"])  # type: ignore[attr-defined]
    query_sent = json.loads(query_runner.calls[0]["kwargs"]["input"])  # type: ignore[attr-defined]
    assert passage_sent["texts"] == ["passage: x"]
    assert query_sent["texts"] == ["query: x"]
    assert passage_sent["texts"] != query_sent["texts"]


def test_an_empty_prefix_leaves_the_text_alone():
    """Most models want no prefix at all, and `""` is the default — an accidental
    space would change every stored vector."""
    runner = _runner(_payload([[1.0, 0.0]]))
    embed_passages(["uno"], command="e", model=None, prefix="", runner=runner)
    sent = json.loads(runner.calls[0]["kwargs"]["input"])  # type: ignore[attr-defined]
    assert sent["texts"] == ["uno"]


def test_embed_query_returns_a_single_vector_batch():
    runner = _runner(_payload([[1.0, 0.0]]))
    batch = embed_query("hola", command="e", model=None, prefix="", runner=runner)
    assert isinstance(batch, EmbeddingBatch)
    assert len(batch.vectors) == 1


def test_embed_query_refuses_a_backend_that_answers_with_two_vectors():
    """One query in, one vector out — and the refusal comes from the COUNT CHECK in
    `embed_texts`, which is why `embed_query` carries no guard of its own.

    Asserted on the message, because that is what distinguishes the two possible
    mechanisms. A `len(vectors) == 1` check inside `embed_query` would be a second
    copy of an invariant `embed_texts` already owns, it could never fire, and a test
    that looked like it exercised such a guard would in fact be satisfied upstream.
    """
    runner = _runner(_payload([[1.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_query("hola", command="e", model=None, prefix="", runner=runner)
    assert "2 vectors for 1 texts" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6. Boundaries
# ---------------------------------------------------------------------------


def test_no_texts_never_invokes_the_subprocess():
    """Plan 03 §9, first row. There is nothing to embed and nothing the response
    could be validated against — `dimension` would have to be invented."""
    runner = _runner(_payload([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        embed_texts([], command="xbrain-embed", model=None, runner=runner)
    assert runner.calls == []  # type: ignore[attr-defined]


def test_the_batch_is_immutable():
    """It is handed to the index writer, which must not be able to edit history."""
    runner = _runner(_payload([[1.0, 0.0]]))
    batch = embed_texts(["hola"], command="e", model=None, runner=runner)
    with pytest.raises(Exception):
        batch.model = "other"  # type: ignore[misc]
    assert isinstance(batch.vectors, tuple)
    assert isinstance(batch.vectors[0], tuple)


@pytest.mark.parametrize(
    "stdout",
    [
        "not json",
        json.dumps(
            {
                "schema_version": "99",
                "model": "m",
                "dimension": 2,
                "normalized": True,
                "vectors": [[1.0, 0.0]],
            }
        ),
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "model": "m",
                "dimension": 2,
                "normalized": True,
                "vectors": [],
            }
        ),
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "model": "m",
                "dimension": 2,
                "normalized": False,
                "vectors": [[0.0, 0.0]],
            }
        ),
    ],
)
def test_no_failure_message_ever_quotes_the_embedded_text(stdout: str):
    """Plan 03 §10.5/§10.8: the corpus is personal, and an operator error ends up
    in a terminal, a log or a pasted issue. Messages name COUNTS, INDICES and
    DIMENSIONS — never content.

    Seen red by f-stringing `texts` into any of these four messages.
    """
    with pytest.raises(EmbedderFailed) as excinfo:
        embed_texts([CORPUS_SENTINEL], command="xbrain-embed", model=None, runner=_runner(stdout))
    assert CORPUS_SENTINEL not in str(excinfo.value)


# ---------------------------------------------------------------------------
# 7. The architecture guards — no ML in core, no real embedder in the suite
# ---------------------------------------------------------------------------


def test_embeddings_imports_no_ml_library():
    """The locked architecture (Plan 03 §0): xbrain core carries NO embedding/ML
    dependency — the model is an external subprocess. Guard the module so a future
    edit cannot quietly `import sentence_transformers`."""
    import xbrain.embeddings as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import sentence_transformers",
        "import torch",
        "import transformers",
        "import numpy",
        "import onnxruntime",
    ):
        assert forbidden not in source


def _imported_module_roots(path: Path) -> set[str]:
    """Every top-level package name imported by a Python file, via the AST.

    The AST, not a substring scan: `"sentence_transformers"` appears legitimately
    inside a monkeypatch target and inside this very docstring, and a grep-shaped
    guard would either fire on those or be loosened until it fired on nothing. Only
    a real `Import` / `ImportFrom` node counts.
    """
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_test_in_the_suite_constructs_a_real_embedder():
    """Plan 03 §12, as a SUITE guard rather than a sentence in a plan.

    The whole test suite runs offline in seconds, and it stays that way only if
    nothing ever loads a model. A single `import sentence_transformers` in one test
    file downloads hundreds of megabytes on a cold CI runner and turns a green gate
    into a flaky one — and it would arrive as a plausible-looking test of the very
    module this file covers.

    Derived on both sides: the left is every test file on disk, the right is the
    forbidden set. Adding such an import to ANY test makes this go red.
    """
    forbidden = {"sentence_transformers", "torch", "transformers", "onnxruntime", "mlx", "mlx_lm"}
    offenders = {
        test_file.name: sorted(_imported_module_roots(test_file) & forbidden)
        for test_file in sorted(Path(__file__).resolve().parent.glob("test_*.py"))
        if _imported_module_roots(test_file) & forbidden
    }
    assert not offenders, (
        "these tests import a real model library, so the suite no longer runs "
        f"offline in seconds: {offenders}"
    )
