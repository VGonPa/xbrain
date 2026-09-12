"""Tests for `scripts/xbrain-embed` — the reference embedding backend (Plan 03 §1.3).

The wrapper lives OUTSIDE the package and its dependency (sentence-transformers)
is deliberately NOT in `pyproject.toml`, so these tests never load a model: the
module's top-level imports are stdlib only, and `_encode` — the one function that
touches sentence-transformers, behind a function-local import — is monkeypatched
everywhere below.

The most valuable test in this file is the last one: it feeds what the wrapper
ACTUALLY emits into what `xbrain.embeddings` actually parses. The two halves of
the contract are written by different hands in different processes, and pinning
them with two lists that "should" match is exactly the divergence CLAUDE.md rule 5
exists to stop.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

# The wrapper is a bare script (no .py suffix), so give importlib an explicit
# source loader. Top-level imports are stdlib only → safe without the model.
_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-embed"
_LOADER = SourceFileLoader("xbrain_embed", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_embed", _LOADER)
xe = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xe)


def _request(texts: list[str], *, model: str | None = None, version: str | None = None) -> str:
    return json.dumps(
        {
            "schema_version": xe.SCHEMA_VERSION if version is None else version,
            "model": model,
            "texts": texts,
        }
    )


# ---------------------------------------------------------------------------
# 1. The request side
# ---------------------------------------------------------------------------


def test_reads_a_well_formed_request():
    request = xe._read_request(io.StringIO(_request(["uno", "dos"])))
    assert request["texts"] == ["uno", "dos"]


def test_empty_stdin_is_an_error_not_an_empty_batch():
    """Exit-0 with no vectors would look to xbrain like a count mismatch on a
    batch that never ran — name the real cause here instead."""
    with pytest.raises(SystemExit):
        xe._read_request(io.StringIO("   \n "))


def test_unparseable_request_is_an_error():
    with pytest.raises(SystemExit):
        xe._read_request(io.StringIO("not json"))


def test_a_json_list_is_not_a_request():
    with pytest.raises(SystemExit):
        xe._read_request(io.StringIO("[1, 2]"))


def test_an_unknown_request_schema_version_is_refused():
    """Both sides name their version, so an upgrade on one side is a legible
    error rather than a misread body."""
    with pytest.raises(SystemExit) as excinfo:
        xe._read_request(io.StringIO(_request(["uno"], version="99")))
    assert "99" in str(excinfo.value)


def test_a_request_without_texts_is_refused():
    with pytest.raises(SystemExit):
        xe._read_request(io.StringIO(json.dumps({"schema_version": xe.SCHEMA_VERSION})))


def test_an_empty_text_list_is_refused():
    with pytest.raises(SystemExit):
        xe._read_request(io.StringIO(_request([])))


def test_non_string_texts_are_refused():
    body = json.dumps({"schema_version": xe.SCHEMA_VERSION, "model": None, "texts": ["a", 7]})
    with pytest.raises(SystemExit):
        xe._read_request(io.StringIO(body))


# ---------------------------------------------------------------------------
# 2. The response side
# ---------------------------------------------------------------------------


def test_response_carries_every_field_of_the_contract():
    body = json.loads(xe._response("test/model", [[1.0, 0.0], [0.0, 1.0]]))
    assert body == {
        "schema_version": xe.SCHEMA_VERSION,
        "model": "test/model",
        "dimension": 2,
        "normalized": True,
        "vectors": [[1.0, 0.0], [0.0, 1.0]],
    }


def test_response_dimension_is_derived_from_the_vectors():
    """Derived, never declared: a hand-set dimension is the one field that can
    disagree with the data it describes, and xbrain refuses the batch when it does."""
    body = json.loads(xe._response("m", [[0.0] * 7]))
    assert body["dimension"] == 7


def test_a_model_that_returns_nothing_is_an_error():
    with pytest.raises(SystemExit):
        xe._response("m", [])


def test_a_model_that_returns_empty_vectors_is_an_error():
    with pytest.raises(SystemExit):
        xe._response("m", [[]])


# ---------------------------------------------------------------------------
# 3. Model selection and the end-to-end run
# ---------------------------------------------------------------------------


def _run_main(monkeypatch, capsys, argv: list[str], stdin: str) -> tuple[int, str, list]:
    seen: list = []

    def _fake_encode(model_name: str, texts: list[str]):
        seen.append((model_name, list(texts)))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(xe, "_encode", _fake_encode)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = xe.main()
    return code, capsys.readouterr().out, seen


def test_main_emits_the_contract_and_exits_zero(monkeypatch, capsys):
    code, out, _ = _run_main(monkeypatch, capsys, ["xbrain-embed"], _request(["uno", "dos"]))
    assert code == 0
    body = json.loads(out)
    assert body["schema_version"] == xe.SCHEMA_VERSION
    assert body["dimension"] == 2
    assert len(body["vectors"]) == 2


def test_model_defaults_to_the_wrappers_own(monkeypatch, capsys):
    """xbrain sends `model: null` when `[embeddings].model` is unset, meaning
    'your default' — the response must then name the model it really used, because
    that string is what the index manifest stores."""
    _, out, seen = _run_main(monkeypatch, capsys, ["xbrain-embed"], _request(["uno"]))
    assert seen[0][0] == xe.DEFAULT_MODEL
    assert json.loads(out)["model"] == xe.DEFAULT_MODEL


def test_model_in_the_request_body_is_honoured(monkeypatch, capsys):
    _, out, seen = _run_main(
        monkeypatch, capsys, ["xbrain-embed"], _request(["uno"], model="other/model")
    )
    assert seen[0][0] == "other/model"
    assert json.loads(out)["model"] == "other/model"


def test_the_command_line_flag_wins_over_the_request(monkeypatch, capsys):
    _, _, seen = _run_main(
        monkeypatch,
        capsys,
        ["xbrain-embed", "--model", "flag/model"],
        _request(["uno"], model="body/model"),
    )
    assert seen[0][0] == "flag/model"


def test_the_texts_reach_the_model_unchanged(monkeypatch, capsys):
    """The query / passage prefixes are xbrain's and are already applied when the
    texts arrive. Prefixing again here would double them, and no model expects
    `"query: query: …"` — a defect that produces worse vectors and no error."""
    _, _, seen = _run_main(
        monkeypatch, capsys, ["xbrain-embed"], _request(["query: qué es un transformer"])
    )
    assert seen[0][1] == ["query: qué es un transformer"]


# ---------------------------------------------------------------------------
# 4. The two halves of the contract, bound
# ---------------------------------------------------------------------------


def test_wrapper_and_xbrain_agree_on_the_schema_version():
    """The two sides spell this string INDEPENDENTLY — the wrapper cannot import
    xbrain (it runs under whichever python has the model installed), so the
    duplication is deliberate. Nothing else pins them equal: change one and every
    other test still passes while every real batch dies at the far end.
    """
    from xbrain.embeddings import SCHEMA_VERSION

    assert xe.SCHEMA_VERSION == SCHEMA_VERSION


def test_what_the_wrapper_emits_is_what_xbrain_parses(monkeypatch, capsys):
    """The round trip, through the PUBLIC API of both halves.

    This is the test that would have caught a field renamed on one side only. The
    wrapper's real stdout is fed to `embed_texts` as its subprocess output, so the
    assertion is about two implementations agreeing — not about two literal dicts
    in this file agreeing with each other.
    """
    from xbrain.embeddings import embed_texts

    _, out, _ = _run_main(monkeypatch, capsys, ["xbrain-embed"], _request(["uno", "dos"]))

    def _runner(_argv, **_kwargs):
        import subprocess

        return subprocess.CompletedProcess(args=["e"], returncode=0, stdout=out, stderr="")

    batch = embed_texts(["uno", "dos"], command="xbrain-embed", model=None, runner=_runner)
    assert batch.model == xe.DEFAULT_MODEL
    assert batch.dimension == 2
    assert len(batch.vectors) == 2
    assert batch.renormalized is False


def test_the_wrapper_loads_no_model_library_at_import_time():
    """Its dependency is NOT in `pyproject.toml`, so a top-level
    `from sentence_transformers import …` would make the script unimportable —
    including by this very test file — on every machine that has not installed it.

    Derived from the AST of the file on disk: moving the import to the top makes
    this go red.
    """
    tree = ast.parse(_PATH.read_text(encoding="utf-8"))
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }
    assert top_level <= set(sys.stdlib_module_names), (
        f"scripts/xbrain-embed imports {sorted(top_level - set(sys.stdlib_module_names))} "
        "at module level, so it cannot run without them installed"
    )
