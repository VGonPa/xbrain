# tests/test_knowledge_cli.py
"""`xbrain knowledge inspect` and `xbrain eval` (Plan 01 §8, step 29).

Two commands, the minimum for the plan's exit gate to be DEMONSTRABLE rather than asserted:
one that shows the unified corpus, and one that runs an evaluation which can fail.

THE CONVENTION THEY FOLLOW is the one `list-videos` already established: `--json` writes a
stable JSON document to stdout and NOTHING else, diagnostics go to stderr/logging (spec
§3.7.9), and both commands are strictly read-only — no writes to the store, no snapshot.

A single stray `print` in a `--json` path breaks every consumer downstream, and it breaks
them with a parse error a long way from the cause, so it is tested by parsing the whole of
stdout rather than by looking for a substring in it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from xbrain.cli import app
from xbrain.knowledge import contracts, evaluation

FIXTURES = Path(__file__).parent / "fixtures"
runner = CliRunner()


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch) -> Path:
    """A repo-shaped temp dir with a data/ built from the fixture corpus."""
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(json.dumps(raw["items"], indent=2), encoding="utf-8")
    (data / "topics.json").write_text(json.dumps(raw["topics"], indent=2), encoding="utf-8")
    (data / "vocab.yaml").write_text(
        yaml.safe_dump({"topics": list(raw["vocab"].values())}, allow_unicode=True),
        encoding="utf-8",
    )
    (tmp_path / "eval").mkdir()
    shutil.copy(FIXTURES / "knowledge_goldenset.yaml", tmp_path / "eval" / "golden-set.yaml")
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "vault"\noutput_subdir = "x-knowledge"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _json_stdout(result) -> dict | list:
    """Parse the WHOLE of stdout. A stray log line makes this raise, which is the point."""
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# 29 — knowledge inspect
# ---------------------------------------------------------------------------


def test_inspect_emits_pure_json_on_stdout(workspace: Path) -> None:
    """Spec §3.7.9: JSON results never mix human diagnostics into stdout.

    Parsed as a whole document, so one stray `print` fails this test instead of failing a
    consumer's parser weeks later, far from the cause.
    """
    from xbrain.knowledge.contracts import EVIDENCE_SCHEMA_VERSION

    payload = _json_stdout(runner.invoke(app, ["knowledge", "inspect", "k08", "--json"]))
    assert payload["item"]["item_id"] == "k08"
    # The version of the shapes it dumps, read off the contract and never stamped by hand
    # (U-1): the surfaces and chunks in this payload are the `EvidenceBundle`'s.
    assert payload["schema_version"] == EVIDENCE_SCHEMA_VERSION == "2"


def test_inspect_returns_surfaces_with_provenance_and_locator(workspace: Path) -> None:
    """Acceptance 1: every surface carries origin, trust_class, attribution, locator and
    fingerprint — the five things that make a fragment checkable rather than merely quoted."""
    payload = _json_stdout(
        runner.invoke(app, ["knowledge", "inspect", "k08", "--surfaces", "--json"])
    )
    surfaces = payload["surfaces"]
    assert surfaces
    for surface in surfaces:
        assert surface["origin"] and surface["trust_class"] and surface["locator"]["kind"]
        assert len(surface["fingerprint"]) == 64
    kinds = {s["surface_type"] for s in surfaces}
    assert {"video_transcript", "video_frame", "video_digest"} <= kinds


def test_inspect_shows_the_quoted_author_not_the_poster(workspace: Path) -> None:
    """The attribution rule, visible to a human at the CLI.

    CLAUDE.md rule 7: before building an instrument to detect a defect, ask whether SHOWING
    the evidence makes it self-evident. A reader who sees the quoted post's own handle beside
    the poster's needs no judge to spot a mis-attribution.
    """
    payload = _json_stdout(
        runner.invoke(app, ["knowledge", "inspect", "k07", "--surfaces", "--json"])
    )
    quoted = next(s for s in payload["surfaces"] if s["surface_type"] == "quoted_post")
    assert quoted["attribution"]["handle"] == "othervoice"
    assert payload["item"]["author"]["handle"] == "vgonpa"


def test_inspect_reports_failures_and_unfetched_links(workspace: Path) -> None:
    """A dead link is structured state, not a silence (spec §4)."""
    payload = _json_stdout(runner.invoke(app, ["knowledge", "inspect", "k11", "--json"]))
    assert payload["item"]["failed_sources"][0]["failure_reason"] == "not_found"
    assert payload["item"]["unfetched_links"][0]["reason"] == "http_error"


def test_inspect_can_return_chunks_verbatim(workspace: Path) -> None:
    """Acceptance 4, at the CLI: the chunk offsets slice the surface back to the chunk."""
    payload = _json_stdout(
        runner.invoke(app, ["knowledge", "inspect", "k03", "--chunks", "--json"])
    )
    surfaces = {s["surface_id"]: s["text"] for s in payload["surfaces"]}
    assert payload["chunks"]
    for chunk in payload["chunks"]:
        body = surfaces[chunk["surface_id"]]
        assert body[chunk["char_start"] : chunk["char_end"]] == chunk["text"]


def test_inspect_a_topic(workspace: Path) -> None:
    payload = _json_stdout(
        runner.invoke(
            app, ["knowledge", "inspect", "--topic", "agent-evaluation", "--surfaces", "--json"]
        )
    )
    assert payload["topic"]["slug"] == "agent-evaluation"
    assert payload["topic"]["overview"]["origin"] == "llm"
    assert {s["surface_type"] for s in payload["surfaces"]} >= {"topic_overview", "topic_note"}


@pytest.mark.parametrize(
    ("argv", "surface"),
    [
        (["knowledge", "inspect", "k08", "--json"], "_inspect_item"),
        (["knowledge", "inspect", "--topic", "agent-evaluation", "--json"], "_inspect_topic"),
    ],
)
def test_both_inspect_payloads_read_their_version_off_the_contract(
    workspace: Path, monkeypatch, argv: list[str], surface: str
) -> None:
    """The stamp is DERIVED from `contracts.EVIDENCE_SCHEMA_VERSION`, not a literal that
    currently agrees with it (U-1).

    WHY EQUALITY IS NOT ENOUGH, and this test exists because the equality version was
    measured NOT catching it: reverting `_inspect_topic` alone to a hardcoded `"1"` left the
    whole suite green, because `"1"` is exactly what the contract says today. An assertion
    that a payload equals the current number is satisfied by a payload that will never move
    again — CLAUDE.md rule 1, satisfied for the wrong reason.

    So the version is INJECTED instead. Both inspect helpers import the constant inside the
    function body, at call time, which is what makes it reachable here; a hardcoded literal
    cannot follow an injected value, so this goes red on the exact mutation the equality
    assertion survived. The sentinel is deliberately a string no contract will ever declare,
    so it cannot pass by coincidence at any future version.

    Both payloads are covered because `_inspect_topic` had no assertion on its
    `schema_version` at all: the item path was pinned and the topic path was free to drift.
    """
    import xbrain.knowledge.contracts as contracts

    monkeypatch.setattr(contracts, "EVIDENCE_SCHEMA_VERSION", "sentinel-not-a-version")
    payload = _json_stdout(runner.invoke(app, argv))
    assert payload["schema_version"] == "sentinel-not-a-version", surface


def test_inspect_an_unknown_item_is_an_actionable_error(workspace: Path) -> None:
    result = runner.invoke(app, ["knowledge", "inspect", "nope", "--json"])
    assert result.exit_code != 0


def test_inspect_writes_nothing(workspace: Path) -> None:
    """Read-only, asserted by hash. No snapshot either — there is nothing to snapshot."""
    store = workspace / "data" / "items.json"
    before = hashlib.sha256(store.read_bytes()).hexdigest()
    runner.invoke(app, ["knowledge", "inspect", "k08", "--surfaces", "--chunks", "--json"])
    assert hashlib.sha256(store.read_bytes()).hexdigest() == before
    assert not (workspace / "data" / "snapshots").exists()


# ---------------------------------------------------------------------------
# 29 — eval
# ---------------------------------------------------------------------------


def test_eval_emits_metrics_per_stratum_and_declares_gaps(workspace: Path) -> None:
    """Acceptance 9: per strategy x stratum x provenance, gaps declared, no invented zeros."""
    payload = _json_stdout(runner.invoke(app, ["eval", "--strategy", "lexical", "--json"]))
    assert payload["strategy"] == "lexical"
    assert payload["by_stratum"]["expansion"] == {"coverage": "sin cobertura"}
    assert "recall@10" not in payload
    assert set(payload["by_provenance"]) == {"construido", "real"}


def test_eval_writes_its_report_where_it_is_gitignored(workspace: Path) -> None:
    """The report goes to `data/` (untracked), never to `eval/` (tracked).

    The golden set is versioned because it is questions and ids; the report is not, because
    it carries excerpts of the corpus. Putting the report in `eval/` would quietly publish
    them.
    """
    result = runner.invoke(app, ["eval", "--report", "data/eval-report.json"])
    assert result.exit_code == 0
    assert json.loads((workspace / "data" / "eval-report.json").read_text())["strategy"]
    assert "sin cobertura" in (workspace / "data" / "eval-report.md").read_text()


def test_eval_can_fail_and_says_which_bucket(workspace: Path) -> None:
    """Acceptance 10, at the CLI: a threshold turns the report into a gate that exits non-zero.

    Run over the FIXTURE corpus, so this is the mechanical gate B1 bought — unlike the Plan
    02 and 03 gates, which need the real corpus and are signed measurements instead.
    """
    ok = runner.invoke(app, ["eval", "--min-recall", "0.5", "--k", "20"])
    assert ok.exit_code == 0
    bad = runner.invoke(app, ["eval", "--min-recall", "1.0", "--k", "1"])
    assert bad.exit_code != 0
    assert "recall@1" in bad.output


def test_eval_does_not_write_to_the_store(workspace: Path) -> None:
    """Acceptance 11: the harness never modifies items.json."""
    store = workspace / "data" / "items.json"
    before = hashlib.sha256(store.read_bytes()).hexdigest()
    runner.invoke(app, ["eval", "--json"])
    assert hashlib.sha256(store.read_bytes()).hexdigest() == before


def test_eval_reports_the_corpus_it_measured(workspace: Path) -> None:
    """CLAUDE.md rule 2 at the boundary the user reads."""
    payload = _json_stdout(runner.invoke(app, ["eval", "--json"]))
    assert payload["corpus"]["items"] == 12
    assert payload["corpus"]["chunks"] > 0


# The strategy that CANNOT be measured, injected rather than borrowed (F-2).
#
# The two fail-closed tests below need a case no backend can score. Until the evaluator
# regained its derived `SUPPORTED_FILTERS`, that was free: `lexical` could push only
# `has_surfaces` and `origins`, so trimming the golden set to FX7 — a `source` filter —
# left every bucket empty. Closing that gap removed the construction along with it, which
# is rule 6 at work: the repair invalidated the evidence its own guards stood on.
#
# Borrowing the next unimplemented entry of the frozen `Strategy` literal would rebuild the
# same coupling one level up — a fail-closed guard whose survival depends on Plan 03 not
# landing. So the backend is INVENTED: it exists (`IMPLEMENTED_STRATEGIES`, or
# `resolve_strategy` degrades it to `lexical` and every filter is pushed after all) and it
# can push no filter at all (`SUPPORTED_FILTERS`).
STUB_BACKEND = "stub_backend_that_pushes_no_filter"


@pytest.fixture()
def unscorable_strategy(monkeypatch) -> str:
    """A retrieval backend that runs and can apply no filter, so a filtered case is UNMEASURED."""
    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical", STUB_BACKEND}))
    monkeypatch.setitem(evaluation.SUPPORTED_FILTERS, STUB_BACKEND, frozenset())
    return STUB_BACKEND


def test_eval_with_a_threshold_fails_when_nothing_could_be_measured(
    workspace: Path, unscorable_strategy: str
) -> None:
    """M2: a gate that compared the threshold against NOTHING must not report PASS.

    `_failures` skips every bucket with no coverage and every metric carrying the sentinel —
    correctly, because naming one would be the fabricated zero of spec §8.6.8. But `passed`
    is literally "no failures", so when the threshold reaches no bucket at all the strictest
    gate that exists comes out green over zero comparisons. That is the FAIL-OPEN cell of
    CLAUDE.md rule 11, inside the command whose acceptance criterion 10 is "the evaluation
    can fail".

    Driven through the real CLI, because the exit code is the only surface a caller reads:
    the golden set is trimmed to FX7, and the run is pointed at an INJECTED backend that can
    push no filter at all, so the case is UNMEASURED and every bucket ends up empty. The
    first version of this test used `lexical` for that, which stopped working the day the
    evaluator regained all eight filters — see `unscorable_strategy`.
    """
    golden = yaml.safe_load((workspace / "eval" / "golden-set.yaml").read_text(encoding="utf-8"))
    golden["cases"] = [c for c in golden["cases"] if c["id"] == "FX7"]
    golden.pop("scenarios", None)
    (workspace / "eval" / "golden-set.yaml").write_text(
        yaml.safe_dump(golden, allow_unicode=True), encoding="utf-8"
    )

    result = runner.invoke(app, ["eval", "--min-recall", "1.0", "--strategy", unscorable_strategy])

    assert result.exit_code != 0, (
        "a threshold of 1.0 passed having scored zero cases:\n" + result.output
    )
    assert "0" in result.output and "medid" in result.output, result.output


def test_eval_refuses_a_vector_strategy_without_naming_the_model(workspace: Path) -> None:
    """Plan 03 §3.2: a vector measurement is a measurement OF A MODEL. Without the flag there
    is no model to measure, and answering with lexical numbers under a vector request is the
    F-2 pretence — so the command stops and names the flag."""
    for strategy in ("vector", "hybrid"):
        result = runner.invoke(app, ["eval", "--strategy", strategy])
        assert result.exit_code != 0, result.output
        assert "--embeddings-model" in result.output


def test_eval_refuses_an_embeddings_model_it_would_not_use(workspace: Path) -> None:
    """A model named beside `lexical` would sit in a report none of whose numbers it produced.

    First written asserting only the exit code and the flag's name in the output — and it
    PASSED before the flag existed, on Typer's own «No such option: --embeddings-model». So
    the refusal is asserted by what it says about the strategy, not by the flag being quoted.
    """
    result = runner.invoke(app, ["eval", "--strategy", "lexical", "--embeddings-model", "x/y"])
    assert result.exit_code != 0
    assert "No such option" not in result.output, result.output
    assert "--embeddings-model" in result.output and "lexical" in result.output


def _fake_eval_vectors(seen: list[str]):
    """`cli._eval_vectors` without a subprocess (§13.11): the model named is recorded, and the
    evaluation it returns runs the fake circle embedder over the workspace's own data."""
    from tests.test_knowledge_evaluation import _vectors

    def fake(cfg, model: str):
        seen.append(model)
        return _vectors(cfg.data_dir, cfg.data_dir / "eval-index", requested=model)

    return fake


def test_eval_measures_the_model_it_was_asked_for_and_writes_it_into_the_report(
    workspace: Path, monkeypatch
) -> None:
    """The flag's VALUE reaches the harness and the report file, which is what a reader opens
    to learn whose numbers these are. Seen red with the flag parsed and never passed on."""
    from xbrain import cli

    seen: list[str] = []
    monkeypatch.setattr(cli, "_eval_vectors", _fake_eval_vectors(seen))
    result = runner.invoke(
        app,
        [
            "eval",
            "--strategy",
            "vector",
            "--embeddings-model",
            "fake/cli-model",
            "--report",
            "data/eval-fake.json",
            "--json",
        ],
    )
    payload = _json_stdout(result)

    assert seen == ["fake/cli-model"]
    assert payload["strategy"] == "vector"
    assert payload["embeddings"]["model"] == "fake/cli-model"
    on_disk = json.loads((workspace / "data" / "eval-fake.json").read_text(encoding="utf-8"))
    assert on_disk["embeddings"]["model"] == "fake/cli-model"


def test_eval_fusion_sweep_needs_hybrid_and_writes_its_own_report(
    workspace: Path, monkeypatch
) -> None:
    """`--sweep-fusion` sweeps the constants of the FUSION, so it has nothing to sweep under
    `vector`, which fuses one channel; refused by name. Under `hybrid` it writes its own pair
    of files, never the ordinary report's."""
    from xbrain import cli

    monkeypatch.setattr(cli, "_eval_vectors", _fake_eval_vectors([]))
    refused = runner.invoke(
        app,
        ["eval", "--strategy", "vector", "--embeddings-model", "m/x", "--sweep-fusion", "rrf_k=10"],
    )
    assert refused.exit_code != 0 and "hybrid" in refused.output

    result = runner.invoke(
        app,
        [
            "eval",
            "--strategy",
            "hybrid",
            "--embeddings-model",
            "m/x",
            "--sweep-fusion",
            "rrf_k=10,60",
            "--json",
        ],
    )
    payload = _json_stdout(result)
    assert {row["rrf_k"] for row in payload["rows"]} == {10, 60}
    assert (workspace / "data" / "eval-fusion-sweep.json").exists()
    assert (workspace / "data" / "eval-fusion-sweep.md").exists()
    assert not (workspace / "data" / "eval-report.json").exists()


def test_eval_graph_sweep_needs_hybrid_graph_and_writes_its_own_report(workspace: Path) -> None:
    """Plan 04 §1.3 at the command a reader runs to re-derive the applied threshold: the table
    is `hybrid_graph`'s, so the flag is refused under any other strategy and beside the other
    sweeps; it writes its own pair of files and builds its own index, never `data/index/`."""
    grid = "min_shared_items=2,3 min_weight=0.0"
    for argv, needle in (
        (["eval", "--sweep-graph", grid], "hybrid_graph"),
        (
            ["eval", "--strategy", "hybrid_graph", "--sweep-graph", grid, "--sweep-chunker", "x=1"],
            "--sweep-chunker",
        ),
        (
            ["eval", "--strategy", "hybrid_graph", "--sweep-graph", grid, "--min-recall", "0.5"],
            "--min-recall",
        ),
        (
            ["eval", "--strategy", "hybrid_graph", "--sweep-graph", grid, "--k", "5", "--k", "10"],
            "--k",
        ),
    ):
        refused = runner.invoke(app, argv)
        assert refused.exit_code != 0 and needle in refused.output, (argv, refused.output)
    assert not (workspace / "data" / "eval-graph-sweep.json").exists()

    result = runner.invoke(
        app, ["eval", "--strategy", "hybrid_graph", "--sweep-graph", grid, "--json"]
    )

    payload = _json_stdout(result)
    assert {(row["min_shared_items"], row["min_weight"]) for row in payload["rows"]} >= {
        (2, 0.0),
        (3, 0.0),
    }
    assert payload["base"]["requested_strategy"] == "hybrid"
    on_disk = json.loads((workspace / "data" / "eval-graph-sweep.json").read_text(encoding="utf-8"))
    assert on_disk["verdict"] == payload["verdict"]
    markdown = (workspace / "data" / "eval-graph-sweep.md").read_text(encoding="utf-8")
    assert markdown.splitlines()[-1] == payload["verdict"]
    assert (workspace / "data" / "eval-index" / "graph-sweep").is_dir()
    assert not (workspace / "data" / "index").exists()
    assert not (workspace / "data" / "eval-report.json").exists()


def test_eval_sweep_publishes_the_table_and_writes_both_reports(workspace: Path) -> None:
    """Plan 02 §7 at the command that has to exist for the number to be re-derivable: the
    delivery matrix's row 02.13 lists `M cli.py (--sweep-chunker)` and its outcome is *«el
    baseline léxico está medido y publicado»*. §15.12's signed-measurement half is exempt
    from CI, the INSTRUMENT is not.

    Driven through the real CLI, because the flag is what a reader runs to re-derive `800/0`.
    Seen red before the wiring: `Error: No such option: --sweep-chunker`.
    """
    result = runner.invoke(app, ["eval", "--sweep-chunker", "target=800,1600 overlap=0", "--json"])
    payload = _json_stdout(result)

    assert [row["target"] for row in payload["rows"]] != []
    assert {row["target"] for row in payload["rows"]} == {800, 1600}
    assert all("recall@1" in row and "chunks" in row for row in payload["rows"])
    # The ordinary report's path is NOT reused: a sweep and an evaluation are two documents.
    assert (workspace / "data" / "eval-sweep.json").exists()
    assert (workspace / "data" / "eval-sweep.md").exists()
    assert not (workspace / "data" / "eval-report.json").exists()


def test_the_sweep_ARTEFACTS_on_disk_name_the_retriever_that_ranked_them(
    workspace: Path, monkeypatch
) -> None:
    """End of the chain: the two FILES a reader opens, not the in-process report object.

    `data/eval-sweep.{json,md}` is the artefact Plan 03 has to beat, and it named no retriever
    anywhere — `xbrain eval --strategy vector --sweep-chunker …` wrote a ranked table produced
    entirely by bm25, with `strategy` absent from the JSON and absent from the markdown, while
    the SAME command without `--sweep-chunker` headed its report «`lexical` · solicitada
    `vector`, sin backend». One command, two branches, one of them silent about its instrument
    (F-2). The report object is asserted in `tests/test_knowledge_evaluation.py`; this asserts
    the bytes, because a field that never reaches the file is a field nobody reads.

    The premise is pinned, not inherited: `vector` is the example of a declared-but-
    unimplemented strategy and reading that from production would expire when Plan 03 lands.

    Seen red before the fix: `"strategy" not in payload`, and the written markdown contained
    the word `vector` nowhere.
    """
    monkeypatch.setattr(contracts, "IMPLEMENTED_STRATEGIES", frozenset({"lexical"}))
    result = runner.invoke(
        app, ["eval", "--strategy", "vector", "--sweep-chunker", "target=800,1600", "--json"]
    )
    payload = _json_stdout(result)

    assert payload["strategy"] == "lexical", "what ran"
    assert payload["requested_strategy"] == "vector", "what was asked for"
    assert payload["degraded"] == ["vector_not_implemented"]

    on_disk = json.loads((workspace / "data" / "eval-sweep.json").read_text(encoding="utf-8"))
    assert on_disk["strategy"] == "lexical"
    assert on_disk["requested_strategy"] == "vector"

    markdown = (workspace / "data" / "eval-sweep.md").read_text(encoding="utf-8")
    assert markdown.splitlines()[0].startswith("Recuperador: `lexical`")
    assert "vector_not_implemented" in markdown.splitlines()[0]


def test_eval_sweep_honours_and_publishes_the_limit(workspace: Path) -> None:
    """`xbrain eval --limit 150 --sweep-chunker …` produced a report byte-identical to
    `--limit 10` on the snapshot's real corpus, because `_run_sweep` never passed the option
    the command advertised. The report carries the depth it ran at.
    """
    payload = _json_stdout(
        runner.invoke(app, ["eval", "--limit", "150", "--sweep-chunker", "target=800", "--json"])
    )
    assert payload["limit"] == 150
    default = _json_stdout(runner.invoke(app, ["eval", "--sweep-chunker", "target=800", "--json"]))
    assert default["limit"] == 10


def test_eval_sweep_ranks_at_the_k_the_command_was_given(workspace: Path) -> None:
    """F2-1 of the final gate on #177: `--k` never reached the sweep. `_run_sweep` took no `k`
    and called `sweep_chunker` without one, so the ranking always happened at the default 10:

        eval --k 5 --sweep-chunker "target=800,1600" --json  ->  report k = 10
        eval       --sweep-chunker "target=800,1600" --json  ->  report k = 10
        payloads byte-identical: True

    It is the same defect, in the same function, with the same byte-identical tell as the
    `--limit` one the sweep commit says it fixed — and every `k=` in the sweep's own tests was
    the default, so no test at any layer could have caught it. `sweep_chunker(k=…)` was always
    correct; only the wiring was missing.

    The DEFAULT is asserted as a control, so this cannot pass because 5 happened to be what
    the command does anyway.
    """
    payload = _json_stdout(
        runner.invoke(app, ["eval", "--k", "5", "--sweep-chunker", "target=800,1600", "--json"])
    )
    assert payload["k"] == 5, payload
    assert all("recall@5" in row for row in payload["rows"]), payload["rows"]

    default = _json_stdout(
        runner.invoke(app, ["eval", "--sweep-chunker", "target=800,1600", "--json"])
    )
    assert default["k"] == 10, "the control moved: 5 was not distinguishable from the default"


def test_eval_sweep_refuses_more_than_one_k_instead_of_picking_one(workspace: Path) -> None:
    """`--k` is repeatable on the ordinary path — a report carries several columns — and the
    sweep ranks by exactly ONE `recall@k`. Taking `max(k)` would discard the others in silence,
    which is the defect this PR exists to close, one layer up. So the combination is REFUSED,
    by name and with a non-zero exit.
    """
    result = runner.invoke(
        app, ["eval", "--k", "1", "--k", "5", "--sweep-chunker", "target=800,1600"]
    )

    assert result.exit_code != 0, result.output
    assert "--k" in result.output and "--sweep-chunker" in result.output, result.output
    # And ONE value is honoured rather than refused along with the rest.
    assert (
        _json_stdout(
            runner.invoke(app, ["eval", "--k", "5", "--sweep-chunker", "target=800", "--json"])
        )["k"]
        == 5
    )


def test_eval_sweep_refuses_a_threshold_it_cannot_apply(workspace: Path) -> None:
    """F2-2 of the final gate on #177: `--min-recall` was accepted on the sweep path and
    silently ignored, because `if sweep_chunker: _run_sweep(...); return` happens before the
    threshold is ever used. Measured there, with the ordinary path as the control:

        | threshold        | ordinary `eval` | `eval --sweep-chunker` |
        | --min-recall 1.1 | exit 1          | exit 0                 |
        | --min-recall 2.0 | exit 1          | exit 0                 |

    The flag's own help promises *«si algún bucket queda por debajo, el comando falla»*, and
    on the sweep path it judged nothing and exited 0 — the fail-open CLAUDE.md already records
    for this exact flag. The threshold judges the BUCKETS of one evaluation; a sweep publishes
    a table of combinations and has no bucket to compare against, so inventing a meaning for
    it here would be a gate whose green a reader would misread. It is refused instead, which
    is the one reading that cannot mislead.

    Both halves are asserted: the sweep refuses, and the ordinary path still fails — so the
    flag's promise is kept somewhere and the refusal is a narrowing, not a deletion.
    """
    swept = runner.invoke(
        app, ["eval", "--min-recall", "2.0", "--sweep-chunker", "target=800,1600"]
    )

    assert swept.exit_code != 0, "the sweep accepted a threshold it never applied:\n" + swept.output
    assert "--min-recall" in swept.output and "--sweep-chunker" in swept.output, swept.output

    plain = runner.invoke(app, ["eval", "--min-recall", "2.0"])
    assert plain.exit_code != 0, plain.output


def test_eval_sweep_without_a_winner_is_not_reported_as_a_success(
    workspace: Path, unscorable_strategy: str
) -> None:
    """F2-3 at the surface a caller reads. A sweep whose every combination was unscorable
    printed «PLANO: todas las combinaciones puntúan igual» and exited 0 — a positive claim
    about a ranking that never happened, reported as a success.

    The golden set is trimmed to FX7 and the run is pointed at an INJECTED backend that can
    push no filter at all, so every case is UNMEASURED for every combination — the same
    construction the threshold's own fail-closed test uses, and for the same reason it is an
    injection rather than `lexical`.

    The empty grid is asserted beside it because it is the same predicate — no winner — and
    it used to publish `rows: []` with exit 0 as well.
    """
    golden = yaml.safe_load((workspace / "eval" / "golden-set.yaml").read_text(encoding="utf-8"))
    golden["cases"] = [c for c in golden["cases"] if c["id"] == "FX7"]
    golden.pop("scenarios", None)
    (workspace / "eval" / "golden-set.yaml").write_text(
        yaml.safe_dump(golden, allow_unicode=True), encoding="utf-8"
    )

    unmeasured = runner.invoke(
        app,
        ["eval", "--sweep-chunker", "target=800,1600", "--strategy", unscorable_strategy],
    )

    assert unmeasured.exit_code != 0, "a sweep that scored nothing exited 0:\n" + unmeasured.output
    assert "SIN MEDICIÓN" in unmeasured.output, unmeasured.output
    assert "PLANO" not in unmeasured.output, unmeasured.output
    # The table is still published: the run failed, the evidence is not withheld.
    payload = json.loads((workspace / "data" / "eval-sweep.json").read_text(encoding="utf-8"))
    assert payload["winner"] is None and payload["measured"] is False
    assert len(payload["rows"]) == 2

    empty = runner.invoke(app, ["eval", "--sweep-chunker", "target="])
    assert empty.exit_code != 0, empty.output
    assert "SIN COMBINACIONES" in empty.output, empty.output


def test_eval_sweep_refuses_an_unknown_axis_through_the_command(workspace: Path) -> None:
    """A typo that swept nothing would publish the DEFAULT's numbers under the name of a
    sweep. The refusal has to reach the CLI's exit code, not only `parse_sweep`.
    """
    result = runner.invoke(app, ["eval", "--sweep-chunker", "targt=800"])

    assert result.exit_code != 0, result.output
    assert "desconocido" in result.output, result.output


def test_inspect_chunks_an_article_on_its_block_boundaries(workspace: Path) -> None:
    """m8, the other production caller: `knowledge inspect --chunks`.

    The same defect and the same blind spot as the harness — measured, deleting
    `blocks_by_surface_id=article_block_texts(item)` from `cli._inspect_item` left the full
    suite green. Driven through the real command, because the CLI is what a consumer runs
    and the argument lives in the CLI, not in the chunker.

    The article fixture is IMPORTED, not copied: two literals that must agree about where a
    block ends are two definitions that will drift (rule 5).
    """
    from tests.test_knowledge_chunking import _article_item, _discriminating_blocks

    blocks = _discriminating_blocks()
    item = _article_item(blocks)
    store_path = workspace / "data" / "items.json"
    store = json.loads(store_path.read_text(encoding="utf-8"))
    store[item.id] = item.model_dump(mode="json")
    store_path.write_text(json.dumps(store, indent=2, default=str), encoding="utf-8")

    payload = _json_stdout(
        runner.invoke(app, ["knowledge", "inspect", item.id, "--chunks", "--json"])
    )
    article = [c for c in payload["chunks"] if c["surface_type"] == "x_article"]

    assert article, "the command emitted no article chunk at all"
    edges, cursor = {0}, 0
    for block in blocks:
        cursor += len(block.text)
        edges.add(cursor)
    for chunk in article:
        assert chunk["char_start"] in edges, (
            f"`inspect --chunks` cut inside a block at {chunk['char_start']}: the command is "
            "not handing the chunker the block boundaries"
        )
        assert chunk["char_end"] in edges


# ---------------------------------------------------------------------------
# Plan 03.6 — `search --strategy` when the vector channel cannot run (§5, §13)
# ---------------------------------------------------------------------------

SEARCH_QUERY = "Quillfeather"


def _configure_embeddings(workspace: Path, command: str) -> None:
    config = workspace / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + f"[embeddings]\ncommand = {json.dumps(command)}\n",
        encoding="utf-8",
    )


def _build_index_with_a_plane(workspace: Path) -> None:
    """The index `xbrain index build --embeddings` would leave, embedded by a hash fake.

    Built through the CLI's OWN loaders (`_index_inputs`, `_index_options`), so the manifest it
    seals is the one the query door compares against — not a second description of it.
    """
    import math

    from xbrain import cli
    from xbrain.config import load_config
    from xbrain.knowledge import index_build
    from xbrain.knowledge.vector_index import VectorSpec

    def embed(texts):  # noqa: ANN001, ANN202 - the `Embedder` shape
        vectors = []
        for text in texts:
            angle = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
            vectors.append((math.cos(angle * 2 * math.pi), math.sin(angle * 2 * math.pi)))
        return vectors

    cfg = load_config(workspace)
    spec = VectorSpec(
        model="fake-model", dimension=2, normalized=True, query_prefix="", passage_prefix=""
    )
    index_build.build(
        cfg.index_dir,
        cli._index_inputs(cfg),
        options=cli._index_options(cfg),
        vectors=index_build.VectorBuild(spec=spec, embed=embed),
    )


def _vector_matches(payload: dict) -> list[dict]:
    return [
        match
        for result in payload["results"]
        for match in result["matches"]
        if "vector" in match["matched_by"] or match["vector_rank"] is not None
    ]


def test_search_hybrid_without_an_embeddings_section_is_lexical_and_says_so(
    workspace: Path,
) -> None:
    """Criterion §13.2 through the command: no `[embeddings]` at all, and everything still works.

    Before 03.6 the command never bound an embedder, so the answer said `hybrid_not_implemented`
    — a false statement about the build that no setting could ever make true.
    """
    assert runner.invoke(app, ["index", "build"]).exit_code == 0

    payload = _json_stdout(
        runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "hybrid", "--json"])
    )

    assert payload["strategy"] == "lexical"
    assert "embeddings_not_configured" in payload["index"]["degraded"]
    assert "hybrid_not_implemented" not in payload["index"]["degraded"]
    assert payload["results"], "lexical stays operational"
    assert not _vector_matches(payload)


def test_search_vector_without_vectors_is_an_error_and_prints_no_response(
    workspace: Path,
) -> None:
    """Criterion §13.3: `--strategy vector` was ASKED for; answering lexically would hide it."""
    assert runner.invoke(app, ["index", "build"]).exit_code == 0

    result = runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "vector", "--json"])

    assert result.exit_code == 1, result.output
    assert result.stdout == "", "an error must not also print a response a consumer could parse"
    assert "xbrain index build --embeddings" in result.output


def test_search_hybrid_with_the_embedder_binary_missing_is_lexical_and_declares_it(
    workspace: Path,
) -> None:
    """Criterion §13.4 end to end: a configured command whose binary is gone, a plane on disk.

    The command binds the REAL adapter, the REAL subprocess boundary refuses to start, and the
    response is `lexical` with `embedder_unavailable` and not one `vector` in any `matched_by`.
    """
    _configure_embeddings(workspace, str(workspace / "bin" / "xbrain-embed-gone"))
    _build_index_with_a_plane(workspace)

    payload = _json_stdout(
        runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "hybrid", "--json"])
    )

    assert payload["strategy"] == "lexical"
    assert payload["index"]["degraded"] == ["embedder_unavailable"]
    assert payload["results"]
    assert not _vector_matches(payload)


def _index_lines(output: str) -> list[str]:
    """The human view's warnings: the lines between the header and the first result (spec §7.6)."""
    header, _, rest = output.partition("\n")
    assert "estrategia" in header, f"premise: the first line is the search header — {header!r}"
    return rest.split("\n\n", 1)[0].splitlines()


@pytest.mark.parametrize(
    "binary, flag",
    [(None, "embeddings_not_configured"), ("xbrain-embed-gone", "embedder_unavailable")],
)
def test_search_hybrid_human_view_names_what_to_fix_not_a_bare_code(
    workspace: Path, binary: str | None, flag: str
) -> None:
    """A degraded `hybrid` read WITHOUT `--json` says what failed and which setting fixes it.

    The JSON carries the code; the human view is the surface a reader actually reads, and a bare
    `⚠ embedder_unavailable` is a flag they have to look up — i.e. one they will ignore. So the
    warning line, above the first result, must name `[embeddings].command`, and must say these
    results are lexical, not hybrid.
    """
    if binary is not None:
        _configure_embeddings(workspace, str(workspace / "bin" / binary))
    _build_index_with_a_plane(workspace)

    result = runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "hybrid"])

    assert result.exit_code == 0, result.output
    warnings = _index_lines(result.stdout)
    assert f"⚠ {flag}" not in warnings
    naming = [line for line in warnings if "`[embeddings].command`" in line]
    assert len(naming) == 1, warnings
    assert "`lexical`" in naming[0] and "`hybrid`" in naming[0], naming[0]


def _a_working_embedder(workspace: Path, monkeypatch) -> list[list[str]]:
    """`[embeddings].command` configured and answering in the plane's own model — nothing runs.

    `subprocess.run` is replaced, as in the build test below (`xbrain.embeddings` resolves it at
    call time), and every call is recorded, so a test can tell a query that reached the backend
    from one that never did.
    """
    import subprocess

    from xbrain.embeddings import SCHEMA_VERSION

    calls: list[list[str]] = []

    def run(argv, **kwargs):  # noqa: ANN001, ANN003, ANN202 - a subprocess.run stand-in
        calls.append(list(argv))
        texts = json.loads(kwargs["input"])["texts"]
        body = {
            "schema_version": SCHEMA_VERSION,
            "model": "fake-model",
            "dimension": 2,
            "normalized": True,
            "vectors": [[1.0, 0.0] for _ in texts],
        }
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(body), stderr="")

    _configure_embeddings(workspace, "xbrain-embed")
    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_search_without_a_strategy_is_the_lexical_request_even_where_hybrid_would_run(
    workspace: Path, monkeypatch
) -> None:
    """Audit A2: the COMMAND's default is `lexical`, bound where it is published.

    `test_hybrid_graph_existe_es_desactivable_y_el_default_no_cambia` reads the default off
    `search_service.search`, but the command declares its OWN (`typer.Option("lexical", …)`),
    and changing it to `"hybrid"` kept the whole suite green (audit M02). So this runs where a
    promoted default would SHOW — a plane on disk and an embedder that answers, which the control
    proves by really running `hybrid` — and omitting the flag must be the very same request as
    `--strategy lexical`: the same envelope, nothing degraded, the embedder never called.
    `strategy == "lexical"` alone would not bind it: an unconfigured `hybrid` default answers
    `lexical` too, and confesses only in `degraded`.
    """
    _build_index_with_a_plane(workspace)
    calls = _a_working_embedder(workspace, monkeypatch)
    hybrid = _json_stdout(
        runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "hybrid", "--json"])
    )
    assert hybrid["strategy"] == "hybrid", "premise: a non-lexical default would run here"
    calls.clear()

    omitted = _json_stdout(runner.invoke(app, ["search", SEARCH_QUERY, "--json"]))
    explicit = _json_stdout(
        runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "lexical", "--json"])
    )

    assert omitted == explicit
    assert omitted["strategy"] == "lexical"
    assert omitted["index"]["degraded"] == []
    assert omitted["results"]
    assert calls == []


def test_search_hybrid_graph_through_the_command_leaves_the_graph_switched_off(
    workspace: Path,
) -> None:
    """Audit A3: the command asks for `hybrid_graph` with the switch OFF — never its own `True`.

    The service binds `GRAPH_ENABLED_BY_DEFAULT` (audit M01, red), but the command is a second
    door: passing `graph_enabled=True` from `cli.py` kept the suite green (audit M03), because no
    test ever ran `xbrain search --strategy hybrid_graph`. Here it runs, and its envelope must be
    the one the service serves with the switch explicitly OFF — `lexical`, declaring
    `hybrid_graph_not_implemented`, and no match reached through `graph`. The control proves the
    comparison can tell the two apart: switched ON over the same index, the service answers
    `hybrid_graph`.
    """
    from xbrain import cli
    from xbrain.config import load_config
    from xbrain.knowledge.search_service import search

    assert runner.invoke(app, ["index", "build"]).exit_code == 0
    cfg = load_config(workspace)
    context = cli._query_context(cfg, cli._index_inputs(cfg))
    switched_on = search(SEARCH_QUERY, context, strategy="hybrid_graph", graph_enabled=True)
    assert switched_on.strategy == "hybrid_graph", "premise: the graph runs when switched on"

    payload = _json_stdout(
        runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "hybrid_graph", "--json"])
    )

    switched_off = search(SEARCH_QUERY, context, strategy="hybrid_graph", graph_enabled=False)
    assert payload == switched_off.model_dump(mode="json")
    assert payload["strategy"] == "lexical"
    assert "hybrid_graph_not_implemented" in payload["index"]["degraded"]
    assert payload["results"]
    assert not [m for r in payload["results"] for m in r["matches"] if "graph" in m["matched_by"]]


def test_search_vector_with_a_filter_through_the_command_is_lexical_and_never_says_vector(
    workspace: Path, monkeypatch
) -> None:
    """Audit A1 through the command: a filtered `--strategy vector` answers `lexical`.

    `test_a_filtered_vector_request_is_answered_lexically_and_never_says_vector` binds the
    service's filter branch; this binds the door a reader actually types. The plane is on disk
    and the embedder answers — the control proves the vector channel RUNS here without a filter —
    so under `--source bookmark` the only reason it does not run is the filter, and the response
    must say so: `lexical`, `degraded == ["vector_filters_unsupported"]`, the lexical page for the
    same filter, no match claiming `vector`, and the embedder never called. Naming `vector` in
    that branch (audit M06) kept the whole suite green.
    """
    _build_index_with_a_plane(workspace)
    calls = _a_working_embedder(workspace, monkeypatch)
    unfiltered = _json_stdout(
        runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", "vector", "--json"])
    )
    assert unfiltered["strategy"] == "vector", "premise: the vector channel runs without a filter"
    calls.clear()

    filtered = _json_stdout(
        runner.invoke(
            app, ["search", SEARCH_QUERY, "--strategy", "vector", "--source", "bookmark", "--json"]
        )
    )
    lexical = _json_stdout(
        runner.invoke(
            app, ["search", SEARCH_QUERY, "--strategy", "lexical", "--source", "bookmark", "--json"]
        )
    )

    assert filtered["strategy"] == "lexical"
    assert filtered["index"]["degraded"] == ["vector_filters_unsupported"]
    assert filtered["filters"]["source"] == "bookmark"
    assert filtered["results"], "lexical stays operational"
    assert filtered["results"] == lexical["results"]
    assert not _vector_matches(filtered)
    assert calls == []


def test_index_build_embeddings_refuses_a_batch_from_another_model_than_the_probe(
    workspace: Path, monkeypatch
) -> None:
    """Plan 03 §13.4 at build time: the probe fixes the model; a later batch from ANOTHER model
    of the same dimension is refused, not appended into a matrix sealed under the first name.

    `subprocess.run` is replaced (resolved at call time by `xbrain.embeddings`), so nothing runs:
    the probe answers `model-a`, every later batch `model-b`, all 2-dimensional.
    """
    import subprocess

    from xbrain.embeddings import SCHEMA_VERSION

    answered: list[str] = []

    def run(argv, **kwargs):  # noqa: ANN001, ANN003, ANN202 - a subprocess.run stand-in
        texts = json.loads(kwargs["input"])["texts"]
        model = "model-a" if not answered else "model-b"
        answered.append(model)
        body = {
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "dimension": 2,
            "normalized": True,
            "vectors": [[1.0, 0.0] if model == "model-a" else [0.0, 1.0] for _ in texts],
        }
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(body), stderr="")

    _configure_embeddings(workspace, "xbrain-embed")
    monkeypatch.setattr(subprocess, "run", run)

    result = runner.invoke(app, ["index", "build", "--embeddings"])

    assert answered[:2] == ["model-a", "model-b"], "premise: a second model did answer"
    assert result.exit_code == 1, result.output
    assert "model-b" in result.output and "model-a" in result.output
    assert "Traceback" not in result.output
    assert not _manifest_exists(workspace)


@pytest.mark.parametrize("strategy", ["vector", "hybrid"])
def test_search_needing_vectors_without_the_extra_names_the_install_command(
    workspace: Path, monkeypatch, strategy: str
) -> None:
    """Criterion §13.12, second half: never a raw `ImportError`, always the one install command.

    The plane is on disk and the command is configured, so the ONLY thing missing is `numpy` —
    blocked in `sys.modules`, which makes its import raise exactly as an absent package does.
    """
    import sys

    _configure_embeddings(workspace, str(workspace / "bin" / "xbrain-embed"))
    _build_index_with_a_plane(workspace)
    monkeypatch.setitem(sys.modules, "numpy", None)

    result = runner.invoke(app, ["search", SEARCH_QUERY, "--strategy", strategy, "--json"])

    assert result.exit_code == 1, result.output
    assert "uv pip install -e '.[embeddings]'" in result.output
    assert "xbrain[" not in result.output, "names a package this repo does not publish"
    assert not isinstance(result.exception, ImportError)
    assert "Traceback" not in result.output


def _manifest_exists(workspace: Path) -> bool:
    from xbrain.config import load_config
    from xbrain.knowledge.index_build import manifest_path

    return manifest_path(load_config(workspace).index_dir).exists()


def test_index_build_embeddings_force_without_numpy_refuses_and_leaves_the_index_standing(
    workspace: Path, monkeypatch
) -> None:
    """Backlog #6 of PR #206, reproduced in Plan 06.3: the missing extra used to cost the index.

    `build` discarded manifest, base and plane first and imported `numpy` only when the matrix
    was written, so the refusal arrived over an index that was already gone and even a lexical
    `search` answered «no hay manifest». The refusal must come BEFORE anything is deleted.
    """
    import sys

    assert runner.invoke(app, ["index", "build"]).exit_code == 0
    _a_working_embedder(workspace, monkeypatch)
    monkeypatch.setitem(sys.modules, "numpy", None)

    result = runner.invoke(app, ["index", "build", "--embeddings", "--force"])

    assert result.exit_code == 1, result.output
    assert "numpy" in result.output
    assert _manifest_exists(workspace), "the good index was discarded before the refusal"
    searched = runner.invoke(app, ["search", SEARCH_QUERY, "--json"])
    assert searched.exit_code == 0, searched.output
    assert _json_stdout(searched)["results"], "lexical search no longer answers"


def test_index_build_embeddings_without_a_command_names_the_setting_and_builds_nothing(
    workspace: Path,
) -> None:
    """`--embeddings` with no `[embeddings].command`: a refusal naming the setting, not a lexical
    index quietly sealed under a flag that asked for vectors."""
    result = runner.invoke(app, ["index", "build", "--embeddings"])

    assert result.exit_code == 1, result.output
    assert "[embeddings].command" in result.output
    assert not _manifest_exists(workspace)


@pytest.mark.parametrize("executable", [False, None])
def test_index_build_embeddings_with_the_binary_absent_is_embedder_not_found(
    workspace: Path, executable: bool | None
) -> None:
    """Plan 03 §5, row 2, through the command: missing (`None`) or not executable (`False`).

    The REAL subprocess boundary refuses to start the process; the command exits 1 naming the
    setting to fix, and no manifest is sealed that a query door could trust.
    """
    binary = workspace / "bin" / "xbrain-embed"
    if executable is not None:
        binary.parent.mkdir()
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o644)
    _configure_embeddings(workspace, str(binary))

    result = runner.invoke(app, ["index", "build", "--embeddings"])

    assert result.exit_code == 1, result.output
    assert "[embeddings].command" in result.output
    assert "Traceback" not in result.output
    assert not _manifest_exists(workspace)


# ---------------------------------------------------------------------------
# Plan 04.3 — graph-expand
# ---------------------------------------------------------------------------


def test_graph_expand_returns_nodes_edges_and_explicit_paths(workspace: Path) -> None:
    """`graph-expand --item` over a REAL built index: every reached node carries its path.

    k02 is assigned `agent-evaluation` (primary) and k08 shares it, so two hops from `item:k02`
    must reach `item:k08` THROUGH the topic — a path that names both hops, not a bare neighbour.
    """
    assert runner.invoke(app, ["index", "build"]).exit_code == 0

    payload = _json_stdout(
        runner.invoke(app, ["graph-expand", "--item", "k02", "--max-hops", "2", "--json"])
    )

    assert payload["seeds"] == ["item:k02"]
    node_ids = {node["node_id"] for node in payload["nodes"]}
    assert {"item:k02", "topic:agent-evaluation", "item:k08"} <= node_ids
    assert payload["edges"], "a seed with topic assignments has incident edges"
    for path in payload["paths"]:
        assert path["nodes"][0] == "item:k02"
        assert len(path["nodes"]) == len(path["edges"]) + 1
        for (a, b), edge in zip(zip(path["nodes"], path["nodes"][1:]), path["edges"]):
            assert {edge["source"], edge["target"]} == {a, b}
    to_k08 = next(p for p in payload["paths"] if p["nodes"][-1] == "item:k08")
    assert to_k08["nodes"] == ["item:k02", "topic:agent-evaluation", "item:k08"]


def test_graph_expand_response_carries_semantics_and_disclaimer_key(workspace: Path) -> None:
    """Plan 04 §11.6 on a REAL command response, not on a model built in the test.

    The values are read off the CONTRACT's defaults, never retyped here, and the key must name
    the i18n sentence the human render prints — so dropping either field from what the command
    serialises, or printing a sentence of its own, goes red.
    """
    from xbrain.i18n import strings_for

    assert runner.invoke(app, ["index", "build"]).exit_code == 0
    fields = contracts.GraphExpansionResponse.model_fields

    payload = _json_stdout(runner.invoke(app, ["graph-expand", "--item", "k02", "--json"]))

    assert payload["semantics"] == fields["semantics"].default
    assert payload["disclaimer_key"] == fields["disclaimer_key"].default
    sentence = getattr(strings_for("English"), payload["disclaimer_key"])
    human = runner.invoke(app, ["graph-expand", "--item", "k02"])
    assert human.exit_code == 0, human.output
    assert human.stdout.splitlines()[0] == sentence
