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
    payload = _json_stdout(runner.invoke(app, ["knowledge", "inspect", "k08", "--json"]))
    assert payload["item"]["item_id"] == "k08"
    assert payload["schema_version"] == "1"


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


def test_eval_with_a_threshold_fails_when_nothing_could_be_measured(workspace: Path) -> None:
    """M2: a gate that compared the threshold against NOTHING must not report PASS.

    `_failures` skips every bucket with no coverage and every metric carrying the sentinel —
    correctly, because naming one would be the fabricated zero of spec §8.6.8. But `passed`
    is literally "no failures", so when the threshold reaches no bucket at all the strictest
    gate that exists comes out green over zero comparisons. That is the FAIL-OPEN cell of
    CLAUDE.md rule 11, inside the command whose acceptance criterion 10 is "the evaluation
    can fail".

    Driven through the real CLI, because the exit code is the only surface a caller reads:
    the golden set is trimmed to FX7, whose `source` filter the lexical baseline cannot
    push into `WHERE`, so the case is UNMEASURED and every bucket ends up empty.
    """
    golden = yaml.safe_load((workspace / "eval" / "golden-set.yaml").read_text(encoding="utf-8"))
    golden["cases"] = [c for c in golden["cases"] if c["id"] == "FX7"]
    golden.pop("scenarios", None)
    (workspace / "eval" / "golden-set.yaml").write_text(
        yaml.safe_dump(golden, allow_unicode=True), encoding="utf-8"
    )

    result = runner.invoke(app, ["eval", "--min-recall", "1.0"])

    assert result.exit_code != 0, (
        "a threshold of 1.0 passed having scored zero cases:\n" + result.output
    )
    assert "0" in result.output and "medid" in result.output, result.output


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
