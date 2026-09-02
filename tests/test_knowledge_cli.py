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
import shlex
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

    THE DRIVER CHANGED IN PLAN 02, and the reason is worth recording. It used to trim the
    golden set to FX7, whose `source` filter the lexical baseline could not push into
    `WHERE` — so the case was unmeasured and every bucket ended up empty. Plan 02 gave the
    baseline all eight filters, so FX7 now scores and that driver stopped exercising anything
    (rule 1: a test green for a reason unrelated to its name). The driver is now a golden set
    with every case ARCHIVED AS A SCENARIO, which is a shape the file can legitimately take —
    Plan 01 §4.4 archives exactly this way — and which reaches the same zero comparisons.

    Driven through the real CLI, because the exit code is the only surface a caller reads.
    """
    golden = yaml.safe_load((workspace / "eval" / "golden-set.yaml").read_text(encoding="utf-8"))
    golden["scenarios"] = [
        {
            "id": case["id"],
            "question": case["query"],
            "provenance": case["provenance"],
            "reason": "archivado para este test: sin verdad de terreno enumerada",
        }
        for case in golden["cases"]
    ]
    golden["cases"] = []
    (workspace / "eval" / "golden-set.yaml").write_text(
        yaml.safe_dump(golden, allow_unicode=True), encoding="utf-8"
    )

    result = runner.invoke(app, ["eval", "--min-recall", "1.0"])

    assert result.exit_code != 0, (
        "a threshold of 1.0 passed having scored zero cases:\n" + result.output
    )
    assert "no se comparó contra nada" in result.output, (
        "the gate must SAY it measured nothing, not merely exit 1"
    )


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
# 26, 28 — `index`, `search`, `get` at the CLI (Plan 02 §6)
# ---------------------------------------------------------------------------


def _store_hash(workspace: Path) -> str:
    """The hash of the store file, for the before/after comparison of step 28."""
    return hashlib.sha256((workspace / "data" / "items.json").read_bytes()).hexdigest()


def test_index_build_then_search_then_get_end_to_end(workspace: Path) -> None:
    """The chain the plan's exit gate is about, run as a user would run it.

    Three commands in sequence against a real workspace, because each of them is green on its
    own and the interesting failures are at the seams — a config field the CLI does not
    thread, an index directory resolved differently by two commands, a store loaded twice
    with different vocabularies. CLAUDE.md rule 3: the judge must EXECUTE.
    """
    assert runner.invoke(app, ["index", "build"]).exit_code == 0

    found = _json_stdout(runner.invoke(app, ["search", "Quillfeather", "--json"]))
    assert found["schema_version"] == "1" and found["strategy"] == "lexical"
    assert found["results"], "the built index answered nothing"
    item_id = found["results"][0]["item_id"]

    bundle = _json_stdout(runner.invoke(app, ["get", item_id, "--json"]))
    assert bundle["item"]["item_id"] == item_id


def test_every_json_path_writes_only_json_to_stdout(workspace: Path) -> None:
    """Step 26 / spec §3.7.9: `--json` never mixes human diagnostics into stdout.

    Parsed as a WHOLE document per command, so one stray `print` fails here instead of
    failing a consumer's parser weeks later, far from the cause. Every command of the plan is
    covered, because the one that regresses will be whichever is not.
    """
    runner.invoke(app, ["index", "build"])
    for argv in (
        ["index", "status", "--json"],
        ["index", "update", "--json"],
        ["search", "Quillfeather", "--json"],
        ["get", "k03", "--json"],
        ["get", "k03", "--surface", "external_article", "--json"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"{argv}: {result.output}"
        json.loads(result.stdout)


def test_no_command_of_this_plan_writes_to_the_store(workspace: Path) -> None:
    """Step 28 / acceptance 13: a hash of `items.json` before and after every command.

    A claim about the FILE, not about the code. "We never call `save_store`" is what someone
    believes; an unchanged sha256 is what happened. `vocab.yaml` and `topics.json` are
    included for the same reason.
    """
    before = {
        name: hashlib.sha256((workspace / "data" / name).read_bytes()).hexdigest()
        for name in ("items.json", "topics.json", "vocab.yaml")
    }
    for argv in (
        ["index", "build"],
        ["index", "update"],
        ["index", "status"],
        ["search", "Quillfeather"],
        ["get", "k03", "--surface", "external_article"],
        ["index", "build", "--force", "--dry-run"],
    ):
        assert runner.invoke(app, argv).exit_code == 0, argv
    after = {
        name: hashlib.sha256((workspace / "data" / name).read_bytes()).hexdigest()
        for name in ("items.json", "topics.json", "vocab.yaml")
    }
    assert after == before


def test_index_build_takes_no_snapshot(workspace: Path) -> None:
    """Plan 02 §6: no command here is destructive, so none of them snapshots.

    `data/index/` is derived and reconstructible by definition (spec §5.6). A snapshot would
    copy a store nothing touched, and would train the reader to ignore the ones that matter.
    """
    runner.invoke(app, ["index", "build"])
    snapshots = workspace / "data" / "snapshots"
    assert not snapshots.exists() or not list(snapshots.iterdir())


def test_search_without_an_index_names_the_build_command(workspace: Path) -> None:
    """Step 30 at the CLI: an actionable message, and a non-zero exit.

    Both halves. A command that prints its own failure and exits 0 is the `gh pr checks` trap
    of CLAUDE.md rule 9 reproduced locally — the exit status is what a script reads.
    """
    result = runner.invoke(app, ["search", "Quillfeather"])
    assert result.exit_code != 0
    assert "xbrain index build" in result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "Quillfeather"],
        ["index", "status"],
        ["index", "update"],
    ],
)
def test_a_corrupt_database_names_the_rebuild_command_on_every_command(
    workspace: Path, argv: list[str]
) -> None:
    """Plan 02 §11 / spec §9.3: *base corrupta -> error accionable con `index build --force`*.

    The row was tabulated and not implemented (F-3). `_OPERATOR_ERRORS` catches `IndexError_`,
    which covers a MISSING index and an incompatible MANIFEST — but `sqlite3.DatabaseError` is
    neither, so all three commands printed a raw traceback. The manifest half had a test; the
    database half had none, and that is why nobody noticed.

    THE CORRUPTION IS REAL, not a mock: the SQLite header is overwritten in place on a
    database that was really built, so `sqlite3.connect` still succeeds — it is lazy — and the
    failure lands on the first read, which is exactly where it lands in production.

    BOTH HALVES ARE ASSERTED, text AND exit code, for CLAUDE.md rule 9's reason: a command
    that prints its own failure and exits 0 is read as success by every script above it.

    Seen red before the fix on all three parametrisations: `sqlite3.DatabaseError: file is not
    a database` escaped `_handle_cli_errors`, `result.exception` was the raw `DatabaseError`
    and the output named no command at all.
    """
    assert runner.invoke(app, ["index", "build"]).exit_code == 0
    database = workspace / "data" / "index" / "knowledge.db"
    with database.open("r+b") as handle:
        handle.write(b"this is not a sqlite database at all, not even close, really!!")

    result = runner.invoke(app, argv)

    assert result.exit_code != 0, result.output
    assert "xbrain index build --force" in result.output, result.output


@pytest.mark.parametrize(
    "argv",
    [
        ["search", "Quillfeather"],
        ["index", "status"],
        ["index", "update", "--dry-run"],
    ],
)
def test_a_corrupt_fts_structure_names_the_rebuild_command_on_every_command(
    workspace: Path, argv: list[str]
) -> None:
    """G-4 at the CLI: corruption BEYOND page 1 — an FTS5 shadow table dropped — is the
    actionable sentence on all three commands, with a non-zero exit, and no traceback.

    Measured on the real corpus before the fix: `search` exit 1 with a 68-line Rich traceback
    (`sqlite3.DatabaseError` is not in `_OPERATOR_ERRORS`), `status` exit 0 and healthy,
    `update --dry-run` exit 0. The traceback is asserted absent through `result.exception`:
    a clean exit is a `SystemExit`, a crash is the raw error.

    Seen red before the fix on all three parametrisations.
    """
    import sqlite3

    assert runner.invoke(app, ["index", "build"]).exit_code == 0
    connection = sqlite3.connect(workspace / "data" / "index" / "knowledge.db")
    connection.execute("DROP TABLE chunks_fts_data")
    connection.commit()
    connection.close()

    result = runner.invoke(app, argv)

    assert result.exit_code != 0, result.output
    assert "xbrain index build --force" in result.output, result.output
    assert isinstance(result.exception, SystemExit), repr(result.exception)


def test_mine_maps_to_own_tweet(workspace: Path) -> None:
    """Spec §7.2's shortcut, asserted through the FILTER the response echoes back.

    The response carries the filters it applied, so this checks the mapping the service
    received rather than the flag the CLI parsed.
    """
    runner.invoke(app, ["index", "build"])
    payload = _json_stdout(runner.invoke(app, ["search", "thread", "--mine", "--json"]))
    assert payload["filters"]["source"] == "own_tweet"


def test_mine_and_a_conflicting_source_are_refused(workspace: Path) -> None:
    """Two flags that mean different things must not silently pick one."""
    runner.invoke(app, ["index", "build"])
    result = runner.invoke(app, ["search", "agents", "--mine", "--source", "bookmark"])
    assert result.exit_code != 0
    assert "incompatibles" in result.output


def test_update_dry_run_over_a_deleted_database_leaves_search_closed(workspace: Path) -> None:
    """G-2 at the CLI, the exact operator sequence the gate reproduced on the real corpus.

    `knowledge.db` deleted by hand (52 MB, a natural clean-up target), `manifest.json` kept:
    `index update --dry-run` exited 1 with the right message and CREATED an empty database;
    the next `search` exited 0 with «Sin resultados» over zero rows. The instrument that says
    "let me see what would happen" must not change what happens next.

    Seen red before the fix: `knowledge.db` existed after the dry run and `search` exited 0.
    """
    assert runner.invoke(app, ["index", "build"]).exit_code == 0
    database = workspace / "data" / "index" / "knowledge.db"
    database.unlink()

    result = runner.invoke(app, ["index", "update", "--dry-run"])
    assert result.exit_code != 0, result.output
    assert "xbrain index build --force" in result.output, result.output
    assert not database.exists(), "the dry run created the database"

    result = runner.invoke(app, ["search", "Quillfeather"])
    assert result.exit_code != 0, result.output
    assert "xbrain index build --force" in result.output, result.output


def test_index_status_reports_the_store_delta(workspace: Path) -> None:
    """Step 10c at the CLI: `status --json` says HOW MANY items changed — a number.

    The clean half (`0`) is satisfied by a boolean too (G-3), so the store on disk is then
    edited on TWO items and the count asserted `== 2`. Seen red under the `int(bool(…))`
    mutation of `status` in an isolated copy: `1 == 2` fails.
    """
    runner.invoke(app, ["index", "build"])
    payload = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert payload["items_changed"] == 0 and payload["behind"] is False
    assert payload["manifest"]["chunker_version"]
    assert payload["counts"]["chunks"] > 0

    items_path = workspace / "data" / "items.json"
    raw = json.loads(items_path.read_text(encoding="utf-8"))
    for item_id in ("k02", "k03"):
        raw[item_id]["enriched"]["summary"] = f"un resumen completamente distinto para {item_id}"
    items_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    payload = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert payload["items_changed"] == 2 and payload["behind"] is True


def test_index_status_and_update_report_the_topic_rows_behind(workspace: Path) -> None:
    """H1 at the CLI: a topic move is declared by `status` and repaired by `update`.

    Editing two SUMMARIES leaves `topics_changed` at 0 — the count is about the topic plane,
    not a copy of `items_changed`; moving k02's assignment puts it at 2. Then `update`
    reports the two rows it refreshed, and `status` is clean again. The human `status` line
    carries the number too, because a count only the JSON shows is a count nobody reads.

    Seen red before the fix: `topics_changed` absent from the JSON, `status` clean after
    the move, and `update` reporting nothing about the topic plane.
    """
    runner.invoke(app, ["index", "build"])
    items_path = workspace / "data" / "items.json"
    raw = json.loads(items_path.read_text(encoding="utf-8"))
    for item_id in ("k03", "k04"):
        raw[item_id]["enriched"]["summary"] = f"un resumen completamente distinto para {item_id}"
    items_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    payload = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert payload["items_changed"] == 2 and payload["topics_changed"] == 0

    raw["k02"]["enriched"]["primary_topic"] = "ai-policy"
    raw["k02"]["enriched"]["topics"] = ["ai-policy"]
    items_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    payload = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert payload["items_changed"] == 3 and payload["topics_changed"] == 2
    human = runner.invoke(app, ["index", "status"])
    assert "2 topics" in human.output, human.output

    updated = _json_stdout(runner.invoke(app, ["index", "update", "--json"]))
    assert updated["items_changed"] == 3 and updated["topics_refreshed"] == 2
    assert updated["topics_rebuilt"] is False
    payload = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert payload["items_changed"] == 0 and payload["topics_changed"] == 0


@pytest.mark.parametrize("moved", ["topics.json", "vocab.yaml"])
def test_topics_or_vocab_rewritten_on_disk_makes_search_warn_and_status_say_behind(
    workspace: Path, moved: str
) -> None:
    """P1a at the CLI (gate Codex, round 05): the operator path is `xbrain topics`, which
    writes `data/topics.json` and never `items.json` (`cli._topics_run` / `_topics_apply`),
    and `xbrain vocab`, which writes `vocab.yaml`. After either, `search` must warn and
    `index status` must say `behind`, and one `index update` must clear both — the promise
    `docs/tutorial.md` made and the code did not keep.

    The files are rewritten through the SAME writers the commands use, with a change the
    index would actually serve (a note, a description). Seen red before the fix: `search`
    printed no warning and `status --json` said `"behind": false` on both.
    """
    from xbrain.models import Topic, TopicPage
    from xbrain.rubrics import load_vocab, save_vocab
    from xbrain.store import load_topic_pages, save_topic_pages

    assert runner.invoke(app, ["index", "build"]).exit_code == 0
    assert "index update" not in runner.invoke(app, ["search", "Quillfeather"]).output

    path = workspace / "data" / moved
    if moved == "topics.json":
        pages: dict[str, TopicPage] = load_topic_pages(path)
        slug = sorted(pages)[0]
        pages[slug] = pages[slug].model_copy(update={"notes": [*pages[slug].notes, "nuevo"]})
        save_topic_pages(pages, path)
    else:
        vocab: list[Topic] = load_vocab(path)
        vocab[0] = vocab[0].model_copy(update={"description": vocab[0].description + " más"})
        save_vocab(vocab, path)

    human = runner.invoke(app, ["search", "Quillfeather"])
    assert human.exit_code == 0, human.output
    assert "xbrain index update" in human.output, human.output
    payload = _json_stdout(runner.invoke(app, ["search", "Quillfeather", "--json"]))
    assert "index_behind_store" in payload["index"]["degraded"]
    assert _json_stdout(runner.invoke(app, ["index", "status", "--json"]))["behind"] is True

    assert runner.invoke(app, ["index", "update"]).exit_code == 0
    assert _json_stdout(runner.invoke(app, ["index", "status", "--json"]))["behind"] is False
    assert "index update" not in runner.invoke(app, ["search", "Quillfeather"]).output


@pytest.mark.parametrize("command", [["index", "build"], ["index", "update"]])
def test_index_build_and_update_seal_the_manifest_with_the_snapshot_they_loaded(
    workspace: Path, monkeypatch, command: list[str]
) -> None:
    """P1b at the CLI: the window is between the CLI's load and the writer's commit, and the
    CLI is the caller that has to hand the loader's signal through. Staged by replacing
    `data/items.json` from inside the store parser, i.e. after the bytes were read and before
    the command builds — the same race the gate ran by hand. Afterwards `search` must warn
    and `status` must count the one item the base never saw.

    Seen red before the fix: `search --json` carried no `index_behind_store` and `status`
    said `behind: false` with `items_changed: 1` — two instruments, opposite answers.
    """
    from xbrain.knowledge import index_build

    if command == ["index", "update"]:
        assert runner.invoke(app, ["index", "build"]).exit_code == 0
    items_path = workspace / "data" / "items.json"
    real = index_build.parse_store

    def replace_then_parse(text: str):
        raw = json.loads(items_path.read_text(encoding="utf-8"))
        raw["k01"]["text"] = raw["k01"]["text"] + " raceonlytoken"
        items_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        monkeypatch.setattr(index_build, "parse_store", real)
        return real(text)

    monkeypatch.setattr(index_build, "parse_store", replace_then_parse)
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output

    payload = _json_stdout(runner.invoke(app, ["search", "Quillfeather", "--json"]))
    assert "index_behind_store" in payload["index"]["degraded"], payload["index"]
    status = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert status["behind"] is True and status["items_changed"] == 1


def test_get_works_after_the_index_is_removed(workspace: Path) -> None:
    """Acceptance 9 at the CLI: `get` reads the store, so the index can be gone."""
    runner.invoke(app, ["index", "build"])
    shutil.rmtree(workspace / "data" / "index")
    payload = _json_stdout(
        runner.invoke(app, ["get", "k03", "--surface", "external_article", "--json"])
    )
    assert payload["surfaces"][0]["surface_type"] == "external_article"


@pytest.mark.parametrize(
    "request_args",
    [
        pytest.param(["--surface", "external_article"], id="positional"),
        pytest.param(["--surface", "external_article", "--query", "retrieval"], id="query"),
    ],
)
def test_the_printed_continuation_command_continues_the_same_sequence(
    workspace: Path, request_args: list[str]
) -> None:
    """H2 at the CLI: the continuation `get` prints is FOLLOWED LITERALLY, page after page.

    The gate did exactly this and got an empty page on the positional route and a refused
    cursor on the query route, because the printed line was `xbrain get ID --cursor C` with
    no `--surface` and no `--query`. Each page's line is parsed with `shlex`, run as printed
    (plus `--json` to read it back), and the walk must deliver something on every page,
    repeat nothing, and reassemble what one unbounded call returns: the surface's text
    verbatim on the positional route, the ranked chunk list on the query route.

    Seen red before the fix: `the printed command led to an empty page` (positional) and
    exit code 1 with `Cursor inválido` (query).
    """
    request = ["get", "k03", *request_args, "--budget", "500"]
    first = runner.invoke(app, request)
    assert first.exit_code == 0, first.output
    first_page = _json_stdout(runner.invoke(app, [*request, "--json"]))
    assert first_page["truncated"] is True, "the budget must force a continuation"
    pages = [first_page]

    line = first.output.splitlines()[-1]
    for _ in range(20):
        assert "Continúa con: xbrain get " in line, line
        command = shlex.split(line.split("Continúa con: ", 1)[1])[1:]
        followed = runner.invoke(app, [*command, "--json"])
        assert followed.exit_code == 0, f"{command}: {followed.output}"
        page = json.loads(followed.stdout)
        assert page["chunks"] or page["surfaces"], f"{command} led to an empty page"
        pages.append(page)
        if not page["truncated"]:
            break
        line = runner.invoke(app, command).output.splitlines()[-1]
    else:
        pytest.fail("the printed continuations never finished paginating")

    ids = [chunk["chunk_id"] for page in pages for chunk in page["chunks"]]
    assert len(ids) == len(set(ids)), "a page repeated a chunk"
    whole = _json_stdout(
        runner.invoke(app, ["get", "k03", *request_args, "--budget", "10000000", "--json"])
    )
    if "--query" in request_args:
        assert ids == [chunk["chunk_id"] for chunk in whole["chunks"]]
    else:
        texts = [chunk["text"] for page in pages for chunk in page["chunks"]]
        texts += [surface["text"] for page in pages for surface in page["surfaces"]]
        assert "".join(texts) == whole["surfaces"][0]["text"]


def test_a_quoted_chunk_from_get_query_names_the_quoted_author(workspace: Path) -> None:
    """H3 at the CLI, on the fixture the gate used: `get k07 --surface quoted_post --query
    weights` returns the quoted post as a CHUNK, and the human view placed it under
    `@vgonpa` with no author of its own. `@othervoice` wrote it. The header of the chunk must
    say so, on the same line as the surface label, while the bundle header keeps naming the
    poster — a reader sees both in two seconds (CLAUDE.md rule 7).

    Seen red before the fix: no line of the output mentioned `othervoice`.
    """
    result = runner.invoke(app, ["get", "k07", "--surface", "quoted_post", "--query", "weights"])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("k07  @vgonpa (Victor Gonzalez)"), lines[0]
    (header,) = [line for line in lines if line.startswith("[quoted_post")]
    assert "autor: @othervoice (Other Voice)" in header, header


def test_the_human_search_output_names_the_get_command(workspace: Path) -> None:
    """Step 27 at the CLI: the human view is rendered from the SAME response model."""
    runner.invoke(app, ["index", "build"])
    result = runner.invoke(app, ["search", "Quillfeather"])
    assert result.exit_code == 0
    assert "xbrain get " in result.output


def test_get_reports_the_configured_transcriber_as_the_transcripts_producer(
    workspace: Path,
) -> None:
    """A-4 at the adapter: the ONE place the producer is defined is `config.toml`, and both
    `index build` and `get` must read it from there (rule 5). Before the fix `_query_context`
    dropped it and `get` answered `producer: null` for a transcript the build had stamped.

    Seen red before the fix: `producer` was `None`.
    """
    config = workspace / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + '[transcribe]\ncommand = "review-transcriber"\n',
        encoding="utf-8",
    )
    bundle = _json_stdout(
        runner.invoke(app, ["get", "k08", "--surface", "video_transcript", "--json"])
    )
    assert bundle["surfaces"][0]["surface_type"] == "video_transcript"
    assert bundle["surfaces"][0]["producer"] == "review-transcriber"
