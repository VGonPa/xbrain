# tests/test_knowledge_index_cli.py
"""`xbrain index build|update|status`, `xbrain search` and `xbrain get` (Plan 02 §6).

THE CLI OF THIS PLAN IS AN ADAPTER, and every test here holds it to that. Each one either
proves a command actually reaches its service, or proves the two renderings come from ONE
model — never that some string turns up somewhere in the output, which is the assertion
CLAUDE.md rule 1 was written about.

The convention is the one the rest of the CLI already follows: `--json` writes a stable
document to stdout and NOTHING else (spec §3.7.9), diagnostics go to stderr, and none of
these five commands writes to `items.json`, `vocab.yaml` or `topics.json` (spec §3.7.12).
`index build` does write — to `data/index/`, which is derived and reconstructible — which is
why the read-only test digests the three inputs and not the directory.
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
    """A repo-shaped temp dir with a `data/` built from the committed fixture corpus.

    The corpus is a FIXTURE, never `data/`: in CI there is no store at all, and a test that
    reached for one would be a test that silently stops running there (Plan 02 §1).
    """
    raw = json.loads((FIXTURES / "knowledge_corpus.json").read_text(encoding="utf-8"))
    data = tmp_path / "data"
    data.mkdir()
    (data / "items.json").write_text(json.dumps(raw["items"], indent=2), encoding="utf-8")
    (data / "topics.json").write_text(json.dumps(raw["topics"], indent=2), encoding="utf-8")
    (data / "vocab.yaml").write_text(
        yaml.safe_dump({"topics": list(raw["vocab"].values())}, allow_unicode=True),
        encoding="utf-8",
    )
    (tmp_path / "config.toml").write_text(
        '[paths]\nvault = "vault"\noutput_subdir = "x-knowledge"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _json_stdout(result) -> dict:
    """Parse the WHOLE of stdout. A stray log line makes this raise, which is the point."""
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _build(workspace: Path) -> None:
    result = runner.invoke(app, ["index", "build"])
    assert result.exit_code == 0, result.output


_STORE_FILES = ("data/items.json", "data/vocab.yaml", "data/topics.json")


def _store_digest(root: Path) -> dict[str, str]:
    """sha256 of the three inputs — the evidence that a command wrote to none of them."""
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _STORE_FILES}


# ---------------------------------------------------------------------------
# The two refusals the plan names, and the read-only guarantee
# ---------------------------------------------------------------------------


def test_no_command_of_this_plan_writes_to_the_store(workspace: Path) -> None:
    """Acceptance 13 / step 28: `search`, `get` and the three `index` commands are read-only.

    Hashed before and after over all THREE inputs, not just `items.json`: `vocab.yaml` and
    `topics.json` are equally the store, and a command that rewrote one of them while leaving
    `items.json` alone would pass a one-file check.
    """
    before = _store_digest(workspace)
    for argv in (
        ["index", "build"],
        ["index", "update"],
        ["index", "status"],
        ["search", "retrieval"],
        ["get", "k01"],
        ["index", "build", "--force", "--dry-run"],
    ):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, f"{argv} -> {result.output}"
    assert _store_digest(workspace) == before


def test_search_without_an_index_names_the_build_command(workspace: Path) -> None:
    """Step 30 / §11: an absent index is an ACTIONABLE error, never a raw traceback.

    And never an empty result set either, which is the failure that matters: `0 resultados`
    over a corpus that was simply never indexed is a claim about the corpus, and it is
    indistinguishable from a query nothing matched.
    """
    result = runner.invoke(app, ["search", "retrieval"])
    assert result.exit_code == 1
    assert "xbrain index build" in result.stderr
    assert "Traceback" not in result.stderr


def test_an_index_error_prints_exactly_one_error_line(workspace: Path) -> None:
    """One failure, one `Error:` line — the user-visible half of a wrapper fix.

    `_handle_index_errors` reports the fault and then raises `typer.Exit(code=1)`, and it is
    stacked UNDER `_handle_cli_errors`. `click.exceptions.Exit` subclasses `RuntimeError`,
    which the outer wrapper's `_OPERATOR_ERRORS` lists — so that deliberate exit was caught
    and re-reported as a second, EMPTY `Error: ` line after the real message. The exit code
    was 1 either way, which is exactly why no existing test saw it.
    """
    result = runner.invoke(app, ["search", "retrieval"])
    assert result.exit_code == 1
    assert result.stderr.count("Error:") == 1, result.stderr


# ---------------------------------------------------------------------------
# index build / update / status
# ---------------------------------------------------------------------------


def test_index_build_dry_run_touches_no_file(workspace: Path) -> None:
    """Step 4: `--dry-run` reports what a build WOULD do and creates nothing."""
    payload = _json_stdout(runner.invoke(app, ["index", "build", "--dry-run", "--json"]))
    assert payload["dry_run"] is True
    assert payload["chunks_written"] > 0
    assert not (workspace / "data" / "index").exists()


def test_index_build_refuses_a_second_build_and_names_both_commands(workspace: Path) -> None:
    """A rebuild throws away minutes of work, so it is opt-in — and the error says how.

    `index build` over an existing index names BOTH ways out (`index update` for the usual
    case, `--force` for a real rebuild); an error naming neither would leave the operator
    guessing at the one command that is not destructive.
    """
    _build(workspace)
    result = runner.invoke(app, ["index", "build"])
    assert result.exit_code == 1
    assert "xbrain index update" in result.stderr
    assert "--force" in result.stderr
    assert runner.invoke(app, ["index", "build", "--force"]).exit_code == 0


def test_index_status_json_carries_the_manifest_the_file_holds(workspace: Path) -> None:
    """Acceptance 2: counts, versions, date and skipped — from the manifest, not retyped.

    The assertion is IDENTITY against `data/index/manifest.json`, not a field-by-field
    spot-check: `status --json` and the manifest on disk are ONE document, so a CLI that
    rebuilt the payload by hand — dropping a field, or renaming one — goes red here instead
    of quietly publishing a second shape of the same thing (rule 5).
    """
    _build(workspace)
    payload = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    on_disk = json.loads((workspace / "data" / "index" / "manifest.json").read_text("utf-8"))
    assert payload["manifest"] == on_disk
    assert payload["counts"]["chunks"] > 0
    assert payload["incomplete"] is False


def test_index_status_without_an_index_says_so_and_names_build(workspace: Path) -> None:
    """`status` is the instrument an operator runs to find out, so it never raises here.

    The other four commands refuse a missing index; this one REPORTS it. Two instruments
    with opposite answers on one state is rule 9, and the way out is for the diagnostic one
    to answer while the doors refuse — not for it to refuse too.
    """
    payload = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert payload["manifest"] is None
    assert payload["incomplete"] is True
    assert "xbrain index build" in payload["advice"]


def test_index_update_reports_the_item_that_changed(workspace: Path) -> None:
    """Acceptance 4b in its CLI form: a changed store is COUNTED, not merely noticed.

    `status` must say HOW MANY items moved before the update, and `update` must say it
    touched exactly that one afterwards. A command reporting "something changed" would leave
    the operator unable to decide whether the rebuild is worth its minutes.
    """
    _build(workspace)
    items = json.loads((workspace / "data" / "items.json").read_text("utf-8"))
    items["k01"]["text"] = "Zephyrine is the codename, now rewritten."
    (workspace / "data" / "items.json").write_text(json.dumps(items, indent=2), "utf-8")

    before = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert before["items_changed"] == 1
    assert before["behind"] is True

    report = _json_stdout(runner.invoke(app, ["index", "update", "--json"]))
    assert report["items_changed"] == 1
    assert report["items_added"] == 0 and report["items_removed"] == 0

    after = _json_stdout(runner.invoke(app, ["index", "status", "--json"]))
    assert after["items_changed"] == 0
    assert after["behind"] is False


def test_index_update_dry_run_leaves_the_index_where_it_was(workspace: Path) -> None:
    """A dry run reports the delta it WOULD apply and re-seals nothing."""
    _build(workspace)
    items = json.loads((workspace / "data" / "items.json").read_text("utf-8"))
    items["k01"]["text"] = "Zephyrine, rewritten again."
    (workspace / "data" / "items.json").write_text(json.dumps(items, indent=2), "utf-8")
    manifest = (workspace / "data" / "index" / "manifest.json").read_bytes()

    report = _json_stdout(runner.invoke(app, ["index", "update", "--dry-run", "--json"]))
    assert report["dry_run"] is True and report["items_changed"] == 1
    assert (workspace / "data" / "index" / "manifest.json").read_bytes() == manifest


# ---------------------------------------------------------------------------
# search — the response model, the human view, the filters
# ---------------------------------------------------------------------------


def test_search_json_is_the_response_model_and_nothing_else(workspace: Path) -> None:
    """Step 26 + spec §3.7.9: stdout is the document, diagnostics never mix into it.

    Parsed WHOLE, and then re-validated against the frozen contract, which is the half a
    substring check cannot do: `extra="forbid"` means a CLI that added a convenience key to
    the payload — or dropped one — fails here rather than downstream in a consumer, with a
    parse error a long way from the cause. The human view's degradation sentence is asserted
    ABSENT because it is the line most likely to leak into a `--json` run: the renderer
    prints it BEFORE the results, not after them.
    """
    from xbrain.knowledge.contracts import SearchResponse

    _build(workspace)
    result = runner.invoke(app, ["search", "retrieval", "--json"])
    payload = _json_stdout(result)
    response = SearchResponse.model_validate(payload)
    assert response.query == "retrieval"
    assert response.strategy == "lexical"
    assert [r.item_id for r in response.results]
    assert "Estrategia léxica" not in result.stdout


def test_search_human_view_is_the_renderer_over_the_same_response(workspace: Path) -> None:
    """Step 27 + spec §7.6: ONE model, TWO renderings — never a second shape in `cli.py`.

    The human run's stdout is compared against `render.render_search` applied to the model
    rebuilt from the `--json` run. That is the binding (rule 5): a CLI that formatted its own
    lines, or that derived the human view from anything other than the response it
    serialises, goes red. Asserting instead that "the output contains `xbrain get`" would
    have passed on a hand-rolled formatter that happened to print the same suggestion.
    """
    from xbrain.knowledge import render
    from xbrain.knowledge.contracts import SearchResponse

    _build(workspace)
    payload = _json_stdout(runner.invoke(app, ["search", "retrieval", "--json"]))
    human = runner.invoke(app, ["search", "retrieval"])
    assert human.exit_code == 0
    assert human.stdout.rstrip("\n") == render.render_search(SearchResponse.model_validate(payload))
    # And what §7.6 requires be shown IS shown, through that renderer: the origin of every
    # match, and the command that fetches the source behind it.
    assert "origin=" in human.stdout
    assert "xbrain get" in human.stdout


def test_search_mine_is_the_own_tweet_filter_and_it_is_applied(workspace: Path) -> None:
    """`--mine` is the spec §7.2 shortcut for `source=own_tweet` — APPLIED, not just echoed.

    The same query runs twice, and the unfiltered run must return items the filtered one does
    not. Asserting only the echoed `filters.source` would pass on a CLI that set the field and
    never handed it to the service, which is the fail-open shape the plan warns about for all
    eight filters.
    """
    _build(workspace)
    everything = _json_stdout(runner.invoke(app, ["search", "how", "--json"]))
    mine = _json_stdout(runner.invoke(app, ["search", "how", "--mine", "--json"]))
    assert mine["filters"]["source"] == "own_tweet"
    mine_ids = {r["item_id"] for r in mine["results"]}
    all_ids = {r["item_id"] for r in everything["results"]}
    assert mine_ids == {"k06"}
    assert all_ids > mine_ids


def test_search_topic_filter_reaches_the_service(workspace: Path) -> None:
    """A repeatable filter is a tuple on `SearchFilters`, and it narrows the answer.

    The unfiltered run is the control: without it, a topic filter that silently matched
    everything would look identical to one that worked.
    """
    _build(workspace)
    everything = _json_stdout(runner.invoke(app, ["search", "sobre", "--json"]))
    scoped = _json_stdout(runner.invoke(app, ["search", "sobre", "--topic", "ai-policy", "--json"]))
    assert scoped["filters"]["topics"] == ["ai-policy"]
    scoped_ids = {r["item_id"] for r in scoped["results"]}
    all_ids = {r["item_id"] for r in everything["results"]}
    assert scoped_ids and scoped_ids < all_ids


def test_search_rejects_mine_together_with_an_explicit_source(workspace: Path) -> None:
    """Two ways to set one field is one way to set it to two different things."""
    _build(workspace)
    result = runner.invoke(app, ["search", "retrieval", "--mine", "--source", "bookmark"])
    assert result.exit_code == 1
    assert "--mine" in result.stderr


@pytest.mark.parametrize(
    ("argv", "needle"),
    [
        (["search", "retrieval", "--limit", "0"], "--limit"),
        (["search", "   "], "vacía"),
        (["search", "retrieval", "--topic", "no-such-topic"], "agent-evaluation"),
    ],
)
def test_search_refuses_what_it_cannot_answer_honestly(
    workspace: Path, argv: list[str], needle: str
) -> None:
    """§11: a validation error is stable and names what WOULD have been valid.

    The unknown topic is the one that matters: answering it with zero results would be a
    claim about the corpus manufactured by a typo, so the error lists the vocabulary instead.
    """
    _build(workspace)
    result = runner.invoke(app, argv)
    assert result.exit_code == 1
    assert needle in result.stderr


# ---------------------------------------------------------------------------
# get — the store, never the index
# ---------------------------------------------------------------------------


def test_get_answers_with_the_index_directory_deleted(workspace: Path) -> None:
    """Step 21 / acceptance 9: `get` reads the STORE, so it works with no index at all.

    Spec §3.7.7's invariant 7 in its operational form. The index is built FIRST and then
    removed, so the test cannot pass by never having had one: an implementation that read the
    index when present and the store when absent would still be two sources of truth, and
    this ordering is what catches it the day the two bundles differ.
    """
    _build(workspace)
    argv = ["get", "k04", "--surface", "external_article", "--json"]
    with_index = _json_stdout(runner.invoke(app, argv))
    shutil.rmtree(workspace / "data" / "index")
    assert _json_stdout(runner.invoke(app, argv)) == with_index


def test_get_json_round_trips_the_frozen_evidence_bundle(workspace: Path) -> None:
    """The `--json` document validates against the contract Plan 01 froze, key for key."""
    from xbrain.knowledge.contracts import EvidenceBundle

    payload = _json_stdout(runner.invoke(app, ["get", "k04", "--json"]))
    bundle = EvidenceBundle.model_validate(payload)
    assert bundle.item.item_id == "k04"
    assert "external_article" in bundle.item.available_surfaces


def test_get_human_view_is_the_renderer_over_the_same_bundle(workspace: Path) -> None:
    """The §7.6 binding on the other service, with the REQUEST its continuation needs.

    `--surface` and `--query` are passed to the renderer because the printed continuation is
    an offset into the sequence THEY define: a continuation rendered without them resumes
    inside the default selection and returns a different page.
    """
    from xbrain.knowledge import render
    from xbrain.knowledge.contracts import EvidenceBundle

    argv = ["get", "k04", "--surface", "external_article", "--query", "retrieval"]
    payload = _json_stdout(runner.invoke(app, [*argv, "--json"]))
    human = runner.invoke(app, argv)
    assert human.exit_code == 0
    assert human.stdout.rstrip("\n") == render.render_get(
        EvidenceBundle.model_validate(payload),
        surfaces=["external_article"],
        query="retrieval",
    )


def test_get_names_the_valid_surfaces_when_asked_for_one_the_item_lacks(
    workspace: Path,
) -> None:
    """A surface an item does not have is refused BY NAME, never answered with an empty page.

    An empty bundle would be indistinguishable from a transcript full of nothing, which is
    the distinction `UnknownSurfaceError` exists to keep.
    """
    result = runner.invoke(app, ["get", "k01", "--surface", "video_transcript"])
    assert result.exit_code == 1
    assert "post" in result.stderr


def test_get_honours_the_configured_char_budget_and_publishes_a_cursor(
    workspace: Path,
) -> None:
    """`[index].get_char_budget` reaches `get`, and a cut page is DECLARED (spec §9.3).

    The budget is lowered in `config.toml`, so the assertion is that configuration travels to
    the service — not that some default happens to truncate. A silent cut is the failure the
    cursor exists to prevent, so both halves are asserted: `truncated` AND a continuation.
    """
    (workspace / "config.toml").write_text(
        '[paths]\nvault = "vault"\noutput_subdir = "x-knowledge"\ndata_dir = "data"\n'
        '[x]\nhandle = "vgonpa"\n[index]\nget_char_budget = 400\n',
        encoding="utf-8",
    )
    payload = _json_stdout(
        runner.invoke(app, ["get", "k04", "--surface", "external_article", "--json"])
    )
    assert payload["truncated"] is True
    assert payload["cursor"]
