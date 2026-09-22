# tests/test_jev_cli.py
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from tests.jev_fakes import FakeJevClient
from xbrain import cli
from xbrain.cli import app
from xbrain.config import Config
from xbrain.jev.client import JevClient
from xbrain.jev.store import load_assessments
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.rubrics import save_vocab
from xbrain.store import save_store

runner = CliRunner()
DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _refusing_client(reason: str) -> Callable[[Config], JevClient]:
    """A `_jev_client` stand-in that fails the test if the CLI builds a client at all.

    The factory is the seam where money starts being spent, so "did not call Jev" is
    asserted by making the CALL impossible, not by counting calls afterwards.
    """

    def _build(cfg: Config) -> JevClient:
        raise AssertionError(reason)

    return _build


def _setup_repo(tmp_path: Path, monkeypatch) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        "[paths]\n"
        f'vault = "{vault}"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("XBRAIN_REPO_ROOT", str(tmp_path))
    return vault


def _item(item_id: str, text: str, topics=("ai-coding",)) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="Resumen.",
            primary_topic=topics[0],
            topics=list(topics),
        ),
    )


def _seed_vocab(tmp_path: Path) -> None:
    save_vocab(
        [
            Topic(slug="ai-coding", description="Construir software con IA."),
            Topic(slug="startups", description="Fundar empresas."),
        ],
        tmp_path / "data" / "vocab.yaml",
    )


def _seed(tmp_path: Path) -> None:
    save_store(
        {"1": _item("1", "Claude Code hooks"), "2": _item("2", "Seed round tips", ("startups",))},
        tmp_path / "data" / "items.json",
    )
    _seed_vocab(tmp_path)


def test_jev_topics_assesses_saves_and_skips_current_on_rerun(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(nouls={"ai-coding": 0.93}, primary="ai-coding")
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "2 items por evaluar · 0 vigentes · 0 sin evidencia" in result.output
    assert "2 evaluadas · 0 fallidas" in result.output
    assert "jev-1.13.0" in result.output
    saved = load_assessments(tmp_path / "data" / "jev" / "topics.json")
    assert saved["1"].membership == {"ai-coding": 0.93, "startups": 0.05}
    # The judge that ANSWERED, not the one the config asked for: provenance travels on the
    # result, so a record can never be attributed to the wrong provider.
    assert saved["1"].provider == "fake"
    assert len(fake.calls) == 2

    monkeypatch.setattr(
        cli, "_jev_client", _refusing_client("no client expected when nothing is pending")
    )
    again = runner.invoke(app, ["jev", "topics"])
    assert again.exit_code == 0, again.output
    assert "0 items por evaluar · 2 vigentes · 0 sin evidencia" in again.output


def test_jev_topics_dry_run_counts_without_calling(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    monkeypatch.setattr(cli, "_jev_client", _refusing_client("--dry-run must not call Jev"))
    result = runner.invoke(app, ["jev", "topics", "--dry-run", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "1 items por evaluar · 0 vigentes · 0 sin evidencia" in result.output
    assert not (tmp_path / "data" / "jev").exists()


def test_jev_topics_without_key_is_a_clean_operator_error(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 1
    assert "TYPESAFE_API_KEY" in result.output


def test_jev_topics_reports_failed_items_and_keeps_the_rest(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(fail_when=lambda state: "Seed" in state["post"])
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 0, result.output
    assert "1 evaluadas · 1 fallidas" in result.output
    assert "FALLO 2: fake failure" in result.output
    assert list(load_assessments(tmp_path / "data" / "jev" / "topics.json")) == ["1"]


def test_jev_topics_closes_the_client_it_built(tmp_path: Path, monkeypatch):
    """The run releases the connection pool it opened, on the success path.

    `_jev_client` builds a client per invocation; without this the pool leaks once per run
    and nothing goes red, because a leaked socket does not fail a test that only reads the
    side-car.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert fake.closed is True


def test_jev_topics_closes_the_client_even_when_the_run_dies(tmp_path: Path, monkeypatch):
    """...and on the failure path, which is the one a `with`-less run forgets.

    Every item failing makes `run_assessments` raise, so the close must be in a `finally`
    and not on the line after the call.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    fake = FakeJevClient(fail_when=lambda state: True)
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)
    result = runner.invoke(app, ["jev", "topics"])
    assert result.exit_code == 1
    assert "ninguna de las 2 evaluaciones terminó" in result.output
    assert fake.closed is True


def test_jev_topics_never_touches_items_json(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    _seed(tmp_path)
    before = (tmp_path / "data" / "items.json").read_bytes()
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: FakeJevClient())
    assert runner.invoke(app, ["jev", "topics"]).exit_code == 0
    assert (tmp_path / "data" / "items.json").read_bytes() == before
    assert not list((tmp_path / "data").glob("snapshots/*"))


def test_jev_topics_truncates_a_long_failure_list(tmp_path: Path, monkeypatch):
    """Only the first ten failures are printed; the rest are COUNTED, never dropped.

    A dead key or a bad vocabulary fails nearly every item. One line each would bury the
    summary under thousands of identical lines, and dropping the overflow silently would
    understate the damage — so the tail is reported as a number.
    """
    _setup_repo(tmp_path, monkeypatch)
    _seed_vocab(tmp_path)
    store = {"01": _item("01", "El unico que pasa")}
    store.update({f"{n:02d}": _item(f"{n:02d}", f"Fallo numero {n}") for n in range(2, 13)})
    save_store(store, tmp_path / "data" / "items.json")
    fake = FakeJevClient(fail_when=lambda state: "unico" not in state["post"])
    monkeypatch.setattr(cli, "_jev_client", lambda cfg: fake)

    result = runner.invoke(app, ["jev", "topics"])

    assert result.exit_code == 0, result.output
    assert "1 evaluadas · 11 fallidas" in result.output
    # Ten shown, one accounted for — 11 failures, not 10 and a shrug.
    assert result.output.count("  FALLO ") == 10
    assert "… y 1 fallos más" in result.output
    # The one record that succeeded is still saved: a noisy failure list never costs work
    # that was already paid for.
    assert list(load_assessments(tmp_path / "data" / "jev" / "topics.json")) == ["01"]
