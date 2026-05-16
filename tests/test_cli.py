# tests/test_cli.py
from pathlib import Path

from typer.testing import CliRunner

from xkb.cli import app

runner = CliRunner()


def _setup_repo(tmp_path: Path, monkeypatch) -> Path:
    vault = tmp_path / "vault"
    vault.mkdir()
    (tmp_path / "config.toml").write_text(
        '[paths]\n'
        f'vault = "{vault}"\n'
        'output_subdir = "x-knowledge"\n'
        'data_dir = "data"\n'
        '[x]\n'
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )
    (tmp_path / "courses.yaml").write_text("courses: []\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    monkeypatch.setenv("XKB_REPO_ROOT", str(tmp_path))
    return vault


def test_status_runs_on_empty_store(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Items: 0" in result.stdout


def test_generate_creates_output_dir(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    result = runner.invoke(app, ["generate"])
    assert result.exit_code == 0
    assert (vault / "x-knowledge" / "_index.md").exists()
