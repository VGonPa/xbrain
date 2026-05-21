# tests/test_snapshot_auto.py
"""Integration tests for the auto-snapshot hooks in cli.py.

Every destructive flag (`vocab --regenerate`, `topics --resynth`,
`fetch --force`) must create exactly one snapshot under data/snapshots/
before any mutation happens. Non-destructive runs must not.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from xbrain.cli import app
from xbrain.models import Author, Item, Link
from xbrain.snapshot import snapshot_list
from xbrain.store import save_store

runner = CliRunner()


def _setup_repo(tmp_path: Path, monkeypatch) -> Path:
    """Mirror of tests/test_cli.py: minimal config.toml + data dir, repo root via env."""
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


def _linked_item(item_id: str = "1") -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Note {item_id}",
        created_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        captured_at=datetime(2026, 5, 16, tzinfo=timezone.utc),
        links=[Link(url="https://example.com/p", domain="example.com")],
    )


def _write_vocab(tmp_path: Path) -> None:
    """Write a minimal vocab.yaml (load_vocab expects `topics:` root key)."""
    (tmp_path / "data" / "vocab.yaml").write_text(
        "topics:\n  - slug: misc\n    description: Posts that do not fit a specific topic.\n",
        encoding="utf-8",
    )


def test_vocab_regenerate_with_worksheet_creates_pre_snapshot(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    _write_vocab(tmp_path)

    # Build a minimal vocab worksheet that satisfies apply_vocab_worksheet
    # (the loader expects a flat `topics` list at the JSON root)
    worksheet = tmp_path / "data" / "vocab-worksheet.json"
    worksheet.write_text(
        json.dumps(
            {"topics": [{"slug": "misc", "description": "Posts that do not fit a specific topic."}]}
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["vocab", "--apply", str(worksheet), "--regenerate"])

    assert result.exit_code == 0, result.stdout
    snapshots = snapshot_list(tmp_path / "data")
    assert any(p.name.endswith("-pre-vocab-regenerate") for p, _ in snapshots), snapshots


def test_vocab_without_regenerate_creates_no_snapshot(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")

    # vocab without --regenerate, claude-code executor (worksheet export, no mutation
    # outside writing vocab-worksheet.json, which is not an artifact under snapshot scope)
    result = runner.invoke(app, ["vocab", "--executor", "claude-code"])

    assert result.exit_code == 0, result.stdout
    snapshots = snapshot_list(tmp_path / "data")
    assert snapshots == []


def test_topics_resynth_creates_pre_snapshot(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    _write_vocab(tmp_path)
    (vault / "x-knowledge").mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["topics", "--resynth", "--executor", "manual"])

    # Topics may succeed or report nothing pending; either way the snapshot
    # must have been taken before the resynth path branched.
    assert result.exit_code == 0, result.stdout
    snapshots = snapshot_list(tmp_path / "data")
    assert any(p.name.endswith("-pre-topics-resynth") for p, _ in snapshots), snapshots


def test_topics_without_resynth_creates_no_snapshot(tmp_path: Path, monkeypatch):
    vault = _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    _write_vocab(tmp_path)
    (vault / "x-knowledge").mkdir(parents=True, exist_ok=True)

    result = runner.invoke(app, ["topics", "--executor", "manual"])

    assert result.exit_code == 0, result.stdout
    assert snapshot_list(tmp_path / "data") == []


def test_fetch_force_creates_pre_snapshot(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    # An item with no links → fetch_pending no-ops; the snapshot still fires
    item = _linked_item("1")
    item.links = []
    save_store({"1": item}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["fetch", "--force"])

    assert result.exit_code == 0, result.stdout
    snapshots = snapshot_list(tmp_path / "data")
    assert any(p.name.endswith("-pre-fetch-force") for p, _ in snapshots), snapshots


def test_fetch_without_force_creates_no_snapshot(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    item = _linked_item("1")
    item.links = []
    save_store({"1": item}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["fetch"])

    assert result.exit_code == 0, result.stdout
    assert snapshot_list(tmp_path / "data") == []


def test_snapshot_create_cli_writes_manifest(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")

    result = runner.invoke(app, ["snapshot", "create", "--name", "before-rubric-v2"])

    assert result.exit_code == 0, result.stdout
    snapshots = snapshot_list(tmp_path / "data")
    assert len(snapshots) == 1
    assert snapshots[0][0].name.endswith("-before-rubric-v2")
    assert snapshots[0][1].item_count == 1


def test_snapshot_list_cli_reports_each_snapshot(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    runner.invoke(app, ["snapshot", "create", "--name", "a"])
    runner.invoke(app, ["snapshot", "create", "--name", "b"])

    result = runner.invoke(app, ["snapshot", "list"])

    assert result.exit_code == 0
    assert "-a" in result.stdout
    assert "-b" in result.stdout


def test_snapshot_restore_cli_brings_back_items(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")

    create = runner.invoke(app, ["snapshot", "create", "--name", "checkpoint"])
    assert create.exit_code == 0

    # Mutate items.json out from under us
    save_store({}, tmp_path / "data" / "items.json")

    name = next(p.name for p, _ in snapshot_list(tmp_path / "data"))
    restore = runner.invoke(app, ["snapshot", "restore", name])

    assert restore.exit_code == 0, restore.stdout
    data = json.loads((tmp_path / "data" / "items.json").read_text())
    assert set(data) == {"1"}


def test_snapshot_prune_cli_deletes_older(tmp_path: Path, monkeypatch):
    _setup_repo(tmp_path, monkeypatch)
    save_store({"1": _linked_item("1")}, tmp_path / "data" / "items.json")
    for i in range(4):
        runner.invoke(app, ["snapshot", "create", "--name", f"s{i}"])

    result = runner.invoke(app, ["snapshot", "prune", "--keep-last", "2"])

    assert result.exit_code == 0, result.stdout
    assert len(snapshot_list(tmp_path / "data")) == 2
