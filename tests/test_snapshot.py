# tests/test_snapshot.py
"""Unit tests for xbrain.snapshot — pure I/O, no CLI, no mocking the FS."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from xbrain.snapshot import (
    SnapshotManifest,
    snapshot_create,
    snapshot_list,
    snapshot_pre,
    snapshot_prune,
    snapshot_restore,
    snapshot_show,
    snapshots_dir,
)


def _seed_data_dir(data_dir: Path) -> None:
    """Populate data_dir with realistic, parseable artifacts."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "items.json").write_text(
        json.dumps({"1": {"id": "1"}, "2": {"id": "2"}}), encoding="utf-8"
    )
    (data_dir / "state.json").write_text(json.dumps({"bookmarks": {}}), encoding="utf-8")
    (data_dir / "vocab.yaml").write_text(
        "topics:\n"
        "  - slug: ai-coding\n    description: foo\n"
        "  - slug: software\n    description: bar\n"
        "  - slug: misc\n    description: baz\n",
        encoding="utf-8",
    )
    (data_dir / "topics.json").write_text(
        json.dumps({"ai-coding": {"slug": "ai-coding"}}), encoding="utf-8"
    )


def test_snapshot_create_on_empty_data_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = snapshot_create(data_dir, command="manual")

    assert path.exists()
    assert (path / "snapshot.json").exists()
    # No artifacts to copy when data/ is empty
    assert not (path / "items.json").exists()
    manifest = SnapshotManifest.model_validate_json((path / "snapshot.json").read_text())
    assert manifest.item_count == 0
    assert manifest.topic_count == 0
    assert manifest.vocab_size == 0
    assert manifest.command == "manual"


def test_snapshot_create_copies_every_existing_artifact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_data_dir(data_dir)

    path = snapshot_create(data_dir, command="manual")

    for artifact in ("items.json", "state.json", "vocab.yaml", "topics.json"):
        assert (path / artifact).exists()
    manifest = SnapshotManifest.model_validate_json((path / "snapshot.json").read_text())
    assert manifest.item_count == 2
    assert manifest.topic_count == 1
    assert manifest.vocab_size == 3


def test_snapshot_pre_prefixes_command_with_pre(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = snapshot_pre(data_dir, command="vocab-regenerate")

    assert "pre-vocab-regenerate" in path.name
    manifest = SnapshotManifest.model_validate_json((path / "snapshot.json").read_text())
    assert manifest.command == "pre-vocab-regenerate"


def test_snapshot_create_with_name_uses_name_in_dirname(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path = snapshot_create(data_dir, command="manual", name="milestone-v1")

    assert path.name.endswith("-milestone-v1")


def test_snapshot_list_returns_newest_first(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    snapshot_create(data_dir, command="manual", name="first")
    time.sleep(1.1)  # ensure distinct timestamps at second granularity
    snapshot_create(data_dir, command="manual", name="second")

    rows = snapshot_list(data_dir)

    assert len(rows) == 2
    assert rows[0][0].name.endswith("-second")
    assert rows[1][0].name.endswith("-first")


def test_snapshot_list_skips_directories_without_manifest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    snapshots_dir(data_dir).mkdir(parents=True)
    (snapshots_dir(data_dir) / "bogus").mkdir()  # no manifest
    snapshot_create(data_dir, command="manual", name="real")

    rows = snapshot_list(data_dir)

    assert len(rows) == 1
    assert rows[0][0].name.endswith("-real")


def test_snapshot_restore_brings_back_original_content(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_data_dir(data_dir)
    snap = snapshot_create(data_dir, command="manual", name="checkpoint")

    # Mutate items.json
    (data_dir / "items.json").write_text(json.dumps({"999": {"id": "999"}}), encoding="utf-8")
    assert "999" in (data_dir / "items.json").read_text()

    snapshot_restore(data_dir, snap.name)

    restored = json.loads((data_dir / "items.json").read_text())
    assert set(restored) == {"1", "2"}


def test_snapshot_restore_deletes_live_file_missing_from_snapshot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Snapshot with no vocab.yaml present
    snap = snapshot_create(data_dir, command="manual", name="pre-vocab")

    # Now there IS a vocab.yaml — simulating "after vocab ran"
    (data_dir / "vocab.yaml").write_text(
        "topics:\n  - slug: ai-coding\n    description: x\n", encoding="utf-8"
    )
    assert (data_dir / "vocab.yaml").exists()

    snapshot_restore(data_dir, snap.name)

    assert not (data_dir / "vocab.yaml").exists()


def test_snapshot_restore_does_not_touch_files_outside_artifacts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sentinel = data_dir / "unrelated.txt"
    sentinel.write_text("important user data", encoding="utf-8")
    snap = snapshot_create(data_dir, command="manual", name="check")

    sentinel.write_text("mutated", encoding="utf-8")
    snapshot_restore(data_dir, snap.name)

    # Restore must not touch files outside the four artifacts
    assert sentinel.read_text() == "mutated"


def test_snapshot_prune_keeps_only_the_n_newest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(5):
        snapshot_create(data_dir, command="manual", name=f"s{i}")
        time.sleep(1.05)

    deleted = snapshot_prune(data_dir, keep_last=2)

    rows = snapshot_list(data_dir)
    assert deleted == 3
    assert len(rows) == 2
    # The two newest should remain — s4 and s3
    names = {row[0].name for row in rows}
    assert any(n.endswith("-s4") for n in names)
    assert any(n.endswith("-s3") for n in names)


def test_snapshot_prune_zero_keeps_nothing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    snapshot_create(data_dir, command="manual", name="a")
    snapshot_create(data_dir, command="manual", name="b")

    deleted = snapshot_prune(data_dir, keep_last=0)

    assert deleted == 2
    assert snapshot_list(data_dir) == []


def test_snapshot_prune_rejects_negative_keep_last(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(ValueError):
        snapshot_prune(data_dir, keep_last=-1)


def test_snapshot_show_raises_for_unknown_snapshot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        snapshot_show(data_dir, "does-not-exist")


def test_snapshot_show_returns_manifest_for_known_snapshot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_data_dir(data_dir)
    snap = snapshot_create(data_dir, command="manual", name="known")

    snap_path, manifest = snapshot_show(data_dir, snap.name)

    assert snap_path == snap
    assert manifest.item_count == 2
