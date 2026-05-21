# tests/test_snapshot.py
"""Unit tests for xbrain.snapshot — pure I/O, no CLI, no mocking the FS."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xbrain.snapshot import (
    RESTORE_COPIED,
    RESTORE_DELETED,
    RESTORE_SKIPPED,
    snapshot_create,
    snapshot_list,
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

    path, manifest = snapshot_create(data_dir, command="manual")

    assert path.exists()
    assert (path / "snapshot.json").exists()
    assert not (path / "items.json").exists()  # nothing to copy
    assert manifest.item_count == 0
    assert manifest.topic_count == 0
    assert manifest.vocab_size == 0
    assert manifest.command == "manual"
    # xbrain_version is always populated (real version or 'unknown')
    assert manifest.xbrain_version != ""


def test_snapshot_create_copies_every_existing_artifact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_data_dir(data_dir)

    path, manifest = snapshot_create(data_dir, command="manual")

    for artifact in ("items.json", "state.json", "vocab.yaml", "topics.json"):
        assert (path / artifact).exists()
    assert manifest.item_count == 2
    assert manifest.topic_count == 1
    assert manifest.vocab_size == 3


def test_snapshot_create_uses_dir_label_for_directory_and_command_for_manifest(
    tmp_path: Path,
) -> None:
    """dir_label drives the directory name; command drives manifest.command."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path, manifest = snapshot_create(
        data_dir,
        command="vocab-regenerate",
        dir_label="pre-vocab-regenerate",
    )

    assert "pre-vocab-regenerate" in path.name
    # manifest.command is the destructive op name only — no `pre-` prefix
    assert manifest.command == "vocab-regenerate"


def test_snapshot_create_dir_label_defaults_to_command(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path, _ = snapshot_create(data_dir, command="manual")

    assert path.name.endswith("-manual")


def test_snapshot_create_with_explicit_dir_label_uses_it(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path, _ = snapshot_create(data_dir, command="manual", dir_label="milestone-v1")

    assert path.name.endswith("-milestone-v1")


def test_snapshot_create_propagates_corrupt_json(tmp_path: Path) -> None:
    """A corrupt items.json must abort the snapshot — not record count=0."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "items.json").write_text("not json at all", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        snapshot_create(data_dir, command="manual")
    # No snapshot directory should have been created
    assert not snapshots_dir(data_dir).exists() or list(snapshots_dir(data_dir).iterdir()) == []


def test_snapshot_create_millisecond_precision_avoids_collisions(tmp_path: Path) -> None:
    """Two snapshots in the same second must NOT collide on directory name."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    path_a, _ = snapshot_create(data_dir, command="manual", dir_label="a")
    path_b, _ = snapshot_create(data_dir, command="manual", dir_label="b")

    # No FileExistsError, both directories present
    assert path_a.exists() and path_b.exists()
    assert path_a != path_b


def test_snapshot_list_returns_newest_first(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    snapshot_create(data_dir, command="manual", dir_label="first")
    snapshot_create(data_dir, command="manual", dir_label="second")

    rows = snapshot_list(data_dir)

    assert len(rows) == 2
    # Both should have a manifest; newest is `second`.
    assert all(m is not None for _, m in rows)
    assert rows[0][0].name.endswith("-second")
    assert rows[1][0].name.endswith("-first")


def test_snapshot_list_returns_corrupt_dirs_with_none_manifest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    snapshots_dir(data_dir).mkdir(parents=True)
    bogus = snapshots_dir(data_dir) / "bogus"
    bogus.mkdir()
    snapshot_create(data_dir, command="manual", dir_label="real")

    rows = snapshot_list(data_dir)

    assert len(rows) == 2
    # Real one sorts first (newest); corrupt sorts last with manifest=None.
    real_path, real_manifest = rows[0]
    corrupt_path, corrupt_manifest = rows[1]
    assert real_path.name.endswith("-real")
    assert real_manifest is not None
    assert corrupt_path == bogus
    assert corrupt_manifest is None


def test_snapshot_restore_brings_back_all_four_artifacts(tmp_path: Path) -> None:
    """Restore round-trip verifies every artifact, not just items.json."""
    data_dir = tmp_path / "data"
    _seed_data_dir(data_dir)
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="checkpoint")

    # Mutate every artifact
    (data_dir / "items.json").write_text(json.dumps({"999": {"id": "999"}}), encoding="utf-8")
    (data_dir / "state.json").write_text(json.dumps({"mutated": True}), encoding="utf-8")
    (data_dir / "vocab.yaml").write_text("topics: []\n", encoding="utf-8")
    (data_dir / "topics.json").write_text(json.dumps({}), encoding="utf-8")

    actions = snapshot_restore(data_dir, snap.name)

    # All four came back to their pre-snapshot content
    assert set(json.loads((data_dir / "items.json").read_text())) == {"1", "2"}
    assert json.loads((data_dir / "state.json").read_text()) == {"bookmarks": {}}
    assert "ai-coding" in (data_dir / "vocab.yaml").read_text()
    assert "ai-coding" in (data_dir / "topics.json").read_text()
    # Every artifact reported as copied
    assert all(action == RESTORE_COPIED for _, action in actions)


def test_snapshot_restore_returns_per_artifact_actions(tmp_path: Path) -> None:
    """The action codes are observable so the CLI can echo every decision."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "items.json").write_text(json.dumps({"1": {}}), encoding="utf-8")
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="partial")

    # Add a live vocab.yaml that the snapshot doesn't have
    (data_dir / "vocab.yaml").write_text("topics: []\n", encoding="utf-8")

    actions = snapshot_restore(data_dir, snap.name)

    by_artifact = dict(actions)
    assert by_artifact["items.json"] == RESTORE_COPIED
    assert by_artifact["vocab.yaml"] == RESTORE_DELETED
    assert by_artifact["state.json"] == RESTORE_SKIPPED
    assert by_artifact["topics.json"] == RESTORE_SKIPPED


def test_snapshot_restore_deletes_live_file_missing_from_snapshot(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="pre-vocab")

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
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="check")

    sentinel.write_text("mutated", encoding="utf-8")
    snapshot_restore(data_dir, snap.name)

    assert sentinel.read_text() == "mutated"


def test_snapshot_restore_uses_copy2_preserving_metadata(tmp_path: Path) -> None:
    """Symmetric with create: shutil.copy2, not text round-trip — binary-safe."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Bytes that are not 7-bit ASCII but ARE valid UTF-8 — proves the path works
    # for non-trivial content. A future binary artifact would also survive.
    (data_dir / "items.json").write_text(
        json.dumps({"é": "España 🇪🇸"}, ensure_ascii=False), encoding="utf-8"
    )
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="utf")
    original_bytes = (snap / "items.json").read_bytes()

    (data_dir / "items.json").write_text("{}", encoding="utf-8")
    snapshot_restore(data_dir, snap.name)

    assert (data_dir / "items.json").read_bytes() == original_bytes


def test_snapshot_prune_keeps_only_the_n_newest(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    paths = [snapshot_create(data_dir, command="manual", dir_label=f"s{i}")[0] for i in range(5)]

    deleted = snapshot_prune(data_dir, keep_last=2)

    rows = snapshot_list(data_dir)
    assert deleted == 3
    assert len(rows) == 2
    # The two newest should remain — the last two created
    remaining = {row[0] for row in rows}
    assert remaining == {paths[3], paths[4]}


def test_snapshot_prune_with_fewer_than_keep_last(tmp_path: Path) -> None:
    """No-op when there are fewer snapshots than keep_last."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    snapshot_create(data_dir, command="manual", dir_label="a")
    snapshot_create(data_dir, command="manual", dir_label="b")

    deleted = snapshot_prune(data_dir, keep_last=10)

    assert deleted == 0
    assert len(snapshot_list(data_dir)) == 2


def test_snapshot_prune_zero_keeps_nothing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    snapshot_create(data_dir, command="manual", dir_label="a")
    snapshot_create(data_dir, command="manual", dir_label="b")

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
    snap, _ = snapshot_create(data_dir, command="manual", dir_label="known")

    snap_path, manifest = snapshot_show(data_dir, snap.name)

    assert snap_path == snap
    assert manifest.item_count == 2


def test_manifest_records_xbrain_version(tmp_path: Path) -> None:
    """PRD §5 requires xbrain_version in the manifest."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    _, manifest = snapshot_create(data_dir, command="manual")

    # Either a real version or the fallback — never empty
    assert manifest.xbrain_version != ""
    assert isinstance(manifest.xbrain_version, str)
