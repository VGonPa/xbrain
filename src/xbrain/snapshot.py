"""Snapshot lifecycle for the data/ directory.

Auto-snapshots before destructive operations (`vocab --regenerate`, `topics
--resynth`, `fetch --force`) are XBrain's recovery boundary: every destructive
write is preceded by a complete directory copy at `data/snapshots/<ts>-pre-<cmd>/`.

Manual snapshots (`xbrain snapshot create [--name X]`) let the user mark a
known-good state on demand.

Both kinds are reversible via `snapshot_restore`. The whole module is pure I/O
plus pydantic — no CLI side-effects.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel

from xbrain.store import _atomic_write

# The mutable artifacts that live in data/. Order matters for restore: items
# first (the source of truth), then derived stores.
_ARTIFACTS = ("items.json", "state.json", "vocab.yaml", "topics.json")

_MANIFEST_FILENAME = "snapshot.json"
_SNAPSHOTS_DIRNAME = "snapshots"


class SnapshotManifest(BaseModel):
    """Metadata persisted alongside every snapshot as snapshot.json."""

    created_at: datetime
    command: str
    item_count: int
    topic_count: int
    vocab_size: int
    xbrain_version: str = "unknown"


def snapshots_dir(data_dir: Path) -> Path:
    """Return the snapshots directory under data/."""
    return data_dir / _SNAPSHOTS_DIRNAME


def snapshot_create(data_dir: Path, *, command: str, name: str | None = None) -> Path:
    """Create a snapshot directory inside data/snapshots/ and return its path.

    Directory naming:
      - if name is None: `<UTC-timestamp>-<command>` (command typically `manual` or `pre-<op>`)
      - if name is set:  `<UTC-timestamp>-<name>`

    Copies every existing artifact + writes snapshot.json. Raises if the
    snapshots dir is unwritable or any artifact copy fails — the caller is
    expected to let that propagate (a snapshot failure must abort the
    destructive op that triggered it).
    """
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    label = name if name is not None else command
    snapshot_dir = snapshots_dir(data_dir) / f"{timestamp}-{label}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    for artifact in _ARTIFACTS:
        src = data_dir / artifact
        if src.exists():
            shutil.copy2(src, snapshot_dir / artifact)

    manifest = SnapshotManifest(
        created_at=now,
        command=command,
        item_count=_count_items(data_dir),
        topic_count=_count_topics(data_dir),
        vocab_size=_count_vocab(data_dir),
        xbrain_version=_xbrain_version(),
    )
    _atomic_write(
        snapshot_dir / _MANIFEST_FILENAME,
        manifest.model_dump_json(indent=2),
    )
    return snapshot_dir


def snapshot_pre(data_dir: Path, *, command: str) -> Path:
    """Snapshot before a destructive op. `command` is the op name (e.g. `vocab-regenerate`).

    The directory is labelled `<ts>-pre-<command>` so the intent is obvious in `list`.
    """
    return snapshot_create(data_dir, command=f"pre-{command}")


def snapshot_list(data_dir: Path) -> list[tuple[Path, SnapshotManifest]]:
    """Return every snapshot under data/snapshots/, newest first.

    Snapshots without a readable manifest are skipped (defensive — never
    crash the list view because of one bad directory).
    """
    root = snapshots_dir(data_dir)
    if not root.exists():
        return []
    rows: list[tuple[Path, SnapshotManifest]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        manifest_path = entry / _MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            manifest = SnapshotManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            continue
        rows.append((entry, manifest))
    rows.sort(key=lambda row: row[1].created_at, reverse=True)
    return rows


def snapshot_show(data_dir: Path, name: str) -> tuple[Path, SnapshotManifest]:
    """Look up one snapshot by its directory name. Raises FileNotFoundError if missing."""
    snapshot_dir = snapshots_dir(data_dir) / name
    manifest_path = snapshot_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No snapshot named {name!r} under {snapshots_dir(data_dir)}")
    manifest = SnapshotManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return snapshot_dir, manifest


def snapshot_restore(data_dir: Path, name: str) -> None:
    """Replace data/<artifact> from the snapshot for each of the four artifacts.

    Artifacts present in the snapshot are copied over the live files (atomically
    via the existing _atomic_write pattern). Artifacts MISSING in the snapshot
    cause the live file to be deleted — restoring a pre-vocab snapshot to a
    repo that now has a vocab.yaml drops the live vocab.yaml.

    The snapshots/ subdir, auth/, the vault and any unrelated files are untouched.
    """
    snapshot_dir, _ = snapshot_show(data_dir, name)
    for artifact in _ARTIFACTS:
        src = snapshot_dir / artifact
        dst = data_dir / artifact
        if src.exists():
            _atomic_write(dst, src.read_text(encoding="utf-8"))
        elif dst.exists():
            dst.unlink()


def snapshot_prune(data_dir: Path, *, keep_last: int) -> int:
    """Delete all but the keep_last newest snapshots. Returns the count deleted.

    `keep_last` must be >= 0. A value of 0 prunes everything.
    """
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")
    rows = snapshot_list(data_dir)
    to_delete = rows[keep_last:]
    for snapshot_dir, _ in to_delete:
        shutil.rmtree(snapshot_dir)
    return len(to_delete)


# --------------------------------------------------------------------- internals


def _count_items(data_dir: Path) -> int:
    return _count_json_dict(data_dir / "items.json")


def _count_topics(data_dir: Path) -> int:
    return _count_json_dict(data_dir / "topics.json")


def _count_vocab(data_dir: Path) -> int:
    """Count topics in vocab.yaml. Mirrors load_vocab's shape contract."""
    path = data_dir / "vocab.yaml"
    if not path.exists():
        return 0
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return 0
    topics = data.get("topics", []) if isinstance(data, dict) else []
    return len(topics) if isinstance(topics, list) else 0


def _count_json_dict(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return 0
    return len(data) if isinstance(data, dict) else 0


def _xbrain_version() -> str:
    try:
        return metadata.version("xbrain")
    except metadata.PackageNotFoundError:
        return "unknown"
