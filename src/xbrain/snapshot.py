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

import json
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml
from pydantic import BaseModel

from xbrain.store import _atomic_write

# The mutable artifacts that live in data/. Order matters for restore: items
# first (the source of truth), then derived stores.
_ARTIFACTS = ("items.json", "state.json", "vocab.yaml", "topics.json")

_MANIFEST_FILENAME = "snapshot.json"
_SNAPSHOTS_DIRNAME = "snapshots"

# Action codes returned by snapshot_restore for each artifact.
RESTORE_COPIED = "copied"
RESTORE_DELETED = "deleted"
RESTORE_SKIPPED = "skipped"


class SnapshotManifest(BaseModel):
    """Metadata persisted alongside every snapshot as snapshot.json.

    `command` records *what triggered this snapshot*: the destructive op name
    (e.g. `vocab-regenerate`) for auto-snapshots, `manual` for hand-taken ones.
    The `pre-` prefix lives only in the directory label, never in this field.
    """

    created_at: datetime
    command: str
    item_count: int
    topic_count: int
    vocab_size: int
    xbrain_version: str = "unknown"


def snapshots_dir(data_dir: Path) -> Path:
    """Return the snapshots directory under data/."""
    return data_dir / _SNAPSHOTS_DIRNAME


def snapshot_create(
    data_dir: Path,
    *,
    command: str,
    dir_label: str | None = None,
) -> tuple[Path, SnapshotManifest]:
    """Create a snapshot directory inside data/snapshots/.

    - `command` is recorded in the manifest as the destructive-op name (e.g.
      `vocab-regenerate`) or `manual` for hand-taken snapshots.
    - `dir_label` becomes the human-readable suffix of the directory name. When
      `None` it defaults to `command`. Auto-snapshots pass `pre-<command>` so
      the directory listing makes the intent visible at a glance.

    Counts are read from the live `data/` files BEFORE any copy happens — a
    corrupt artifact propagates as an exception (the destructive op that
    triggered the snapshot must abort, not proceed with a lying manifest).

    Returns the snapshot path and its parsed manifest.
    """
    # Read counts first — corrupt artifacts propagate here, before any copy.
    item_count = _count_json_dict(data_dir / "items.json")
    topic_count = _count_json_dict(data_dir / "topics.json")
    vocab_size = _count_vocab(data_dir / "vocab.yaml")

    now = datetime.now(timezone.utc)
    # Millisecond precision avoids same-second collisions in scripted use.
    timestamp = now.strftime("%Y-%m-%dT%H-%M-%S-") + f"{now.microsecond // 1000:03d}Z"
    label = dir_label if dir_label is not None else command
    snapshot_dir = snapshots_dir(data_dir) / f"{timestamp}-{label}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    for artifact in _ARTIFACTS:
        src = data_dir / artifact
        if src.exists():
            shutil.copy2(src, snapshot_dir / artifact)

    manifest = SnapshotManifest(
        created_at=now,
        command=command,
        item_count=item_count,
        topic_count=topic_count,
        vocab_size=vocab_size,
        xbrain_version=_xbrain_version(),
    )
    _atomic_write(
        snapshot_dir / _MANIFEST_FILENAME,
        manifest.model_dump_json(indent=2),
    )
    return snapshot_dir, manifest


def snapshot_list(data_dir: Path) -> list[tuple[Path, SnapshotManifest | None]]:
    """Return every snapshot under data/snapshots/, newest first.

    Snapshots whose manifest is missing or unreadable are returned with
    `manifest=None` and sorted to the end of the list — the caller decides
    how to surface the corruption. A directory we cannot read is data the
    user still needs to know exists.
    """
    root = snapshots_dir(data_dir)
    if not root.exists():
        return []
    rows: list[tuple[Path, SnapshotManifest | None]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        rows.append((entry, _load_manifest(entry / _MANIFEST_FILENAME)))
    # Corrupt entries (manifest=None) sort to the end via the sentinel epoch.
    sentinel = datetime.min.replace(tzinfo=timezone.utc)
    rows.sort(
        key=lambda row: row[1].created_at if row[1] is not None else sentinel,
        reverse=True,
    )
    return rows


def snapshot_show(data_dir: Path, name: str) -> tuple[Path, SnapshotManifest]:
    """Look up one snapshot by its directory name. Raises FileNotFoundError if missing."""
    snapshot_dir = snapshots_dir(data_dir) / name
    manifest_path = snapshot_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No snapshot named {name!r} under {snapshots_dir(data_dir)}")
    manifest = SnapshotManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return snapshot_dir, manifest


def snapshot_restore(data_dir: Path, name: str) -> list[tuple[str, str]]:
    """Restore data/ artifacts from the snapshot. Returns the per-artifact actions.

    For each of the four artifacts:
      - if the snapshot has it → live file is replaced via `shutil.copy2`
        (RESTORE_COPIED). Symmetric with `snapshot_create` and binary-safe.
      - if the snapshot lacks it but the live file exists → live file is
        deleted (RESTORE_DELETED) — restoring a pre-vocab snapshot to a repo
        that now has a vocab.yaml drops the live vocab.yaml.
      - if neither has it → noop (RESTORE_SKIPPED).

    Returns a list of (artifact, action) tuples so the CLI can echo every
    decision. The function itself is silent — no print, no logging.

    The snapshots/ subdir, auth/, the vault and any unrelated files are
    untouched.
    """
    snapshot_dir, _ = snapshot_show(data_dir, name)
    actions: list[tuple[str, str]] = []
    for artifact in _ARTIFACTS:
        src = snapshot_dir / artifact
        dst = data_dir / artifact
        if src.exists():
            shutil.copy2(src, dst)
            actions.append((artifact, RESTORE_COPIED))
        elif dst.exists():
            dst.unlink()
            actions.append((artifact, RESTORE_DELETED))
        else:
            actions.append((artifact, RESTORE_SKIPPED))
    return actions


def snapshot_prune(data_dir: Path, *, keep_last: int) -> int:
    """Delete all but the keep_last newest snapshots. Returns the count deleted.

    `keep_last` must be >= 0. A value of 0 prunes everything. Corrupt
    snapshots (manifest=None) sort to the end and are deleted first.
    """
    if keep_last < 0:
        raise ValueError(f"keep_last must be >= 0, got {keep_last}")
    rows = snapshot_list(data_dir)
    to_delete = rows[keep_last:]
    for snapshot_dir, _ in to_delete:
        shutil.rmtree(snapshot_dir)
    return len(to_delete)


# --------------------------------------------------------------------- internals


def _load_manifest(path: Path) -> SnapshotManifest | None:
    """Read a manifest from disk; return None if missing or unreadable."""
    if not path.exists():
        return None
    try:
        return SnapshotManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _count_json_dict(path: Path) -> int:
    """Return len(json.load(path)) for a top-level dict, or 0 if the file is absent.

    A corrupt JSON file propagates the exception — `snapshot_create` callers
    rely on this to abort the destructive op rather than persisting a manifest
    with a falsely-zero count.
    """
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, dict) else 0


def _count_vocab(path: Path) -> int:
    """Count topics in a vocab.yaml file. Mirrors load_vocab's shape contract.

    A corrupt YAML propagates the exception (same rationale as `_count_json_dict`).
    """
    if not path.exists():
        return 0
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    topics = data.get("topics", []) if isinstance(data, dict) else []
    return len(topics) if isinstance(topics, list) else 0


def _xbrain_version() -> str:
    """Return the installed xbrain version, or 'unknown' if not packaged."""
    try:
        return metadata.version("xbrain")
    except metadata.PackageNotFoundError:
        return "unknown"
