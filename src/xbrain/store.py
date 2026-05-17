"""JSON-backed store for XBrain items and extraction state."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from xbrain.models import Item, State


def load_store(path: Path) -> dict[str, Item]:
    """Load the item store; an empty dict if the file does not exist."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {item_id: Item.model_validate(data) for item_id, data in raw.items()}


def save_store(store: dict[str, Item], path: Path) -> None:
    """Persist the item store as pretty, sorted, UTF-8 JSON (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {item_id: item.model_dump(mode="json") for item_id, item in store.items()}
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    _atomic_write(path, text)


def merge_items(store: dict[str, Item], new_items: list[Item]) -> int:
    """Add items not already present; never overwrite. Returns count added."""
    added = 0
    for item in new_items:
        if item.id not in store:
            store[item.id] = item
            added += 1
    return added


def load_state(path: Path) -> State:
    """Load extraction state; a fresh State if the file does not exist."""
    if not path.exists():
        return State()
    return State.model_validate_json(path.read_text(encoding="utf-8"))


def save_state(state: State, path: Path) -> None:
    """Persist extraction state as JSON (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, state.model_dump_json(indent=2))


def _atomic_write(path: Path, text: str) -> None:
    """Write to a temp file in the same directory, then atomically rename."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
