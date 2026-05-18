"""Worksheet handoff for the manual / claude-code enrichment tracks.

`xbrain enrich --executor manual|claude-code` exports the pending items, the
vocabulary and the rubrics into one JSON worksheet. A human or a Claude Code
session fills the `judgments` array. `xbrain enrich --apply <file>` reads it
back. XBrain never handles a Claude OAuth token — it only moves JSON.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from xbrain.models import Item, Topic
from xbrain.rubrics import load_rubric


def _article_text(item: Item) -> str | None:
    if not item.content:
        return None
    for src in item.content.sources:
        if src.ok and src.text:
            return src.text[:4000]
    return None


def export_worksheet(items: list[Item], vocab: list[Topic], path: Path) -> None:
    """Write a worksheet with everything needed to enrich `items` by hand."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instructions": (
            "For each entry in `items`, append one object to `judgments` with "
            "keys {item_id, summary, primary_topic, topics}. Use only slugs from "
            "`vocab`. Then run: xbrain enrich --apply <this file>."
        ),
        "rubrics": {
            "summary": load_rubric("summary"),
            "topics": load_rubric("topics"),
        },
        "vocab": [t.model_dump() for t in vocab],
        "items": [
            {
                "item_id": it.id,
                "author": it.author.handle,
                "text": it.text,
                "bookmark_folder": it.bookmark_folder,
                "links": [{"url": ln.url, "domain": ln.domain}
                          for ln in it.links],
                "article": _article_text(it),
            }
            for it in items
        ],
        "judgments": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def import_worksheet(path: Path) -> list[dict]:
    """Read the `judgments` array from a filled worksheet."""
    if not path.exists():
        raise FileNotFoundError(f"Worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    judgments = data.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("worksheet `judgments` must be a list")
    return judgments
