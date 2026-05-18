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
from xbrain.rubrics import ARTICLE_CHAR_LIMIT, load_rubric


def _article_text(item: Item) -> str | None:
    if not item.content:
        return None
    for src in item.content.sources:
        if src.ok and src.text:
            return src.text[:ARTICLE_CHAR_LIMIT]
    return None


def export_worksheet(items: list[Item], vocab: list[Topic], path: Path, executor: str) -> None:
    """Write a worksheet with everything needed to enrich `items` by hand.

    `executor` is recorded in the payload so `--apply` attributes the
    judgments to the track that produced the worksheet.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "executor": executor,
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
                "links": [{"url": ln.url, "domain": ln.domain} for ln in it.links],
                "article": _article_text(it),
            }
            for it in items
        ],
        "judgments": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_worksheet(path: Path) -> tuple[str, list[dict]]:
    """Read `(executor, judgments)` from a filled worksheet.

    `executor` falls back to ``"claude-code"`` for worksheets written before
    the executor key existed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    judgments = data.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("worksheet `judgments` must be a list")
    executor = data.get("executor", "claude-code")
    return executor, judgments
