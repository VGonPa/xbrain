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

from xbrain.executors.api import _content_image_descriptions, _video_frame_descriptions
from xbrain.models import ContentSourceSuccess, Item, Topic
from xbrain.rubrics import ARTICLE_CHAR_LIMIT, load_rubric


def _article_text(item: Item) -> str | None:
    """First successful fetched article body, truncated, or None.

    Only the success variant of `ContentSource` carries `text`. The
    isinstance narrowing satisfies mypy and silently skips broken-link
    sources, matching the pre-#20 ``src.ok and src.text`` behaviour.
    `x_video` sources are skipped — the transcript rides in its own
    `video_transcript` field so the enrich track sees it as a transcript,
    not an article (#44).
    """
    if not item.content:
        return None
    for src in item.content.sources:
        if isinstance(src, ContentSourceSuccess) and src.kind != "x_video" and src.text:
            return src.text[:ARTICLE_CHAR_LIMIT]
    return None


def _video_transcript(item: Item) -> str | None:
    """First with-speech `x_video` transcript body, FULL (untruncated), or None (#44).

    Unlike the `api` executor prompt — a bounded per-item model call that truncates
    to `TRANSCRIPT_CHAR_LIMIT` — the worksheet is judged by a full-context agent, so
    it carries the whole talk. The summary/topics are then not front-biased to the
    first ~13 min of a long video. A no-speech source (`has_speech=False`, empty
    text) yields None — no transcript to enrich from.
    """
    if not item.content:
        return None
    for src in item.content.sources:
        if (
            isinstance(src, ContentSourceSuccess)
            and src.kind == "x_video"
            and src.has_speech
            and src.text
        ):
            return src.text
    return None


def export_worksheet(
    items: list[Item],
    vocab: list[Topic],
    path: Path,
    executor: str,
    output_language: str,
) -> None:
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
            "`vocab`. An item may also carry `links`, `article`, `video_transcript` "
            "(the full talk — what the video SAYS), `video_frame_descriptions` (what "
            "the video SHOWS: slides/screens, present even when there is no "
            "transcript) and `image_descriptions` (content-bearing photos) — weigh "
            "them all as topic signal, not just `text`. Then run: xbrain enrich "
            "--apply <this file>."
        ),
        "rubrics": {
            "summary": load_rubric("summary", language=output_language),
            "topics": load_rubric("topics", language=output_language),
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
                "video_transcript": _video_transcript(it),
                # Content-bearing photo descriptions (#34): the same non-decorative
                # selection the `api` executor injects as its `Images in this post:`
                # section (`_content_image_descriptions`), so the manual/claude-code
                # enrich track sees the identical visual signal. Empty list when the
                # item has no described photos or only decorative ones.
                "image_descriptions": _content_image_descriptions(it),
                # Video key-frame descriptions: what the video SHOWS (slides/screens),
                # the same signal the `api` executor injects as its `Video frames`
                # section (`_video_frame_descriptions`). Present even for a mute
                # screen-share (no transcript). Empty list when the video has no
                # described frames. All frames are carried (agent-judged, full
                # context) — the `api` engine bounds its block instead.
                "video_frame_descriptions": _video_frame_descriptions(it),
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
    if not isinstance(data, dict):
        raise ValueError("worksheet must be a JSON object")
    judgments = data.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("worksheet `judgments` must be a list")
    executor = data.get("executor", "claude-code")
    return executor, judgments
