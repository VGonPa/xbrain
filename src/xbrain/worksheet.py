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

from xbrain.executors.api import (
    QUOTED_CONTENT_UNFETCHED_NOTE,
    _content_image_descriptions,
    _video_frame_descriptions,
    first_source_text,
    quoted_content_unfetched,
    thread_text,
    unfetched_links_note,
)
from xbrain.models import LINK_CONTENT_KINDS, ContentSourceSuccess, Item, Topic
from xbrain.rubrics import load_rubric


def _link_content_source(item: Item) -> ContentSourceSuccess | None:
    """First successfully-fetched LINKED page source (`LINK_CONTENT_KINDS`), or None.

    Only the success variant of `ContentSource` carries `text`. The isinstance
    narrowing satisfies mypy and silently skips broken-link sources, matching the
    pre-#20 ``src.ok and src.text`` behaviour. Only a linked page counts: an
    `x_video` transcript rides in `video_transcript`, a `thread` in `thread` and a
    quoted post in its own marker — none of them is a fetched article, and serving
    one as `article` would tell the reader (agent or judge) a link was downloaded
    when none was.
    """
    if not item.content:
        return None
    for src in item.content.sources:
        if isinstance(src, ContentSourceSuccess) and src.kind in LINK_CONTENT_KINDS and src.text:
            return src
    return None


def _article_title(item: Item) -> str | None:
    """The FETCHED linked article's title, or None.

    The summary rubric declares "the fetched article body and its title" as evidence, the
    api prompt ships it and the judge's `_source_text` ships it — but the worksheet track,
    the one that actually runs, shipped the body alone. The rubric was naming a surface the
    running generator never received. Cost, measured: 8 summaries flagged ungrounded for a
    name that sits in the article's TITLE.
    """
    if not item.content:
        return None
    for source in item.content.sources:
        if isinstance(source, ContentSourceSuccess) and source.kind in LINK_CONTENT_KINDS:
            return source.title
    return None


def _video_title(item: Item) -> str | None:
    """The `x_video` source's title, or None.

    The judge's `_source_text` carries `[Video title]` (#86), and the summary rubric
    admits it as evidence — so the generator must be handed it too, or the rubric names a
    surface the generator was never given. Defined locally: importing `video_digest`
    would be circular (it imports `_video_transcript` from here).
    """
    if not item.content:
        return None
    for source in item.content.sources:
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_video":
            return source.title
    return None


def _article_text(item: Item) -> str | None:
    """First successfully-fetched LINKED article body, truncated, or None."""
    return first_source_text(item, LINK_CONTENT_KINDS)


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
            "`vocab`. An item may also carry `links`, `article` (a FETCHED linked "
            "page), `thread` (the poster's own full thread text — not a linked "
            "page), `video_transcript` (the full talk — what the video SAYS), "
            "`video_frame_descriptions` (what the video SHOWS: slides/screens, "
            "present even when there is no transcript) and `image_descriptions` "
            "(content-bearing photos) — weigh them all as topic signal, not just "
            "`text`. `unfetched_links_note` / `quoted_content_note` mark content "
            "that was NEVER downloaded: obey them — never describe or guess it. "
            "Name no entity that none of these surfaces names: `links` carry a URL "
            "and a domain, which are topic signal only — never a name, never content. "
            "Then run: xbrain enrich --apply <this file>."
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
                # The display name travels WITH the handle: the summary rubric admits the
                # author metadata as evidence for WHO POSTED, and the judge's
                # `_source_text` holds `@handle (Display Name)` since #86. Handle-only
                # would have the generator de-abbreviating `@lexfridman` into a name it
                # was never shown, while the judge checks the name it WAS shown.
                "author_name": it.author.name,
                "text": it.text,
                "bookmark_folder": it.bookmark_folder,
                "links": [{"url": ln.url, "domain": ln.domain} for ln in it.links],
                "article": _article_text(it),
                "article_title": _article_title(it),
                # The poster's OWN expanded thread — real signal, but NOT a fetched
                # linked page. It ships in its own field so the agent never reads the
                # author's own words as the body of an article nobody downloaded.
                "thread": thread_text(it),
                # Guardrail: when the item links out and some link's content is
                # missing, say so explicitly (with counts on a partial fetch) — the
                # agent must not reconstruct the linked content from the URL/domain
                # (see `links_content_unfetched`).
                "unfetched_links_note": unfetched_links_note(it),
                # Guardrail: `quoted_id` is captured but no fetcher downloads the
                # quoted post, so the shared content is NOT in this worksheet. Without
                # this note the summary rubric's "summarise the shared content" rule
                # would be an order to invent it.
                "quoted_content_note": (
                    QUOTED_CONTENT_UNFETCHED_NOTE if quoted_content_unfetched(it) else None
                ),
                # The judge sees `[Video title]` (#86); the generator must hold the same
                # surface, or the rubric's evidence set is a promise we do not keep.
                "video_title": _video_title(it),
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
