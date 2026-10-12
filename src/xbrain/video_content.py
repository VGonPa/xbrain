"""Helpers for consuming `x_video` summaries across enrichment and synthesis."""

from __future__ import annotations

from xbrain.models import ContentSourceSuccess
from xbrain.rubrics import truncate_transcript


def video_content_text(source: ContentSourceSuccess, limit: int | None = None) -> str | None:
    """Return the text signal an `x_video` source contributes downstream.

    For videos, `source.text` is the executive summary. The raw transcript lives
    in `source.raw_transcript` for reference/vault rendering only and must not
    feed enrich/topics/dashboard/Ask. Legacy frame descriptions are ignored by
    design: XBrain no longer stores or analyses visual video artifacts.
    """
    if source.kind != "x_video":
        return None
    text = source.text.strip()
    if not text:
        return None
    return truncate_transcript(text, limit) if limit is not None else text
