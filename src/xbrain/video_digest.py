"""The `video-digest` step: a long-form readable digest per `x_video` source.

Mirrors the `enrich` worksheet handoff (`worksheet.py`) but for a different
judgment shape: `{item_id, digest}`. `xbrain video-digest --executor manual|
claude-code` exports the pending videos + the rubric into one JSON worksheet; a
human or a Claude Code session fills the `judgments` array; `xbrain video-digest
--apply <file>` reads it back and writes each `digest` onto the item's `x_video`
source. XBrain never handles a Claude OAuth token — it only moves JSON.

The digest is what `generate` renders as the headline of the `## Video digest`
note section (PR-A), demoting the raw transcript + frames to collapsible evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from xbrain.executors.api import _video_frame_descriptions
from xbrain.models import ContentSourceSuccess, Item
from xbrain.rubrics import load_rubric
from xbrain.worksheet import _video_source, _video_transcript


def _has_digestible_content(source: ContentSourceSuccess) -> bool:
    """True when the `x_video` source carries content the worksheet will actually
    serialise to digest.

    Aligned with the exporter so selection never outruns the payload: the
    worksheet's transcript comes from `worksheet._video_transcript` (requires
    `has_speech` truthy + non-empty `text`) and its frames from
    `_video_frame_descriptions` (keeps only frames with a NON-EMPTY description).
    A looser predicate (e.g. counting frames whose descriptions are all empty)
    would export an item with nothing to digest and re-select it every run,
    forever. A silent video with no described frames has nothing to synthesise.
    """
    has_transcript = bool(source.has_speech) and bool(source.text)
    has_frame_descriptions = any(frame.description for frame in source.frames)
    return has_transcript or has_frame_descriptions


def items_pending_video_digest(store: dict[str, Item]) -> list[Item]:
    """Items whose `x_video` source has content but no digest yet.

    Selection is `has content AND empty digest`. Re-digesting a video whose
    transcript later changed is out of scope here (there is no digest timestamp);
    clear the `digest` field (or a future `--regenerate`) to force one.
    """
    pending: list[Item] = []
    for item in store.values():
        source = _video_source(item)
        if source is None or not _has_digestible_content(source):
            continue
        if not source.digest:
            pending.append(item)
    return pending


def export_video_digest_worksheet(
    items: list[Item],
    path: Path,
    executor: str,
    output_language: str,
) -> None:
    """Write a worksheet with everything needed to digest `items` by hand.

    Each entry carries the FULL transcript (what the video says) + the frame
    descriptions (what it shows) + the tweet text — the same visual/spoken signal
    the digest must synthesise. `executor` is recorded so `--apply` attributes the
    judgments to the track that produced the worksheet.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "executor": executor,
        "instructions": (
            "For each entry in `items`, append one object to `judgments` with keys "
            "{item_id, digest}. Write the `digest` following `rubric` — grounded in "
            "`video_transcript` (what the video says) and `video_frame_descriptions` "
            "(what it shows), NOT the tweet `text`/caption. Then run: xbrain "
            "video-digest --apply <this file>."
        ),
        "rubric": load_rubric("video-digest", language=output_language),
        "items": [
            {
                "item_id": item.id,
                "author": item.author.handle,
                # Evidence, not decoration: the rubric promises the judge "@handle +
                # display name", so the generator must be handed it too (D6).
                "author_name": item.author.name,
                "text": item.text,
                "title": (source.title if (source := _video_source(item)) else None),
                "video_transcript": _video_transcript(item),
                "video_frame_descriptions": _video_frame_descriptions(item),
            }
            for item in items
        ],
        "judgments": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_video_digest_worksheet(path: Path) -> list[dict]:
    """Read the `judgments` list from a filled worksheet."""
    if not path.exists():
        raise FileNotFoundError(f"Worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("worksheet must be a JSON object")
    judgments = data.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("worksheet `judgments` must be a list")
    return judgments


def apply_video_digest_judgments(
    store: dict[str, Item], judgments: list[dict]
) -> tuple[int, list[tuple[str, list[str]]]]:
    """Write each `{item_id, digest}` onto its item's `x_video` source.

    Returns `(applied_count, invalid)` where `invalid` is `(item_id, errors)`.
    An unknown item, an item without an `x_video` source, or an empty/non-string
    digest is rejected (recorded in `invalid`), never silently dropped.
    """
    applied = 0
    invalid: list[tuple[str, list[str]]] = []
    for judgment in judgments:
        item_id = str(judgment.get("item_id"))
        digest = judgment.get("digest")
        item = store.get(item_id)
        if item is None:
            invalid.append((item_id, ["unknown item id"]))
            continue
        source = _video_source(item)
        if source is None:
            invalid.append((item_id, ["item has no x_video source"]))
            continue
        if not isinstance(digest, str) or not digest.strip():
            invalid.append((item_id, ["digest is empty or not a string"]))
            continue
        source.digest = digest.strip()
        applied += 1
    return applied, invalid
