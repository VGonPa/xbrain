"""Build and render the self-contained interactive HTML metrics dashboard.

`compute_dashboard_data` is pure — it turns the store (plus topic overviews and
an id→note map) into the JSON blob the vendored template consumes.
`collect_thumbnails` does the photo file IO. `render_dashboard_html` injects the
blob and the vendored ECharts library into the template. `generate` wires these
together and writes `<output_dir>/dashboard.html`; nothing here touches a browser.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
from collections import Counter, defaultdict
from itertools import combinations
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from xbrain.content_policy import has_video
from xbrain.models import (
    ArticleImageBlock,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
    TopicPage,
)

logger = logging.getLogger(__name__)

# Slug words rendered upper-case (acronyms) or as an ampersand in topic labels.
_ACRONYMS = frozenset({"ai", "ml", "llm", "mcp", "api", "ux", "ui", "gpu", "seo", "vc", "3d", "x"})
_THUMB_LIMIT = 18
# The three video media variants share `type="video"`, `thumbnail_url` and
# `duration_millis`; discriminating on the union (not a `getattr("type")`
# sniff) keeps the typed access mypy-checked, matching `generate._render_media_lines`.
_VIDEO_TYPES = (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)
# A described photo (`MediaPhotoDescribed`) IS a downloaded photo — it carries
# the same `local_path`/bytes and only adds a vision caption. `xbrain describe`
# transitions Downloaded -> Described in place, so every downloaded-photo count,
# thumbnail source, and photo-post filter must accept BOTH variants or the whole
# corpus of described photos vanishes from the dashboard. Mirrors the isinstance
# grouping in `generate._render_media_lines`.
_DOWNLOADED_PHOTO_TYPES = (MediaPhotoDownloaded, MediaPhotoDescribed)


def _article_image_blocks(item: Item) -> list[ArticleImageBlock]:
    """Return inline image blocks from X Articles on an item."""
    if item.content is None:
        return []
    return [
        block
        for source in item.content.sources
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_article"
        for block in source.blocks
        if isinstance(block, ArticleImageBlock)
    ]


def _downloaded_image_entries(
    item: Item,
) -> list[tuple[MediaPhotoDownloaded | MediaPhotoDescribed, str]]:
    """Downloaded images for thumbnailing: post photos first, then article images."""
    entries: list[tuple[MediaPhotoDownloaded | MediaPhotoDescribed, str]] = [
        (entry, "post") for entry in item.media if isinstance(entry, _DOWNLOADED_PHOTO_TYPES)
    ]
    entries.extend(
        (block.media, "article")
        for block in _article_image_blocks(item)
        if isinstance(block.media, _DOWNLOADED_PHOTO_TYPES)
    )
    return entries


def _article_media_counts(item: Item) -> tuple[int, int, int]:
    """Return downloaded, pending and undescribed counts for inline Article images."""
    downloaded = pending = undescribed = 0
    for block in _article_image_blocks(item):
        entry = block.media
        if isinstance(entry, _DOWNLOADED_PHOTO_TYPES):
            downloaded += 1
        if isinstance(entry, MediaPhotoPending):
            pending += 1
        if isinstance(entry, MediaPhotoDownloaded):
            undescribed += 1
    return downloaded, pending, undescribed


def _article_image_rows(item: Item) -> list[dict[str, Any]]:
    """Downloaded inline Article images exposed to document detail views.

    Keep the dashboard payload lightweight: image bytes remain in the canonical
    ``_media`` tree and the UI receives only the safe relative path plus the
    already-persisted description metadata.
    """
    rows: list[dict[str, Any]] = []
    for index, block in enumerate(_article_image_blocks(item), start=1):
        entry = block.media
        if not isinstance(entry, _DOWNLOADED_PHOTO_TYPES):
            continue
        description = entry.description.strip() if isinstance(entry, MediaPhotoDescribed) else ""
        alt = (block.alt or "").strip()
        rows.append(
            {
                "path": f"_media/{entry.local_path}",
                "description": description or alt,
                "alt": alt or f"Article image {index}",
                "width": entry.width,
                "height": entry.height,
                "decorative": isinstance(entry, MediaPhotoDescribed) and entry.is_decorative,
            }
        )
    return rows


def humanize_topic(slug: str) -> str:
    """Turn a topic slug into a display label (``ai-coding`` → ``AI Coding``)."""
    out: list[str] = []
    for word in slug.split("-"):
        if word in _ACRONYMS:
            out.append(word.upper())
        elif word == "and":
            out.append("&")
        else:
            out.append(word.capitalize())
    return " ".join(out)


def _summary(item: Item) -> str:
    """The item's Spanish enrichment summary, or a text fallback."""
    if item.enriched and item.enriched.summary:
        return item.enriched.summary
    return item.text[:200]


def _date(item: Item) -> str:
    """Library acquisition date kept under the legacy row key ``date``."""
    return item.captured_at.date().isoformat()


def _month(item: Item) -> str:
    """Library acquisition month, not the source publication month."""
    return item.captured_at.strftime("%Y-%m")


def _published_date(item: Item) -> str:
    return item.created_at.date().isoformat()


def _primary(item: Item) -> str | None:
    return item.enriched.primary_topic if item.enriched else None


def _confidence(item: Item) -> str | None:
    return item.enriched.topic_confidence if item.enriched else None


def _suggested_topics(item: Item) -> list[str]:
    return item.enriched.suggested_new_topics if item.enriched else []


def _topics(item: Item) -> list[str]:
    """Unique topic slugs assigned to an item, primary first when present."""
    if item.enriched is None:
        return []
    topics = [topic for topic in [item.enriched.primary_topic, *item.enriched.topics] if topic]
    return list(dict.fromkeys(topics))


def _recent(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The `n` most recent rows by their ``date`` field."""
    return sorted(rows, key=lambda r: r["date"], reverse=True)[:n]


def _row(
    item: Item,
    id2note: dict[str, str],
    slug2label: dict[str, str],
    item_storage: dict[str, dict[str, Any]],
    *,
    include_article_images: bool = False,
) -> dict[str, Any]:
    """A drill-down post row: who/when/what plus deep links to X and the vault."""
    topic = _primary(item)
    suggestions = _suggested_topics(item)
    row: dict[str, Any] = {
        "id": item.id,
        "title": _graph_item_label(item),
        "handle": item.author.handle,
        "name": item.author.name,
        "source": item.source,
        "content_type": _content_type(item),
        "date": _date(item),
        "captured_date": _date(item),
        "published_date": _published_date(item),
        "topic": slug2label.get(topic or "", topic or "—"),
        "topic_slug": topic,
        "topics": _topics(item),
        "topic_labels": [humanize_topic(slug) for slug in _topics(item)],
        "confidence": _confidence(item) or "unknown",
        "suggested_topics": suggestions,
        "needs_review": _needs_topic_review(item),
        "processing_ready": _processing_ready(item),
        "summary": _summary(item),
        "post_text": _compact_text(item.text, 600),
        "domain": _graph_domain(item),
        "word_count": _item_word_count(item),
        "article_sources": _item_article_source_counts(item),
        "media_counts": _item_media_counts(item),
        "url": item.url,
        "note": id2note.get(item.id),
        "storage": item_storage.get(item.id),
    }
    if include_article_images:
        row["article_images"] = _article_image_rows(item)
    return row


def _article_sources(item: Item) -> list[ContentSourceSuccess | ContentSourceFailure]:
    """Article-like fetched sources attached to an item."""
    if item.content is None:
        return []
    return [
        source
        for source in item.content.sources
        if source.kind in ("external_article", "x_article")
        and isinstance(source, (ContentSourceSuccess, ContentSourceFailure))
    ]


def _saved_article_sources(item: Item) -> list[ContentSourceSuccess]:
    return [source for source in _article_sources(item) if isinstance(source, ContentSourceSuccess)]


def _failed_article_sources(item: Item) -> list[ContentSourceFailure]:
    return [source for source in _article_sources(item) if isinstance(source, ContentSourceFailure)]


def _item_article_source_counts(item: Item) -> dict[str, int]:
    counts = {"ext_saved": 0, "ext_failed": 0, "x_saved": 0, "x_failed": 0}
    for source in _article_sources(item):
        prefix = "ext" if source.kind == "external_article" else "x"
        outcome = "saved" if isinstance(source, ContentSourceSuccess) else "failed"
        counts[f"{prefix}_{outcome}"] += 1
    return counts


def _item_media_counts(item: Item) -> dict[str, int]:
    article_downloaded, article_pending, _ = _article_media_counts(item)
    return {
        "photos_downloaded": sum(
            1 for entry in item.media if isinstance(entry, _DOWNLOADED_PHOTO_TYPES)
        ),
        "photos_pending": sum(1 for entry in item.media if isinstance(entry, MediaPhotoPending)),
        "article_images_downloaded": article_downloaded,
        "article_images_pending": article_pending,
        "videos": sum(1 for entry in item.media if isinstance(entry, _VIDEO_TYPES)),
    }


def _has_processed_content(item: Item) -> bool:
    """True when an item already has useful generated content for the note."""
    if item.enriched is not None:
        return True
    if item.content is None:
        return False
    return any(isinstance(source, ContentSourceSuccess) for source in item.content.sources)


def _blocking_article_failures(item: Item) -> list[ContentSourceFailure]:
    """Article failures that still leave the item without useful processing."""
    if _has_processed_content(item):
        return []
    return _failed_article_sources(item)


def _item_word_count(item: Item) -> int:
    """Approximate readable words across retained successful content."""
    sources = item.content.sources if item.content else []
    source_words = sum(
        len(source.text.split()) for source in sources if isinstance(source, ContentSourceSuccess)
    )
    return source_words or len(item.text.split())


def _content_type(item: Item) -> str:
    """Dashboard content bucket: article, video, article_failed or post_only."""
    if _saved_article_sources(item):
        return "article"
    if any(
        isinstance(source, ContentSourceSuccess) and source.kind == "x_video"
        for source in (item.content.sources if item.content else [])
    ) or has_video(item):
        return "video"
    if _blocking_article_failures(item):
        return "article_failed"
    return "post_only"


def collect_thumbnails(
    items: list[Item], media_root: Path | None, id2note: dict[str, str], limit: int = _THUMB_LIMIT
) -> list[dict[str, Any]]:
    """Base64-encode a sample of downloaded images with their post metadata.

    Reads at most `limit` downloaded images from `media_root`, downscaling each
    to a small JPEG data URI so the dashboard stays self-contained. Returns an
    empty list when `media_root` is None. Unreadable files are skipped.
    """
    if media_root is None:
        return []
    thumbs: list[dict[str, Any]] = []
    for item in items:
        for entry, location in _downloaded_image_entries(item):
            if len(thumbs) >= limit:
                return thumbs
            path = media_root / entry.local_path
            if not path.exists():
                continue
            try:
                with Image.open(path) as opened:
                    rgb = opened.convert("RGB")
                rgb.thumbnail((200, 200))
                buffer = io.BytesIO()
                rgb.save(buffer, "JPEG", quality=78)
            except Exception:  # noqa: BLE001 - a bad image file must not break the dashboard
                logger.debug("Skipping unreadable thumbnail %s", path, exc_info=True)
                continue
            # Surface the vision caption on the thumbnail so the photo drawer says
            # what each image actually is. Only `MediaPhotoDescribed` carries a
            # caption, and the model validator forces a decorative one to "" —
            # so a single typed isinstance suffices (a plain `MediaPhotoDownloaded`
            # yields ""). Typed access, not a `getattr` sniff, per the `_VIDEO_TYPES`
            # note above.
            desc = entry.description if isinstance(entry, MediaPhotoDescribed) else ""
            thumbs.append(
                {
                    "thumb": "data:image/jpeg;base64,"
                    + base64.b64encode(buffer.getvalue()).decode(),
                    "url": item.url,
                    "note": id2note.get(item.id),
                    "handle": item.author.handle,
                    "date": _date(item),
                    "summary": _summary(item),
                    "desc": desc,
                    "kind": location,
                }
            )
    return thumbs


def _growth(items: list[Item]) -> dict[str, Any]:
    """Monthly acquisitions and cumulative totals, including zero-count months."""
    per_month: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        per_month[_month(item)].append(item)
    populated_months = sorted(per_month)
    months = _month_range(populated_months[0], populated_months[-1]) if populated_months else []
    new_total, cum_total, cum_bm, cum_own = [], [], [], []
    run_t = run_b = run_o = 0
    for month in months:
        group = per_month[month]
        run_t += len(group)
        run_b += sum(1 for i in group if i.source == "bookmark")
        run_o += sum(1 for i in group if i.source == "own_tweet")
        new_total.append(len(group))
        cum_total.append(run_t)
        cum_bm.append(run_b)
        cum_own.append(run_o)
    return {
        "months": months,
        "new_total": new_total,
        "cum_total": cum_total,
        "cum_bm": cum_bm,
        "cum_own": cum_own,
        "_per_month": per_month,
    }


def _month_range(first: str, last: str) -> list[str]:
    """Return every calendar month in an inclusive ``YYYY-MM`` interval."""
    year, month = map(int, first.split("-"))
    last_year, last_month = map(int, last.split("-"))
    out: list[str] = []
    while (year, month) <= (last_year, last_month):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out


def _longform(
    items: list[Item], id2note: dict[str, str], item_storage: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Captured long-form counts (external vs X) plus a recent article list."""
    counts = {"ext_saved": 0, "ext_failed": 0, "x_saved": 0, "x_failed": 0}
    articles: list[dict[str, Any]] = []
    for item in items:
        for source in _article_sources(item):
            prefix = "ext" if source.kind == "external_article" else "x"
            if isinstance(source, ContentSourceSuccess):
                counts[f"{prefix}_saved"] += 1
                articles.append(
                    {
                        "title": source.title or source.url[:80],
                        "id": item.id,
                        "url": source.url,
                        "source": "External" if source.kind == "external_article" else "X Article",
                        "handle": item.author.handle,
                        "date": _date(item),
                        "summary": _summary(item),
                        "topic": humanize_topic(_primary(item) or "misc"),
                        "confidence": _confidence(item) or "unknown",
                        "suggested_topics": _suggested_topics(item),
                        "post": item.url,
                        "note": id2note.get(item.id),
                        "storage": item_storage.get(item.id),
                    }
                )
            else:
                counts[f"{prefix}_failed"] += 1
    saved = counts["ext_saved"] + counts["x_saved"]
    total = saved + counts["ext_failed"] + counts["x_failed"]
    return {
        **counts,
        "saved": saved,
        "total": total,
        "saved_pct": round(saved / total * 100, 1) if total else 0.0,
        "items": _recent(articles, 60),
    }


def _article_failures(items: list[Item], id2note: dict[str, str]) -> list[dict[str, Any]]:
    """Recent article failures, whether or not another source was processed."""
    rows: list[dict[str, Any]] = []
    for item in items:
        blocking_sources = set(id(source) for source in _blocking_article_failures(item))
        for source in _failed_article_sources(item):
            rows.append(
                {
                    "handle": item.author.handle,
                    "id": item.id,
                    "date": _date(item),
                    "kind": source.kind,
                    "reason": source.failure_reason,
                    "url": source.url,
                    "post": item.url,
                    "note": id2note.get(item.id),
                    "summary": _summary(item),
                    "blocking": id(source) in blocking_sources,
                }
            )
    return _recent(rows, 20)


def _article_failure_count(items: list[Item]) -> int:
    """Total failed linked article sources, including non-blocking failures."""
    return sum(len(_failed_article_sources(item)) for item in items)


def _blocking_article_failure_count(items: list[Item]) -> int:
    return sum(len(_blocking_article_failures(item)) for item in items)


def _failed_bookmark_count(items: list[Item]) -> int:
    """Bookmarks containing any article failure, matching ``retry-failed``."""
    return sum(1 for item in items if item.source == "bookmark" and _failed_article_sources(item))


def _media_counts(items: list[Item]) -> dict[str, int]:
    """Downloaded/pending photo counts and captured-video count across the store."""
    downloaded = pending = videos = article_downloaded = article_pending = 0
    for item in items:
        for entry in item.media:
            if isinstance(entry, _DOWNLOADED_PHOTO_TYPES):
                downloaded += 1
            elif isinstance(entry, MediaPhotoPending):
                pending += 1
            elif isinstance(entry, _VIDEO_TYPES):
                videos += 1
        item_article_downloaded, item_article_pending, _ = _article_media_counts(item)
        article_downloaded += item_article_downloaded
        article_pending += item_article_pending
    return {
        "photos_downloaded": downloaded,
        "photos_pending": pending,
        "article_images_downloaded": article_downloaded,
        "article_images_pending": article_pending,
        "videos": videos,
    }


def _ops(items: list[Item], id2note: dict[str, str], rows: _Rows) -> dict[str, Any]:
    """Mobile-first operational status for the ingestion workflow."""
    downloaded_undescribed = sum(
        1 for item in items for entry in item.media if isinstance(entry, MediaPhotoDownloaded)
    )
    article_pending = article_undescribed = 0
    for item in items:
        _, item_article_pending, item_article_undescribed = _article_media_counts(item)
        article_pending += item_article_pending
        article_undescribed += item_article_undescribed
    pending_fetch = sum(1 for item in items if item.links and item.content is None)
    pending_enrich = sum(1 for item in items if item.enriched is None)
    failures = _article_failures(items, id2note)
    failure_count = _article_failure_count(items)
    blocking_failure_count = _blocking_article_failure_count(items)
    failed_bookmarks = _failed_bookmark_count(items)
    recent_bookmarks = _recent(rows([item for item in items if item.source == "bookmark"]), 8)
    return {
        "command": "uv run xbrain refresh-all --headless --video-max-size 4GB",
        "retry_failed_command": "uv run xbrain retry-failed --source bookmarks --headless",
        "serve_command": "uv run xbrain serve-dashboard --host 127.0.0.1 --port 8765",
        "retry_failed_bookmarks": failed_bookmarks,
        "pending": {
            "fetch": pending_fetch,
            "media": sum(
                1 for item in items for entry in item.media if isinstance(entry, MediaPhotoPending)
            )
            + article_pending,
            "describe": downloaded_undescribed + article_undescribed,
            "enrich": pending_enrich,
            "article_failures": failure_count,
            "blocking_article_failures": blocking_failure_count,
        },
        "recent_bookmarks": recent_bookmarks,
        "article_failures": failures,
    }


def _videos(
    items: list[Item], id2note: dict[str, str], item_storage: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """A recent sample of video posts with poster, duration and deep links."""
    rows: list[dict[str, Any]] = []
    for item in items:
        for entry in item.media:
            if not isinstance(entry, _VIDEO_TYPES):
                continue
            dur = entry.duration_millis
            rows.append(
                {
                    "id": item.id,
                    "handle": item.author.handle,
                    "date": _date(item),
                    "summary": _summary(item),
                    "dur": round(dur / 1000) if dur else None,
                    "poster": entry.thumbnail_url,
                    "url": item.url,
                    "note": id2note.get(item.id),
                    "storage": item_storage.get(item.id),
                }
            )
            break
    return _recent(rows, 12)


_Rows = Callable[[list[Item]], list[dict[str, Any]]]


def _primaries(items: list[Item]) -> list[str]:
    """The primary-topic slugs of the enriched items (drops items without one)."""
    return [p for item in items if (p := _primary(item)) is not None]


def _topics_section(
    items: list[Item], topic_freq: "Counter[str]", topic_pages: dict[str, TopicPage], rows: _Rows
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The topics bar chart data plus per-topic drill-down (overview + posts)."""
    topics_sorted = [
        {"slug": s, "label": humanize_topic(s), "count": c} for s, c in topic_freq.most_common()
    ]
    by_topic: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if (p := _primary(item)) is not None:
            by_topic[p].append(item)
    topic_data = {
        s: {
            "label": humanize_topic(s),
            "count": topic_freq[s],
            "overview": topic_pages[s].overview if s in topic_pages else "",
            "samples": _recent(rows(by_topic[s]), 5),
        }
        for s in topic_freq
    }
    return topics_sorted, topic_data


def _authors_section(items: list[Item], rows: _Rows) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Top bookmarked authors plus per-author drill-down."""
    by_author: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        if item.source == "bookmark":
            by_author[item.author.handle].append(item)
    top = Counter({h: len(v) for h, v in by_author.items()}).most_common(10)
    authors = [{"handle": h, "name": by_author[h][0].author.name, "count": c} for h, c in top]
    author_data = {
        h: {
            "name": by_author[h][0].author.name,
            "count": len(by_author[h]),
            "samples": _recent(rows(by_author[h]), 8),
        }
        for h, _ in top
    }
    return authors, author_data


def _domains_section(items: list[Item], rows: _Rows) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Top linked domains (x.com excluded) plus per-domain drill-down."""
    by_domain: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        for link in item.links:
            by_domain[link.domain].append(item)
    top = Counter({d: len(v) for d, v in by_domain.items() if d != "x.com"}).most_common(10)
    domains = [{"domain": d, "count": c} for d, c in top]
    domain_data = {d: {"count": c, "samples": _recent(rows(by_domain[d]), 7)} for d, c in top}
    return domains, domain_data


def _months_section(per_month: dict[str, list[Item]], rows: _Rows) -> dict[str, Any]:
    """Per-month drill-down: counts, top topics, top authors and sample posts."""
    out: dict[str, Any] = {}
    for month, group in per_month.items():
        tt = Counter(_primaries(group))
        ta = Counter(i.author.handle for i in group if i.source == "bookmark")
        out[month] = {
            "count": len(group),
            "bm": sum(1 for i in group if i.source == "bookmark"),
            "own": sum(1 for i in group if i.source == "own_tweet"),
            "top_topics": [{"label": humanize_topic(s), "count": c} for s, c in tt.most_common(6)],
            "top_authors": [{"handle": h, "count": c} for h, c in ta.most_common(6)],
            "samples": _recent(rows(group), 6),
        }
    return out


_META_LONGFORM_KEYS = (
    "ext_saved",
    "ext_failed",
    "x_saved",
    "x_failed",
    "saved",
    "total",
    "saved_pct",
)


def _library_section(items: list[Item]) -> dict[str, Any]:
    """Counts by retained content type for the article/video-first product."""
    counts = Counter(_content_type(item) for item in items)
    return {
        "articles": counts["article"],
        "videos": counts["video"],
        "article_failed": counts["article_failed"],
        "source_failures": _article_failure_count(items),
        "blocking_failures": _blocking_article_failure_count(items),
        "post_only": counts["post_only"],
    }


def _taxonomy_section(items: list[Item], rows: _Rows) -> dict[str, Any]:
    """Topic-assignment quality signals exposed in the dashboard."""
    confidence = Counter(_confidence(item) or "unknown" for item in items)
    misc_items = [item for item in items if _primary(item) == "misc"]
    low_items = [item for item in items if _confidence(item) == "low"]
    suggested = Counter(suggestion for item in items for suggestion in _suggested_topics(item))
    review_items = {
        item.id: item
        for item in sorted(
            [item for item in items if _needs_topic_review(item)],
            key=lambda i: i.captured_at,
            reverse=True,
        )
    }
    return {
        "confidence": {
            "high": confidence["high"],
            "medium": confidence["medium"],
            "low": confidence["low"],
            "unknown": confidence["unknown"],
        },
        "misc": len(misc_items),
        "low": len(low_items),
        "review_count": len(review_items),
        "suggestion_assignments": sum(suggested.values()),
        "suggestion_unique": len(suggested),
        "suggested": [{"slug": slug, "count": count} for slug, count in suggested.most_common(20)],
        "review_items": rows(list(review_items.values()))[:12],
    }


def _needs_topic_review(item: Item) -> bool:
    enrichment = item.enriched
    if enrichment is None:
        return False
    if enrichment.topic_confidence in (None, "low"):
        return True
    if enrichment.suggested_new_topics:
        return True
    return (enrichment.primary_topic == "misc" or "misc" in enrichment.topics) and (
        enrichment.topic_confidence != "high"
    )


def _compact_text(value: str | None, limit: int) -> str:
    text = " ".join((value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _graph_domain(item: Item) -> str | None:
    for link in item.links:
        if link.domain != "x.com":
            return link.domain
    return item.links[0].domain if item.links else None


def _graph_item_label(item: Item) -> str:
    for source in _saved_article_sources(item):
        if source.title:
            return _compact_text(source.title, 74)
    return _compact_text(_summary(item) or item.text or item.id, 74) or item.id


def _graph_topic_nodes(
    by_topic: dict[str, list[Item]],
    topic_edges: list[dict[str, Any]],
    topic_pages: dict[str, TopicPage],
    latest_date: str | None,
) -> list[dict[str, Any]]:
    neighbours: dict[str, set[str]] = defaultdict(set)
    for edge in topic_edges:
        source = edge["source"].removeprefix("topic:")
        target = edge["target"].removeprefix("topic:")
        neighbours[source].add(target)
        neighbours[target].add(source)

    out: list[dict[str, Any]] = []
    for slug in sorted(by_topic):
        group = by_topic[slug]
        confidence = Counter(_confidence(item) or "unknown" for item in group)
        content_types = Counter(_content_type(item) for item in group)
        last_activity = max((_date(item) for item in group), default=None)
        recent_count = 0
        previous_count = 0
        if latest_date is not None:
            latest_month = latest_date[:7]
            previous_month = _previous_month(latest_month)
            recent_count = sum(1 for item in group if _date(item)[:7] == latest_month)
            previous_count = sum(1 for item in group if _date(item)[:7] == previous_month)
        growth_delta = recent_count - previous_count
        growth_rate = (
            round(growth_delta / previous_count * 100, 1)
            if previous_count
            else (100.0 if recent_count else 0.0)
        )
        out.append(
            {
                "id": f"topic:{slug}",
                "slug": slug,
                "label": humanize_topic(slug),
                "count": len(group),
                "related_count": len(neighbours[slug]),
                "recent_count": recent_count,
                "previous_count": previous_count,
                "growth_delta": growth_delta,
                "growth_rate": growth_rate,
                "last_activity": last_activity,
                "confidence": {
                    "high": confidence["high"],
                    "medium": confidence["medium"],
                    "low": confidence["low"],
                    "unknown": confidence["unknown"],
                },
                "content_types": {
                    "article": content_types["article"],
                    "video": content_types["video"],
                    "article_failed": content_types["article_failed"],
                    "post_only": content_types["post_only"],
                },
                "overview": topic_pages[slug].overview if slug in topic_pages else "",
                "note": f"topics/{slug}.md" if slug in topic_pages else None,
            }
        )
    return out


def _previous_month(month: str) -> str:
    year, number = map(int, month.split("-"))
    if number == 1:
        return f"{year - 1:04d}-12"
    return f"{year:04d}-{number - 1:02d}"


def _graph_item_nodes(items: list[Item], id2note: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda i: (i.captured_at, i.id), reverse=True):
        topics = _topics(item)
        if not topics:
            continue
        out.append(
            {
                "id": f"item:{item.id}",
                "item_id": item.id,
                "label": _graph_item_label(item),
                "summary": _compact_text(_summary(item), 180),
                "date": _date(item),
                "captured_date": _date(item),
                "published_date": _published_date(item),
                "primary_topic": _primary(item),
                "topics": topics,
                "content_type": _content_type(item),
                "confidence": _confidence(item) or "unknown",
                "author": item.author.handle,
                "author_name": item.author.name,
                "domain": _graph_domain(item),
                "note": id2note.get(item.id),
                "url": item.url,
                "needs_review": _needs_topic_review(item),
                "suggested_topics": _suggested_topics(item),
            }
        )
    return out


def _graph_topic_edges(items: list[Item], limit: int = 200) -> list[dict[str, Any]]:
    weights: Counter[tuple[str, str]] = Counter()
    topic_counts: Counter[str] = Counter()
    for item in items:
        topics = sorted(_topics(item))
        topic_counts.update(topics)
        for source, target in combinations(topics, 2):
            weights[(source, target)] += 1

    candidates: list[dict[str, Any]] = [
        {
            "source": f"topic:{s}",
            "target": f"topic:{t}",
            "kind": "cooccurrence",
            "weight": w,
            "affinity": round(w / (topic_counts[s] + topic_counts[t] - w), 4),
        }
        for (s, t), w in weights.items()
        if w >= 1
    ]
    candidates.sort(key=lambda e: (-int(e["weight"]), str(e["source"]), str(e["target"])))

    per_topic: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for edge in candidates:
        source = str(edge["source"])
        target = str(edge["target"])
        if per_topic[source] >= 5 and per_topic[target] >= 5:
            continue
        selected.append(edge)
        per_topic[source] += 1
        per_topic[target] += 1
        if len(selected) >= limit:
            break
    return selected


def _graph_membership_edges(items: list[Item], per_topic_limit: int = 40) -> list[dict[str, Any]]:
    used: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda i: (i.captured_at, i.id), reverse=True):
        primary = _primary(item)
        for slug in _topics(item):
            if used[slug] >= per_topic_limit:
                continue
            out.append(
                {
                    "source": f"topic:{slug}",
                    "target": f"item:{item.id}",
                    "kind": "primary" if slug == primary else "secondary",
                    "weight": 2 if slug == primary else 1,
                }
            )
            used[slug] += 1
    return out


def _graph_emerging_topics(items: list[Item]) -> list[dict[str, Any]]:
    suggested: dict[str, set[str]] = defaultdict(set)
    related: dict[str, set[str]] = defaultdict(set)
    for item in items:
        for suggestion in _suggested_topics(item):
            suggested[suggestion].add(item.id)
            related[suggestion].update(_topics(item))
    return [
        {
            "id": f"emerging:{slug}",
            "slug": slug,
            "label": humanize_topic(slug),
            "count": len(item_ids),
            "item_ids": sorted(item_ids),
            "related_topics": sorted(related[slug]),
        }
        for slug, item_ids in sorted(
            suggested.items(), key=lambda entry: (-len(entry[1]), entry[0])
        )
    ]


def _graph_facets(items: list[Item]) -> dict[str, Any]:
    content_types = Counter(_content_type(item) for item in items if _topics(item))
    confidence = Counter(_confidence(item) or "unknown" for item in items if _topics(item))
    months = Counter(_month(item) for item in items if _topics(item))
    return {
        "content_types": dict(sorted(content_types.items())),
        "confidence": {
            "high": confidence["high"],
            "medium": confidence["medium"],
            "low": confidence["low"],
            "unknown": confidence["unknown"],
        },
        "months": [{"month": month, "count": count} for month, count in sorted(months.items())],
    }


def _graph_insights(
    topic_nodes: list[dict[str, Any]],
    topic_edges: list[dict[str, Any]],
    emerging_topics: list[dict[str, Any]],
) -> dict[str, Any]:
    by_slug = {node["slug"]: node for node in topic_nodes}
    by_degree: Counter[str] = Counter()
    by_weight: Counter[str] = Counter()
    for edge in topic_edges:
        for key in ("source", "target"):
            slug = edge[key].removeprefix("topic:")
            by_degree[slug] += 1
            by_weight[slug] += edge["weight"]

    connected_slug = by_weight.most_common(1)[0][0] if by_weight else None
    isolated = [node for node in topic_nodes if by_degree[node["slug"]] <= 1]
    review = sorted(
        topic_nodes,
        key=lambda node: (
            node["confidence"]["low"]
            + node["confidence"]["unknown"]
            + (node["count"] if node["slug"] == "misc" else 0),
            node["count"],
        ),
        reverse=True,
    )
    active = sorted(topic_nodes, key=lambda node: node["last_activity"] or "", reverse=True)
    fastest = sorted(
        topic_nodes,
        key=lambda node: (node["growth_delta"], node["growth_rate"], node["recent_count"]),
        reverse=True,
    )
    return {
        "most_connected": by_slug.get(connected_slug) if connected_slug else None,
        "fastest_growing": fastest[0] if fastest else None,
        "isolated": isolated[:8],
        "needs_review": review[:8],
        "recently_active": active[:8],
        "emerging_count": sum(topic["count"] for topic in emerging_topics),
    }


def _graph_version(graph: dict[str, Any]) -> str:
    """Stable taxonomy version so new documents do not discard saved positions."""
    payload = json.dumps(
        {
            "schema": 2,
            "topics": sorted(topic["slug"] for topic in graph["topics"]),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _graph_section(
    items: list[Item], topic_pages: dict[str, TopicPage], id2note: dict[str, str]
) -> dict[str, Any]:
    by_topic: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        for slug in _topics(item):
            by_topic[slug].append(item)

    topic_edges = _graph_topic_edges(items)
    latest_date = max((_date(item) for item in items), default=None)
    emerging_topics = _graph_emerging_topics(items)
    topic_nodes = _graph_topic_nodes(by_topic, topic_edges, topic_pages, latest_date)
    graph = {
        "version": "",
        "topics": topic_nodes,
        "items": _graph_item_nodes(items, id2note),
        "topic_edges": topic_edges,
        "membership_edges": _graph_membership_edges(items),
        "emerging_topics": emerging_topics,
        "facets": _graph_facets(items),
        "insights": {},
    }
    graph["insights"] = _graph_insights(topic_nodes, topic_edges, emerging_topics)
    graph["version"] = _graph_version(graph)
    return graph


def _meta(
    items: list[Item],
    topic_freq: "Counter[str]",
    longform: dict[str, Any],
    media: dict[str, int],
    library: dict[str, Any],
    taxonomy: dict[str, Any],
    updated: str,
    bookmarks: int,
    own: int,
    storage: dict[str, Any],
) -> dict[str, Any]:
    """The KPI header block (totals, enrichment, long-form, media, timestamp)."""
    assigned_topics = {slug for item in items for slug in _topics(item)}
    ready_items = sum(1 for item in items if _processing_ready(item))
    return {
        "total": len(items),
        "bookmarks": bookmarks,
        "own": own,
        "enriched": sum(1 for i in items if i.enriched),
        "topics_count": len(assigned_topics),
        "primary_topics_count": len(topic_freq),
        "processing_ready": ready_items,
        "processing_ready_pct": round(ready_items / len(items) * 100, 1) if items else 0.0,
        "longform": {k: longform[k] for k in _META_LONGFORM_KEYS},
        "media": media,
        "library": library,
        "taxonomy": taxonomy,
        "storage": storage,
        "updated": updated,
    }


def _processing_ready(item: Item) -> bool:
    """True when enrichment exists and is not older than fetched content."""
    if item.enriched is None:
        return False
    if item.links and item.content is None:
        return False
    return item.content is None or item.enriched.enriched_at >= item.content.fetched_at


def compute_dashboard_data(
    items: list[Item],
    topic_pages: dict[str, TopicPage],
    id2note: dict[str, str],
    thumbs: list[dict[str, Any]],
    updated: str,
    storage: dict[str, Any] | None = None,
    item_storage: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the full JSON blob the dashboard template consumes.

    Pure: no file or network IO (photo thumbnails are computed by
    `collect_thumbnails` and injected via `thumbs`). `updated` is a display
    string (the caller stamps the generation date).
    """
    growth = _growth(items)
    per_month: dict[str, list[Item]] = growth.pop("_per_month")
    topic_freq: "Counter[str]" = Counter(_primaries(items))
    slug2label = {s: humanize_topic(s) for s in topic_freq}
    storage = storage or {}
    item_storage = item_storage or {}

    def rows(group: list[Item]) -> list[dict[str, Any]]:
        return [_row(i, id2note, slug2label, item_storage) for i in group]

    topics_sorted, topic_data = _topics_section(items, topic_freq, topic_pages, rows)
    authors, author_data = _authors_section(items, rows)
    domains, domain_data = _domains_section(items, rows)
    months_data = _months_section(per_month, rows)
    longform = _longform(items, id2note, item_storage)
    media = _media_counts(items)
    library = _library_section(items)
    taxonomy = _taxonomy_section(items, rows)
    graph = _graph_section(items, topic_pages, id2note)
    bookmark_items = [i for i in items if i.source == "bookmark"]
    own_items = [i for i in items if i.source == "own_tweet"]
    photo_posts = [i for i in items if _downloaded_image_entries(i)]
    library_items = sorted(
        [
            _row(
                item,
                id2note,
                slug2label,
                item_storage,
                include_article_images=True,
            )
            for item in items
        ],
        key=lambda row: (str(row["captured_date"]), str(row["id"])),
        reverse=True,
    )

    return {
        "meta": _meta(
            items,
            topic_freq,
            longform,
            media,
            library,
            taxonomy,
            updated,
            len(bookmark_items),
            len(own_items),
            storage,
        ),
        **growth,
        "topics_sorted": topics_sorted,
        "topic_data": topic_data,
        "authors": authors,
        "author_data": author_data,
        "domains": domains,
        "domain_data": domain_data,
        "months_data": months_data,
        "library_items": library_items,
        "longform_full": longform,
        "taxonomy": taxonomy,
        "graph": graph,
        "photos": {
            "downloaded": media["photos_downloaded"],
            "pending": media["photos_pending"],
            "article_downloaded": media["article_images_downloaded"],
            "article_pending": media["article_images_pending"],
            "thumbs": thumbs,
            "samples": _recent(rows(photo_posts), 6),
        },
        "videos": {"count": media["videos"], "items": _videos(items, id2note, item_storage)},
        "sources": {
            "bookmark": {"count": len(bookmark_items), "samples": _recent(rows(bookmark_items), 6)},
            "own_tweet": {"count": len(own_items), "samples": _recent(rows(own_items), 6)},
        },
        "ops": _ops(items, id2note, rows),
        "storage": storage,
    }


def _resource(name: str) -> str:
    return (resources.files("xbrain") / "resources" / name).read_text(encoding="utf-8")


def _escape_for_script(payload: str) -> str:
    """Make a JSON payload safe to inline inside a ``<script>`` block.

    ``json.dumps(ensure_ascii=False)`` leaves ``<``/``>``/``&`` raw and lets the
    JS line terminators U+2028/U+2029 and lone UTF-16 surrogates (from mangled
    emoji in scraped X text) through. Un-escaped, a ``</script>`` in any post
    summary/title/handle would close the tag at HTML-parse time — a stored-XSS
    break-out — and a lone surrogate would crash the UTF-8 write. The escaped
    forms ``JSON.parse`` back to the originals, so displayed content is identical.
    """
    escaped = (
        payload.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace(chr(0x2028), "\\u2028")
        .replace(chr(0x2029), "\\u2029")
    )
    # Round-trip through UTF-8 to drop lone surrogates (illegal in UTF-8) so
    # writing the dashboard can never abort the whole `generate` run.
    return escaped.encode("utf-8", "replace").decode("utf-8")


def render_dashboard_html(
    data: dict[str, Any], template: str | None = None, echarts: str | None = None
) -> str:
    """Inject the data blob and the ECharts library into the vendored template.

    Loads the template and library from `xbrain/resources/` when not supplied.
    The result is a self-contained HTML document (no external scripts except the
    Google Fonts stylesheet).
    """
    template = template if template is not None else _resource("dashboard.template.html")
    echarts = echarts if echarts is not None else _resource("echarts.min.js")
    payload = _escape_for_script(json.dumps(data, ensure_ascii=False))
    # Inject the trusted (fixed) library first and the user-derived payload LAST,
    # so a summary/title containing the literal `/*__ECHARTS__*/` sentinel can
    # never splice the library into the JSON on a re-scan.
    return template.replace("/*__ECHARTS__*/", echarts).replace("/*__DATA__*/", payload)
