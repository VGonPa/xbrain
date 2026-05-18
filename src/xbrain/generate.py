"""Render the JSON store into Obsidian markdown notes."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from xbrain.models import Content, ContentSource, FailureReason, Item
from xbrain.notes_io import DEFAULT_TAIL, note_filename, slugify, title_of, user_tail, wrap

logger = logging.getLogger(__name__)

_FAILURE_ES: dict[FailureReason, str] = {
    "not_found": "no encontrado",
    "forbidden": "acceso denegado",
    "paywall": "muro de pago",
    "timeout": "tiempo de espera agotado",
    "dns_error": "dominio no resuelto",
    "js_required": "requiere JavaScript",
    "empty_content": "sin contenido extraíble",
}


def _broken_link_line(source: ContentSource, fetched_at: datetime) -> str:
    """A one-line, human-readable record of a link that could not be fetched."""
    bits: list[str] = []
    if source.http_status:
        bits.append(f"HTTP {source.http_status}")
    if source.failure_reason:
        bits.append(_FAILURE_ES.get(source.failure_reason, source.failure_reason))
    detail = " · ".join(bits) or "no se pudo recuperar"
    date = fetched_at.date().isoformat()
    return f"> ⚠ Enlace roto: <{source.url}> — {detail} (verificado {date})"


def generate(
    store: dict[str, Item],
    output_dir: Path,
    since: datetime | None = None,
    until: datetime | None = None,
) -> None:
    """Write _index.md, log.md and one note per noted item.

    A note is written for any item that has links or has been enriched. The
    index and log always reflect the whole store; `since`/`until` only narrow
    which item notes are (re)generated.
    """
    items = sorted(store.values(), key=lambda i: i.created_at, reverse=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_index.md").write_text(_render_index(items), encoding="utf-8")
    (output_dir / "log.md").write_text(_render_log(items), encoding="utf-8")
    for item in items:
        if _has_note(item) and _in_range(item, since, until):
            _write_note(items_dir, item)


def _has_note(item: Item) -> bool:
    """An item gets its own note if it has links or has been enriched."""
    return bool(item.links) or item.enriched is not None


def _in_range(item: Item, since: datetime | None, until: datetime | None) -> bool:
    if since and item.created_at < since:
        return False
    if until and item.created_at > until:
        return False
    return True


def _write_note(items_dir: Path, item: Item) -> None:
    """Write an item's note, replacing only the generated region.

    The filename ends with the item's globally unique ``id``. That makes
    every note path collision-free and lets us locate a note written for
    this item under a previous title or date: that stale note is migrated
    so the user's hand-written tail follows the item instead of being
    orphaned.
    """
    path = items_dir / note_filename(item)
    block = wrap(_render_note(item))
    source = path if path.exists() else _stale_note(items_dir, item, path)
    if source is not None:
        tail = user_tail(source.read_text(encoding="utf-8"), DEFAULT_TAIL)
        if source != path:
            source.unlink()
            logger.info("Migrated note %s -> %s", source.name, path.name)
    else:
        tail = DEFAULT_TAIL
    path.write_text(block + tail, encoding="utf-8")


def _stale_note(items_dir: Path, item: Item, current: Path) -> Path | None:
    """Find this item's previous note when a filename component changed.

    The filename ends with the item's globally unique ``id``, so a glob on
    that id matches at most one file. If that file is not the item's
    current note path, the title slug or date changed and the note must be
    migrated; otherwise there is nothing to migrate.
    """
    for candidate in items_dir.glob(f"*-{item.id}.md"):
        if candidate != current:
            return candidate
    return None


def _enrichment_lines(item: Item) -> list[str]:
    """Summary + topic links for an enriched item (empty if not enriched)."""
    if not item.enriched:
        return []
    lines: list[str] = []
    if item.enriched.summary:
        lines += [item.enriched.summary, ""]
    if item.enriched.topics:
        links = " · ".join(f"[[{t}]]" for t in item.enriched.topics)
        lines += [f"**Temas:** {links}", ""]
    return lines


def _content_lines(content: Content) -> list[str]:
    """Rendered article bodies + broken-link evidence for a fetched item."""
    lines: list[str] = []
    for source in content.sources:
        if source.ok and source.text:
            heading = source.title or source.url
            lines += [f"## Contenido: {heading}", "", source.text, ""]
        elif not source.ok and source.kind in ("external_article", "x_article"):
            lines += [_broken_link_line(source, content.fetched_at), ""]
    return lines


def _render_note(item: Item) -> str:
    lines = [_frontmatter(item), "", f"# {title_of(item)}", ""]
    lines += _enrichment_lines(item)
    lines += ["## Tweet", "", item.text, ""]
    if item.links:
        lines.append("## Enlaces")
        lines += [f"- <{link.url}>" for link in item.links]
        lines.append("")
    lines += [f"[Ver tweet original]({item.url})", ""]
    if item.content:
        lines += _content_lines(item.content)
    return "\n".join(lines).rstrip()


def _frontmatter(item: Item) -> str:
    domains = ", ".join(sorted({link.domain for link in item.links}))
    tags = ["x-knowledge"]
    if item.enriched:
        tags += item.enriched.topics  # topics already includes primary_topic
    if item.bookmark_folder:
        tags.append(slugify(item.bookmark_folder))
    tags = list(dict.fromkeys(tags))
    lines = [
        "---",
        f'id: "{item.id}"',
        f"source: {item.source}",
        f"url: {item.url}",
        f"created: {item.created_at.date().isoformat()}",
        f"author: {item.author.handle}",
        f"domains: [{domains}]",
        f"tags: [{', '.join(tags)}]",
    ]
    if item.bookmark_folder:
        lines.append(f"bookmark_folder: {item.bookmark_folder}")
    lines.append("---")
    return "\n".join(lines)


def _render_index(items: list[Item]) -> str:
    bookmarks = sum(1 for i in items if i.source == "bookmark")
    own = sum(1 for i in items if i.source == "own_tweet")
    noted = sum(1 for i in items if _has_note(i))
    enriched = sum(1 for i in items if i.enriched)
    topic_freq: dict[str, int] = {}
    for item in items:
        if item.enriched:
            for topic in item.enriched.topics:
                topic_freq[topic] = topic_freq.get(topic, 0) + 1
    lines = [
        "# XBrain",
        "",
        f"> Generado: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Resumen",
        "",
        f"- Items totales: {len(items)}",
        f"- Bookmarks: {bookmarks} · Tweets propios: {own}",
        f"- Con nota propia: {noted}",
        f"- Enriquecidos: {enriched}",
        "",
        "## Índices",
        "",
        "- [[log|Log cronológico completo]]",
        "",
        "## Temas",
        "",
    ]
    for topic, count in sorted(topic_freq.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- [[{topic}]] ({count})")
    return "\n".join(lines) + "\n"


def _render_log(items: list[Item]) -> str:
    lines = ["# Log cronológico", ""]
    for item in items:
        date = item.created_at.date().isoformat()
        snippet = item.text.replace("\n", " ")[:120]
        link = f" → [[items/{Path(note_filename(item)).stem}|nota]]" if _has_note(item) else ""
        lines.append(f"- `{date}` @{item.author.handle}: {snippet}{link}")
    return "\n".join(lines) + "\n"
