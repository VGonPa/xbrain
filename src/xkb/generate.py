"""Render the JSON store into Obsidian markdown notes."""
from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from xkb.models import Item

logger = logging.getLogger(__name__)

GEN_START = "<!-- xkb:generated:start -->"
GEN_END = "<!-- xkb:generated:end -->"
_DEFAULT_TAIL = (
    "\n\n## Mis notas\n\n"
    "*(Escribe debajo. El bloque por encima de este punto se regenera "
    "automáticamente; no lo edites.)*\n\n"
)


def generate(store: dict[str, Item], output_dir: Path) -> None:
    """Write _index.md, log.md and one note per item that has links."""
    items = sorted(store.values(), key=lambda i: i.created_at, reverse=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_index.md").write_text(_render_index(items), encoding="utf-8")
    (output_dir / "log.md").write_text(_render_log(items), encoding="utf-8")
    for item in items:
        if item.links:
            _write_note(items_dir, item)


def _write_note(items_dir: Path, item: Item) -> None:
    """Write an item's note, replacing only the generated region.

    The filename ends with the item's globally unique ``id``. That makes
    every note path collision-free and lets us locate a note written for
    this item under a previous title or date: that stale note is migrated
    so the user's hand-written tail follows the item instead of being
    orphaned.
    """
    path = items_dir / _note_filename(item)
    block = f"{GEN_START}\n{_render_note(item)}\n{GEN_END}"
    source = path if path.exists() else _stale_note(items_dir, item, path)
    if source is not None:
        tail = _user_tail(source.read_text(encoding="utf-8"))
        if source != path:
            source.unlink()
            logger.info("Migrated note %s -> %s", source.name, path.name)
    else:
        tail = _DEFAULT_TAIL
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


def _user_tail(existing: str) -> str:
    """Return the content to preserve after the generated block.

    Normally everything after GEN_END. If GEN_END is missing (markers
    deleted or corrupted) but the file has content, preserve the whole
    file rather than discarding the user's work.
    """
    idx = existing.find(GEN_END)
    if idx != -1:
        return existing[idx + len(GEN_END):]
    if existing.strip():
        return "\n\n" + existing
    return _DEFAULT_TAIL


def _render_note(item: Item) -> str:
    lines = [_frontmatter(item), "", f"# {_title(item)}", "", item.text, ""]
    if item.links:
        lines.append("## Enlaces")
        lines += [f"- <{link.url}>" for link in item.links]
        lines.append("")
    lines += [f"[Ver tweet original]({item.url})", ""]
    if item.content:
        for source in item.content.sources:
            if source.ok and source.text:
                heading = source.title or source.url
                lines += [f"## Contenido: {heading}", "", source.text, ""]
    lines.append("## Enrichment")
    if item.enriched:
        lines += [
            f"- **Resumen:** {item.enriched.summary or '—'}",
            f"- **Temas:** {', '.join(item.enriched.topics) or '—'}",
            f"- **Cursos:** {', '.join(item.enriched.course_suggestions) or '—'}",
        ]
    else:
        lines.append("_Pendiente de enriquecer._")
    return "\n".join(lines)


def _frontmatter(item: Item) -> str:
    topics = ", ".join(item.enriched.topics) if item.enriched else ""
    course = ", ".join(item.enriched.course_suggestions) if item.enriched else ""
    domains = ", ".join(sorted({link.domain for link in item.links}))
    tags = "x-knowledge" + (f", {topics}" if topics else "")
    return "\n".join([
        "---",
        f'id: "{item.id}"',
        f"source: {item.source}",
        f"url: {item.url}",
        f"created: {item.created_at.date().isoformat()}",
        f"author: {item.author.handle}",
        f"domains: [{domains}]",
        f"tags: [{tags}]",
        f'course: "{course}"',
        "---",
    ])


def _render_index(items: list[Item]) -> str:
    bookmarks = sum(1 for i in items if i.source == "bookmark")
    own = sum(1 for i in items if i.source == "own_tweet")
    with_links = sum(1 for i in items if i.links)
    fetched = sum(1 for i in items if i.content)
    enriched = sum(1 for i in items if i.enriched)
    domains: dict[str, int] = {}
    for item in items:
        for link in item.links:
            domains[link.domain] = domains.get(link.domain, 0) + 1
    top = sorted(domains.items(), key=lambda kv: kv[1], reverse=True)[:15]
    lines = [
        "# X Knowledge Base",
        "",
        f"> Generado: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Resumen",
        "",
        f"- Items totales: {len(items)}",
        f"- Bookmarks: {bookmarks} · Tweets propios: {own}",
        f"- Con enlace (tienen nota): {with_links}",
        f"- Con contenido descargado: {fetched}",
        f"- Enriquecidos: {enriched}",
        "",
        "## Índices",
        "",
        "- [[log|Log cronológico completo]]",
        "",
        "## Dominios más enlazados",
        "",
    ]
    lines += [f"- {domain}: {count}" for domain, count in top]
    return "\n".join(lines) + "\n"


def _render_log(items: list[Item]) -> str:
    lines = ["# Log cronológico", ""]
    for item in items:
        date = item.created_at.date().isoformat()
        snippet = item.text.replace("\n", " ")[:120]
        link = (
            f" → [[items/{Path(_note_filename(item)).stem}|nota]]"
            if item.links
            else ""
        )
        lines.append(f"- `{date}` @{item.author.handle}: {snippet}{link}")
    return "\n".join(lines) + "\n"


def _title(item: Item) -> str:
    if item.content:
        for source in item.content.sources:
            if source.title:
                return source.title
    return item.text.replace("\n", " ")[:80] or item.id


def _note_filename(item: Item) -> str:
    return (
        f"{item.created_at.date().isoformat()}-"
        f"{_slugify(_title(item))}-{item.id}.md"
    )


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug[:60] or "item"
