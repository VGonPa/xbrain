"""Render the JSON store into Obsidian markdown notes."""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from pathlib import Path

from xkb.models import Item

GEN_START = "<!-- xkb:generated:start -->"
GEN_END = "<!-- xkb:generated:end -->"
_DEFAULT_TAIL = "\n\n## Mis notas\n\n"


def generate(store: dict[str, Item], output_dir: Path) -> None:
    """Write _index.md, log.md and one note per item that has links."""
    items = sorted(store.values(), key=lambda i: i.created_at, reverse=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_index.md").write_text(_render_index(items), encoding="utf-8")
    (output_dir / "log.md").write_text(_render_log(items), encoding="utf-8")
    for item in items:
        if item.links:
            _write_note(items_dir / _note_filename(item), _render_note(item))


def _write_note(path: Path, generated_block: str) -> None:
    """Write a note, replacing only the generated region if the file exists."""
    block = f"{GEN_START}\n{generated_block}\n{GEN_END}"
    if path.exists():
        tail = _user_tail(path.read_text(encoding="utf-8"))
    else:
        tail = _DEFAULT_TAIL
    path.write_text(block + tail, encoding="utf-8")


def _user_tail(existing: str) -> str:
    """Return everything the user wrote after the generated end marker."""
    idx = existing.find(GEN_END)
    if idx == -1:
        return _DEFAULT_TAIL
    return existing[idx + len(GEN_END):]


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
        f"> Generado: {datetime.now().date().isoformat()}",
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
        link = f" → [[items/{_note_filename(item)[:-3]}|nota]]" if item.links else ""
        lines.append(f"- `{date}` @{item.author.handle}: {snippet}{link}")
    return "\n".join(lines) + "\n"


def _title(item: Item) -> str:
    if item.content:
        for source in item.content.sources:
            if source.title:
                return source.title
    return item.text.replace("\n", " ")[:80] or item.id


def _note_filename(item: Item) -> str:
    return f"{item.created_at.date().isoformat()}-{_slugify(_title(item))}.md"


def _slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug[:60] or "item"
