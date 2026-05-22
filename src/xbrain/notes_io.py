"""Shared markdown helpers for the generated wiki pages.

The generated-block markers, user-tail preservation and the slug / filename
helpers — used by both `xbrain.generate` (item notes) and `xbrain.topics`
(topic pages).
"""

from __future__ import annotations

import re
import unicodedata

from xbrain.models import Item

GEN_START = "<!-- xbrain:generated:start -->"
GEN_END = "<!-- xbrain:generated:end -->"

# The default "Mis notas" tail appended after the generated block of a freshly
# created page — shared by item notes (`xbrain.generate`) and topic pages
# (`xbrain.topics`).
DEFAULT_TAIL = (
    "\n\n## Mis notas\n\n"
    "*(Escribe debajo. El bloque por encima de este punto se regenera "
    "automáticamente; no lo edites.)*\n\n"
)


def wrap(body: str) -> str:
    """Surround a generated body with the start / end markers."""
    return f"{GEN_START}\n{body}\n{GEN_END}"


def user_tail(existing: str, default_tail: str) -> str:
    """Return the content to preserve after the generated block.

    Normally everything after `GEN_END`. If `GEN_END` is missing (markers
    deleted or corrupted) but the file has content, preserve the whole file
    rather than discarding the user's work. An empty file gets `default_tail`.
    """
    idx = existing.find(GEN_END)
    if idx != -1:
        return existing[idx + len(GEN_END) :]
    if existing.strip():
        return "\n\n" + existing
    return default_tail


def slugify(text: str) -> str:
    """A lowercase ASCII kebab-case slug, capped at 60 characters."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug[:60] or "item"


def title_of(item: Item) -> str:
    """A display title for an item — a fetched article title, else the text.

    Only the success variant of `ContentSource` carries a `title` field; the
    failure variant has no title, so the isinstance narrowing both satisfies
    mypy and silently skips broken-link sources.
    """
    from xbrain.models import ContentSourceSuccess

    if item.content:
        for source in item.content.sources:
            if isinstance(source, ContentSourceSuccess) and source.title:
                return source.title
    return item.text.replace("\n", " ")[:80] or item.id


def note_filename(item: Item) -> str:
    """The collision-free filename of an item's note (ends with its id)."""
    return f"{item.created_at.date().isoformat()}-{slugify(title_of(item))}-{item.id}.md"
