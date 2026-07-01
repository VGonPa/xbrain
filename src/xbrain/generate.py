"""Render the JSON store into Obsidian markdown notes."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import assert_never

from xbrain.config import SUPPORTED_TOPIC_STYLES
from xbrain.dashboard import collect_thumbnails, compute_dashboard_data, render_dashboard_html
from xbrain.i18n import Strings, strings_for
from xbrain.models import (
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    FailureReason,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
    TopicPage,
)
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
    "unknown_error": "error desconocido",
}


# Subdirectory under `output_dir` where downloaded photos are mirrored at
# generate time, so an Obsidian vault is fully self-contained. Photos are
# canonically stored under `data/media/<id>/<n>.<ext>` and copied to
# `<output_dir>/_media/<id>/<n>.<ext>` whenever `generate` runs with a
# `media_root` argument. The leading underscore keeps the directory at
# the top of file listings and matches the convention used by static-
# site generators (Hugo, Jekyll) for non-content assets.
_VAULT_MEDIA_SUBDIR = "_media"


def _broken_link_line(source: ContentSourceFailure, fetched_at: datetime) -> str:
    """A one-line, human-readable record of a link that could not be fetched.

    Accepts only the failure variant — the type system enforces that
    `failure_reason` is present (no Optional check needed).
    """
    bits: list[str] = []
    if source.http_status:
        bits.append(f"HTTP {source.http_status}")
    bits.append(_FAILURE_ES.get(source.failure_reason, source.failure_reason))
    detail = " · ".join(bits) or "no se pudo recuperar"
    date = fetched_at.date().isoformat()
    return f"> ⚠ Enlace roto: <{source.url}> — {detail} (verificado {date})"


def generate(
    store: dict[str, Item],
    output_dir: Path,
    since: datetime | None = None,
    until: datetime | None = None,
    output_language: str = "English",
    topic_style: str = "wikilink",
    media_root: Path | None = None,
    topic_pages: dict[str, TopicPage] | None = None,
) -> None:
    """Write _index.md, log.md and one note per noted item.

    A note is written for any item that has links or has been enriched. The
    index and log always reflect the whole store; `since`/`until` only narrow
    which item notes are (re)generated. `output_language` drives the section
    headers (Topics:, Content:, Summary, ...) via `xbrain.i18n`.

    `topic_style` controls how the in-body ``**Topics:**`` line is rendered:
    ``"wikilink"`` (default) emits ``[[slug]]`` links, ``"hashtag"`` emits
    Obsidian ``#slug`` tags. The toggle does not affect frontmatter ``tags:``,
    the index ``## Topics`` section, or the topic-page post lists — those
    stay wikilinks by design.

    `media_root` is the directory under which `xbrain media` downloads
    photos as `<item-id>/<index>.<ext>`. When provided, photos for each
    item being rendered are copied to
    `<output_dir>/_media/<item-id>/<index>.<ext>` and embedded in the
    note body via Obsidian wikilink embeds. When `None`, photo entries
    render as if no `xbrain media` run had taken place — pending photos
    are silent, failed and video-pending photos still produce their
    warning lines (the URL is in the data; only the file bytes are
    missing).
    """
    if topic_style not in SUPPORTED_TOPIC_STYLES:
        raise ValueError(
            f"Unsupported topic_style: {topic_style!r}. Supported: {SUPPORTED_TOPIC_STYLES}"
        )
    strings = strings_for(output_language)
    items = sorted(store.values(), key=lambda i: i.created_at, reverse=True)
    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "_index.md").write_text(_render_index(items, strings), encoding="utf-8")
    (output_dir / "log.md").write_text(_render_log(items), encoding="utf-8")
    for item in items:
        if _has_note(item) and _in_range(item, since, until):
            if media_root is not None:
                _mirror_item_media(item, media_root, output_dir / _VAULT_MEDIA_SUBDIR)
            _write_note(items_dir, item, strings, topic_style)
    try:
        _write_dashboard(items, output_dir, items_dir, topic_pages or {}, media_root)
    except Exception:  # noqa: BLE001 - the dashboard is a best-effort secondary artifact
        logger.warning("Dashboard generation failed; item notes were written.", exc_info=True)


def _write_dashboard(
    items: list[Item],
    output_dir: Path,
    items_dir: Path,
    topic_pages: dict[str, TopicPage],
    media_root: Path | None,
) -> None:
    """Write the self-contained interactive `dashboard.html` from the store.

    The id→note map uses the same `note_filename` the item notes are written
    under, so the dashboard's ``note ↗`` deep links point at real vault files.
    Photo thumbnails come from `media_root`; topic overviews from `topic_pages`.
    No browser is involved — the HTML is template + injected JSON.
    """
    # Absolute paths: `obsidian://open?path=` requires them, and `output_dir`
    # can be relative when the configured vault is relative.
    id2note = {
        item.id: str((items_dir / note_filename(item)).resolve())
        for item in items
        if _has_note(item)
    }
    thumbs = collect_thumbnails(items, media_root, id2note)
    now = datetime.now(timezone.utc)
    updated = f"{now:%b} {now.day}, {now.year}".upper()
    data = compute_dashboard_data(items, topic_pages, id2note, thumbs, updated)
    (output_dir / "dashboard.html").write_text(render_dashboard_html(data), encoding="utf-8")


def _has_note(item: Item) -> bool:
    """An item gets its own note if it has links, media, or has been enriched.

    A tweet whose only payload is a photo (no link, no LLM enrichment
    yet) was previously invisible in the wiki. Including it surfaces
    the photo as soon as `xbrain media` populates the variant — the
    natural read flow.
    """
    return bool(item.links) or bool(item.media) or item.enriched is not None


def _in_range(item: Item, since: datetime | None, until: datetime | None) -> bool:
    if since and item.created_at < since:
        return False
    if until and item.created_at > until:
        return False
    return True


def _write_note(items_dir: Path, item: Item, strings: Strings, topic_style: str) -> None:
    """Write an item's note, replacing only the generated region.

    The filename ends with the item's globally unique ``id``. That makes
    every note path collision-free and lets us locate a note written for
    this item under a previous title or date: that stale note is migrated
    so the user's hand-written tail follows the item instead of being
    orphaned.
    """
    path = items_dir / note_filename(item)
    block = wrap(_render_note(item, strings, topic_style))
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


def _enrichment_lines(item: Item, strings: Strings, topic_style: str) -> list[str]:
    """Summary + topic refs for an enriched item (empty if not enriched).

    `topic_style` selects the in-body topic-line rendering:
    - ``"wikilink"`` → ``**Topics:** [[ai-coding]] · [[software-engineering]]``
    - ``"hashtag"``  → ``**Topics:** #ai-coding #software-engineering``

    The hashtag mode uses a bare space as separator: Obsidian's tag parser
    consumes a trailing middle-dot as part of the tag boundary on some
    renderers, which produces broken tags. Frontmatter ``tags:`` are emitted
    by ``_frontmatter`` and are independent of this toggle.
    """
    if not item.enriched:
        return []
    lines: list[str] = []
    if item.enriched.summary:
        lines += [item.enriched.summary, ""]
    if item.enriched.topics:
        if topic_style == "hashtag":
            refs = " ".join(f"#{t}" for t in item.enriched.topics)
        else:
            refs = " · ".join(f"[[{t}]]" for t in item.enriched.topics)
        lines += [f"**{strings.topics_label}:** {refs}", ""]
    return lines


def _render_media_lines(item: Item) -> list[str]:
    """One line per `Item.media` entry, ready to splice into the Tweet section.

    Variant handling:
    - `MediaPhotoDownloaded` / `MediaPhotoDescribed` / `MediaVideoDownloaded`
      → Obsidian embed `![[_media/<id>/<n>.<ext>]]`. The vault is
      self-contained: `generate()` mirrors the file from `data/media/` into
      `<output_dir>/_media/` before rendering, so the embed resolves
      with no user configuration. A downloaded video embeds its local
      mp4 exactly like a photo (Obsidian renders an inline player). The
      described variant inherits the same on-disk file — the description
      is consumed by the LLM prompts in `executors/api.py` /
      `topic_synth.py`, NOT shown as alt-text in this phase. Decorative
      photos are still embedded; the `is_decorative` flag only filters
      them out of the LLM prompts, never out of the visual rendering.
    - `MediaPhotoFailed` / `MediaVideoFailed` → one-line ⚠ warning carrying
      the failure reason and the original URL — visible evidence, not a
      silent drop.
    - `MediaPhotoPending`     → silent. Not an error, just "the next
      `xbrain media` run will pick it up".
    - `MediaVideoPending`     → a clickable "Ver vídeo" link to the playable
      stream (the mp4/HLS URL from `video_info.variants`, not the poster),
      flagged as pending local download until `xbrain download-videos`
      fetches the bytes (mp4) — HLS stays a link pending the ffmpeg follow-up.

    The output is intentionally plain markdown; the caller (`_render_note`)
    wraps it in a blank line on either side for readability.
    """
    lines: list[str] = []
    for entry in item.media:
        if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed, MediaVideoDownloaded)):
            lines.append(f"![[{_VAULT_MEDIA_SUBDIR}/{entry.local_path}]]")
        elif isinstance(entry, MediaPhotoFailed):
            reason = _FAILURE_ES_MEDIA.get(entry.failure_reason, entry.failure_reason)
            lines.append(f"> ⚠ Foto no disponible ({reason}): <{entry.url}>")
        elif isinstance(entry, MediaVideoFailed):
            reason = _FAILURE_ES_MEDIA.get(entry.failure_reason, entry.failure_reason)
            lines.append(f"> ⚠ Vídeo no disponible ({reason}): <{entry.url}>")
        elif isinstance(entry, MediaPhotoPending):
            # Silent: a future `xbrain media` run will advance this entry.
            continue
        elif isinstance(entry, MediaVideoPending):
            # `entry.url` is the playable stream (mp4 or HLS), not the poster,
            # so surface it as a clickable link; bytes are not saved yet.
            lines.append(f"> 🎥 [Ver vídeo]({entry.url}) (pendiente de descarga)")
        else:
            assert_never(entry)
    return lines


# Translations for media failure reasons — symmetric with `_FAILURE_ES`
# (content-source failures). Kept separate because the vocabularies differ:
# media has `http_4xx` and `format_error`, content has `js_required` and
# `paywall`, etc. A wrong translation here doesn't break anything (the slug
# itself is a fallback), but the operator-facing line should read cleanly.
_FAILURE_ES_MEDIA: dict[str, str] = {
    "http_4xx": "URL no encontrada (HTTP 4xx)",
    "http_5xx": "error del servidor (HTTP 5xx)",
    "timeout": "tiempo de espera agotado",
    "format_error": "formato no reconocido",
    "unknown_error": "error desconocido",
}


def _mirror_item_media(item: Item, media_root: Path, vault_media_dir: Path) -> None:
    """Copy every downloaded photo/video on `item` into the vault's `_media/` tree.

    The canonical store is `data/media/<id>/<n>.<ext>` (under `media_root`);
    the vault mirror is `<output_dir>/_media/<id>/<n>.<ext>`. Mirroring
    happens at render time, not download time, so the vault stays in sync
    with whichever subset of items `--since`/`--until` is regenerating.

    Files are copied with `shutil.copy2` (preserves mtime) and silently
    skipped when the source is missing — that should never happen on a
    healthy `data/media/` tree, but a manual cleanup of the bytes must not
    crash the generator. The variant on disk still drives the embed line,
    so a missing-bytes-but-marked-downloaded record renders as a broken
    embed Obsidian shows as an empty rectangle; that is loud enough.
    """
    vault_media_dir.mkdir(parents=True, exist_ok=True)
    for entry in item.media:
        # The described variant inherits the on-disk bytes from the prior
        # downloaded state; a downloaded video carries its mp4 the same way —
        # all three shapes hit the same mirror path.
        if not isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed, MediaVideoDownloaded)):
            continue
        source = media_root / entry.local_path
        destination = vault_media_dir / entry.local_path
        if not source.exists():
            # Marked downloaded in items.json but the file is gone — log
            # and move on. The Obsidian embed will render as a broken
            # image, which is the right user signal.
            logger.warning(
                "Photo bytes missing for item %s at %s — embed will render broken.",
                item.id,
                source,
            )
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _content_lines(content: Content, strings: Strings) -> list[str]:
    """Rendered article bodies + broken-link evidence for a fetched item.

    Switches on the `ContentSource` variant: the success variant is
    rendered as a content block; the failure variant is rendered as a
    broken-link line *only* for external articles and X articles (a
    failed thread fetch is silently elided, matching the pre-refactor
    behaviour — `source.kind` is what guarded that path before).
    """
    lines: list[str] = []
    for source in content.sources:
        if isinstance(source, ContentSourceSuccess):
            heading = source.title or source.url
            lines += [f"## {strings.content_header}: {heading}", "", source.text, ""]
        elif source.kind in ("external_article", "x_article"):
            lines += [_broken_link_line(source, content.fetched_at), ""]
    return lines


def _render_note(item: Item, strings: Strings, topic_style: str) -> str:
    """Render the wiki-side note for one item.

    The media block lives between the tweet text and the `## Enlaces`
    section: photos appear immediately under the tweet body, matching
    how X itself renders them — natural read order, no jumping.
    """
    lines = [_frontmatter(item), "", f"# {title_of(item)}", ""]
    lines += _enrichment_lines(item, strings, topic_style)
    lines += ["## Tweet", "", item.text, ""]
    media_lines = _render_media_lines(item)
    if media_lines:
        lines += media_lines
        lines.append("")
    if item.links:
        lines.append("## Enlaces")
        lines += [f"- <{link.url}>" for link in item.links]
        lines.append("")
    lines += [f"[Ver tweet original]({item.url})", ""]
    if item.content:
        lines += _content_lines(item.content, strings)
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


def _count_topic_frequency(items: list[Item]) -> dict[str, int]:
    """Tally how often each topic appears across the enriched items.

    Items without enrichment contribute nothing. The result maps topic slug
    to the number of enriched items that include it.
    """
    topic_freq: dict[str, int] = {}
    for item in items:
        if item.enriched:
            for topic in item.enriched.topics:
                topic_freq[topic] = topic_freq.get(topic, 0) + 1
    return topic_freq


def _render_index(items: list[Item], strings: Strings) -> str:
    """Render the top-level index note: corpus stats and the topic list."""
    bookmarks = sum(1 for i in items if i.source == "bookmark")
    own = sum(1 for i in items if i.source == "own_tweet")
    noted = sum(1 for i in items if _has_note(i))
    enriched = sum(1 for i in items if i.enriched)
    topic_freq = _count_topic_frequency(items)
    lines = [
        "# XBrain",
        "",
        f"> Generado: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        f"## {strings.summary_header}",
        "",
        f"- Items totales: {len(items)}",
        f"- Bookmarks: {bookmarks} · Tweets propios: {own}",
        f"- Con nota propia: {noted}",
        f"- Enriquecidos: {enriched}",
        "",
        "## Índices",
        "",
        "- [[log|Log cronológico completo]]",
        "- [📊 Dashboard interactivo](dashboard.html) — métricas, drill-down y enlaces (ábrelo en el navegador)",
        "",
        f"## {strings.topics_label}",
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
