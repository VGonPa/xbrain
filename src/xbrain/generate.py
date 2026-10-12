"""Render the JSON store into Obsidian markdown notes."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, assert_never

from xbrain.config import SUPPORTED_TOPIC_STYLES
from xbrain.dashboard import collect_thumbnails, compute_dashboard_data, render_dashboard_html
from xbrain.i18n import Strings, strings_for
from xbrain.models import (
    ARTICLE_PARAGRAPH_SEP,
    ArticleImageBlock,
    ArticleTextBlock,
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
from xbrain.notes_io import DEFAULT_TAIL, GEN_END, note_filename, slugify, title_of, user_tail, wrap
from xbrain.video_content import video_content_text

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


# Subdirectory under `output_dir` where note embeds expect media bytes. In local
# mode this is a render-time mirror of `data/media`; in Drive mode it is the
# canonical media location, so no duplicate `data/media` tree is needed.
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
    data_dir: Path | None = None,
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
    _remove_stale_item_notes(items_dir, items)
    # Absolute file:// URI so Obsidian opens the dashboard in the external browser
    # on click — a relative `dashboard.html` link is unreliable for non-markdown
    # files (Obsidian hides .html from the explorer and won't render its JS inline).
    # This pins the link to the machine that ran `generate` (the URI is absolute
    # and `_index.md` syncs via iCloud), which is the unavoidable cost of opening a
    # local file from Obsidian; it self-heals on the next `generate` per machine.
    dashboard_href = (output_dir / "dashboard.html").resolve().as_uri()
    (output_dir / "_index.md").write_text(
        _render_index(items, strings, dashboard_href), encoding="utf-8"
    )
    (output_dir / "log.md").write_text(_render_log(items), encoding="utf-8")
    item_storage: dict[str, dict[str, int | str]] = {}
    for _ in range(3):
        for item in items:
            if _has_note(item) and _in_range(item, since, until):
                if media_root is not None:
                    vault_media_dir = output_dir / _VAULT_MEDIA_SUBDIR
                    _mirror_item_media(item, media_root, vault_media_dir)
                    _mirror_item_article_images(item, media_root, vault_media_dir)
                _write_note(items_dir, item, strings, topic_style, item_storage.get(item.id))
        next_storage = _item_storage_map(items, items_dir, media_root)
        if _storage_labels(next_storage) == _storage_labels(item_storage):
            item_storage = next_storage
            break
        item_storage = next_storage
    _write_video_artifacts(items, output_dir / "videos", strings)
    item_storage = _item_storage_map(items, items_dir, media_root, output_dir / "videos")
    try:
        _write_dashboard(
            items,
            output_dir,
            items_dir,
            topic_pages or {},
            media_root,
            data_dir=data_dir,
            item_storage=item_storage,
        )
    except Exception:  # noqa: BLE001 - the dashboard is a best-effort secondary artifact
        logger.warning("Dashboard generation failed; item notes were written.", exc_info=True)


def _write_dashboard(
    items: list[Item],
    output_dir: Path,
    items_dir: Path,
    topic_pages: dict[str, TopicPage],
    media_root: Path | None,
    *,
    data_dir: Path | None = None,
    item_storage: dict[str, dict[str, int | str]] | None = None,
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
    updated = f"{now:%b} {now.day}, {now.year} {now:%H:%M} UTC".upper()
    storage = _storage_summary(data_dir, output_dir)
    data = compute_dashboard_data(
        items,
        topic_pages,
        id2note,
        thumbs,
        updated,
        storage=storage,
        item_storage=item_storage,
    )
    (output_dir / "dashboard.html").write_text(render_dashboard_html(data), encoding="utf-8")
    _write_canvas(data, output_dir, items)


def _canvas_color(seed: str) -> str:
    return str(sum(ord(char) for char in seed) % 6 + 1)


def _canvas_node_id(kind: str, identifier: str) -> str:
    safe = "".join(char if char.isalnum() else "-" for char in identifier).strip("-")
    return f"{kind}-{safe[:48]}"


def _canvas_topic_file(output_dir: Path, slug: str) -> str | None:
    rel = Path("topics") / f"{slug}.md"
    return rel.as_posix() if (output_dir / rel).exists() else None


def _canvas_graph_parts(
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    graph = data.get("graph")
    if not isinstance(graph, dict):
        return None
    topics = [topic for topic in graph.get("topics", []) if isinstance(topic, dict)]
    if not topics:
        return None
    graph_items = [item for item in graph.get("items", []) if isinstance(item, dict)]
    topic_edges = [edge for edge in graph.get("topic_edges", []) if isinstance(edge, dict)]
    topics = sorted(topics, key=lambda t: (-int(t.get("count", 0)), str(t.get("slug", ""))))[:40]
    return topics, graph_items, topic_edges


def _canvas_items_by_topic(
    graph_items: list[dict[str, Any]], topics: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    topic_by_slug = {str(topic["slug"]): topic for topic in topics if topic.get("slug")}
    items_by_topic: dict[str, list[dict[str, Any]]] = {}
    for item in graph_items:
        primary = item.get("primary_topic")
        if isinstance(primary, str) and primary in topic_by_slug:
            items_by_topic.setdefault(primary, []).append(item)
    for rows in items_by_topic.values():
        rows.sort(
            key=lambda item: (str(item.get("date", "")), str(item.get("item_id", ""))), reverse=True
        )
    return items_by_topic


def _canvas_topic_node(output_dir: Path, topic: dict[str, Any], x: int, y: int) -> dict[str, Any]:
    slug = str(topic.get("slug", "topic"))
    node_id = _canvas_node_id("topic", slug)
    base: dict[str, Any] = {
        "id": node_id,
        "x": x,
        "y": y,
        "width": 300,
        "height": 120,
        "color": _canvas_color(slug),
    }
    topic_file = _canvas_topic_file(output_dir, slug)
    if topic_file:
        return {**base, "type": "file", "file": topic_file}
    text = f"{topic.get('label', slug)}\n{topic.get('count', 0)} documents"
    return {**base, "type": "text", "text": text}


def _canvas_document_nodes(
    topic_node_id: str,
    slug: str,
    topic_items: list[dict[str, Any]],
    item_note_names: dict[str, str],
    x: int,
    y: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    item_node_ids: set[str] = set()
    for doc_index, item in enumerate(topic_items[:3]):
        item_id = str(item.get("item_id", ""))
        note_name = item_note_names.get(item_id)
        if not note_name:
            continue
        doc_node_id = _canvas_node_id("item", item_id)
        item_node_ids.add(doc_node_id)
        nodes.append(
            {
                "id": doc_node_id,
                "type": "file",
                "file": (Path("items") / note_name).as_posix(),
                "x": x + 340,
                "y": y + doc_index * 130,
                "width": 300,
                "height": 108,
                "color": _canvas_color(slug),
            }
        )
        edges.append(
            {
                "id": f"edge-{topic_node_id}-{doc_node_id}",
                "fromNode": topic_node_id,
                "fromSide": "right",
                "toNode": doc_node_id,
                "toSide": "left",
            }
        )
    return nodes, edges, item_node_ids


def _canvas_topic_edges(
    topic_edges: list[dict[str, Any]], topics: list[dict[str, Any]], included_topic_nodes: set[str]
) -> list[dict[str, Any]]:
    topic_by_id = {str(topic["id"]): topic for topic in topics if topic.get("id")}
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for edge in topic_edges[:160]:
        source = topic_by_id.get(str(edge.get("source")))
        target = topic_by_id.get(str(edge.get("target")))
        if source is None or target is None:
            continue
        source_id = _canvas_node_id("topic", str(source.get("slug", "")))
        target_id = _canvas_node_id("topic", str(target.get("slug", "")))
        if source_id not in included_topic_nodes or target_id not in included_topic_nodes:
            continue
        edge_id = f"edge-{source_id}-{target_id}"
        if edge_id in seen:
            continue
        seen.add(edge_id)
        edges.append(
            {
                "id": edge_id,
                "fromNode": source_id,
                "fromSide": "right",
                "toNode": target_id,
                "toSide": "left",
            }
        )
    return edges


def _write_canvas(data: dict[str, Any], output_dir: Path, items: list[Item]) -> None:
    """Write a derived Obsidian Canvas map without touching user-editable copies."""
    parts = _canvas_graph_parts(data)
    if parts is None:
        return
    topics, graph_items, topic_edges = parts
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    item_note_names = {item.id: note_filename(item) for item in items if _has_note(item)}
    items_by_topic = _canvas_items_by_topic(graph_items, topics)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    included_topic_nodes: set[str] = set()
    included_item_nodes: set[str] = set()
    columns = 5
    for index, topic in enumerate(topics):
        slug = str(topic.get("slug", "topic"))
        node_id = _canvas_node_id("topic", slug)
        x = (index % columns) * 520
        y = (index // columns) * 430
        included_topic_nodes.add(node_id)
        nodes.append(_canvas_topic_node(output_dir, topic, x, y))
        doc_nodes, doc_edges, doc_ids = _canvas_document_nodes(
            node_id, slug, items_by_topic.get(slug, []), item_note_names, x, y
        )
        nodes.extend(doc_nodes)
        edges.extend(doc_edges)
        included_item_nodes.update(doc_ids)

    edges.extend(_canvas_topic_edges(topic_edges, topics, included_topic_nodes))

    canvas = {
        "nodes": nodes,
        "edges": [
            edge
            for edge in edges
            if edge["fromNode"] in included_topic_nodes | included_item_nodes
            and edge["toNode"] in included_topic_nodes | included_item_nodes
        ],
    }
    (maps_dir / "xbrain-map.generated.canvas").write_text(
        json.dumps(canvas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _storage_labels(storage: dict[str, dict[str, int | str]]) -> dict[str, tuple[str, ...]]:
    return {
        item_id: (
            str(row.get("record_label", "")),
            str(row.get("stored_files_label", "")),
            str(row.get("note_label", "")),
        )
        for item_id, row in storage.items()
    }


def _remove_stale_item_notes(items_dir: Path, items: list[Item]) -> int:
    """Delete item notes no longer present in the rendered corpus."""
    expected = {note_filename(item) for item in items if _has_note(item)}
    removed = 0
    for path in items_dir.glob("*.md"):
        if path.name not in expected:
            path.unlink()
            removed += 1
    return removed


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


def _write_note(
    items_dir: Path,
    item: Item,
    strings: Strings,
    topic_style: str,
    storage: dict[str, int | str] | None = None,
) -> None:
    """Write an item's note, replacing only the generated region.

    The filename ends with the item's globally unique ``id``. That makes
    every note path collision-free and lets us locate a note written for
    this item under a previous title or date: that stale note is migrated
    so the user's hand-written tail follows the item instead of being
    orphaned.
    """
    path = items_dir / note_filename(item)
    block = wrap(_render_note(item, strings, topic_style, storage))
    source = path if path.exists() else _stale_note(items_dir, item, path)
    if source is not None:
        tail = user_tail(source.read_text(encoding="utf-8"), DEFAULT_TAIL)
        if source != path:
            source.unlink()
            logger.info("Migrated note %s -> %s", source.name, path.name)
    else:
        tail = DEFAULT_TAIL
    path.write_text(block + tail, encoding="utf-8")


def _human_bytes(size: int) -> str:
    """Compact byte label for generated notes and dashboard storage payloads."""
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            if value < 10:
                return f"{value:.1f} {unit}"
            return f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.0f} GB"


def _local_media_size(media_root: Path | None, local_path: str) -> int:
    if media_root is None:
        return 0
    path = media_root / local_path
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _item_media_file_bytes(item: Item, media_root: Path | None) -> int:
    total = 0
    for entry in item.media:
        if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
            total += _local_media_size(media_root, entry.local_path)
    if item.content is None:
        return total
    for source in item.content.sources:
        if not (isinstance(source, ContentSourceSuccess) and source.kind == "x_article"):
            continue
        for block in source.blocks:
            if isinstance(block, ArticleImageBlock) and isinstance(
                block.media, (MediaPhotoDownloaded, MediaPhotoDescribed)
            ):
                total += _local_media_size(media_root, block.media.local_path)
    return total


def _item_record_bytes(item: Item) -> int:
    return len(item.model_dump_json().encode("utf-8"))


def _path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _generated_markdown_bytes(path: Path) -> int:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return 0
    end = text.find(GEN_END)
    if end == -1:
        return len(text.encode("utf-8"))
    return len(text[: end + len(GEN_END)].encode("utf-8"))


def _item_video_artifact_bytes(item: Item, videos_dir: Path | None) -> tuple[int, int]:
    if videos_dir is None:
        return 0, 0
    summary = transcript = 0
    for source in _video_sources(item):
        if not source.raw_transcript:
            continue
        folder = videos_dir / _video_folder_name(item, source)
        summary += _path_size(folder / "summary.md")
        transcript += _path_size(folder / "transcript.md")
    return summary, transcript


def _item_storage_map(
    items: list[Item],
    items_dir: Path,
    media_root: Path | None,
    videos_dir: Path | None = None,
) -> dict[str, dict[str, int | str]]:
    out: dict[str, dict[str, int | str]] = {}
    for item in items:
        if not _has_note(item):
            continue
        note_bytes = _generated_markdown_bytes(items_dir / note_filename(item))
        record_bytes = _item_record_bytes(item)
        stored_files_bytes = _item_media_file_bytes(item, media_root)
        video_summary_bytes, video_transcript_bytes = _item_video_artifact_bytes(item, videos_dir)
        total_bytes = (
            note_bytes
            + record_bytes
            + stored_files_bytes
            + video_summary_bytes
            + video_transcript_bytes
        )
        out[item.id] = {
            "record_bytes": record_bytes,
            "stored_files_bytes": stored_files_bytes,
            "note_bytes": note_bytes,
            "video_summary_bytes": video_summary_bytes,
            "video_transcript_bytes": video_transcript_bytes,
            "total_bytes": total_bytes,
            "record_label": _human_bytes(record_bytes),
            "stored_files_label": _human_bytes(stored_files_bytes),
            "note_label": _human_bytes(note_bytes),
            "video_summary_label": _human_bytes(video_summary_bytes),
            "video_transcript_label": _human_bytes(video_transcript_bytes),
            "total_label": _human_bytes(total_bytes),
        }
    return out


def _iter_regular_files(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    return [path for path in root.rglob("*") if path.is_file()]


def _sum_files(paths: list[Path]) -> int:
    return sum(_path_size(path) for path in paths)


def _partition_data_files(data_dir: Path | None) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {"media": [], "snapshots": [], "store": [], "other": []}
    if data_dir is None or not data_dir.exists():
        return buckets
    for path in _iter_regular_files(data_dir):
        rel = path.relative_to(data_dir)
        first = rel.parts[0] if rel.parts else ""
        if first == "media":
            buckets["media"].append(path)
        elif first == "snapshots":
            buckets["snapshots"].append(path)
        elif path.suffix in {".json", ".yaml", ".yml"}:
            buckets["store"].append(path)
        else:
            buckets["other"].append(path)
    return buckets


def _output_media_files(output_files: list[Path], output_dir: Path) -> list[Path]:
    media_root = (output_dir / _VAULT_MEDIA_SUBDIR).resolve(strict=False)
    return [path for path in output_files if path.resolve(strict=False).is_relative_to(media_root)]


def _storage_categories(
    *,
    store_bytes: int,
    media_bytes: int,
    snapshot_bytes: int,
    note_bytes: int,
    dashboard_bytes: int,
    mirrored_media_bytes: int,
    other_data_bytes: int,
) -> list[dict[str, int | str]]:
    categories: list[dict[str, int | str]] = [
        {"label": "Store JSON/YAML", "bytes": store_bytes, "value": _human_bytes(store_bytes)},
        {"label": "Stored media", "bytes": media_bytes, "value": _human_bytes(media_bytes)},
        {"label": "Snapshots", "bytes": snapshot_bytes, "value": _human_bytes(snapshot_bytes)},
        {"label": "Generated notes", "bytes": note_bytes, "value": _human_bytes(note_bytes)},
        {
            "label": "Dashboard HTML",
            "bytes": dashboard_bytes,
            "value": _human_bytes(dashboard_bytes),
        },
    ]
    if mirrored_media_bytes:
        categories.append(
            {
                "label": "Vault media mirror",
                "bytes": mirrored_media_bytes,
                "value": _human_bytes(mirrored_media_bytes),
            }
        )
    if other_data_bytes:
        categories.append(
            {
                "label": "Other data",
                "bytes": other_data_bytes,
                "value": _human_bytes(other_data_bytes),
            }
        )
    return categories


def _storage_summary(
    data_dir: Path | None, output_dir: Path
) -> dict[str, int | str | list[dict[str, int | str]]]:
    """Global storage footprint shown in the dashboard.

    `data_dir` carries JSON/YAML records and snapshots. Media may live there in
    local mode (`data/media`) or directly in the generated vault in Drive mode
    (`<output_dir>/_media`) to avoid duplicate Drive storage.
    """
    data_files = _partition_data_files(data_dir)
    output_files = _iter_regular_files(output_dir)
    output_media_files = _output_media_files(output_files, output_dir)
    data_media_files = data_files["media"]
    media_files = data_media_files if data_media_files else output_media_files
    mirrored_media_files = output_media_files if data_media_files else []
    media_storage = "data/media" if data_media_files else "vault/_media"
    if not media_files:
        media_storage = "none"
    output_media_paths = set(output_media_files)
    note_files = [
        path for path in output_files if path.suffix == ".md" and path not in output_media_paths
    ]
    dashboard_files = [
        path
        for path in output_files
        if path.name == "dashboard.html" and path not in output_media_paths
    ]
    store_bytes = _sum_files(data_files["store"])
    media_bytes = _sum_files(media_files)
    mirrored_media_bytes = _sum_files(mirrored_media_files)
    snapshot_bytes = _sum_files(data_files["snapshots"])
    other_data_bytes = _sum_files(data_files["other"])
    note_bytes = _sum_files(note_files)
    dashboard_bytes = _sum_files(dashboard_files)
    data_media_bytes = _sum_files(data_media_files)
    output_media_bytes = _sum_files(output_media_files)
    data_bytes = store_bytes + data_media_bytes + snapshot_bytes + other_data_bytes
    vault_bytes = note_bytes + dashboard_bytes + output_media_bytes
    total_bytes = data_bytes + vault_bytes
    categories = _storage_categories(
        store_bytes=store_bytes,
        media_bytes=media_bytes,
        snapshot_bytes=snapshot_bytes,
        note_bytes=note_bytes,
        dashboard_bytes=dashboard_bytes,
        mirrored_media_bytes=mirrored_media_bytes,
        other_data_bytes=other_data_bytes,
    )
    return {
        "total_bytes": total_bytes,
        "data_bytes": data_bytes,
        "vault_bytes": vault_bytes,
        "store_bytes": store_bytes,
        "media_bytes": media_bytes,
        "media_storage": media_storage,
        "mirrored_media_bytes": mirrored_media_bytes,
        "snapshot_bytes": snapshot_bytes,
        "note_bytes": note_bytes,
        "dashboard_bytes": dashboard_bytes,
        "total_label": _human_bytes(total_bytes),
        "data_label": _human_bytes(data_bytes),
        "vault_label": _human_bytes(vault_bytes),
        "store_label": _human_bytes(store_bytes),
        "media_label": _human_bytes(media_bytes),
        "media_storage_label": media_storage,
        "mirrored_media_label": _human_bytes(mirrored_media_bytes),
        "snapshot_label": _human_bytes(snapshot_bytes),
        "note_label": _human_bytes(note_bytes),
        "dashboard_label": _human_bytes(dashboard_bytes),
        "categories": categories,
    }


def _video_sources(item: Item) -> list[ContentSourceSuccess]:
    if item.content is None:
        return []
    return [
        source
        for source in item.content.sources
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_video"
    ]


def _write_text_artifact(path: Path, body: str) -> None:
    """Write a fully generated text artifact, preserving any user tail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_tail = user_tail(path.read_text(encoding="utf-8"), "\n") if path.exists() else "\n"
    path.write_text(wrap(body) + existing_tail, encoding="utf-8")


def _write_video_artifacts(items: list[Item], videos_dir: Path, strings: Strings) -> None:
    """Write per-video summary/transcript markdown under `videos/`.

    The dashboard chat indexes only `items/` and `topics/`, so the raw transcript
    files are available for reading/audit without entering retrieval context.
    """
    for item in items:
        for source in _video_sources(item):
            if not source.raw_transcript:
                continue
            folder = videos_dir / _video_folder_name(item, source)
            _write_text_artifact(
                folder / "summary.md", _render_video_summary_artifact(item, source)
            )
            _write_text_artifact(
                folder / "transcript.md",
                _render_video_transcript_artifact(item, source, strings),
            )


def _video_artifact_frontmatter(
    item: Item, source: ContentSourceSuccess, *, kind: str, exclude: bool = False
) -> str:
    tags = ["x-knowledge", "video"]
    if item.enriched:
        tags += item.enriched.topics
    fields = [
        "---",
        f'id: "{item.id}"',
        f"kind: {kind}",
        f"url: {item.url}",
        f"video_source: {source.url}",
        f"created: {item.created_at.date().isoformat()}",
        f"author: {item.author.handle}",
        f"tags: [{', '.join(dict.fromkeys(tags))}]",
    ]
    if source.language:
        fields.append(f"language: {source.language}")
    if exclude:
        fields.append("xbrain_exclude: true")
    fields.append("---")
    return "\n".join(fields)


def _render_video_summary_artifact(item: Item, source: ContentSourceSuccess) -> str:
    title = source.title or title_of(item)
    transcript_link = "transcript.md" if source.raw_transcript else ""
    lines = [
        _video_artifact_frontmatter(item, source, kind="video_summary"),
        "",
        f"# {title}",
        "",
        f"[Tweet original]({item.url})",
    ]
    if transcript_link:
        lines.append(f"[Transcripción original]({transcript_link})")
    lines += ["", source.text]
    return "\n".join(lines).rstrip()


def _render_video_transcript_artifact(
    item: Item, source: ContentSourceSuccess, strings: Strings
) -> str:
    title = source.title or title_of(item)
    summary_link = "summary.md"
    lines = [
        _video_artifact_frontmatter(item, source, kind="video_transcript", exclude=True),
        "",
        f"# Original Transcript: {title}",
        "",
        "> Referencia raw. Este documento no se usa para dashboard, topics ni Ask XBrain.",
        "",
        f"[{strings.video_digest_header}]({summary_link}) · [Tweet original]({item.url})",
    ]
    if source.raw_transcript_url:
        lines += ["", f"Transcript source: <{source.raw_transcript_url}>"]
    lines += ["", "## Transcript", "", source.raw_transcript or ""]
    return "\n".join(lines).rstrip()


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
    - `MediaPhotoDownloaded` / `MediaPhotoDescribed`
      → Obsidian embed `![[_media/<id>/<n>.<ext>]]`. The vault is
      self-contained: local mode mirrors from `data/media`, while Drive mode
      stores the canonical bytes directly in `<output_dir>/_media`, so the
      embed resolves with no user configuration. The described variant inherits the same
      on-disk file — the description is consumed by the LLM prompts in `executors/api.py` /
      `topic_synth.py`, NOT shown as alt-text in this phase. Decorative
      photos are still embedded; the `is_decorative` flag only filters
      them out of the LLM prompts, never out of the visual rendering.
    - `MediaPhotoFailed` / `MediaVideoFailed` → one-line ⚠ warning carrying
      the failure reason and the original URL — visible evidence, not a
      silent drop.
    - `MediaPhotoPending`     → silent. Not an error, just "the next
      `xbrain media` run will pick it up".
    - `MediaVideoPending` / `MediaVideoDownloaded`
      → a clickable "Ver vídeo" external link only. XBrain never embeds or
      mirrors video bytes in the vault.

    The output is intentionally plain markdown; the caller (`_render_note`)
    wraps it in a blank line on either side for readability.
    """
    lines: list[str] = []
    for entry in item.media:
        if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
            lines.append(f"![[{_VAULT_MEDIA_SUBDIR}/{entry.local_path}]]")
            # A described (non-decorative) photo carries a short vision caption
            # under the embed. Full OCR/transcribed text stays in the JSON store;
            # rendering it after every image makes the note noisy.
            # One `>` per physical line: Markdown blockquotes scope to a single
            # line, so a multi-line description must re-prefix every line or the
            # trailing lines leak into the note body (worst case: a line that
            # starts with `#`/`-`/`![[` injects unintended structure).
            if isinstance(entry, MediaPhotoDescribed):
                lines.extend(_described_photo_text_lines(entry))
        elif isinstance(entry, MediaPhotoFailed):
            reason = _FAILURE_ES_MEDIA.get(entry.failure_reason, entry.failure_reason)
            lines.append(f"> ⚠ Foto no disponible ({reason}): <{entry.url}>")
        elif isinstance(entry, MediaVideoFailed):
            reason = _FAILURE_ES_MEDIA.get(entry.failure_reason, entry.failure_reason)
            lines.append(f"> ⚠ Vídeo no disponible ({reason}): <{entry.url}>")
        elif isinstance(entry, MediaPhotoPending):
            # Silent: a future `xbrain media` run will advance this entry.
            continue
        elif isinstance(entry, (MediaVideoPending, MediaVideoDownloaded)):
            # `entry.url` is the playable stream (mp4 or HLS), not the poster,
            # so surface it as a clickable link; bytes are never embedded.
            lines.append(f"> 🎥 [Ver vídeo externo]({entry.url}) (no se descarga)")
        else:
            assert_never(entry)
    return lines


def _described_photo_text_lines(entry: MediaPhotoDescribed) -> list[str]:
    """Render a described image's short visible caption."""
    lines: list[str] = []
    if entry.description:
        lines.extend(f"> {line}" for line in entry.description.splitlines())
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


def _mirror_file(item_id: str, source: Path, destination: Path) -> None:
    """Ensure one media file is available in the vault's `_media/` tree.

    Local mode copies from `data/media`. Drive mode commonly passes the same
    path for source and destination because `_media` is canonical there; that is
    treated as already mirrored. Missing bytes warn but do not abort generation,
    so the note renders a visible broken embed instead of losing the whole run.
    """
    if not source.exists():
        logger.warning(
            "Media bytes missing for item %s at %s — embed will render broken.",
            item_id,
            source,
        )
        return
    if source.resolve(strict=False) == destination.resolve(strict=False):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _mirror_item_media(item: Item, media_root: Path, vault_media_dir: Path) -> None:
    """Ensure every downloaded photo on `item` exists in the vault `_media` tree.

    `media_root` is `data/media` in local mode and `<output_dir>/_media` in
    Drive mode. The same function handles both by skipping self-copies.
    """
    for entry in item.media:
        # The described variant inherits the on-disk bytes from the prior
        # downloaded state. Video bytes are intentionally excluded.
        if not isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
            continue
        _mirror_file(item.id, media_root / entry.local_path, vault_media_dir / entry.local_path)


def _mirror_item_frames(item: Item, media_root: Path, vault_media_dir: Path) -> None:
    """Mirror every `x_video` key-frame slide on `item` into the vault (#44 PR4).

    Slides are stored at `data/media/<id>/frames/<n>.<ext>` (persisted by
    `digest-video --frames`) and mirrored to `<output_dir>/_media/<id>/frames/…`
    exactly like a downloaded photo, so the `![[_media/…]]` embed in the Video
    digest section resolves in a self-contained vault. A missing byte renders a
    broken embed (via `_mirror_file`), never a crash.
    """
    if item.content is None:
        return
    for source in item.content.sources:
        if not isinstance(source, ContentSourceSuccess):
            continue
        for frame in source.frames:
            _mirror_file(item.id, media_root / frame.local_path, vault_media_dir / frame.local_path)


def _mirror_item_article_images(item: Item, media_root: Path, vault_media_dir: Path) -> None:
    """Ensure every downloaded inline Article image is available in `_media`.

    X long-form Article images live outside `item.media`, on the `x_article`
    blocks as `ArticleImageBlock`s. The stored `local_path` is copied verbatim
    under `_media`; in Drive mode it already lives there and the self-copy is
    skipped.
    """
    if item.content is None:
        return
    for source in item.content.sources:
        if not (isinstance(source, ContentSourceSuccess) and source.kind == "x_article"):
            continue
        for block in source.blocks:
            if not isinstance(block, ArticleImageBlock):
                continue
            entry = block.media
            # Only the on-disk states (downloaded / described) carry a
            # `local_path` to mirror; pending/failed/video variants have no bytes.
            if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
                _mirror_file(
                    item.id, media_root / entry.local_path, vault_media_dir / entry.local_path
                )


def _video_folder_name(item: Item, source: ContentSourceSuccess) -> str:
    """Stable generated folder for one video's text artifacts."""
    title = source.title or title_of(item)
    return f"{item.created_at.date().isoformat()}-{slugify(title)}-{item.id}"


def _video_digest_lines(item: Item, source: ContentSourceSuccess, strings: Strings) -> list[str]:
    """Render an `x_video` source as an executive-summary section.

    `source.text` is the summary used by dashboard/enrich/topics. The raw
    transcript is linked as a generated artifact under `videos/`, but its text is
    deliberately not included here so Ask XBrain cannot retrieve it from item
    notes.
    """
    content_text = video_content_text(source)
    if not content_text:
        return [f"> {strings.silent_video}", ""]
    heading = source.title or source.url
    lines = [f"## {strings.video_digest_header}: {heading}", ""]
    if source.raw_transcript:
        folder = _video_folder_name(item, source)
        lines += [
            f"[Resumen ejecutivo](../videos/{folder}/summary.md) · "
            f"[Transcripción original](../videos/{folder}/transcript.md)",
            "",
        ]
    lines += [content_text, ""]
    return lines


def _article_image_lines(block: ArticleImageBlock) -> list[str]:
    """Render one inline Article image block (#39 PR5) — embed, warning, or silent.

    Mirrors the photo convention in `_render_media_lines`:
    - `MediaPhotoDownloaded` / `MediaPhotoDescribed` → the `![[_media/<id>/article/<n>.<ext>]]`
      embed (the STORED `local_path` carries the `article/` namespace), followed
      by any caption lines: the author's `alt` text and — for a described image —
      the vision description, each as `> …` blockquote lines (one `>` per physical
      line so a multi-line caption can't spill out of the blockquote).
    - `MediaPhotoFailed` → a one-line `⚠ Imagen no disponible (<reason>): <url>`
      note (reason via `_FAILURE_ES_MEDIA`) — visible evidence, never a silent drop.
    - `MediaPhotoPending` → silent (a future `xbrain media` run advances it).

    A video variant never appears on an article image (the PR3 producer only ever
    emits photo states); if a malformed record carries one, it is logged and
    skipped rather than crashing generation.
    """
    entry = block.media
    if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
        lines = [f"![[{_VAULT_MEDIA_SUBDIR}/{entry.local_path}]]"]
        lines += _article_caption_lines(block, entry)
        return lines
    if isinstance(entry, MediaPhotoFailed):
        reason = _FAILURE_ES_MEDIA.get(entry.failure_reason, entry.failure_reason)
        return [f"> ⚠ Imagen no disponible ({reason}): <{entry.url}>"]
    if isinstance(entry, MediaPhotoPending):
        return []  # Silent: a future `xbrain media` run will advance this image.
    logger.warning(
        "Article image carries an unexpected %s media variant; skipping its embed.",
        type(entry).__name__,
    )
    return []


def _article_caption_lines(
    block: ArticleImageBlock, entry: MediaPhotoDownloaded | MediaPhotoDescribed
) -> list[str]:
    """Caption lines under an inline Article image: the author's `alt` then, for a
    described image, its vision description — each `> …`, one line per physical line."""
    lines: list[str] = []
    if block.alt:
        lines += [f"> {line}" for line in block.alt.splitlines()]
    if isinstance(entry, MediaPhotoDescribed):
        described_lines = _described_photo_text_lines(entry)
        if lines and described_lines:
            lines.append("")
        lines += described_lines
    return lines


def _article_blocks_lines(source: ContentSourceSuccess, strings: Strings) -> list[str]:
    """Render an `x_article` source with structured `blocks` as a blogpost (#39 PR5).

    Walks `source.blocks` IN ORDER under a `## <content_header>: <title>` heading:
    each `ArticleTextBlock` becomes a body paragraph (with the baked `\\n\\n`
    separator stripped — see `ARTICLE_PARAGRAPH_SEP`), each `ArticleImageBlock`
    an inline `![[_media/…]]` embed (or a warning / silence) via `_article_image_lines`.
    The result reads as authored — text and images interleaved where the author
    placed them. Only called for a NON-empty `blocks`; the empty-`blocks`
    (trafilatura fallback) path renders `source.text` in `_content_lines`.

    The body is computed first: if every block renders to nothing (e.g. an
    image-only Article whose sole image is still `MediaPhotoPending` — the normal
    post-`fetch`/pre-`media` state), the bare `## <content_header>:` heading is
    NOT emitted, mirroring how `_video_digest_lines` avoids an empty digest block.
    """
    body: list[str] = []
    for block in source.blocks:
        if isinstance(block, ArticleTextBlock):
            text = block.text.removeprefix(ARTICLE_PARAGRAPH_SEP)
            if text:
                body += [text, ""]
        else:
            image_lines = _article_image_lines(block)
            if image_lines:
                body += image_lines
                body.append("")
    if not body:
        return []
    heading = source.title or source.url
    return [f"## {strings.content_header}: {heading}", "", *body]


def _content_lines(item: Item, strings: Strings) -> list[str]:
    """Rendered article bodies + broken-link evidence for a fetched item.

    Switches on the `ContentSource` variant: the success variant is
    rendered as a content block; the failure variant is rendered as a
    broken-link line *only* for external articles and X articles (a
    failed thread fetch is silently elided, matching the pre-refactor
    behaviour — `source.kind` is what guarded that path before). An
    `x_video` success is rendered as a `Video digest` section rather than
    a generic content block (#44); an `x_article` success with structured
    `blocks` renders as an ordered blogpost (text + inline image embeds)
    rather than a plain text block (#39 PR5), while an `x_article` with
    empty `blocks` (trafilatura fallback) keeps the plain `source.text`
    block — byte-unchanged.
    """
    lines: list[str] = []
    content = item.content
    if content is None:
        return lines
    for source in content.sources:
        if isinstance(source, ContentSourceSuccess):
            if source.kind == "x_video":
                lines += _video_digest_lines(item, source, strings)
            elif source.kind == "x_article" and source.blocks:
                # Structured Article (#39): render the ordered text+image blocks
                # as a blogpost. An `x_article` with EMPTY blocks (trafilatura
                # fallback, or a pre-#39 record) falls through to the plain
                # `source.text` path below — byte-unchanged, no regression.
                lines += _article_blocks_lines(source, strings)
            else:
                heading = source.title or source.url
                lines += [f"## {strings.content_header}: {heading}", "", source.text, ""]
        elif source.kind in ("external_article", "x_article"):
            lines += [_broken_link_line(source, content.fetched_at), ""]
    return lines


def _storage_note_lines(storage: dict[str, int | str] | None) -> list[str]:
    if not storage:
        return []
    return [
        (
            "> XBrain storage: "
            f"DB record {storage['record_label']} · "
            f"files {storage['stored_files_label']} · "
            f"note {storage['note_label']}"
        ),
        "",
    ]


def _render_note(
    item: Item,
    strings: Strings,
    topic_style: str,
    storage: dict[str, int | str] | None = None,
) -> str:
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
    lines += _storage_note_lines(storage)
    if item.content:
        lines += _content_lines(item, strings)
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


def _render_index(items: list[Item], strings: Strings, dashboard_href: str) -> str:
    """Render the top-level index note: corpus stats and the topic list.

    `dashboard_href` is the absolute ``file://`` URI of ``dashboard.html`` so the
    index link opens the self-contained dashboard in the external browser (see
    `generate`); Obsidian neither lists nor renders the raw ``.html`` itself.
    """
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
        f"- [📊 Dashboard interactivo]({dashboard_href}) — métricas, drill-down y enlaces (se abre en el navegador)",
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
