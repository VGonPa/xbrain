"""Email notifications for completed XBrain updates."""

from __future__ import annotations

import smtplib
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from html import escape
from pathlib import Path
from urllib.parse import urlparse

from xbrain.config import Config
from xbrain.models import Item
from xbrain.notes_io import note_filename


@dataclass(frozen=True)
class BookmarkEmailRow:
    author: str
    url: str
    short_url: str
    summary: str
    kilobytes: float


@dataclass(frozen=True)
class BookmarkUpdateEmail:
    command: str
    item_count: int
    updated_count: int
    rows: list[BookmarkEmailRow]
    drive_megabytes: float
    data_megabytes: float
    generated_at: str


def send_bookmark_update_email(
    cfg: Config,
    *,
    command: str,
    store: dict[str, Item],
    updated_item_ids: list[str],
) -> None:
    """Send a short update email through the configured SMTP account."""
    payload = build_bookmark_update_email(
        cfg,
        command=command,
        store=store,
        updated_item_ids=updated_item_ids,
    )
    message = EmailMessage()
    message["From"] = cfg.email_sender
    message["To"] = cfg.email_recipient
    message["Subject"] = (
        f"XBrain actualizado: {payload.updated_count} bookmarks nuevos/actualizados ({command})"
    )
    message.set_content(_render_text(payload, cfg))
    message.add_alternative(_render_html(payload, cfg), subtype="html")

    if cfg.email_smtp_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            cfg.email_smtp_host, cfg.email_smtp_port, context=context, timeout=30
        ) as smtp:
            smtp.login(cfg.email_smtp_username, cfg.email_smtp_password)
            smtp.send_message(message)
        return

    with smtplib.SMTP(cfg.email_smtp_host, cfg.email_smtp_port, timeout=30) as smtp:
        if cfg.email_smtp_starttls:
            smtp.starttls(context=ssl.create_default_context())
        smtp.login(cfg.email_smtp_username, cfg.email_smtp_password)
        smtp.send_message(message)


def build_bookmark_update_email(
    cfg: Config,
    *,
    command: str,
    store: dict[str, Item],
    updated_item_ids: list[str],
) -> BookmarkUpdateEmail:
    rows = [_row_for_item(cfg, store[item_id]) for item_id in updated_item_ids if item_id in store]
    return BookmarkUpdateEmail(
        command=command,
        item_count=len(store),
        updated_count=len(rows),
        rows=rows,
        drive_megabytes=_bytes_to_mb(_tree_bytes(cfg.drive_cache_dir)),
        data_megabytes=_bytes_to_mb(_tree_bytes(cfg.data_dir)),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _row_for_item(cfg: Config, item: Item) -> BookmarkEmailRow:
    return BookmarkEmailRow(
        author=_author_label(item),
        url=item.url,
        short_url=_short_url(item.url),
        summary=_summary(item),
        kilobytes=_bytes_to_kb(_item_bytes(cfg, item)),
    )


def _author_label(item: Item) -> str:
    name = item.author.name.strip()
    handle = item.author.handle.strip()
    if name and handle and name != handle:
        return f"{name} (@{handle})"
    if handle:
        return f"@{handle}"
    return name or "unknown"


def _summary(item: Item) -> str:
    if item.enriched is not None and item.enriched.summary:
        return item.enriched.summary
    text = " ".join(item.text.split())
    return text[:300] + ("..." if len(text) > 300 else "")


def _short_url(url: str) -> str:
    parsed = urlparse(url)
    label = f"{parsed.netloc}{parsed.path}" if parsed.netloc else url
    return label[:80] + ("..." if len(label) > 80 else "")


def _item_bytes(cfg: Config, item: Item) -> int:
    total = len(item.model_dump_json().encode("utf-8"))
    note = cfg.output_dir / "items" / note_filename(item)
    total += _path_bytes(note)
    seen: set[Path] = set()
    for local_path in _local_paths(item.model_dump(mode="json")):
        path = cfg.media_dir / local_path
        if path in seen:
            continue
        seen.add(path)
        total += _path_bytes(path)
    return total


def _local_paths(value: object) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "local_path" and isinstance(child, str):
                paths.append(child)
            else:
                paths.extend(_local_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_local_paths(child))
    return paths


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return _path_bytes(path)
    return sum(_path_bytes(child) for child in path.rglob("*") if child.is_file())


def _path_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _bytes_to_kb(size: int) -> float:
    return round(size / 1024, 1)


def _bytes_to_mb(size: int) -> float:
    return round(size / (1024 * 1024), 2)


def _render_text(payload: BookmarkUpdateEmail, cfg: Config) -> str:
    lines = [
        "XBrain ha completado una actualizacion de bookmarks.",
        "",
        "Metadatos",
        f"- Comando: {payload.command}",
        f"- Fecha UTC: {payload.generated_at}",
        f"- Items totales en biblioteca: {payload.item_count}",
        f"- Bookmarks nuevos/actualizados en esta ejecucion: {payload.updated_count}",
        f"- Google Drive root: {cfg.drive_root_folder_id}",
        f"- Cache local: {cfg.drive_cache_dir}",
        "",
        "Bookmarks actualizados",
    ]
    if not payload.rows:
        lines.append("- No se detectaron bookmarks nuevos en esta ejecucion.")
    for row in payload.rows:
        lines.extend(
            [
                f"- {row.author}",
                f"  Link: {row.short_url} ({row.url})",
                f"  Peso: {row.kilobytes:.1f} KB",
                f"  Resumen: {row.summary}",
            ]
        )
    lines.extend(
        [
            "",
            "Conclusiones",
            f"- Total ocupado por la biblioteca/cache de Google Drive: {payload.drive_megabytes:.2f} MB",
            f"- Total ocupado por data/ dentro de la biblioteca: {payload.data_megabytes:.2f} MB",
            "- El correo se envio porque [drive].enabled=true y [email].enabled=true.",
        ]
    )
    return "\n".join(lines)


def _render_html(payload: BookmarkUpdateEmail, cfg: Config) -> str:
    rows = "\n".join(
        (
            "<li>"
            f"<strong>{escape(row.author)}</strong><br>"
            f'<a href="{escape(row.url)}">{escape(row.short_url)}</a><br>'
            f"Peso: {row.kilobytes:.1f} KB<br>"
            f"Resumen: {escape(row.summary)}"
            "</li>"
        )
        for row in payload.rows
    )
    if not rows:
        rows = "<li>No se detectaron bookmarks nuevos en esta ejecucion.</li>"
    return f"""<!doctype html>
<html>
  <body>
    <h2>XBrain ha completado una actualizacion de bookmarks</h2>
    <h3>Metadatos</h3>
    <ul>
      <li>Comando: {escape(payload.command)}</li>
      <li>Fecha UTC: {escape(payload.generated_at)}</li>
      <li>Items totales en biblioteca: {payload.item_count}</li>
      <li>Bookmarks nuevos/actualizados: {payload.updated_count}</li>
      <li>Google Drive root: {escape(cfg.drive_root_folder_id)}</li>
      <li>Cache local: {escape(str(cfg.drive_cache_dir))}</li>
    </ul>
    <h3>Bookmarks actualizados</h3>
    <ol>
      {rows}
    </ol>
    <h3>Conclusiones</h3>
    <ul>
      <li>Total ocupado por la biblioteca/cache de Google Drive: {payload.drive_megabytes:.2f} MB</li>
      <li>Total ocupado por data/ dentro de la biblioteca: {payload.data_megabytes:.2f} MB</li>
    </ul>
  </body>
</html>"""
