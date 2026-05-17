"""Command-line interface for X Knowledge Base."""
from __future__ import annotations

import enum
import functools
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import typer

from xbrain.archive import parse_archive
from xbrain.config import Config, load_config
from xbrain.enrich import enrich as run_enrich
from xbrain.extract.browser import login as run_login
from xbrain.extract.browser import x_context
from xbrain.extract.extractor import extract_source
from xbrain.extract.threads import expand_threads
from xbrain.fetch import fetch_pending
from xbrain.generate import generate as run_generate
from xbrain.models import ArchiveImport, Author
from xbrain.store import load_state, load_store, merge_items, save_state, save_store

app = typer.Typer(help="X Knowledge Base — bookmarks y tweets de X a un wiki de Obsidian")

_BOOKMARKS_URL = "https://x.com/i/bookmarks"


class Source(str, enum.Enum):
    bookmarks = "bookmarks"
    tweets = "tweets"
    all = "all"


def _repo_root() -> Path:
    """Repo root — overridable via XKB_REPO_ROOT for tests."""
    override = os.environ.get("XKB_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def _config() -> Config:
    return load_config(_repo_root())


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


_OPERATOR_ERRORS = (
    FileNotFoundError,
    ValueError,
    KeyError,
    RuntimeError,
    NotImplementedError,
)


def _handle_cli_errors(func: Callable) -> Callable:
    """Surface expected operator errors as a clean message + exit code 1."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except _OPERATOR_ERRORS as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    return wrapper


def _run_extract(cfg: Config, source: str,
                  since: datetime | None, until: datetime | None) -> None:
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    targets = {
        "bookmark": _BOOKMARKS_URL,
        "own_tweet": f"https://x.com/{cfg.x_handle}",
    }
    chosen = {
        "bookmarks": ["bookmark"],
        "tweets": ["own_tweet"],
        "all": ["bookmark", "own_tweet"],
    }[source]
    known_ids = set(store)
    with x_context(cfg.storage_state_path) as context:
        for src in chosen:
            cursor = state.bookmarks if src == "bookmark" else state.own_tweets
            first_run = cursor.last_seen_id is None
            items = extract_source(context, src, targets[src], known_ids,
                                   since, until)
            if not items and first_run:
                typer.echo(
                    f"AVISO: {src} devolvió 0 items en una extracción inicial — "
                    "revisa la sesión de X o el parser GraphQL (spec §6).",
                    err=True,
                )
            added = merge_items(store, items)
            if items:
                cursor.last_seen_id = max(items, key=lambda i: int(i.id)).id
            cursor.last_run = datetime.now(timezone.utc)
            typer.echo(f"{src}: {added} nuevos items")
    save_store(store, cfg.items_path)
    save_state(state, cfg.state_path)


def _run_fetch(cfg: Config, since: datetime | None,
               until: datetime | None, force: bool) -> None:
    store = load_store(cfg.items_path)
    articles = fetch_pending(store, since, until, force)
    threads = expand_threads(store, cfg.storage_state_path, force)
    save_store(store, cfg.items_path)
    typer.echo(f"Contenido descargado: {articles} artículos, {threads} hilos")


def _run_generate(cfg: Config, since: datetime | None,
                  until: datetime | None) -> None:
    store = load_store(cfg.items_path)
    run_generate(store, cfg.output_dir, since, until)
    typer.echo(f"Markdown generado en {cfg.output_dir}")


@app.command()
@_handle_cli_errors
def login() -> None:
    """Abre un navegador para iniciar sesión en X y guarda la sesión."""
    run_login(_config().storage_state_path)


@app.command()
@_handle_cli_errors
def extract(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date, e.g. 2025-12-31"),
) -> None:
    """Extrae bookmarks y/o tweets propios desde X."""
    _run_extract(_config(), source.value, _parse_date(since), _parse_date(until))


@app.command(name="import-archive")
@_handle_cli_errors
def import_archive(zip_path: Path) -> None:
    """Backfill del histórico de tweets desde el archivo oficial de X."""
    cfg = _config()
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    author = Author(handle=cfg.x_handle, name=cfg.x_handle)
    added = merge_items(store, parse_archive(zip_path, author))
    state.archive_imported = ArchiveImport(file=zip_path.name, at=datetime.now(timezone.utc))
    save_store(store, cfg.items_path)
    save_state(state, cfg.state_path)
    typer.echo(f"Archivo importado: {added} tweets nuevos")


@app.command()
@_handle_cli_errors
def fetch(
    since: str = typer.Option(None),
    until: str = typer.Option(None),
    force: bool = typer.Option(False, help="Volver a descargar lo ya descargado"),
) -> None:
    """Descarga el contenido de los artículos enlazados."""
    _run_fetch(_config(), _parse_date(since), _parse_date(until), force)


@app.command()
@_handle_cli_errors
def enrich(
    executor: str = typer.Option("manual", help="manual | api | claude-code"),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date, e.g. 2025-12-31"),
) -> None:
    """Enriquecimiento con LLM (en pausa — ver spec §9)."""
    pending = run_enrich(
        load_store(_config().items_path), executor,
        _parse_date(since), _parse_date(until),
    )
    typer.echo(f"{len(pending)} items pendientes de enriquecer")


@app.command()
@_handle_cli_errors
def generate(
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date, e.g. 2025-12-31"),
) -> None:
    """Genera las notas markdown en el vault."""
    _run_generate(_config(), _parse_date(since), _parse_date(until))


@app.command()
@_handle_cli_errors
def sync() -> None:
    """extract + fetch + generate en orden."""
    cfg = _config()
    _run_extract(cfg, "all", None, None)
    _run_fetch(cfg, None, None, False)
    _run_generate(cfg, None, None)


@app.command()
@_handle_cli_errors
def status() -> None:
    """Muestra contadores y última ejecución."""
    cfg = _config()
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    typer.echo(f"Items: {len(store)}")
    typer.echo(f"  con enlace: {sum(1 for i in store.values() if i.links)}")
    typer.echo(f"  con contenido: {sum(1 for i in store.values() if i.content)}")
    typer.echo(f"  enriquecidos: {sum(1 for i in store.values() if i.enriched)}")
    typer.echo(f"  última extracción bookmarks: {state.bookmarks.last_run}")
    typer.echo(f"  última extracción tweets: {state.own_tweets.last_run}")


if __name__ == "__main__":
    app()
