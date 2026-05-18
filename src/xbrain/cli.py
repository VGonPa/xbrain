"""Command-line interface for XBrain."""
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
from xbrain.enrich import (apply_worksheet_judgments, enrich_with_executor,
                           items_pending_enrichment)
from xbrain.executors.api import ApiExecutor
from xbrain.extract.browser import login as run_login
from xbrain.extract.browser import x_context
from xbrain.extract.extractor import extract_source
from xbrain.extract.threads import expand_threads
from xbrain.fetch import fetch_pending
from xbrain.generate import generate as run_generate
from xbrain.models import ArchiveImport, Author
from xbrain.rubrics import load_vocab, save_vocab
from xbrain.store import load_state, load_store, merge_items, save_state, save_store
from xbrain.vocab import induce_vocab
from xbrain.worksheet import export_worksheet, import_worksheet

app = typer.Typer(help="XBrain — bookmarks y tweets de X a un wiki de Obsidian")

_BOOKMARKS_URL = "https://x.com/i/bookmarks"


class Source(str, enum.Enum):
    bookmarks = "bookmarks"
    tweets = "tweets"
    all = "all"


def _repo_root() -> Path:
    """Repo root — overridable via XBRAIN_REPO_ROOT for tests."""
    override = os.environ.get("XBRAIN_REPO_ROOT")
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


def _report_invalid(invalid: list[tuple[str, list[str]]]) -> None:
    if invalid:
        typer.echo(f"Rechazados por el validador: {len(invalid)}", err=True)
        for item_id, errors in invalid:
            typer.echo(f"  {item_id}: {'; '.join(errors)}", err=True)


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
    executor: str = typer.Option(
        None, help="api | manual | claude-code (default: the enrich executor set in config.toml)"),
    apply: Path = typer.Option(
        None, "--apply", help="Import a filled worksheet and apply it"),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date, e.g. 2025-12-31"),
) -> None:
    """Enriquece los items con resumen + topics."""
    cfg = _config()
    store = load_store(cfg.items_path)
    vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")

    if apply is not None:
        if not vocab_topics:
            raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
        enriched, invalid = apply_worksheet_judgments(
            store, import_worksheet(apply), vocab_topics)
        save_store(store, cfg.items_path)
        typer.echo(f"Worksheet aplicada: {enriched} items enriquecidos")
        _report_invalid(invalid)
        return

    if not vocab_topics:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    chosen = executor or cfg.enrich_executor

    if chosen in ("manual", "claude-code"):
        pending = items_pending_enrichment(
            store, _parse_date(since), _parse_date(until))
        if not pending:
            typer.echo("No hay items pendientes de enriquecer.")
            return
        worksheet = cfg.data_dir / "enrich-worksheet.json"
        export_worksheet(pending, vocab_topics, worksheet)
        typer.echo(
            f"{len(pending)} items exportados a {worksheet}\n"
            f"Rellena el array `judgments` (con Claude Code o a mano) y ejecuta:\n"
            f"  xbrain enrich --apply {worksheet}")
        return

    if chosen != "api":
        raise ValueError(f"Ejecutor desconocido: {chosen!r}")

    enriched, invalid = enrich_with_executor(
        store, ApiExecutor(model=cfg.enrich_model), vocab_topics,
        _parse_date(since), _parse_date(until))
    save_store(store, cfg.items_path)
    typer.echo(f"Enriquecidos: {enriched} items")
    _report_invalid(invalid)


@app.command()
@_handle_cli_errors
def vocab(
    regenerate: bool = typer.Option(
        False, help="Regenerate the taxonomy and re-enrich every item"),
) -> None:
    """Induce the topic vocabulary (data/vocab.yaml) from the corpus."""
    cfg = _config()
    store = load_store(cfg.items_path)
    if not store:
        raise RuntimeError("El store está vacío — ejecuta `xbrain extract` antes.")
    topics = induce_vocab(store, cfg.vocab_target_count, cfg.enrich_model)
    save_vocab(topics, cfg.data_dir / "vocab.yaml")
    if regenerate:
        for item in store.values():
            item.enriched = None
        save_store(store, cfg.items_path)
        typer.echo("Todos los items marcados para re-enriquecer.")
    typer.echo(f"Vocabulario inducido: {len(topics)} topics "
               f"→ {cfg.data_dir / 'vocab.yaml'}")


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
