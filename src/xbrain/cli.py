"""Command-line interface for XBrain."""

from __future__ import annotations

import enum
import functools
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import typer

from xbrain import snapshot
from xbrain.archive import parse_archive
from xbrain.config import Config, load_config
from xbrain.enrich import apply_worksheet_judgments, enrich_with_executor, items_pending_enrichment
from xbrain.executors.api import ApiExecutor
from xbrain.extract.browser import login as run_login
from xbrain.extract.browser import x_context
from xbrain.extract.extractor import extract_source
from xbrain.extract.threads import expand_threads
from xbrain.fetch import fetch_pending
from xbrain.fetch_x import fetch_x_articles
from xbrain.generate import generate as run_generate
from xbrain.models import ArchiveImport, Author, SourceName
from xbrain.rubrics import load_vocab, save_vocab
from xbrain.store import (
    load_state,
    load_store,
    load_topic_pages,
    merge_items,
    save_state,
    save_store,
    save_topic_pages,
)
from xbrain.topic_synth import (
    apply_overview_judgments,
    export_topic_worksheet,
    import_topic_worksheet,
    synthesize_overviews_api,
)
from xbrain.topics import (
    build_topic_inputs,
    compute_topic_posts,
    merge_overviews,
    topics_needing_synth,
    write_topic_pages,
)
from xbrain.vocab import (
    apply_vocab_worksheet,
    export_vocab_worksheet,
    import_vocab_worksheet,
    induce_vocab,
)
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
    # OSError covers PermissionError, FileExistsError, IsADirectoryError, etc.
    # The snapshot module hits these on permission or disk issues — they should
    # surface as a clean exit-1, not a raw traceback.
    OSError,
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


def _run_extract(cfg: Config, source: str, since: datetime | None, until: datetime | None) -> None:
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    targets = {
        "bookmark": _BOOKMARKS_URL,
        "own_tweet": f"https://x.com/{cfg.x_handle}",
    }
    source_sets: dict[str, list[SourceName]] = {
        "bookmarks": ["bookmark"],
        "tweets": ["own_tweet"],
        "all": ["bookmark", "own_tweet"],
    }
    chosen = source_sets[source]
    known_ids = set(store)
    with x_context(cfg.storage_state_path) as context:
        for src in chosen:
            cursor = state.bookmarks if src == "bookmark" else state.own_tweets
            first_run = cursor.last_seen_id is None
            items = extract_source(context, src, targets[src], known_ids, since, until)
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


def _auto_snapshot(cfg: Config, command: str) -> None:
    """Snapshot data/ before a destructive op and echo the path + item count.

    Called from every destructive code path (vocab --regenerate, topics
    --resynth, fetch --force). The manifest's `command` field carries the
    destructive op name (e.g. `vocab-regenerate`); the directory label uses
    the `pre-<op>` prefix so the listing is self-describing.

    Any failure here propagates and aborts the destructive op — a snapshot we
    can't take must not be silently skipped.
    """
    path, manifest = snapshot.snapshot_create(
        cfg.data_dir,
        command=command,
        dir_label=f"pre-{command}",
    )
    typer.echo(f"Snapshot created: {path.name} ({manifest.item_count} items)")


def _run_fetch(cfg: Config, since: datetime | None, until: datetime | None, force: bool) -> None:
    if force:
        _auto_snapshot(cfg, "fetch-force")
    store = load_store(cfg.items_path)
    try:
        articles = fetch_pending(store, since, until, force)
        x_articles = fetch_x_articles(store, cfg.storage_state_path, force, since, until)
        threads = expand_threads(store, cfg.storage_state_path, force)
    finally:
        # Persist whatever was fetched even if a later stage raised — a stage
        # error (e.g. an expired X session) must not discard in-memory work.
        save_store(store, cfg.items_path)
    typer.echo(f"Contenido descargado: {articles} artículos, {x_articles} de X, {threads} hilos")


def _run_generate(cfg: Config, since: datetime | None, until: datetime | None) -> None:
    store = load_store(cfg.items_path)
    run_generate(store, cfg.output_dir, since, until, cfg.output_language, cfg.topic_style)
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
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: the enrich executor set in config.toml)"
    ),
    apply: Path | None = typer.Option(
        None, "--apply", help="Import a filled worksheet and apply it"
    ),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date, e.g. 2025-12-31"),
) -> None:
    """Enriquece los items con resumen + topics."""
    cfg = _config()
    store = load_store(cfg.items_path)
    vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab_topics:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")

    if apply is not None:
        executor_name, judgments = import_worksheet(apply)
        enriched, invalid = apply_worksheet_judgments(store, judgments, vocab_topics, executor_name)
        save_store(store, cfg.items_path)
        typer.echo(f"Worksheet aplicada: {enriched} items enriquecidos")
        _report_invalid(invalid)
        return

    chosen = executor or cfg.enrich_executor

    if chosen in ("manual", "claude-code"):
        pending = items_pending_enrichment(store, _parse_date(since), _parse_date(until))
        if not pending:
            typer.echo("No hay items pendientes de enriquecer.")
            return
        worksheet = cfg.data_dir / "enrich-worksheet.json"
        export_worksheet(pending, vocab_topics, worksheet, chosen, cfg.output_language)
        typer.echo(
            f"{len(pending)} items exportados a {worksheet}\n"
            f"Rellena el array `judgments` (con Claude Code o a mano) y ejecuta:\n"
            f"  xbrain enrich --apply {worksheet}"
        )
        return

    if chosen != "api":
        raise ValueError(f"Ejecutor desconocido: {chosen!r}")

    enriched, invalid = enrich_with_executor(
        store,
        ApiExecutor(model=cfg.enrich_model, output_language=cfg.output_language),
        vocab_topics,
        _parse_date(since),
        _parse_date(until),
    )
    save_store(store, cfg.items_path)
    typer.echo(f"Enriquecidos: {enriched} items")
    _report_invalid(invalid)


def _mark_for_regenerate(store: dict, cfg: Config, regenerate: bool) -> None:
    """When `--regenerate` is set, drop every item's enrichment and persist."""
    if regenerate:
        for item in store.values():
            item.enriched = None
        save_store(store, cfg.items_path)
        typer.echo("Todos los items marcados para re-enriquecer.")


def _vocab_apply(cfg: Config, store: dict, apply: Path, regenerate: bool) -> None:
    """`xbrain vocab --apply` — import a filled vocab worksheet."""
    topics, invalid = apply_vocab_worksheet(import_vocab_worksheet(apply))
    _report_invalid(invalid)
    if not topics:
        raise RuntimeError("La worksheet no produjo ningún topic válido.")
    if regenerate:
        _auto_snapshot(cfg, "vocab-regenerate")
    # Mark the store first: a crash here leaves items pending (a re-run re-marks
    # idempotently) — safer than vocab.yaml updated while items stay stale.
    _mark_for_regenerate(store, cfg, regenerate)
    save_vocab(topics, cfg.data_dir / "vocab.yaml")
    typer.echo(f"Vocabulario aplicado: {len(topics)} topics → {cfg.data_dir / 'vocab.yaml'}")


def _vocab_run(cfg: Config, store: dict, executor: str | None, regenerate: bool) -> None:
    """`xbrain vocab` — induce the taxonomy (worksheet export, or `api`)."""
    chosen = executor or cfg.enrich_executor
    if chosen in ("manual", "claude-code"):
        worksheet = cfg.data_dir / "vocab-worksheet.json"
        export_vocab_worksheet(store, cfg.vocab_target_count, worksheet, cfg.output_language)
        regen = " --regenerate" if regenerate else ""
        typer.echo(
            f"Corpus exportado a {worksheet}\n"
            f"Induce la taxonomía (con Claude Code o a mano) y ejecuta:\n"
            f"  xbrain vocab --apply {worksheet}{regen}"
        )
        return
    if chosen != "api":
        raise ValueError(f"Ejecutor desconocido: {chosen!r}")
    if regenerate:
        _auto_snapshot(cfg, "vocab-regenerate")
    topics = induce_vocab(store, cfg.vocab_target_count, cfg.enrich_model, cfg.output_language)
    save_vocab(topics, cfg.data_dir / "vocab.yaml")
    _mark_for_regenerate(store, cfg, regenerate)
    typer.echo(f"Vocabulario inducido: {len(topics)} topics → {cfg.data_dir / 'vocab.yaml'}")


@app.command()
@_handle_cli_errors
def vocab(
    regenerate: bool = typer.Option(
        False, help="Marca todos los items para re-enriquecer contra la taxonomía nueva"
    ),
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: el de config.toml)"
    ),
    apply: Path | None = typer.Option(None, "--apply", help="Importar una vocab worksheet rellena"),
) -> None:
    """Induce el vocabulario de topics (data/vocab.yaml) desde el corpus."""
    cfg = _config()
    store = load_store(cfg.items_path)
    if not store:
        raise RuntimeError("El store está vacío — ejecuta `xbrain extract` antes.")
    if apply is not None:
        _vocab_apply(cfg, store, apply, regenerate)
    else:
        _vocab_run(cfg, store, executor, regenerate)


def _topics_apply(cfg: Config, store: dict, vocab: list, apply: Path) -> None:
    """`xbrain topics --apply` — import a filled overview worksheet."""
    pages = load_topic_pages(cfg.topics_path)
    posts = compute_topic_posts(store, vocab)
    valid, invalid = apply_overview_judgments(import_topic_worksheet(apply))
    merge_overviews(pages, valid, posts)
    save_topic_pages(pages, cfg.topics_path)
    written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
    typer.echo(f"Worksheet aplicada: {len(valid)} overviews · {written} páginas escritas")
    _report_invalid(invalid)


def _topics_run(cfg: Config, store: dict, vocab: list, resynth: bool, executor: str | None) -> None:
    """`xbrain topics` — update lists and (re)synthesize stale overviews."""
    if resynth:
        _auto_snapshot(cfg, "topics-resynth")
    pages = load_topic_pages(cfg.topics_path)
    posts = compute_topic_posts(store, vocab)
    stale = topics_needing_synth(vocab, posts, pages, cfg.topics_resynth_threshold, resynth)
    inputs = build_topic_inputs(stale, vocab, posts)

    if not inputs:
        written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
        typer.echo(f"Topic pages actualizadas: {written} páginas (sin overviews pendientes).")
        return

    chosen = executor or cfg.enrich_executor
    if chosen in ("manual", "claude-code"):
        worksheet = cfg.data_dir / "topic-worksheet.json"
        export_topic_worksheet(inputs, worksheet, cfg.output_language)
        written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
        typer.echo(
            f"{len(inputs)} topics exportados a {worksheet} · {written} páginas escritas\n"
            f"Rellena el array `judgments` y ejecuta:\n"
            f"  xbrain topics --apply {worksheet}"
        )
        return
    if chosen != "api":
        raise ValueError(f"Ejecutor desconocido: {chosen!r}")

    judgments = synthesize_overviews_api(inputs, cfg.enrich_model, cfg.output_language)
    merge_overviews(pages, judgments, posts)
    save_topic_pages(pages, cfg.topics_path)
    written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
    typer.echo(f"Topics sintetizados: {len(judgments)}/{len(inputs)} · {written} páginas escritas")


@app.command()
@_handle_cli_errors
def topics(
    resynth: bool = typer.Option(False, help="Re-sintetizar todos los overviews obsoletos"),
    apply: Path | None = typer.Option(
        None, "--apply", help="Importar un worksheet de overviews relleno"
    ),
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: el de config.toml)"
    ),
) -> None:
    """Genera las páginas de topic: listas de posts + overviews sintetizados."""
    cfg = _config()
    store = load_store(cfg.items_path)
    vocab = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab:
        raise RuntimeError("No hay vocabulario — ejecuta `xbrain vocab` antes.")
    if apply is not None:
        _topics_apply(cfg, store, vocab, apply)
    else:
        _topics_run(cfg, store, vocab, resynth, executor)


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


snapshot_app = typer.Typer(help="Gestionar snapshots de data/")
app.add_typer(snapshot_app, name="snapshot")


@snapshot_app.command("create")
@_handle_cli_errors
def snapshot_create_cmd(
    name: str | None = typer.Option(None, help="Optional directory label (default: 'manual')"),
) -> None:
    """Create a snapshot of data/ right now."""
    cfg = _config()
    path, manifest = snapshot.snapshot_create(
        cfg.data_dir,
        command="manual",
        dir_label=name,
    )
    typer.echo(f"Snapshot created: {path.name} ({manifest.item_count} items)")


@snapshot_app.command("list")
@_handle_cli_errors
def snapshot_list_cmd() -> None:
    """List snapshots, newest first. Corrupt entries surface as CORRUPT."""
    cfg = _config()
    rows = snapshot.snapshot_list(cfg.data_dir)
    if not rows:
        typer.echo("No snapshots.")
        return
    for path, manifest in rows:
        if manifest is None:
            typer.echo(
                f"{path.name}  CORRUPT — manifest missing or unreadable",
                err=True,
            )
            continue
        typer.echo(
            f"{path.name}  {manifest.command:<28}  "
            f"items={manifest.item_count}  topics={manifest.topic_count}  "
            f"vocab={manifest.vocab_size}"
        )


@snapshot_app.command("show")
@_handle_cli_errors
def snapshot_show_cmd(name: str = typer.Argument(..., help="Snapshot directory name")) -> None:
    """Print the manifest of one snapshot."""
    cfg = _config()
    _, manifest = snapshot.snapshot_show(cfg.data_dir, name)
    typer.echo(manifest.model_dump_json(indent=2))


@snapshot_app.command("restore")
@_handle_cli_errors
def snapshot_restore_cmd(name: str = typer.Argument(..., help="Snapshot directory name")) -> None:
    """Restore data/ from a snapshot.

    The vault is NOT touched — run `xbrain generate` next to refresh it.
    Every per-artifact action is echoed so 'a file vanished' never happens
    silently.
    """
    cfg = _config()
    actions = snapshot.snapshot_restore(cfg.data_dir, name)
    for artifact, action in actions:
        typer.echo(f"  {artifact}: {action}")
    typer.echo(f"Restored {name}. Run `xbrain generate` to refresh the vault.")


@snapshot_app.command("prune")
@_handle_cli_errors
def snapshot_prune_cmd(
    keep_last: int = typer.Option(10, "--keep-last", help="Keep the N newest snapshots"),
) -> None:
    """Delete older snapshots, keeping the N newest."""
    cfg = _config()
    deleted = snapshot.snapshot_prune(cfg.data_dir, keep_last=keep_last)
    typer.echo(f"Snapshots deleted: {deleted}")


if __name__ == "__main__":
    app()
