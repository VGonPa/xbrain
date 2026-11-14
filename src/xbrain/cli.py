"""Command-line interface for XBrain."""

from __future__ import annotations

import enum
import functools
import json
import logging
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import typer

from xbrain import snapshot
from xbrain.archive import parse_archive
from xbrain.config import Config, load_config
from xbrain.describe import apply_describe_worksheet, export_describe_worksheet
from xbrain.describe import describe_all as run_describe_all
from xbrain.describe import emit_summary_line as describe_emit_summary_line
from xbrain.diff import diff_snapshots, format_json, format_text
from xbrain.digest import VisualConfig, digest_videos, format_digest_summary
from xbrain.enrich import apply_worksheet_judgments, enrich_with_executor, items_pending_enrichment
from xbrain.executors.api import ApiExecutor
from xbrain.extract.browser import login as run_login
from xbrain.extract.browser import x_context
from xbrain.extract.extractor import (
    OperationNotCaptured,
    RateLimitTruncated,
    extract_source,
)
from xbrain.extract.threads import expand_threads
from xbrain.extract.graphql import items_needing_refetch
from xbrain.fetch import (
    FIRECRAWL_CREDENTIAL_PATHS,
    RetryPlan,
    fetch_pending,
    firecrawl_available,
    plan_retry_failed,
    retry_failed,
    revalidate_stored_bodies,
)
from xbrain.fetch_x import fetch_x_articles, refetch_full_texts_pooled
from xbrain.generate import generate as run_generate
from xbrain.media import download_all as run_media_download
from xbrain.media import emit_summary_line as media_emit_summary_line
from xbrain.payloads import payload_stats, reextract_from_payloads
from xbrain.refetch_pool import PAUSE_MAX_MS, PAUSE_MIN_MS, clamp_tabs
from xbrain.models import ArchiveImport, Author, Item, SourceName
from xbrain.redescribe import (
    RedescribeReport,
    format_redescribe_summary,
    redescribe_frames,
    stale_video_sources,
)
from xbrain.refresh import (
    backfill_quoted_from_store,
    backfill_quoted_sources,
    estimate_download_size,
    refresh_video_media,
)
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
from xbrain.transcribe import Transcript, transcribe_media
from xbrain.video_fetch import (
    FetchReport,
    fetch_result_to_json,
    fetch_videos,
    format_fetch_summary,
)
from xbrain.video_frames import (
    FOOTAGE_CAP_SETTING,
    KeyFrame,
    extract_key_frames,
    select_frames,
)
from xbrain.video_media import (
    VideoDownloadPlan,
    VideoReport,
    emit_video_summary_line,
    format_size_gate,
    parse_size_to_bytes,
    plan_video_downloads,
)
from xbrain.video_media import download_videos as run_download_videos
from xbrain.video_select import (
    _primary_topic,
    _scope_by_source,
    format_video_table,
    list_video_entries,
    row_to_json,
)
from xbrain.vision import describe_image
from xbrain.vocab import (
    apply_vocab_worksheet,
    export_vocab_worksheet,
    import_vocab_worksheet,
    induce_vocab,
)
from xbrain.verification import (
    aggregate_verify_judgments,
    apply_verdicts_to_store,
    count_invalidated_verdicts,
    cross_check_fingerprints,
    export_verify_worksheet,
    import_verify_fingerprints,
    import_verify_judgments,
    items_for_verification,
    parse_targets,
    record_fingerprints,
    render_verify_report,
    stamp_record_fingerprints,
)
from xbrain.entity_grounding import (
    load_ensemble_verdicts,
    outputs_present,
    render_entity_report,
    scan_store,
    summarise_scan,
)
from xbrain.verification_audit import (
    consequential_records,
    export_audit_worksheet,
    import_audit_judgments,
    load_report_records,
    merge_audit,
)
from xbrain.video_digest import (
    apply_video_digest_judgments,
    export_video_digest_worksheet,
    import_video_digest_worksheet,
    items_pending_video_digest,
)
from xbrain.worksheet import export_worksheet, import_worksheet

app = typer.Typer(help="XBrain — X bookmarks and tweets into an Obsidian wiki")

_BOOKMARKS_URL = "https://x.com/i/bookmarks"

_HEADLESS_HELP = (
    "Hidden browser. Headful (visible) by default — harder to fingerprint "
    "as a bot. Use --headless for unattended runs without a display."
)


@app.callback()
def _configure_logging() -> None:
    """Surface library `logging` warnings (e.g. the 429 backoff notice) cleanly.

    Without a configured handler these fall to Python's last-resort handler with
    an ugly `WARNING:logger:` prefix; route warnings through a plain stderr stream
    so the user sees the backoff message during a long pause.
    """
    logging.basicConfig(level=logging.WARNING, format="%(message)s")


class Source(str, enum.Enum):
    bookmarks = "bookmarks"
    tweets = "tweets"
    all = "all"


class VideoStatus(str, enum.Enum):
    """The `list-videos --status` filter values (mirrors the four `VideoState`s)."""

    downloaded = "downloaded"
    failed = "failed"
    pending = "pending"
    poster_era = "poster-era"


def _repo_root() -> Path:
    """Repo root — overridable via XBRAIN_REPO_ROOT for tests."""
    override = os.environ.get("XBRAIN_REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]


def _config() -> Config:
    return load_config(_repo_root())


def _parse_date(value: str | None, *, end_of_day: bool = False) -> datetime | None:
    """Parse an ISO date/datetime into a UTC-aware datetime.

    A *date-only* ``value`` (e.g. ``2025-12-31``) carries no time component,
    so it parses to that day's midnight. For a ``since`` bound that is the
    correct day start. For an ``until`` bound (``end_of_day=True``) midnight
    would exclude the whole final day, so we snap it to the last microsecond
    (``23:59:59.999999`` UTC) — the ``item.created_at > until`` filters then
    include every item created on that day. An explicit time
    (e.g. ``2025-12-31T09:00``) is respected as-is and never snapped.
    """
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if end_of_day and _is_date_only(value):
        parsed = parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


# A bare ISO date (``YYYY-MM-DD``) optionally carrying a tz offset (``+00:00``,
# ``-0500``, ``Z``) but NO time-of-day. A time-of-day is always introduced by a
# ``T``/space separator, so ``2025-12-31T09:00:00`` and ``2025-12-31 120000``
# never match — only whole-day bounds do.
_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[Zz]|[+-]\d{2}:?\d{2})?")


def _is_date_only(value: str) -> bool:
    """True when an ISO string is a bare date (no time-of-day), so an ``until``
    bound should cover the whole day. See ``_DATE_ONLY_RE``."""
    return _DATE_ONLY_RE.fullmatch(value) is not None


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
    # NOTE: MemoryError is deliberately NOT here — a global catch would swallow
    # OOM stacks for every command and print an empty "Error: ". `download-videos`
    # handles a too-large body LOCALLY in `_download_one_video` (records the cause
    # + continues the batch); see `xbrain.video_media`.
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
        typer.echo(f"Rejected by the validator: {len(invalid)}", err=True)
        for item_id, errors in invalid:
            typer.echo(f"  {item_id}: {'; '.join(errors)}", err=True)


def _run_extract(
    cfg: Config,
    source: str,
    since: datetime | None,
    until: datetime | None,
    *,
    headless: bool = False,
) -> None:
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
    truncated: list[str] = []
    with x_context(cfg.storage_state_path, headless=headless) as context:
        for src in chosen:
            cursor = state.bookmarks if src == "bookmark" else state.own_tweets
            first_run = cursor.last_seen_id is None
            try:
                items = extract_source(
                    context, src, targets[src], known_ids, since, until, cfg.payload_dir
                )
            except RateLimitTruncated as exc:
                # A truncated run is a partial, non-contiguous batch. Merging it
                # (and advancing the cursor) would seal a permanent gap in the
                # incremental store, so persist NOTHING for this source and fail
                # loud; the next run re-scrolls the window cleanly.
                typer.echo(f"ERROR: {exc} (nothing was saved for {src})", err=True)
                truncated.append(src)
                continue
            except OperationNotCaptured as exc:
                # Nothing was captured at all — the run learned NOTHING about this
                # source. Advancing `last_run` would make the next run look fresh
                # and hide the breakage, so leave the cursor untouched and fail
                # loud. Other sources still save: a rename hits one operation.
                typer.echo(f"ERROR: {exc} (nothing was saved for {src})", err=True)
                truncated.append(src)
                continue
            if not items and first_run:
                typer.echo(
                    f"WARNING: {src} returned 0 items on an initial extraction — "
                    "check the X session or the GraphQL parser (spec §6).",
                    err=True,
                )
            added = merge_items(store, items)
            if items:
                cursor.last_seen_id = max(items, key=lambda i: int(i.id)).id
            cursor.last_run = datetime.now(timezone.utc)
            typer.echo(f"{src}: {added} new items")
    save_store(store, cfg.items_path)
    save_state(state, cfg.state_path)
    if truncated:
        raise RuntimeError(
            f"Incomplete extraction for: {', '.join(truncated)} (rate limit, block, or "
            "X renamed the GraphQL operation — the message above says which). "
            "The completed sources were saved; resume later for the rest."
        )


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


def _format_size_estimate(estimated_bytes: int, n_estimable: int, n_unknown: int) -> str:
    """The human download-size line; never prints '~0.0 GB' when nothing is estimable.

    With at least one estimable video, reports the GB sum plus the unknown
    count. With none estimable, says the size is unknown for the N videos that
    carry no bitrate/duration (so a large unknown count never misreads as
    "~0.0 GB, nothing to download"), and reports "no videos" when there are none.
    """
    if n_estimable == 0:
        if n_unknown == 0:
            return "Estimated video download: no videos in the store."
        return (
            f"Estimated video download: size unknown for {n_unknown} videos "
            "(no bitrate/duration captured)."
        )
    gigabytes = estimated_bytes / 1_000_000_000
    return (
        f"Estimated video download: ~{gigabytes:.1f} GB across {n_estimable} videos; "
        f"{n_unknown} with unknown size."
    )


def _recapture_history(
    cfg: Config, source: str, *, label: str, headless: bool = False
) -> list[Item]:
    """Scroll the FULL X history and return every freshly-parsed item.

    The shared capture harness behind every backfill (`refresh-media`,
    `refresh-quoted`): one scroll, one parser, so a second backfill cannot drift
    into its own subtly different ingest path.

    `known_ids` is EMPTY, which disables `extract_source`'s skip-known early-stop —
    the whole timeline is walked, not just what is newer than the cursor. Unlike
    `_run_extract`, the `state.json` cursors are deliberately left untouched: a
    backfill revisits existing records, so the next `extract` cursor must not move.
    """
    # Mirrors `_run_extract` — the source → (target URL, GraphQL source) mapping.
    targets = {
        "bookmark": _BOOKMARKS_URL,
        "own_tweet": f"https://x.com/{cfg.x_handle}",
    }
    source_sets: dict[str, list[SourceName]] = {
        "bookmarks": ["bookmark"],
        "tweets": ["own_tweet"],
        "all": ["bookmark", "own_tweet"],
    }
    typer.echo(
        f"{label} scrolls the FULL X history with no skip-known — this is "
        "slow and human-paced and can take many minutes. Leave it running."
    )
    fresh: list[Item] = []
    with x_context(cfg.storage_state_path, headless=headless) as context:
        for src in source_sets[source]:
            # Persist the raw payloads on the way past. This scroll has NO skip-known, so it
            # re-sees the ENTIRE timeline — it is the cheapest payload backfill available for
            # the items ingested before payload persistence existed, and every backfill
            # (`refresh-media`, `refresh-quoted`) runs through here. A scroll that walked the
            # whole history and stored nothing would have to be paid for twice.
            fresh.extend(
                extract_source(context, src, targets[src], set(), payload_dir=cfg.payload_dir)
            )
    return fresh


def _guard_empty_recapture(
    store: dict[str, Item], items_seen: int, *, label: str, force: bool
) -> None:
    """Abort a backfill that re-saw 0 known items against a non-empty store.

    `extract_source` returns `[]` (it does NOT raise) when the session is logged in
    but the GraphQL parser drifts or the scroll is interrupted. Re-seeing nothing is
    therefore a likely-broken run, not a successful no-op — and since the merge was a
    no-op the store on disk is untouched, so aborting without saving is byte-identical
    (and the pre-snapshot already fired). `--force` downgrades it to a warning.
    """
    if not (store and items_seen == 0):
        return
    warning = (
        f"{label} re-saw 0 of the {len(store)} already-known items — "
        "the X session probably expired or the GraphQL parser has drifted "
        "(spec §6); nothing was updated."
    )
    if not force:
        raise RuntimeError(f"{warning} Use --force to save anyway.")
    typer.echo(f"WARNING: {warning}", err=True)


def _run_refresh_quoted_from_store(cfg: Config) -> None:
    """Backfill the quoted post from items ALREADY in the store — no browser, no network.

    A quote-tweet's `quoted_id` often names a post we captured in its own right, so the
    evidence is one dict lookup away. Free, instant, and re-runnable. What it cannot
    reach (a quoted post we never captured) is left for `refresh-quoted`.

    Destructive (rewrites `items.json`) → auto-snapshots first.
    """
    _auto_snapshot(cfg, "refresh-quoted-from-store")
    store = load_store(cfg.items_path)
    report = backfill_quoted_from_store(store)
    save_store(store, cfg.items_path)
    typer.echo(
        f"refresh-quoted --from-store: {report.sources_attached} quoted posts attached "
        f"from items already in the store; {report.already_present} already had one; "
        f"{report.quoted_items_not_seen} quote-tweets quote a post we do NOT hold "
        "(run `xbrain refresh-quoted` to capture those)."
    )
    if report.readable:
        typer.echo(
            f"Next: `xbrain enrich` regenerates the {report.readable} summaries with the "
            "new evidence (only those advance `content.fetched_at`; an unreadable quoted "
            "post is recorded but does not re-enrich)."
        )


def _run_refresh_quoted(cfg: Config, source: str, *, force: bool, headless: bool = False) -> None:
    """Re-capture X and backfill the QUOTED POST onto already-stored quote-tweets.

    No per-item fetch: X embeds the quoted post — body AND author — in the same
    timeline payload as the tweet quoting it, so one re-capture carries everything.
    Items that gain a quoted post get a bumped `content.fetched_at`, so the next
    `xbrain enrich` re-generates exactly those summaries against the evidence they
    were previously written without.

    Destructive (rewrites `items.json` in place) → auto-snapshots first.
    """
    _auto_snapshot(cfg, "refresh-quoted")
    store = load_store(cfg.items_path)
    fresh = _recapture_history(cfg, source, label="refresh-quoted", headless=headless)
    report = backfill_quoted_sources(store, fresh)

    _guard_empty_recapture(store, report.items_seen, label="refresh-quoted", force=force)
    save_store(store, cfg.items_path)
    typer.echo(
        f"refresh-quoted: {report.items_seen} known items re-seen; "
        f"{report.sources_attached} quoted posts attached "
        f"({report.readable} readable, {report.unreadable} unavailable); "
        f"{report.already_present} already had one; "
        f"{report.quoted_items_not_seen} quote-tweets NOT re-seen (still evidence-less)."
    )
    if report.readable:
        typer.echo(
            f"Next: `xbrain enrich` regenerates the {report.readable} summaries with the "
            "new evidence (only those advance `content.fetched_at`; an unreadable quoted "
            "post is recorded but does not re-enrich)."
        )


def _run_refresh_media(cfg: Config, source: str, *, force: bool, headless: bool = False) -> None:
    """Re-capture the FULL X history and backfill playable video media in place.

    Destructive — it overwrites the video entries on existing items — so it
    auto-snapshots `data/` first (label `pre-refresh-media`); a snapshot failure
    propagates and aborts before any capture or write (CONTRIBUTING §Safety).

    Then it scrolls each chosen source with an EMPTY `known_ids` set, so
    `extract_source` does NOT stop at the first known id and the whole timeline
    is walked. The freshly-parsed items are merged onto the store by
    `refresh_video_media` — video entries only; photos and every enrichment /
    description / fetch field are preserved. The store is saved and a
    download-size estimate is printed. Video DOWNLOAD is out of scope here.

    Empty-capture guard: `extract_source` returns `[]` (it does NOT raise) when
    the session is logged in but the GraphQL parser drifts or the scroll is
    interrupted. Re-seeing 0 known items against a NON-EMPTY store is therefore
    a likely-broken run, not success — it surfaces a loud warning and aborts
    non-zero WITHOUT saving (the merge was a no-op, so the store on disk is
    untouched and the pre-snapshot already fired). `--force` downgrades this to
    a warning and proceeds. An empty store (fresh project) and any non-zero
    capture (monotonic, re-runnable progress) are left to save normally.
    """
    _auto_snapshot(cfg, "refresh-media")
    store = load_store(cfg.items_path)
    fresh = _recapture_history(cfg, source, label="refresh-media", headless=headless)
    report = refresh_video_media(store, fresh)

    _guard_empty_recapture(store, report.items_seen, label="refresh-media", force=force)
    save_store(store, cfg.items_path)
    estimated_bytes, n_estimable, n_unknown = estimate_download_size(store)
    typer.echo(
        f"refresh-media: {report.items_seen} known items re-seen, "
        f"{report.items_refreshed} refreshed, {report.videos_updated} videos updated; "
        f"{report.items_with_video_not_seen} video items not re-seen (still poster-era)."
    )
    typer.echo(_format_size_estimate(estimated_bytes, n_estimable, n_unknown))


def _run_fetch(
    cfg: Config,
    since: datetime | None,
    until: datetime | None,
    force: bool,
    *,
    headless: bool = False,
) -> None:
    if force:
        _auto_snapshot(cfg, "fetch-force")
    store = load_store(cfg.items_path)
    try:
        articles = fetch_pending(store, since, until, force)
        x_articles = fetch_x_articles(
            store, cfg.storage_state_path, force, since, until, headless=headless
        )
        threads = expand_threads(store, cfg.storage_state_path, force, headless=headless)
    finally:
        # Persist whatever was fetched even if a later stage raised — a stage
        # error (e.g. an expired X session) must not discard in-memory work.
        save_store(store, cfg.items_path)
    typer.echo(f"Content fetched: {articles} articles, {x_articles} from X, {threads} threads")


def _run_generate(cfg: Config, since: datetime | None, until: datetime | None) -> None:
    store = load_store(cfg.items_path)
    topic_pages = load_topic_pages(cfg.topics_path) if cfg.topics_path.exists() else {}
    run_generate(
        store,
        cfg.output_dir,
        since,
        until,
        cfg.output_language,
        cfg.topic_style,
        media_root=cfg.media_dir,
        topic_pages=topic_pages,
    )
    typer.echo(f"Markdown generated in {cfg.output_dir}")


@app.command()
@_handle_cli_errors
def login() -> None:
    """Open a browser to log in to X and save the session."""
    run_login(_config().storage_state_path)


@app.command()
@_handle_cli_errors
def extract(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Extract bookmarks and/or your own tweets from X."""
    _run_extract(
        _config(),
        source.value,
        _parse_date(since),
        _parse_date(until, end_of_day=True),
        headless=headless,
    )


@app.command(name="import-archive")
@_handle_cli_errors
def import_archive(zip_path: Path) -> None:
    """Backfill the tweet history from the official X archive."""
    cfg = _config()
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    author = Author(handle=cfg.x_handle, name=cfg.x_handle)
    added = merge_items(store, parse_archive(zip_path, author))
    state.archive_imported = ArchiveImport(file=zip_path.name, at=datetime.now(timezone.utc))
    save_store(store, cfg.items_path)
    save_state(state, cfg.state_path)
    typer.echo(f"Archive imported: {added} new tweets")


def _echo_retry_plan(plan: RetryPlan, *, has_key: bool) -> None:
    """Report the plan — including, loudly, what it will NOT attempt and why."""
    reasons = ", ".join(f"{n} {reason}" for reason, n in sorted(plan.reasons.items()))
    typer.echo(f"Retryable: {len(plan.retryable)} items" + (f" ({reasons})" if reasons else ""))
    if plan.blocked_on_firecrawl:
        looked_in = "\n".join(f"    - {p}" for p in FIRECRAWL_CREDENTIAL_PATHS)
        typer.echo(
            f"BLOCKED for lack of a Firecrawl key: {len(plan.blocked_on_firecrawl)} items "
            "with js_required/empty_content failures that NEVER went through the fallback "
            "(attempts=1). Without the key, retrying them repeats the same failure.\n"
            "  Looked in $FIRECRAWL_API_KEY and in the CLI credentials:\n"
            f"{looked_in}\n"
            "  Configure it (or run `firecrawl login`) and run again."
        )
    elif has_key:
        typer.echo("Firecrawl key resolved — the JS fallback joins the retries.")
    typer.echo(
        f"Terminal (no extractor can fix them): {len(plan.terminal)} items. "
        "Their guardrail note already names the cause."
    )


def _run_retry_failed(cfg: Config, *, dry_run: bool) -> None:
    """`fetch --retry-failed`: re-fetch ONLY the link failures a retry could actually repair.

    Distinct from `--force`, which re-hits every link in the store (400 items in the real
    corpus) and re-downloads the ones that already succeeded. This targets the recorded
    failures, so it is the safe way to pick up the Firecrawl fallback on the
    `js_required`/`empty_content` bucket that `_should_refetch` calls terminal and therefore
    never retries.
    """
    has_key = firecrawl_available()
    store = load_store(cfg.items_path)
    plan = plan_retry_failed(store, firecrawl_configured=has_key)
    _echo_retry_plan(plan, has_key=has_key)
    if dry_run:
        typer.echo("--dry-run: the store was not touched.")
        return
    if not plan.retryable:
        return
    _auto_snapshot(cfg, "fetch-retry-failed")
    try:
        refetched = retry_failed(store, plan, firecrawl_configured=has_key)
    finally:
        save_store(store, cfg.items_path)
    # What actually LANDED. A retry that "succeeded" into a cookie wall is now recorded as a
    # `blocked_interstitial` failure rather than evidence, so this tally is the operator's
    # direct read on whether the run repaired anything or merely re-confirmed the walls.
    after = plan_retry_failed(store, firecrawl_configured=has_key)
    repaired = [i for i in plan.retryable if i not in set(after.retryable) | set(after.terminal)]
    typer.echo(
        f"Retried: {refetched} items → {cfg.items_path}\n"
        f"  repaired (now have an article): {len(repaired)}\n"
        f"  still without content: {refetched - len(repaired)} "
        "(their guardrail note stays active and names the cause)"
    )


def _run_revalidate(cfg: Config, *, write: bool) -> None:
    """`fetch --revalidate`: re-judge the article bodies ALREADY in the store and demote the junk.

    REPORT-ONLY unless `--write`. This one rewrites recorded evidence, so the safe default is to
    show the operator exactly which items and which domains would change, and let them decide.

    Local — no network, no extractor. A junk body that was accepted as a success is invisible to
    `--retry-failed` (which selects FAILURES), so without this the 28 measured junk bodies in the
    real corpus keep serving as `[Linked article]` evidence forever. Demoting cannot lose
    anything: the body was never evidence in the first place.
    """
    store = load_store(cfg.items_path)
    found = revalidate_stored_bodies(store)
    domains = Counter(urlparse(u).hostname or u for u in found.urls)
    typer.echo(
        f"Bodies that are NOT articles: {len(found.items)} items. Their 'article' is really a "
        "wall (cookies/login), an anti-bot challenge or pure page chrome — today they are served "
        "to the judge as `[Linked article]` and the guardrail does NOT fire for them."
    )
    for domain, count in domains.most_common(12):
        typer.echo(f"  {count:>3}  {domain}")
    if not write:
        typer.echo(
            "\nReport only — the store was NOT touched. Re-run with --write to demote them "
            "to a `blocked_interstitial` failure (a snapshot is taken first)."
        )
        return
    if not found.items:
        return
    _auto_snapshot(cfg, "fetch-revalidate")
    save_store(store, cfg.items_path)
    typer.echo(f"\nDemoted to `blocked_interstitial` → {cfg.items_path}")


@app.command()
@_handle_cli_errors
def fetch(
    since: str = typer.Option(None),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
    force: bool = typer.Option(False, help="Re-download what was already downloaded"),
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Retry ONLY the links whose recorded failure a retry can repair "
        "(transient ones, plus js_required/empty_content when FIRECRAWL_API_KEY is set). "
        "Unlike --force, it does not re-download what already worked.",
    ),
    revalidate: bool = typer.Option(
        False,
        "--revalidate",
        help="Re-judge the bodies ALREADY stored and demote the ones that are cookie/login walls "
        "or page chrome (28 measured on the real corpus). Local, no network.",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        help="With --revalidate: apply the demotions (by default it only reports).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="With --retry-failed: report the plan without touching the store."
    ),
) -> None:
    """Download the content of linked articles."""
    if dry_run and not retry_failed:
        raise typer.BadParameter("--dry-run requires --retry-failed.")
    if write and not revalidate:
        raise typer.BadParameter("--write requires --revalidate.")
    if revalidate:
        if retry_failed or force:
            raise typer.BadParameter(
                "--revalidate re-judges the bodies already in the store (no network); it does "
                "not combine with --retry-failed or --force."
            )
        _run_revalidate(_config(), write=write)
        return
    if retry_failed:
        if force:
            raise typer.BadParameter(
                "--retry-failed and --force are mutually exclusive: --force re-fetches EVERY "
                "link (including the ones that already succeeded), --retry-failed targets only "
                "the recorded failures a retry could repair."
            )
        _run_retry_failed(_config(), dry_run=dry_run)
        return
    _run_fetch(
        _config(), _parse_date(since), _parse_date(until, end_of_day=True), force, headless=headless
    )


def _run_media(
    cfg: Config,
    *,
    force: bool,
    limit: int | None,
    items_filter: list[str] | None,
    verbose: bool = False,
) -> None:
    """Run the photo downloader: snapshot, load, download, persist, summarise.

    Always snapshots `data/` first (the same recovery boundary as
    `vocab --regenerate` etc): a botched run can be undone with
    `xbrain snapshot restore`.

    Persistence happens twice: once after every photo transition (the
    `on_progress` callback writes the store atomically, so Ctrl-C mid-run
    leaves `items.json` coherent), and once unconditionally at the end so
    the elapsed timestamp on the last `MediaPhotoDownloaded` is captured
    even if no transition fired (e.g. a `--limit 0` no-op).

    Persistence failure semantics: if `save_store` raises inside the
    `on_progress` callback (e.g. disk full), the exception propagates and
    aborts the run. The state of `items.json` for the photo currently
    being processed is whatever the previous successful write captured;
    later items remain in their pre-run variant. The `finally` block
    below still attempts a final write, but on a disk-full condition that
    too may fail — in which case the in-memory transitions for the
    interrupted batch are lost. This is acceptable: a re-run after the
    operator clears the disk picks up every still-pending photo cleanly.
    """
    if items_filter:
        target = set(items_filter)
        store_ids = set(load_store(cfg.items_path))
        missing = target - store_ids
        if missing and not (target & store_ids):
            typer.echo(
                f"WARNING: --items {','.join(items_filter)} matches no item "
                f"in the store ({len(store_ids)} items). The run will be a no-op.",
                err=True,
            )
    _auto_snapshot(cfg, "media")
    store = load_store(cfg.items_path)

    def _persist() -> None:
        save_store(store, cfg.items_path)

    try:
        report = run_media_download(
            store,
            cfg.media_dir,
            force=force,
            limit=limit,
            items_filter=items_filter,
            on_progress=_persist,
        )
    finally:
        # Persist whatever changed, even if `download_all` raised. A
        # RuntimeError on total failure must not discard the per-photo
        # MediaPhotoFailed records that landed before the raise.
        save_store(store, cfg.items_path)
    media_emit_summary_line(report)
    article_failed = report.article_images_failed_permanent + report.article_images_failed_transient
    typer.echo(
        f"Media: downloaded {report.photos_downloaded}, "
        f"failed {report.photos_failed_permanent + report.photos_failed_transient}, "
        f"skipped {report.photos_skipped_already_downloaded} "
        f"(article images: downloaded {report.article_images_downloaded}, "
        f"failed {article_failed}, skipped {report.article_images_skipped})"
    )
    if verbose and report.per_item_failures:
        typer.echo("Failed media:", err=True)
        for item_id, failures in sorted(report.per_item_failures.items()):
            for url, reason in failures:
                typer.echo(f"  {item_id}  {reason:<14}  {url}", err=True)


@app.command()
@_handle_cli_errors
def media(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download every photo, even those already downloaded or permanently "
        "failed (HTTP 4xx, format_error).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Maximum number of downloads to attempt in this run.",
    ),
    items: str | None = typer.Option(
        None,
        "--items",
        help="Comma-separated item IDs to limit the scope of the run.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Print every failed photo (item_id, reason, URL) at the end of the run.",
    ),
) -> None:
    """Download the photos of the X posts referenced in `items.json`.

    Only photos are downloaded (`MediaPhotoPending` + transient retries).
    Videos stay in their `MediaVideoPending` variant for a later stage
    — `--force` does NOT download them.
    """
    cfg = _config()
    items_filter = [s.strip() for s in items.split(",") if s.strip()] if items else None
    _run_media(cfg, force=force, limit=limit, items_filter=items_filter, verbose=verbose)


@app.command(name="refresh-media")
@_handle_cli_errors
def refresh_media(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Save even if 0 known items are re-seen (expired session / "
        "GraphQL drift). By default that case aborts without writing.",
    ),
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Re-capture X and refresh the video URL/metadata of already-stored items.

    Walks the FULL history (without skipping known ids) and rewrites the
    poster-era video entries with the playable stream + bitrate + duration.
    It does not touch photos or the enrichment/description state, and it
    never demotes a good video to its poster if X stops serving the stream.

    Destructive (rewrites `items.json` in place) → auto-snapshot before
    writing. If 0 known items are re-seen against a non-empty store (likely
    an expired session or parser drift), it aborts without saving unless
    `--force`. It does NOT download video (that is a later stage): it only
    prints an estimate of the total download size. The scroll is slow and
    human-paced; it can take several minutes.
    """
    _run_refresh_media(_config(), source.value, force=force, headless=headless)


@app.command(name="refresh-quoted")
@_handle_cli_errors
def refresh_quoted(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    from_store: bool = typer.Option(
        False,
        "--from-store",
        help="No network: attach the quoted post ONLY when it is already in the store as "
        "an item of its own (measured: 199 of 762). Instant and re-runnable.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Save even if 0 known items are re-seen (expired session / "
        "GraphQL drift). By default that case aborts without writing.",
    ),
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """Re-capture X and attach the QUOTED POST to already-stored quote-tweets.

    With `--from-store` it does NOT open the browser: it joins `quoted_id → item`
    against the store itself, which already holds the quoted post in 199 of the
    762 cases (26.1%). Always start there — it is free — then re-capture for the rest.

    Existing items only store `quoted_id`: the quoted post's body and its author
    were lost, so the generator saw a bare reaction ("Read this and you'll
    understand") with nothing to summarise — and filled the gap by inventing.
    X embeds the quoted post in the SAME timeline payload, so this makes NOT ONE
    extra request per item: it walks the full history (without skipping known
    ids) and re-parses.

    It only touches `quoted_tweet` sources: articles, transcripts, threads and all
    enrichment are preserved. Idempotent (an already-readable quoted post is left
    intact; a failed one is retried). Items that gain evidence advance their
    `content.fetched_at`, so the next `xbrain enrich` regenerates exactly those
    summaries.

    Destructive (rewrites `items.json` in place) → auto-snapshot before writing.
    The scroll is slow and human-paced; it can take several minutes.
    """
    if from_store:
        _run_refresh_quoted_from_store(_config())
        return
    _run_refresh_quoted(_config(), source.value, force=force, headless=headless)


def _run_describe(
    cfg: Config,
    *,
    force: bool,
    limit: int | None,
    items_filter: list[str] | None,
    model: str,
    batch_size: int,
    verbose: bool,
) -> None:
    """Run the vision-describe orchestrator and persist after every batch.

    Always snapshots `data/` first (the same recovery boundary as
    `xbrain media`): a botched run — a wrong model, a runaway prompt
    — can be undone with `xbrain snapshot restore`. Coherence on a
    Ctrl-C mid-run is held by the outer `try/finally` below, which
    saves the store unconditionally even when the orchestrator raises;
    the `on_progress` callback is for incremental persistence between
    batches on a clean run (so a long describe run never loses more
    than one batch of work to a process death).
    """
    if items_filter:
        target = set(items_filter)
        store_ids = set(load_store(cfg.items_path))
        missing = target - store_ids
        if missing and not (target & store_ids):
            typer.echo(
                f"WARNING: --items {','.join(items_filter)} matches no item "
                f"in the store ({len(store_ids)} items). The run will be a no-op.",
                err=True,
            )
    _auto_snapshot(cfg, "describe")
    store = load_store(cfg.items_path)

    def _persist() -> None:
        save_store(store, cfg.items_path)

    try:
        report = run_describe_all(
            store,
            cfg.media_dir,
            model=model,
            output_language=cfg.output_language,
            description_version=cfg.describe_version,
            force=force,
            limit=limit,
            items_filter=items_filter,
            batch_size=batch_size,
            on_progress=_persist,
        )
    finally:
        # Persist whatever transitioned, even if `describe_all` raised. A
        # RuntimeError on total failure must not discard the per-photo
        # MediaPhotoDescribed records that landed before the raise.
        save_store(store, cfg.items_path)
    describe_emit_summary_line(report)
    typer.echo(
        f"Describe: described {report.photos_described}, "
        f"failed {report.photos_failed}, "
        f"skipped {report.photos_skipped_already_described}"
    )
    if verbose and report.per_item_failures:
        typer.echo("Failed photos:", err=True)
        for item_id, failures in sorted(report.per_item_failures.items()):
            for url, error in failures:
                typer.echo(f"  {item_id}  {url}  {error}", err=True)


@app.command()
@_handle_cli_errors
def describe(
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-describe every photo, even those already described at the current version.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Maximum number of photos to describe in this run.",
    ),
    items: str | None = typer.Option(
        None,
        "--items",
        help="Comma-separated item IDs to limit the scope of the run.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Vision model to use. If omitted, the config's (`describe.model`) is used.",
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        min=1,
        help="Images per API call. 5 is the sweet spot (12-15%% token savings).",
    ),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="api | manual | claude-code (default: api). manual/claude-code export a "
        "worksheet to describe without an API key (like enrich/topics).",
    ),
    apply: Path | None = typer.Option(
        None,
        "--apply",
        help="Import a filled description worksheet and apply it (no API key).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Print every failed photo (item_id, URL, error) at the end of the run.",
    ),
) -> None:
    """Describe the downloaded photos with a vision LLM.

    Only photos with bytes on disk (`MediaPhotoDownloaded`) are described.
    Entries already described at the current version are skipped; bumping
    `[describe].version` in `config.toml` forces an automatic re-describe
    without `--force`. Descriptions are persisted in `items.json` and
    consumed by `xbrain enrich` and `xbrain topics` in later LLM calls.
    """
    cfg = _config()
    items_filter = [s.strip() for s in items.split(",") if s.strip()] if items else None
    worksheet_path = cfg.data_dir / "describe-worksheet.json"
    if apply is not None:
        _auto_snapshot(cfg, "describe-apply")
        store = load_store(cfg.items_path)
        applied, invalid = apply_describe_worksheet(store, apply)
        save_store(store, cfg.items_path)
        typer.echo(f"Describe worksheet applied: {applied} photos described")
        _report_invalid(invalid)
        return
    if executor is not None and executor not in ("api", "manual", "claude-code"):
        raise ValueError(f"Unknown executor: {executor!r}")
    if executor in ("manual", "claude-code"):
        store = load_store(cfg.items_path)
        n = export_describe_worksheet(
            store,
            cfg.media_dir,
            worksheet_path,
            version=cfg.describe_version,
            output_language=cfg.output_language,
            force=force,
            limit=limit,
            items_filter=items_filter,
        )
        typer.echo(
            f"{n} photos exported to {worksheet_path}\n"
            "Fill in the `judgments` array (with Claude Code or by hand) and run:\n"
            f"  xbrain describe --apply {worksheet_path}"
        )
        return
    chosen_model = model or cfg.describe_model
    _run_describe(
        cfg,
        force=force,
        limit=limit,
        items_filter=items_filter,
        model=chosen_model,
        batch_size=batch_size,
        verbose=verbose,
    )


def _warn_items_filter_no_match(cfg: Config, items_filter: list[str]) -> None:
    """Echo a no-op warning when `--items` matches nothing (shared by media/video)."""
    target = set(items_filter)
    store_ids = set(load_store(cfg.items_path))
    if (target - store_ids) and not (target & store_ids):
        typer.echo(
            f"WARNING: --items {','.join(items_filter)} matches no item "
            f"in the store ({len(store_ids)} items). The run will be a no-op.",
            err=True,
        )


def _skip_only_report(plan: VideoDownloadPlan) -> VideoReport:
    """A `VideoReport` carrying only `plan`'s skip counts (no attempts).

    Lets the skip-only path emit the same `SUMMARY:` line as a real run, so a
    monitor grepping stderr sees `download-videos` and `media` consistently.
    """
    return VideoReport(
        videos_skipped_hls=plan.n_hls_skipped,
        videos_skipped_poster_era=plan.n_poster_skipped,
        videos_skipped_already_downloaded=plan.n_already_downloaded,
        videos_skipped_too_large=plan.n_too_large,
        videos_skipped_size_unknown=plan.n_size_unknown_skipped,
    )


def _run_download_videos(
    cfg: Config,
    source: str,
    *,
    force: bool,
    limit: int | None,
    items_filter: list[str] | None,
    yes: bool,
    max_size_bytes: int | None,
) -> None:
    """Download the mp4 bytes for `MediaVideoPending` entries; persist + summarise.

    Flow: load → plan (no network, no write) → print the size gate → confirm
    (unless `--yes`) → snapshot `data/` → download → persist. The snapshot is the
    same recovery boundary as `xbrain media`, but taken AFTER the confirm so a
    declined gate never leaves a stray snapshot; a snapshot failure still
    propagates and aborts before any write (CONTRIBUTING §Safety). A run with no
    downloadable mp4 (only HLS / poster-era / already-downloaded / over-cap /
    unknown-size) writes nothing, so it skips both the confirm and the snapshot —
    but still emits the `SUMMARY:` line for monitor parity with `media`.

    `--source` scopes the run to bookmark / own-tweet items; `scoped` shares the
    same `Item` objects as `store`, so the in-place transitions are persisted by
    saving the full `store`. mp4 ONLY: HLS entries are reported as deferred to
    the ffmpeg follow-up, never downloaded here. `max_size_bytes` caps the
    per-video estimated size.
    """
    if items_filter:
        _warn_items_filter_no_match(cfg, items_filter)
    store = load_store(cfg.items_path)
    source_sets: dict[str, list[SourceName]] = {
        "bookmarks": ["bookmark"],
        "tweets": ["own_tweet"],
        "all": ["bookmark", "own_tweet"],
    }
    chosen = set(source_sets[source])
    scoped = {item_id: item for item_id, item in store.items() if item.source in chosen}

    plan = plan_video_downloads(
        scoped, force=force, limit=limit, items_filter=items_filter, max_size_bytes=max_size_bytes
    )
    if plan.n_to_download == 0:
        typer.echo(
            f"No mp4 videos to download "
            f"({plan.n_hls_skipped} HLS awaiting ffmpeg, "
            f"{plan.n_poster_skipped} poster-era, "
            f"{plan.n_already_downloaded} already downloaded, "
            f"{plan.n_too_large} > --max-size, "
            f"{plan.n_size_unknown_skipped} without a size)."
        )
        emit_video_summary_line(_skip_only_report(plan))
        return
    typer.echo(format_size_gate(plan))
    if not yes:
        typer.confirm("Proceed with the download?", abort=True)
    _auto_snapshot(cfg, "download-videos")

    def _persist() -> None:
        save_store(store, cfg.items_path)

    try:
        report = run_download_videos(
            scoped,
            cfg.media_dir,
            force=force,
            limit=limit,
            items_filter=items_filter,
            max_size_bytes=max_size_bytes,
            on_progress=_persist,
        )
    finally:
        # Persist whatever transitioned even if `download_videos` raised — a
        # total-failure RuntimeError must not discard the MediaVideoFailed
        # records that landed before the raise.
        save_store(store, cfg.items_path)
    emit_video_summary_line(report)
    typer.echo(
        f"Videos: downloaded {report.videos_downloaded}, "
        f"failed {report.videos_failed_permanent + report.videos_failed_transient}, "
        f"HLS skipped {report.videos_skipped_hls}, "
        f"poster-era skipped {report.videos_skipped_poster_era}, "
        f"already downloaded {report.videos_skipped_already_downloaded}, "
        f"> --max-size {report.videos_skipped_too_large}, "
        f"without a size {report.videos_skipped_size_unknown}"
    )


@app.command(name="download-videos")
@_handle_cli_errors
def download_videos(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Maximum number of videos to download in this run.",
    ),
    items: str | None = typer.Option(
        None,
        "--items",
        help="Comma-separated item IDs to limit the scope of the run.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-download already-downloaded videos and retry permanent failures.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Do not ask to confirm the download size (non-interactive mode).",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Skip videos whose estimated size exceeds this cap. Accepts 500MB / 2GB "
        "(decimal units); a number without a unit is read as MB. With the cap set, "
        "videos of unknown size (no bitrate/duration) are skipped too.",
    ),
) -> None:
    """Download the mp4 bytes of the videos referenced in `items.json`.

    Only playable mp4 streams are downloaded (`MediaVideoPending` entries with a
    real URL, plus transient retries). Before downloading it prints an estimate
    of the total size (~X.X GB) and asks for confirmation unless `--yes`. HLS
    manifests (`.m3u8`) need ffmpeg and are deferred to a follow-up: they are
    counted and skipped, not downloaded here. Poster-era entries (not backfilled:
    run `xbrain refresh-media` first) are skipped too. `--max-size`
    (e.g. `500MB` / `2GB`) skips videos estimated to be too large.
    Destructive (rewrites `items.json`) → auto-snapshot before writing.
    """
    cfg = _config()
    items_filter = [s.strip() for s in items.split(",") if s.strip()] if items else None
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    _run_download_videos(
        cfg,
        source.value,
        force=force,
        limit=limit,
        items_filter=items_filter,
        yes=yes,
        max_size_bytes=max_size_bytes,
    )


@app.command(name="list-videos")
@_handle_cli_errors
def list_videos(
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    topic: str | None = typer.Option(None, "--topic", help="Filter by the item's primary_topic."),
    status: VideoStatus | None = typer.Option(
        None,
        "--status",
        help="Filter by state: downloaded | failed | pending | poster-era.",
    ),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Only videos with a known size <= cap (500MB / 2GB; no unit = MB).",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Maximum number of rows."),
    json_out: bool = typer.Option(
        False, "--json", help="Output a stable JSON array instead of a human table."
    ),
) -> None:
    """Catalogue (read-only) the videos referenced in `items.json`.

    One row per video entry, with its state (downloaded / failed / pending /
    poster-era), estimated size (exact if already downloaded, "unknown" when
    there is no bitrate/duration), the item's `primary_topic` and a text snippet.
    It writes NOTHING and takes no snapshot. With `--json` it emits a stable
    array with the fields `id, url, state, topic, size_bytes|null, mp4_url, text`
    that an agent can parse to choose which videos to pass to `fetch-video`.
    """
    cfg = _config()
    store = load_store(cfg.items_path)
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    rows = list_video_entries(
        store,
        topic=topic,
        status=status.value if status is not None else None,
        max_size_bytes=max_size_bytes,
        source=source.value,
        limit=limit,
    )
    if json_out:
        typer.echo(json.dumps([row_to_json(row) for row in rows], ensure_ascii=False, indent=2))
    else:
        typer.echo(format_video_table(rows))


def _resolve_fetch_ids(
    store: dict[str, Item], ids: str | None, topic: str | None, source: str
) -> list[str]:
    """Resolve `--ids` and/or `--topic` into a de-duplicated, ordered id list.

    Explicit `--ids` are taken verbatim; `--topic` is expanded via the read-only
    catalog (scoped by `--source`). At least one selector is required — an empty
    selection is an operator error, not a silent no-op.
    """
    id_list: list[str] = []
    if ids:
        id_list.extend(part.strip() for part in ids.split(",") if part.strip())
    if topic:
        id_list.extend(row.id for row in list_video_entries(store, topic=topic, source=source))
    if not id_list:
        raise ValueError("fetch-video: pass --ids and/or --topic to select videos.")
    return list(dict.fromkeys(id_list))


def _emit_fetch_report(report: FetchReport, *, json_out: bool) -> None:
    """Print the fetch outcomes: JSON array, or human lines + a SUMMARY."""
    if json_out:
        typer.echo(
            json.dumps(
                [fetch_result_to_json(result) for result in report.results],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    for result in report.results:
        if result.outcome == "fetched":
            typer.echo(f"{result.id}: {result.path}")
        elif result.outcome == "skipped":
            typer.echo(f"{result.id}: skipped ({result.reason})")
        else:
            typer.echo(
                f"{result.id}: failed ({result.reason}) {result.error or ''}".rstrip(), err=True
            )
    typer.echo(format_fetch_summary(report))


@app.command(name="fetch-video")
@_handle_cli_errors
def fetch_video(
    to: Path = typer.Option(
        ..., "--to", help="Destination directory (REQUIRED). Writes <dir>/<id>.mp4."
    ),
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated item IDs."),
    topic: str | None = typer.Option(
        None, "--topic", help="Select videos by the item's primary_topic."
    ),
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    max_size: str | None = typer.Option(
        None,
        "--max-size",
        help="Skip videos whose estimated size exceeds the cap (500MB / 2GB; no unit = MB).",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of downloads in this run."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Output a stable JSON array instead of human lines."
    ),
) -> None:
    """Download (ephemerally) the real mp4 of the chosen videos to `--to`/<id>.mp4.

    Select with `--ids` and/or `--topic` (+ `--max-size`, `--limit`). Reuses the
    `download-videos` primitives (content validation, failure classification,
    atomic write, mp4/HLS/poster discriminator). HLS and poster-era entries are
    skipped and counted. DELIBERATELY non-persisting: it does NOT mutate
    `items.json`, takes NO snapshot and does NOT write to `data/media/` — it only
    writes under `--to`. Meant for an agent to transcribe/analyse the video and
    then discard the bytes.
    """
    cfg = _config()
    store = load_store(cfg.items_path)
    id_list = _resolve_fetch_ids(store, ids, topic, source.value)
    max_size_bytes = parse_size_to_bytes(max_size) if max_size else None
    report = fetch_videos(store, id_list, to, max_size_bytes=max_size_bytes, limit=limit)
    _emit_fetch_report(report, json_out=json_out)
    if report.fetched == 0 and report.failed > 0:
        # Parity with download-videos: a run where every attempted download
        # failed must surface as a non-zero exit, not a silent empty run. A pure
        # all-skips run (nothing attempted) stays exit 0.
        raise RuntimeError(
            f"fetch-video: all {report.failed} download attempt(s) failed; "
            "check network / video.twimg.com availability and the warnings above."
        )


def _resolve_digest_ids(
    store: dict[str, Item],
    ids: str | None,
    topic: str | None,
    all_pending: bool,
    source: str,
    limit: int | None,
) -> list[str]:
    """Resolve the digest selection into a de-duplicated, ordered id list.

    `--all-pending` expands to every fetchable (`pending`) video via the
    read-only catalog; `--ids` are taken verbatim; `--topic` is expanded via the
    catalog (scoped by `--source`). At least one selector is required — an empty
    selection is an operator error, not a silent no-op. `--limit` caps the number
    of items after de-duplication.
    """
    id_list: list[str] = []
    if all_pending:
        id_list.extend(row.id for row in list_video_entries(store, status="pending", source=source))
    if ids:
        id_list.extend(part.strip() for part in ids.split(",") if part.strip())
    if topic:
        id_list.extend(row.id for row in list_video_entries(store, topic=topic, source=source))
    if not id_list:
        raise ValueError("digest-video: pass --ids, --topic or --all-pending to select videos.")
    unique = list(dict.fromkeys(id_list))
    return unique[:limit] if limit is not None else unique


def _build_describe_frame_fn(
    cfg: Config, vision_model: str | None = None, *, operation: str
) -> Callable[[Path], str]:
    """Bind the EXTERNAL vision command into a `path -> caption` callable (#90).

    Shared by `digest-video --frames` and `redescribe-frames` so the two can never
    drift on which model, which command or which rubric language they use. An
    unconfigured `[vision].command` is a clear operator error raised BEFORE any
    work — there is no bundled default vision model.

    `operation` names the CALLER in the guard's error message (#90 review M7),
    e.g. `"digest-video --frames"` or `"redescribe-frames"` — a refactor once
    replaced a caller-specific message ("digest-video --frames requires an
    external vision model: …") with a generic "this operation needs…", and
    the operator lost the one clue telling them WHICH command failed. Required
    (no default) so BOTH call sites must identify themselves.

    The rubric renders in `cfg.output_language` (the wiki's language), NOT in
    `digest-video --language` (the audio language handed to the transcriber).
    """
    if not cfg.vision_command.strip():
        raise ValueError(
            f"{operation} needs an external vision model: set "
            "[vision].command in config.toml (there is no built-in default)."
        )
    model = vision_model or cfg.vision_model

    def _describe(path: Path) -> str:
        return describe_image(
            path, command=cfg.vision_command, model=model, language=cfg.output_language
        )

    return _describe


def _build_visual_config(cfg: Config, vision_model: str | None = None) -> VisualConfig:
    """Build the `--frames` visual-layer config from `[vision]` + `[frames]` (#44 PR4).

    Binds `extract_key_frames` (ffmpeg, threshold/interval from `[frames]`), the two
    reducers — slides capped by `[frames].max_frames`, SILENT non-slide footage by
    `[frames].footage_max_frames`, both after the same dedup — and the shared
    `_build_describe_frame_fn` seam so `digest_videos` calls them with just a
    path. The vision guard, the model override and the rubric language all live
    in that shared helper, so `digest-video --frames` and `redescribe-frames`
    cannot drift apart on any of the three.
    """
    describe_fn = _build_describe_frame_fn(cfg, vision_model, operation="digest-video --frames")

    def _extract(path: Path) -> list[KeyFrame]:
        # RAW frames (cap=False): classification runs on the full distribution; the
        # reduce step below dedupes + caps only what the vision model describes.
        return extract_key_frames(
            path,
            threshold=cfg.frames_scene_threshold,
            interval_seconds=cfg.frames_interval_seconds,
            cap=False,
        )

    def _reduce(frames: list[KeyFrame]) -> list[KeyFrame]:
        return select_frames(
            frames,
            dedupe=cfg.frames_dedupe,
            dedupe_distance=cfg.frames_dedupe_distance,
            max_frames=cfg.frames_max_frames,
        )

    def _reduce_footage(frames: list[KeyFrame]) -> list[KeyFrame]:
        return select_frames(
            frames,
            dedupe=cfg.frames_dedupe,
            dedupe_distance=cfg.frames_dedupe_distance,
            max_frames=cfg.frames_footage_max_frames,
            cap_setting=FOOTAGE_CAP_SETTING,
        )

    return VisualConfig(
        media_root=cfg.media_dir,
        extract_fn=_extract,
        describe_fn=describe_fn,
        reduce_fn=_reduce,
        footage_reduce_fn=_reduce_footage,
    )


def _run_digest_video(
    cfg: Config,
    *,
    ids: str | None,
    topic: str | None,
    all_pending: bool,
    source: str,
    limit: int | None,
    force: bool,
    language: str | None,
    frames: bool,
    vision_model: str | None = None,
    keep_transcript: bool = False,
) -> None:
    """Digest selected videos into `x_video` transcript sources; persist + summarise.

    Flow: load → resolve selection → ephemeral fetch + EXTERNAL transcribe +
    attach (dedup by video identity, in memory) → snapshot → persist. The default
    path fetches + transcribes once per video identity. `--keep-transcript`
    partitions one identity by stored transcript: one fetch per distinct stored
    value (plus a missing-transcript batch), zero ASR for stored batches. The
    transcriber is invoked via `transcribe_media` bound to the `[transcribe]`
    config (command / model) + `--language`. `--frames` (opt-in, #44 PR4) also
    extracts key frames and describes them via the EXTERNAL `[vision]` command,
    attaching them to slide videos and to silent non-slide footage (a talking-head
    with speech is skipped). `--keep-transcript` reuses each item's own stored
    transcript instead of re-running the transcriber. The visual layer is still
    redone; a completed forced re-digest replaces old frames and clears the
    long-form digest even if vision fails or the video is reclassified. The
    `pre-digest-video` snapshot manages the store files `items.json`, `state.json`,
    `vocab.yaml`, and `topics.json`: it restores those present in the snapshot (and
    removes a live one absent there), but never contains `data/media/` or frame PNGs.
    The selected items' `data/media/<id>/frames/` directories have already been
    deleted or overwritten by the time it is taken. Before re-digesting items that
    currently have frames, copy those directories aside and restore them with the snapshot
    if needed; otherwise restored metadata can point to missing files or different
    pixels. Hollow items have no prior frames and are not exposed to this loss. The
    command is destructive (rewrites `items.json`), so it auto-snapshots BEFORE
    the save — but only when something
    was attached (a pure already-digested / no-video run writes nothing, so it
    takes no snapshot). A snapshot failure propagates and aborts before any
    `items.json` write.
    """
    store = load_store(cfg.items_path)
    id_list = _resolve_digest_ids(store, ids, topic, all_pending, source, limit)
    visual = _build_visual_config(cfg, vision_model) if frames else None

    def _transcribe(path: Path) -> Transcript:
        return transcribe_media(
            path,
            command=cfg.transcribe_command,
            model=cfg.transcribe_model,
            language=language,
        )

    report = digest_videos(
        store,
        id_list,
        force=force,
        keep_transcript=keep_transcript,
        transcribe_fn=_transcribe,
        visual=visual,
    )
    if report.changed > 0:
        _auto_snapshot(cfg, "digest-video")
        save_store(store, cfg.items_path)
    typer.echo(format_digest_summary(report))


@app.command(name="digest-video")
@_handle_cli_errors
def digest_video(
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated item IDs."),
    topic: str | None = typer.Option(
        None, "--topic", help="Select videos by the item's primary_topic."
    ),
    all_pending: bool = typer.Option(
        False, "--all-pending", help="Select every video in the pending (fetchable) state."
    ),
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of items to process in this run."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-process items that already have an x_video source; they are re-transcribed "
        "unless --keep-transcript.",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        help="Language to record on the transcript if the transcriber does not report one "
        "(e.g. en, es). The transcriber auto-detects; it is not passed to it as a flag.",
    ),
    frames: bool = typer.Option(
        False,
        "--frames",
        help="Visual layer (opt-in): extract key frames, describe them with the EXTERNAL "
        "vision model (`\\[vision].command`) and embed them in the note. "
        "Slides are described; a talking head WITH speech is skipped (the transcript already "
        "covers it; it is logged); a silent video without slides is described as footage, "
        "capped by `\\[frames].footage_max_frames`.",
    ),
    vision_model: str | None = typer.Option(
        None,
        "--vision-model",
        help="Override `\\[vision].model` for this run: the name is passed as "
        "--model to the vision command. With a multi-backend wrapper this picks the "
        "model per run (e.g. opus → cloud, qwen-7b → local). Requires --frames.",
    ),
    keep_transcript: bool = typer.Option(
        False,
        "--keep-transcript",
        help="With --frames --force: redo the visual layer and reuse the stored transcript "
        "(the ASR is not run again). Like every --force, it clears the long-form digest "
        "and the item goes back through enrich and video-digest. Use it to recover hollow "
        "videos.",
    ),
) -> None:
    """Transcribe stored videos and attach the transcript as an `x_video` source.

    For each selected video: ephemeral download (reuses `fetch-video`) →
    transcribe with a local EXTERNAL transcriber (config `transcribe.command`,
    `parakeet-mlx` by default; the ML does NOT live in xbrain) → attach the
    transcript to the item as `ContentSourceSuccess(kind="x_video")` → discard
    the bytes. Videos are **deduplicated by identity** (the stable id in the mp4
    path, not the signed URL): on the default path, N bookmarks of the same video
    are downloaded and transcribed ONCE and all get the same transcript. With
    `--keep-transcript`, items are split by their stored transcript: one download
    per distinct value (plus the batch without a transcript), zero ASR for the
    stored batches. A video without speech/audio is attached with empty text +
    `has_speech=False` (never a hard failure); if it also ends up without frames,
    the summary counts it in `Hollow (no speech, no frames): N`. Idempotent:
    skips items that already have an x_video source unless `--force`. Destructive
    (rewrites `items.json`) → auto-snapshot before writing. There is never more
    than one video on disk at a time (ephemeral). Select with `--ids`, `--topic`
    or `--all-pending`.

    `--frames` (opt-in, visual layer PR4): extract key frames with ffmpeg
    (EXTERNAL), describe them with the EXTERNAL vision model
    (`\\[vision].command`), attach the descriptions to the `x_video` source and
    embed the frames in the note as photos. Slides are described. A talking head
    is skipped only when the video has speech (the transcript already covers it;
    the reason is logged). A silent video without slides is described as footage,
    capped by `\\[frames].footage_max_frames`. Without `--frames` the flow is the
    PR2/PR3 one, except that the summary may end with
    `Hollow (no speech, no frames): N` (silent videos are left without frames).
    """
    cfg = _config()
    if vision_model and not frames:
        raise typer.BadParameter("--vision-model requires --frames (the visual layer is off)")
    if keep_transcript and not frames:
        raise typer.BadParameter(
            "--keep-transcript requires --frames (it only re-runs the visual layer)"
        )
    if keep_transcript and not force:
        raise typer.BadParameter(
            "--keep-transcript requires --force (it re-digests already-digested videos)"
        )
    _run_digest_video(
        cfg,
        ids=ids,
        topic=topic,
        all_pending=all_pending,
        source=source.value,
        limit=limit,
        force=force,
        language=language,
        frames=frames,
        vision_model=vision_model,
        keep_transcript=keep_transcript,
    )


def _redescribe_wants_whole_corpus(
    ids: str | None, topic: str | None, source: str, limit: int | None
) -> bool:
    """True when NOTHING narrows the `redescribe-frames` selection (#90 finding A).

    Extracted so `_resolve_redescribe_ids` reads as one decision, not four ANDed
    conditions: the "every stale video" shortcut is safe ONLY when no selector
    (`--ids`/`--topic`) AND no cap/scope (`--limit`/`--source`) is set — otherwise
    `--limit 10` or `--source bookmarks` would be silently ignored.
    """
    return not ids and not topic and limit is None and source == "all"


def _warn_missing_ids(store: dict[str, Item], explicit_ids: list[str]) -> None:
    """Echo which of an explicit `--ids` list are absent from the store.

    A warning, not an error (#90 pre-flight finding C): the whole-corpus repair
    must keep working even when one id in an `--ids` list is a typo. Without
    this, an unknown id silently vanishes inside `stale_video_sources`'s
    store-membership filter (`{i: store[i] for i in item_ids if i in store}`),
    and "no such item" reads identically to "already at the contract".
    """
    missing = list(dict.fromkeys(i for i in explicit_ids if i not in store))
    if missing:
        typer.echo(
            f"redescribe-frames: warning, id(s) not found in the store: {', '.join(missing)}",
            err=True,
        )


def _stale_item_ids(
    store: dict[str, Item], *, topic: str | None, source: str, force: bool
) -> list[str]:
    """Item ids carrying a re-caption candidate, scoped by `--topic`/`--source`
    — resolved from the ENGINE's OWN population (#90 review I2).

    `_resolve_redescribe_ids` used to narrow via `list_video_entries`, the
    `list-videos` CATALOG (one row per video *media* entry). That catalog has no
    staleness filter, so `--limit N` truncated it BEFORE `stale_video_sources`
    (inside `redescribe_frames` itself) ever got a look — measured: on a store
    with item 1 already current and item 2 stale, `--limit 1` selected item 1
    (first in store order), the engine then found nothing stale in it, and
    reported "0 vídeos seleccionados" — identically on every repeat, so a staged
    backfill never advanced.

    It also silently drifted from the engine's real population (#90 review M8):
    the catalog is "one row per video *media* entry", while the engine walks
    "every `x_video` success carrying frames" — an item with frames but an empty
    `media` list is repaired by an unscoped run yet invisible to `--limit`/
    `--source` here. Selecting straight from `stale_video_sources` (`force`
    threaded through, so `--force` widens "stale" to "every frame-bearing
    source", exactly like the engine itself) binds the two populations in code
    instead of leaving them to agree by convention (this repo's rule 5).

    `--topic`/`--source` are applied by reusing `video_select._primary_topic`
    and `_scope_by_source` — the SAME functions `list_video_entries` uses
    internally — so this can never drift from the catalog's own notion of
    either filter.
    """
    scoped = _scope_by_source(store, source)
    ids: list[str] = []
    for item_id, _source in stale_video_sources(store, force=force):
        item = scoped.get(item_id)
        if item is None:
            continue
        if topic is not None and _primary_topic(item) != topic:
            continue
        ids.append(item_id)
    return list(dict.fromkeys(ids))


def _resolve_redescribe_ids(
    store: dict[str, Item],
    ids: str | None,
    topic: str | None,
    source: str,
    limit: int | None,
    *,
    force: bool = False,
) -> list[str] | None:
    """Resolve the re-description selection, or `None` for "every stale video".

    Unlike `digest-video`, NO selector means "the whole backfill" rather than an
    error: this is a corpus-wide repair, it is idempotent by contract version, and
    `--dry-run` previews it for free — BUT ONLY when NOTHING narrows the run.
    `--limit`/`--source` must still apply on the whole-corpus path (#90 pre-flight
    finding A): returning `None` the instant `--ids`/`--topic` are both absent —
    without checking `limit`/`source` — would silently ignore `--limit 10`, the
    obvious way to test the waters before a multi-thousand-frame backfill, and
    `--source bookmarks`, describing the WHOLE corpus (every source) at full
    vision cost instead. So the "every stale video" shortcut fires only when
    `limit is None` and `source == "all"` too; otherwise the selection is
    expanded via `_stale_item_ids` — the ENGINE's own stale population, not the
    `list-videos` catalog (#90 review I2 — see `_stale_item_ids` for the bug this
    replaced). `force` is threaded through so `--force` widens the population the
    same way it widens the engine's own selection.

    `--ids` stay verbatim, never re-scoped by `--source`/staleness (matching the
    sibling `_resolve_digest_ids`) — but an id absent from the store, or the
    whole `--ids` value collapsing to nothing after stripping whitespace (e.g.
    `--ids "   "`), is now ECHOED, not silently dropped (#90 pre-flight finding C
    + review M5): `stale_video_sources` filters `{i: store[i] for i in item_ids
    if i in store}`, so a typo'd `--ids 999` would otherwise report zero videos
    and the operator could not tell "no such item" from "already at the
    contract" — and an unmatched `--topic` was the SAME silent blind spot
    (`--topic no-such-topic` printed "0 vídeos seleccionados", indistinguishable
    from an up-to-date corpus). All three stay WARNINGS, not hard errors —
    unlike `_resolve_fetch_ids`'s all-selectors-empty case — because the
    whole-corpus repair must keep working even when one id in an `--ids` list is
    bad.
    """
    if _redescribe_wants_whole_corpus(ids, topic, source, limit):
        return None
    id_list: list[str] = []
    if ids:
        id_list.extend(_resolve_explicit_redescribe_ids(store, ids))
    if topic or not ids:
        # Either an explicit `--topic` filter, or NOTHING selected explicitly at
        # all (the whole-corpus-but-scoped-by-limit/source path from finding A
        # above) — either way, expand from the engine's own stale population.
        # `topic=None` here just means "no topic filter", still scoped by
        # `source`.
        id_list.extend(_resolve_topic_scoped_redescribe_ids(store, topic, source, force))
    unique = list(dict.fromkeys(id_list))
    return unique[:limit] if limit is not None else unique


def _resolve_explicit_redescribe_ids(store: dict[str, Item], ids: str) -> list[str]:
    """Split/strip/dedupe an explicit `--ids` value, warning (#90 review M5)
    when it collapses to nothing after stripping (e.g. `--ids "   "`) and when
    any id is absent from the store (#90 pre-flight finding C, `_warn_missing_
    ids`). Extracted out of `_resolve_redescribe_ids` to keep that function's
    own branching at a glance (radon: this split is what keeps it at grade B).
    """
    explicit = [part.strip() for part in ids.split(",") if part.strip()]
    if not explicit:
        typer.echo(
            "redescribe-frames: warning, --ids holds no valid id after stripping whitespace.",
            err=True,
        )
    _warn_missing_ids(store, explicit)
    return explicit


def _resolve_topic_scoped_redescribe_ids(
    store: dict[str, Item], topic: str | None, source: str, force: bool
) -> list[str]:
    """Expand via `_stale_item_ids`, warning (#90 review M5) when an explicit
    `--topic` matches nothing — the same silent blind spot an unknown `--ids`
    used to have. `topic=None` just means "no topic filter" (no warning)."""
    topic_ids = _stale_item_ids(store, topic=topic, source=source, force=force)
    if topic is not None and not topic_ids:
        typer.echo(
            f"redescribe-frames: warning, --topic {topic!r} matches no video.",
            err=True,
        )
    return topic_ids


def _dry_run_describe_fn(path: Path) -> str:
    """Sentinel `describe_fn` for `redescribe-frames --dry-run` (#90 pre-flight
    finding B).

    `--dry-run` must preview a run WITHOUT `[vision].command` being configured at
    all: `_build_describe_frame_fn` raises the moment it is CALLED when that is
    unset, so building the real `describe_fn` eagerly — above the dry-run branch,
    as the initial plan did — would make the FREE preview demand the same
    configuration as the real run, contradicting this command's own docstring
    promise that `--dry-run` costs nothing. `redescribe_frames`'s dry-run branch
    (`_preview_source`) never calls `describe_fn`, so passing this sentinel is
    also a live regression guard: if a future change ever makes the dry-run path
    call the model after all, it fails loudly here instead of silently spending
    one vision call per frame — 2077 of them, on the corpus that motivated this
    module — on every "free" preview.
    """
    raise RuntimeError(f"redescribe-frames --dry-run must not call the vision model (got {path})")


@app.command(name="redescribe-frames")
@_handle_cli_errors
def redescribe_frames_command(
    ids: str | None = typer.Option(None, "--ids", help="Comma-separated IDs."),
    topic: str | None = typer.Option(None, "--topic", help="Filter by the item's primary_topic."),
    source: Source = typer.Option(Source.all, help="bookmarks | tweets | all"),
    limit: int | None = typer.Option(
        None, "--limit", help="Maximum number of items to re-describe in this run."
    ),
    force: bool = typer.Option(
        False, "--force", help="Also re-describe frames already up to date."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be re-described without writing anything."
    ),
    vision_model: str | None = typer.Option(
        None, "--vision-model", help="Vision model for this run (see scripts/xbrain-vision)."
    ),
) -> None:
    """Re-describe the ALREADY-stored key frames with the current caption rubric.

    Reads the pixels in `data/media/<id>/frames/`: it does NOT download videos,
    does NOT call X and does NOT use ffmpeg. It skips videos whose captions were
    already generated under the current contract (`--force` regenerates them
    anyway), so re-running it on an up-to-date corpus costs zero model calls.

    When a caption actually changes, it bumps the item's `content.fetched_at`,
    the trigger that makes `enrich`, `video-digest` and `generate` run again
    with the new evidence. If none changes, it touches nothing.

    Destructive (rewrites `items.json`), so it auto-snapshots BEFORE saving —
    but only when there is something to save. An abort midway (`RuntimeError`
    from the circuit breaker, `VisionNotFound`, an `OSError`, Ctrl-C) still
    persists what was re-described before the failure — vision work already paid
    for is not thrown away — and the error still propagates with a non-zero exit
    code (#90 review I1).

    `--dry-run` does NOT require `\\[vision].command` to be configured: it previews
    the backfill (selected videos, captions that would be regenerated/fail)
    without calling the model even once. `Re-propagated` in the preview is NOT
    a prediction — it is only known by actually running the model.
    """
    cfg = _config()
    if vision_model and dry_run:
        raise typer.BadParameter(
            "--vision-model requires a real run (--dry-run never calls the vision model)"
        )
    store = load_store(cfg.items_path)
    selected = _resolve_redescribe_ids(store, ids, topic, source.value, limit, force=force)
    # A dry run must not demand `[vision].command` at all — see `_dry_run_
    # describe_fn` for why this can't just be the real describe_fn built eagerly.
    describe_fn = (
        _dry_run_describe_fn
        if dry_run
        else _build_describe_frame_fn(cfg, vision_model, operation="redescribe-frames")
    )
    # `report` is created HERE, before the engine call, so the `finally` below
    # holds the SAME instance the engine populated frame by frame — even if the
    # engine raises partway through (#90 review I1 + M2). Without this, a
    # mid-run `RuntimeError`/`VisionNotFound`/`OSError`/Ctrl-C would discard
    # every already-paid-for vision call: the store was only ever written AFTER
    # `redescribe_frames` returned, and an exception means it never does.
    report = RedescribeReport()
    try:
        redescribe_frames(
            store,
            media_root=cfg.media_dir,
            describe_fn=describe_fn,
            item_ids=selected,
            force=force,
            dry_run=dry_run,
            report=report,
        )
    finally:
        _finish_redescribe_run(cfg, store, report, dry_run=dry_run)


def _finish_redescribe_run(
    cfg: Config, store: dict[str, Item], report: RedescribeReport, *, dry_run: bool
) -> None:
    """The `redescribe-frames` `finally` body (#90 re-review item 4).

    Two invariants must hold even when the snapshot/save step below RAISES:

    1. The summary always prints — it is the FIRST thing this does, before
       anything that can fail — so an abort (in the engine call above, OR
       right here) still tells the operator what landed. Measured bug: with
       the summary echoed AFTER the save, a `save_store` failure meant it
       never ran at all, even though the run's actual counters were sitting
       right there on `report`.
    2. A save/snapshot failure must never REPLACE an exception already
       propagating. That is Python's DEFAULT `finally` behaviour — a new
       exception raised while one is in flight silently becomes the
       reported one (the original demoted to `__context__` and never
       surfaced to the operator) — so this checks `sys.exc_info()`
       explicitly: with an original error in flight, the save failure is
       reported (not swallowed) but the ORIGINAL keeps propagating
       unmodified. With nothing in flight, the save/snapshot failure IS the
       error, and propagates exactly as it did before this fix.
    """
    typer.echo(format_redescribe_summary(report, dry_run=dry_run))
    if dry_run:
        typer.echo("--dry-run: the store was not touched.")
        return
    if report.frames_described == 0:
        return
    # Snapshot strictly BEFORE the write — the operator's only undo for a
    # 2077-caption rewrite.
    original_error = sys.exc_info()[1]
    try:
        _auto_snapshot(cfg, "redescribe-frames")
        save_store(store, cfg.items_path)
    except Exception as save_exc:
        if original_error is None:
            raise
        typer.echo(
            f"WARNING: on top of the error above, saving the store failed: {save_exc}",
            err=True,
        )


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
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
) -> None:
    """Enrich the items with a summary + topics."""
    cfg = _config()
    store = load_store(cfg.items_path)
    vocab_topics = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab_topics:
        raise RuntimeError("No vocabulary — run `xbrain vocab` first.")

    if apply is not None:
        executor_name, judgments = import_worksheet(apply)
        enriched, invalid = apply_worksheet_judgments(store, judgments, vocab_topics, executor_name)
        save_store(store, cfg.items_path)
        typer.echo(f"Worksheet applied: {enriched} items enriched")
        _report_invalid(invalid)
        return

    chosen = executor or cfg.enrich_executor

    if chosen in ("manual", "claude-code"):
        pending = items_pending_enrichment(
            store, _parse_date(since), _parse_date(until, end_of_day=True)
        )
        if not pending:
            typer.echo("No items pending enrichment.")
            return
        worksheet = cfg.data_dir / "enrich-worksheet.json"
        export_worksheet(pending, vocab_topics, worksheet, chosen, cfg.output_language)
        typer.echo(
            f"{len(pending)} items exported to {worksheet}\n"
            f"Fill in the `judgments` array (with Claude Code or by hand) and run:\n"
            f"  xbrain enrich --apply {worksheet}"
        )
        return

    if chosen != "api":
        raise ValueError(f"Unknown executor: {chosen!r}")

    enriched, invalid = enrich_with_executor(
        store,
        ApiExecutor(model=cfg.enrich_model, output_language=cfg.output_language),
        vocab_topics,
        _parse_date(since),
        _parse_date(until, end_of_day=True),
    )
    save_store(store, cfg.items_path)
    typer.echo(f"Enriched: {enriched} items")
    _report_invalid(invalid)


@app.command(name="video-digest")
@_handle_cli_errors
def video_digest_command(
    executor: str | None = typer.Option(
        None, help="manual | claude-code (default: the executor set in config.toml)"
    ),
    apply: Path | None = typer.Option(
        None, "--apply", help="Import a filled worksheet and apply it"
    ),
) -> None:
    """Generate a readable long-form digest per video, from its transcript + frames."""
    cfg = _config()
    store = load_store(cfg.items_path)

    if apply is not None:
        # Snapshot on the APPLY branch — this is the one that mutates `items.json`
        # (writes every `source.digest` + `save_store`); export only writes the
        # worksheet JSON. Mirrors `describe`'s `describe-apply` snapshot.
        _auto_snapshot(cfg, "video-digest-apply")
        judgments = import_video_digest_worksheet(apply)
        applied, invalid = apply_video_digest_judgments(store, judgments)
        save_store(store, cfg.items_path)
        typer.echo(f"Worksheet applied: {applied} video digests")
        _report_invalid(invalid)
        return

    chosen = executor or cfg.enrich_executor
    if chosen not in ("manual", "claude-code"):
        raise ValueError(
            f"Executor {chosen!r} is not supported for video-digest — use manual|claude-code."
        )
    pending = items_pending_video_digest(store)
    if not pending:
        typer.echo("No videos pending a digest.")
        return
    worksheet = cfg.data_dir / "video-digest-worksheet.json"
    export_video_digest_worksheet(pending, worksheet, chosen, cfg.output_language)
    typer.echo(
        f"{len(pending)} videos exported to {worksheet}\n"
        f"Fill in the `judgments` array (with Claude Code or by hand) and run:\n"
        f"  xbrain video-digest --apply {worksheet}"
    )


def _resolve_verify_executor(executor: str | None, cfg: Config) -> str:
    """The worksheet track for `verify`, defaulting to config; only manual/claude-code."""
    chosen = executor or cfg.enrich_executor
    if chosen not in ("manual", "claude-code"):
        raise ValueError(
            f"Executor {chosen!r} is not supported for verify — use manual|claude-code."
        )
    return chosen


def _write_verify_report(cfg: Config, json_report: str, md_report: str) -> None:
    """Persist `verify-report.{json,md}` and echo its headline + path."""
    (cfg.data_dir / "verify-report.json").write_text(json_report, encoding="utf-8")
    (cfg.data_dir / "verify-report.md").write_text(md_report, encoding="utf-8")
    lines = md_report.splitlines()
    typer.echo(lines[2] if len(lines) > 2 else "Report written")
    typer.echo(f"Report: {cfg.data_dir / 'verify-report.md'}")


def _verify_write_verdicts(
    cfg: Config,
    records: list[dict],
    fingerprints: dict[tuple[str, str], str],
    contracts: dict[tuple[str, str], str],
) -> None:
    """Opt-in write path for `verify --apply --write-verdicts` (and its post-audit twin,
    `verify --audit --apply --write-verdicts`): persist each FINAL verdict onto its item with
    the JUDGED output fingerprint (`fingerprints`, stamped at judge-worksheet export), so
    `generate` can badge a still-current FAIL/REVIEW.

    `records` are the verdicts as rendered in the report — the aggregate on the plain path,
    the MERGED post-audit records on the audit path. It never re-derives a verdict: it
    consumes whatever the (guard-enforced) merge produced.

    Mutates `items.json`, so it auto-snapshots `data/` first (label
    `pre-verify-write-verdicts`) — undoable with `xbrain snapshot restore`. Echoes the
    written/skipped tally so a dropped verdict is never silent.
    """
    _auto_snapshot(cfg, "verify-write-verdicts")
    store = load_store(cfg.items_path)
    result = apply_verdicts_to_store(store, records, fingerprints, contracts)
    save_store(store, cfg.items_path)
    typer.echo(f"{result.summary()} → {cfg.items_path}")


def _echo_invalidated_verdicts(cfg: Config, store: dict[str, Item]) -> None:
    """Say how many STORED verdicts the current contract has retired, before exporting a
    fresh worksheet.

    The number is the point. A contract change (a rubric rewrite, a new evidence surface,
    a re-fetched article) silently retires verdicts that were perfectly valid under the old
    rules — they paint no badge and must be re-judged. Reporting it is the difference
    between "the verification layer covers the corpus" and knowing how much of it actually
    does right now.
    """
    invalidated, stored = count_invalidated_verdicts(store, cfg.output_language)
    if invalidated:
        typer.echo(
            f"⚠️  {invalidated} of {stored} stored verdicts are now STALE: they were "
            "judged under another contract (another output, source or rubric). They "
            "paint no badge; they must be re-verified."
        )


def _audit_write_fingerprints(
    records: list[dict], apply: list[Path], stamp: str = "output_fingerprint"
) -> dict[tuple[str, str], str]:
    """The JUDGED fingerprints for the post-audit write.

    The MERGED RECORDS are the authority — they are what the report being written describes,
    and they carry the stamp taken at judge-worksheet export. The applied audit worksheet is
    only a CROSS-CHECK: a second copy of that same stamp, able to DROP a key it disagrees with
    (a hand-edited artifact → `fingerprint-missing` → the record is skipped), never to supply
    one the record lacks. Nothing here recomputes a fingerprint from the live store — that is
    the invariant the whole plumbing exists to protect (#79).
    """
    return cross_check_fingerprints(
        record_fingerprints(records, stamp), import_verify_fingerprints(apply, stamp)
    )


def _verify_audit_apply(
    cfg: Config, aggregated: list[dict], apply: list[Path], force: bool, write_verdicts: bool
) -> None:
    """Merge one auditor's worksheet onto the aggregate, re-render the report, and — with
    `--write-verdicts` — persist the MERGED (post-audit) verdicts.

    The audited verdict is the authoritative one: it is what `--write-verdicts` writes here,
    so a FAIL the auditor REVOKED never badges a note, and a failure the auditor CONFIRMED (or
    added) does. The write consumes `merge_audit`'s OUTPUT — the monotonic floor, confidence
    gate, mass-revocation guard and anti-washing logic all still stand between the auditor and
    the store.

    The audit is a SINGLE independent auditor (judge ≠ party) in ONE pass over the full
    consequential set. More than one `--apply` is rejected, and a second `--audit
    --apply` on an already-audited report is refused (unless `--force`): re-reading the
    already-shrunk FAIL set would let N single-revoke runs bypass the mass-revocation
    guard by splitting revocations across runs. Report-only unless `--write-verdicts`.
    """
    if len(apply) > 1:
        raise ValueError(
            "The audit comes from a single independent auditor — pass exactly one --apply "
            f"(got {len(apply)})."
        )
    if not force and any(isinstance(r, dict) and r.get("audited") for r in aggregated):
        raise ValueError(
            "verify-report.json already contains an applied audit. Re-aggregate the judges "
            "(xbrain verify --apply ...) before re-auditing, or pass --force to "
            "overwrite knowingly (this prevents bypassing the mass-revocation guard "
            "by spreading revocations across passes)."
        )
    audits = import_audit_judgments(apply[0])
    records, audit_log = merge_audit(aggregated, audits)
    _echo_audit_log(audit_log)
    if write_verdicts:
        _reject_unaudited_write(aggregated, audit_log)
        _verify_write_verdicts(
            cfg,
            records,
            _audit_write_fingerprints(records, apply),
            _audit_write_fingerprints(records, apply, "contract_fingerprint"),
        )
    # The report is written LAST, and only once the store write has succeeded. It is the report
    # that carries `audited: True`, which the guard above reads to refuse a second audit — so
    # marking it before a write that then dies would block the retry behind `--force`, which
    # `--write-verdicts` may not use. Report written ⇒ store already written.
    _write_verify_report(cfg, *render_verify_report(records, audit_log))


def _echo_audit_log(audit_log: dict) -> None:
    """The one-line summary of what the merge did — matched, washed, unmatched, guard."""
    unmatched = audit_log["unmatched"]
    typer.echo(
        f"Audit: {audit_log['matched']}/{audit_log['supplied']} applied, "
        f"{len(audit_log['washed'])} reverted to a lower severity"
        + (f", {len(unmatched)} unmatched" if unmatched else "")
        + (" · mass-revocation GUARD TRIGGERED" if audit_log["mass_revocation_guard"] else "")
    )


def _reject_unaudited_write(aggregated: list[dict], audit_log: dict) -> None:
    """Refuse `--write-verdicts` when the audit matched NOTHING while consequential records exist.

    An `audits` block that matches no record is an audit that never happened: `merge_audit` passes
    every record through untouched, so the write would persist the PRE-audit aggregate — the set
    this path exists to keep out of the store — while echoing `0/N applied` and exiting 0. The
    report-only run stays allowed (it is a human-read artifact); the STORE write is refused.
    """
    if audit_log["matched"]:
        return
    pending = consequential_records(aggregated)
    if pending:
        raise ValueError(
            f"The audit matched no record, but there are {len(pending)} consequential verdicts "
            "(FAIL/divergent) left unaudited. Writing them would persist the PRE-audit aggregate. "
            "Fill in `audits` in the worksheet (or drop --write-verdicts to re-render only "
            "the report)."
        )


def _verify_audit(
    cfg: Config, executor: str | None, apply: list[Path], force: bool, write_verdicts: bool
) -> None:
    """The judge≠party audit mode of `verify` (`--audit`): export or apply.

    Reads the aggregated records back from the `verify-report.json` PR-1 wrote.
    With `--apply` it merges the auditor's decisions onto them and re-renders the
    report (and, with `--write-verdicts`, persists the merged post-audit verdicts);
    without it, it exports an audit worksheet for the consequential (FAIL/divergent) subset.
    """
    aggregated = load_report_records(cfg.data_dir / "verify-report.json")
    if apply:
        _verify_audit_apply(cfg, aggregated, apply, force, write_verdicts)
        return
    chosen = _resolve_verify_executor(executor, cfg)
    records = consequential_records(aggregated)
    if not records:
        typer.echo("No consequential verdicts (FAIL/divergent) to audit.")
        return
    worksheet = cfg.data_dir / "verify-audit-worksheet.json"
    exported, skipped = export_audit_worksheet(
        records, load_store(cfg.items_path), worksheet, chosen, cfg.output_language
    )
    skip_note = (
        f" ({len(skipped)} skipped, no item in the store: {', '.join(skipped)})" if skipped else ""
    )
    typer.echo(
        f"{exported} consequential verdicts exported to {worksheet}{skip_note}\n"
        f"Fill in `audits` (independent auditor, judge ≠ party) and run:\n"
        f"  xbrain verify --audit --apply {worksheet.name}"
    )


@app.command(name="verify")
@_handle_cli_errors
def verify_command(
    target: str = typer.Option("all", help="summary | digest | topics | all"),
    executor: str | None = typer.Option(
        None, help="manual | claude-code (default: the executor set in config.toml)"
    ),
    apply: list[Path] = typer.Option(
        None, "--apply", help="Filled worksheet(s), one per judge — aggregated into a report"
    ),
    audit: bool = typer.Option(
        False,
        "--audit",
        help="Judge≠party audit of the consequential (FAIL/divergent) verdicts",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Re-apply an audit onto an already-audited report (--audit --apply)",
    ),
    write_verdicts: bool = typer.Option(
        False,
        "--write-verdicts",
        help="Opt-in: also persist each verdict + output fingerprint onto its item "
        "(so `generate` can badge it). With --audit it writes the AUDITED verdicts. "
        "Requires --apply; snapshots data/ first.",
    ),
) -> None:
    """Verify the enrichment outputs (faithfulness + adherence) with LLM judges."""
    cfg = _config()

    if write_verdicts and not apply:
        raise typer.BadParameter(
            "--write-verdicts requires --apply — there is nothing judged to persist on a "
            "bare export."
        )

    if write_verdicts and force:
        raise typer.BadParameter(
            "--write-verdicts cannot be combined with --force. --force exists to bypass the "
            "already-audited guard, and that guard is what keeps the mass-revocation guard "
            "honest: each forced re-audit re-renders the report from the MERGED records, so the "
            "FAIL set shrinks and N single-revoke runs can clear every FAIL without ever "
            "tripping it (it needs >=2 FAILs). Report-only forced re-audits stay available; "
            "re-aggregate the judges (xbrain verify --apply ...) before writing verdicts."
        )

    if audit:
        _verify_audit(cfg, executor, apply, force, write_verdicts)
        return

    if apply:
        # Aggregate the N judges' passes and write the report. Default is report-only —
        # the store is untouched unless --write-verdicts opts into the additive write.
        # The JUDGED fingerprints come from the worksheet `items` block (stamped at export),
        # so a verdict binds to the output the judge saw — never a live recompute. They are
        # stamped onto the records too, carrying them into verify-report.json → the audit.
        fingerprints = import_verify_fingerprints(apply)
        contracts = import_verify_fingerprints(apply, "contract_fingerprint")
        aggregated = aggregate_verify_judgments([import_verify_judgments(path) for path in apply])
        stamp_record_fingerprints(aggregated, fingerprints)
        stamp_record_fingerprints(aggregated, contracts, "contract_fingerprint")
        _write_verify_report(cfg, *render_verify_report(aggregated))
        if write_verdicts:
            _verify_write_verdicts(cfg, aggregated, fingerprints, contracts)
        return

    chosen = _resolve_verify_executor(executor, cfg)
    store = load_store(cfg.items_path)
    _echo_invalidated_verdicts(cfg, store)
    pairs = items_for_verification(store, parse_targets(target))
    if not pairs:
        typer.echo("No outputs to verify.")
        return
    worksheet = cfg.data_dir / "verify-worksheet.json"
    export_verify_worksheet(pairs, worksheet, chosen, cfg.output_language)
    typer.echo(
        f"{len(pairs)} outputs exported to {worksheet}\n"
        f"Copy it N times (one per judge), fill in `judgments` in each, and run:\n"
        f"  xbrain verify --apply ws1.json --apply ws2.json ..."
    )


def _mark_for_regenerate(store: dict, cfg: Config, regenerate: bool) -> None:
    """When `--regenerate` is set, drop every item's enrichment and persist."""
    if regenerate:
        for item in store.values():
            item.enriched = None
        save_store(store, cfg.items_path)
        typer.echo("All items marked for re-enrichment.")


def _vocab_apply(cfg: Config, store: dict, apply: Path, regenerate: bool) -> None:
    """`xbrain vocab --apply` — import a filled vocab worksheet."""
    topics, invalid = apply_vocab_worksheet(import_vocab_worksheet(apply))
    _report_invalid(invalid)
    if not topics:
        raise RuntimeError("The worksheet produced no valid topic.")
    if regenerate:
        _auto_snapshot(cfg, "vocab-regenerate")
    # Mark the store first: a crash here leaves items pending (a re-run re-marks
    # idempotently) — safer than vocab.yaml updated while items stay stale.
    _mark_for_regenerate(store, cfg, regenerate)
    save_vocab(topics, cfg.data_dir / "vocab.yaml")
    typer.echo(f"Vocabulary applied: {len(topics)} topics → {cfg.data_dir / 'vocab.yaml'}")


def _vocab_run(cfg: Config, store: dict, executor: str | None, regenerate: bool) -> None:
    """`xbrain vocab` — induce the taxonomy (worksheet export, or `api`)."""
    chosen = executor or cfg.enrich_executor
    if chosen in ("manual", "claude-code"):
        worksheet = cfg.data_dir / "vocab-worksheet.json"
        export_vocab_worksheet(store, cfg.vocab_target_count, worksheet, cfg.output_language)
        regen = " --regenerate" if regenerate else ""
        typer.echo(
            f"Corpus exported to {worksheet}\n"
            f"Induce the taxonomy (with Claude Code or by hand) and run:\n"
            f"  xbrain vocab --apply {worksheet}{regen}"
        )
        return
    if chosen != "api":
        raise ValueError(f"Unknown executor: {chosen!r}")
    if regenerate:
        _auto_snapshot(cfg, "vocab-regenerate")
    topics = induce_vocab(store, cfg.vocab_target_count, cfg.enrich_model, cfg.output_language)
    save_vocab(topics, cfg.data_dir / "vocab.yaml")
    _mark_for_regenerate(store, cfg, regenerate)
    typer.echo(f"Vocabulary induced: {len(topics)} topics → {cfg.data_dir / 'vocab.yaml'}")


@app.command()
@_handle_cli_errors
def vocab(
    regenerate: bool = typer.Option(
        False, help="Mark every item for re-enrichment against the new taxonomy"
    ),
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: the one in config.toml)"
    ),
    apply: Path | None = typer.Option(None, "--apply", help="Import a filled vocab worksheet"),
) -> None:
    """Induce the topic vocabulary (data/vocab.yaml) from the corpus."""
    cfg = _config()
    store = load_store(cfg.items_path)
    if not store:
        raise RuntimeError("The store is empty — run `xbrain extract` first.")
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
    typer.echo(f"Worksheet applied: {len(valid)} overviews · {written} pages written")
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
        typer.echo(f"Topic pages updated: {written} pages (no overviews pending).")
        return

    chosen = executor or cfg.enrich_executor
    if chosen in ("manual", "claude-code"):
        worksheet = cfg.data_dir / "topic-worksheet.json"
        export_topic_worksheet(inputs, worksheet, cfg.output_language)
        written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
        typer.echo(
            f"{len(inputs)} topics exported to {worksheet} · {written} pages written\n"
            f"Fill in the `judgments` array and run:\n"
            f"  xbrain topics --apply {worksheet}"
        )
        return
    if chosen != "api":
        raise ValueError(f"Unknown executor: {chosen!r}")

    judgments = synthesize_overviews_api(inputs, cfg.enrich_model, cfg.output_language)
    merge_overviews(pages, judgments, posts)
    save_topic_pages(pages, cfg.topics_path)
    written = write_topic_pages(cfg.output_dir, vocab, posts, pages, cfg.output_language)
    typer.echo(f"Topics synthesized: {len(judgments)}/{len(inputs)} · {written} pages written")


@app.command()
@_handle_cli_errors
def topics(
    resynth: bool = typer.Option(False, help="Re-synthesize every stale overview"),
    apply: Path | None = typer.Option(None, "--apply", help="Import a filled overview worksheet"),
    executor: str | None = typer.Option(
        None, help="api | manual | claude-code (default: the one in config.toml)"
    ),
) -> None:
    """Generate the topic pages: post lists + synthesized overviews."""
    cfg = _config()
    store = load_store(cfg.items_path)
    vocab = load_vocab(cfg.data_dir / "vocab.yaml")
    if not vocab:
        raise RuntimeError("No vocabulary — run `xbrain vocab` first.")
    if apply is not None:
        _topics_apply(cfg, store, vocab, apply)
    else:
        _topics_run(cfg, store, vocab, resynth, executor)


@app.command()
@_handle_cli_errors
def generate(
    since: str = typer.Option(None, help="ISO date, e.g. 2025-01-01"),
    until: str = typer.Option(None, help="ISO date; whole day inclusive, e.g. 2025-12-31"),
) -> None:
    """Generate the markdown notes in the vault."""
    _run_generate(_config(), _parse_date(since), _parse_date(until, end_of_day=True))


@app.command()
@_handle_cli_errors
def sync(
    headless: bool = typer.Option(False, "--headless/--no-headless", help=_HEADLESS_HELP),
) -> None:
    """extract + fetch + generate, in order."""
    cfg = _config()
    _run_extract(cfg, "all", None, None, headless=headless)
    _run_fetch(cfg, None, None, False, headless=headless)
    _run_generate(cfg, None, None)


@app.command()
@_handle_cli_errors
def status() -> None:
    """Show counters and the last run."""
    cfg = _config()
    store = load_store(cfg.items_path)
    state = load_state(cfg.state_path)
    typer.echo(f"Items: {len(store)}")
    typer.echo(f"  with a link: {sum(1 for i in store.values() if i.links)}")
    typer.echo(f"  with content: {sum(1 for i in store.values() if i.content)}")
    typer.echo(f"  enriched: {sum(1 for i in store.values() if i.enriched)}")
    typer.echo(f"  last bookmarks extraction: {state.bookmarks.last_run}")
    typer.echo(f"  last tweets extraction: {state.own_tweets.last_run}")


snapshot_app = typer.Typer(help="Manage snapshots of data/")
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


def _resolve_data_dir(cfg: Config, name: str | None) -> Path:
    """Resolve a snapshot name to its data dir, or `None` to the live `data/`.

    `xbrain diff` accepts a snapshot name (resolved via `snapshot_show`) OR
    `None` to mean "the current live `data/`" — the most common B-side of the
    comparison the user runs after a destructive op.
    """
    if name is None:
        return cfg.data_dir
    snapshot_dir, _ = snapshot.snapshot_show(cfg.data_dir, name)
    return snapshot_dir


@app.command()
@_handle_cli_errors
def diff(
    snapshot_a: str = typer.Argument(..., help="Snapshot name on the A side."),
    snapshot_b: str | None = typer.Argument(
        None,
        help="Snapshot name on the B side. Defaults to the live data/ directory.",
    ),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: 'text' (default) or 'json'.",
    ),
) -> None:
    """Compare two snapshots and surface drift.

    Reports reassigned items, topic-membership shifts, topic-overview drift
    (TF cosine similarity) and vocab changes. The B side defaults to the live
    `data/` directory so `xbrain diff <pre-snapshot>` answers "what did the
    last destructive op move?" with no extra arguments.
    """
    cfg = _config()
    if output_format not in ("text", "json"):
        raise ValueError(f"--format must be 'text' or 'json', got {output_format!r}")
    a_dir = _resolve_data_dir(cfg, snapshot_a)
    b_dir = _resolve_data_dir(cfg, snapshot_b)
    report = diff_snapshots(a_dir, b_dir)
    if output_format == "json":
        typer.echo(format_json(report))
    else:
        b_label = snapshot_b if snapshot_b is not None else "live data/"
        typer.echo("Comparing:")
        typer.echo(f"  A: {snapshot_a}")
        typer.echo(f"  B: {b_label}")
        typer.echo("")
        typer.echo(format_text(report))


@app.command(name="payload-stats")
@_handle_cli_errors
def payload_stats_command() -> None:
    """Measure the raw payloads actually on disk (count, size, projection).

    The disk figures first quoted for this feature were taken from an X *Article* fixture
    that contains no tweets. This measures the real thing.
    """
    cfg = _config()
    stats = payload_stats(cfg.payload_dir)
    if not stats["count"]:
        typer.echo("No payloads on disk yet. Run `xbrain sync` first.")
        return
    mean = stats["mean_gzipped_bytes"]
    typer.echo(
        f"{stats['count']} payloads · {stats['raw_bytes'] / 1e6:.1f} MB raw · "
        f"{stats['gzipped_bytes'] / 1e6:.1f} MB compressed · mean {mean:,} B/item"
    )
    for n in (10_000, 100_000):
        typer.echo(f"  projected to {n:,} items: {n * mean / 1e6:,.0f} MB")


@app.command(name="reextract")
@_handle_cli_errors
def reextract_command(
    apply: bool = typer.Option(False, "--apply", help="Write the re-parsed fields to the store"),
) -> None:
    """Re-run the parser over the STORED raw payloads — offline, no network.

    This is how a parse fix gets validated before it is applied: the dry run prints exactly
    what would change across the whole corpus. Items with no stored payload (everything
    ingested before payload persistence) are listed explicitly — "cannot be re-extracted" is
    never allowed to look like "re-extracted cleanly".
    """
    cfg = _config()
    store = load_store(cfg.items_path)
    if apply:
        _auto_snapshot(cfg, "reextract")
    report = reextract_from_payloads(store, cfg.payload_dir, apply=apply)
    typer.echo(report.summary())
    for item_id, field_name, old, new in report.changed[:20]:
        typer.echo(f"  {item_id} {field_name}: {str(old)[:40]!r} → {str(new)[:40]!r}")
    if apply:
        save_store(store, cfg.items_path)
        typer.echo(f"{len(report.changed)} field(s) updated → {cfg.items_path}")
    else:
        typer.echo("Dry run. Pass --apply to write.")


@app.command(name="refetch-truncated")
@_handle_cli_errors
def refetch_truncated_command(
    apply: bool = typer.Option(False, "--apply", help="Actually re-fetch from X (network)"),
    tabs: int | None = typer.Option(
        None,
        "--tabs",
        help=(
            "Reused tabs in parallel (default 3, max 4). Each one navigates "
            "IN THE SAME tab from one post to the next, with a random 5-30 s pause "
            "between loads: no browser per item and no machine-paced loading."
        ),
    ),
    headless: bool = typer.Option(
        False,
        "--headless/--no-headless",
        help=("Hidden browser. Headful (visible) by default — harder to fingerprint as a bot."),
    ),
) -> None:
    """List (or re-fetch) items whose tweet text was TRUNCATED at ingest.

    `legacy.full_text` is capped at 280 chars: X cuts a long post mid-word and appends a
    t.co self-link. Items ingested before the `note_tweet` fix carry half a sentence, and
    the generator — told to summarise it — finishes the sentence itself.

    The raw GraphQL payloads are NOT on disk (they are captured in-flight), so this is NOT
    a free re-parse: `--apply` re-fetches each affected tweet from X. It snapshots `data/`
    first. Without `--apply` it only reports, and writes the id list.
    """
    cfg = _config()
    store = load_store(cfg.items_path)
    targets = items_needing_refetch(store)
    path = cfg.data_dir / "truncated-items.json"
    path.write_text(
        json.dumps([{"id": i.id, "url": i.url, "text": i.text} for i in targets], indent=2),
        encoding="utf-8",
    )
    typer.echo(f"{len(targets)}/{len(store)} items truncated at ingest → {path}")
    if not apply:
        typer.echo("Dry run. Re-fetching requires the network: pass --apply.")
        return
    _auto_snapshot(cfg, "refetch-truncated")

    # Checkpoint as we go: a session expiry on item 400 of 535 must not discard the first
    # 400 repairs. This is deliberately human-paced browser work — hours of it.
    def _checkpoint() -> None:
        save_store(store, cfg.items_path)

    opened = clamp_tabs(tabs)
    typer.echo(
        f"Re-fetch in {opened} reused tab(s), random pause of "
        f"{PAUSE_MIN_MS // 1000}-{PAUSE_MAX_MS // 1000}s between loads. "
        "Ctrl-C keeps what was repaired up to the last checkpoint."
    )
    # `RefetchRateLimited` is a RuntimeError, so `_handle_cli_errors` already prints it as
    # a clean exit-1. All this has to guarantee is that the repairs made before X started
    # limiting us are on disk when it does.
    try:
        repaired = refetch_full_texts_pooled(
            store,
            targets,
            cfg.storage_state_path,
            headless=headless,
            tabs=opened,
            checkpoint=_checkpoint,
        )
    finally:
        save_store(store, cfg.items_path)
    typer.echo(
        f"{repaired}/{len(targets)} full texts recovered → {cfg.items_path}\n"
        f"{repaired} summaries invalidated: run `xbrain enrich` again."
    )


@app.command(name="verify-entities")
@_handle_cli_errors
def verify_entities_command(
    target: str = typer.Option("digest", help="digest | summary | topics"),
    verdicts: Path | None = typer.Option(
        None,
        "--verdicts",
        help="A verify-report.json to cross-reference (measures the judges' recall)",
    ),
) -> None:
    """Sweep every generated output for entities no evidence surface supports.

    Deterministic and token-free — no LLM, so it cannot inherit the judge ensemble's blind
    spot, and the whole corpus is swept rather than sampled. With `--verdicts` it reports
    how many flagged outputs the judges passed UNANIMOUSLY: that count is the ensemble's
    false-negative floor, the one number it cannot produce about itself.

    Read-only: writes `entity-report.{json,md}` and never touches the store.
    """
    cfg = _config()
    store = load_store(cfg.items_path)
    records = scan_store(store, target)
    ensemble = load_ensemble_verdicts(verdicts, target) if verdicts else {}
    summary = summarise_scan(records, ensemble)
    scanned = outputs_present(store, target)
    json_report, md_report = render_entity_report(records, summary, scanned, target)
    (cfg.data_dir / "entity-report.json").write_text(json_report, encoding="utf-8")
    (cfg.data_dir / "entity-report.md").write_text(md_report, encoding="utf-8")
    typer.echo(
        f"{summary['flagged']} outputs with unsupported entities "
        f"({summary['entities']} entities); "
        f"{summary['unanimous_pass_but_ungrounded']} of them with a UNANIMOUS PASS from the judges."
    )
    # The uncertain tier gets its own line rather than being folded into the headline:
    # it has lower precision, and merging it would let a reader quote one number for two
    # instruments. Printed unconditionally when non-empty, because a finding the check
    # made and the terminal never showed is the failure mode this exists to close.
    if summary["uncertain_only_flagged"]:
        typer.echo(
            f"+ {summary['uncertain_only_flagged']} outputs whose ONLY signal is in the "
            f"uncertain tier (ambiguous capital, typically at the start of a sentence): lower "
            f"precision, and it is where an invented name hides at the opening of a summary."
        )
    typer.echo(f"Report: {cfg.data_dir / 'entity-report.md'}")


# ============================================================================
# knowledge — the read contract (Plan 01)
# ============================================================================

knowledge_app = typer.Typer(help="Inspect the knowledge contract (read-only).")
app.add_typer(knowledge_app, name="knowledge")


def _knowledge_corpus():
    """The live corpus as the knowledge layer sees it. Read-only, no snapshot."""
    from xbrain.knowledge.evaluation import load_corpus_from_store

    cfg = _config()
    return cfg, load_corpus_from_store(
        cfg.items_path, load_vocab(cfg.data_dir / "vocab.yaml"), cfg.topics_path
    )


def _inspect_item(corpus, item_id: str, *, want_surfaces: bool, want_chunks: bool) -> dict:
    """The JSON document for one item: the read projection, its surfaces and its chunks.

    `--chunks` implies `--surfaces` because a chunk is only checkable against the surface it
    was cut from: `surface.text[char_start:char_end] == chunk.text` is the operational form
    of spec §3.8, and it cannot be evaluated by a consumer that was handed only the chunks.
    """
    from xbrain.knowledge.chunking import chunk_surfaces
    from xbrain.knowledge.surfaces import (
        article_block_texts,
        hydrate_verification,
        item_surfaces,
        item_topics,
    )

    cfg = _config()
    item = corpus.items.get(item_id)
    if item is None:
        raise ValueError(
            f"No item {item_id!r} in {cfg.items_path}. "
            "Check the id with `xbrain knowledge inspect <id>`."
        )
    from xbrain.knowledge.surfaces import knowledge_item

    surfaces = item_surfaces(item)
    from xbrain.knowledge.contracts import EVIDENCE_SCHEMA_VERSION

    payload: dict = {
        # The version of the shapes this payload dumps, read off the contract (U-1): the
        # surfaces and chunks here are the `EvidenceBundle`'s, so they carry its number.
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "item": knowledge_item(item, vault_dir=cfg.output_dir).model_dump(mode="json"),
        # Hydrated from the LIVE store, never persisted on a surface (M5): a stored copy
        # could not be invalidated when the verdict changed, so a revoked FAIL would keep
        # being served as the PASS it used to be.
        "verification": {
            target: verdict.model_dump(mode="json")
            for target, verdict in hydrate_verification(item, cfg.output_language).items()
        },
    }
    if want_surfaces or want_chunks:
        payload["surfaces"] = [s.model_dump(mode="json") for s in surfaces]
    if want_chunks:
        payload["chunks"] = [
            c.model_dump(mode="json")
            for c in chunk_surfaces(
                surfaces,
                topics=item_topics(item),
                url=item.url,
                blocks_by_surface_id=article_block_texts(item),
            )
        ]
    return payload


def _topic_membership(corpus, slug: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`(primary_item_ids, secondary_item_ids)` for a topic, both sorted.

    Sorted because `TopicRecord` is compared and fingerprinted downstream, and an order that
    followed dict iteration would make two runs over the same store differ. An item that is
    PRIMARY is excluded from the secondary list rather than appearing in both — the two lists
    answer different questions, and double-counting would inflate any membership figure taken
    from them.
    """
    primary = tuple(
        sorted(
            item.id
            for item in corpus.items.values()
            if item.enriched and item.enriched.primary_topic == slug
        )
    )
    secondary = tuple(
        sorted(
            item.id
            for item in corpus.items.values()
            if item.enriched and slug in item.enriched.topics and item.id not in primary
        )
    )
    return primary, secondary


def _inspect_topic(corpus, slug: str, *, want_surfaces: bool) -> dict:
    from xbrain.knowledge.surfaces import topic_record, topic_surfaces

    topic = next((t for t in corpus.vocab if t.slug == slug), None)
    if topic is None:
        raise ValueError(f"No topic {slug!r} in data/vocab.yaml.")
    page = corpus.topic_pages.get(slug)
    primary, secondary = _topic_membership(corpus, slug)
    from xbrain.knowledge.contracts import EVIDENCE_SCHEMA_VERSION

    payload: dict = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "topic": topic_record(topic, page, primary, secondary).model_dump(mode="json"),
    }
    if want_surfaces:
        payload["surfaces"] = [s.model_dump(mode="json") for s in topic_surfaces(topic, page)]
    return payload


@knowledge_app.command("inspect")
@_handle_cli_errors
def knowledge_inspect(
    item_id: str | None = typer.Argument(None, help="Id of the item to inspect."),
    topic: str | None = typer.Option(None, "--topic", help="Inspect a topic instead."),
    surfaces: bool = typer.Option(False, "--surfaces", help="Include the emitted surfaces."),
    chunks: bool = typer.Option(False, "--chunks", help="Include the chunks (implies --surfaces)."),
    json_out: bool = typer.Option(False, "--json", help="Stable JSON document on stdout."),
) -> None:
    """Show the unified corpus of an item or a topic: surfaces, provenance and chunks.

    READ-ONLY: it does not write to the store, takes no snapshot and calls no model. It is
    the way to check by hand what the knowledge layer offers — and, per rule 7 of
    CLAUDE.md, showing the evidence next to the claim is the cheapest verification layer
    there is: the real author of a quoted post is visible at a glance.
    """
    if (item_id is None) == (topic is None):
        raise ValueError("Pass an item id OR `--topic <slug>`, not both and not neither.")
    _cfg, corpus = _knowledge_corpus()
    payload = (
        _inspect_topic(corpus, topic, want_surfaces=surfaces or chunks)
        if topic is not None
        else _inspect_item(corpus, item_id or "", want_surfaces=surfaces, want_chunks=chunks)
    )
    if json_out:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(_render_inspect(payload))


def _render_inspect(payload: dict) -> str:
    """The human rendering. Same model as `--json` (spec §7.6), never a second shape."""
    lines: list[str] = []
    if "item" in payload:
        item = payload["item"]
        lines += [
            f"{item['item_id']}  @{item['author']['handle']} ({item['author']['name']})",
            f"  {item['url']}",
            f"  created {item['created_at'][:10]} · source {item['source']}"
            f" · topics {', '.join(item['topics']) or '—'}",
            f"  surfaces: {', '.join(item['available_surfaces']) or '—'}",
        ]
        for failure in item["failed_sources"]:
            lines.append(
                f"  ⚠ fetch failed: {failure['kind']} {failure['url']} ({failure['failure_reason']})"
            )
        for link in item["unfetched_links"]:
            lines.append(f"  ⚠ no body: {link['url']} ({link['reason']})")
    else:
        topic = payload["topic"]
        lines += [
            f"topic:{topic['slug']} — {topic['description']['text']}",
            f"  primary {len(topic['primary_item_ids'])}"
            f" · secondary {len(topic['secondary_item_ids'])}"
            f" · {'STALE' if topic['stale'] else 'up to date'}",
        ]
    for surface in payload.get("surfaces", []):
        excerpt = surface["text"][:120].replace("\n", " ")
        lines.append(
            f"  [{surface['surface_type']}] origin={surface['origin']}"
            f" trust={surface['trust_class']}  {excerpt}"
        )
    return "\n".join(lines)


# ============================================================================
# El índice persistente, `search` y `get` (Plan 02 §6)
#
# ESTA CAPA ES UN ADAPTADOR Y NADA MÁS. Cada comando carga las entradas una vez,
# llama a su servicio una vez, y elige UNA de dos salidas sobre el MISMO objeto:
# el documento JSON que el modelo serializa, o la vista humana que `render.py`
# compone desde ese mismo modelo (spec §7.6). Aquí no se formatea una línea: un
# formateador local sería una tercera definición de qué es un resultado, después
# del servicio y del JSON, que es justo la divergencia de la regla 5.
# ============================================================================

index_app = typer.Typer(help="Build and query the persistent index (data/index/).")
app.add_typer(index_app, name="index")


def _handle_index_errors(func: Callable) -> Callable:
    """Convertir los errores accionables del índice en un mensaje limpio + exit 1.

    `IndexError_` hereda de `Exception`, NO de `ValueError`, así que `_handle_cli_errors`
    —que enumera `ValueError`, `KeyError`, `RuntimeError`, `OSError`…— no lo ve: sin esta
    capa, «no hay índice» se imprime como un traceback crudo y el spec §9.3 pide lo
    contrario. Se apila DEBAJO de `_handle_cli_errors`, de modo que cada excepción la
    atiende exactamente uno de los dos y ninguno reimplementa al otro.

    El import es local porque `index_schema` arrastra `sqlite3` y los modelos del contrato,
    y `cli.py` se importa en cada invocación de `xbrain`, incluida `login`.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from xbrain.knowledge.index_schema import IndexError_

        try:
            return func(*args, **kwargs)
        except IndexError_ as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    return wrapper


def _index_inputs(cfg: Config):
    """Las tres entradas del índice, con la señal barata del snapshot que se leyó.

    UN SOLO CARGADOR PARA LOS CINCO COMANDOS. `load_index_inputs` ata las filas y la señal
    al mismo instante y a los mismos descriptores (P1b): dos cargadores distintos dejarían
    que las filas describan un momento y el `stat` otro, y esa diferencia es invisible.
    """
    from xbrain.knowledge.index_build import load_index_inputs

    return load_index_inputs(cfg.items_path, cfg.data_dir / "vocab.yaml", cfg.topics_path)


def _index_options(cfg: Config):
    """Lo que un build necesita y no es el corpus. Idéntico en build, update y status.

    Idéntico a propósito: `item_fingerprint` las consume, así que tres comandos con opciones
    distintas producirían tres huellas distintas del mismo item y `status` declararía
    cambios que no existen.
    """
    from xbrain.knowledge.index_build import IndexOptions

    return IndexOptions(vault_dir=cfg.output_dir)


def _query_context(cfg: Config, inputs):
    """Todo lo que una consulta necesita y no es la consulta (spec §7.2).

    El STORE viaja dentro: `get` lee el store vivo (spec §3.7.7) y `search` hidrata la
    verificación desde él (M5), así que las rutas que van aquí son las mismas que el
    cargador acaba de leer.
    """
    from xbrain.knowledge.search_service import QueryContext

    return QueryContext(
        store=inputs.store,
        vocab=inputs.vocab,
        topic_pages=inputs.topic_pages,
        index_dir=cfg.index_dir,
        items_path=cfg.items_path,
        vocab_path=cfg.data_dir / "vocab.yaml",
        topics_path=cfg.topics_path,
        vault_dir=cfg.output_dir,
        language=cfg.output_language,
        max_matches_per_item=cfg.index_max_matches_per_item,
    )


def _echo_json(payload: object) -> None:
    """El documento estable en stdout, y nada más (spec §3.7.9)."""
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@index_app.command("build")
@_handle_cli_errors
@_handle_index_errors
def index_build_command(
    force: bool = typer.Option(False, "--force", help="Rebuild an existing index from scratch."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count what it would do; touch no file."),
    json_out: bool = typer.Option(False, "--json", help="Stable JSON document on stdout."),
) -> None:
    """Build `data/index/` from scratch and seal it.

    IT IS NOT DESTRUCTIVE TO THE STORE, which is why it takes no snapshot: the only thing it
    writes is `data/index/`, which is derived and rebuildable (Plan 02 §6). `--force` does
    drop the previous index — manifest first, database second — so an interrupted rebuild
    never leaves a manifest standing over an empty database.
    """
    from dataclasses import asdict

    from xbrain.knowledge.index_build import build
    from xbrain.knowledge.render import render_build

    cfg = _config()
    report = build(
        cfg.index_dir, _index_inputs(cfg), options=_index_options(cfg), dry_run=dry_run, force=force
    )
    if json_out:
        _echo_json(asdict(report))
    else:
        typer.echo(render_build(report))


@index_app.command("update")
@_handle_cli_errors
@_handle_index_errors
def index_update_command(
    dry_run: bool = typer.Option(False, "--dry-run", help="Count the delta; write nothing."),
    json_out: bool = typer.Option(False, "--json", help="Stable JSON document on stdout."),
) -> None:
    """Bring the index up to date, touching only what changed (spec §5.6).

    One single transaction for the whole run: a half-done update would leave a partial
    application of a change nobody can name, and the index would look healthy because
    every id it holds still resolves.
    """
    from dataclasses import asdict

    from xbrain.knowledge.index_build import update
    from xbrain.knowledge.render import render_update

    cfg = _config()
    report = update(cfg.index_dir, _index_inputs(cfg), options=_index_options(cfg), dry_run=dry_run)
    if json_out:
        _echo_json(asdict(report))
    else:
        typer.echo(render_update(report))


@index_app.command("status")
@_handle_cli_errors
@_handle_index_errors
def index_status_command(
    json_out: bool = typer.Option(False, "--json", help="Stable JSON document on stdout."),
) -> None:
    """What the index contains and how far behind the store it has fallen.

    IT IS THE COMMAND THAT ANSWERS WHEN THE OTHERS REFUSE. `search`, `get`, `build` and
    `update` reject a missing or incompatible index; this one REPORTS it, because it is the
    instrument you run precisely to find out. Two instruments giving opposite answers about
    the same state is rule 9, and the way out is not for the diagnosis to refuse too: it is
    for it to name the same command the doors name.

    The manifest is published with `to_dict()`, the SAME document `data/index/manifest.json`
    contains, never a parallel dump of the dataclass (which would not even be serialisable:
    it carries a `datetime` and a nested `StoreSignal`).
    """
    from xbrain.knowledge.index_build import status
    from xbrain.knowledge.render import render_status

    cfg = _config()
    report = status(cfg.index_dir, _index_inputs(cfg), options=_index_options(cfg))
    if json_out:
        _echo_json(
            {
                "manifest": report.manifest.to_dict() if report.manifest else None,
                "counts": report.counts,
                "items_added": report.items_added,
                "items_changed": report.items_changed,
                "items_removed": report.items_removed,
                "topics_changed": report.topics_changed,
                "behind": report.behind,
                "incomplete": report.incomplete,
                "advice": report.advice,
            }
        )
    else:
        typer.echo(render_status(report))


def _search_filters(
    created_from: str | None,
    created_to: str | None,
    source: str | None,
    mine: bool,
    author: str | None,
    topics: list[str],
    kinds: list[str],
    origins: list[str],
    has_surfaces: list[str],
):
    """Los ocho filtros del spec §7.2, armados y validados por el contrato.

    `--mine` es el atajo del spec §7.2 para `source=own_tweet`, y es INCOMPATIBLE con
    `--source`: dos maneras de fijar un campo son una manera de fijarlo a dos valores
    distintos, y la resolución silenciosa (gane cuál gane) devolvería un corpus que el
    operador no pidió sin decírselo.

    Los valores de `--kind`, `--origin` y `--has-surface` NO se re-enumeran aquí: los valida
    `SearchFilters`, que es donde viven los `Literal` del contrato. Una lista escrita a mano
    en el CLI sería una segunda copia que envejece el día que el contrato crece (regla 5).
    """
    from xbrain.knowledge.contracts import SearchFilters

    if mine and source is not None:
        raise ValueError("`--mine` already sets `--source own_tweet`; do not combine them.")
    # `model_validate`, no el constructor, y la diferencia es de TIPADO, no de estilo.
    # `--kind`, `--origin`, `--has-surface` y `--source` llegan como texto libre del
    # usuario y los campos que los reciben son `Literal`s del contrato: pasarlos al
    # constructor obliga a un `cast` en el borde, que es decirle al comprobador que el
    # texto ya está validado justo donde todavía no lo está. `model_validate` valida de
    # verdad, en el sitio donde viven los valores válidos, y su mensaje los enumera
    # («Input should be 'external_article', 'x_article', …»). Una lista escrita a mano
    # aquí sería una segunda copia del contrato que envejece sola (regla 5).
    return SearchFilters.model_validate(
        {
            "created_from": _parse_date(created_from),
            "created_to": _parse_date(created_to, end_of_day=True),
            "source": "own_tweet" if mine else source,
            "author": author,
            "topics": tuple(topics),
            "content_kinds": tuple(kinds),
            "origins": tuple(origins),
            "has_surfaces": tuple(has_surfaces),
        }
    )


@app.command("search")
@_handle_cli_errors
@_handle_index_errors
def search_command(
    query: str = typer.Argument(..., help="What to search for. Free text."),
    limit: int = typer.Option(10, "--limit", help="Results per page."),
    created_from: str | None = typer.Option(None, "--from", help="Items created from (ISO)."),
    created_to: str | None = typer.Option(None, "--to", help="Items created until (ISO)."),
    source: str | None = typer.Option(None, "--source", help="bookmark | own_tweet."),
    mine: bool = typer.Option(False, "--mine", help="Shortcut for `--source own_tweet`."),
    author: str | None = typer.Option(None, "--author", help="Author handle."),
    topic: list[str] = typer.Option([], "--topic", help="Vocabulary slug (repeatable)."),
    kind: list[str] = typer.Option([], "--kind", help="Content kind (repeatable)."),
    origin: list[str] = typer.Option([], "--origin", help="Text provenance (repeatable)."),
    has_surface: list[str] = typer.Option(
        [], "--has-surface", help="Only items with this surface (repeatable)."
    ),
    strategy: str = typer.Option("lexical", "--strategy", help="Retrieval strategy."),
    cursor: str | None = typer.Option(None, "--cursor", help="Continue a truncated page."),
    json_out: bool = typer.Option(False, "--json", help="Stable JSON document on stdout."),
) -> None:
    """Search the index and return items with their citable fragments.

    READ-ONLY: the database is opened `mode=ro`, the store is not touched and there is no
    network and no model — spec §13.12 requires `search` to work without a single LLM call.

    Without `--json` it prints the human view of spec §7.6 (item, author, date, URL, surface,
    provenance, excerpt, match channels, warnings, and the `xbrain get` that fetches the
    source); with `--json`, the SAME `SearchResponse` as a document.
    """
    from typing import cast

    from xbrain.knowledge.contracts import Strategy
    from xbrain.knowledge.render import render_search
    from xbrain.knowledge.search_service import search

    cfg = _config()
    inputs = _index_inputs(cfg)
    response = search(
        query,
        _query_context(cfg, inputs),
        filters=_search_filters(
            created_from, created_to, source, mine, author, topic, kind, origin, has_surface
        ),
        limit=limit,
        # El `cast` es honesto porque `search` valida este texto en su primera línea:
        # `resolve_strategy` levanta un `ValueError` que enumera las estrategias
        # declaradas y las implementadas. La firma es `Strategy` pero el contrato acepta
        # y comprueba texto libre a propósito — un typo no es una degradación, y
        # contestarlo con resultados léxicos lo convertiría en una medición.
        strategy=cast(Strategy, strategy),
        cursor=cursor,
    )
    if json_out:
        _echo_json(response.model_dump(mode="json"))
    else:
        typer.echo(render_search(response))


@app.command("get")
@_handle_cli_errors
@_handle_index_errors
def get_command(
    item_id: str = typer.Argument(..., help="Item id."),
    surface: list[str] = typer.Option(
        [], "--surface", help="Surface to deliver whole (repeatable)."
    ),
    query: str | None = typer.Option(
        None, "--query", help="Prioritise the fragments that score for this query."
    ),
    cursor: str | None = typer.Option(None, "--cursor", help="Continue a truncated response."),
    json_out: bool = typer.Option(False, "--json", help="Stable JSON document on stdout."),
) -> None:
    """Deliver an item's evidence, read from the STORE, never from the index.

    This is invariant 7 of spec §3.7: `get` works with `data/index/` deleted, because an
    index able to answer `get` would be a copy of the corpus that nothing invalidates, and
    the day the two disagreed there would be no way to know which one the reader was shown.

    The per-response budget comes from `[index].get_char_budget`; above it the response is
    truncated, SAYING SO, and hands back a cursor (spec §9.3), never silently.
    """
    from typing import Sequence, cast

    from xbrain.knowledge.get_service import GetLimits, get
    from xbrain.knowledge.models import SurfaceType
    from xbrain.knowledge.render import render_get

    cfg = _config()
    inputs = _index_inputs(cfg)
    bundle = get(
        item_id,
        _query_context(cfg, inputs),
        # Mismo trato que `--strategy`: `get` comprueba cada nombre pedido contra las
        # superficies que el item emite y las de las fuentes que fallaron, y rechaza las
        # que no son ninguna de las dos enumerando las disponibles (`UnknownSurfaceError`).
        surfaces=cast("Sequence[SurfaceType] | None", surface or None),
        query=query,
        limits=GetLimits(char_budget=cfg.index_get_char_budget),
        cursor=cursor,
    )
    if json_out:
        _echo_json(bundle.model_dump(mode="json"))
    else:
        typer.echo(render_get(bundle, surfaces=surface, query=query))


def _run_sweep(
    cfg,
    cases,
    corpus,
    axes,
    strategy: str,
    report,
    *,
    json_out: bool,
    limit: int,
    ks: list[int],
    min_recall: float | None,
) -> None:
    """`eval --sweep-chunker`: score every `(target, overlap)` and publish the table.

    A SEPARATE PATH, not a flag threaded through `evaluate`, because the two answer different
    questions: `eval` measures the retriever at the parameters in force, the sweep measures
    the parameters. Folding them would make the ordinary report's numbers depend on whether a
    sweep flag happened to be present.

    The sweep changes `ChunkerParams` as an ARGUMENT and never the module constant, so
    `tests/fixtures/knowledge_ranking.json` — which passes its own pinned parameters — cannot
    be moved by it (M7).

    EVERY OPTION THE COMMAND ACCEPTS EITHER REACHES THIS PATH OR IS REFUSED BY NAME — the class
    of defect the final gate on #177 found three of here, one per option. A separate path that
    silently drops the flags of the command it shares is the worst of both: the user reads the
    command's help, the sweep obeys its own defaults, and the two never meet. `--limit` was
    fixed as an instance (`--limit 10` and `--limit 150` were byte-identical) without closing
    the class, so `--k` had it too, and `--min-recall` exited 0 having judged nothing.
    """
    from xbrain.knowledge.evaluation import (
        DEFAULT_SWEEP_K,
        parse_sweep,
        render_sweep_markdown,
        sweep_chunker as run_sweep,
    )

    # REFUSED, NOT IGNORED, and before anything is computed. The threshold judges the BUCKETS
    # of one evaluation — stratum by stratum, provenance by provenance — and a sweep publishes
    # a TABLE of chunker combinations, which has no bucket to compare it against. Applying it
    # to the winner's overall recall would answer a question nobody asked with a green a reader
    # would read as "every stratum clears the bar", and that misreading is worse than the
    # missing feature. Accepting it in silence was the fail-open CLAUDE.md already records for
    # this exact flag: «a threshold that reached no bucket is a FAILURE, not a pass».
    if min_recall is not None:
        raise ValueError(
            "`--min-recall` does not apply to `--sweep-chunker`: the threshold judges the buckets "
            "of ONE evaluation and the sweep publishes a TABLE of combinations, so there is no "
            "bucket to compare it against. Run `xbrain eval --min-recall …` without the "
            "sweep for the gate, and the sweep separately for the table."
        )
    # `--k` is repeatable because the ordinary report carries a column per k; the sweep RANKS,
    # and a ranking happens at one k. Choosing `max(ks)` would discard the rest in silence,
    # which is the very defect this path is being repaired for.
    if len(ks) > 1:
        raise ValueError(
            "`--sweep-chunker` ranks by a single `recall@k` and received "
            f"{len(ks)} `--k` values ({', '.join(str(value) for value in ks)}). "
            "Picking one would silently discard the rest: repeat the sweep with one `--k` "
            "per run."
        )

    grid = parse_sweep(axes)
    # The depth the command advertises is the depth the sweep runs at: the first version
    # dropped it here, and `--limit 10` and `--limit 150` were byte-identical.
    result = run_sweep(
        cases,
        corpus,
        grid,
        strategy=strategy,
        k=ks[0] if ks else DEFAULT_SWEEP_K,
        limit=limit,
    )
    json_path = report or (cfg.data_dir / "eval-sweep.json")
    if not json_path.is_absolute():
        json_path = _repo_root() / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    json_path.with_suffix(".md").write_text(render_sweep_markdown(result), encoding="utf-8")
    if json_out:
        _echo_json(payload)
    else:
        typer.echo(render_sweep_markdown(result))
        typer.echo(f"Report: {json_path} · {json_path.with_suffix('.md')}")
    # PUBLISHED FIRST, THEN FAILED. A sweep with no winner — nothing scorable, or no
    # combination at all — is not a success, and it used to exit 0 while announcing «PLANO:
    # todas las combinaciones puntúan igual», a positive claim about a ranking that never
    # happened. The exit code is the surface a caller reads (rule 9); the table is still
    # written and echoed, because the run failing is not a reason to withhold its evidence.
    # The message is the report's OWN verdict, so stderr and the artefact say one thing.
    if result.winner is None:
        raise ValueError(payload["verdict"])


@app.command("eval")
@_handle_cli_errors
def eval_command(
    strategy: str = typer.Option("lexical", "--strategy", help="Strategy to evaluate."),
    limit: int = typer.Option(
        10,
        "--limit",
        help="Retrieval depth per case, in OWNERS (never below the largest k).",
    ),
    k: list[int] = typer.Option(
        [], "--k", help="k values to report (repeatable; the sweep accepts only one)."
    ),
    min_recall: float | None = typer.Option(
        None,
        "--min-recall",
        help=(
            "Threshold: if any bucket falls below it, the command fails. Without it, it only "
            "reports. Does not combine with `--sweep-chunker`."
        ),
    ),
    golden_set: Path = typer.Option(
        Path("eval/golden-set.yaml"), "--golden-set", help="Path to the golden set."
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Where to write the report (default data/eval-report.json)."
    ),
    sweep_chunker: list[str] = typer.Option(
        [],
        "--sweep-chunker",
        help="Chunking sweep: `target=800,1200 overlap=0,150` (repeatable or quoted).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Stable JSON document on stdout."),
) -> None:
    """Evaluate retrieval against the golden set and publish the report.

    REPORT ONLY: it does not write to `items.json` or take a snapshot, like `verify` by default.

    Without `--min-recall` it reports and does not judge: spec §8.6 sets the merge thresholds
    AFTER running the baseline, so a default number here would be exactly the metric that
    cannot come out any other way (rule 2 of CLAUDE.md). With a threshold, the command is a
    gate and names the bucket that failed.
    """
    from xbrain.knowledge.evaluation import DEFAULT_KS, evaluate, render_markdown
    from xbrain.knowledge.goldenset import load_cases, load_scenarios, resolve_cases

    cfg, corpus = _knowledge_corpus()
    path = golden_set if golden_set.is_absolute() else _repo_root() / golden_set
    cases = resolve_cases(load_cases(path), corpus.items)
    if sweep_chunker:
        _run_sweep(
            cfg,
            cases,
            corpus,
            sweep_chunker,
            strategy,
            report,
            json_out=json_out,
            limit=limit,
            ks=k,
            min_recall=min_recall,
        )
        return
    result = evaluate(
        cases,
        corpus,
        strategy=strategy,
        ks=tuple(k) if k else DEFAULT_KS,
        threshold=min_recall,
        scenarios=load_scenarios(path),
        limit=limit,
    )
    payload = result.to_dict()
    json_path = report or (cfg.data_dir / "eval-report.json")
    if not json_path.is_absolute():
        json_path = _repo_root() / json_path
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    json_path.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")

    if json_out:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(render_markdown(result))
        typer.echo(f"Report: {json_path} · {json_path.with_suffix('.md')}")
    if not result.passed:
        # A gate that reports its own failure on stdout and exits 0 is the `gh pr checks`
        # trap of CLAUDE.md rule 9, reproduced locally. The non-zero exit is the signal.
        raise ValueError(
            "The evaluation does not reach the threshold:\n  " + "\n  ".join(result.failures)
        )


if __name__ == "__main__":
    app()
