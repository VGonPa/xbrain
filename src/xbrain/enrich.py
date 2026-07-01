"""Enrichment phase — two tracks (API executor, worksheet) feed one validator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast, get_args

from xbrain.executors.base import EnrichmentExecutor
from xbrain.models import Enrichment, ExecutorName, Item, Topic
from xbrain.validate import validate_judgment


def _needs_reenrichment(item: Item) -> bool:
    """True when an already-enriched item's content was (re)fetched since (#44).

    A video bookmark is often enriched from its ~2-line tweet first, then gains
    an `x_video` transcript when `digest-video` runs — which bumps
    `content.fetched_at` to attach time. Keying re-enrichment on
    ``content.fetched_at > enriched.enriched_at`` means the richer transcript is
    NOT treated as already-processed: the item flows back through enrich and
    finally gets a real `primary_topic` instead of "—". The normal order
    (fetch → enrich) leaves `fetched_at` *before* `enriched_at`, so nothing
    re-enriches spuriously. A `fetch --force` refresh benefits from the same
    trigger. Both timestamps are UTC-aware by construction, so the comparison is
    well-defined.
    """
    return (
        item.enriched is not None
        and item.content is not None
        and item.content.fetched_at > item.enriched.enriched_at
    )


def items_pending_enrichment(
    store: dict[str, Item],
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Item]:
    """Return items needing enrichment, optionally within a date range.

    An item is pending when it has no enrichment yet, OR when its content was
    (re)fetched after its last enrichment (`_needs_reenrichment` — the #44
    re-enrichment trigger for a transcript attached post-enrich).
    """
    pending: list[Item] = []
    for item in store.values():
        if item.enriched is not None and not _needs_reenrichment(item):
            continue
        if since and item.created_at < since:
            continue
        if until and item.created_at > until:
            continue
        pending.append(item)
    return pending


def apply_enrichment(item: Item, enrichment: Enrichment) -> None:
    """Attach an enrichment result to an item."""
    item.enriched = enrichment


def _validate_and_attach(
    store: dict[str, Item],
    item_id: str,
    summary: str,
    primary_topic: str,
    topics: object,
    vocab_slugs: set[str],
    executor_name: str,
) -> list[str]:
    """Validate one judgment; attach it if valid. Return errors (empty = ok)."""
    errors = validate_judgment(
        {"summary": summary, "primary_topic": primary_topic, "topics": topics},
        vocab_slugs,
    )
    if errors:
        return errors
    # validate_judgment guarantees `topics` is a list once it returns no errors;
    # the isinstance guard makes that proof visible to the type checker.
    if not isinstance(topics, list):
        return ["topics must be a list"]
    item = store.get(item_id)
    if item is None:
        return [f"unknown item id: {item_id}"]
    apply_enrichment(
        item,
        Enrichment(
            enriched_at=datetime.now(timezone.utc),
            # Both callers supply a validated executor name: enrich_with_executor
            # passes the literal "api"; apply_worksheet_judgments checks the value
            # against get_args(ExecutorName) before calling.
            executor=cast(ExecutorName, executor_name),
            summary=summary,
            primary_topic=primary_topic,
            topics=list(topics),
        ),
    )
    return []


def enrich_with_executor(
    store: dict[str, Item],
    executor: EnrichmentExecutor,
    vocab: list[Topic],
    since: datetime | None = None,
    until: datetime | None = None,
) -> tuple[int, list[tuple[str, list[str]]]]:
    """Enrich pending items with an in-process executor (the `api` track).

    Returns `(enriched_count, invalid)` where `invalid` is `(item_id, errors)`.
    """
    pending = items_pending_enrichment(store, since, until)
    vocab_slugs = {t.slug for t in vocab}
    enriched = 0
    invalid: list[tuple[str, list[str]]] = []
    for j in executor.enrich_items(pending, vocab):
        errors = _validate_and_attach(
            store, j.item_id, j.summary, j.primary_topic, j.topics, vocab_slugs, "api"
        )
        if errors:
            invalid.append((j.item_id, errors))
        else:
            enriched += 1
    return enriched, invalid


def apply_worksheet_judgments(
    store: dict[str, Item],
    judgments: list[dict],
    vocab: list[Topic],
    executor_name: str = "claude-code",
) -> tuple[int, list[tuple[str, list[str]]]]:
    """Validate + attach judgments from a filled worksheet (the worksheet track)."""
    if executor_name not in get_args(ExecutorName):
        raise ValueError(f"worksheet has an invalid executor: {executor_name!r}")
    vocab_slugs = {t.slug for t in vocab}
    enriched = 0
    invalid: list[tuple[str, list[str]]] = []
    for j in judgments:
        item_id = str(j.get("item_id", ""))
        errors = _validate_and_attach(
            store,
            item_id,
            str(j.get("summary", "")),
            str(j.get("primary_topic", "")),
            j.get("topics"),
            vocab_slugs,
            executor_name,
        )
        if errors:
            invalid.append((item_id, errors))
        else:
            enriched += 1
    return enriched, invalid
