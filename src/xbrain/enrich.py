"""Enrichment phase — two tracks (API executor, worksheet) feed one validator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import get_args

from xbrain.executors.base import EnrichmentExecutor
from xbrain.models import Enrichment, ExecutorName, Item, Topic
from xbrain.validate import validate_judgment


def items_pending_enrichment(
    store: dict[str, Item],
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Item]:
    """Return items with no enrichment yet, optionally within a date range."""
    pending: list[Item] = []
    for item in store.values():
        if item.enriched is not None:
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
    item = store.get(item_id)
    if item is None:
        return [f"unknown item id: {item_id}"]
    apply_enrichment(
        item,
        Enrichment(
            enriched_at=datetime.now(timezone.utc),
            executor=executor_name,
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
