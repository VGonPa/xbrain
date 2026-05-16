"""Enrichment phase — DESIGNED BUT IN PAUSE (see spec §9).

The schema slot `Item.enriched` and the helpers here are ready. The actual
LLM executor (`api`) is intentionally not implemented until authorised.
"""
from __future__ import annotations

from datetime import datetime

from xkb.models import Enrichment, Item


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
    """Attach an enrichment result to an item (used by every executor)."""
    item.enriched = enrichment


def enrich(
    store: dict[str, Item],
    executor: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[Item]:
    """Run the enrichment phase. v1 supports inspection only.

    `executor="manual"` returns the items pending enrichment so they can be
    filled in by hand or by Claude Code. `executor="api"` / `"claude-code"`
    are intentionally unimplemented — the LLM executor is in pause (spec §9).
    """
    pending = items_pending_enrichment(store, since, until)
    if executor == "manual":
        return pending
    if executor in ("api", "claude-code"):
        raise NotImplementedError(
            f"El ejecutor '{executor}' no está habilitado. "
            "El enriquecimiento con LLM está en pausa (ver spec §9)."
        )
    raise ValueError(f"Ejecutor desconocido: {executor!r}")
