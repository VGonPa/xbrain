"""Enrichment phase — DESIGNED BUT IN PAUSE (see spec §9).

The schema slot `Item.enriched` and the helpers here are ready. The actual
LLM executor (`api`) is intentionally not implemented until authorised.
"""
from __future__ import annotations

from xkb.models import Enrichment, Item


def items_pending_enrichment(store: dict[str, Item]) -> list[Item]:
    """Return items that have no enrichment yet."""
    return [item for item in store.values() if item.enriched is None]


def apply_enrichment(item: Item, enrichment: Enrichment) -> None:
    """Attach an enrichment result to an item (used by every executor)."""
    item.enriched = enrichment


def enrich(store: dict[str, Item], executor: str) -> list[Item]:
    """Run the enrichment phase. v1 supports inspection only.

    `executor="manual"` returns the items pending enrichment so they can be
    filled in by hand or by Claude Code. `executor="api"` / `"claude-code"`
    are intentionally unimplemented — the LLM executor is in pause (spec §9).
    """
    pending = items_pending_enrichment(store)
    if executor == "manual":
        return pending
    if executor in ("api", "claude-code"):
        raise NotImplementedError(
            f"El ejecutor '{executor}' no está habilitado. "
            "El enriquecimiento con LLM está en pausa (ver spec §9)."
        )
    raise ValueError(f"Ejecutor desconocido: {executor!r}")
