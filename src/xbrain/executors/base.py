"""The in-process executor interface — what `enrich` calls for LLM judgment.

An executor receives a batch of items plus the topic vocabulary and returns one
judgment per item. It produces *only* judgment; the validator handles structure.
The worksheet handoff (manual / claude-code) does not go through this Protocol —
it is a two-step export/import, see xbrain.worksheet.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from xbrain.models import Item, Topic


class EnrichmentJudgment(BaseModel):
    """One executor result for one item — judgment only, no identifiers.

    `topics` always carries at least the primary topic, so an empty list is an
    illegal state and fails construction.
    """
    item_id: str
    summary: str
    primary_topic: str
    topics: list[str] = Field(min_length=1)


class EnrichmentExecutor(Protocol):
    """Anything that can turn items into enrichment judgments in-process."""

    def enrich_items(
        self, items: list[Item], vocab: list[Topic]
    ) -> list[EnrichmentJudgment]:
        """Return one judgment per input item, in any order."""
        ...
