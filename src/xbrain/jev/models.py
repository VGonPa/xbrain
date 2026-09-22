"""What one Jev assessment of one item looks like on disk.

Provider-agnostic on purpose: a second judge answering the same questions (another model,
a panel) produces the same record with another `provider`/`model`, so they can be compared.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class PrimaryChoice(BaseModel):
    """The Choice answer for the primary topic: the winner, its confidence, and the full
    distribution over every option (the vocabulary slugs plus the fallback)."""

    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class TopicAssessment(BaseModel):
    """One item, one call. `membership` holds a Noul (probability) per vocabulary slug."""

    item_id: str
    provider: str = "typesafe"
    model: str
    asked_at: datetime
    # `verification.fingerprint_output(item, "topics")` at ask time — which assignment existed.
    output_fingerprint: str | None = None
    # `assess.topic_contract(...)`: binds the record to the state + vocabulary it was asked on.
    contract: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_chars: int = Field(ge=0)
    truncated: bool = False
    membership: dict[str, float]
    primary: PrimaryChoice
    input_tokens: int | None = None
    output_tokens: int | None = None

    @field_validator("asked_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("asked_at must be UTC-aware")
        return value
