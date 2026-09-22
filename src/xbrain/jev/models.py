"""What one Jev assessment of one item looks like on disk.

Provider-agnostic on purpose: a second judge answering the same questions (another model,
a panel) produces the same record with another `provider`/`model`, so they can be compared.
The record is FROZEN and refuses unknown fields, like every other persisted envelope in the
repo (`knowledge/models.py`): a side-car file that a later run rewrites wholesale must not
lose a field a newer writer added, and a record mutated after its `contract` was computed
would carry a contract describing something else.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from xbrain.models import _require_utc_aware

_FROZEN = ConfigDict(frozen=True, extra="forbid")
_SHA256 = r"^[0-9a-f]{64}$"

#: Every float in this record is a probability the provider returned, so all of them carry
#: the same bound. An out-of-range value means the record is not what the provider answered;
#: NaN and infinity are refused too, and a NaN would otherwise compare False against every
#: threshold forever — silently absent from every report.
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
#: An identifier that must actually identify something.
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PrimaryChoice(BaseModel):
    """The Choice answer for the primary topic: the winner, its confidence, and the
    probabilities the provider returned.

    Jev normally returns one probability per option (the vocabulary slugs plus the
    fallback), but full coverage is NOT enforced — read a option's probability with
    `.get(option, 0.0)`. The winner's own entry IS required: without it a reader either
    raises `KeyError` or silently scores the winner at zero.
    """

    model_config = _FROZEN

    choice: NonEmpty
    confidence: Probability
    probabilities: dict[str, Probability]

    @model_validator(mode="after")
    def _choice_is_in_the_distribution(self) -> PrimaryChoice:
        if self.choice not in self.probabilities:
            raise ValueError(
                f"primary.probabilities has no entry for the chosen option {self.choice!r}"
            )
        return self


class TopicAssessment(BaseModel):
    """One item, one call. `membership` holds a Noul (probability) per vocabulary slug."""

    model_config = _FROZEN

    item_id: NonEmpty
    # Which judge answered. No default: provenance is what the judge REPORTED (it travels on
    # `JevResult`), never a value the record assumes about itself.
    provider: NonEmpty
    model: NonEmpty
    asked_at: datetime
    # `verification.fingerprint_output(item, "topics")` at ask time — which assignment
    # existed. Pattern-constrained like its twin `VerificationVerdict.output_fingerprint`:
    # this is the staleness key, and a malformed stamp can never equal a fresh digest, so
    # the "this assessment is stale" signal would silently never fire.
    output_fingerprint: str | None = Field(default=None, pattern=_SHA256)
    # `assess.topic_contract(...)` = sha256(version ∥ state ∥ questions_digest), where the
    # digest is `assess.questions_digest(questions)`: the canonical JSON of every question
    # asked — each one's TYPE, instructions and criteria. A stored assessment stays valid
    # while the state and that digest are unchanged: re-enriching the item does NOT
    # invalidate it, while rewording a question, changing a topic description or the
    # fallback option, adding or removing a topic, or new evidence text does.
    contract: str = Field(pattern=_SHA256)
    # Length of the evidence BEFORE the cut, so a reader can tell how much was dropped;
    # `truncated` says whether the cut happened. Task 2 computes both.
    state_chars: int = Field(ge=0)
    truncated: bool = False
    membership: dict[str, Probability] = Field(min_length=1)
    primary: PrimaryChoice
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @field_validator("asked_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        """Naive is refused by the shared helper; a non-UTC offset is refused here.

        We do NOT coerce — `xbrain.models._require_utc_aware` says why in as many words:
        "that would mask the bug". Storing every instant as UTC is also what keeps two
        records written on machines in different zones comparable by their string form.
        """
        _require_utc_aware("asked_at", value)
        if value.utcoffset() != timedelta(0):
            raise ValueError(
                f"asked_at must be UTC (offset +00:00), got {value.utcoffset()} in {value!r}"
            )
        return value
