"""What one Jev assessment of one item looks like on disk.

Provider-agnostic on purpose: a second judge answering the same questions (another model,
a panel) produces the same record with another `provider`/`model`, which are recorded for
PROVENANCE — which judge said this, and under which version. Being able to hold two judges'
answers side by side is a different thing and is not what the side-car does today: it is keyed
by `item_id` alone, so a second judge overwrites the first (see `jev/store.py`).

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
    ValidationInfo,
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


def _require_utc(name: str, value: datetime) -> datetime:
    """Aware AND at offset zero.

    We do NOT coerce — `xbrain.models._require_utc_aware` says why in as many words: "that
    would mask the bug". Storing every instant as UTC is also what keeps two records written on
    machines in different zones comparable by their string form.
    """
    _require_utc_aware(name, value)
    if value.utcoffset() != timedelta(0):
        raise ValueError(
            f"{name} must be UTC (offset +00:00), got {value.utcoffset()} in {value!r}"
        )
    return value


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
    # `truncated` says whether the cut happened. `assess.build_topic_state` computes both.
    state_chars: int = Field(ge=0)
    truncated: bool = False
    membership: dict[str, Probability] = Field(min_length=1)
    primary: PrimaryChoice
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @field_validator("asked_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        """Naive or non-UTC is refused (see `_require_utc`)."""
        return _require_utc("asked_at", value)


class JevRun(BaseModel):
    """One `xbrain jev topics` pass that SENT at least one request — a line of `runs.jsonl`.

    The side-car keeps only the latest assessment per item, so it cannot say what has been
    billed over time; this record can. It stores TOKENS, never dollars: the cost is computed
    at read time (`report.run_history`), so a price correction reprices the whole history.

    `requests` counts the calls xbrain SENT, each item once. The SDK retries internally and
    those retries are not visible here, so they are not counted. `ok` and `failed` are the
    calls that came back; on an interrupted pass the difference is the calls still in flight
    when Ctrl-C landed, whose outcome nobody observed.

    `input_tokens_by_provider` rather than one `provider`: every answer carries the provider
    that gave it, and the price is per provider (`defaults.tokens_cost_usd`), so a single
    field would misprice a mixed pass. It is EMPTY when nothing answered — a pass where every
    call failed (a 402, a dead key) reports no provider at all, and is still history.
    `input_tokens` is its sum, stored so a plain reader of the file gets the total;
    `input_tokens_unknown` counts answers that reported no usage.
    """

    model_config = _FROZEN

    started_at: datetime
    finished_at: datetime
    models: list[NonEmpty]
    requests: int = Field(ge=0)
    ok: int = Field(ge=0)
    failed: int = Field(ge=0)
    input_tokens_by_provider: dict[str, int]
    input_tokens: int = Field(ge=0)
    input_tokens_unknown: int = Field(ge=0)
    interrupted: bool

    @field_validator("started_at", "finished_at")
    @classmethod
    def _utc(cls, value: datetime, info: ValidationInfo) -> datetime:
        return _require_utc(str(info.field_name), value)

    @model_validator(mode="after")
    def _consistent(self) -> JevRun:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is earlier than started_at")
        if self.models != sorted(set(self.models)):
            raise ValueError(f"models must be sorted and distinct, got {self.models!r}")
        answered = self.ok + self.failed
        if answered > self.requests:
            raise ValueError(f"ok + failed ({answered}) exceeds requests ({self.requests})")
        if not self.interrupted and answered != self.requests:
            raise ValueError(
                f"a pass that was not interrupted accounts for every request: "
                f"ok + failed = {answered}, requests = {self.requests}"
            )
        if any(tokens < 0 for tokens in self.input_tokens_by_provider.values()):
            raise ValueError("input_tokens_by_provider holds a negative count")
        if sum(self.input_tokens_by_provider.values()) != self.input_tokens:
            raise ValueError(
                f"input_tokens ({self.input_tokens}) is not the sum of "
                f"input_tokens_by_provider {self.input_tokens_by_provider!r}"
            )
        return self
