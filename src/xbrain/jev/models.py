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
from typing import Annotated, Literal

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


#: A pass's token count for one provider: named, and never negative.
ProviderTokens = dict[NonEmpty, Annotated[int, Field(ge=0)]]


class JevRun(BaseModel):
    """One pass that SENT at least one request to Jev — a line of `runs.jsonl`.

    The side-car keeps only the latest assessment per item, so it cannot say what has been
    billed over time; this record can. It stores TOKENS, never dollars: the cost is computed
    at read time (`report.run_history`), so a price correction reprices the whole history.

    `kind` says which command made the pass. Only `topics` exists today; the field exists now
    so a later kind of pass can share the file, and a line written before it existed reads
    as `topics`.

    Every count is read at the CLIENT SEAM (`client.CountingJevClient`), because a call is
    billed the moment it is answered, whatever xbrain then does with the answer:

    * `requests` — calls SENT, each item once. The SDK retries internally and those retries
      are invisible here, so they are not counted.
    * `ok` — answers kept (banked into the side-car).
    * `failed` — calls that raised, plus answers xbrain refused (a malformed answer set).
    * `unsaved` — only on an interrupted pass: answers that came back but were never
      drained into the side-car before Ctrl-C landed (a refused answer that was not yet
      drained lands here too — it was not kept either way). Billed, not kept, not failed.
    * `requests - ok - failed - unsaved` — on an interrupted pass, calls still in flight.
      A worker that dequeues its item AFTER Ctrl-C can still send a call the log never sees;
      SIGTERM or a kill logs nothing at all. Both show up in `report.run_history` as
      assessments outside the log.

    `input_tokens_by_provider` covers EVERY answer that came back — refused and unsaved ones
    included — keyed by the provider that gave it, because the price is per provider
    (`defaults.tokens_cost_usd`). It is EMPTY when nothing answered: a pass where every call
    failed (a 402, a dead key) reports no provider at all, and is still history.
    `input_tokens` is its sum; `input_tokens_unknown` counts answers with no usage.
    """

    model_config = _FROZEN

    kind: Literal["topics"] = "topics"
    started_at: datetime
    finished_at: datetime
    models: list[NonEmpty]
    requests: int = Field(ge=0)
    ok: int = Field(ge=0)
    failed: int = Field(ge=0)
    unsaved: int = Field(default=0, ge=0)
    input_tokens_by_provider: ProviderTokens
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
        came_back = self.ok + self.failed + self.unsaved
        if came_back > self.requests:
            raise ValueError(
                f"ok + failed + unsaved ({came_back}) exceeds requests ({self.requests})"
            )
        if not self.interrupted and self.unsaved:
            raise ValueError("unsaved answers exist only on an interrupted pass")
        if not self.interrupted and came_back != self.requests:
            raise ValueError(
                f"a pass that was not interrupted accounts for every request: "
                f"ok + failed = {self.ok + self.failed}, requests = {self.requests}"
            )
        if sum(self.input_tokens_by_provider.values()) != self.input_tokens:
            raise ValueError(
                f"input_tokens ({self.input_tokens}) is not the sum of "
                f"input_tokens_by_provider {self.input_tokens_by_provider!r}"
            )
        return self
