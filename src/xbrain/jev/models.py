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
    SerializerFunctionWrapHandler,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_serializer,
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


#: Which command made a pass: `xbrain jev topics` or `xbrain jev ask`. They share `runs.jsonl`,
#: and every reader filters by it — a topics cost view never bills a query, and vice versa.
RunKind = Literal["topics", "ask"]

#: A pass's token count for one provider: named, and never negative.
ProviderTokens = dict[NonEmpty, Annotated[int, Field(ge=0)]]


class JevRun(BaseModel):
    """One pass that SENT at least one request to Jev — a line of `runs.jsonl`.

    The side-car keeps only the latest assessment per item, so it cannot say what has been
    billed over time; this record can. It stores TOKENS, never dollars: the cost is computed
    at read time (`report.run_history`), so a price correction reprices the whole history.

    `kind` says which command made the pass (`RunKind`); a line written before the field
    existed reads as `topics`. An `ask` pass also names its query by `query_sha` (the file
    name of `data/jev/asks/<query_sha>.json`), so what one query has cost is a sum over the
    log; a topics pass has none.

    Every count is read at the CLIENT SEAM (`client.CountingJevClient`), because a call is
    billed the moment it is answered, whatever xbrain then does with the answer:

    * `requests` — calls SENT, each item once. The SDK retries internally and those retries
      are invisible here, so they are not counted.
    * `ok` — answers kept (banked into the side-car).
    * `failed` — calls that raised, plus answers xbrain refused (a malformed answer set).
    * `unsaved` — answers that came back and never reached the disk: on an interrupted pass,
      those not drained into the side-car before Ctrl-C landed (a refused answer that was not
      yet drained lands here too — it was not kept either way); on any pass, those whose
      final save failed (a full disk). Billed, not kept, not failed.
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

    kind: RunKind = "topics"
    query_sha: str | None = Field(default=None, pattern=_SHA256)
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

    @model_serializer(mode="wrap")
    def _no_null_query(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        """A topics line has NO `query_sha` key, rather than `null`: this model refuses unknown
        keys, so a copy of xbrain from before `jev ask` would otherwise refuse the whole log."""
        data: dict[str, object] = handler(self)
        if data.get("query_sha") is None:
            data.pop("query_sha", None)
        return data

    @model_validator(mode="after")
    def _consistent(self) -> JevRun:
        if (self.kind == "ask") != (self.query_sha is not None):
            raise ValueError("query_sha is required on an ask pass and absent on a topics pass")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at is earlier than started_at")
        if self.models != sorted(set(self.models)):
            raise ValueError(f"models must be sorted and distinct, got {self.models!r}")
        came_back = self.ok + self.failed + self.unsaved
        if came_back > self.requests:
            raise ValueError(
                f"ok + failed + unsaved ({came_back}) exceeds requests ({self.requests})"
            )
        if not self.interrupted and came_back != self.requests:
            raise ValueError(
                f"a pass that was not interrupted accounts for every request: "
                f"ok + failed + unsaved = {came_back}, requests = {self.requests}"
            )
        if sum(self.input_tokens_by_provider.values()) != self.input_tokens:
            raise ValueError(
                f"input_tokens ({self.input_tokens}) is not the sum of "
                f"input_tokens_by_provider {self.input_tokens_by_provider!r}"
            )
        return self


class AskAssessment(BaseModel):
    """One post, one call, one query: the probability that the post answers the request.

    Kept in `data/jev/asks/<query_sha>.json` (`store.save_asks`), one file per query, keyed by
    post id. `contract` = `ask.ask_contract(state, questions_digest)`: the same state a topics
    pass sends and the ask's one question, under its own version string — so repeating a query
    never re-pays a post whose evidence is unchanged, and new evidence re-asks it.
    `prompt_chars` is what was sent (state as cut + the question's text): with `input_tokens`
    it is how the next estimate measures characters per token (`ask.chars_per_token`).
    """

    model_config = _FROZEN

    item_id: NonEmpty
    provider: NonEmpty
    model: NonEmpty
    asked_at: datetime
    contract: str = Field(pattern=_SHA256)
    state_chars: int = Field(ge=0)
    truncated: bool = False
    prompt_chars: int = Field(ge=0)
    probability: Probability
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    @field_validator("asked_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _require_utc("asked_at", value)


class AskFile(BaseModel):
    """`data/jev/asks/<query_sha>.json`: the query it answers, and its records by post id.

    The query is IN the file so the file says what it is without the history, and so a file
    renamed or copied under another query's sha is refused (`store.load_asks`) instead of
    passing its answers off as another question's.
    """

    model_config = _FROZEN

    query: NonEmpty
    assessments: dict[str, AskAssessment]

    @model_validator(mode="after")
    def _keyed_by_post(self) -> AskFile:
        for key, record in self.assessments.items():
            if key != record.item_id:
                raise ValueError(f"record for {record.item_id!r} filed under {key!r}")
        return self


class AskHistoryEntry(BaseModel):
    """One query in `data/jev/asks/index.json`: what was asked, when, and its LAST USE.

    One entry per query, refreshed each time it is asked: `first_asked_at`, `last_asked_at`
    and `times` span every use; the `last_*` fields are the LAST use only — the posts it had a
    current answer for, how many were results, at which threshold, under which filters. The
    full history of uses is the run log (`JevRun.query_sha`), which is also where its COST
    lives (`report.ask_cost`), priced when read like every other bill. `rebuilt` marks an entry
    reconstructed from the query's answer file because the index had lost it (a crash between
    the save and the history write, a deleted `index.json`): its `last_*` are then counted over
    the whole file at `[jev].threshold`, with no filters.
    """

    model_config = _FROZEN

    query_sha: str = Field(pattern=_SHA256)
    query: NonEmpty
    first_asked_at: datetime
    last_asked_at: datetime
    times: int = Field(ge=1)
    last_evaluated: int = Field(ge=0)
    last_results: int = Field(ge=0)
    last_threshold: Probability
    last_filters: dict[str, str | bool]
    rebuilt: bool = False

    @field_validator("first_asked_at", "last_asked_at")
    @classmethod
    def _utc(cls, value: datetime, info: ValidationInfo) -> datetime:
        return _require_utc(str(info.field_name), value)


class AskCalibration(BaseModel):
    """What paid ask answers say about the bill: the sufficient sums of a straight-line fit
    `tokens = per_call + chars / chars_per_token` over `(prompt_chars, input_tokens)`.

    Kept in `index.json` so an estimate never has to parse every answer file (one unreadable
    file of another query would otherwise block every estimate). `answers` counts the pairs;
    an answer with no usage is not one. `ask.cost_model` turns it into a model.
    """

    model_config = _FROZEN

    answers: int = Field(default=0, ge=0)
    chars: float = Field(default=0.0, ge=0)
    tokens: float = Field(default=0.0, ge=0)
    chars_sq: float = Field(default=0.0, ge=0)
    chars_tokens: float = Field(default=0.0, ge=0)

    def add(self, chars: int, tokens: int) -> AskCalibration:
        """This calibration with one more paid answer folded in."""
        return AskCalibration(
            answers=self.answers + 1,
            chars=self.chars + chars,
            tokens=self.tokens + tokens,
            chars_sq=self.chars_sq + chars * chars,
            chars_tokens=self.chars_tokens + chars * tokens,
        )


class AskIndex(BaseModel):
    """`data/jev/asks/index.json`: the history of queries by sha, and the cost calibration."""

    model_config = _FROZEN

    queries: dict[str, AskHistoryEntry] = Field(default_factory=dict)
    calibration: AskCalibration = Field(default_factory=AskCalibration)

    @model_validator(mode="after")
    def _keyed_by_sha(self) -> AskIndex:
        for key, entry in self.queries.items():
            if key != entry.query_sha:
                raise ValueError(f"entry for {entry.query_sha!r} filed under {key!r}")
        return self
