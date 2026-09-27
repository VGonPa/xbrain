"""`xbrain jev ask`: ask the corpus a question; Jev says, post by post, whether it answers it.

ONE NOUL PER POST (`questions.build_ask_questions`): the user's query verbatim as the true
criterion, over the SAME state a topics pass sends (`assess.build_topic_state`). Each post is
judged alone, so its probability does not depend on the company it was asked in. The results
are the posts whose CURRENT answer is at or above the threshold, best first.

THE FLOW, shared by `xbrain jev ask` and the server's ask job: `plan_ask` (load, filter,
select, estimate — nothing paid, no lock needed) → the caller confirms → under the pass lock,
`plan_ask` again and `same_selection` (refuse if the posts moved) → `run.run_ask` (the pass:
pool, checkpoint, save, run log — `run.run_pass`, the same loop as `jev topics`) →
`finish_ask` (the results, and the history under the one rule of what is recorded).

What this module owns:

* the query (`AskQuery`: normalised text, its sha — the file name — and its question digest);
* the contract (`ask_contract` = `assess.contract` under its own version), so a repeated
  query never re-pays an unchanged post and new evidence re-asks it;
* the pre-filters (`AskFilters`, `filter_posts`), the funnel (`select_ask_items`, which is
  `assess.select_by_contract`), the cost model and estimate (`cost_model`, `estimate_ask`,
  priced by `defaults.tokens_cost_usd`), the results and the history (`load_history`).

What a query has COST is not here: it is the run log's (`report.ask_cost`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from xbrain.config import Config
from xbrain.jev.assess import (
    Selection,
    build_topic_state,
    contract,
    record_refusal,
    sha256,
    questions_digest,
    select_by_contract,
)
from xbrain.jev.client import JevClient, JevError, NoulAnswer, Question
from xbrain.jev.defaults import (
    DEFAULT_ASK_TOKENS_PER_CALL,
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_PROVIDER,
    tokens_cost_usd,
)
from xbrain.jev.load import JevPairs, load_jev_pairs
from xbrain.jev.models import (
    AskAssessment,
    AskCalibration,
    AskFile,
    AskHistoryEntry,
    AskIndex,
    TopicAssessment,
)
from xbrain.jev.questions import ASK_KEY, STATE_KEY, build_ask_questions, normalize_query
from xbrain.jev.report import jev_assigned
from xbrain.jev.store import (
    ASK_INDEX,
    ask_files,
    load_ask_file,
    load_ask_index,
    load_asks,
    save_ask_index,
)
from xbrain.models import Item
from xbrain.store import load_store

if TYPE_CHECKING:
    from xbrain.jev.run import RunOutcome

#: Bumped only when the COMPOSITION of the contract changes (see `assess._CONTRACT_VERSION`).
#: Distinct from the topics version, so a topics contract can never pass for an ask one.
_CONTRACT_VERSION = "xbrain-jev-ask/v1"


@dataclass(frozen=True)
class AskQuery:
    """One query, as asked and as filed. Build it with `AskQuery.of(raw)`.

    `text` is `normalize_query(raw)` — what Jev receives and what the file stores; `sha` is
    its sha256, the file name under `data/jev/asks/` and the run log's `query_sha`; `digest`
    is `assess.questions_digest` of `questions`, hashed into every answer's contract.
    """

    text: str
    sha: str
    questions: dict[str, Question] = field(compare=False)
    digest: str

    @classmethod
    def of(cls, raw: str) -> AskQuery:
        text = normalize_query(raw)
        questions = build_ask_questions(text)
        return cls(
            text=text, sha=sha256(text), questions=questions, digest=questions_digest(questions)
        )


def ask_contract(state_text: str, digest: str) -> str:
    """sha256(version ∥ state as sent ∥ question digest) — `assess.topic_contract`'s shape.

    The judge is outside it, as for topics: re-pointing `[jev].model` does not retire answers.
    """
    return contract(_CONTRACT_VERSION, state_text, digest)


def question_chars(questions: dict[str, Question]) -> int:
    """The characters of the question text sent with every post: instructions plus criteria.

    Part of `prompt_chars`, so an estimate can account for a long query as well as a long post.
    """
    total = 0
    for question in questions.values():
        total += len(question.instructions)
        total += sum(len(text or "") for text in (question.criteria or {}).values())
    return total


def assess_post(
    item: Item,
    query: AskQuery,
    client: JevClient,
    *,
    char_limit: int,
    now: datetime | None = None,
) -> AskAssessment:
    """One call for one post; total with respect to `JevError` for everything Jev answered."""
    state, state_chars = build_topic_state(item, char_limit)
    result = client.ask(state, query.questions)
    answer = result.answers.get(ASK_KEY)
    if not isinstance(answer, NoulAnswer):
        raise JevError(f"Jev no contestó la pregunta {ASK_KEY!r}")
    try:
        return AskAssessment(
            item_id=item.id,
            provider=result.provider,
            model=result.model,
            asked_at=now or datetime.now(timezone.utc),
            contract=ask_contract(state[STATE_KEY], query.digest),
            state_chars=state_chars,
            truncated=state_chars > char_limit,
            prompt_chars=len(state[STATE_KEY]) + question_chars(query.questions),
            probability=answer.noul,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    except ValidationError as exc:
        raise record_refusal(exc) from exc


def _is_current(record: AskAssessment | None, state_text: str, query: AskQuery) -> bool:
    """THE ONE definition of "this stored answer still answers this query about this post"."""
    return record is not None and record.contract == ask_contract(state_text, query.digest)


# --------------------------------------------------------------------------- pre-filters

_FILTER_KEYS = ("topic", "since", "until", "author", "only_evaluated")


@dataclass(frozen=True)
class AskFilters:
    """What narrows the posts BEFORE anything is paid for. Every field is optional.

    `topic` keeps a post that enrich assigned it (primary or not) OR whose CURRENT Jev topics
    answer backs it at `[jev].threshold` — never at a query's own results threshold, so a low
    results bar cannot widen, and re-bill, the posts asked; `since`/`until` are calendar days,
    both INCLUDED, read on `created_at` in UTC (every stored instant is UTC); `author` is a
    handle, `@` and case ignored; `only_evaluated` keeps posts with a current topics answer.

    Refused on construction when they could only select nothing by mistake: `since` after
    `until`, a blank `author` or `topic` (`ValueError`, which the CLI prints as an operator
    error). `from_json` is how a caller that received them as data (the server) builds them.
    """

    topic: str | None = None
    since: date | None = None
    until: date | None = None
    author: str | None = None
    only_evaluated: bool = False

    def __post_init__(self) -> None:
        if self.since is not None and self.until is not None and self.since > self.until:
            raise ValueError(f"--since {self.since} es posterior a --until {self.until}")
        if self.author is not None and not self.author.strip().lstrip("@").strip():
            raise ValueError("--author está vacío")
        if self.topic is not None and not self.topic.strip():
            raise ValueError("--topic está vacío")

    @property
    def needs_jev(self) -> bool:
        """Whether applying these filters needs the topics side-car (`load_jev_pairs`)."""
        return self.topic is not None or self.only_evaluated

    def as_json(self) -> dict[str, str | bool]:
        """The filters that are set, as the history stores them (dates as `AAAA-MM-DD`)."""
        out: dict[str, str | bool] = {}
        if self.topic is not None:
            out["topic"] = self.topic
        if self.since is not None:
            out["since"] = self.since.isoformat()
        if self.until is not None:
            out["until"] = self.until.isoformat()
        if self.author is not None:
            out["author"] = self.author
        if self.only_evaluated:
            out["only_evaluated"] = True
        return out

    @classmethod
    def from_json(cls, data: object) -> AskFilters:
        """`as_json`'s inverse, refusing (`ValueError`) an unknown key or a value of the wrong
        type — a filter silently dropped would pay for more posts than were asked for."""
        if not isinstance(data, dict):
            raise ValueError("los filtros deben ser un objeto")
        unknown = sorted(set(data) - set(_FILTER_KEYS))
        if unknown:
            raise ValueError(f"filtro desconocido: {unknown[0]}")
        text: dict[str, str | None] = {}
        for key in ("topic", "author", "since", "until"):
            value = data.get(key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"el filtro {key} debe ser un texto")
            text[key] = value
        only = data.get("only_evaluated", False)
        if not isinstance(only, bool):
            raise ValueError("el filtro only_evaluated debe ser true o false")
        return cls(
            topic=text["topic"],
            since=_day(text["since"], "since"),
            until=_day(text["until"], "until"),
            author=text["author"],
            only_evaluated=only,
        )


def _day(value: str | None, key: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"el filtro {key} debe ser un día AAAA-MM-DD, no {value!r}") from exc


def _enrich_topics(item: Item) -> set[str]:
    if item.enriched is None:
        return set()
    topics = set(item.enriched.topics)
    if item.enriched.primary_topic:
        topics.add(item.enriched.primary_topic)
    return topics


def _known_topics(store: dict[str, Item], jev: JevPairs | None) -> set[str]:
    known = {slug for item in store.values() for slug in _enrich_topics(item)}
    if jev is not None:
        known |= {topic.slug for topic in jev.vocab}
    return known


def _on_topic(
    item: Item, assessment: TopicAssessment | None, filters: AskFilters, threshold: float
) -> bool:
    """`only_evaluated` and `topic`: the two filters that read the topics side-car."""
    if filters.only_evaluated and assessment is None:
        return False
    if filters.topic is None or filters.topic in _enrich_topics(item):
        return True
    return assessment is not None and filters.topic in jev_assigned(
        assessment.membership, threshold
    )


def _in_days(item: Item, filters: AskFilters) -> bool:
    day = item.created_at.astimezone(timezone.utc).date()
    return (filters.since is None or day >= filters.since) and (
        filters.until is None or day <= filters.until
    )


def _by_author(item: Item, author: str | None) -> bool:
    return author is None or item.author.handle.casefold() == author.lstrip("@").casefold()


def _refuse_unusable(store: dict[str, Item], filters: AskFilters, jev: JevPairs | None) -> None:
    """A caller that forgot the side-car (a bug), or a `topic` nobody uses (a typo)."""
    if filters.needs_jev and jev is None:
        raise ValueError("filter_posts: --topic y --only-evaluated necesitan las evaluaciones")
    if filters.topic is not None and filters.topic not in _known_topics(store, jev):
        raise JevError(f"topic desconocido: {filters.topic!r} (ni en el vocabulario ni en enrich)")


def filter_posts(
    store: dict[str, Item],
    filters: AskFilters,
    *,
    jev: JevPairs | None,
    threshold: float,
) -> tuple[list[Item], int]:
    """The posts `filters` keep, in store order, and how many they dropped.

    `jev` is required when `filters.needs_jev`; only its CURRENT pairs count (a stale topics
    answer describes other evidence). `threshold` is the TOPIC bar — `plan_ask` passes
    `[jev].threshold`. A `topic` nobody uses — not in the vocabulary and never assigned by
    enrich — is refused: it would select nothing and look like "no post answers".
    """
    _refuse_unusable(store, filters, jev)
    current = {item.id: assessment for item, assessment in (jev.pairs if jev else [])}
    kept = [
        item
        for item in store.values()
        if _on_topic(item, current.get(item.id), filters, threshold)
        and _in_days(item, filters)
        and _by_author(item, filters.author)
    ]
    return kept, len(store) - len(kept)


# --------------------------------------------------------------------------- the funnel


def select_ask_items(
    candidates: Sequence[Item],
    records: dict[str, AskAssessment],
    query: AskQuery,
    *,
    char_limit: int,
    limit: int | None,
    state_of: Callable[[Item], tuple[str, int]] | None = None,
) -> Selection:
    """The posts to pay for: every candidate with evidence and no current answer to `query`,
    cut by `limit` (below 1 is refused). The same funnel as topics
    (`assess.select_by_contract`), never forced: a current answer is the cache, and re-asking
    it would pay for the same probability."""

    def _current(item: Item, state_text: str) -> bool:
        return _is_current(records.get(item.id), state_text, query)

    return select_by_contract(
        candidates, _current, limit=limit, force=False, char_limit=char_limit, state_of=state_of
    )


# --------------------------------------------------------------------------- estimate


@dataclass(frozen=True)
class CostModel:
    """`tokens = posts × per_call + chars / chars_per_token` — how an ask is billed.

    `measured` is False while it is the prior (`defaults.DEFAULT_ASK_TOKENS_PER_CALL`,
    `DEFAULT_CHARS_PER_TOKEN`); `answers` is how many paid answers it was fitted on.
    """

    per_call: float
    chars_per_token: float
    measured: bool
    answers: int


_PRIOR = CostModel(
    per_call=float(DEFAULT_ASK_TOKENS_PER_CALL),
    chars_per_token=DEFAULT_CHARS_PER_TOKEN,
    measured=False,
    answers=0,
)


def cost_model(calibration: AskCalibration) -> CostModel:
    """The least-squares line through the paid answers' `(prompt_chars, input_tokens)`, or the
    prior until it means something.

    A real provider bills a fixed prompt around the state and question on every call, so a
    ratio alone (tokens ∝ chars) underestimates short posts; the intercept is that fixed part.
    Two DIFFERENT sizes are needed to separate the two terms — one size (or none) keeps the
    prior. A slope at or below zero (a bill that shrinks with longer posts) is noise, never a
    model, and keeps the prior too. A negative intercept is refitted through zero.
    """
    n = calibration.answers
    spread = n * calibration.chars_sq - calibration.chars**2
    if n < 2 or spread <= 1e-9 * max(1.0, calibration.chars_sq * n):
        return _PRIOR
    slope = (n * calibration.chars_tokens - calibration.chars * calibration.tokens) / spread
    if slope <= 0:
        return _PRIOR
    intercept = (calibration.tokens - slope * calibration.chars) / n
    if intercept < 0:
        intercept, slope = 0.0, calibration.chars_tokens / calibration.chars_sq
    return CostModel(per_call=intercept, chars_per_token=1 / slope, measured=True, answers=n)


@dataclass(frozen=True)
class AskEstimate:
    """What asking `posts` posts will cost: tokens by `model` from the characters that will be
    sent (each post's state as cut + the question), priced by THE price formula. An estimate,
    never a bill."""

    posts: int
    chars: int
    tokens: int
    usd: float
    model: CostModel


def estimate_ask(prompt_chars: Sequence[int], model: CostModel) -> AskEstimate:
    """The estimate for posts whose calls will send `prompt_chars` characters each."""
    chars = sum(prompt_chars)
    tokens = round(len(prompt_chars) * model.per_call + chars / model.chars_per_token)
    return AskEstimate(
        posts=len(prompt_chars),
        chars=chars,
        tokens=tokens,
        usd=tokens_cost_usd(tokens, DEFAULT_PROVIDER),
        model=model,
    )


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class AskResults:
    """The posts that answer, best first; how many candidates have a current answer; and
    whether this use of the query was written to the history."""

    ranked: tuple[tuple[Item, AskAssessment], ...]
    answered: int
    recorded: bool = False


def _rank(
    candidates: Sequence[Item],
    records: dict[str, AskAssessment],
    query: AskQuery,
    state_text: Callable[[Item], str | None],
    threshold: float,
) -> AskResults:
    current: list[tuple[Item, AskAssessment]] = []
    for item in candidates:
        record = records.get(item.id)
        if record is None:
            continue
        text = state_text(item)
        if text is not None and _is_current(record, text, query):
            current.append((item, record))
    ranked = sorted(
        ((item, record) for item, record in current if record.probability >= threshold),
        key=lambda pair: (-pair[1].probability, pair[0].id),
    )
    return AskResults(ranked=tuple(ranked), answered=len(current))


def ask_results(
    candidates: Sequence[Item],
    records: dict[str, AskAssessment],
    query: AskQuery,
    *,
    char_limit: int,
    threshold: float,
) -> AskResults:
    """The candidates whose CURRENT answer is at or above `threshold`, by probability then id.

    A stale answer (the post's evidence moved since) is not a result: it answered another
    post. Candidates, not the whole file: the filters of THIS use decide what is shown.
    """
    return _rank(
        candidates,
        records,
        query,
        lambda item: build_topic_state(item, char_limit)[0][STATE_KEY],
        threshold,
    )


def saved_results(
    store: dict[str, Item],
    jev: JevPairs | None,
    query: AskQuery,
    filters: AskFilters,
    records: dict[str, AskAssessment],
    *,
    topic_threshold: float,
    threshold: float,
    state_text: Callable[[Item], str | None],
) -> AskResults:
    """The results of a query asked before, over the filters it was asked with: what the page
    shows and `finish_ask` ranked. TWO BARS, never one: `topic_threshold` is `[jev].threshold`,
    the bar `--topic` and `--only-evaluated` judge Jev's topics answers at (as `plan_ask`
    does); `threshold` is the results bar that use asked for, which only ranks. `state_text`
    is each post's state as sent, so a caller ranking many queries builds it once."""
    candidates, _ = filter_posts(store, filters, jev=jev, threshold=topic_threshold)
    return _rank(candidates, records, query, state_text, threshold)


# --------------------------------------------------------------------------- history


def _similar_key(text: str) -> str:
    """A query with case, punctuation and spacing taken out: what makes two queries "the same
    question" to a reader, and two different paid files to the cache."""
    return "".join(char for char in text.casefold() if char.isalnum())


def similar_queries(index: AskIndex, query: AskQuery) -> tuple[str, ...]:
    """Queries already asked that differ from `query` only in case, punctuation or spacing."""
    key = _similar_key(query.text)
    return tuple(
        sorted(
            entry.query
            for sha, entry in index.queries.items()
            if sha != query.sha and _similar_key(entry.query) == key
        )
    )


def _rebuilt_entry(stored: AskFile, sha: str, threshold: float) -> AskHistoryEntry:
    records = list(stored.assessments.values())
    moments = [record.asked_at for record in records] or [datetime.now(timezone.utc)]
    return AskHistoryEntry(
        query_sha=sha,
        query=stored.query,
        first_asked_at=min(moments),
        last_asked_at=max(moments),
        times=1,
        last_evaluated=len(records),
        last_results=sum(1 for record in records if record.probability >= threshold),
        last_threshold=threshold,
        last_filters={},
        rebuilt=True,
    )


def _calibrate(calibration: AskCalibration, records: Iterable[AskAssessment]) -> AskCalibration:
    for record in records:
        if record.input_tokens:
            calibration = calibration.add(record.prompt_chars, record.input_tokens)
    return calibration


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return None


def _newer(file: Path, written: int | None) -> bool:
    """Whether `file` was written after the history (`written`: its mtime; `None`: absent)."""
    return written is None or (_mtime(file) or 0) > written


def _unseen(stored: AskFile, entry: AskHistoryEntry | None) -> list[AskAssessment]:
    """The answers the calibration has not folded: all of them for a query with no entry,
    else those asked after the entry's last use."""
    return [
        record
        for record in stored.assessments.values()
        if entry is None or record.asked_at > entry.last_asked_at
    ]


def load_history(cfg: Config) -> AskIndex:
    """`asks/index.json`, with what it lost put back from the answer files.

    `finish_ask` writes the history AFTER the pass saved the query's file, so a file NEWER
    than `index.json` (or any file, when there is no index) holds answers the history never
    recorded: a crash between the two writes. Only those files are opened — a history in step
    costs one read and a `stat` per query. From each:

    * a query with no entry gets one rebuilt from its file (`AskHistoryEntry.rebuilt`);
    * the answers the calibration has not seen are folded in: every answer of a query with no
      entry, or those asked after its entry's `last_asked_at` (the older ones were folded when
      they were paid).

    An entry missing from a NEWER index (removed by hand) is rebuilt too, but its answers are
    not re-counted. A file that cannot be read is refused and named as another query's
    (`store.load_ask_file`). Nothing is written here: the next `finish_ask` saves it, under
    the lock.
    """
    path = cfg.jev_asks_dir / ASK_INDEX
    index = load_ask_index(path)
    written = _mtime(path)
    queries = dict(index.queries)
    calibration = index.calibration
    for file in ask_files(cfg.jev_asks_dir):
        entry = queries.get(file.stem)
        newer = _newer(file, written)
        if entry is not None and not newer:
            continue
        stored = load_ask_file(file)
        if newer:
            calibration = _calibrate(calibration, _unseen(stored, entry))
        if entry is None:
            queries[file.stem] = _rebuilt_entry(stored, file.stem, cfg.jev_threshold)
    return AskIndex(queries=queries, calibration=calibration)


# --------------------------------------------------------------------------- plan / finish


@dataclass(frozen=True)
class AskPlan:
    """Everything one use of a query decided before paying: THE shared first half of
    `jev ask` and of the server's ask job.

    `records` is the query's file in memory — `run.run_ask` updates it in place. `states`
    holds each candidate's state text AS SENT, for every candidate with evidence: built once
    and reused for the funnel, the estimate and the results. `history` is the validated
    index (lost entries rebuilt), which `finish_ask` extends and saves. `similar` names
    queries already asked that differ only in case, punctuation or spacing.
    """

    query: AskQuery
    filters: AskFilters
    candidates: tuple[Item, ...]
    dropped: int
    records: dict[str, AskAssessment]
    selection: Selection
    estimate: AskEstimate
    states: dict[str, str]
    history: AskIndex
    similar: tuple[str, ...]


def ask_path(cfg: Config, query: AskQuery) -> Path:
    """`data/jev/asks/<sha>.json` — where one query's answers live."""
    return cfg.jev_asks_dir / f"{query.sha}.json"


def plan_ask(
    cfg: Config,
    query: AskQuery,
    filters: AskFilters,
    limit: int | None,
    *,
    jev: JevPairs | None = None,
) -> AskPlan:
    """Load, filter, select and estimate — everything before a confirmation, nothing paid.

    Reads only; safe without the lock (a CLI plans, confirms, then takes the lock and plans
    AGAIN — `same_selection` — so a prompt never holds the lock). The history is validated
    here, so a corrupt one refuses before any confirmation. The Jev side of `--topic` and
    `--only-evaluated` is judged at `[jev].threshold` over CURRENT topics answers; `jev` is
    loaded when those filters need it and not handed in.
    """
    char_limit = cfg.jev_state_char_limit
    if filters.needs_jev and jev is None:
        jev = load_jev_pairs(cfg)
    store = jev.store if jev is not None else load_store(cfg.items_path)
    candidates, dropped = filter_posts(store, filters, jev=jev, threshold=cfg.jev_threshold)
    history = load_history(cfg)
    records = load_asks(ask_path(cfg, query), query)
    built = {item.id: build_topic_state(item, char_limit) for item in candidates}
    cut = {item_id: (state[STATE_KEY], chars) for item_id, (state, chars) in built.items()}
    selection = select_ask_items(
        candidates,
        records,
        query,
        char_limit=char_limit,
        limit=limit,
        state_of=lambda item: cut[item.id],
    )
    per_question = question_chars(query.questions)
    estimate = estimate_ask(
        [len(cut[item.id][0]) + per_question for item in selection.items],
        cost_model(history.calibration),
    )
    return AskPlan(
        query=query,
        filters=filters,
        candidates=tuple(candidates),
        dropped=dropped,
        records=records,
        selection=selection,
        estimate=estimate,
        states={item_id: text for item_id, (text, chars) in cut.items() if chars},
        history=history,
        similar=similar_queries(history, query),
    )


def same_selection(before: AskPlan, after: AskPlan) -> bool:
    """Whether a plan re-made under the lock would pay for exactly the posts confirmed."""
    return [item.id for item in before.selection.items] == [
        item.id for item in after.selection.items
    ]


def finish_ask(
    cfg: Config,
    plan: AskPlan,
    outcome: RunOutcome[AskAssessment] | None,
    *,
    threshold: float,
    now: datetime | None = None,
) -> AskResults:
    """The results of this use, and its line in the history. THE CALLER HOLDS THE LOCK.

    `outcome` is `run.run_ask`'s, or `None` when nothing needed asking (every answer cached).
    The ONE rule for what is recorded: this use goes into the history — and its paid answers
    into the cost calibration — whenever it can say something true, i.e. unless it was
    interrupted before banking anything (Ctrl-C or a soft cancel with nothing kept). A pass
    whose every call failed raises out of `run_ask` and never gets here: it banked nothing.
    An interrupted pass that banked answers IS recorded, so the history never lags the file.
    """
    results = _rank(plan.candidates, plan.records, plan.query, _state_lookup(plan), threshold)
    banked = outcome.assessed if outcome is not None else ()
    if outcome is not None and outcome.interrupted and not banked:
        return results
    moment = now or datetime.now(timezone.utc)
    previous = plan.history.queries.get(plan.query.sha)
    entry = AskHistoryEntry(
        query_sha=plan.query.sha,
        query=plan.query.text,
        first_asked_at=previous.first_asked_at if previous else moment,
        last_asked_at=moment,
        times=(previous.times + 1) if previous else 1,
        last_evaluated=results.answered,
        last_results=len(results.ranked),
        last_threshold=threshold,
        last_filters=plan.filters.as_json(),
    )
    index = AskIndex(
        queries={**plan.history.queries, plan.query.sha: entry},
        calibration=_calibrate(plan.history.calibration, banked),
    )
    save_ask_index(index, cfg.jev_asks_dir / ASK_INDEX)
    return AskResults(ranked=results.ranked, answered=results.answered, recorded=True)


def _state_lookup(plan: AskPlan) -> Callable[[Item], str | None]:
    return lambda item: plan.states.get(item.id)
