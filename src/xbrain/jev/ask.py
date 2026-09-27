"""`xbrain jev ask`: ask the corpus a question; Jev says, post by post, whether it answers it.

ONE NOUL PER POST (`questions.build_ask_questions`): the user's query verbatim as the true
criterion, over the SAME state a topics pass sends (`assess.build_topic_state`). Each post is
judged alone, so its probability does not depend on the company it was asked in. The results
are the posts whose CURRENT answer is at or above the threshold, best first.

What this module owns, and what it borrows:

* the query (`AskQuery`: normalised text, its sha — the file name — and its question digest);
* the contract (`ask_contract`), built like `assess.topic_contract` under its own version, so
  a repeated query never re-pays an unchanged post and new evidence re-asks it;
* the pre-filters (`filter_posts`), the funnel (`select_ask_items`, which is
  `assess.select_by_contract`), the estimate (`estimate_ask`, priced by
  `defaults.tokens_cost_usd`), the results (`ask_results`) and the history (`record_ask`).

The pass itself — pool, checkpoint, save, run log, lock — is `run.run_ask`, which is
`run.run_pass`, the same loop as `jev topics`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from pydantic import ValidationError

from xbrain.config import Config
from xbrain.jev.assess import (
    Selection,
    _nfc,
    _record_refusal,
    _sha256,
    build_topic_state,
    questions_digest,
    select_by_contract,
)
from xbrain.jev.client import JevClient, JevError, NoulAnswer, Question
from xbrain.jev.defaults import DEFAULT_CHARS_PER_TOKEN, DEFAULT_PROVIDER, tokens_cost_usd
from xbrain.jev.load import JevPairs
from xbrain.jev.models import AskAssessment, AskHistoryEntry, TopicAssessment
from xbrain.jev.report import jev_assigned
from xbrain.jev.questions import ASK_KEY, STATE_KEY, build_ask_questions, normalize_query
from xbrain.jev.store import ASK_INDEX, load_ask_index, save_ask_index
from xbrain.models import Item

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
            text=text, sha=_sha256(text), questions=questions, digest=questions_digest(questions)
        )


def ask_contract(state_text: str, digest: str) -> str:
    """sha256(version ∥ state as sent ∥ question digest) — `assess.topic_contract`'s shape.

    The judge is outside it, as for topics: re-pointing `[jev].model` does not retire answers.
    """
    return _sha256("\x1f".join((_CONTRACT_VERSION, _nfc(state_text), digest)))


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
        raise _record_refusal(exc) from exc


def _is_current(record: AskAssessment | None, state_text: str, query: AskQuery) -> bool:
    """THE ONE definition of "this stored answer still answers this query about this post"."""
    return record is not None and record.contract == ask_contract(state_text, query.digest)


# --------------------------------------------------------------------------- pre-filters


@dataclass(frozen=True)
class AskFilters:
    """What narrows the posts BEFORE anything is paid for. Every field is optional.

    `topic` keeps a post that enrich assigned it (primary or not) OR whose current Jev topics
    answer backs it at the threshold; `since`/`until` are calendar days, both INCLUDED, read
    on `created_at` in UTC (every stored instant is UTC);
    `author` is a handle, `@` and case ignored; `only_evaluated` keeps posts with a current
    Jev topics answer.
    """

    topic: str | None = None
    since: date | None = None
    until: date | None = None
    author: str | None = None
    only_evaluated: bool = False

    @property
    def needs_jev(self) -> bool:
        """Whether applying these filters needs the topics side-car (`load_jev_pairs`)."""
        return self.topic is not None or self.only_evaluated

    @classmethod
    def from_json(cls, stored: dict[str, str | bool]) -> AskFilters:
        """The filters `as_json` wrote — how the history's entry is asked again."""
        since, until = stored.get("since"), stored.get("until")
        topic, author = stored.get("topic"), stored.get("author")
        return cls(
            topic=topic if isinstance(topic, str) else None,
            since=date.fromisoformat(since) if isinstance(since, str) else None,
            until=date.fromisoformat(until) if isinstance(until, str) else None,
            author=author if isinstance(author, str) else None,
            only_evaluated=stored.get("only_evaluated") is True,
        )

    def as_json(self) -> dict[str, str | bool]:
        """The filters that are set, as the history stores them."""
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

    `jev` is required when `filters.needs_jev`. A `topic` nobody uses — not in the vocabulary
    and never assigned by enrich — is refused: it would select nothing and look like "no post
    answers", after the operator mistyped a slug.
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
) -> Selection:
    """The posts to pay for: every candidate with evidence and no current answer to `query`,
    cut by `limit`. The same funnel as topics (`assess.select_by_contract`), never forced: a
    current answer is the cache, and re-asking it would pay for the same probability."""

    def _current(item: Item, state_text: str) -> bool:
        return _is_current(records.get(item.id), state_text, query)

    return select_by_contract(candidates, _current, limit=limit, force=False, char_limit=char_limit)


# --------------------------------------------------------------------------- estimate


@dataclass(frozen=True)
class AskEstimate:
    """What asking `posts` posts will cost: input tokens from the characters that will be
    sent, at `chars_per_token`, priced by THE price formula. An estimate, never a bill."""

    posts: int
    chars: int
    chars_per_token: float
    tokens: int
    usd: float


@dataclass(frozen=True)
class TokenRatio:
    """Characters per input token, and how many paid answers it was measured on (0: the
    default, `DEFAULT_CHARS_PER_TOKEN`)."""

    value: float
    measured: int


def token_ratio(records: Iterable[AskAssessment]) -> TokenRatio:
    """Characters per input token, measured on paid ask answers that reported their usage;
    `DEFAULT_CHARS_PER_TOKEN` (low on purpose) until one exists."""
    chars = tokens = measured = 0
    for record in records:
        if record.input_tokens:
            chars += record.prompt_chars
            tokens += record.input_tokens
            measured += 1
    if not tokens:
        return TokenRatio(DEFAULT_CHARS_PER_TOKEN, 0)
    return TokenRatio(chars / tokens, measured)


def chars_per_token(records: Iterable[AskAssessment]) -> float:
    """`token_ratio`'s value: what an estimate divides the characters it will send by."""
    return token_ratio(records).value


def estimate_ask(
    selection: Selection, query: AskQuery, *, char_limit: int, chars_per_token: float
) -> AskEstimate:
    """The estimate for exactly `selection.items`, from the SAME builder the call uses: each
    post's state as it will be sent, plus the question — `AskAssessment.prompt_chars`'s sum."""
    per_question = question_chars(query.questions)
    chars = sum(
        len(build_topic_state(item, char_limit)[0][STATE_KEY]) + per_question
        for item in selection.items
    )
    tokens = round(chars / chars_per_token)
    return AskEstimate(
        posts=len(selection.items),
        chars=chars,
        chars_per_token=chars_per_token,
        tokens=tokens,
        usd=tokens_cost_usd(tokens, DEFAULT_PROVIDER),
    )


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class AskResults:
    """The posts that answer, best first, and how many candidates have a current answer."""

    ranked: tuple[tuple[Item, AskAssessment], ...]
    answered: int


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
    post. Candidates, not the whole file: the filters of THIS run decide what is shown.
    """
    current: list[tuple[Item, AskAssessment]] = []
    for item in candidates:
        record = records.get(item.id)
        if record is None:
            continue
        state, _ = build_topic_state(item, char_limit)
        if _is_current(record, state[STATE_KEY], query):
            current.append((item, record))
    ranked = sorted(
        ((item, record) for item, record in current if record.probability >= threshold),
        key=lambda pair: (-pair[1].probability, pair[0].id),
    )
    return AskResults(ranked=tuple(ranked), answered=len(current))


# --------------------------------------------------------------------------- history


def record_ask(
    cfg: Config,
    query: AskQuery,
    *,
    filters: AskFilters,
    evaluated: int,
    results: int,
    threshold: float,
    now: datetime | None = None,
) -> AskHistoryEntry:
    """Add (or refresh) `query` in `asks/index.json`. The caller holds the pass lock."""
    path = cfg.jev_asks_dir / ASK_INDEX
    index = load_ask_index(path)
    moment = now or datetime.now(timezone.utc)
    previous = index.get(query.sha)
    entry = AskHistoryEntry(
        query_sha=query.sha,
        query=query.text,
        first_asked_at=previous.first_asked_at if previous else moment,
        last_asked_at=moment,
        times=(previous.times + 1) if previous else 1,
        evaluated=evaluated,
        results=results,
        threshold=threshold,
        filters=filters.as_json(),
    )
    index[query.sha] = entry
    save_ask_index(index, path)
    return entry
