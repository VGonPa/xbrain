"""From an item to a stored `TopicAssessment`: build the state, ask, parse, stamp the contract.

The state is `evidence_text(item, "topics")` — the SAME evidence the verify judge reads for the
`topics` target. One definition of evidence, not two.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from xbrain.evidence import evidence_text
from xbrain.jev.client import (
    ChoiceAnswer,
    JevClient,
    JevError,
    JevResult,
    NoulAnswer,
    Question,
)
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.questions import PRIMARY_KEY, STATE_KEY, TOPIC_PREFIX, build_topic_questions
from xbrain.models import Item, Topic, VerifyTarget
from xbrain.verification import fingerprint_output

TARGET: VerifyTarget = "topics"
# Bumped ONLY when the COMPOSITION of the contract hash changes — not when a question's
# wording or the vocabulary changes, since those are hashed. Mirrors
# `verification._CONTRACT_VERSION`: it makes the retirement of every previously stored
# contract explicit and greppable instead of an accident of collision-freedom.
_CONTRACT_VERSION = "xbrain-jev-topics/v1"


def build_topic_state(item: Item, char_limit: int) -> tuple[dict[str, str], int]:
    """`{"post": evidence}` cut to `char_limit`, plus the evidence length BEFORE the cut.

    Jev's window is 32k tokens, so a long item is cut. The PRE-cut length is what gets
    stored (`TopicAssessment.state_chars`): the post-cut one is `char_limit` handed back
    and would tell a report nothing about how much was dropped. `truncated` is derived
    from it at the single place that records it, so the two can never disagree.
    """
    text = evidence_text(item, TARGET)
    return {STATE_KEY: text[:char_limit]}, len(text)


def topic_contract(state_text: str, questions: dict[str, Question]) -> str:
    """sha256 binding an assessment to what Jev was ACTUALLY asked: the state, plus the
    canonical form of every question sent with it.

    Hashing the QUESTIONS — rather than the vocabulary they were built from — is what makes
    the binding total. A reworded instruction, an edited topic description, a different
    fallback option and a new topic all change this digest, and none of them can be missed
    by a hand-maintained list of "things that ought to invalidate". Re-enriching the item
    does NOT change it (the comparison against `enrich` is recomputed at report time); new
    evidence text does. `sort_keys` makes the digest independent of dict ordering, so
    reshuffling `vocab.yaml` does not retire every stored assessment; `ensure_ascii=False`
    keeps a Spanish description hashing as its own characters.
    """
    payload = json.dumps(
        {key: asdict(question) for key, question in questions.items()},
        sort_keys=True,
        ensure_ascii=False,
    )
    parts = [_CONTRACT_VERSION, state_text, payload]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def parse_topic_result(
    result: JevResult, vocab: list[Topic], fallback: str
) -> tuple[dict[str, float], PrimaryChoice]:
    """The membership map and the primary choice, or `JevError` when an answer is missing or
    outside the options — a partial answer set is never stored as if it were complete."""
    membership: dict[str, float] = {}
    for topic in vocab:
        answer = result.answers.get(TOPIC_PREFIX + topic.slug)
        if not isinstance(answer, NoulAnswer):
            raise JevError(f"Jev no contestó el noul de {topic.slug!r}")
        membership[topic.slug] = answer.noul
    primary = result.answers.get(PRIMARY_KEY)
    if not isinstance(primary, ChoiceAnswer):
        raise JevError("Jev no contestó la pregunta 'primary'")
    if primary.choice not in {topic.slug for topic in vocab} | {fallback}:
        raise JevError(f"Jev eligió {primary.choice!r}, que no está entre las opciones")
    # `PrimaryChoice` refuses a winner with no entry in its own distribution, and it refuses
    # it with a `ValidationError` — not a `JevError`. Caught HERE, a truncated or renormalised
    # distribution is one recorded failure; left to the record's validator it would escape the
    # seam's error type and abort the whole batch. Same class of fault as a missing Noul, so
    # the same exception.
    if primary.choice not in primary.probabilities:
        raise JevError(f"Jev eligió {primary.choice!r} pero no está en su propia distribución")
    return membership, PrimaryChoice(
        choice=primary.choice,
        confidence=primary.confidence,
        probabilities=dict(primary.probabilities),
    )


def assess_topics(
    item: Item,
    vocab: list[Topic],
    client: JevClient,
    *,
    fallback: str,
    char_limit: int,
    now: datetime | None = None,
) -> TopicAssessment:
    """One call for one item.

    The questions are built ONCE and then both asked and hashed, so a stored contract can
    never describe a question other than the one that was sent.
    """
    questions = build_topic_questions(vocab, fallback)
    state, state_chars = build_topic_state(item, char_limit)
    result = client.ask(state, questions)
    membership, primary = parse_topic_result(result, vocab, fallback)
    return TopicAssessment(
        item_id=item.id,
        # Provenance as the judge REPORTED it, never a value this record assumes about itself.
        provider=result.provider,
        model=result.model,
        asked_at=now or datetime.now(timezone.utc),
        output_fingerprint=fingerprint_output(item, TARGET),
        contract=topic_contract(state[STATE_KEY], questions),
        state_chars=state_chars,
        truncated=state_chars > char_limit,
        membership=membership,
        primary=primary,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _contract_matches(
    assessment: TopicAssessment | None, state_text: str, questions: dict[str, Question]
) -> bool:
    """The ONE definition of "this stored assessment still describes this ask"; `None` (no
    stored assessment) never matches. `assessment_is_current` and `select_items` both route
    through it so the skip rule and the public predicate cannot drift apart."""
    return assessment is not None and assessment.contract == topic_contract(state_text, questions)


def assessment_is_current(
    assessment: TopicAssessment, item: Item, vocab: list[Topic], *, fallback: str, char_limit: int
) -> bool:
    """Whether `assessment` still describes what asking about `item` today would ask."""
    state, _ = build_topic_state(item, char_limit)
    return _contract_matches(assessment, state[STATE_KEY], build_topic_questions(vocab, fallback))


def _candidates(store: dict[str, Item], ids: list[str] | None) -> list[Item]:
    """The items `ids` names, in the order asked, or every item when `ids` is empty.

    An unknown id is an error, never a silent omission: a typo'd `--id` that quietly
    selected nothing would report "nada que evaluar" and look like a successful no-op.
    """
    if not ids:
        return list(store.values())
    missing = [item_id for item_id in ids if item_id not in store]
    if missing:
        raise JevError(f"ids desconocidos: {', '.join(missing)}")
    return [store[item_id] for item_id in ids]


def select_items(
    store: dict[str, Item],
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    *,
    ids: list[str] | None,
    limit: int | None,
    force: bool,
    fallback: str,
    char_limit: int,
) -> list[Item]:
    """Items to ask about: the requested ids (every item when None), skipping items with no
    evidence and — unless `force` — items whose stored assessment is still current.

    The questions are built ONCE for the whole selection: they do not depend on the item,
    so rebuilding them per item would redo identical work for every post in the corpus.
    """
    questions = build_topic_questions(vocab, fallback)
    selected: list[Item] = []
    for item in _candidates(store, ids):
        state, _ = build_topic_state(item, char_limit)
        state_text = state[STATE_KEY]
        # Nothing for Jev to read: asking would spend a call to have it judge an empty post.
        if not state_text.strip():
            continue
        if not force and _contract_matches(assessments.get(item.id), state_text, questions):
            continue
        selected.append(item)
    return selected[:limit] if limit is not None else selected


@dataclass(frozen=True)
class RunResult:
    """What one run produced: the records, and the reason each failed item failed."""

    assessed: list[TopicAssessment]
    failed: list[tuple[str, str]]  # (item_id, reason)


def run_assessments(
    items: list[Item],
    vocab: list[Topic],
    client: JevClient,
    *,
    fallback: str,
    char_limit: int,
    concurrency: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> RunResult:
    """Ask about every item, `concurrency` at a time. A failed item is recorded, never dropped;
    a run where EVERY item failed raises, so a dead key or a dead API is an error, not an
    empty success. The questions are validated once, before any call."""
    build_topic_questions(vocab, fallback)
    assessed: list[TopicAssessment] = []
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                assess_topics, item, vocab, client, fallback=fallback, char_limit=char_limit
            ): item
            for item in items
        }
        for done, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                assessed.append(future.result())
            except JevError as exc:
                failed.append((item.id, str(exc)))
            except ValidationError as exc:
                # An answer the seam accepted but the RECORD refuses (a probability outside
                # [0, 1], an empty model name). One bad item must not discard the assessments
                # that already completed, so it is recorded exactly like a `JevError`. The
                # first complaint only: `str(exc)` is a four-line banner and `failed` is
                # printed one line per item.
                failed.append((item.id, exc.errors()[0]["msg"]))
            if on_progress is not None:
                on_progress(done, len(items))
    # Sorted, not completion-ordered: the same corpus must produce the same file whatever
    # order the pool happened to finish in, or every run would show a spurious diff.
    assessed.sort(key=lambda assessment: assessment.item_id)
    failed.sort()
    if items and not assessed:
        raise JevError(
            f"ninguna de las {len(items)} evaluaciones terminó; primer error: {failed[0][1]}"
        )
    return RunResult(assessed=assessed, failed=failed)
