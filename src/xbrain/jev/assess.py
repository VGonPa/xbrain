"""From an item to a stored `TopicAssessment`: build the state, ask, parse, stamp the contract.

The state is built from `xbrain.evidence.evidence_surfaces(item, "topics")` — the same evidence
SURFACES the verify judge admits for this target — rendered as plain values, without the judge's
`[Author]`-style labels and without the not-fetched markers `verification._source_text` adds.
One definition of what may ground a claim; two renderings of it, and this module owns neither.
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from pydantic import ValidationError

from xbrain.evidence import evidence_surfaces
from xbrain.jev.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevClient,
    JevError,
    JevResult,
    NoulAnswer,
    NoulQuestion,
    Question,
)
from xbrain.jev.defaults import plural
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.questions import PRIMARY_KEY, STATE_KEY, TOPIC_PREFIX, build_topic_questions
from xbrain.models import Item, Topic, VerifyTarget
from xbrain.verification import fingerprint_output

logger = logging.getLogger(__name__)

TARGET: VerifyTarget = "topics"
# Bumped ONLY when the COMPOSITION of the contract hash changes — not when a question's
# wording or the vocabulary changes, since those are hashed. Mirrors
# `verification._CONTRACT_VERSION`: it makes the retirement of every previously stored
# contract explicit and greppable instead of an accident of collision-freedom.
_CONTRACT_VERSION = "xbrain-jev-topics/v1"
# The surface carrying the post's own words. It is LAST in the canonical evidence order and
# FIRST in the state — see `_state_text`.
_POST_SURFACE = "tweet"


def _nfc(text: str) -> str:
    """Unicode-normalise before hashing.

    The same visible description in NFC and NFD is two different byte strings. Without this,
    an editor that rewrites `vocab.yaml` in another normal form retires every stored
    assessment and re-pays for the whole corpus, with nothing visibly changed.
    """
    return unicodedata.normalize("NFC", text)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_text(item: Item) -> str:
    """Every admitted surface's atomic values, the post's OWN WORDS first.

    `evidence_surfaces` returns the canonical reading order, in which `tweet` comes last and
    the unbounded video transcript comes third. Cutting that blob at `char_limit` therefore
    eats the transcript and throws away the post: measured on the corpus, 7 of 7 items over
    the 100k default lost their own tweet, and Jev was then asked whether "the post in
    `post`" was about a topic with the post no longer in it. Leading with the tweet makes
    that structurally impossible — a cut can only ever reach the supporting surfaces.
    """
    surfaces = evidence_surfaces(item, TARGET)
    # A stable sort on a boolean: the post surface moves to the front, everything else keeps
    # its canonical relative order.
    ordered = sorted(surfaces, key=lambda surface: surface.key != _POST_SURFACE)
    return "\n".join(value for surface in ordered for value in surface.values)


def build_topic_state(item: Item, char_limit: int) -> tuple[dict[str, str], int]:
    """`{"post": evidence}` cut to `char_limit`, plus the evidence length BEFORE the cut.

    Jev's request budget covers `state` plus ALL the questions, and this call sends one Noul
    per vocabulary topic plus the Choice — so the budget tightens as `[vocab].target_count`
    grows, and cutting the state only shrinks the state half of it. `[jev].state_char_limit`
    is a bound on the EVIDENCE, not a defence of the budget. The budget's two figures, their
    source and their date live once in `docs/jev.md` § Vendor facts; do not restate them here.

    A cut is SIGNPOSTED with `[… evidencia recortada: N caracteres omitidos …]`, following
    `rubrics.truncate_transcript`: an unmarked slice tells the model the cut point is the end
    of the post. The marker is added on top of `char_limit` — the limit bounds the evidence,
    and the few dozen characters that say evidence was dropped are not evidence.

    The returned length is the PRE-cut one, which is what gets stored: the post-cut length
    saturates at `char_limit`, so for exactly the items that were cut it is the one number
    that cannot say how much went. `truncated` is derived from it at the single site that
    records it, so the two can never disagree.
    """
    text = _state_text(item)
    if len(text) <= char_limit:
        return {STATE_KEY: text}, len(text)
    dropped = len(text) - char_limit
    cut = f"{text[:char_limit]}\n[… evidencia recortada: {dropped} caracteres omitidos …]"
    return {STATE_KEY: cut}, len(text)


def questions_digest(questions: dict[str, Question]) -> str:
    """sha256 of the canonical JSON of a whole question set — computed ONCE per run.

    The entry carries the question's TYPE as well as its fields: `dataclasses.asdict` keeps
    the fields and drops the class, so a Noul and a Choice with equal fields would otherwise
    be the same bytes, and a key that changed type would keep its stored contract.

    `sort_keys` is safe here precisely because `build_topic_questions` is canonical: there is
    exactly one wire order for a given vocabulary, so an order-independent hash is not
    ignoring a difference that reaches the model.
    """
    payload = json.dumps(
        {
            key: {"type": type(question).__name__, **asdict(question)}
            for key, question in questions.items()
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return _sha256(_nfc(payload))


def topic_contract(state_text: str, digest: str) -> str:
    """sha256 binding an assessment to what Jev was ACTUALLY asked: the state as sent (the
    truncation marker included) and the digest of the questions that went with it.

    Hashing the QUESTIONS rather than the vocabulary they were built from is what removes the
    hand-maintained list of "things that ought to invalidate": a reworded instruction, an
    edited topic description, a different fallback option and a new topic all move the digest
    on their own. Two things are deliberately OUTSIDE it. The option ORDER, because
    `build_topic_questions` is canonical and there is only one. And the JUDGE — `provider`
    and `model` are recorded on the assessment but never hashed, so re-pointing `[jev].model`
    does not retire stored work. The contract binds what Jev was ASKED, never who answered.

    `digest` is `questions_digest(questions)`, taken as a parameter rather than the question
    map so a run serialises the canonical JSON once instead of once per item.
    """
    return _sha256("\x1f".join((_CONTRACT_VERSION, _nfc(state_text), digest)))


def _record_refusal(exc: ValidationError) -> JevError:
    """The seam's Spanish message for an answer this repo's own validators refuse.

    Both records built from a Jev answer — `PrimaryChoice` here and `TopicAssessment` in
    `assess_topics` — go through it, so a refusal reads the same wherever it happens: one
    line naming the field, never pydantic's four-line English banner in a failure table.
    """
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or "<registro>"
    return JevError(f"Jev devolvió una respuesta que el registro rechaza: {field}: {first['msg']}")


def parse_topic_result(
    result: JevResult, questions: dict[str, Question]
) -> tuple[dict[str, float], PrimaryChoice]:
    """The membership map and the primary choice, read against the questions ACTUALLY asked.

    Keys and options come from `questions`, never re-derived from the vocabulary: the
    contract is a digest of this same dict, so the parser and the contract cannot end up
    describing different asks. An answer to a question nobody asked is ignored.

    `JevError` when an answer is missing, is the wrong answer type, is outside the offered
    options, is absent from its own distribution, is not an argmax of it, or carries a value
    `PrimaryChoice` refuses — a partial or self-contradictory answer set is never stored as
    if it were complete. A `questions` map with no `primary` Choice is a caller bug and
    raises `ValueError` instead: nothing about it is the provider's fault.
    """
    membership: dict[str, float] = {}
    for key, question in questions.items():
        if not isinstance(question, NoulQuestion):
            continue
        answer = result.answers.get(key)
        slug = key.removeprefix(TOPIC_PREFIX)
        if not isinstance(answer, NoulAnswer):
            raise JevError(f"Jev no contestó el noul de {slug!r}")
        membership[slug] = answer.noul
    offered = questions.get(PRIMARY_KEY)
    if not isinstance(offered, ChoiceQuestion):
        # `ValueError`, not `JevError`: the provider did nothing wrong. `build_topic_questions`
        # cannot produce this map, so reaching here is a caller bug, and dressing it as an
        # operator-facing Jev failure would send whoever reads `failed` looking at the API.
        raise ValueError("el juego de preguntas no incluye la Choice 'primary'")
    primary = result.answers.get(PRIMARY_KEY)
    if not isinstance(primary, ChoiceAnswer):
        raise JevError("Jev no contestó la pregunta 'primary'")
    if primary.choice not in offered.criteria:
        raise JevError(f"Jev eligió {primary.choice!r}, que no está entre las opciones")
    probability = primary.probabilities.get(primary.choice)
    if probability is None:
        raise JevError(f"Jev eligió {primary.choice!r} pero no está en su propia distribución")
    # Presence is not enough. A winner sitting at 0.0 while a loser holds 1.0 scores zero for
    # every reader using the `.get(option, 0.0)` access `PrimaryChoice` prescribes — the very
    # outcome the presence check was written to prevent. Ties are fine; being beaten is not.
    if probability < max(primary.probabilities.values()):
        raise JevError(
            f"Jev eligió {primary.choice!r} con p={probability}, por debajo del máximo de su "
            f"propia distribución"
        )
    try:
        choice = PrimaryChoice(
            choice=primary.choice,
            confidence=primary.confidence,
            probabilities=dict(primary.probabilities),
        )
    except ValidationError as exc:
        # The guards above cover the answer's SHAPE; the model still owns its own bounds (a
        # confidence outside [0, 1], a probability that is not one). Wrapped here so this
        # function is total with respect to `JevError` for a direct caller too.
        raise _record_refusal(exc) from exc
    return membership, choice


def assess_topics(
    item: Item,
    questions: dict[str, Question],
    client: JevClient,
    *,
    char_limit: int,
    digest: str | None = None,
    now: datetime | None = None,
) -> TopicAssessment:
    """One call for one item, TOTAL with respect to `JevError` for everything Jev answered.

    The same `questions` object is asked and hashed, so a stored contract can never describe
    a question other than the one that was sent. A record the model refuses is converted at
    this seam: `JevError` is documented as the only exception the seam emits, and a caller
    with no batch around it must get the operator's message rather than a pydantic banner.

    One thing is deliberately NOT a `JevError`: a malformed `questions` map raises
    `ValueError`, exactly as `build_topic_questions` does for a malformed vocabulary. Both
    are caller bugs, and neither is something to tell an operator about the provider.

    `output_fingerprint` records WHICH enrich assignment existed when Jev was asked. It is
    informational — the report recomputes the comparison against `enrich` at report time —
    and is never consulted for currency, which is `contract`'s job alone.

    `digest` is `questions_digest(questions)`; pass it to serialise the canonical JSON once
    per run instead of once per item. It must be the digest of THESE questions.
    """
    contract_digest = questions_digest(questions) if digest is None else digest
    state, state_chars = build_topic_state(item, char_limit)
    result = client.ask(state, questions)
    membership, primary = parse_topic_result(result, questions)
    try:
        return TopicAssessment(
            item_id=item.id,
            provider=result.provider,
            model=result.model,
            asked_at=now or datetime.now(timezone.utc),
            output_fingerprint=fingerprint_output(item, TARGET),
            contract=topic_contract(state[STATE_KEY], contract_digest),
            state_chars=state_chars,
            truncated=state_chars > char_limit,
            membership=membership,
            primary=primary,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    except ValidationError as exc:
        raise _record_refusal(exc) from exc


def _contract_matches(assessment: TopicAssessment | None, state_text: str, digest: str) -> bool:
    """The ONE definition of "this stored assessment still describes this ask"; `None` (no
    stored assessment) never matches. `assessment_is_current` and `select_items` both route
    through it so the public predicate and the skip rule cannot drift apart."""
    return assessment is not None and assessment.contract == topic_contract(state_text, digest)


def assessment_is_current(
    assessment: TopicAssessment, item: Item, vocab: list[Topic], *, fallback: str, char_limit: int
) -> bool:
    """Whether `assessment` still describes what asking about `item` today would ask."""
    state, _ = build_topic_state(item, char_limit)
    digest = questions_digest(build_topic_questions(vocab, fallback))
    return _contract_matches(assessment, state[STATE_KEY], digest)


@dataclass(frozen=True)
class CurrentPairs:
    """The still-current `(item, assessment)` pairs, and what was DROPPED getting there.

    THE THREE NUMBERS PARTITION THE SIDE-CAR: `len(pairs) + stale + orphans` is every stored
    record. That is the whole point of the dataclass. A drop with no counter is the failure
    `Selection` refuses one screen up — after a vocabulary edit, a side-car holding thousands
    of paid records and a side-car nobody ever wrote both produce `0 evaluaciones`, and the
    operator re-pays for the corpus to find out which one they had.

    `stale` is a record whose contract no longer describes today's ask. `orphans` is a record
    whose item is no longer in the store — a different event with the same symptom, so it gets
    its own number rather than hiding inside the first.

    `fallback` and `char_limit` are PROVENANCE, not parameters: they are the two inputs the
    verdict was computed from, and they travel with it so a consumer that accepts a hand-in
    (`dashboard.compute_jev_dashboard_data`) can refuse one decided under other options. The
    counts look identical whatever they were computed with, so nothing in the shape of this
    object could catch the swap — the same argument `TopicAssessment.contract` answers for a
    stored record.
    """

    pairs: tuple[tuple[Item, TopicAssessment], ...]
    stale: int
    orphans: int
    fallback: str
    char_limit: int


def current_pairs(
    items: list[Item],
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    *,
    fallback: str,
    char_limit: int,
) -> CurrentPairs:
    """The pairs whose stored assessment still describes today's ask, and the two drop counts.

    THE ONE DEFINITION OF CURRENCY for a whole side-car — `assessment_is_current` is the
    single-record predicate, and both route through `_contract_matches`, so the report and the
    assessment side can never disagree about what "still valid" means.

    The questions and their digest are built ONCE for the whole side-car, exactly as
    `select_items` hoists them and for the same reason: they do not depend on the item, so
    doing it per record re-validates the vocabulary, re-serialises the canonical JSON and
    re-hashes it thousands of times in a command that is meant to be a cheap read.
    """
    digest = questions_digest(build_topic_questions(vocab, fallback))
    known = {item.id for item in items}
    pairs: list[tuple[Item, TopicAssessment]] = []
    stale = 0
    for item in items:
        assessment = assessments.get(item.id)
        if assessment is None:
            # Not a drop: this item simply has no stored assessment. It is counted by the
            # assessment side (`Selection`), never here.
            continue
        state, _ = build_topic_state(item, char_limit)
        if _contract_matches(assessment, state[STATE_KEY], digest):
            pairs.append((item, assessment))
        else:
            stale += 1
    return CurrentPairs(
        pairs=tuple(pairs),
        stale=stale,
        orphans=sum(1 for item_id in assessments if item_id not in known),
        fallback=fallback,
        char_limit=char_limit,
    )


def _candidates(store: dict[str, Item], ids: list[str] | None) -> list[Item]:
    """The items `ids` names, in the order asked and de-duplicated; every item when `ids` is
    `None` or empty (`[]` is deliberately "the whole corpus" — the CLI maps a missing `--id`
    to it).

    A repeated id is collapsed rather than asked twice: two paid calls for one post produce
    two records under one `item_id`, which the side-car would then reconcile silently. An
    unknown id is an error, never a silent omission — one that quietly selected nothing would
    look like a successful no-op.
    """
    if not ids:
        return list(store.values())
    unique = list(dict.fromkeys(ids))
    missing = [item_id for item_id in unique if item_id not in store]
    if missing:
        raise JevError(f"ids desconocidos: {', '.join(missing)}")
    return [store[item_id] for item_id in unique]


@dataclass(frozen=True)
class Selection:
    """What `select_items` picked, and how much it passed over.

    The counts are not decoration: without them an empty selection caused by a regression in
    the evidence layer is indistinguishable from a clean "everything is up to date", and the
    run exits 0 either way.

    THEY PARTITION THE CANDIDATE SET: `len(items) + skipped_current + skipped_no_evidence +
    remaining` is every item considered. `forced` is a SUB-count of `items`, not a fifth
    bucket — the items in it are being asked about, they are just being asked again.
    """

    items: tuple[Item, ...]
    skipped_current: int
    skipped_no_evidence: int
    #: Selected items whose stored assessment was STILL CURRENT and is being re-asked anyway
    #: because `force` is set. Under `force`, `skipped_current` is structurally 0 — it means
    #: "we did not look", not "nothing was current" — so without this the operator cannot
    #: tell "I just re-paid for 2,998 perfectly current assessments" from "2,998 contracts
    #: had genuinely expired". That is the counter failing at the one moment it is worth
    #: money. Counts only items that SURVIVED `limit`: a forced item that was cut is not
    #: going to be re-asked, and reporting it would promise a re-bill that never happens.
    forced: int = 0
    #: Evaluable items left out by `limit`. The two skips above cost nothing; this is the one
    #: that means "there is more backlog still to pay for". Without it `--limit 200` run
    #: nightly cannot say whether the backlog is draining, stalled or growing, and `--dry-run`
    #: reports the same truncated number so it cannot be used to find out either.
    remaining: int = 0


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
) -> Selection:
    """Items to ask about: the requested ids (every item when None or empty), skipping items
    with no evidence and — unless `force` — items whose stored assessment is still current.

    Naming an id does not imply `--force`, and naming an evidence-free item is not an error:
    both are COUNTED, so the CLI can say why nothing happened instead of printing a bare
    "nothing to do". `limit=None` means no limit; a limit below 1 is an operator error, not
    an empty success.

    Every candidate ends in exactly one of four places — selected, current, evidence-free, or
    cut by `limit` — and `Selection` reports all four, so the operator's line accounts for the
    whole corpus rather than for the part that happened to survive.

    The questions and their digest are built ONCE for the whole selection: they do not depend
    on the item, so doing it per item would redo identical work for every post in the corpus.

    `build_topic_state` deliberately is NOT hoisted the same way, and the asymmetry is the
    point. The digest is one value for the whole run, so building it per item is pure waste;
    the state is per ITEM, and threading it from here to `assess_topics` would mean the
    function that SENDS a state receives it from elsewhere — the one place a future edit to
    the cut or the evidence set could make the state that was judged for currency differ from
    the state that was billed. Measured rather than assumed: a full pass over 2,609 items is
    0.011 s (2026-09-22, this machine), against a command whose unit of work is a network
    call. Paying it twice buys `assess_topics` the property that it computes what it sends.
    """
    if limit is not None and limit < 1:
        raise JevError("--limit debe ser >= 1")
    digest = questions_digest(build_topic_questions(vocab, fallback))
    selected: list[Item] = []
    # Parallel to `selected`: was this item's stored assessment still current when we took
    # it anyway? Recorded per item rather than counted on the spot so `limit` can cut both
    # lists together — `forced` must describe the items that will actually be re-asked.
    was_current: list[bool] = []
    skipped_current = 0
    skipped_no_evidence = 0
    for item in _candidates(store, ids):
        state, state_chars = build_topic_state(item, char_limit)
        # Judged on the PRE-cut evidence: an item with evidence but a tiny window has
        # something to ask about, and `evidence_surfaces` already drops blank values.
        if state_chars == 0:
            skipped_no_evidence += 1
            continue
        # Computed even under `force`, which costs nothing and is the whole point: the
        # answer is what tells a forced re-bill of a current corpus apart from a corpus
        # whose contracts had expired. Skipping the check would make the two identical.
        current = _contract_matches(assessments.get(item.id), state[STATE_KEY], digest)
        if current and not force:
            skipped_current += 1
            continue
        selected.append(item)
        was_current.append(current)
    cut = len(selected) if limit is None else min(limit, len(selected))
    return Selection(
        items=tuple(selected[:cut]),
        skipped_current=skipped_current,
        skipped_no_evidence=skipped_no_evidence,
        forced=sum(was_current[:cut]),
        remaining=len(selected) - cut,
    )


@dataclass(frozen=True)
class RunResult:
    """What one run produced: the records, and the reason each failed item failed.

    `failed` is not "provider faults" — it also holds records this repo's own validators
    refused. The reason is stamped with the exception type for anything that is not a
    `JevError`, so an xbrain bug does not read as N bad answers.
    """

    assessed: tuple[TopicAssessment, ...]
    failed: tuple[tuple[str, str], ...]  # (item_id, reason)


def _report_progress(on_progress: Callable[[int, int], None] | None, done: int, total: int) -> None:
    """Report progress AFTER the item's record is stored, and never let it fail the run.

    `xbrain jev topics | head` closes the pipe under the writer; a display failure must not
    throw away work that has already been paid for.
    """
    if on_progress is None:
        return
    try:
        on_progress(done, total)
    except Exception as exc:
        logger.warning(
            "on_progress falló en %d/%d (%s: %s); la evaluación continúa",
            done,
            total,
            type(exc).__name__,
            exc,
        )


def _deliver_result(
    on_result: Callable[[TopicAssessment], None] | None, assessment: TopicAssessment
) -> None:
    """Hand a stored record to the caller's checkpoint, and never let it fail the run.

    Guarded exactly like `_report_progress`, for a stronger reason: this hook exists so an
    interrupted run keeps what it has already paid for, and a checkpoint that threw would
    discard the very record it was called to save. The record stays in `assessed` either
    way — a broken checkpoint costs durability, never the result.
    """
    if on_result is None:
        return
    try:
        on_result(assessment)
    except Exception as exc:
        logger.warning(
            "on_result falló para %s (%s: %s); la evaluación continúa",
            assessment.item_id,
            type(exc).__name__,
            exc,
        )


def run_assessments(
    items: list[Item],
    vocab: list[Topic],
    client: JevClient,
    *,
    fallback: str,
    char_limit: int,
    concurrency: int,
    on_progress: Callable[[int, int], None] | None = None,
    on_result: Callable[[TopicAssessment], None] | None = None,
) -> RunResult:
    """Ask about every item, `concurrency` at a time.

    The questions are built and validated ONCE, before the pool opens, and that same object
    is handed to every worker: a bad vocabulary fails as a single `ValueError` in the
    caller's thread, not as N identical failures buried in futures.

    EVERY worker exception is recorded as that item's failure and the run continues — a
    `JevError` keeps its operator message, anything else is stamped with its type. A run
    where every item failed raises, so a dead key is an error and not an empty success; an
    empty `items` is simply an empty result and never reaches the client.

    An interrupt discards THIS FUNCTION'S collection, by design. `KeyboardInterrupt` cancels
    every queued call — every item is submitted up front, so a plain shutdown would drain the
    whole queue and the operator's Ctrl-C would still pay the full bill — and then propagates
    without a `RunResult`.

    What survives an interrupt is whatever `on_result` was already handed. Every successful
    record is delivered to it the moment it is stored, in arrival order, so a caller that
    checkpoints there keeps the work it has paid for; a caller that passes none keeps
    nothing, which is the same bargain as before. Holding the records back to deliver them
    sorted at the end would make the hook useless for the one event it exists for.

    `client.ask` is called from `concurrency` threads at once and must be safe to do so; see
    the `JevClient` protocol.
    """
    questions = build_topic_questions(vocab, fallback)
    if not items:
        return RunResult(assessed=(), failed=())
    digest = questions_digest(questions)
    assessed: list[TopicAssessment] = []
    failed: list[tuple[str, str]] = []
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        futures = {
            pool.submit(
                assess_topics, item, questions, client, char_limit=char_limit, digest=digest
            ): item
            for item in items
        }
        for done, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                assessment = future.result()
            except JevError as exc:
                failed.append((item.id, str(exc)))
            except Exception as exc:
                failed.append((item.id, f"{type(exc).__name__}: {exc}"))
            else:
                # Stored first, delivered second: the checkpoint is told about a record
                # that is already in `assessed`, never the other way round.
                assessed.append(assessment)
                _deliver_result(on_result, assessment)
            _report_progress(on_progress, done, len(items))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    # Sorted, not completion-ordered: the same corpus must produce the same file whatever
    # order the pool happened to finish in, or every run would show a spurious diff.
    assessed.sort(key=lambda assessment: assessment.item_id)
    failed.sort(key=lambda failure: failure[0])
    if not assessed:
        # The DISTINCT reason count is the diagnosis, and one quoted reason cannot carry it:
        # N failures with one reason is a key, a quota or an outage — fix one thing and
        # re-run — while N failures with forty is the corpus, and every one has to be read.
        # Showing `failed[0][1]` alone reads as the first case whichever it is.
        reasons = {reason for _, reason in failed}
        raise JevError(
            f"ninguna de las {len(items)} evaluaciones terminó: "
            f"{plural(len(failed), 'fallo', 'fallos')}, "
            f"{plural(len(reasons), 'motivo distinto', 'motivos distintos')}; "
            f"primero: {failed[0][1]}"
        )
    return RunResult(assessed=tuple(assessed), failed=tuple(failed))
