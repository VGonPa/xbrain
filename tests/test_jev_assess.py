# tests/test_jev_assess.py
import logging
import time
import unicodedata
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tests.jev_fakes import FakeJevClient
from xbrain.evidence import evidence_surfaces
from xbrain.jev.assess import (
    RunResult,
    Selection,
    assess_topics,
    assessment_is_current,
    build_topic_state,
    parse_topic_result,
    questions_digest,
    run_assessments,
    select_items,
    topic_contract,
)
from xbrain.jev.client import ChoiceAnswer, ChoiceQuestion, JevError, NoulAnswer, NoulQuestion
from xbrain.jev.questions import PRIMARY_KEY, build_topic_questions
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.verification import fingerprint_output

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)


def _vocab():
    return [
        Topic(slug="ai-coding", description="Construir software con IA."),
        Topic(slug="startups", description="Fundar y financiar empresas."),
    ]


def _questions(fallback: str = "otro"):
    return build_topic_questions(_vocab(), fallback)


def _item(
    item_id: str = "1",
    text: str = "Claude Code ships hooks",
    topics=("ai-coding",),
    author: Author | None = None,
) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=author if author is not None else Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="Resumen.",
            primary_topic=topics[0],
            topics=list(topics),
        ),
    )


# --------------------------------------------------------------------------- state


def test_the_post_text_leads_the_state_so_a_cut_can_never_drop_it():
    """`evidence_surfaces` puts the tweet LAST; the state puts it FIRST.

    Measured on the real corpus, 7 of 7 items over the 100k default lost their own tweet to
    `text[:char_limit]` — Jev was asked "is the post in `post` about X" about a `post` that
    no longer contained the post, and answered confidently.
    """
    item = _item()
    assert [surface.key for surface in evidence_surfaces(item, "topics")] == ["author", "tweet"]
    state, state_chars = build_topic_state(item, char_limit=100_000)
    assert state["post"] == "\n".join(["Claude Code ships hooks", "alice", "Alice"])
    assert state["post"].startswith(item.text)
    assert state_chars == len(state["post"])


def test_the_state_signposts_what_was_cut_and_reports_the_pre_cut_length():
    item = _item(text="x" * 50)
    # tweet (50) + "\n" + "alice" + "\n" + "Alice" = 62; the author surface is evidence too.
    assert len(build_topic_state(item, char_limit=100_000)[0]["post"]) == 62
    state, state_chars = build_topic_state(item, char_limit=10)
    # PRE-cut: how much evidence EXISTED. The post-cut length saturates at `char_limit`, so
    # for exactly the items that were cut it is the one number that cannot say how much went.
    assert state_chars == 62
    assert state["post"] == "x" * 10 + "\n[… evidencia recortada: 52 caracteres omitidos …]"
    assessment = assess_topics(item, _questions(), FakeJevClient(), char_limit=10)
    assert assessment.truncated is True
    assert assessment.state_chars == 62


def test_evidence_exactly_at_the_limit_is_not_truncated():
    item = _item(text="x" * 50)
    state, state_chars = build_topic_state(item, char_limit=62)
    assert state_chars == 62
    assert "recortada" not in state["post"]
    assert assess_topics(item, _questions(), FakeJevClient(), char_limit=62).truncated is False


# --------------------------------------------------------------------------- contract


def test_contract_is_a_stable_golden_vector():
    """Changing this hex is a DELIBERATE retirement of every stored assessment.

    Pinned so a prompt edit, a criterion edit or a change to the hash's composition shows up
    as an edit to this line, and not as a silent run that re-pays for the whole corpus.

    A pin nobody has seen fail is a decoration: the three tests below demonstrate, in
    process, that a reworded instruction, a changed question type and a different state
    each move this hex.
    """
    assert (
        topic_contract("post", questions_digest(_questions()))
        == "ddfe455b9eab61fa787a8ffb448a21189b7e7647fba1d0a21eb675ed5e58ae40"
    )


def test_contract_moves_when_only_the_instruction_wording_changes():
    """The whole justification for hashing the questions instead of the vocabulary."""
    questions = _questions()
    reworded = dict(questions)
    primary = reworded[PRIMARY_KEY]
    reworded[PRIMARY_KEY] = replace(primary, instructions=primary.instructions + " Be strict.")
    assert topic_contract("post", questions_digest(reworded)) != topic_contract(
        "post", questions_digest(questions)
    )


def test_contract_is_independent_of_vocabulary_order():
    """Reshuffling `vocab.yaml` must not retire the corpus — and now cannot ASK differently
    either, because `build_topic_questions` is canonical."""
    vocab = _vocab()
    assert questions_digest(build_topic_questions(vocab, "otro")) == questions_digest(
        build_topic_questions([vocab[1], vocab[0]], "otro")
    )


def test_the_question_type_is_inside_the_digest():
    """`asdict` keeps the fields, not the class: without the type, a Noul and a Choice with
    equal fields would hash alike and a key changing type would not retire its contract."""
    fields = {"instructions": "I", "criteria": {"true": "A", "false": "B"}}
    assert questions_digest({"k": NoulQuestion(**fields)}) != questions_digest(
        {"k": ChoiceQuestion(**fields)}
    )


def test_the_digest_is_unicode_normalised():
    """NFC and NFD of the same visible text must not be two different asks — an editor that
    rewrites `vocab.yaml` in another normal form would otherwise re-pay for the corpus."""
    composed, decomposed = (unicodedata.normalize(form, "Descripción.") for form in ("NFC", "NFD"))
    assert composed != decomposed
    assert questions_digest(build_topic_questions([Topic(slug="a", description=composed)], "otro"))
    assert questions_digest(
        build_topic_questions([Topic(slug="a", description=composed)], "otro")
    ) == questions_digest(build_topic_questions([Topic(slug="a", description=decomposed)], "otro"))
    assert topic_contract(composed, "d" * 64) == topic_contract(decomposed, "d" * 64)


# --------------------------------------------------------------------------- assess_topics


def test_assess_topics_asks_on_the_evidence_and_stamps_the_contract():
    item, questions = _item(), _questions()
    client = FakeJevClient(nouls={"ai-coding": 0.97}, primary="ai-coding")
    assessment = assess_topics(item, questions, client, char_limit=100_000, now=DT)
    state, asked = client.calls[0]
    assert state == build_topic_state(item, 100_000)[0]
    # What was SENT is what the digest below is taken over.
    assert asked == questions
    assert assessment.membership == {"ai-coding": 0.97, "startups": 0.05}
    assert assessment.primary.choice == "ai-coding"
    assert assessment.primary.confidence == 0.9
    assert assessment.primary.probabilities == {"ai-coding": 1.0, "startups": 0.0, "otro": 0.0}
    assert assessment.provider == "fake"
    assert assessment.model == "jev-1.13.0"
    assert assessment.output_fingerprint == fingerprint_output(item, "topics")
    assert assessment.contract == topic_contract(state["post"], questions_digest(questions))
    assert assessment.truncated is False
    assert assessment.state_chars == len(state["post"])
    assert (assessment.input_tokens, assessment.output_tokens) == (100, 10)
    assert assessment.asked_at == DT


def test_an_un_enriched_item_stores_no_output_fingerprint():
    """`output_fingerprint` is informational — which enrich assignment existed at ask time —
    and is never consulted for currency, so `None` here is a record, not a staleness signal.
    """
    bare = _item().model_copy(update={"enriched": None})
    assessment = assess_topics(bare, _questions(), FakeJevClient(), char_limit=100)
    assert assessment.output_fingerprint is None
    assert assessment.membership  # the item is still assessable: evidence is raw surfaces


def test_a_record_the_model_refuses_becomes_a_jev_error_naming_the_field():
    """`assess_topics` is TOTAL with respect to `JevError`: no pydantic banner reaches a
    caller, including the single-item path that has no batch around it."""

    class _OutOfRange(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            result.answers["topic__ai-coding"] = NoulAnswer(noul=1.7)
            return result

    with pytest.raises(JevError, match="el registro rechaza: membership.ai-coding") as caught:
        assess_topics(_item(), _questions(), _OutOfRange(), char_limit=100)
    assert "\n" not in str(caught.value)


# --------------------------------------------------------------------------- parse


def test_parse_reads_the_questions_actually_asked_not_the_vocabulary():
    """Membership keys come from the Noul question keys, so the parser and the contract can
    never describe different asks. An answer to a question nobody asked is ignored."""
    questions = _questions()
    client = FakeJevClient(nouls={"ai-coding": 0.9})
    result = client.ask({"post": "x"}, questions)
    result.answers["topic__ghost"] = NoulAnswer(noul=1.0)
    membership, primary = parse_topic_result(result, questions)
    assert membership == {"ai-coding": 0.9, "startups": 0.05}
    assert primary.choice == "otro"


def test_parse_rejects_a_missing_noul():
    class _MissingNoul(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            del result.answers["topic__startups"]
            return result

    with pytest.raises(JevError, match="no contestó el noul de 'startups'"):
        assess_topics(_item(), _questions(), _MissingNoul(), char_limit=100)


def test_parse_rejects_a_noul_answered_with_the_wrong_answer_type():
    class _ChoiceForANoul(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            result.answers["topic__startups"] = ChoiceAnswer(
                choice="x", confidence=1.0, probabilities={"x": 1.0}
            )
            return result

    with pytest.raises(JevError, match="no contestó el noul de 'startups'"):
        assess_topics(_item(), _questions(), _ChoiceForANoul(), char_limit=100)


def test_parse_rejects_a_missing_primary_answer():
    """Without this guard `None.choice` raises `AttributeError` — which is not the `JevError`
    the seam promises, so a direct caller gets a traceback instead of the operator's message.
    """

    class _NoPrimary(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            del result.answers["primary"]
            return result

    with pytest.raises(JevError, match="no contestó la pregunta 'primary'"):
        assess_topics(_item(), _questions(), _NoPrimary(), char_limit=100)


def test_parse_rejects_a_choice_outside_the_offered_options():
    with pytest.raises(JevError, match="no está entre las opciones"):
        assess_topics(_item(), _questions(), FakeJevClient(primary="banana"), char_limit=100)


def test_parse_rejects_a_choice_missing_from_its_own_distribution():
    class _ChoiceOffItsOwnDistribution(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            choice = result.answers["primary"]
            del choice.probabilities[choice.choice]
            return result

    with pytest.raises(JevError, match="no está en su propia distribución"):
        assess_topics(_item(), _questions(), _ChoiceOffItsOwnDistribution(), char_limit=100)


def test_parse_rejects_a_winner_its_own_distribution_does_not_favour():
    """Membership is not enough: a winner sitting at 0.0 while a loser holds 1.0 scores zero
    for every reader using the `.get(option, 0.0)` access the record prescribes — the exact
    outcome the presence check was written to prevent, one notch short."""

    class _LosingWinner(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            choice = result.answers["primary"]
            result.answers["primary"] = replace(
                choice, probabilities={**choice.probabilities, choice.choice: 0.0, "startups": 1.0}
            )
            return result

    with pytest.raises(JevError, match="por debajo del máximo"):
        assess_topics(_item(), _questions(), _LosingWinner(), char_limit=100)


def test_parse_accepts_a_tie_for_the_argmax():
    class _Tied(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            choice = result.answers["primary"]
            result.answers["primary"] = replace(
                choice, probabilities={**choice.probabilities, choice.choice: 0.5, "startups": 0.5}
            )
            return result

    assessment = assess_topics(_item(), _questions(), _Tied(), char_limit=100)
    assert assessment.primary.probabilities["startups"] == 0.5


def test_an_out_of_range_confidence_is_wrapped_like_any_other_refusal():
    """`PrimaryChoice` is built inside `parse_topic_result`, not inside `assess_topics`' try.

    A confidence above 1 therefore escaped as a raw pydantic banner and landed in
    `RunResult.failed` as four lines of English — the exact shape the Spanish wrapper exists
    to keep out of the operator's failure table.
    """

    class _OverConfident(FakeJevClient):
        def ask(self, state, questions):
            result = super().ask(state, questions)
            choice = result.answers["primary"]
            result.answers["primary"] = replace(choice, confidence=1.7)
            return result

    with pytest.raises(JevError, match="el registro rechaza: confidence") as caught:
        assess_topics(_item(), _questions(), _OverConfident(), char_limit=100)
    assert "\n" not in str(caught.value)
    # `parse_topic_result` is public and promises `JevError`; a direct caller gets it too.
    with pytest.raises(JevError, match="el registro rechaza: confidence"):
        parse_topic_result(_OverConfident().ask({"post": "x"}, _questions()), _questions())


def test_a_question_map_without_the_primary_choice_is_a_programmer_error():
    """`ValueError`, not `JevError`: the provider did nothing wrong. The caller handed over a
    question map `build_topic_questions` could not have produced, and an operator-facing Jev
    message would send whoever reads `failed` looking at the API."""
    questions = _questions()
    result = FakeJevClient().ask({"post": "x"}, questions)
    noul_only = {key: q for key, q in questions.items() if key != PRIMARY_KEY}
    with pytest.raises(ValueError, match="no incluye la Choice 'primary'"):
        parse_topic_result(result, noul_only)


# --------------------------------------------------------------------------- currency


def test_currency_tracks_vocab_and_evidence_but_not_re_enrichment():
    item, vocab = _item(), _vocab()
    assessment = assess_topics(item, _questions(), FakeJevClient(), char_limit=100)
    kw = {"fallback": "otro", "char_limit": 100}
    assert assessment_is_current(assessment, item, vocab, **kw)
    assert assessment_is_current(assessment, _item(topics=("startups",)), vocab, **kw)
    changed_vocab = [Topic(slug="ai-coding", description="Otra descripción."), vocab[1]]
    assert not assessment_is_current(assessment, item, changed_vocab, **kw)
    assert not assessment_is_current(assessment, _item(text="otro texto"), vocab, **kw)
    assert not assessment_is_current(assessment, item, vocab, fallback="ninguno", char_limit=100)


def test_a_truncated_item_stays_current_until_the_window_changes():
    """Without this, an item longer than `state_char_limit` is never current: re-picked,
    re-asked and re-billed on every run, and the run looks like it simply had work to do."""
    item, vocab = _item(text="x" * 500), _vocab()
    assessment = assess_topics(item, _questions(), FakeJevClient(), char_limit=20)
    assert assessment.truncated is True
    kw = {"fallback": "otro", "char_limit": 20}
    assert assessment_is_current(assessment, item, vocab, **kw)
    assert (
        select_items(
            {item.id: item}, {item.id: assessment}, vocab, ids=None, limit=None, force=False, **kw
        ).items
        == ()
    )
    # A different window is a different ask, in BOTH directions.
    assert not assessment_is_current(assessment, item, vocab, fallback="otro", char_limit=10)
    assert not assessment_is_current(assessment, item, vocab, fallback="otro", char_limit=100_000)


# --------------------------------------------------------------------------- select_items


def _store():
    return {
        "1": _item("1"),
        "2": _item("2", text="Fundraising tips"),
        # No handle and blank text: the author surface is evidence too, so an item with only
        # a handle would NOT be evidence-free.
        "3": _item("3", text="   ", author=Author(handle="", name="")),
    }


def test_select_items_skips_current_unless_forced_and_counts_what_it_skipped():
    vocab, store = _vocab(), _store()
    assert build_topic_state(store["3"], 100)[1] == 0
    kw = {"fallback": "otro", "char_limit": 100}
    current = assess_topics(store["1"], _questions(), FakeJevClient(), char_limit=100)
    assessments = {"1": current}
    picked = select_items(store, assessments, vocab, ids=None, limit=None, force=False, **kw)
    assert isinstance(picked, Selection)
    assert [item.id for item in picked.items] == ["2"]
    # The counts are the whole point: a silent funnel (evidence regressed, everything skips)
    # would otherwise be indistinguishable from a clean "nothing to do".
    assert (picked.skipped_current, picked.skipped_no_evidence) == (1, 1)
    forced = select_items(store, assessments, vocab, ids=None, limit=None, force=True, **kw)
    assert [item.id for item in forced.items] == ["1", "2"]
    # `force` does NOT override the evidence skip — there is nothing to ask about.
    assert (forced.skipped_current, forced.skipped_no_evidence) == (0, 1)
    limited = select_items(store, assessments, vocab, ids=None, limit=1, force=True, **kw)
    assert [item.id for item in limited.items] == ["1"]


def test_select_items_honours_the_order_asked_and_dedupes_repeats():
    """A repeated id would otherwise be asked, PAID for and recorded once per repetition,
    producing two records under one `item_id` for the side-car to reconcile silently."""
    vocab, store = _vocab(), _store()
    kw = {"fallback": "otro", "char_limit": 100}
    picked = select_items(store, {}, vocab, ids=["2", "1", "2"], limit=None, force=False, **kw)
    assert [item.id for item in picked.items] == ["2", "1"]


def test_select_items_rejects_unknown_ids_and_bad_limits():
    vocab, store = _vocab(), _store()
    kw = {"fallback": "otro", "char_limit": 100}
    with pytest.raises(JevError, match="ids desconocidos: 9, 8"):
        select_items(store, {}, vocab, ids=["9", "8", "9"], limit=None, force=False, **kw)
    for bad in (0, -1):
        with pytest.raises(JevError, match=r"--limit debe ser >= 1"):
            select_items(store, {}, vocab, ids=None, limit=bad, force=True, **kw)


def test_an_explicitly_named_id_is_counted_never_an_error():
    """Naming an id does NOT imply `--force`, and naming an evidence-free one is not a
    failure — both are counted so the CLI can say why nothing happened."""
    vocab, store = _vocab(), _store()
    kw = {"fallback": "otro", "char_limit": 100}
    current = assess_topics(store["1"], _questions(), FakeJevClient(), char_limit=100)
    by_id = select_items(store, {"1": current}, vocab, ids=["1"], limit=None, force=False, **kw)
    assert by_id.items == () and by_id.skipped_current == 1
    blank = select_items(store, {}, vocab, ids=["3"], limit=None, force=True, **kw)
    assert blank.items == () and blank.skipped_no_evidence == 1


def test_an_empty_id_list_means_the_whole_corpus():
    vocab, store = _vocab(), _store()
    picked = select_items(
        store, {}, vocab, ids=[], limit=None, force=False, fallback="otro", char_limit=100
    )
    assert [item.id for item in picked.items] == ["1", "2"]


# --------------------------------------------------------------------------- run_assessments


def test_run_assessments_records_failures_and_keeps_going():
    items = [_item("1"), _item("2", text="BOOM"), _item("3")]
    client = FakeJevClient(fail_when=lambda state: "BOOM" in state["post"])
    result = run_assessments(
        items, _vocab(), client, fallback="otro", char_limit=100, concurrency=2
    )
    assert isinstance(result, RunResult)
    assert [a.item_id for a in result.assessed] == ["1", "3"]
    assert result.failed == (("2", "fake failure"),)


def test_run_assessments_records_an_unexpected_exception_with_its_type():
    """Never dropped, and never filed as a provider fault: the type is stamped into the
    reason so an xbrain bug reads as one instead of as N bad answers."""

    class _Broken(FakeJevClient):
        def ask(self, state, questions):
            if "BOOM" in state["post"]:
                raise RuntimeError("el adaptador explotó")
            return super().ask(state, questions)

    result = run_assessments(
        [_item("1"), _item("2", text="BOOM")],
        _vocab(),
        _Broken(),
        fallback="otro",
        char_limit=100,
        concurrency=2,
    )
    assert [a.item_id for a in result.assessed] == ["1"]
    assert result.failed == (("2", "RuntimeError: el adaptador explotó"),)


def test_results_are_sorted_by_item_id_whatever_the_pool_finished_first():
    items = [_item("3"), _item("1", text="BOOM"), _item("2"), _item("4", text="BOOM")]
    result = run_assessments(
        items,
        _vocab(),
        FakeJevClient(fail_when=lambda state: "BOOM" in state["post"]),
        fallback="otro",
        char_limit=100,
        concurrency=4,
    )
    assert [a.item_id for a in result.assessed] == ["2", "3"]
    assert [item_id for item_id, _ in result.failed] == ["1", "4"]


def test_run_assessments_raises_when_every_item_fails():
    client = FakeJevClient(fail_when=lambda state: True)
    with pytest.raises(JevError, match="ninguna de las 2 evaluaciones"):
        run_assessments(
            [_item("1"), _item("2")],
            _vocab(),
            client,
            fallback="otro",
            char_limit=100,
            concurrency=2,
        )


def test_the_all_failed_rule_covers_an_unexpected_exception_too():
    """`failed[0]` must always exist when `assessed` is empty.

    A catch-all branch that forgot to append would leave both lists empty and turn the
    all-failed raise into an `IndexError` — a dead provider reported as a crash.
    """

    class _AlwaysBroken(FakeJevClient):
        def ask(self, state, questions):
            raise RuntimeError("cayó el adaptador")

    with pytest.raises(JevError, match="ninguna de las 2 evaluaciones"):
        run_assessments(
            [_item("1"), _item("2")],
            _vocab(),
            _AlwaysBroken(),
            fallback="otro",
            char_limit=100,
            concurrency=2,
        )


def test_an_empty_run_is_an_empty_result_not_a_dead_provider():
    client = FakeJevClient()
    result = run_assessments([], _vocab(), client, fallback="otro", char_limit=100, concurrency=2)
    assert result == RunResult(assessed=(), failed=())
    assert client.calls == []


def test_run_assessments_validates_questions_before_any_call():
    client = FakeJevClient()
    with pytest.raises(ValueError, match="choca"):
        run_assessments(
            [_item()], _vocab(), client, fallback="startups", char_limit=100, concurrency=1
        )
    assert client.calls == []


def test_progress_counts_failures_too_and_is_reported_from_the_main_thread():
    """Reported from the `as_completed` loop, not from the workers, so the sequence is
    monotonic whatever the pool's finishing order — and a failed item still ticks."""
    seen: list[tuple[int, int]] = []
    run_assessments(
        [_item("1"), _item("2", text="BOOM"), _item("3")],
        _vocab(),
        FakeJevClient(fail_when=lambda state: "BOOM" in state["post"]),
        fallback="otro",
        char_limit=100,
        concurrency=4,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_a_failing_progress_callback_never_costs_a_record(caplog):
    """`xbrain jev topics | head` closes the pipe under a Rich console. A display failure
    must not throw away work that was already paid for."""

    def _broken(done: int, total: int) -> None:
        raise BrokenPipeError("head cerró el pipe")

    with caplog.at_level(logging.WARNING, logger="xbrain.jev.assess"):
        result = run_assessments(
            [_item("1"), _item("2")],
            _vocab(),
            FakeJevClient(),
            fallback="otro",
            char_limit=100,
            concurrency=2,
            on_progress=_broken,
        )
    assert [a.item_id for a in result.assessed] == ["1", "2"]
    # The warning has to say WHAT broke: "on_progress falló" alone sends nobody anywhere.
    assert "BrokenPipeError" in caplog.text
    assert "head cerró el pipe" in caplog.text


def test_an_interrupt_cancels_the_queued_calls_instead_of_paying_for_them():
    """Every item is submitted up front, so a plain `shutdown(wait=True)` would drain the
    whole queue — the operator's Ctrl-C would not interrupt anything and the full bill would
    still arrive. The records already collected ARE lost: that is what an interrupt means."""

    class _Interrupting(FakeJevClient):
        def ask(self, state, questions):
            if "BOOM" in state["post"]:
                raise KeyboardInterrupt
            time.sleep(0.05)
            return super().ask(state, questions)

    items = [_item("00", text="BOOM")] + [_item(f"{n:02d}") for n in range(1, 30)]
    client = _Interrupting()
    with pytest.raises(KeyboardInterrupt):
        run_assessments(items, _vocab(), client, fallback="otro", char_limit=100, concurrency=2)
    assert len(client.calls) < len(items)
