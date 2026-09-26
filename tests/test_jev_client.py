# tests/test_jev_client.py — the vendor-free seam: the persisted record and the test double.
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tests.jev_fakes import FakeJevClient
from xbrain.jev.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    CountingJevClient,
    JevClient,
    JevError,
    NoulAnswer,
    NoulQuestion,
)
from xbrain.jev.env import typesafe_api_key
from xbrain.jev.models import PrimaryChoice, TopicAssessment

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
CONTRACT = hashlib.sha256(b"state||vocab||otro").hexdigest()


def _assessment(**overrides) -> TopicAssessment:
    """A valid `TopicAssessment`, with the named fields replaced."""
    fields = {
        "item_id": "1",
        "provider": "typesafe",
        "model": "jev-1.13.0",
        "asked_at": DT,
        "output_fingerprint": None,
        "contract": CONTRACT,
        "state_chars": 12,
        "membership": {"ai": 0.9},
        "primary": PrimaryChoice(
            choice="ai", confidence=0.8, probabilities={"ai": 0.8, "otro": 0.2}
        ),
    }
    return TopicAssessment(**{**fields, **overrides})


# --------------------------------------------------------------------------- the record


def test_topic_assessment_round_trips_json():
    assessment = _assessment()
    restored = TopicAssessment.model_validate(assessment.model_dump(mode="json"))
    assert restored == assessment
    assert restored.provider == "typesafe"
    assert restored.truncated is False


def test_provider_is_required_so_a_record_cannot_default_its_own_provenance():
    """Provenance is a fact the judge reports, never a value the record assumes."""
    payload = _assessment().model_dump(mode="json")
    del payload["provider"]
    with pytest.raises(ValidationError, match="provider"):
        TopicAssessment.model_validate(payload)


def test_a_stored_record_is_frozen():
    """The `contract` is computed over the record's inputs; a record mutated afterwards
    would carry a contract that describes something else."""
    with pytest.raises(ValidationError):
        _assessment().item_id = "2"


def test_an_unknown_field_is_refused_rather_than_silently_dropped():
    """A record written by a newer xbrain must not lose fields when an older one reads and
    rewrites the file — `extra="ignore"` would destroy them with no error."""
    with pytest.raises(ValidationError, match="future_field"):
        TopicAssessment.model_validate(
            {**_assessment().model_dump(mode="json"), "future_field": 42}
        )


def test_contract_accepts_what_hashlib_emits():
    digest = hashlib.sha256(b"state||vocab||otro").hexdigest()
    assert _assessment(contract=digest).contract == digest


@pytest.mark.parametrize(
    "bad", ["0" * 63, "0" * 65, "A" * 64, "z" * 64, "sha256:" + "0" * 64, "", "0" * 64 + "\n"]
)
def test_contract_rejects_anything_that_is_not_a_lowercase_sha256(bad: str):
    with pytest.raises(ValidationError, match="contract"):
        _assessment(contract=bad)


def test_output_fingerprint_round_trips_a_real_fingerprint():
    digest = hashlib.sha256(b"topics output").hexdigest()
    restored = TopicAssessment.model_validate(
        _assessment(output_fingerprint=digest).model_dump(mode="json")
    )
    assert restored.output_fingerprint == digest


@pytest.mark.parametrize("bad", ["not-a-sha", "", "A" * 64, "0" * 63])
def test_output_fingerprint_rejects_a_value_no_fingerprint_could_produce(bad: str):
    """It is the staleness key: a malformed stamp can never equal a fresh digest, so the
    "assessment is stale" signal would silently never fire."""
    with pytest.raises(ValidationError, match="output_fingerprint"):
        _assessment(output_fingerprint=bad)


@pytest.mark.parametrize("offset_hours", [2, -5])
def test_asked_at_refuses_an_offset_that_is_not_utc(offset_hours: int):
    """The repo does not coerce a wrong instant into a right-looking one — that masks the
    bug. Every persisted instant is UTC, so string forms compare."""
    aware_elsewhere = datetime(2026, 9, 22, 12, tzinfo=timezone(timedelta(hours=offset_hours)))
    with pytest.raises(ValidationError, match="UTC"):
        _assessment(asked_at=aware_elsewhere)


def test_asked_at_accepts_an_explicit_zero_offset():
    stored = _assessment(asked_at=datetime(2026, 9, 22, 12, tzinfo=timezone.utc)).asked_at
    assert stored.utcoffset() == timedelta(0)


def test_asked_at_refuses_a_naive_datetime():
    with pytest.raises(ValidationError, match="timezone-aware"):
        TopicAssessment.model_validate(
            {**_assessment().model_dump(mode="json"), "asked_at": "2026-09-22T00:00:00"}
        )


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"membership": {"ai": 1.7}}, "less than or equal to 1"),
        ({"membership": {"ai": -0.1}}, "greater than or equal to 0"),
        (
            {"primary": {"choice": "ai", "confidence": 1.7, "probabilities": {"ai": 1.0}}},
            "less than or equal to 1",
        ),
        (
            {"primary": {"choice": "ai", "confidence": 0.8, "probabilities": {"ai": -0.1}}},
            "greater than or equal to 0",
        ),
        ({"state_chars": -1}, "greater than or equal to 0"),
        ({"input_tokens": -5}, "greater than or equal to 0"),
        ({"output_tokens": -5}, "greater than or equal to 0"),
    ],
)
def test_out_of_range_numbers_are_rejected(overrides: dict, needle: str):
    with pytest.raises(ValidationError, match=needle):
        _assessment(**overrides)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_membership_that_is_not_a_real_probability_is_rejected(bad: float):
    """A NaN membership compares False against every threshold — silently absent forever."""
    with pytest.raises(ValidationError):
        _assessment(membership={"ai": bad})


def test_an_assessment_covering_no_topics_is_refused():
    """ "No topics asked" and "every topic scored zero" render identically downstream."""
    with pytest.raises(ValidationError, match="membership"):
        _assessment(membership={})


def test_the_primary_choice_must_appear_in_its_own_distribution():
    """A later `probabilities[choice]` is a KeyError; a `.get(choice, 0.0)` is a silent 0."""
    with pytest.raises(ValidationError, match="probabilities"):
        PrimaryChoice(choice="ai", confidence=0.9, probabilities={"otro": 1.0})


@pytest.mark.parametrize("field", ["item_id", "provider", "model"])
def test_the_identifying_strings_cannot_be_empty(field: str):
    with pytest.raises(ValidationError, match=field):
        _assessment(**{field: "  "})


# --------------------------------------------------------------------------- the double


def test_fake_jev_client_answers_as_configured_and_records_calls():
    client: JevClient = FakeJevClient(nouls={"ai": 0.93}, primary="ai", confidence=0.7)
    questions = {
        "topic__ai": NoulQuestion("Is it about ai?"),
        "topic__other": NoulQuestion("Is it about other?"),
        "primary": ChoiceQuestion("Main?", {"ai": "AI", "otro": "other"}),
    }
    result = client.ask({"post": "hola"}, questions)
    assert result.answers["topic__ai"] == NoulAnswer(noul=0.93)
    assert result.answers["topic__other"] == NoulAnswer(noul=0.05)
    assert result.answers["primary"] == ChoiceAnswer(
        choice="ai", confidence=0.7, probabilities={"ai": 1.0, "otro": 0.0}
    )
    assert (result.provider, result.model) == ("fake", "jev-1.13.0")
    assert result.answers.keys() == questions.keys()


def test_fake_jev_client_records_a_snapshot_not_a_reference():
    """Task 2 builds `questions` per item in a loop; a log holding the caller's dict would
    show the final state for every recorded call and the assertion would pass anyway."""
    client = FakeJevClient()
    state = {"post": "first"}
    questions = {"topic__ai": NoulQuestion("Is it about ai?")}
    client.ask(state, questions)
    state["post"] = "mutated"
    questions["topic__ai"] = NoulQuestion("rewritten")
    assert client.calls == [({"post": "first"}, {"topic__ai": NoulQuestion("Is it about ai?")})]


def test_fake_jev_client_can_fail_per_item():
    client = FakeJevClient(fail_when=lambda state: state["post"] == "boom")
    questions = {"topic__ai": NoulQuestion("Is it about ai?")}
    assert client.ask({"post": "ok"}, questions).answers.keys() == questions.keys()
    with pytest.raises(JevError, match="fake failure"):
        client.ask({"post": "boom"}, questions)


def test_fake_jev_client_refuses_an_empty_question_map_like_the_real_one():
    with pytest.raises(JevError, match="sin preguntas"):
        FakeJevClient().ask({"post": "x"}, {})


def test_fake_jev_client_can_report_unreported_usage():
    """The real client returns `None` when the API reports no usage; a cost consumer has to
    be able to meet that path with the double."""
    client = FakeJevClient(input_tokens=None, output_tokens=None)
    result = client.ask({"post": "x"}, {"topic__ai": NoulQuestion("q")})
    assert (result.input_tokens, result.output_tokens) == (None, None)


def test_fake_jev_client_may_answer_an_option_outside_the_criteria():
    """Deliberate, and depended upon: it is how task 2 drives the "the primary must be a
    vocabulary slug or the fallback" guard."""
    client = FakeJevClient(primary="banana")
    answer = client.ask({"post": "x"}, {"primary": ChoiceQuestion("Main?", {"ai": "AI"})})
    assert answer.answers["primary"] == ChoiceAnswer(
        choice="banana", confidence=0.9, probabilities={"ai": 0.0, "banana": 1.0}
    )


def test_the_suite_cannot_reach_a_real_key(tmp_path):
    """`conftest._isolate_typesafe_credentials` is autouse and this module does not opt
    out, so neither the environment variable nor a `.env` beside the caller is visible.
    Asserted rather than assumed: the failure it prevents is a green suite spending money
    against the live API."""
    (tmp_path / ".env").write_text(
        "TYPESAFE_API_KEY=ts-would-cost-money\n",  # pragma: allowlist secret
        encoding="utf-8",
    )
    assert typesafe_api_key(tmp_path) is None


def test_fake_jev_client_snapshots_deeply_enough_to_survive_a_mutated_question():
    """A shallow `dict(questions)` still shares each question's `criteria` dict, so a caller
    that edits one in place would rewrite history the log already recorded."""
    client = FakeJevClient()
    question = NoulQuestion("Is it about ai?", {"true": "sobre IA"})
    client.ask({"post": "first"}, {"topic__ai": question})
    question.criteria["true"] = "MUTATED"
    assert client.calls[0][1]["topic__ai"].criteria == {"true": "sobre IA"}


def test_fake_jev_client_refuses_a_question_type_it_does_not_model():
    """`assert` is stripped under `python -O`; an unmodelled variant has to raise the same
    error the real adapter would, not vanish into a silently wrong answer set."""
    with pytest.raises(JevError, match="tipo de pregunta"):
        FakeJevClient().ask({"post": "x"}, {"q": object()})  # type: ignore[dict-item]


# --------------------------------------------------------------------------- the counting seam


def test_the_counting_seam_folds_every_answer_it_forwards():
    """Tokens, usage-less answers and models are read off EVERY returned result — including
    answers xbrain later refuses — because each of them was billed."""
    counting = CountingJevClient(FakeJevClient(input_tokens=120, model="jev-1.13.0"))
    counting.ask({"post": "a"}, {"q": NoulQuestion(instructions="?")})
    counting.ask({"post": "b"}, {"q": NoulQuestion(instructions="?")})

    counts = counting.snapshot()

    assert (counts.sent, counts.answered, counts.raised) == (2, 2, 0)
    assert counts.input_tokens_by_provider == {"fake": 240}
    assert counts.input_tokens_unknown == 0
    assert counts.models == ("jev-1.13.0",)


def test_the_counting_seam_counts_a_raised_call_and_an_answer_without_usage():
    failing = CountingJevClient(FakeJevClient(fail_when=lambda state: True))
    with pytest.raises(JevError):
        failing.ask({"post": "a"}, {"q": NoulQuestion(instructions="?")})
    silent = CountingJevClient(FakeJevClient(input_tokens=None))
    silent.ask({"post": "a"}, {"q": NoulQuestion(instructions="?")})

    assert (failing.snapshot().sent, failing.snapshot().raised) == (1, 1)
    assert failing.snapshot().input_tokens_by_provider == {}
    # The provider answered and reported no usage: its row exists at zero, and it is counted.
    assert silent.snapshot().input_tokens_by_provider == {"fake": 0}
    assert silent.snapshot().input_tokens_unknown == 1


def test_a_keyboard_interrupt_inside_a_call_is_in_flight_not_raised():
    counting = CountingJevClient(FakeJevClient(interrupt_after=0))
    with pytest.raises(KeyboardInterrupt):
        counting.ask({"post": "a"}, {"q": NoulQuestion(instructions="?")})

    counts = counting.snapshot()
    assert (counts.sent, counts.answered, counts.raised) == (1, 0, 0)
