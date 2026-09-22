# tests/test_jev_client.py
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from typesafe_sdk import Choice, Noul, SystemOneResponse, TypeSafeError, Usage
from typesafe_sdk import ChoiceAnswer as SdkChoiceAnswer
from typesafe_sdk import NoulAnswer as SdkNoulAnswer
from typesafe_sdk import ScoreAnswer as SdkScoreAnswer

from tests.jev_fakes import FakeJevClient
from xbrain.jev.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevClient,
    JevError,
    NoulAnswer,
    NoulQuestion,
    TypeSafeJevClient,
)
from xbrain.jev.models import PrimaryChoice, TopicAssessment

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
SHA = "0" * 64


class _FakeSdk:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def system_one(self, state, questions, *, model=None):
        self.calls.append((state, questions, model))
        if self.error is not None:
            raise self.error
        return self.response


def _response(**answers):
    return SystemOneResponse(
        model="jev-1.13.0", usage=Usage(input_tokens=321, output_tokens=7), answers=answers
    )


_QUESTIONS = {
    "topic__ai": NoulQuestion("Is it about ai?", {"true": "AI", "false": "not"}),
    "primary": ChoiceQuestion("Main?", {"ai": "AI", "otro": "other"}),
}


def test_ask_maps_sdk_answers_and_usage():
    sdk = _FakeSdk(
        _response(
            topic__ai=SdkNoulAnswer(noul=0.91),
            primary=SdkChoiceAnswer(
                choice="ai", confidence=0.8, probabilities={"ai": 0.8, "otro": 0.2}
            ),
        )
    )
    client = TypeSafeJevClient(api_key="k", model="jev-latest", sdk_client=sdk)
    result = client.ask({"post": "hola"}, _QUESTIONS)
    assert result.model == "jev-1.13.0"
    assert result.answers["topic__ai"] == NoulAnswer(noul=0.91)
    assert result.answers["primary"] == ChoiceAnswer(
        choice="ai", confidence=0.8, probabilities={"ai": 0.8, "otro": 0.2}
    )
    assert (result.input_tokens, result.output_tokens) == (321, 7)


def test_ask_sends_sdk_question_objects_state_and_model():
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.5)))
    client = TypeSafeJevClient(api_key="k", model="jev-latest", sdk_client=sdk)
    client.ask({"post": "hola"}, {"topic__ai": _QUESTIONS["topic__ai"]})
    state, questions, model = sdk.calls[0]
    assert state == {"post": "hola"}
    assert model == "jev-latest"
    assert isinstance(questions["topic__ai"], Noul)
    assert questions["topic__ai"].instructions == "Is it about ai?"
    assert questions["topic__ai"].criteria == {"true": "AI", "false": "not"}


def test_ask_sends_choice_criteria_verbatim():
    sdk = _FakeSdk(
        _response(primary=SdkChoiceAnswer(choice="ai", confidence=0.9, probabilities={"ai": 1.0}))
    )
    client = TypeSafeJevClient(api_key="k", model="jev-latest", sdk_client=sdk)
    client.ask({"post": "x"}, {"primary": _QUESTIONS["primary"]})
    _, questions, _ = sdk.calls[0]
    assert isinstance(questions["primary"], Choice)
    assert questions["primary"].criteria == {"ai": "AI", "otro": "other"}


def test_ask_wraps_sdk_errors_as_jev_error():
    client = TypeSafeJevClient(
        api_key="k", model="m", sdk_client=_FakeSdk(error=TypeSafeError("boom"))
    )
    with pytest.raises(JevError, match="boom"):
        client.ask({"post": "x"}, _QUESTIONS)


def test_ask_rejects_answer_types_that_were_not_asked():
    sdk = _FakeSdk(
        _response(
            x=SdkScoreAnswer(
                score=1.0, confidence=0.5, legend={0: "a", 1: "b"}, probabilities={0: 0.0, 1: 1.0}
            )
        )
    )
    client = TypeSafeJevClient(api_key="k", model="m", sdk_client=sdk)
    with pytest.raises(JevError, match="score"):
        client.ask({"post": "x"}, _QUESTIONS)


def test_topic_assessment_round_trips_json_and_requires_aware_datetime():
    assessment = TopicAssessment(
        item_id="1",
        model="jev-1.13.0",
        asked_at=DT,
        output_fingerprint=None,
        contract=SHA,
        state_chars=12,
        membership={"ai": 0.9},
        primary=PrimaryChoice(choice="ai", confidence=0.8, probabilities={"ai": 0.8, "otro": 0.2}),
    )
    restored = TopicAssessment.model_validate(assessment.model_dump(mode="json"))
    assert restored == assessment
    assert restored.provider == "typesafe"
    assert restored.truncated is False
    with pytest.raises(ValueError, match="UTC-aware"):
        TopicAssessment.model_validate(
            {**assessment.model_dump(mode="json"), "asked_at": "2026-09-22T00:00:00"}
        )


def _assessment(**overrides):
    """A valid `TopicAssessment`, with the named fields replaced."""
    fields = {
        "item_id": "1",
        "model": "jev-1.13.0",
        "asked_at": DT,
        "output_fingerprint": None,
        "contract": SHA,
        "state_chars": 12,
        "membership": {"ai": 0.9},
        "primary": PrimaryChoice(
            choice="ai", confidence=0.8, probabilities={"ai": 0.8, "otro": 0.2}
        ),
    }
    return TopicAssessment(**{**fields, **overrides})


def test_ask_rejects_a_response_missing_an_asked_question():
    """The SDK drops answers whose type it does not model, so a short answer set is
    reported instead of silently becoming a topic with no membership."""
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.91)))
    client = TypeSafeJevClient(api_key="k", model="m", sdk_client=sdk)
    with pytest.raises(JevError, match="primary"):
        client.ask({"post": "x"}, _QUESTIONS)


def test_ask_maps_a_noul_question_with_partial_criteria():
    """`NoulCriteria` is a total=False TypedDict: one side may be left undescribed."""
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.5)))
    client = TypeSafeJevClient(api_key="k", model="m", sdk_client=sdk)
    client.ask({"post": "x"}, {"topic__ai": NoulQuestion("Is it about ai?", {"true": "AI"})})
    _, questions, _ = sdk.calls[0]
    assert questions["topic__ai"].criteria == {"true": "AI", "false": None}


def test_asked_at_is_normalized_to_utc():
    assessment = _assessment(
        asked_at=datetime(2026, 9, 22, 12, tzinfo=timezone(timedelta(hours=2)))
    )
    assert assessment.asked_at == datetime(2026, 9, 22, 10, tzinfo=timezone.utc)
    assert assessment.asked_at.utcoffset() == timedelta(0)


@pytest.mark.parametrize("field", ["membership", "primary"])
def test_probabilities_outside_zero_to_one_are_rejected(field: str):
    bad = (
        {"membership": {"ai": 1.7}}
        if field == "membership"
        else {"primary": {"choice": "ai", "confidence": 0.8, "probabilities": {"ai": 1.7}}}
    )
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        _assessment(**bad)


def test_fake_jev_client_answers_as_configured_and_records_calls():
    """The shared fake is exercised here because mypy does not cover `tests/`."""
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
    assert result.model == "jev-1.13.0"
    assert client.calls == [({"post": "hola"}, questions)]


def test_fake_jev_client_can_fail_per_item():
    client = FakeJevClient(fail_when=lambda state: state["post"] == "boom")
    assert client.ask({"post": "ok"}, {}).answers == {}
    with pytest.raises(JevError, match="fake failure"):
        client.ask({"post": "boom"}, {})
