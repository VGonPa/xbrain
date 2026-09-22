# tests/test_jev_client.py
from datetime import datetime, timezone

import pytest
from typesafe_sdk import Choice, Noul, SystemOneResponse, TypeSafeError, Usage
from typesafe_sdk import ChoiceAnswer as SdkChoiceAnswer
from typesafe_sdk import NoulAnswer as SdkNoulAnswer
from typesafe_sdk import ScoreAnswer as SdkScoreAnswer

from xbrain.jev.client import (
    ChoiceAnswer,
    ChoiceQuestion,
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
