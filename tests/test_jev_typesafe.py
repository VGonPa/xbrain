# tests/test_jev_typesafe.py — the TypeSafe adapter: every test that touches the SDK.
import json

import pytest
from httpx2 import Headers
from typesafe_sdk import (
    Choice,
    Noul,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeError,
    TypeSafeRateLimitError,
    Usage,
)
from typesafe_sdk import ChoiceAnswer as SdkChoiceAnswer
from typesafe_sdk import NoulAnswer as SdkNoulAnswer
from typesafe_sdk import ScoreAnswer as SdkScoreAnswer

from xbrain.jev.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevClient,
    JevError,
    NoulAnswer,
    NoulQuestion,
)
from xbrain.jev.typesafe import TypeSafeJevClient


class _FakeSdk:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []
        self.closed = False

    def system_one(self, state, questions, *, model=None):
        self.calls.append((state, questions, model))
        if self.error is not None:
            raise self.error
        return self.response

    def close(self):
        self.closed = True


def _response(**answers):
    return SystemOneResponse(
        model="jev-1.13.0", usage=Usage(input_tokens=321, output_tokens=7), answers=answers
    )


_QUESTIONS = {
    "topic__ai": NoulQuestion("Is it about ai?", {"true": "AI", "false": "not"}),
    "primary": ChoiceQuestion("Main?", {"ai": "AI", "otro": "other"}),
}


def _client(sdk) -> TypeSafeJevClient:
    return TypeSafeJevClient(api_key="k", model="jev-latest", sdk_client=sdk)


# --------------------------------------------------------------------------- mapping


def test_ask_maps_sdk_answers_usage_and_provenance():
    sdk = _FakeSdk(
        _response(
            topic__ai=SdkNoulAnswer(noul=0.91),
            primary=SdkChoiceAnswer(
                choice="ai", confidence=0.8, probabilities={"ai": 0.8, "otro": 0.2}
            ),
        )
    )
    result = _client(sdk).ask({"post": "hola"}, _QUESTIONS)
    assert (result.provider, result.model) == ("typesafe", "jev-1.13.0")
    assert result.answers["topic__ai"] == NoulAnswer(noul=0.91)
    assert result.answers["primary"] == ChoiceAnswer(
        choice="ai", confidence=0.8, probabilities={"ai": 0.8, "otro": 0.2}
    )
    assert (result.input_tokens, result.output_tokens) == (321, 7)


def test_ask_reports_unreported_usage_as_none():
    """`Usage` fields are `int | None`; a cost consumer must meet that path in tests."""
    sdk = _FakeSdk(
        SystemOneResponse(
            model="jev-1.13.0", usage=Usage(), answers={"topic__ai": SdkNoulAnswer(noul=0.5)}
        )
    )
    result = _client(sdk).ask({"post": "x"}, {"topic__ai": _QUESTIONS["topic__ai"]})
    assert (result.input_tokens, result.output_tokens) == (None, None)


def test_ask_sends_sdk_question_objects_state_and_model():
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.5)))
    _client(sdk).ask({"post": "hola"}, {"topic__ai": _QUESTIONS["topic__ai"]})
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
    _client(sdk).ask({"post": "x"}, {"primary": _QUESTIONS["primary"]})
    _, questions, _ = sdk.calls[0]
    assert isinstance(questions["primary"], Choice)
    assert questions["primary"].criteria == {"ai": "AI", "otro": "other"}


def test_a_noul_side_left_out_is_sent_as_undescribed():
    """`NoulCriteria` values are `JSONContent | None` and the SDK preserves a nested
    `None`, so an omitted side reaches the wire as `null` = undescribed, not as a
    missing key. The wire form is asserted because the in-memory dict cannot tell the two
    apart."""
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.5)))
    _client(sdk).ask({"post": "x"}, {"topic__ai": NoulQuestion("Is it about ai?", {"true": "AI"})})
    _, questions, _ = sdk.calls[0]
    assert questions["topic__ai"].criteria == {"true": "AI", "false": None}
    assert json.loads(questions["topic__ai"].model_dump_json())["criteria"] == {
        "true": "AI",
        "false": None,
    }


def test_choice_options_with_no_description_are_still_offered():
    """An undescribed vocabulary topic must stay on the ballot; dropping it would make the
    primary choice unable to be that topic, with nothing saying an option vanished."""
    sdk = _FakeSdk(
        _response(primary=SdkChoiceAnswer(choice="ai", confidence=0.9, probabilities={"ai": 1.0}))
    )
    _client(sdk).ask({"post": "x"}, {"primary": ChoiceQuestion("Main?", {"ai": None, "otro": "o"})})
    _, questions, _ = sdk.calls[0]
    assert questions["primary"].criteria == {"ai": None, "otro": "o"}
    assert json.loads(questions["primary"].model_dump_json())["criteria"] == {
        "ai": None,
        "otro": "o",
    }


@pytest.mark.parametrize(
    "criteria",
    [{"tru": "AI", "false": "not"}, {"yes": "AI", "no": "not"}, {"true": "a", "maybe": "c"}],
)
def test_a_misspelt_noul_side_is_refused_instead_of_shipping_an_undescribed_question(criteria):
    """`None` is a LEGAL wire value meaning "undescribed", so a wrong key would silently
    delete the topic description and Jev would score against a bare instruction."""
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.5)))
    with pytest.raises(JevError, match="true"):
        _client(sdk).ask({"post": "x"}, {"topic__ai": NoulQuestion("Is it about ai?", criteria)})
    assert sdk.calls == []


def test_a_malformed_question_fails_as_a_jev_error_not_a_pydantic_error():
    """`_to_sdk` runs the SDK's validators; a batch caller catching `JevError` per item must
    not lose the whole run to a `ValidationError` from one bad vocabulary entry."""
    sdk = _FakeSdk(_response())
    with pytest.raises(JevError, match="pregunta"):
        _client(sdk).ask({"post": "x"}, {"primary": ChoiceQuestion("x", {"ai": 3})})


# --------------------------------------------------------------------------- answer set


def test_ask_rejects_a_response_missing_an_asked_question():
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.91)))
    with pytest.raises(JevError, match="primary"):
        _client(sdk).ask({"post": "x"}, _QUESTIONS)


def test_ask_rejects_an_answer_to_a_question_that_was_not_asked():
    """An extra key becomes a topic with a probability no question ever posed."""
    sdk = _FakeSdk(
        _response(
            topic__ai=SdkNoulAnswer(noul=0.9),
            primary=SdkChoiceAnswer(choice="ai", confidence=0.8, probabilities={"ai": 1.0}),
            topic__ghost=SdkNoulAnswer(noul=0.99),
        )
    )
    with pytest.raises(JevError, match="topic__ghost"):
        _client(sdk).ask({"post": "x"}, _QUESTIONS)


def test_ask_rejects_answer_types_that_were_not_asked():
    sdk = _FakeSdk(
        _response(
            x=SdkScoreAnswer(
                score=1.0, confidence=0.5, legend={0: "a", 1: "b"}, probabilities={0: 0.0, 1: 1.0}
            )
        )
    )
    with pytest.raises(JevError, match="score.*'x'"):
        _client(sdk).ask({"post": "x"}, _QUESTIONS)


# --------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "error",
    [
        TypeSafeError("boom"),
        TypeSafeAuthenticationError(401, None, Headers(), message="bad key"),
        TypeSafeRateLimitError(429, None, Headers(), message="slow down"),
        TypeSafeAPITimeoutError(60.0),
    ],
)
def test_ask_wraps_every_sdk_error_as_jev_error_preserving_the_cause(error):
    """Every SDK exception subclasses `TypeSafeError`; narrowing the `except` to one
    subclass must go red here, and the original stays reachable as `__cause__`."""
    client = _client(_FakeSdk(error=error))
    with pytest.raises(JevError) as excinfo:
        client.ask({"post": "x"}, _QUESTIONS)
    assert str(error) in str(excinfo.value)
    assert excinfo.value.__cause__ is error


# --------------------------------------------------------------------------- construction


class _RecordingSdkClient:
    """Stands in for the real `TypeSafeClient` so the un-injected branch runs offline."""

    built: dict = {}

    def __init__(self, **kwargs):
        _RecordingSdkClient.built = kwargs
        self.closed = False

    def close(self):
        self.closed = True


def test_the_real_sdk_client_gets_the_key_model_timeout_and_a_bounded_retry_budget(monkeypatch):
    """Every other test injects `sdk_client`, so this is the ONLY place the key, the model
    and the retry policy actually reach the SDK constructor."""
    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeClient", _RecordingSdkClient)
    TypeSafeJevClient(api_key="ts-abc", model="jev-latest")  # pragma: allowlist secret
    built = _RecordingSdkClient.built
    assert built["api_key"] == "ts-abc"  # pragma: allowlist secret
    assert built["model"] == "jev-latest"
    assert built["timeout"] == 60.0
    assert built["retry"].max_retries == 3
    # Per-attempt 60 s × 4 attempts + 10 s of backoff: long enough that the attempt count
    # is what stops a retry, short enough that a wedged call cannot run forever.
    assert built["retry"].timeout == 60.0 * 4 + 10


def test_the_retry_budget_follows_a_custom_timeout(monkeypatch):
    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeClient", _RecordingSdkClient)
    TypeSafeJevClient(api_key="ts-abc", model="m", timeout=5.0)  # pragma: allowlist secret
    assert _RecordingSdkClient.built["retry"] == RetryPolicy(max_retries=3, timeout=5.0 * 4 + 10)


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_api_key_fails_as_a_jev_error_before_the_sdk_is_touched(monkeypatch, blank):
    """The SDK would raise its own English `TypeSafeError` here — and would also silently
    fall back to the TYPESAFE_API_KEY environment variable."""
    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeClient", _RecordingSdkClient)
    _RecordingSdkClient.built = {}
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        TypeSafeJevClient(api_key=blank, model="jev-latest")
    assert _RecordingSdkClient.built == {}


def test_a_key_the_sdk_refuses_fails_as_a_jev_error(monkeypatch):
    """A key carrying a non-ASCII or whitespace character survives `.env` parsing and is
    refused by the SDK at construction — which must not surface as an SDK traceback."""

    def _raise(**_kwargs):
        raise TypeSafeError("API key must contain only printable ASCII characters")

    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeClient", _raise)
    with pytest.raises(JevError, match="ASCII") as excinfo:
        TypeSafeJevClient(api_key="ts-good key", model="m")  # pragma: allowlist secret
    assert isinstance(excinfo.value.__cause__, TypeSafeError)


def test_close_closes_the_client_we_own(monkeypatch):
    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeClient", _RecordingSdkClient)
    client = TypeSafeJevClient(api_key="k", model="m")
    owned = client._client
    assert owned.closed is False
    client.close()
    assert owned.closed is True


def test_close_converts_an_sdk_failure_into_a_jev_error(monkeypatch):
    """The module docstring promises every SDK exception becomes a `JevError`, and `close`
    is no exception to that — a vendor type escaping here would reach a `finally` in the CLI
    and, unguarded, replace the outcome of a run that has already been paid for."""

    class _ExplodingSdkClient(_RecordingSdkClient):
        def close(self):
            raise TypeSafeError("transport teardown: connection reset by peer")

    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeClient", _ExplodingSdkClient)
    client = TypeSafeJevClient(api_key="k", model="m")
    with pytest.raises(JevError, match="transport teardown") as excinfo:
        client.close()
    assert isinstance(excinfo.value.__cause__, TypeSafeError)


def test_close_is_idempotent_so_a_second_release_is_not_a_second_failure(monkeypatch):
    """The Protocol says idempotent. The CLI releases in a `finally` that can run after an
    earlier release on the interrupt path, and a double close must not become an error."""
    monkeypatch.setattr("xbrain.jev.typesafe.TypeSafeClient", _RecordingSdkClient)
    client = TypeSafeJevClient(api_key="k", model="m")
    client.close()
    client.close()
    assert client._client.closed is True


def test_close_leaves_an_injected_client_alone():
    """The caller that injected it owns its lifetime — closing it would be a surprise."""
    sdk = _FakeSdk(_response())
    _client(sdk).close()
    assert sdk.closed is False


def test_the_adapter_satisfies_the_jev_client_protocol():
    def _assert_conforms(client: JevClient) -> str:
        return type(client).__name__

    assert _assert_conforms(_client(_FakeSdk(_response()))) == "TypeSafeJevClient"


def test_a_choice_with_no_options_is_refused():
    """An empty vocabulary would otherwise ship a primary-topic question with no ballot,
    and the provider would have to invent an answer."""
    sdk = _FakeSdk(_response())
    with pytest.raises(JevError, match="sin opciones"):
        _client(sdk).ask({"post": "x"}, {"primary": ChoiceQuestion("Main?", {})})
    assert sdk.calls == []


@pytest.mark.parametrize(
    ("criteria", "expected"),
    [(None, None), ({}, {"true": None, "false": None})],
)
def test_no_criteria_omits_the_key_while_an_empty_dict_sends_both_sides_undescribed(
    criteria, expected
):
    """Documented because the two look alike in Python and differ on the wire."""
    sdk = _FakeSdk(_response(topic__ai=SdkNoulAnswer(noul=0.5)))
    _client(sdk).ask({"post": "x"}, {"topic__ai": NoulQuestion("q", criteria)})
    _, questions, _ = sdk.calls[0]
    assert questions["topic__ai"].criteria == expected
    assert json.loads(questions["topic__ai"].model_dump_json()).get("criteria") == expected


def _ask_expecting_error(answers: dict) -> str:
    sdk = _FakeSdk(_response(**answers))
    with pytest.raises(JevError) as excinfo:
        _client(sdk).ask({"post": "x"}, _QUESTIONS)
    return str(excinfo.value)


_PRIMARY = SdkChoiceAnswer(choice="ai", confidence=0.8, probabilities={"ai": 1.0})


def test_a_missing_answer_is_reported_on_its_own():
    """Only one thing went wrong, so only one sentence — an empty `[]` for the other half
    reads as a second fault that did not happen."""
    message = _ask_expecting_error({"topic__ai": SdkNoulAnswer(noul=0.9)})
    assert "Jev no respondió a ['primary']" in message
    assert "contestó" not in message
    assert "[]" not in message


def test_an_unasked_answer_is_reported_on_its_own():
    message = _ask_expecting_error(
        {
            "topic__ai": SdkNoulAnswer(noul=0.9),
            "primary": _PRIMARY,
            "ghost": SdkNoulAnswer(noul=0.1),
        }
    )
    assert "Jev contestó preguntas no formuladas: ['ghost']" in message
    assert "no respondió" not in message


def test_both_halves_are_reported_when_both_happen():
    message = _ask_expecting_error(
        {"topic__ai": SdkNoulAnswer(noul=0.9), "ghost": SdkNoulAnswer(noul=0.1)}
    )
    assert "Jev no respondió a ['primary']" in message
    assert "Jev contestó preguntas no formuladas: ['ghost']" in message
