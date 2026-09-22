"""The Jev client seam: xbrain's own question/answer types on one side, `typesafe-sdk` on
the other. Nothing above this module imports the SDK, so a second provider answering the
same questions is a second class with the same `ask`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from typesafe_sdk import (
    Choice,
    Noul,
    NoulCriteria,
    RetryPolicy,
    SystemOneResponse,
    TypeSafeClient,
    TypeSafeError,
)
from typesafe_sdk import ChoiceAnswer as SdkChoiceAnswer
from typesafe_sdk import NoulAnswer as SdkNoulAnswer

# TypeSafe's list price on 2026-09-22 (docs.typesafe.ai/models): input only, output is free.
USD_PER_MTOK_INPUT = 0.042


class JevError(RuntimeError):
    """An operator-facing Jev failure: missing key, API error, or an answer set we cannot use."""


@dataclass(frozen=True)
class NoulQuestion:
    instructions: str
    criteria: dict[str, str] | None = None  # keys "true" / "false"


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: str
    criteria: dict[str, str]  # option -> description


Question = NoulQuestion | ChoiceQuestion


@dataclass(frozen=True)
class NoulAnswer:
    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]


Answer = NoulAnswer | ChoiceAnswer


@dataclass(frozen=True)
class JevResult:
    model: str
    answers: dict[str, Answer]
    input_tokens: int | None = None
    output_tokens: int | None = None


class JevClient(Protocol):
    """One call: a `state` and a map of typed questions, all answered against that state."""

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult: ...


def _to_sdk(question: Question) -> Noul | Choice:
    if isinstance(question, NoulQuestion):
        criteria = (
            NoulCriteria(true=question.criteria["true"], false=question.criteria["false"])
            if question.criteria is not None
            else None
        )
        return Noul(instructions=question.instructions, criteria=criteria)
    return Choice(instructions=question.instructions, criteria=dict(question.criteria))


def _from_sdk(response: SystemOneResponse) -> JevResult:
    answers: dict[str, Answer] = {}
    for key, answer in response.answers.items():
        if isinstance(answer, SdkNoulAnswer):
            answers[key] = NoulAnswer(noul=answer.noul)
        elif isinstance(answer, SdkChoiceAnswer):
            answers[key] = ChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=dict(answer.probabilities),
            )
        else:
            raise JevError(
                f"Jev devolvió una respuesta de tipo {answer.type!r} para {key!r}, que no se pidió"
            )
    return JevResult(
        model=response.model,
        answers=answers,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


class TypeSafeJevClient:
    """`JevClient` over the official SDK. `sdk_client` is the injection seam for tests,
    mirroring `executors.api.ApiExecutor(client=...)`."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        sdk_client: TypeSafeClient | None = None,
    ) -> None:
        self._model = model
        self._client = (
            sdk_client
            if sdk_client is not None
            else TypeSafeClient(
                api_key=api_key, model=model, retry=RetryPolicy(max_retries=3), timeout=timeout
            )
        )

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        sdk_questions = {key: _to_sdk(question) for key, question in questions.items()}
        try:
            response = self._client.system_one(state, sdk_questions, model=self._model)
        except TypeSafeError as exc:
            raise JevError(f"Jev API: {exc}") from exc
        return _from_sdk(response)
