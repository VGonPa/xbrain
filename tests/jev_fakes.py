# tests/jev_fakes.py — the JevClient test double shared by the jev test modules.
from __future__ import annotations

from collections.abc import Callable

from xbrain.jev.client import (
    ChoiceAnswer,
    JevError,
    JevResult,
    NoulAnswer,
    NoulQuestion,
    Question,
)


class FakeJevClient:
    """Every `topic__<slug>` Noul answers `nouls.get(slug, default_noul)`; the `primary`
    Choice answers `primary` with `confidence` and a one-hot distribution. `fail_when(state)`
    True raises `JevError`, so batch failure paths can be exercised per item."""

    def __init__(
        self,
        *,
        nouls: dict[str, float] | None = None,
        default_noul: float = 0.05,
        primary: str = "otro",
        confidence: float = 0.9,
        model: str = "jev-1.13.0",
        fail_when: Callable[[dict[str, str]], bool] | None = None,
    ) -> None:
        self.nouls = dict(nouls or {})
        self.default_noul = default_noul
        self.primary = primary
        self.confidence = confidence
        self.model = model
        self.fail_when = fail_when
        self.calls: list[tuple[dict[str, str], dict[str, Question]]] = []

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        self.calls.append((state, questions))
        if self.fail_when is not None and self.fail_when(state):
            raise JevError("fake failure")
        answers: dict[str, NoulAnswer | ChoiceAnswer] = {}
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                slug = key.removeprefix("topic__")
                answers[key] = NoulAnswer(noul=self.nouls.get(slug, self.default_noul))
            else:
                probabilities = {option: 0.0 for option in question.criteria}
                probabilities[self.primary] = 1.0
                answers[key] = ChoiceAnswer(
                    choice=self.primary, confidence=self.confidence, probabilities=probabilities
                )
        return JevResult(model=self.model, answers=answers, input_tokens=100, output_tokens=10)
