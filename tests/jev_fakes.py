# tests/jev_fakes.py — the JevClient test double shared by the jev test modules.
from __future__ import annotations

import copy
from collections.abc import Callable

from xbrain.jev.client import (
    ChoiceAnswer,
    ChoiceQuestion,
    JevError,
    JevResult,
    NoulAnswer,
    NoulQuestion,
    Question,
)


class FakeJevClient:
    """A `JevClient` that answers without a network, as strictly as the real adapter.

    Every Noul answers `nouls.get(slug, default_noul)`, where `slug` is its key minus an
    optional `topic__` prefix; every Choice answers `primary` with `confidence` and a
    one-hot distribution. Dispatch is on the question TYPE, not on the key, so a second
    Choice under any name gets the primary treatment. `fail_when(state)` True raises
    `JevError`, so batch failure paths can be exercised per item, and an empty question map
    is refused exactly as the adapter refuses it.

    `primary` is answered even when it is NOT one of the question's options, and that is
    deliberate: it is how a caller (task 2) drives the "the primary must be a vocabulary
    slug or the fallback" guard, e.g. `FakeJevClient(primary="banana")`. Constraining the
    fake to the offered options would make that guard untestable.

    `input_tokens`/`output_tokens` are configurable and may be `None`, because the real
    provider reports no usage sometimes and cost code must be able to meet that path.
    """

    def __init__(
        self,
        *,
        nouls: dict[str, float] | None = None,
        default_noul: float = 0.05,
        primary: str = "otro",
        confidence: float = 0.9,
        provider: str = "fake",
        model: str = "jev-1.13.0",
        input_tokens: int | None = 100,
        output_tokens: int | None = 10,
        fail_when: Callable[[dict[str, str]], bool] | None = None,
    ) -> None:
        self.nouls = dict(nouls or {})
        self.default_noul = default_noul
        self.primary = primary
        self.confidence = confidence
        self.provider = provider
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.fail_when = fail_when
        self.calls: list[tuple[dict[str, str], dict[str, Question]]] = []

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        # Deep snapshots, not references: a caller that builds `questions` in a loop and
        # reuses the dict would otherwise see the final state in every recorded call, and a
        # shallow copy still shares each question's `criteria`, so editing one in place
        # would rewrite history the log had already recorded.
        self.calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
        if not questions:
            raise JevError("Jev: llamada sin preguntas")
        if self.fail_when is not None and self.fail_when(state):
            raise JevError("fake failure")
        answers: dict[str, NoulAnswer | ChoiceAnswer] = {}
        for key, question in questions.items():
            if isinstance(question, NoulQuestion):
                slug = key.removeprefix("topic__")
                answers[key] = NoulAnswer(noul=self.nouls.get(slug, self.default_noul))
            elif isinstance(question, ChoiceQuestion):
                probabilities = {option: 0.0 for option in question.criteria}
                probabilities[self.primary] = 1.0
                answers[key] = ChoiceAnswer(
                    choice=self.primary, confidence=self.confidence, probabilities=probabilities
                )
            else:
                raise JevError(f"tipo de pregunta no modelado para {key!r}: {type(question)}")
        # The same completeness the adapter enforces, so a test can never assert against an
        # answer set the real client would have refused. `raise`, not `assert`: `python -O`
        # strips an assert, and this double is what every later task asserts against.
        if answers.keys() != questions.keys():
            unanswered = sorted(questions.keys() - answers.keys())
            raise JevError(f"la doble no respondió a {unanswered!r}")
        return JevResult(
            provider=self.provider,
            model=self.model,
            answers=answers,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
