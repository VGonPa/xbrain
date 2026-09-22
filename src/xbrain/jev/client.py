"""The Jev seam, vendor-free: the question and answer types xbrain itself speaks.

Nothing here imports a provider SDK. `JevClient` is the whole contract — a provider is a
class with an `ask`, and `typesafe.py` is the first one. Keeping the protocol and the
adapter apart is what lets the rest of the package (questions, assessment, store, report)
import this module without paying for, or depending on, anybody's HTTP stack. Mirrors
`executors/base.py` vs `executors/api.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

#: The two sides of a Noul's criteria. A `Literal` key type, not a bare `str`, because a
#: misspelt side is not a typo the wire can report: the provider accepts `None` as
#: "undescribed", so `{"tru": …}` would ship a question with its description silently gone.
NoulSide = Literal["true", "false"]


class JevError(RuntimeError):
    """An operator-facing Jev failure, in the language the CLI prints.

    Raised for an unusable key or client configuration, a question we refuse to send, a
    provider error, and an answer set that does not match the questions asked. It is the
    ONLY exception type the seam emits: no provider exception reaches a caller.
    """


@dataclass(frozen=True)
class NoulQuestion:
    """A yes/no question. `criteria` describes the two outcomes; either side may be left
    out, which asks the question with that outcome undescribed."""

    instructions: str
    criteria: dict[NoulSide, str] | None = None


@dataclass(frozen=True)
class ChoiceQuestion:
    """A pick-one question: `criteria` maps each option to its description."""

    instructions: str
    criteria: dict[str, str | None]


Question = NoulQuestion | ChoiceQuestion


@dataclass(frozen=True)
class NoulAnswer:
    """A probability in [0, 1] that the answer to a `NoulQuestion` is yes."""

    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """The option a `ChoiceQuestion` picked, with its confidence and the distribution."""

    choice: str
    confidence: float
    probabilities: dict[str, float]


Answer = NoulAnswer | ChoiceAnswer


@dataclass(frozen=True)
class JevResult:
    """One answered call. `provider` and `model` travel WITH the answers, so a record built
    from this result can never be attributed to the wrong judge; token counts are `None`
    when the provider did not report usage."""

    provider: str
    model: str
    answers: dict[str, Answer]
    input_tokens: int | None = None
    output_tokens: int | None = None


class JevClient(Protocol):
    """One call: a `state` and a map of typed questions, all answered against that state."""

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult: ...
