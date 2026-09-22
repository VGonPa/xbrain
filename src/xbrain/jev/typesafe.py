"""The TypeSafe (Jev) adapter — the ONLY module in `src/` that imports `typesafe_sdk`.

It maps xbrain's vendor-free questions onto the SDK's and the SDK's answers back, and it
converts every SDK exception into `JevError`, so nothing above this file handles a vendor
type. A second provider answering the same questions is a second module like this one.
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING

from pydantic import ValidationError

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

from xbrain.jev.client import (
    Answer,
    ChoiceAnswer,
    JevClient,
    JevError,
    JevResult,
    NoulAnswer,
    NoulQuestion,
    Question,
)

PROVIDER = "typesafe"
#: Retries for a call that failed transiently. The budget below is derived from it.
MAX_RETRIES = 3
#: Slack for the backoff delays between attempts, in seconds.
_BACKOFF_ALLOWANCE = 10.0
_NOUL_SIDES = ("true", "false")


def _to_sdk(question: Question) -> Noul | Choice:
    """One xbrain question as the SDK's, refusing anything that would ship a lie.

    Each `NoulCriteria` side is `JSONContent | None`, and `None` is a LEGAL value meaning
    "undescribed" — the SDK preserves a nested `None` rather than dropping the key. So a
    misspelt side would not fail: it would send the question with its description deleted
    and get back a confident probability judged against a bare instruction. An omitted side
    is deliberate and stays `None`; an unknown side is refused here.
    """
    if isinstance(question, NoulQuestion):
        criteria = None
        if question.criteria is not None:
            unknown = set(question.criteria) - set(_NOUL_SIDES)
            if unknown:
                raise JevError(
                    f"NoulQuestion.criteria solo admite {list(_NOUL_SIDES)}; "
                    f"llegó {sorted(unknown)!r}"
                )
            criteria = NoulCriteria(
                true=question.criteria.get("true"), false=question.criteria.get("false")
            )
        return Noul(instructions=question.instructions, criteria=criteria)
    if not question.criteria:
        raise JevError("ChoiceQuestion sin opciones: no hay nada que elegir")
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
                f"Jev devolvió un tipo de respuesta {answer.type!r} para {key!r}, no pedido"
            )
    return JevResult(
        provider=PROVIDER,
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
        self._owns_client = sdk_client is None
        if sdk_client is not None:
            self._client = sdk_client
            return
        # Checked before the SDK, which would otherwise raise its own English error — and,
        # worse, silently fall back to the TYPESAFE_API_KEY environment variable, so a run
        # meant to use a configured key could quietly use a different one.
        if not api_key.strip():
            raise JevError("TYPESAFE_API_KEY vacía: ponla en el entorno o en <repo>/.env")
        try:
            self._client = TypeSafeClient(
                api_key=api_key,
                model=model,
                # TWO CLOCKS, AND THEY MUST AGREE. `timeout` bounds each HTTP operation
                # within one attempt; `RetryPolicy.timeout` is the TOTAL budget across
                # attempts and defaults to 30 s — shorter than a single attempt here, so
                # it would cancel the retries of exactly the slow calls retries exist for.
                # The budget is therefore derived from the per-attempt timeout instead of
                # disabled: every attempt may run, and a wedged call still ends.
                retry=RetryPolicy(
                    max_retries=MAX_RETRIES,
                    timeout=timeout * (MAX_RETRIES + 1) + _BACKOFF_ALLOWANCE,
                ),
                timeout=timeout,
            )
        except TypeSafeError as exc:
            # A key with a non-ASCII or whitespace character survives `.env` parsing and
            # dies here; `JevError` is what the docstring promises and what the CLI knows.
            raise JevError(f"Jev: configuración inválida ({exc})") from exc

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        """Ask every question against one state, or raise `JevError`.

        The answer set must match the questions EXACTLY. Short: the SDK drops answers whose
        type it does not model (forward-compat) with only a log line, and the API may omit
        one — silence would be stored as "this topic scored nothing" instead of "this topic
        was never answered". Long: an unasked key would become a topic carrying a
        probability no question ever posed.
        """
        try:
            sdk_questions = {key: _to_sdk(question) for key, question in questions.items()}
        except ValidationError as exc:
            # The SDK validates question shape. A batch caller catches `JevError` per item;
            # a raw `ValidationError` from one bad vocabulary entry would kill the run.
            raise JevError(f"Jev: pregunta mal formada ({exc})") from exc
        try:
            response = self._client.system_one(state, sdk_questions, model=self._model)
        except TypeSafeError as exc:
            raise JevError(f"Jev API: {exc}") from exc
        result = _from_sdk(response)
        if result.answers.keys() != questions.keys():
            # One sentence per fault: a combined message would report an empty `[]` for the
            # half that did not happen, which reads as a second failure to chase.
            faults = []
            missing = sorted(questions.keys() - result.answers.keys())
            if missing:
                faults.append(f"Jev no respondió a {missing!r}")
            extra = sorted(result.answers.keys() - questions.keys())
            if extra:
                faults.append(f"Jev contestó preguntas no formuladas: {extra!r}")
            raise JevError(". ".join(faults))
        return result

    def close(self) -> None:
        """Close the SDK's HTTP pool, but only if we opened it.

        An injected `sdk_client` belongs to whoever injected it; closing it here would be a
        surprise. A run at `[jev].concurrency` holds a pool worth releasing.
        """
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TypeSafeJevClient:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        # The three are the context-manager protocol's, unused: an error on the way out
        # still has to release the pool, and it is not ours to swallow (we return None).
        self.close()


if TYPE_CHECKING:

    def _assert_conforms_to_protocol(client: TypeSafeJevClient) -> JevClient:
        """mypy enforces the Protocol HERE, in `src/`, where the gate actually looks.

        `tests/` is outside `mypy.files`, so an annotation there checks nothing; returning
        the concrete client as a `JevClient` makes any drift in `ask` a type error.
        """
        return client
