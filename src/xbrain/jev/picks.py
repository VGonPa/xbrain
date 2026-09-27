"""What a request to `xbrain jev serve` may ask a pass about: a `TopicsPick` or an `AskPick`.

A pick names posts the way the page does — some ids, a topic, the next N without a current
answer, the posts behind a pair (`post_sets` `cx` / `px` / `pd`) or a confidence band — plus
whether current answers are re-asked (`force`). It is parsed here, strictly, and resolved to
post ids from the page's own blob, so a pick means exactly the posts the page lists under it.

The «Preguntar» ask (`/api/ask/…`) names its posts another way: a query, the pre-filters of
`xbrain jev ask` and a limit (`AskPick`, `parse_ask`), plus the minimum probability its
results are recorded with (`min`). What the tab offers beside each topic is `parse_filters`'
(the same filters, nothing else). The server's job slot and the pass lock are shared by both
kinds.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from xbrain.jev.ask import AskFilters
from xbrain.jev.errors import refuse
from xbrain.jev.questions import normalize_query
from xbrain.models import Topic

#: Posts one pick may name by id. The page never needs more; the corpus is ~2.6k posts.
MAX_IDS = 5000
#: The pick kinds, in the order a body is checked for them (`_VALUES` checks each value).
PICK_KINDS = ("ids", "topic", "unevaluated", "pair", "band")
#: `post_sets` keys a pair may name: topic confusion, primary confusion, primary agreement.
PAIR_KINDS = ("cx", "px", "pd")


@dataclass(frozen=True)
class TopicsPick:
    """One selection for the topics pass: which posts, and whether current ones are re-asked.

    `value` is the ids (a tuple), the topic slug, N, the pair's `(kind, key)` or the band key.
    Frozen and comparable, so a confirmation is bound to exactly the pick it priced — «post 3»
    and «the next unevaluated post» are different picks even on a day they are the same post.
    """

    kind: str
    value: Any
    force: bool

    def as_json(self) -> dict[str, Any]:
        value = self.value
        if self.kind == "ids":
            value = list(value)
        elif self.kind == "pair":
            value = {"kind": value[0], "key": value[1]}
        return {self.kind: value, "force": self.force}


def _ids_value(value: Any, force: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_IDS:
        raise refuse(f"`ids` debe ser una lista de 1 a {MAX_IDS} ids")
    if not all(isinstance(item_id, str) and item_id for item_id in value):
        raise refuse("`ids` debe contener ids de post (texto)")
    return tuple(dict.fromkeys(value))


def _unevaluated_value(value: Any, force: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise refuse("`unevaluated` debe ser un entero >= 1")
    if force:
        raise refuse(
            "«los N siguientes sin evaluar» no se fuerza: re-evaluar se pide por post, topic o cruce"
        )
    return value


def _pair_value(value: Any, force: bool) -> tuple[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != {"kind", "key"}
        or value["kind"] not in PAIR_KINDS
        or not isinstance(value["key"], str)
    ):
        raise refuse(f"`pair` debe ser {{kind: {'|'.join(PAIR_KINDS)}, key: texto}}")
    return (value["kind"], value["key"])


def _text_value(value: Any, force: bool) -> str:
    if not isinstance(value, str) or not value:
        raise refuse("`topic` y `band` deben ser texto")
    return value


#: Each pick kind's value check: the value normalised (hashable, so a pick compares), or a 400
#: naming what is wrong.
_VALUES: dict[str, Callable[[Any, bool], Any]] = {
    "ids": _ids_value,
    "topic": _text_value,
    "unevaluated": _unevaluated_value,
    "pair": _pair_value,
    "band": _text_value,
}


def parse_pick(body: Any) -> TopicsPick:
    """The pick a request body names — exactly one kind, plus `force` — or a 400."""
    if not isinstance(body, dict):
        raise refuse("el cuerpo debe ser un objeto JSON")
    unknown = sorted(set(body) - set(PICK_KINDS) - {"force"})
    if unknown:
        raise refuse(f"campos desconocidos: {', '.join(unknown)}")
    kinds = [kind for kind in PICK_KINDS if kind in body]
    if len(kinds) != 1:
        raise refuse(f"indica exactamente uno de: {', '.join(PICK_KINDS)}")
    force = body.get("force", False)
    if not isinstance(force, bool):
        raise refuse("`force` debe ser true o false")
    kind = kinds[0]
    return TopicsPick(kind=kind, value=_VALUES[kind](body[kind], force), force=force)


def _topic_posts(slug: str, blob: dict[str, Any], vocab: list[Topic]) -> list[str]:
    """The posts a topic filter shows on the page: every card whose `slugs` carry it (enrich's
    topics and Jev's rows alike), in the page's order."""
    if slug not in {topic.slug for topic in vocab}:
        raise refuse(f"topic desconocido: {slug}")
    return [card["id"] for card in blob["posts"] if slug in card["slugs"]]


def _set_posts(sets: dict[str, dict[str, list[str]]], kind: str, key: str, what: str) -> list[str]:
    """The posts behind a pair or band, from the page's `post_sets` (`report.post_sets`)."""
    if key not in sets.get(kind, {}):
        raise refuse(f"{what} desconocido: {key}")
    return list(sets[kind][key])


def pick_ids(pick: TopicsPick, blob: dict[str, Any], vocab: list[Topic]) -> list[str] | None:
    """The posts `pick` names, in order; `None` for «the next N unevaluated» (a limit)."""
    if pick.kind == "ids":
        return list(pick.value)
    if pick.kind == "unevaluated":
        return None
    if pick.kind == "topic":
        return _topic_posts(pick.value, blob, vocab)
    kind, key = pick.value if pick.kind == "pair" else ("bands", pick.value)
    return _set_posts(blob["post_sets"], kind, key, "cruce" if pick.kind == "pair" else "banda")


# --------------------------------------------------------------------------- the ask's pick

#: The longest query a request may send, in characters (after normalising). A query is a
#: sentence or two; the bound keeps a pasted document from being sent with every post.
MAX_QUERY_CHARS = 2000
#: The fields an ask body may carry: the query, `xbrain jev ask`'s pre-filters and its limit.
#: `topics` is a list (OR within topics); `topic` is the single topic an older page sends.
FILTER_FIELDS = ("topics", "topic", "since", "until", "author", "only_evaluated")
ASK_FIELDS = ("query", *FILTER_FIELDS, "limit", "min")


@dataclass(frozen=True)
class AskPick:
    """One «preguntar» request: the query as filed (`normalize_query`), the pre-filters, the
    limit and the results' minimum — `xbrain jev ask QUERY [--topic …] [--since] [--until]
    [--author] [--only-evaluated] [--limit] [--min]`. Frozen and comparable, so a
    confirmation is bound to exactly the query, filters and minimum it priced. `minimum`
    `None` = not sent (the page never sends one): the use keeps the query's last minimum
    (`ask.finish_ask`), so asking again from the page never wipes a terminal's `--min`."""

    query: str
    filters: AskFilters
    limit: int | None
    minimum: float | None = None

    def as_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"query": self.query, **self.filters.as_json()}
        if self.limit is not None:
            out["limit"] = self.limit
        if self.minimum is not None:
            out["min"] = self.minimum
        return out


def _query(value: Any) -> str:
    if not isinstance(value, str):
        raise refuse("falta `query`: la pregunta, como texto")
    try:
        text = normalize_query(value)
    except ValueError as exc:
        raise refuse(str(exc)) from exc
    if len(text) > MAX_QUERY_CHARS:
        raise refuse(f"la consulta pasa de {MAX_QUERY_CHARS} caracteres")
    return text


def _limit(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise refuse("`limit` debe ser un entero >= 1")
    return value


def _minimum(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise refuse("`min` debe ser un número entre 0 y 1 (la relevancia mínima)")
    return float(value)


def parse_filters(body: Any) -> AskFilters:
    """The pre-filters a request names, alone (`GET /api/ask/counts`'s query string), or a 400."""
    if not isinstance(body, dict):
        raise refuse("el cuerpo debe ser un objeto JSON")
    unknown = sorted(set(body) - set(FILTER_FIELDS))
    if unknown:
        raise refuse(f"campos desconocidos: {', '.join(unknown)}")
    try:
        return AskFilters.from_json(body)
    except ValueError as exc:
        raise refuse(str(exc)) from exc


def parse_ask(body: Any) -> AskPick:
    """The ask a request body names — a query, optional filters, an optional limit — or a 400.

    The filters are `AskFilters.from_json`'s, the command's own checks (a range that runs
    backwards, an empty topic or author, a malformed day); on top, a query longer than
    `MAX_QUERY_CHARS` is refused (it would travel with every post)."""
    if not isinstance(body, dict):
        raise refuse("el cuerpo debe ser un objeto JSON")
    unknown = sorted(set(body) - set(ASK_FIELDS))
    if unknown:
        raise refuse(f"campos desconocidos: {', '.join(unknown)}")
    query = _query(body.get("query"))
    filters = parse_filters({key: body[key] for key in FILTER_FIELDS if key in body})
    return AskPick(
        query=query,
        filters=filters,
        limit=_limit(body.get("limit")),
        minimum=_minimum(body.get("min")),
    )
