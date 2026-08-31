"""The golden set: two-stage loading of the retrieval ground truth (spec §8.1, §8.3).

WHY TWO STAGES, AND WHY IT IS NOT TIDINESS (M-B). `load_cases(path)` validates STRUCTURE and
never opens the store; `resolve_cases(cases, store)` checks that every relevant id exists.

The golden set is versioned in Git (`eval/golden-set.yaml`) so that the evaluation can be a
real gate rather than the author's assurance. It carries ids from the REAL corpus, and CI has
no `data/` (`.gitignore: data/*`). With ONE function that resolved ids at load time, the very
test that proves the evaluation runs in CI could not run: it would either explode opening a
store that does not exist, or be handed the FIXTURE store, where every id of the real golden
set fails to resolve. The comfortable escape — pointing the test at a fixture golden set —
gives back exactly what versioning bought: that the file audited in every PR is the real one.

So the structural checks run in CI against the real file, and resolution runs locally against
the real corpus (and in CI only against fixtures). `xbrain eval` chains them; nothing calls
the second without the first.

WHAT MAKES A CASE SCORABLE (spec §8.1). Its ground truth must be ENUMERATED — ids listed and
verified one by one. A case describing its population in prose does not score: with
`relevant_items: []` the recall@k is 0/0, which comes out as 1.0 or 0.0 depending on the
implementation and measures nothing either way (CLAUDE.md rule 2). Those are archived as
`scenarios` with their reason — never dropped, never scored zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any

import yaml

from xbrain.knowledge.contracts import SearchFilters
from xbrain.knowledge.models import SurfaceType
from xbrain.models import Item

# The versioned golden set. A path relative to the repo root, not an absolute one, so the
# same constant works from a checkout, a worktree and a CI runner.
GOLDEN_SET_PATH = Path("eval/golden-set.yaml")

# Spec §8.2's strata, as a closed vocabulary. A typo would otherwise create a silent new
# bucket that the report would show with one case in it — a stratum nobody defined,
# presented as if it had been measured.
STRATA: frozenset[str] = frozenset(
    {
        "exacto",
        "semantico",
        "cruzado_idioma",
        "enterrado",
        "multimodal",
        "resumen",
        "topic",
        "filtros",
        "expansion",
    }
)

# Spec §8.2 closing paragraph: results are reported SEPARATELY by provenance, because
# constructed cases serve as regression and real ones decide usefulness. A third value would
# quietly open a third column that means nothing.
PROVENANCES: frozenset[str] = frozenset({"real", "construido"})

# A short identifying fragment, never a copy of the body. The ceiling is what keeps "we only
# version questions and ids" true over time rather than true on the day it was written.
MAX_EXPECTED_TEXT = 300

_CASE_KEYS = {
    "id",
    "query",
    "provenance",
    "strata",
    "filters",
    "relevant_items",
    "relevant_topics",
    "relevant_surfaces",
    "expected_text",
    "notes",
}


class GoldenSetError(ValueError):
    """The golden set could not be loaded or resolved. Always names the case."""


@dataclass(frozen=True)
class RelevantSurface:
    """One surface that must be retrieved, beyond the item that owns it.

    Spec §8.4 asks for SURFACE recall alongside item recall, because returning the right item
    through the wrong surface is a different (and often worse) answer: it means the evidence
    the consumer would verify is not the evidence the fact is in.
    """

    owner_type: str
    owner_id: str
    surface_type: SurfaceType


@dataclass(frozen=True)
class GoldenCase:
    """One SCORABLE case: an enumerated, verified ground truth."""

    id: str
    query: str
    provenance: str
    strata: tuple[str, ...]
    filters: SearchFilters
    relevant_items: tuple[str, ...] = ()
    relevant_topics: tuple[str, ...] = ()
    relevant_surfaces: tuple[RelevantSurface, ...] = ()
    expected_text: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class GoldenScenario:
    """An ARCHIVED question with no enumerable ground truth (spec §8.1).

    Kept with its `reason` because dropping it silently is what spec §8.1 forbids, and
    because the reason is what tells the next person what enumerating it would cost. A
    scenario is NOT a case scoring zero: it does not enter any metric.
    """

    id: str
    question: str
    provenance: str
    reason: str


def load_cases(path: Path) -> tuple[GoldenCase, ...]:
    """Parse and validate STRUCTURE. Never opens the store.

    Rejects, with `GoldenSetError`: a bad schema version, an unknown stratum or filter, an
    unfilled `<X>` template, a scorable case with an empty relevant set, an `expected_text`
    over the ceiling, a topic pseudo-id in `relevant_items`, an unknown provenance and a
    duplicate id.

    Runs in CI against `eval/golden-set.yaml` itself. Keep it store-free: the signature is
    asserted by the suite for exactly that reason.
    """
    raw = _read(path)
    seen: set[str] = set()
    cases: list[GoldenCase] = []
    for entry in raw.get("cases") or []:
        case = _case(entry, path)
        if case.id in seen:
            raise GoldenSetError(f"{path}: id duplicado {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return tuple(cases)


def load_scenarios(path: Path) -> tuple[GoldenScenario, ...]:
    """The archived, unscored questions, with the reason each one is archived."""
    raw = _read(path)
    scenarios = []
    for entry in raw.get("scenarios") or []:
        missing = {"id", "question", "reason"} - set(entry)
        if missing:
            raise GoldenSetError(f"{path}: escenario incompleto, falta {sorted(missing)}")
        scenarios.append(
            GoldenScenario(
                id=str(entry["id"]),
                question=str(entry["question"]),
                provenance=str(entry.get("provenance", "construido")),
                reason=str(entry["reason"]),
            )
        )
    return tuple(scenarios)


def resolve_cases(cases: tuple[GoldenCase, ...], store: dict[str, Item]) -> tuple[GoldenCase, ...]:
    """Check that every relevant id exists in the store of THIS run (spec §8.1).

    An id that does not resolve is an ERROR, not a case that scores zero. Scoring it zero
    would blame the retriever for a stale ground truth, and the number would look like a
    permanent regression that no change to retrieval could ever fix.

    Runs locally against the real corpus, and in CI only against a fixture store.
    """
    for case in cases:
        missing = [item_id for item_id in case.relevant_items if item_id not in store]
        surface_missing = [
            surface.owner_id
            for surface in case.relevant_surfaces
            if surface.owner_type == "item" and surface.owner_id not in store
        ]
        if missing or surface_missing:
            raise GoldenSetError(
                f"caso {case.id}: id relevante que no existe en el store: "
                f"{sorted(set(missing) | set(surface_missing))}"
            )
    return cases


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GoldenSetError(f"golden set no encontrado: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise GoldenSetError(f"{path}: la raíz debe ser un mapa")
    version = str(raw.get("schema_version", ""))
    if version != "1":
        raise GoldenSetError(f"{path}: schema_version {version!r}, se esperaba '1'")
    return raw


def _case(entry: dict[str, Any], path: Path) -> GoldenCase:
    """Validate one entry. Every rejection names the case, so a 23-case file is debuggable.

    Split into one validator per concern rather than one long branch: the chunker and the
    loader are the two functions Plan 01 §13 flags as the complexity risk, and a validator
    nobody can read is how a wrong rejection message survives review.
    """
    _ = path
    case_id = str(entry.get("id", "<sin id>"))
    unknown = set(entry) - _CASE_KEYS
    if unknown:
        raise GoldenSetError(f"caso {case_id}: claves desconocidas {sorted(unknown)}")

    items, topics, surfaces = _validate_relevants(entry, case_id)
    return GoldenCase(
        id=case_id,
        query=_validate_query(entry, case_id),
        provenance=_validate_provenance(entry, case_id),
        strata=_validate_strata(entry, case_id),
        filters=_filters(entry.get("filters") or {}, case_id),
        relevant_items=items,
        relevant_topics=topics,
        relevant_surfaces=surfaces,
        expected_text=_validate_expected_text(entry, case_id),
        notes=str(entry.get("notes", "")),
    )


def _validate_query(entry: dict[str, Any], case_id: str) -> str:
    """The question itself — and never an unfilled template.

    A `<X>` placeholder rejects the whole FILE rather than being skipped. Skipping is the
    worst of the three options: the case does not score, nobody is told, and the report shows
    one fewer case than the file visibly contains.
    """
    query = str(entry.get("query", "")).strip()
    if not query:
        raise GoldenSetError(f"caso {case_id}: sin `query`")
    if "<" in query and ">" in query:
        raise GoldenSetError(
            f"caso {case_id}: es una plantilla sin rellenar ({query!r}); no es una pregunta"
        )
    return query


def _validate_provenance(entry: dict[str, Any], case_id: str) -> str:
    """`real` or `construido` — the value decides how the result is READ.

    Spec §8.2 reports metrics separately by provenance: constructed cases are regression,
    real ones decide usefulness. A third value would quietly open a third column that means
    nothing.
    """
    provenance = str(entry.get("provenance", ""))
    if provenance not in PROVENANCES:
        raise GoldenSetError(
            f"caso {case_id}: procedencia {provenance!r}; admitidas {sorted(PROVENANCES)}"
        )
    return provenance


def _validate_strata(entry: dict[str, Any], case_id: str) -> tuple[str, ...]:
    """A closed vocabulary (spec §8.2).

    A typo would create a silent new bucket, and the report would show it with one case in
    it — a stratum nobody defined, presented as if it had been measured.
    """
    strata = tuple(str(s) for s in entry.get("strata") or ())
    unknown = set(strata) - STRATA
    if unknown:
        raise GoldenSetError(
            f"caso {case_id}: estrato desconocido {sorted(unknown)}; admitidos {sorted(STRATA)}"
        )
    return strata


def _validate_relevants(
    entry: dict[str, Any], case_id: str
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[RelevantSurface, ...]]:
    """The enumerated ground truth — the thing that makes a case scorable at all (B2)."""
    items = _validate_item_ids(entry, case_id)
    topics = tuple(str(t) for t in entry.get("relevant_topics") or ())
    surfaces = tuple(_surface(s, case_id) for s in entry.get("relevant_surfaces") or ())
    if not (items or topics or surfaces):
        raise GoldenSetError(
            f"caso {case_id}: sin conjunto relevante enumerado. Con el conjunto vacío el "
            "recall@k es 0/0 y no mide nada; archívalo como `scenario` con su razón"
        )
    return items, topics, surfaces


def _validate_item_ids(entry: dict[str, Any], case_id: str) -> tuple[str, ...]:
    """The relevant ITEM ids — and never a topic pseudo-id among them.

    Spec §8.1: a topic pseudo-id is not a retrievable item. v1 wrote
    `items_relevantes: ["topic:agency-and-mindset"]` for S7/S8/S9; left there it would be
    counted forever as an item retrieval failed to return — a permanent, invisible penalty
    that no change to retrieval could ever clear.
    """
    items = tuple(str(i) for i in entry.get("relevant_items") or ())
    pseudo = [i for i in items if i.startswith("topic:")]
    if pseudo:
        raise GoldenSetError(
            f"caso {case_id}: pseudo-id de topic en relevant_items {pseudo}; "
            "migra a relevant_topics + relevant_surfaces"
        )
    return items


def _validate_expected_text(entry: dict[str, Any], case_id: str) -> str | None:
    """A short identifying fragment, never a copy of the body.

    The ceiling is what keeps "we only version questions and ids" true over time rather than
    true on the day it was written — the golden set is in Git, so anything that lands in this
    field is published.
    """
    expected = entry.get("expected_text")
    if expected is None:
        return None
    if len(str(expected)) > MAX_EXPECTED_TEXT:
        raise GoldenSetError(
            f"caso {case_id}: expected_text de {len(str(expected))} chars supera el techo de "
            f"{MAX_EXPECTED_TEXT}; es un fragmento identificador, no una copia del cuerpo"
        )
    return str(expected)


def _surface(raw: Any, case_id: str) -> RelevantSurface:
    if not isinstance(raw, dict):
        raise GoldenSetError(f"caso {case_id}: relevant_surfaces debe ser una lista de mapas")
    try:
        return RelevantSurface(
            owner_type=str(raw["owner_type"]),
            owner_id=str(raw["owner_id"]),
            surface_type=raw["surface_type"],
        )
    except KeyError as exc:
        raise GoldenSetError(f"caso {case_id}: superficie relevante sin {exc}") from exc


def _filters(raw: dict[str, Any], case_id: str) -> SearchFilters:
    """Turn the case's `filters:` block into the frozen `SearchFilters` contract.

    Spec §8.1: *windows and filters are part of the case and are not discarded when it is
    loaded*. v1 kept them under a `ventana:` key that no loader read, so a temporal case
    silently became an untemporal one and its result was reported as if the window had been
    applied. Parsing them into the SAME `SearchFilters` the search API takes is what makes
    that impossible — there is no second place for a filter to sit unread.

    A bare `YYYY-MM-DD` end date becomes end-of-day UTC: `created_to: 2025-10-31` plainly
    means "through the 31st", and midnight would silently drop a day.
    """
    unknown = set(raw) - set(SearchFilters.model_fields)
    if unknown:
        raise GoldenSetError(
            f"caso {case_id}: filtro desconocido {sorted(unknown)}; "
            f"admitidos {sorted(SearchFilters.model_fields)}"
        )
    payload = dict(raw)
    for key, end_of_day in (("created_from", False), ("created_to", True)):
        if key in payload:
            payload[key] = _as_datetime(payload[key], end_of_day=end_of_day)
    for key in ("topics", "content_kinds", "origins", "has_surfaces"):
        if key in payload:
            payload[key] = tuple(payload[key])
    try:
        return SearchFilters(**payload)
    except Exception as exc:  # pydantic validation, re-raised with the case named
        raise GoldenSetError(f"caso {case_id}: filtros inválidos — {exc}") from exc


def _as_datetime(value: Any, *, end_of_day: bool) -> datetime:
    """A YAML date or datetime as a UTC-aware datetime.

    PyYAML parses a bare `2025-10-31` into a `date`, which is not comparable with the store's
    aware datetimes — a silent `TypeError` at filter time, or worse, a naive comparison.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    moment = time(23, 59, 59) if end_of_day else time(0, 0)
    return datetime.combine(value, moment, tzinfo=timezone.utc)
