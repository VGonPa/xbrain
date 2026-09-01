# tests/test_knowledge_goldenset.py
"""The golden-set loader, in TWO stages, and why that is the whole point (M-B, steps 21-23).

`load_cases(path)` validates STRUCTURE and never opens the store. `resolve_cases(cases,
store)` checks that every relevant id exists. Fusing them would look tidier and would break
the thing B1 bought.

Here is the mechanism. `eval/golden-set.yaml` is versioned in Git so that the evaluation can
be a real CI gate. It carries ids from the REAL corpus, and CI has no `data/` (`.gitignore:
data/*`). With one function that resolved ids at load time, the test proving "the evaluation
runs in CI" would be the first one that could not run: either it explodes opening a store
that does not exist, or it is handed the FIXTURE store and then every id in the real golden
set fails to resolve. And the comfortable way out — pointing the test at a fixture golden set
— silently gives back exactly what B1 paid for: that the file audited in every PR is the
REAL one.

So: the structural tests below run against `eval/golden-set.yaml` itself, in CI, with no
store. The resolution test runs against a fixture store. Both see red for their own defect.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.knowledge.goldenset import (
    GOLDEN_SET_PATH,
    GoldenSetError,
    load_cases,
    load_scenarios,
    resolve_cases,
)
from xbrain.models import Author, Item

REPO_ROOT = Path(__file__).resolve().parent.parent
UTC = timezone.utc


@pytest.fixture(scope="module")
def real_cases():
    """The REAL, versioned golden set — parsed structurally, with no store in sight."""
    return load_cases(REPO_ROOT / GOLDEN_SET_PATH)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "gs.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _store(*ids: str) -> dict[str, Item]:
    return {
        item_id: Item(
            id=item_id,
            source="bookmark",
            url=f"https://x.com/a/status/{item_id}",
            author=Author(handle="a", name="A"),
            text="t",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            captured_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        for item_id in ids
    }


# ---------------------------------------------------------------------------
# 21d — where the files live (B1)
# ---------------------------------------------------------------------------


def test_the_golden_set_is_tracked_and_the_reports_are_not() -> None:
    """The decision of B1, asserted from both sides.

    Tracked ground truth is what makes the migration reviewable, the evaluation runnable in
    CI and an edited case visible in history. Untracked reports are what keeps corpus
    excerpts out of Git. Seen red by inverting either. `git check-ignore` exits 1 for a path
    that is NOT ignored, 0 for one that is.
    """

    def ignored(path: str) -> bool:
        return (
            subprocess.run(
                ["git", "check-ignore", "-q", path], cwd=REPO_ROOT, check=False
            ).returncode
            == 0
        )

    assert not ignored("eval/golden-set.yaml"), "the golden set must be tracked"
    assert ignored("data/eval-report.json"), "the report must NOT be tracked"
    assert ignored("data/items.json"), "the corpus must NOT be tracked"


# ---------------------------------------------------------------------------
# 21e — the real file, validated in CI, with no store
# ---------------------------------------------------------------------------


def test_the_real_golden_set_loads_structurally_without_a_store(real_cases) -> None:
    """This is the test B1 exists to make possible.

    It runs in CI, against `eval/golden-set.yaml` — the file a reviewer sees in the diff —
    and it does not touch `data/`, which does not exist there.
    """
    assert len(real_cases) >= 21
    assert {case.id for case in real_cases} >= {"S1", "P2", "D1c", "X1", "F1"}


def test_no_expected_text_is_a_corpus_body(real_cases) -> None:
    """A hard 300-char ceiling, so a body cannot be versioned through the back door.

    `expected_text` is the minimum fragment that identifies the fact, not a copy of it. The
    ceiling is what keeps "we only version questions and ids" true over time, rather than
    true on the day it was written. Seen red by pasting a paragraph into a case.
    """
    assert all(len(case.expected_text or "") <= 300 for case in real_cases)


def test_the_real_golden_set_covers_the_strata_that_were_empty(real_cases) -> None:
    """`exacto` and `filtros` had ZERO cases in v1/v2 (spec anexo A.4).

    A stratum with no case cannot say whether indexing for it helped — which is why the
    migration had to add them rather than report a stratum at 0.0.
    """
    strata = {stratum for case in real_cases for stratum in case.strata}
    assert {"exacto", "filtros"} <= strata
    assert "expansion" not in strata, (
        "the expansion stratum has no mechanism until Plan 04; a case for it would be "
        "measuring something that does not exist"
    )


def test_every_scorable_case_has_an_enumerated_ground_truth(real_cases) -> None:
    """B2: only a case enumerated id by id scores.

    With `relevant_items: []` and no surfaces, recall@k is 0/0 — which comes out as 1.0 or
    0.0 depending on the implementation and measures nothing either way (CLAUDE.md rule 2).
    """
    for case in real_cases:
        assert case.relevant_items or case.relevant_surfaces or case.relevant_topics, case.id


def test_the_archived_scenarios_keep_their_reason() -> None:
    """Spec §8.1: a case without enumerable ground truth is archived, never dropped silently.

    The reason is the part that matters: it is what tells the next person what enumerating
    it would take, and it is why C1-C5 can be re-read later instead of rediscovered.
    """
    scenarios = load_scenarios(REPO_ROOT / GOLDEN_SET_PATH)
    assert {s.id for s in scenarios} >= {"C1", "C2", "C3", "C4", "C5"}
    assert all(s.reason.strip() for s in scenarios)


def test_d1e_the_unfilled_template_is_absent_from_the_file(real_cases) -> None:
    """D1e was `¿Cuál es el recurso más completo que tengo sobre <X>?` — not a question.

    It is not archived as a scenario either: a template is not a case that lacks ground
    truth, it is a case that lacks a QUESTION.
    """
    scenarios = load_scenarios(REPO_ROOT / GOLDEN_SET_PATH)
    assert "D1e" not in {c.id for c in real_cases} | {s.id for s in scenarios}


# ---------------------------------------------------------------------------
# 21b, 21c, 22, 23 — structural rejections, each seen red on a fixture file
# ---------------------------------------------------------------------------


def test_a_scorable_case_without_relevants_is_a_load_error(tmp_path: Path) -> None:
    """B2, as a structural error: it is visible in the file, so it needs no store."""
    path = _write(
        tmp_path,
        'schema_version: "1"\ncases:\n'
        '  - id: Q1\n    query: "q"\n    provenance: construido\n'
        "    strata: [exacto]\n    relevant_items: []\n    relevant_surfaces: []\n",
    )
    with pytest.raises(GoldenSetError, match="sin conjunto relevante"):
        load_cases(path)


def test_an_unfilled_template_makes_the_whole_file_a_load_error(tmp_path: Path) -> None:
    """A `<X>` placeholder rejects the FILE — it is never skipped in silence.

    Skipping would be the worst of the three options: the case does not score, nobody is
    told, and the report shows one fewer case than the file contains.
    """
    path = _write(
        tmp_path,
        'schema_version: "1"\ncases:\n'
        '  - id: D1e\n    query: "¿Cuál es el recurso más completo que tengo sobre <X>?"\n'
        "    provenance: construido\n    strata: [semantico]\n"
        '    relevant_items: ["1"]\n',
    )
    with pytest.raises(GoldenSetError, match="plantilla"):
        load_cases(path)


def test_a_topic_pseudo_id_in_relevant_items_is_a_load_error(tmp_path: Path) -> None:
    """Spec §8.1: a topic pseudo-id is not a retrievable item.

    v1 wrote `items_relevantes: ["topic:agency-and-mindset"]` for S7/S8/S9. Left as an item
    id it would be counted as an item that retrieval failed to return, forever — a
    permanent, invisible penalty. It belongs in `relevant_topics` plus the surface that
    holds the text. Seen red by feeding the unmigrated shape.
    """
    path = _write(
        tmp_path,
        'schema_version: "1"\ncases:\n'
        '  - id: S7\n    query: "q"\n    provenance: construido\n'
        '    strata: [topic]\n    relevant_items: ["topic:agency-and-mindset"]\n',
    )
    with pytest.raises(GoldenSetError, match="pseudo-id"):
        load_cases(path)


def test_a_window_becomes_filters_and_is_never_discarded(real_cases) -> None:
    """Spec §8.1: windows and filters are PART of the case.

    v1 kept them under a `ventana:` key that no loader read, so a temporal case silently
    became an untemporal one. F1 carries a real window; this asserts it survived the
    migration as typed datetimes rather than as text nobody applies.
    """
    f1 = next(case for case in real_cases if case.id == "F1")
    assert f1.filters.created_from == datetime(2025, 10, 1, tzinfo=UTC)
    assert f1.filters.created_to.date().isoformat() == "2025-10-31"
    assert f1.filters.source == "own_tweet"
    f2 = next(case for case in real_cases if case.id == "F2")
    assert f2.filters.content_kinds == ("x_video",)


def test_an_unknown_stratum_is_a_load_error(tmp_path: Path) -> None:
    """The strata are a closed vocabulary (spec §8.2).

    A typo'd stratum would create a silent new bucket, and the report would show it with one
    case in it — a stratum nobody defined, reported as if it were measured.
    """
    path = _write(
        tmp_path,
        'schema_version: "1"\ncases:\n'
        '  - id: Q1\n    query: "q"\n    provenance: construido\n'
        '    strata: [semantiko]\n    relevant_items: ["1"]\n',
    )
    with pytest.raises(GoldenSetError, match="estrato"):
        load_cases(path)


def test_an_unknown_filter_is_a_load_error(tmp_path: Path) -> None:
    """Spec §9.3: invalid filters are a stable validation error, not a silent drop."""
    path = _write(
        tmp_path,
        'schema_version: "1"\ncases:\n'
        '  - id: Q1\n    query: "q"\n    provenance: construido\n'
        '    strata: [filtros]\n    filters: {mood: happy}\n    relevant_items: ["1"]\n',
    )
    with pytest.raises(GoldenSetError):
        load_cases(path)


def test_an_unknown_provenance_is_a_load_error(tmp_path: Path) -> None:
    """`real` vs `construido` decides how the result is READ, so it cannot be free text.

    Spec §8.2 requires the metrics to be reported separately by provenance: constructed cases
    serve as regression, real ones decide usefulness. A third value would quietly create a
    third column that means nothing.
    """
    path = _write(
        tmp_path,
        'schema_version: "1"\ncases:\n'
        '  - id: Q1\n    query: "q"\n    provenance: inventado\n'
        '    strata: [exacto]\n    relevant_items: ["1"]\n',
    )
    with pytest.raises(GoldenSetError, match="procedencia"):
        load_cases(path)


def test_a_duplicate_case_id_is_a_load_error(tmp_path: Path) -> None:
    """Two cases under one id means one of them silently disappears from every report."""
    path = _write(
        tmp_path,
        'schema_version: "1"\ncases:\n'
        '  - id: Q1\n    query: "a"\n    provenance: construido\n'
        '    strata: [exacto]\n    relevant_items: ["1"]\n'
        '  - id: Q1\n    query: "b"\n    provenance: construido\n'
        '    strata: [exacto]\n    relevant_items: ["2"]\n',
    )
    with pytest.raises(GoldenSetError, match="duplicado"):
        load_cases(path)


# ---------------------------------------------------------------------------
# 21 — resolution, the stage that DOES need a store
# ---------------------------------------------------------------------------


def test_a_relevant_id_missing_from_the_store_is_a_resolution_error(tmp_path: Path) -> None:
    """Spec §8.1: an id that does not resolve is an ERROR, not a case that scores zero.

    Scoring it zero would blame the retriever for the ground truth being stale, and the
    number would look like a regression forever. Seen red by returning the cases unchecked.
    """
    cases = load_cases(
        _write(
            tmp_path,
            'schema_version: "1"\ncases:\n'
            '  - id: Q1\n    query: "q"\n    provenance: construido\n'
            '    strata: [exacto]\n    relevant_items: ["999"]\n',
        )
    )
    with pytest.raises(GoldenSetError, match="no existe"):
        resolve_cases(cases, _store("1", "2"))


def test_resolution_passes_when_every_id_exists(tmp_path: Path) -> None:
    cases = load_cases(
        _write(
            tmp_path,
            'schema_version: "1"\ncases:\n'
            '  - id: Q1\n    query: "q"\n    provenance: construido\n'
            '    strata: [exacto]\n    relevant_items: ["1", "2"]\n',
        )
    )
    assert resolve_cases(cases, _store("1", "2", "3")) == cases


def test_load_cases_does_not_read_the_store_at_all() -> None:
    """The separation, asserted on the SIGNATURE rather than by hoping.

    If `load_cases` grew a store parameter, the CI test above would have to be handed one —
    and the only store available there is the fixture, at which point every id in the real
    golden set fails and someone "fixes" it by pointing the test at a fixture file. This
    assertion is what makes that regression loud.
    """
    import inspect

    assert set(inspect.signature(load_cases).parameters) == {"path"}
    assert "store" in inspect.signature(resolve_cases).parameters
