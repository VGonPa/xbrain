# tests/test_jev_dashboard.py
"""What `jev.html` ships to the browser, and whether the browser can be trusted with it.

The page recomputes every threshold-dependent number client-side (that is what the slider
IS), so the one risk worth a test suite is the browser and `jev/report.py` disagreeing. Two
defences, in order of strength:

* `test_the_browser_derives_the_same_buckets_as_the_report` runs the template's OWN `derive`
  through node and compares its buckets with the summary the same fixture produced in
  Python. When node is present this is the real check.
* `test_the_template_compares_with_ge_and_keeps_an_unjudged_bucket` pins the two rules a
  drift would break as TEXT. It is the weaker check, and it is here for the machine that
  has no node.

THE SKIP IS FAIL-CLOSED WHERE IT MATTERS. A developer laptop without node may skip the
node-executed half; a runner may not. `_NODE_IS_REQUIRED` reads `CI` and
`XBRAIN_REQUIRE_NODE` (quality.yml sets the second), and where either is set
`_requires_node` stops skipping — so a missing node goes RED instead of quietly removing
the only real check the page has. `test_node_is_available_where_the_mirror_is_required`
is the one that says so in words; without it the 19 failures below would all read as
`FileNotFoundError: node` and none of them would name the cause.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

import pytest

from xbrain.dashboard import _resource
from xbrain.jev.assess import (
    build_topic_state,
    current_pairs,
    questions_digest,
    topic_contract,
)
from xbrain.jev.dashboard import (
    DERIVE_END,
    DERIVE_START,
    GUARD_END,
    GUARD_START,
    compute_jev_dashboard_data,
    render_jev_dashboard_html,
)
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.questions import STATE_KEY, build_topic_questions
from xbrain.jev.report import THRESHOLD_DEPENDENT_KEYS
from xbrain.models import Author, Enrichment, Item, Topic

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
#: A fixed clock. The blob carries `summary["generated_at"]`, so a function that reads the
#: clock itself cannot be asserted against — and is not the pure function it claims to be.
NOW = datetime(2026, 9, 22, 18, 30, 5, tzinfo=timezone.utc)
VOCAB = [
    Topic(slug="ai-coding", description="IA."),
    Topic(slug="startups", description="Empresas."),
]
FALLBACK = "otro"
CHAR_LIMIT = 100_000
#: The one provider `jev.defaults.INPUT_USD_PER_MTOK` prices, so `cost_usd` is a real number.
PRICED_PROVIDER = "typesafe"

#: Is node a REQUIREMENT here, or a convenience? `CI` is set by GitHub Actions itself and
#: `XBRAIN_REQUIRE_NODE` is set explicitly by `quality.yml`, so both mean "this machine is a
#: gate, not a laptop". Two variables rather than one: `CI` covers any runner that forgets
#: the explicit flag, and the explicit flag lets a developer reproduce the gate's behaviour
#: locally (`XBRAIN_REQUIRE_NODE=1 uv run pytest tests/test_jev_dashboard.py`).
_NODE_IS_REQUIRED = bool(os.environ.get("CI") or os.environ.get("XBRAIN_REQUIRE_NODE"))
#: ONE mark for the 19 node-executed tests, so the skip rule has one definition. Nineteen
#: copies of a condition is nineteen places for a future edit to restore the fail-open one.
_requires_node = pytest.mark.skipif(
    shutil.which("node") is None and not _NODE_IS_REQUIRED,
    reason="no JS engine on this machine (set XBRAIN_REQUIRE_NODE=1 to make this fail instead)",
)


def _item(item_id: str = "1", text: str = "Claude Code hooks", topics=("ai-coding",)) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=text,
        created_at=DT,
        captured_at=DT,
        enriched=Enrichment(
            enriched_at=DT,
            executor="claude-code",
            summary="s",
            primary_topic=topics[0],
            topics=list(topics),
        ),
    )


def _assessment(
    item: Item,
    contract: str | None = None,
    membership: dict[str, float] | None = None,
    provider: str = PRICED_PROVIDER,
    vocab: list[Topic] | None = None,
) -> TopicAssessment:
    """A record whose contract is the one TODAY'S ask would stamp, so it reads as current.

    Built the way `assess.assess_topics` builds it — the state as sent and the digest of the
    questions that went with it — so a change to either composition shows up here as a stale
    fixture instead of a test passing over a contract nobody computes any more.
    """
    state, state_chars = build_topic_state(item, CHAR_LIMIT)
    # The digest is over the vocabulary the ask ACTUALLY used: a description is part of a
    # question, so a fixture built against one vocabulary and compared against another reads
    # as stale — correctly, and confusingly if it is not the subject of the test.
    digest = questions_digest(build_topic_questions(vocab or VOCAB, FALLBACK))
    return TopicAssessment(
        item_id=item.id,
        provider=provider,
        model="jev-1.13.0",
        asked_at=DT,
        contract=contract or topic_contract(state[STATE_KEY], digest),
        state_chars=state_chars,
        membership=membership or {"startups": 0.123456, "ai-coding": 0.9},
        primary=PrimaryChoice(
            choice="ai-coding",
            confidence=0.77,
            probabilities={"ai-coding": 0.7, "startups": 0.2, "otro": 0.1},
        ),
        input_tokens=2000,
    )


def _data(items: list[Item], assessments: dict[str, TopicAssessment], **kwargs) -> dict[str, Any]:
    options: dict[str, Any] = {
        "threshold": 0.85,
        "fallback": FALLBACK,
        "char_limit": CHAR_LIMIT,
        "id2note": {},
        "updated": "SEP 22, 2026",
        "now": NOW,
    }
    options.update(kwargs)
    return compute_jev_dashboard_data(items, assessments, VOCAB, **options)


# --------------------------------------------------------------------------- the blob


def test_blob_aligns_membership_with_topics_and_drops_stale():
    fresh, stale = _item("1"), _item("2", text="old")
    data = _data(
        [fresh, stale],
        # Two providers, so `providers` and `unpriced_providers` are pinned by more than one
        # value: with one, the assertion cannot fail while the fixture supplies both sides.
        {
            "1": _assessment(fresh),
            "2": _assessment(stale, contract="e" * 64, provider="otro-juez"),
        },
        id2note={"1": "/v/items/1.md"},
    )

    assert [t["slug"] for t in data["topics"]] == ["ai-coding", "startups"]
    assert data["topics"][0]["label"] == "AI Coding"
    assert [row["id"] for row in data["items"]] == ["1"]
    row = data["items"][0]
    # Same order as `data["topics"]`, and NOT rounded: the browser compares these against the
    # slider with the same `>=` `report.py` uses, so a rounded 0.8496 would read as backed at
    # 0.85 where the report reads doubtful — a divergence in the data, not in the logic.
    assert row["m"] == [0.9, 0.123456]
    assert row["cmp"] is True
    assert row["assigned"] == ["ai-coding"] and row["primary"] == "ai-coding"
    assert row["jp"] == "ai-coding" and row["jc"] == 0.77
    assert row["top"] == [["ai-coding", 0.7], ["startups", 0.2], ["otro", 0.1]]
    assert row["note"] == "/v/items/1.md" and row["handle"] == "alice"
    # Against the SOURCE, not a constant: blanked, every `X` link on the page would point
    # at the page itself, and the row would look identical either way.
    assert row["url"] == fresh.url
    assert row["truncated"] is False
    assert row["model"] == "jev-1.13.0" and row["provider"] == PRICED_PROVIDER
    assert data["threshold"] == 0.85 and data["fallback"] == FALLBACK
    assert data["totals"] == {
        "items": 2,
        "assessed": 2,
        "current": 1,
        "stale": 1,
        "orphans": 0,
        "models": {"jev-1.13.0": 1},
        "providers": {PRICED_PROVIDER: 1},
        "truncated": 0,
        "input_tokens": 2000,
        "input_tokens_unknown": 0,
        "cost_usd": round(2000 / 1e6 * 0.042, 4),
        "unpriced_providers": [],
        # The ONE sentence that quotes a bill — `defaults.jev_cost_fragment`, the same words
        # `jev topics` and `jev report` print, never a second formatter in the template.
        "cost_text": "2000 tokens de entrada (~0.0001 $)",
    }


def test_blob_ships_the_report_summary_and_the_threshold_dependent_keys():
    """The browser cannot check itself against numbers it was not given.

    `summary` is the server-side truth at the DEFAULT threshold and the key split says which
    of its numbers the slider invalidates; `selfCheck` needs both, and without them the page
    would be free to show numbers `xbrain jev report` would never print.
    """
    item = _item()
    data = _data([item], {"1": _assessment(item)})

    assert data["summary"]["threshold"] == 0.85
    assert data["summary"]["assigned_backed"] == 1  # ai-coding at 0.9 >= 0.85
    assert data["summary"]["doubtful_pairs"] == 0
    assert data["threshold_dependent_keys"] == sorted(THRESHOLD_DEPENDENT_KEYS)
    assert data["summary"]["input_tokens"] == data["totals"]["input_tokens"]


def test_a_topic_that_left_the_vocabulary_is_unjudged_not_a_zero():
    """An assigned slug absent from `membership` gets its own bucket in BOTH halves.

    `enrich` validates topics at write time, so an item enriched under an older `vocab.yaml`
    carries a slug today's questions do not. Scoring it 0.0 would put it at the top of the
    "most doubtful" queue as the strongest disagreement in the corpus — an invented doubt.
    """
    item = _item("1", topics=("ai-coding", "web3"))
    data = _data([item], {"1": _assessment(item)})

    row = data["items"][0]
    assert row["assigned"] == ["ai-coding", "web3"]
    # `m` is positional over the VOCABULARY, so a retired slug has no slot at all — the
    # browser reads "not in `slugs`" as unjudged rather than inventing a probability.
    assert row["m"] == [0.9, 0.123456]
    assert data["summary"]["assigned_unjudged"] == 1
    assert data["summary"]["assigned_pairs"] == 2
    assert data["summary"]["assigned_backed"] == 1


def test_an_item_without_enrichment_is_shipped_but_not_comparable():
    """`compare_item` returns None for it, so the browser must not count it either."""
    item = _item()
    item.enriched = None
    data = _data([item], {"1": _assessment(item)})

    assert data["items"][0]["cmp"] is False
    assert data["items"][0]["primary"] is None and data["items"][0]["assigned"] == []
    assert data["totals"]["current"] == 1
    assert data["summary"]["items_compared"] == 0


def test_a_post_past_the_budget_is_cut_with_an_ellipsis_like_the_markdown_report():
    """A post that simply stops mid-word reads as a broken record, not as a cut one.

    `report._snippet` already solved this for the markdown table — cut to `width - 1`, then
    an ellipsis — and the drawer is the "see the whole item" surface, so the silent version
    is worse here than it is there.
    """
    long_post = "palabra " * 80  # 640 chars, well past the budget
    short_post = "Claude Code hooks"
    items = [_item("1", text=long_post), _item("2", text=short_post)]
    data = _data(items, {i.id: _assessment(i) for i in items})

    cut, whole = data["items"][0]["text"], data["items"][1]["text"]
    assert len(cut) == 240 and cut.endswith("…")
    assert cut[:239] == " ".join(long_post.split())[:239]
    # A post that fits is NOT decorated: an ellipsis on a complete post is the same lie in
    # the other direction.
    assert whole == short_post


@pytest.mark.parametrize(
    ("length", "expected"),
    [(239, "x" * 239), (240, "x" * 240), (241, "x" * 239 + "\u2026")],
)
def test_the_cut_is_tested_at_the_boundary_not_four_hundred_characters_away(length, expected):
    """240 exactly must NOT be decorated: off by one here claims a cut that never happened."""
    item = _item("1", text="x" * length)

    assert _data([item], {"1": _assessment(item)})["items"][0]["text"] == expected


def test_the_summary_carries_the_clock_it_was_handed_and_never_reads_one():
    """A function that reads the clock inside itself is not pure, and `generated_at` is the
    one blob field that would silently differ between two runs of the same inputs."""
    item = _item()

    data = _data([item], {"1": _assessment(item)})

    assert data["summary"]["generated_at"] == NOW.isoformat()


def test_the_caller_may_hand_in_the_currency_it_already_computed():
    """`_jev_pairs` has already run `current_pairs` before the CLI gets here.

    Recomputing it is a second `build_topic_state` + sha256 over the whole corpus for an
    answer the caller is holding. Handing it in must produce a byte-identical blob: if the
    two paths could differ, passing it in would be a way to make the page disagree with the
    refusal that just let it through.
    """
    fresh, stale = _item("1"), _item("2", text="old")
    assessments = {"1": _assessment(fresh), "2": _assessment(stale, contract="e" * 64)}
    current = current_pairs(
        [fresh, stale], assessments, VOCAB, fallback=FALLBACK, char_limit=CHAR_LIMIT
    )

    handed_in = _data([fresh, stale], assessments, current=current)

    assert handed_in == _data([fresh, stale], assessments)
    assert handed_in["totals"]["stale"] == 1


def test_a_side_car_record_whose_item_left_the_corpus_is_an_orphan_not_a_stale_one():
    """`assessed == current + stale + orphans`, asserted as the equation `_totals` claims.

    The two ways of losing a record differ: a vocabulary edit RETIRES an assessment and
    re-running `jev topics` buys it back; a deleted item ORPHANS one and nothing does.
    """
    fresh, gone = _item("1"), _item("2")

    data = _data([fresh], {"1": _assessment(fresh), "2": _assessment(gone)})

    totals = data["totals"]
    assert (totals["items"], totals["assessed"]) == (1, 2)
    assert (totals["current"], totals["stale"], totals["orphans"]) == (1, 0, 1)
    assert totals["assessed"] == totals["current"] + totals["stale"] + totals["orphans"]


def test_an_item_enrich_left_without_a_primary_is_still_compared():
    """No primary is not "nothing to compare": its assigned topics still have nouls to back.

    `_row`'s docstring names this hazard — reading a null primary as "skip" would drop the
    item from the browser's side while `report.compare_item` keeps it, so the banner would
    fire on a corpus that is fine.
    """
    item = _item("1", topics=("ai-coding",))
    item.enriched.primary_topic = None

    data = _data([item], {"1": _assessment(item)})

    assert data["items"][0]["cmp"] is True and data["items"][0]["primary"] is None
    assert data["summary"]["items_compared"] == 1
    assert data["summary"]["assigned_pairs"] == 1


def test_a_vocabulary_slug_the_assessment_never_answered_is_null_not_zero():
    """A fabricated 0.0 is the strongest possible disagreement, invented out of silence.

    Different from a RETIRED slug, which has no slot in `m` at all: this one is in today's
    vocabulary and the stored answer simply has no entry for it.
    """
    item = _item("1", topics=("ai-coding", "startups"))

    data = _data([item], {"1": _assessment(item, membership={"ai-coding": 0.9})})

    assert data["items"][0]["m"] == [0.9, None]
    assert data["summary"]["assigned_unjudged"] == 1


def test_a_truncated_assessment_is_flagged_on_its_own_row():
    """The `TRUNCADO` badge is the only per-item sign Jev judged a CUT post — which is
    exactly the item whose verdict a reviewer should distrust."""
    item = _item()
    assessment = _assessment(item).model_copy(update={"truncated": True})

    data = _data([item], {"1": assessment})

    assert data["items"][0]["truncated"] is True
    assert data["totals"]["truncated"] == 1


def test_the_choice_distribution_is_cut_to_five_and_ties_break_by_name():
    """`report._primary_rank` breaks ties by option name so two runs of one distribution can
    never report different ranks; the drawer sorts the same list and must not disagree."""
    item = _item()
    assessment = _assessment(item).model_copy(
        update={
            "primary": PrimaryChoice(
                choice="ai-coding",
                confidence=0.77,
                probabilities={
                    "startups": 0.3,
                    "ai-coding": 0.3,
                    "otro": 0.2,
                    "web3": 0.1,
                    "ml": 0.05,
                    "zzz": 0.05,
                },
            )
        }
    )

    top = _data([item], {"1": assessment})["items"][0]["top"]

    assert len(top) == 5  # six options offered, five shown
    assert [option for option, _ in top] == ["ai-coding", "startups", "otro", "web3", "ml"]


def test_a_topic_assigned_twice_is_counted_twice_on_both_sides():
    """`report._pair_totals` counts every assigned slot, duplicates included, while
    `jev_backed` intersects SETS. Four independent choices line up; this pins the alignment."""
    item = _item("1", topics=("ai-coding", "ai-coding"))

    data = _data([item], {"1": _assessment(item)})

    assert data["items"][0]["assigned"] == ["ai-coding", "ai-coding"]
    assert data["summary"]["assigned_pairs"] == 2 and data["summary"]["assigned_backed"] == 2
    assert data["summary"]["jev_backed"] == 1  # the set intersection, not the multiset


# --------------------------------------------------------------------------- the page


def test_render_inlines_data_and_library_without_leaving_sentinels():
    item = _item()
    data = _data([item], {"1": _assessment(item)})

    html = render_jev_dashboard_html(data)

    assert "/*__DATA__*/" not in html and "/*__ECHARTS__*/" not in html
    assert "const DATA = " in html
    assert json.dumps(data["topics"][0]["slug"]) in html
    assert 'id="threshold"' in html
    assert "echarts" in html.lower()
    # Self-contained: no external script, and the fonts stylesheet is the only thing the
    # markup fetches — exactly as `dashboard.template.html` is. A CDN slipped into the
    # template would make the page useless on a plane and leak the corpus's shape to a third
    # party, and neither shows up as a failing assertion anywhere else.
    fetched = re.findall(r'<(?:script|link|img|iframe)[^>]*\s(?:src|href)="(https?://[^"]*)"', html)
    assert fetched and all(url.startswith("https://fonts.g") for url in fetched)


def test_scraped_text_cannot_close_the_script_tag_or_break_the_parse():
    """This is the page that inlines raw scraped X text into `const DATA = ...`.

    Two different failures, one escaper: an un-escaped `</script>` closes the tag at HTML-parse
    time (stored XSS), and an un-escaped U+2028 is a hard `SyntaxError` — a blank page, not a
    degraded one. `render_dashboard_html` delegates both to `dashboard._escape_for_script`;
    what is new here is the exposure, so the round-trip is pinned on this surface too.
    """
    vocab = [
        Topic(slug="ai-coding", description="IA.\u2028Fin.</script><img src=x>"),
        Topic(slug="startups", description="Empresas."),
    ]
    item = _item("1", text="cierra aquí </script><img src=x onerror=alert(1)>")
    data = compute_jev_dashboard_data(
        [item],
        {"1": _assessment(item, vocab=vocab)},
        vocab,
        threshold=0.85,
        fallback=FALLBACK,
        char_limit=CHAR_LIMIT,
        id2note={},
        updated="SEP 22, 2026",
        now=NOW,
    )

    html = render_jev_dashboard_html(data)

    # The vendored library's tag and the data tag. A third one is a break-out.
    assert html.count("</script>") == 2
    assert "\u2028" not in html
    assert "<img src=x" not in html
    # …and the text is not mangled: it round-trips through `JSON.parse` unchanged.
    assert data["items"][0]["text"] == "cierra aquí </script><img src=x onerror=alert(1)>"


def test_the_template_compares_with_ge_and_keeps_an_unjudged_bucket():
    """The WEAKER check, for a machine with no node: the two rules pinned as text.

    `>` instead of `>=` moves every pair sitting exactly on the threshold, and a missing
    `unjudged` bucket folds retired topics into `backed` or `doubtful`. Both are silent.
    """
    template = _resource("jev.template.html")

    # TWICE, once per side of the mirror (`enrichSide` and `jevSide`). Asserting mere presence
    # let either one be flipped alone while the other kept the pin satisfied — and a
    # one-function edit is exactly what drift looks like.
    assert template.count("noul >= t") == 2
    assert "noul > t" not in template
    assert "unjudged" in template
    assert DERIVE_START in template and DERIVE_END in template
    # ONE `topicRows`, and it is the one inside the region. A second declaration later in the
    # script silently wins (function declarations hoist, the last one binds), so the page ran
    # an unpinned copy while the node test pinned a dead one — the mirror was there and not
    # connected to anything.
    assert template.count("function topicRows") == 1


def test_the_boot_guard_is_registered_before_anything_that_can_throw():
    """A guard installed after the work it guards is not a guard.

    `echarts.init`, the histogram and the load-time latch all run at script evaluation, and an
    uncaught throw in any of them aborts the script — leaving the full masthead, five titled
    panels and zero rows, with the banner element still `hidden`. That page reads as "the
    corpus is empty", which is wrong and sends an operator to re-pay for the side-car.
    """
    template = _resource("jev.template.html")
    guard = template.index("addEventListener('error'")

    # Before the payload, before the mirror, before ECharts, before any rendering.
    for later in ("const DATA = ", DERIVE_START, "echarts.init(", "function boot("):
        assert guard < template.index(later), later
    # And the eager work is inside `boot()`, whose single call is the wrapped one.
    assert template.count("boot();") == 1


# --------------------------------------------------------------------------- browser vs report

_HARNESS = """
const DATA = %s;
%s
console.log(JSON.stringify((() => { %s })()));
"""


def _run_in_node(data: dict[str, Any], body: str) -> Any:
    """Run `body` against the template's OWN pure region, over `data`, in node.

    Only the delimited region is extracted, so nothing here needs a DOM or ECharts: the
    region is the pure half of the script on purpose, and the delimiters are exported by
    `jev/dashboard.py` so this test and the template can never disagree about where it is.
    `body` is a statement block ending in `return`.
    """
    template = _resource("jev.template.html")
    region = template.split(DERIVE_START, 1)[1].split(DERIVE_END, 1)[0]
    script = _HARNESS % (json.dumps(data, ensure_ascii=False), region, body)
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(out.stdout)


def _derive_in_node(data: dict[str, Any]) -> dict[str, int]:
    """The template's own buckets at the threshold its summary was built at."""
    return _run_in_node(data, "return buckets(derive(DATA.threshold));")


def _self_check_in_node(
    data: dict[str, Any], patch: dict[str, Any] | None = None, *, at: float | None = None
) -> list[str]:
    """The template's own `selfCheck` against `data["summary"]`, optionally corrupted.

    `patch` is merged over the summary BEFORE the call, which is how the safety net is shown
    to fire: a check that is never observed failing is indistinguishable from one that cannot.

    `at` is where the SLIDER is; the summary stays at the threshold it was built at. Without
    it every call had `d.t === summary.threshold`, so `atReportThreshold` was always true and
    the entire key split — the reason `THRESHOLD_DEPENDENT_KEYS` is shipped at all — was never
    executed in the state it exists for.
    """
    overrides = json.dumps(patch or {}, ensure_ascii=False)
    derived_at = "s.threshold" if at is None else repr(float(at))
    return _run_in_node(
        data,
        f"const s = Object.assign({{}}, DATA.summary, {overrides});"
        f" return selfCheck(s, derive({derived_at}));",
    )


def test_node_is_available_where_the_mirror_is_required():
    """On a GATE, a missing node is a failure; on a laptop it is a skip.

    This is the whole fail-open fix in one assertion. The 19 tests below carry the page's
    only real correctness guarantee — that the browser's `derive` and `jev/report.py` agree
    — and a `skipif` hands that guarantee to whatever happens to be installed. `node` is
    present on `ubuntu-latest` because the image ships it, not because anything asked for
    it; the day the image drops it, or the day the gate moves to a leaner runner, all 19
    disappear and the gate still reports green. A check that cannot tell you whether it
    checked is the failure mode this repo has already paid for (`quality.yml`'s #103
    comment block, and the skipped-required-check battery in `tests/test_ci_workflow.py`).

    Kept separate from the 19 rather than folded into them: when node is genuinely missing
    on a runner, those 19 fail with `FileNotFoundError: node` and none of them says why it
    matters. This one names the cause once.
    """
    if not _NODE_IS_REQUIRED:
        pytest.skip("node is optional on a developer machine; CI sets CI/XBRAIN_REQUIRE_NODE")
    assert shutil.which("node") is not None, (
        "node is required in CI: the template's derive/selfCheck tests would silently skip"
    )


@_requires_node
def test_the_browser_derives_the_same_buckets_as_the_report():
    """The page and `xbrain jev report` must never quote different numbers.

    The fixture deliberately carries every bucket at once: a backed pair, a doubtful one, a
    missing candidate, a retired topic and a primary that does not coincide. A `derive` that
    mirrors only the easy half still passes an all-agreeing corpus.
    """
    data = _every_bucket_fixture()

    assert _derive_in_node(data) == {
        "backed": data["summary"]["assigned_backed"],
        "doubtful": data["summary"]["doubtful_pairs"],
        "missing": data["summary"]["missing_pairs"],
        "unjudged": data["summary"]["assigned_unjudged"],
        "jev_pairs": data["summary"]["jev_pairs"],
        # `jev_backed` and `items_compared` are DISPLAYED (the "Jev respaldado por enrich"
        # KPI, and the denominator of "primario coincide"), so they are checked like the
        # rest: a number on screen that nothing compares is a number free to be wrong.
        "jev_backed": data["summary"]["jev_backed"],
        "primary_agree": data["summary"]["primary_agree"],
        # `primary_fallback` is what the queue's own toggle filters on, so the predicate
        # behind a filtered count is checked even though the count itself is display-only.
        "primary_fallback": data["summary"]["primary_fallback"],
        "assigned_pairs": data["summary"]["assigned_pairs"],
        "items_compared": data["summary"]["items_compared"],
    }


@_requires_node
def test_the_browser_recomputes_the_buckets_when_the_threshold_moves():
    """At another threshold the page must still agree with what the report WOULD print.

    Pinning only the default threshold would certify a `derive` that ignores its argument —
    the one function whose whole job is to take it.
    """
    item = _item("1", topics=("ai-coding", "startups"))
    rows = _data([item], {"1": _assessment(item)}, threshold=0.10)

    assert _derive_in_node(rows)["backed"] == 2  # 0.9 and 0.123456 both clear 0.10
    assert _derive_in_node(rows)["doubtful"] == 0


def _every_bucket_fixture() -> dict[str, Any]:
    """A corpus carrying every bucket at once: backed, doubtful, missing, unjudged, mismatch.

    `ai-coding` sits EXACTLY ON THE THRESHOLD (0.85 against an umbral of 0.85), which is the
    one value that tells `>=` from `>`. `report._doubtful` uses `membership[slug] < threshold`,
    so at equality the report says BACKED — and a page spelling it `>` says doubtful, while
    `assigned_backed` is threshold-dependent and therefore only checked at one slider
    position. A probability of exactly 0.85 is not exotic for an LLM asked for a confidence.
    """
    backed = _item("1", topics=("ai-coding", "web3"))
    doubtful = _item("2", text="Seed round", topics=("startups",))
    return _data(
        [backed, doubtful],
        {
            "1": _assessment(backed, membership={"ai-coding": 0.85, "startups": 0.123456}),
            "2": _assessment(doubtful, membership={"ai-coding": 0.91, "startups": 0.4}),
        },
    )


@_requires_node
def test_self_check_is_silent_when_the_page_agrees_with_the_report():
    """The net must not cry wolf: over its own summary it reports nothing at all."""
    assert _self_check_in_node(_every_bucket_fixture()) == []


@_requires_node
@pytest.mark.parametrize(
    "key",
    [
        "assigned_backed",
        "doubtful_pairs",
        "missing_pairs",
        "jev_pairs",
        "jev_backed",
        "assigned_unjudged",
        "primary_agree",
        "assigned_pairs",
        "items_compared",
    ],
)
def test_self_check_names_the_bucket_the_report_disagrees_about(key: str):
    """Every checked key, off by one, must come back NAMED.

    A check that is never observed failing is indistinguishable from one that cannot fail:
    a wrong pairing in the tables (`['missing_pairs', 'doubtful']`) would ship a banner that
    never fires, which is the exact failure the banner exists to prevent. One case per key,
    because one key standing for the other eight is how a mis-pairing survives.
    """
    data = _every_bucket_fixture()

    problems = _self_check_in_node(data, {key: data["summary"][key] + 1})

    assert [p for p in problems if p.startswith(f"{key}:")], problems


@_requires_node
def test_self_check_compares_the_per_topic_rows_not_just_the_totals():
    """Chart 01 is a per-topic number, so it is checked per topic.

    The corpus totals can agree while a single topic's row is wrong — that is exactly what a
    per-topic bug looks like, and a totals-only net would pass it.
    """
    data = _every_bucket_fixture()
    rows = [dict(row) for row in data["summary"]["per_topic"]]
    target = next(row for row in rows if row["assigned"])
    target["backed"] += 1

    problems = _self_check_in_node(data, {"per_topic": rows})

    assert [p for p in problems if p.startswith(f"per_topic[{target['slug']}].backed:")], problems


@_requires_node
def test_self_check_reports_a_key_that_changed_sides_in_the_report():
    """The key split is itself a contract.

    `selfCheck` checks the threshold-FREE buckets at EVERY threshold and the dependent ones
    only at the report's. If `report.py` ever moves one across, the page's two tables become
    quietly wrong about when they apply — so the mismatch between the shipped
    `THRESHOLD_DEPENDENT_KEYS` and the page's own tables is reported rather than assumed.
    """
    data = _every_bucket_fixture()
    data["threshold_dependent_keys"] = [
        k for k in data["threshold_dependent_keys"] if k != "jev_backed"
    ]

    problems = _self_check_in_node(data)

    assert [p for p in problems if "jev_backed" in p], problems


@_requires_node
def test_the_browser_recomputes_against_a_summary_built_at_another_threshold():
    """The slider is the page's whole reason to exist, and it was certified by a test that
    could not fail: every node case built the summary at the threshold it then derived at, so
    `function derive(t) { t = DATA.threshold; ... }` passed them all.

    Here the summary is built at 0,85 and the browser derives at 0,95, against numbers
    computed by hand from the fixture:

      ai-coding 0,90 assigned  -> backed at 0,85 · DOUBTFUL at 0,95
      startups  0,91 unassigned -> missing at 0,85 · NOTHING at 0,95 (0,91 < 0,95)

    so every bucket in the pair differs between the two thresholds and a `derive` that ignores
    its argument answers with the 0,85 column.
    """
    item = _item("1", topics=("ai-coding",))
    data = _data([item], {"1": _assessment(item, membership={"ai-coding": 0.9, "startups": 0.91})})

    at_report = _run_in_node(data, "return buckets(derive(0.85));")
    moved = _run_in_node(data, "return buckets(derive(0.95));")

    assert (at_report["backed"], at_report["doubtful"], at_report["missing"]) == (1, 0, 1)
    assert (moved["backed"], moved["doubtful"], moved["missing"]) == (0, 1, 0)
    assert (at_report["jev_pairs"], moved["jev_pairs"]) == (2, 0)
    # And the 0,85 column is what the report prints for the same pairs.
    assert at_report["backed"] == data["summary"]["assigned_backed"]
    assert at_report["missing"] == data["summary"]["missing_pairs"]


@_requires_node
def test_self_check_is_silent_when_only_the_slider_moved():
    """Moving the slider changes the threshold-dependent buckets BY DESIGN.

    A banner there would be the page crying wolf on its own feature, and a reader who learns
    to ignore the banner has lost the one signal that matters.
    """
    assert _self_check_in_node(_every_bucket_fixture(), at=0.10) == []


@_requires_node
@pytest.mark.parametrize(
    "key",
    ["assigned_unjudged", "primary_agree", "primary_fallback", "assigned_pairs", "items_compared"],
)
def test_the_threshold_free_net_still_fires_after_the_slider_moves(key: str):
    """These do not move with the threshold, so they are checked at EVERY one.

    This is the branch that goes dark if `: FREE_CHECKS` ever becomes `: []` — the net would
    switch itself off at exactly the moment nothing else is checking.
    """
    data = _every_bucket_fixture()

    problems = _self_check_in_node(data, {key: data["summary"][key] + 1}, at=0.10)

    assert [p for p in problems if p.startswith(f"{key}:")], problems


@_requires_node
def test_a_mismatch_found_on_load_is_never_withdrawn_when_the_slider_moves():
    """THE BANNER DOES NOT RETRACT.

    `selfCheck` is right that it may only compare the threshold-dependent keys at the umbral
    the summary was built at. But a divergence found there is a fact about the PAGE — the same
    arithmetic runs at every umbral — so showing only the current check's output meant the
    page said "no te fíes de los números de esta página" on load and then withdrew it the
    moment the reader touched the slider. The numbers stayed wrong; only the warning left.
    """
    data = _every_bucket_fixture()
    data["summary"]["assigned_backed"] += 1000  # a divergence visible only at 0,85

    on_load = _run_in_node(data, "return currentMismatches(derive(DATA.threshold));")
    after_moving = _run_in_node(data, "return currentMismatches(derive(0.10));")

    assert [p for p in on_load if p.startswith("assigned_backed:")], on_load
    assert [p for p in after_moving if p.startswith("assigned_backed:")], after_moving


@_requires_node
@pytest.mark.parametrize("field", ["assigned", "backed", "doubtful", "unjudged", "missing"])
def test_self_check_compares_every_per_topic_field(field: str):
    """All five, one case each. The suite's own rule, one level down: mutating only `backed`
    left `TOPIC_FIELDS = ['backed']` passing, and chart 01's tooltip displays four of them."""
    data = _every_bucket_fixture()
    rows = [dict(row) for row in data["summary"]["per_topic"]]
    target = next(row for row in rows if row["assigned"])
    target[field] += 1

    problems = _self_check_in_node(data, {"per_topic": rows})

    assert [p for p in problems if p.startswith(f"per_topic[{target['slug']}].{field}:")], problems


@_requires_node
def test_chart_one_orders_its_rows_the_way_the_report_orders_its_table():
    """Chart 01's ROW ORDER mirrors `report._backing_order`, so it is executed like the rest.

    Worst backing first, never-assigned last, keyed on the EXACT ratio — `_pct` rounds to one
    decimal, so ordering on the rendered percentage puts the one topic with a real
    disagreement below every perfect topic whose slug sorts earlier.
    """
    data = _every_bucket_fixture()

    order = _run_in_node(data, "return topicRows(derive(DATA.threshold)).map(r => r.slug);")

    assert order == [row["slug"] for row in data["summary"]["per_topic"]]


@_requires_node
@pytest.mark.parametrize(
    ("part", "whole"), [(1, 16), (5, 16), (2, 32), (9, 16), (1, 3), (2, 3), (0, 0), (1, 1)]
)
def test_the_page_rounds_percentages_the_way_python_does(part: int, whole: int):
    """One side-car, one number. `toFixed` rounds half AWAY from zero and Python's `round` is
    half-to-EVEN, so one backed assignment of sixteen printed `6,3 %` in the KPI band and
    `6.2 %` in `topics-report.md`. 152 of the 80,199 pairs under whole<=400 diverge that way.

    `(0, 0)` is the other half of the mirror: `report._pct` scores an empty whole as `0.0` and
    the markdown prints `0.0 %`, so the page may not print an em dash there.
    """
    data = _every_bucket_fixture()
    expected = f"{round(part / whole * 100, 1) if whole else 0.0:.1f}".replace(".", ",") + " %"

    assert _run_in_node(data, f"return pct1({part}, {whole});") == expected


@_requires_node
def test_the_partition_invariant_is_observable():
    """Belt-and-braces over checks that would already fire — but by the suite's own standard a
    check never observed failing is indistinguishable from one that cannot.

    It reads only the page's own buckets, so no corrupted summary reaches it; the rule is its
    own function precisely so a hand-made bucket can.
    """
    data = _every_bucket_fixture()
    whole = "{backed: 3, assigned_pairs: 5, doubtful: 1, unjudged: 1}"
    broken = "{backed: 1, assigned_pairs: 5, doubtful: 1, unjudged: 1}"

    assert _run_in_node(data, f"return partitionProblems({whole});") == []
    assert [
        p
        for p in _run_in_node(data, f"return partitionProblems({broken});")
        if "no se reparten" in p
    ]


@_requires_node
def test_the_histograms_assigned_half_is_checked_against_the_report():
    """Panel 02 derives 40 bin counts. The `asignados` half has a counterpart in the summary —
    every assigned pair Jev answered — so it is checked rather than merely drawn."""
    data = _every_bucket_fixture()

    clean = _self_check_in_node(data)
    corrupted = _self_check_in_node(
        data, {"assigned_unjudged": data["summary"]["assigned_unjudged"] + 1}
    )

    assert clean == []
    assert [p for p in corrupted if p.startswith("histograma:")], corrupted


@_requires_node
def test_chart_one_orders_by_backing_not_by_slug_or_vocabulary_order():
    """A fixture where all three orders differ, so the sort is pinned and not coincidence.

    Vocabulary order is `ai-coding, startups`; alphabetical is the same; `report._backing_order`
    puts the WORST backing first, which here is `startups` (0 of 1) ahead of `ai-coding`
    (1 of 1). A `topicRows` that forgot to sort would answer with the vocabulary order.
    """
    item = _item("1", topics=("ai-coding", "startups"))
    data = _data([item], {"1": _assessment(item, membership={"ai-coding": 0.9, "startups": 0.1})})

    order = _run_in_node(data, "return topicRows(derive(DATA.threshold)).map(r => r.slug);")

    assert order == ["startups", "ai-coding"]
    assert order != [topic["slug"] for topic in data["topics"]]  # not the vocabulary order
    assert order == [row["slug"] for row in data["summary"]["per_topic"]]


@_requires_node
def test_the_histogram_counts_assigned_pairs_as_a_multiset_like_the_report():
    """`report._pair_totals` counts every assigned SLOT, duplicates included.

    Binning through a `Set` dropped the duplicate, so the histogram total came up one short of
    `assigned_pairs - assigned_unjudged` and the page latched a red banner over a corpus that
    is perfectly fine — a false alarm from the check that exists to prevent false calm.
    """
    item = _item("1", topics=("ai-coding", "ai-coding"))
    data = _data([item], {"1": _assessment(item)})

    assert data["summary"]["assigned_pairs"] == 2
    assert _self_check_in_node(data) == []


@_requires_node
def test_the_page_and_python_round_every_percentage_in_the_corpus_the_same_way():
    """A sweep, not a handful: every (part, whole) with whole <= 120, against Python's `round`.

    `x * 10` is not exact for most x, so the earlier half-even helper decided "is this a tie"
    on a value the multiplication had already moved. The rule now is Python's: an EXACT binary
    tie (x is a quarter and x*20 is odd) rounds half-to-even; everything else goes through
    `toFixed`, which — like `round` — reads the exact binary value.
    """
    pairs = [(part, whole) for whole in range(0, 121) for part in range(0, whole + 1)]
    expected = [
        f"{round(part / whole * 100, 1) if whole else 0.0:.1f}".replace(".", ",") + " %"
        for part, whole in pairs
    ]

    got = _run_in_node(
        _every_bucket_fixture(),
        "return " + json.dumps(pairs) + ".map(([a, b]) => pct1(a, b));",
    )

    first_bad = next((i for i, (g, e) in enumerate(zip(got, expected)) if g != e), None)
    assert first_bad is None, f"{pairs[first_bad]}: {got[first_bad]} != {expected[first_bad]}"


_GUARD_HARNESS = """
const painted = {hidden: true, innerHTML: '', appendChild(n) { this.innerHTML += n.textContent; }};
globalThis.document = {
  getElementById: (id) => (id === 'banner' ? painted : null),
  createTextNode: (text) => ({textContent: text}),
};
globalThis.addEventListener = () => {};
%s
function boot() { throw new Error('echarts no está: la librería no se inyectó'); }
try { boot(); } catch (err) { paintBootFailure(err.message); }
console.log(JSON.stringify({hidden: painted.hidden, html: painted.innerHTML}));
"""


@_requires_node
def test_a_throw_during_boot_paints_the_banner_instead_of_a_blank_page():
    """The guard, executed — with a `boot()` that does what a missing ECharts would do.

    Registering the listener is not the same as it working: `paintBootFailure` runs before most
    of the file exists, so anything it reached for (`esc`, `$`, `f3`) would be a second failure
    on the failure path. The stub below provides only `document.getElementById` and
    `createTextNode`, which is the whole of what it may use.
    """
    template = _resource("jev.template.html")
    guard = template.split(GUARD_START, 1)[1].split(GUARD_END, 1)[0]

    out = subprocess.run(
        ["node", "-e", _GUARD_HARNESS % guard],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    result = json.loads(out.stdout)
    assert result["hidden"] is False
    assert "La página no pudo dibujarse" in result["html"]
    assert "echarts no está" in result["html"]
    assert "xbrain jev report" in result["html"]
