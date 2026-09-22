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
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any

import pytest

from xbrain.dashboard import _resource
from xbrain.jev.assess import build_topic_state, questions_digest, topic_contract
from xbrain.jev.dashboard import (
    DERIVE_END,
    DERIVE_START,
    compute_jev_dashboard_data,
    render_jev_dashboard_html,
)
from xbrain.jev.models import PrimaryChoice, TopicAssessment
from xbrain.jev.questions import STATE_KEY, build_topic_questions
from xbrain.jev.report import THRESHOLD_DEPENDENT_KEYS
from xbrain.models import Author, Enrichment, Item, Topic

DT = datetime(2026, 9, 22, tzinfo=timezone.utc)
VOCAB = [
    Topic(slug="ai-coding", description="IA."),
    Topic(slug="startups", description="Empresas."),
]
FALLBACK = "otro"
CHAR_LIMIT = 100_000
#: The one provider `jev.defaults.INPUT_USD_PER_MTOK` prices, so `cost_usd` is a real number.
PRICED_PROVIDER = "typesafe"


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
) -> TopicAssessment:
    """A record whose contract is the one TODAY'S ask would stamp, so it reads as current.

    Built the way `assess.assess_topics` builds it — the state as sent and the digest of the
    questions that went with it — so a change to either composition shows up here as a stale
    fixture instead of a test passing over a contract nobody computes any more.
    """
    state, state_chars = build_topic_state(item, CHAR_LIMIT)
    digest = questions_digest(build_topic_questions(VOCAB, FALLBACK))
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
    }
    options.update(kwargs)
    return compute_jev_dashboard_data(items, assessments, VOCAB, **options)


# --------------------------------------------------------------------------- the blob


def test_blob_aligns_membership_with_topics_and_drops_stale():
    fresh, stale = _item("1"), _item("2", text="old")
    data = _data(
        [fresh, stale],
        {"1": _assessment(fresh), "2": _assessment(stale, contract="e" * 64)},
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


def test_the_template_compares_with_ge_and_keeps_an_unjudged_bucket():
    """The WEAKER check, for a machine with no node: the two rules pinned as text.

    `>` instead of `>=` moves every pair sitting exactly on the threshold, and a missing
    `unjudged` bucket folds retired topics into `backed` or `doubtful`. Both are silent.
    """
    template = _resource("jev.template.html")

    assert "noul >= t" in template
    assert "unjudged" in template
    assert DERIVE_START in template and DERIVE_END in template


# --------------------------------------------------------------------------- browser vs report

_DERIVE_HARNESS = """
const DATA = %s;
%s
console.log(JSON.stringify(buckets(derive(DATA.threshold))));
"""


def _derive_in_node(data: dict[str, Any]) -> dict[str, int]:
    """Run the template's OWN `derive` over `data` in node and return its buckets.

    Only the delimited region is extracted, so nothing here needs a DOM or ECharts: the
    region is the pure half of the script on purpose, and the delimiters are exported by
    `jev/dashboard.py` so this test and the template can never disagree about where it is.
    """
    template = _resource("jev.template.html")
    region = template.split(DERIVE_START, 1)[1].split(DERIVE_END, 1)[0]
    script = _DERIVE_HARNESS % (json.dumps(data, ensure_ascii=False), region)
    out = subprocess.run(
        ["node", "-e", script], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="no JS engine on this machine")
def test_the_browser_derives_the_same_buckets_as_the_report():
    """The page and `xbrain jev report` must never quote different numbers.

    The fixture deliberately carries every bucket at once: a backed pair, a doubtful one, a
    missing candidate, a retired topic and a primary that does not coincide. A `derive` that
    mirrors only the easy half still passes an all-agreeing corpus.
    """
    backed = _item("1", topics=("ai-coding", "web3"))
    doubtful = _item("2", text="Seed round", topics=("startups",))
    data = _data(
        [backed, doubtful],
        {
            "1": _assessment(backed),  # ai-coding 0.9 backed · startups 0.12 missing? no: <t
            "2": _assessment(
                doubtful, membership={"ai-coding": 0.91, "startups": 0.4}
            ),  # startups doubtful, ai-coding a missing candidate, primary disagrees
        },
    )

    assert _derive_in_node(data) == {
        "backed": data["summary"]["assigned_backed"],
        "doubtful": data["summary"]["doubtful_pairs"],
        "missing": data["summary"]["missing_pairs"],
        "unjudged": data["summary"]["assigned_unjudged"],
        "jev_pairs": data["summary"]["jev_pairs"],
        "primary_agree": data["summary"]["primary_agree"],
        "assigned_pairs": data["summary"]["assigned_pairs"],
    }


@pytest.mark.skipif(shutil.which("node") is None, reason="no JS engine on this machine")
def test_the_browser_recomputes_the_buckets_when_the_threshold_moves():
    """At another threshold the page must still agree with what the report WOULD print.

    Pinning only the default threshold would certify a `derive` that ignores its argument —
    the one function whose whole job is to take it.
    """
    item = _item("1", topics=("ai-coding", "startups"))
    rows = _data([item], {"1": _assessment(item)}, threshold=0.10)

    assert _derive_in_node(rows)["backed"] == 2  # 0.9 and 0.123456 both clear 0.10
    assert _derive_in_node(rows)["doubtful"] == 0
