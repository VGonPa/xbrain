"""The `jev.html` page: where enrich's topics and Jev disagree, post by post, and what it cost.

ONE JOB. The page answers "which posts have topics I should fix, and what did Jev cost me to
find out", in plain words. It does that at ONE threshold — `[jev].threshold` from config — and
recomputes nothing in the browser: the rows arrive ordered (most disagreement first, then
by id — decided here), and the browser only filters and searches them.

NOTHING HERE RE-IMPLEMENTS A NUMBER.

* The comparison is `report.build_report`, the entry point `xbrain jev report` also goes
  through, at the same threshold, over the same current pairs. The headline numbers are its
  `summary`, shipped whole, and each post row is built from its `ItemComparison` — which
  topics are doubtful, missing or unjudged is read off the comparison, never re-derived.
* The cost is `report.run_history` (the run log, priced now from its tokens) and
  `report.assessment_cost_usd` / `report.post_cost_view` (a post's own stored tokens). One
  price formula for everything, and `ItemComparison.disagreements` is what a row counts.

Rendered through the same mechanism as `dashboard.html` — `render_dashboard_html` with a
second template — but without ECharts: the page is a table and a few numbers, so the library
is not injected. Nothing is fetched at runtime except the Google Fonts stylesheet.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from xbrain.dashboard import _resource, humanize_topic, render_dashboard_html
from xbrain.jev.assess import CurrentPairs, current_pairs
from xbrain.jev.defaults import unpriced
from xbrain.jev.models import JevRun, TopicAssessment
from xbrain.jev.report import (
    ItemComparison,
    assessment_cost_usd,
    bill,
    build_report,
    chose_fallback,
    post_cost_view,
    run_history,
)
from xbrain.models import Item, Topic

#: The post text carried into the blob, in characters: long enough to recognise the post in a
#: row, short enough that the whole corpus of them stays a fraction of the page.
_TEXT_CHARS = 240
#: Where the page's "docs" link points: the operator guide, which explains every number here.
DOCS_URL = "https://github.com/VGonPa/xbrain/blob/develop/docs/jev.md"


def _snippet(text: str, width: int = _TEXT_CHARS) -> str:
    """The post on one line, cut to `width` — `width - 1` characters plus an ellipsis.

    `report._snippet`'s rule at this surface's own width. Not called, because that one also
    escapes for a markdown table cell, and this destination is HTML: the template escapes.
    """
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= width else one_line[: width - 1] + "…"


def _enrich_marks(comparison: ItemComparison, membership: dict[str, float]) -> list[dict[str, Any]]:
    """Each topic enrich assigned, in enrich's order: backed (`True`), doubtful (`False`) or
    never asked (`None`) — read off the comparison's own buckets."""
    doubtful = {pair.slug for pair in comparison.doubtful}
    unjudged = set(comparison.unjudged)

    def mark(slug: str) -> bool | None:
        if slug in unjudged:
            return None
        return slug not in doubtful

    return [
        {"slug": slug, "p": membership.get(slug), "ok": mark(slug)} for slug in comparison.assigned
    ]


def _post_row(
    item: Item,
    assessment: TopicAssessment,
    comparison: ItemComparison,
    slugs: set[str],
    id2note: dict[str, str],
) -> dict[str, Any]:
    """One table row, RESHAPED from `report`: what enrich said, where Jev disagrees
    (`ItemComparison.disagreements` — the one definition), and what this answer cost
    (`report.assessment_cost_usd`: `None` for unknown usage or an unpriced provider)."""
    return {
        "id": item.id,
        "handle": item.author.handle,
        "text": _snippet(item.text),
        "url": item.url,
        "note": id2note.get(item.id),
        "enrich": _enrich_marks(comparison, assessment.membership),
        "adds": [{"slug": pair.slug, "p": pair.noul} for pair in comparison.jev_only],
        "primary": comparison.primary_topic,
        "jev_primary": comparison.jev_primary,
        "jev_fallback": chose_fallback(comparison, slugs),
        "primary_agrees": comparison.primary_agrees,
        "disagreements": comparison.disagreements,
        "tokens": assessment.input_tokens,
        "cost_usd": assessment_cost_usd(assessment),
        "unpriced": bool(unpriced([assessment.provider])),
        "truncated": assessment.truncated,
    }


def _cost_block(
    runs: Sequence[JevRun],
    assessments: dict[str, TopicAssessment],
    current: list[TopicAssessment],
    runs_error: str | None,
) -> dict[str, Any]:
    """The cost strip's data: the run history (or the error that kept it out), the mean per
    post, and the CURRENT answers — counted and priced as one set, the set `summary` covers
    (`bill` is unrounded; the summary's `cost_usd` is rounded for the JSON report)."""
    block: dict[str, Any] = {
        "per_post": post_cost_view(current),
        "current": bill(current),
    }
    if runs_error is not None:
        block["error"] = runs_error
    else:
        block.update(run_history(runs, assessments))
    return block


def compute_jev_dashboard_data(
    items: list[Item],
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    *,
    threshold: float,
    fallback: str,
    char_limit: int,
    id2note: dict[str, str],
    updated: str,
    runs: Sequence[JevRun],
    runs_error: str | None = None,
    now: datetime | None = None,
    current: CurrentPairs | None = None,
) -> dict[str, Any]:
    """Pure: items + side-car + run log + vocabulary in, the JSON blob the template reads.

    Currency is `assess.current_pairs`, the call `jev report` goes through; stale and orphaned
    records are EXCLUDED from the rows and COUNTED in `totals`. `current` is that decision
    handed in by a caller that already holds it (`cli._jev_pairs`); one computed under other
    options is REFUSED, because the counts look the same whatever produced them.

    `assessments` is the RAW side-car on purpose: the cost history prices every record that
    was paid for, stale or not (`report.run_history`).

    `runs_error` is why the run log could not be read (a corrupt line). The page then shows
    that message in place of the cost strip and keeps everything else: one torn line must
    not cost the operator the disagreement table.

    `now` is the clock, threaded from the caller so `summary["generated_at"]` is a function
    of the arguments.
    """
    if current is None:
        current = current_pairs(items, assessments, vocab, fallback=fallback, char_limit=char_limit)
    elif (current.fallback, current.char_limit) != (fallback, char_limit):
        raise ValueError(
            f"`current` se calculó con fallback={current.fallback!r} y "
            f"char_limit={current.char_limit}, pero el blob se construye con "
            f"fallback={fallback!r} y char_limit={char_limit}"
        )
    pairs = list(current.pairs)
    summary, comparisons = build_report(
        pairs, vocab, threshold, now=now, stale=current.stale, orphans=current.orphans
    )
    by_id = {item.id: (item, assessment) for item, assessment in pairs}
    slugs = {topic.slug for topic in vocab}
    posts = [
        _post_row(*by_id[comparison.item_id], comparison, slugs, id2note)
        for comparison in comparisons
    ]
    # Most disagreement first; ties by id so two renders of one side-car are identical.
    posts.sort(key=lambda post: (-post["disagreements"], post["id"]))
    return {
        "updated": updated,
        "threshold": threshold,
        "fallback": fallback,
        "docs_url": DOCS_URL,
        "topics": [
            {"slug": t.slug, "label": humanize_topic(t.slug), "description": t.description}
            for t in vocab
        ],
        # `xbrain jev report`'s numbers, whole: the page quotes them and computes none.
        "summary": summary,
        "totals": {
            "items": len(items),
            "assessed": len(assessments),
            "current": len(pairs),
            "stale": current.stale,
            "orphans": current.orphans,
            "compared": summary["items_compared"],
            "not_compared": summary["items_assessed"] - summary["items_compared"],
            "models": summary["models"],
        },
        "cost": _cost_block(runs, assessments, [a for _, a in pairs], runs_error),
        "posts": posts,
    }


def render_jev_dashboard_html(data: dict[str, Any]) -> str:
    """Inject the blob into `jev.template.html` (same sentinel as the dashboard; no ECharts)."""
    return render_dashboard_html(data, template=_resource("jev.template.html"), echarts="")
