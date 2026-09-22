"""The `jev.html` page: a self-contained HTML report of Jev against `enrich`.

Rendered through the same mechanism as `dashboard.html` — `render_dashboard_html` with a
second template, the vendored ECharts injected through the same sentinels, no network.

THE SLIDER IS THE REASON THIS MODULE SHIPS FACTS AND NOT VERDICTS. `jev report` prints one
threshold; this page lets a reader move it, so every threshold-dependent number is recomputed
in the browser from the raw probabilities. What `compute_jev_dashboard_data` ships is the
membership row as Jev answered it, plus the server-side `summary` at the DEFAULT threshold so
the page can check its own arithmetic against `jev/report.py` on load (see `selfCheck` in the
template, and `report.THRESHOLD_DEPENDENT_KEYS`, which says exactly which numbers the slider
invalidates).

TWO THINGS ARE DELIBERATELY NOT RE-IMPLEMENTED HERE.

* Every number comes from `report.build_report`. The comparison between Jev and `enrich` has
  ONE definition, and a dashboard that recomputed it in Python would be a second one that
  drifts — the browser's recompute is already one mirror too many, which is why it is checked
  against the summary rather than trusted.
* The bill is `report.cost_fragment`, the same Spanish sentence `jev topics`, `jev report` and
  `topics-report.md` print. A KPI that formats it itself is a fourth definition of the bill.

MEMBERSHIP IS SHIPPED UNROUNDED. Rounding `m` for size would be a genuine divergence and not
a cosmetic one: a noul of 0.8496 renders as 0.85 and then clears a 0.85 slider, so the page
would call backed what the report calls doubtful — and the consistency banner would fire on
the data instead of on the logic it exists to police. Display rounding is the template's job.
"""

from __future__ import annotations

from typing import Any

from xbrain.dashboard import _resource, humanize_topic, render_dashboard_html
from xbrain.jev.assess import current_pairs
from xbrain.jev.models import TopicAssessment
from xbrain.jev.report import THRESHOLD_DEPENDENT_KEYS, build_report, cost_fragment
from xbrain.models import Item, Topic

#: The post text carried into the blob, in characters. Long enough to recognise the post in a
#: queue row, short enough that ~3,000 of them do not double the page.
_TEXT_CHARS = 240
#: How much of the Choice distribution the drawer shows. `PrimaryChoice.probabilities` can
#: carry one entry per vocabulary topic plus the fallback; the tail is noise.
_TOP_CHOICES = 5

#: The delimiters around the template's PURE half — the part that mirrors `jev/report.py` and
#: touches neither the DOM nor ECharts. `tests/test_jev_dashboard.py` extracts exactly this
#: region and runs it through node against the same fixture the summary was built from, which
#: is what turns "the template is the one file pytest cannot test-drive" into a file whose
#: load-bearing half is test-driven. They live here so the test and the template can never
#: disagree about where the region starts.
DERIVE_START = "/* ===== derive: mirrors jev/report.py ===== */"
DERIVE_END = "/* ===== end derive ===== */"


def _snippet(text: str, width: int = _TEXT_CHARS) -> str:
    """The post on one line, cut to `width` — `width - 1` characters plus an ellipsis.

    A post that simply stops mid-word reads as a BROKEN record rather than as a cut one, and
    the drawer is the "see the whole item" surface, which makes the silent version worse here
    than in a queue row. The cut rule is `report._snippet`'s, so the same post is cut at the
    same place in the markdown report and on the page.

    MIRRORED RATHER THAN CALLED, and the reason is the half that does not travel:
    `report._snippet` finishes by running `_escape_cell` over the result, doubling backslashes
    and escaping pipes so the text survives a markdown table row. Shipping that into the JSON
    blob would put a literal `a\\|b` on screen. Escaping belongs to the destination, and this
    destination is HTML — the template escapes for it (`esc`), at render time.
    """
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= width else one_line[: width - 1] + "…"


def _row(
    item: Item, assessment: TopicAssessment, slugs: list[str], id2note: dict[str, str]
) -> dict[str, Any]:
    """One item as the template consumes it: what `enrich` said, what Jev answered.

    `m` is POSITIONAL over the vocabulary and carries `null`, never `0.0`, for a slug absent
    from `membership` — the same distinction `report._doubtful` draws by skipping it. A
    fabricated zero would put a topic Jev was never asked about at the top of the "most
    doubtful" queue as the strongest disagreement in the corpus.

    `cmp` is whether the item can be compared at all (`report.compare_item` returns None
    without an enrichment). It is shipped rather than inferred from `primary`, because an item
    enrich left WITHOUT a primary topic is still compared — its assigned topics still have
    memberships to back — and reading `primary === null` as "skip" would silently drop it.
    """
    enriched = item.enriched
    ranked = sorted(assessment.primary.probabilities.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "id": item.id,
        "handle": item.author.handle,
        "text": _snippet(item.text),
        "url": item.url,
        "note": id2note.get(item.id),
        "cmp": enriched is not None,
        "assigned": list(enriched.topics) if enriched else [],
        "primary": enriched.primary_topic if enriched else None,
        "m": [assessment.membership.get(slug) for slug in slugs],
        "jp": assessment.primary.choice,
        "jc": round(assessment.primary.confidence, 3),
        "top": [[option, round(p, 3)] for option, p in ranked[:_TOP_CHOICES]],
        "truncated": assessment.truncated,
        "model": assessment.model,
        "provider": assessment.provider,
    }


def _totals(summary: dict[str, Any], items: int, assessed: int, current: Any) -> dict[str, Any]:
    """The header facts: how much of the side-car this page is about, and what it cost.

    `assessed`, `current`, `stale` and `orphans` are a real partition (`assessed == current +
    stale + orphans`), and all four are shipped because a side-car retired by a vocabulary
    edit must never render identically to one nobody ever wrote — the second reading sends an
    operator to re-pay for the whole corpus.
    """
    return {
        "items": items,
        "assessed": assessed,
        "current": len(current.pairs),
        "stale": current.stale,
        "orphans": current.orphans,
        "models": summary["models"],
        "providers": summary["providers"],
        "truncated": summary["truncated"],
        "input_tokens": summary["input_tokens"],
        "input_tokens_unknown": summary["input_tokens_unknown"],
        "cost_usd": summary["cost_usd"],
        "unpriced_providers": summary["unpriced_providers"],
        "cost_text": cost_fragment(summary),
    }


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
) -> dict[str, Any]:
    """Pure: items + side-car + vocabulary in, the JSON blob the template consumes out.

    Currency is decided by `assess.current_pairs`, the same call `jev report` goes through, and
    the counts it drops travel into `totals` rather than being discarded. Stale records are
    EXCLUDED from `items`: a comparison against a question Jev is no longer asked is not a
    weaker signal, it is a wrong one.

    Each row's `m` ships the membership UNROUNDED — the page costs a few hundred KB more for
    it, and rounding to three decimals would let a `0.8496` render as `0.85` and then clear a
    `0.85` slider, so the browser would call backed what `report.py` calls doubtful and the
    template's consistency banner would report a divergence in the DATA as one in the logic.
    """
    current = current_pairs(items, assessments, vocab, fallback=fallback, char_limit=char_limit)
    pairs = list(current.pairs)
    # `build_report` is the entry point, and the summary is the half this page needs: the rows
    # a reader clicks are rebuilt in the browser at whatever threshold the slider is on, so the
    # server-side comparisons would only be the default threshold's, redone.
    summary, _comparisons = build_report(
        pairs, vocab, threshold, stale=current.stale, orphans=current.orphans
    )
    slugs = [topic.slug for topic in vocab]
    return {
        "updated": updated,
        "threshold": threshold,
        "fallback": fallback,
        "topics": [
            {"slug": t.slug, "label": humanize_topic(t.slug), "description": t.description}
            for t in vocab
        ],
        "items": [_row(item, assessment, slugs, id2note) for item, assessment in pairs],
        # The page's own oracle: the numbers `xbrain jev report` prints for this side-car at
        # the DEFAULT threshold, and the split saying which of them the slider invalidates.
        # Without both, the browser's recompute has nothing to be wrong against.
        "summary": summary,
        "threshold_dependent_keys": sorted(THRESHOLD_DEPENDENT_KEYS),
        "totals": _totals(summary, len(items), len(assessments), current),
    }


def render_jev_dashboard_html(data: dict[str, Any]) -> str:
    """Inject the blob and ECharts into `jev.template.html` (same sentinels as the dashboard)."""
    return render_dashboard_html(data, template=_resource("jev.template.html"))
