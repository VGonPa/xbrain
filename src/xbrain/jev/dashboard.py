"""The `jev.html` page: a self-contained HTML report of Jev against `enrich`.

Rendered through the same mechanism as `dashboard.html` — `render_dashboard_html` with a
second template, the vendored ECharts injected through the same sentinels. Nothing is fetched
at runtime except the Google Fonts stylesheet, exactly as `dashboard.html`: the data and the
library are in the file.

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
  The price is therefore a LOCALE SEAM: the fragment is formatted in Python with a decimal
  point and no grouping, while every number the template formats goes through its own es-ES
  `nf` (decimal comma, thousands point). One definition of the bill beats one separator, and
  the next reader should not "fix" it by re-formatting the fragment in the template.

MEMBERSHIP IS SHIPPED UNROUNDED. Rounding `m` for size would be a genuine divergence and not
a cosmetic one: a noul of 0.8496 renders as 0.85 and then clears a 0.85 slider, so the page
would call backed what the report calls doubtful — and the consistency banner would fire on
the data instead of on the logic it exists to police. Display rounding is the template's job.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from xbrain.dashboard import _resource, humanize_topic, render_dashboard_html
from xbrain.jev.assess import CurrentPairs, current_pairs
from xbrain.jev.models import TopicAssessment
from xbrain.jev.report import (
    THRESHOLD_DEPENDENT_KEYS,
    build_report,
    cost_fragment,
    primary_rank,
)
from xbrain.models import Item, Topic

#: The post text carried into the blob, in characters. Long enough to recognise the post in a
#: queue row, short enough that the whole corpus of them stays a fraction of the page: measured
#: at 0.42 MB of the 4.53 MB page for 2,609 items (see `compute_jev_dashboard_data`).
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

#: The delimiters around the template's BOOT GUARD — the first statements the script runs, and
#: the only ones that may not depend on anything declared later. Extracted the same way, so the
#: guard is exercised rather than merely read: a guard installed after the work it guards is not
#: a guard, and one that calls a helper declared below it has the same hole again.
GUARD_START = "/* ===== boot guard ===== */"
GUARD_END = "/* ===== end boot guard ===== */"


def _snippet(text: str, width: int = _TEXT_CHARS) -> str:
    """The post on one line, cut to `width` — `width - 1` characters plus an ellipsis.

    A post that simply stops mid-word reads as a BROKEN record rather than as a cut one, and
    the drawer is the "see the whole item" surface, which makes the silent version worse here
    than in a queue row.

    THE RULE is `report._snippet`'s — cut to `width - 1`, then an ellipsis — applied at this
    surface's OWN width: 80 characters in a markdown table cell, 240 in a queue row. A cell is
    scanned, a row is read to recognise the post, so the budgets differ on purpose; what must
    not differ is that both say when they cut.

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

    `primary_rank` is IMPORTED rather than re-derived from `ranked` above, even though the
    sort is already in hand: the rank rule (descending probability, ties by option name, so two
    runs of one distribution can never report different ranks) is `report`'s, and a second copy
    is the one that drifts.

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
        # How many options the Choice actually offered, so the drawer can say `5 de 46`
        # instead of presenting a cut as the whole distribution.
        "options": len(assessment.primary.probabilities),
        # enrich's primary, even when the cut left it out: its probability and its 1-based
        # rank. Without them the section answers Jev's question and not the reader's — the
        # bars show Jev's five favourites and nothing says whether enrich's pick came sixth
        # or last, which is the distinction `report.primary_rank` exists to draw.
        "pp": (
            None
            if enriched is None or enriched.primary_topic is None
            else assessment.primary.probabilities.get(enriched.primary_topic)
        ),
        "pr": primary_rank(
            None if enriched is None else enriched.primary_topic,
            assessment.primary.probabilities,
        ),
        "truncated": assessment.truncated,
        "model": assessment.model,
        "provider": assessment.provider,
    }


def _totals(
    summary: dict[str, Any], items: int, assessed: int, current: CurrentPairs
) -> dict[str, Any]:
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
    now: datetime | None = None,
    current: CurrentPairs | None = None,
) -> dict[str, Any]:
    """Pure: items + side-car + vocabulary in, the JSON blob the template consumes out.

    Currency is decided by `assess.current_pairs`, the same call `jev report` goes through, and
    the counts it drops travel into `totals` rather than being discarded. Stale records are
    EXCLUDED from `items`: a comparison against a question Jev is no longer asked is not a
    weaker signal, it is a wrong one.

    `current` is that decision, HANDED IN. A caller that already holds it — `cli._jev_pairs`
    computes it to decide whether to refuse at all — would otherwise pay a second
    `build_topic_state` and sha256 over the whole corpus for an answer it has. Omitted, it is
    computed here, which is what keeps this function callable with nothing but its arguments.
    Both paths must produce the same blob, and a test asserts it.

    `char_limit` is then unused — it is an input to that computation and to nothing else —
    but `fallback` is NOT: it ships into the blob, where the page names it. A hand-in
    computed under other options is REFUSED rather than trusted, because the counts look
    identical whatever they were decided with, so the swap has no symptom: the page would
    show one fallback and a currency verdict reached under another.

    `now` is the clock, threaded from the caller rather than read here: `_summarize` stamps
    `generated_at`, and a function that reads the clock inside itself is not a pure function of
    its arguments and cannot be asserted against.

    SIZE, measured rather than asserted (2026-09-22, the live corpus): 4,531,327 bytes — about
    4.53 MB — for 2,609 items × 45 topics. ECharts 1.03 MB, the JSON blob 3.43 MB, of which the
    unrounded memberships are 1.16 MB, the note deep links 0.45 MB and the post text 0.42 MB.
    (Rendered with synthetic six-decimal nouls, so the membership figure is an upper bound; a
    provider that answers in two or three decimals ships less. The deep links are present only
    where `xbrain generate` has written the note, so a vault without notes is ~0.45 MB lighter
    — an earlier reading of 4.05 MB was this same page measured with none of them.) The
    memberships are the deliberate cost and the module docstring says why; `_TEXT_CHARS` is the
    cheap lever if the page ever has to shrink. ARCHITECTURE.md § jev quotes this measurement.

    Each row's `m` ships the membership UNROUNDED — the page costs a few hundred KB more for
    it, and rounding to three decimals would let a `0.8496` render as `0.85` and then clear a
    `0.85` slider, so the browser would call backed what `report.py` calls doubtful and the
    template's consistency banner would report a divergence in the DATA as one in the logic.
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
    # `build_report` is the entry point, and the summary is the half this page needs: the rows
    # a reader clicks are rebuilt in the browser at whatever threshold the slider is on, so the
    # server-side comparisons would only be the default threshold's, redone.
    summary, _comparisons = build_report(
        pairs, vocab, threshold, now=now, stale=current.stale, orphans=current.orphans
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
