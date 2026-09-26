"""The `jev.html` page: a browser of the whole corpus, enrich vs Jev post by post, and the cost.

ONE JOB. The page answers "which posts have topics I should fix, and what did Jev cost me to
find out", in plain words. It does that at ONE threshold — `[jev].threshold` from config — and
recomputes nothing in the browser: every post arrives as a card, ordered (most disagreement
first, then status, then id — decided here), and the browser only filters, searches and
re-sorts them.

NOTHING HERE RE-IMPLEMENTS A NUMBER.

* The comparison is `report.build_report`, the entry point `xbrain jev report` also goes
  through, at the same threshold, over the same current pairs. The headline and filter counts
  are its `summary`, shipped whole, and each card's Jev-vs-enrich rows are built from its
  `ItemComparison` — which topics are doubtful, missing or unjudged is read off the
  comparison, never re-derived.
* The cost is `report.run_history` (the run log, priced now from its tokens) and
  `report.assessment_cost_usd` / `report.post_cost_view` (a post's own stored tokens).
* What Jev read is `assess.state_surfaces`: the state `build_topic_state` sends, split back
  into its evidence surfaces — the same order and the same cut.

The share card is built from data XBrain already holds: photos are RELATIVE paths into the
vault's `_media/` mirror (the files the notes embed), linked only when the file exists and
never inlined; the quoted post is `quoted_source`'s; the link card is the evidence's own
fetched page. Rendered through the same mechanism as `dashboard.html` — `render_dashboard_html`
with a second template — without ECharts. Nothing is fetched at runtime except the Google
Fonts stylesheet.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from xbrain.dashboard import _resource, humanize_topic, render_dashboard_html
from xbrain.executors.api import quoted_source
from xbrain.generate import VAULT_MEDIA_SUBDIR
from xbrain.jev.assess import CurrentPairs, current_pairs, state_surfaces
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
from xbrain.models import (
    ContentSourceSuccess,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
    Topic,
)
from xbrain.worksheet import _link_content_source

#: How much of each evidence surface, of a quoted post, the static page ships, in characters.
#: Enough to recognise what Jev read; the whole corpus of them must stay a fraction of the
#: page. The full text is what a local server (`xbrain jev serve`) can hand out on demand.
PAGE_SURFACE_CHARS = 600
#: A photo's vision caption, as the page's alt text and tooltip, in characters.
_CAPTION_CHARS = 280
#: Photos and videos per card: the X grid shows four.
_MEDIA_PER_CARD = 4
#: Where the page's "docs" link points: the operator guide, which explains every number here.
DOCS_URL = "https://github.com/VGonPa/xbrain/blob/develop/docs/jev.md"
#: The command a "copiar comando" button completes with a post id — `jev topics --id`.
ASK_COMMAND = "xbrain jev topics --id"
#: Evidence surface → the name the page gives it (the keys are `xbrain.evidence`'s).
SURFACE_LABELS: dict[str, str] = {
    "author": "Autor",
    "video_title": "Título del vídeo",
    "video_transcript": "Transcript del vídeo",
    "video_frames": "Fotogramas del vídeo",
    "images": "Descripciones de imágenes",
    "article_title": "Título del artículo enlazado",
    "article": "Artículo enlazado",
    "thread": "Hilo",
    "quoted": "Post citado",
    "tweet": "Tweet",
}
#: Card order after the disagreement count: compared posts, then the ones Jev has not
#: answered for today, then the ones with nothing to compare against.
_STATUS_RANK = {"compared": 0, "stale": 1, "unevaluated": 2, "not_enriched": 3}


def _topic_rows(comparison: ItemComparison, membership: dict[str, float]) -> list[dict[str, Any]]:
    """One row per topic either side holds: enrich's in enrich's order, then Jev's additions,
    strongest first — each verdict read off the comparison's own buckets."""
    doubtful = {pair.slug for pair in comparison.doubtful}
    unjudged = set(comparison.unjudged)

    def verdict(slug: str) -> str:
        if slug in unjudged:
            return "sin_juzgar"
        return "solo_enrich" if slug in doubtful else "coinciden"

    rows = [
        {"slug": slug, "enrich": True, "p": membership.get(slug), "verdict": verdict(slug)}
        for slug in comparison.assigned
    ]
    rows += [
        {"slug": pair.slug, "enrich": False, "p": pair.noul, "verdict": "solo_jev"}
        for pair in comparison.jev_only
    ]
    return rows


def _surfaces(item: Item, char_limit: int) -> list[dict[str, Any]]:
    """What Jev read, surface by surface (`assess.state_surfaces`), each shipped cut to
    `PAGE_SURFACE_CHARS` with its full size and how much of it the state's cut kept."""
    return [
        {
            "key": part.key,
            "label": SURFACE_LABELS.get(part.key, part.label),
            "chars": part.chars,
            "kept": part.kept,
            "text": part.text[:PAGE_SURFACE_CHARS],
        }
        for part in state_surfaces(item, char_limit)
    ]


def _jev_view(
    item: Item,
    assessment: TopicAssessment,
    comparison: ItemComparison,
    slugs: set[str],
    char_limit: int,
) -> dict[str, Any]:
    """Jev vs enrich on ONE post, RESHAPED from `report`: the rows, both primaries, the three
    disagreement kinds (`ItemComparison`'s — the one definition), and what the answer cost
    (`report.assessment_cost_usd`: `None` for unknown usage or an unpriced provider)."""
    return {
        "topics": _topic_rows(comparison, assessment.membership),
        "primary": comparison.primary_topic,
        "jev_primary": comparison.jev_primary,
        "jev_confidence": comparison.jev_confidence,
        "jev_fallback": chose_fallback(comparison, slugs),
        "primary_agrees": comparison.primary_agrees,
        "enrich_only": len(comparison.enrich_only),
        "jev_only": len(comparison.jev_only),
        "primary_differs": comparison.primary_differs,
        "disagreements": comparison.disagreements,
        "tokens": assessment.input_tokens,
        "cost_usd": assessment_cost_usd(assessment),
        "unpriced": bool(unpriced([assessment.provider])),
        "truncated": assessment.truncated,
        "model": assessment.model,
        "asked_at": assessment.asked_at.isoformat(),
        "state_chars": assessment.state_chars,
        "surfaces": _surfaces(item, char_limit),
    }


def _cut(text: str, width: int) -> str:
    """`text` whole when it fits, else `width - 1` characters and an ellipsis."""
    return text if len(text) <= width else text[: width - 1] + "…"


def _media_href(local_path: str, page_dir: Path | None) -> str | None:
    """`_media/<local_path>`, relative to the page — the file the vault's notes embed — or
    `None` when it is not there: a broken image is worse than a placeholder that says so."""
    if page_dir is None:
        return None
    href = f"{VAULT_MEDIA_SUBDIR}/{local_path}"
    return href if (page_dir / href).is_file() else None


def _first_frame(item: Item) -> str | None:
    """The first key-frame still of the post's video, if one was extracted."""
    for source in item.content.sources if item.content else ():
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_video" and source.frames:
            return source.frames[0].local_path
    return None


def _card_media(item: Item, page_dir: Path | None) -> list[dict[str, Any]]:
    """Up to four photos/videos, as local files only: the page fetches nothing from X.

    A photo is its downloaded file; a video is its first key-frame still. Anything without a
    local file stays in the grid as a placeholder (`src: None`), so the card still says the
    post HAD a picture.
    """
    media: list[dict[str, Any]] = []
    for entry in item.media[:_MEDIA_PER_CARD]:
        if isinstance(entry, (MediaPhotoDownloaded, MediaPhotoDescribed)):
            desc = entry.description if isinstance(entry, MediaPhotoDescribed) else ""
            desc = _cut(desc, _CAPTION_CHARS)
            src = _media_href(entry.local_path, page_dir)
            media.append({"type": "photo", "src": src, "desc": desc})
        elif isinstance(entry, (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)):
            frame = _first_frame(item)
            src = _media_href(frame, page_dir) if frame else None
            media.append({"type": "video", "src": src, "desc": ""})
        else:
            media.append({"type": "photo", "src": None, "desc": ""})
    return media


def _quoted_card(item: Item) -> dict[str, Any] | None:
    """The quoted post as the evidence carries it (`quoted_source`), cut for the page."""
    source = quoted_source(item)
    if source is None:
        return None
    return {
        "handle": source.author.handle if source.author else None,
        "name": source.author.name if source.author else None,
        "url": source.url,
        "text": source.text[:PAGE_SURFACE_CHARS],
        "cut": len(source.text) > PAGE_SURFACE_CHARS,
    }


def _link_card(item: Item) -> dict[str, Any] | None:
    """The fetched linked page (`worksheet._link_content_source`, the evidence's own pick)
    with its `kind` — an `x_article` can hold scraped replies, and the card must not pass it
    off as an article — or else the first outbound link, bare."""
    source = _link_content_source(item)
    if source is not None:
        return {
            "url": source.url,
            "domain": urlsplit(source.url).hostname,
            "title": source.title,
            "kind": source.kind,
        }
    if item.links:
        link = item.links[0]
        return {"url": link.url, "domain": link.domain, "title": None, "kind": None}
    return None


def _card(
    item: Item,
    status: str,
    jev: dict[str, Any] | None,
    id2note: dict[str, str],
    page_dir: Path | None,
) -> dict[str, Any]:
    """One post as the browser shows it: the share card, what enrich said, and — for a
    post with a current answer — Jev vs enrich."""
    enriched = item.enriched
    return {
        "id": item.id,
        "status": status,
        "url": item.url,
        "note": id2note.get(item.id),
        "created": item.created_at.isoformat(),
        "author": {"handle": item.author.handle, "name": item.author.name},
        "text": item.text,
        "media": _card_media(item, page_dir),
        "quoted": _quoted_card(item),
        "link": _link_card(item),
        "enrich": (
            {"topics": list(enriched.topics), "primary": enriched.primary_topic}
            if enriched is not None
            else None
        ),
        "jev": jev,
    }


def _cards(
    items: list[Item],
    assessments: dict[str, TopicAssessment],
    pairs: list[tuple[Item, TopicAssessment]],
    comparisons: list[ItemComparison],
    slugs: set[str],
    *,
    char_limit: int,
    id2note: dict[str, str],
    page_dir: Path | None,
) -> list[dict[str, Any]]:
    """Every item as a card, most disagreement first, ties by status then id."""
    current = {item.id: assessment for item, assessment in pairs}
    compared = {comparison.item_id: comparison for comparison in comparisons}

    def card(item: Item) -> dict[str, Any]:
        if item.id in compared:
            view = _jev_view(item, current[item.id], compared[item.id], slugs, char_limit)
            return _card(item, "compared", view, id2note, page_dir)
        if item.id in current:
            status = "not_enriched"
        else:
            status = "stale" if item.id in assessments else "unevaluated"
        return _card(item, status, None, id2note, page_dir)

    cards = [card(item) for item in items]
    cards.sort(
        key=lambda c: (
            -(c["jev"]["disagreements"] if c["jev"] else 0),
            _STATUS_RANK[c["status"]],
            c["id"],
        )
    )
    return cards


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
    page_dir: Path | None = None,
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

    `page_dir` is the folder the page is written to. Photos are linked RELATIVE to it, into
    the vault's `_media/` mirror, and only when the file is there; `None` links none.
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
    slugs = {topic.slug for topic in vocab}
    posts = _cards(
        items,
        assessments,
        pairs,
        comparisons,
        slugs,
        char_limit=char_limit,
        id2note=id2note,
        page_dir=page_dir,
    )
    return {
        "updated": updated,
        "threshold": threshold,
        "fallback": fallback,
        "docs_url": DOCS_URL,
        "ask_command": ASK_COMMAND,
        "surface_chars": PAGE_SURFACE_CHARS,
        "char_limit": char_limit,
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
            # Posts with no CURRENT answer — never asked, or asked under another state or
            # vocabulary: what "Sin evaluar por Jev" filters, and what `jev topics` would ask.
            "unevaluated": len(items) - len(pairs),
            "models": summary["models"],
        },
        "cost": _cost_block(runs, assessments, [a for _, a in pairs], runs_error),
        "posts": posts,
    }


def render_jev_dashboard_html(data: dict[str, Any]) -> str:
    """Inject the blob into `jev.template.html` (same sentinel as the dashboard; no ECharts)."""
    return render_dashboard_html(data, template=_resource("jev.template.html"), echarts="")
