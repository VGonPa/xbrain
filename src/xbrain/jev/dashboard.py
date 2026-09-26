"""The `jev.html` page: a browser of the whole corpus, enrich vs Jev post by post, and the cost.

ONE JOB. The page answers "which posts have topics I should fix, and what did Jev cost me to
find out", in plain words. It does that at ONE threshold — `[jev].threshold` from config — and
recomputes nothing in the browser: every post arrives as a card, ordered (most disagreement
first, then status, then id — decided here), carrying the filter keys it belongs to, and the
browser only filters by those keys, searches and re-sorts.

NOTHING HERE RE-IMPLEMENTS A NUMBER.

* The comparison is `report.build_report`, the entry point `xbrain jev report` also goes
  through, at the same threshold, over the same current pairs. The headline and filter counts
  are its `summary`, shipped whole, and each card's rows and filter keys are built from its
  `ItemComparison` — which topics are doubtful, missing or unjudged is read off the
  comparison, never re-derived. Tests hold each filter's cards equal to its summary count.
* The cost is `report.run_history` (the run log, priced now from its tokens) and
  `report.assessment_cost_usd` / `report.post_cost_view` (a post's own stored tokens).
* What Jev read is `assess.state_surfaces`: the state `build_topic_state` sends, split back
  into its evidence surfaces — the same order and the same cut.

PURE, EXCEPT ONE FUNCTION. `compute_jev_dashboard_data` touches no disk: which media files
exist is `collect_jev_media`'s answer, handed in. `build_page_data` is the IO shell that loads
everything a page needs from a `Config` — the one call `jev dashboard` makes.

The share card is built from data XBrain already holds: photos are RELATIVE paths into the
vault's `_media/` mirror (the files the notes embed), never inlined; the quoted post is
`quoted_source`'s; the link card is the evidence's own fetched page. Rendered through the
same mechanism as `dashboard.html` — `render_dashboard_html` with a second template — without
ECharts. Nothing is fetched at runtime except the Google Fonts stylesheet.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from xbrain.config import Config
from xbrain.dashboard import _resource, humanize_topic, render_dashboard_html
from xbrain.executors.api import quoted_source
from xbrain.generate import VAULT_MEDIA_SUBDIR
from xbrain.jev.assess import CurrentPairs, build_topic_state, current_pairs, state_surfaces
from xbrain.jev.client import JevError
from xbrain.jev.defaults import unpriced
from xbrain.jev.load import JevPairs, load_jev_pairs
from xbrain.jev.models import JevRun, TopicAssessment
from xbrain.jev.report import (
    ItemComparison,
    assessment_cost_usd,
    bill,
    build_report,
    chose_fallback,
    jev_assigned,
    post_cost_view,
    run_history,
)
from xbrain.jev.store import load_runs
from xbrain.models import (
    LINK_CONTENT_KINDS,
    QUOTED_CONTENT_KINDS,
    ContentSourceFailure,
    ContentSourceSuccess,
    Item,
    MediaPhotoDescribed,
    MediaPhotoDownloaded,
    MediaPhotoFailed,
    MediaPhotoPending,
    MediaVideoDownloaded,
    MediaVideoFailed,
    MediaVideoPending,
    Topic,
)
from xbrain.notes_io import note_filename
from xbrain.worksheet import link_content_source

#: How much of each evidence surface, and of a quoted post, the page ships, in characters.
#: Enough to recognise what Jev read; the whole corpus of them must stay a fraction of the page.
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
    "video_transcript": "Transcripción del vídeo",
    "video_frames": "Fotogramas del vídeo",
    "images": "Descripciones de imágenes",
    "article_title": "Título del artículo enlazado",
    "article": "Artículo enlazado",
    "thread": "Hilo",
    "quoted": "Post citado",
    "tweet": "Tweet",
}
#: Surfaces whose text the card already shows, by the card part that shows it: shipping them
#: again would double the heaviest strings on the page.
_SHOWN_ON_CARD = {"tweet": "post", "author": "author", "quoted": "quoted"}
#: Card order after the disagreement count: compared posts, then the ones Jev has not
#: answered for today, then the ones with nothing to compare against.
_STATUS_RANK = {"compared": 0, "stale": 1, "unevaluated": 2, "not_enriched": 3}
_PHOTO_TYPES = (MediaPhotoDownloaded, MediaPhotoDescribed)
_VIDEO_TYPES = (MediaVideoPending, MediaVideoDownloaded, MediaVideoFailed)


@dataclass(frozen=True)
class MediaFiles:
    """Which media `local_path`s exist where: in the page's `_media/` mirror (`mirrored`,
    what the page can show) and in `data/media/` (`downloaded`, what `xbrain generate` would
    mirror). `collect_jev_media` fills it; an empty one says nothing is on disk."""

    mirrored: frozenset[str] = frozenset()
    downloaded: frozenset[str] = frozenset()


def _first_video_frames(item: Item) -> list[str | None]:
    """The first key-frame still of each `x_video` source, in source order (None when that
    source has no extracted frame) — the k-th video of the post pairs with the k-th entry."""
    sources = item.content.sources if item.content else []
    return [
        source.frames[0].local_path if source.frames else None
        for source in sources
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_video"
    ]


def _media_paths(item: Item) -> Iterator[str]:
    """Every `local_path` a card of `item` may show: its photos and its videos' frames."""
    for entry in item.media[:_MEDIA_PER_CARD]:
        if isinstance(entry, _PHOTO_TYPES):
            yield entry.local_path
    yield from (frame for frame in _first_video_frames(item) if frame)


def collect_jev_media(items: Sequence[Item], page_dir: Path, media_root: Path | None) -> MediaFiles:
    """THE disk look-up for the page's media: which photos and frames sit in
    `<page_dir>/_media/` (linkable) and which only in `media_root` (not mirrored yet).

    Kept out of `compute_jev_dashboard_data` so that one stays pure, and so a caller that
    renders repeatedly (a local server) can decide when to pay for ~thousands of `stat`s.
    """
    paths = {path for item in items for path in _media_paths(item)}
    mirror = page_dir / VAULT_MEDIA_SUBDIR
    return MediaFiles(
        mirrored=frozenset(p for p in paths if (mirror / p).is_file()),
        downloaded=frozenset(
            p for p in paths if media_root is not None and (media_root / p).is_file()
        ),
    )


def _cut(text: str, width: int) -> str:
    """`text` whole when it fits, else `width - 1` characters and an ellipsis."""
    return text if len(text) <= width else text[: width - 1] + "…"


def _file(
    local_path: str | None, files: MediaFiles, why_absent: str
) -> tuple[str | None, str | None]:
    """`(src, None)` for a file the page can show, else `(None, why)`: not mirrored yet
    (`xbrain generate` fixes it) or `why_absent` (gone, or never there)."""
    if local_path is None:
        return None, why_absent
    if local_path in files.mirrored:
        return f"{VAULT_MEDIA_SUBDIR}/{local_path}", None
    if local_path in files.downloaded:
        return None, "not_mirrored"
    return None, why_absent


def _photo(entry: Any, files: MediaFiles) -> dict[str, Any]:
    """One photo slot: its file and caption, or why there is none."""
    if isinstance(entry, _PHOTO_TYPES):
        desc = entry.description if isinstance(entry, MediaPhotoDescribed) else ""
        src, why = _file(entry.local_path, files, "missing")
        return {"type": "photo", "src": src, "desc": _cut(desc, _CAPTION_CHARS), "why": why}
    if isinstance(entry, MediaPhotoFailed):
        why = "failed"
    elif isinstance(entry, MediaPhotoPending):
        why = "not_downloaded"
    else:
        why = "missing"
    return {"type": "photo", "src": None, "desc": "", "why": why}


def _card_media(item: Item, files: MediaFiles) -> list[dict[str, Any]]:
    """Up to four photos/videos, as local files only: the page fetches nothing from X.

    A photo is its downloaded file; the k-th video is the first still of the k-th `x_video`
    source. Anything without a file the page can show stays as a placeholder with a reason,
    so the card still says the post HAD a picture and what would bring it back.
    """
    frames = iter(_first_video_frames(item))
    media: list[dict[str, Any]] = []
    for entry in item.media[:_MEDIA_PER_CARD]:
        if isinstance(entry, _VIDEO_TYPES):
            src, why = _file(next(frames, None), files, "no_frame")
            media.append({"type": "video", "src": src, "desc": "", "why": why})
        else:
            media.append(_photo(entry, files))
    return media


def _quoted_card(item: Item) -> dict[str, Any] | None:
    """The quoted post as the evidence carries it (`quoted_source`), cut for the page — or,
    for a quote-tweet whose quoted post could not be read, a `missing` card linking to X."""
    source = quoted_source(item)
    if source is not None:
        return {
            "handle": source.author.handle if source.author else None,
            "name": source.author.name if source.author else None,
            "url": source.url,
            "text": source.text[:PAGE_SURFACE_CHARS],
            "cut": len(source.text) > PAGE_SURFACE_CHARS,
            "missing": False,
        }
    failed = _failed_source(item, QUOTED_CONTENT_KINDS)
    if failed is None and item.quoted_id is None:
        return None
    url = failed.url if failed else f"https://x.com/i/status/{item.quoted_id}"
    return {"handle": None, "name": None, "url": url, "text": "", "cut": False, "missing": True}


def _failed_source(item: Item, kinds: frozenset[str]) -> ContentSourceFailure | None:
    """The first source of one of `kinds` whose fetch failed."""
    for source in item.content.sources if item.content else []:
        if isinstance(source, ContentSourceFailure) and source.kind in kinds:
            return source
    return None


def _link_card(item: Item) -> dict[str, Any] | None:
    """The fetched linked page (`worksheet.link_content_source`, the evidence's own pick) with
    its `kind` — an `x_article` can hold scraped replies, and the card must not pass it off
    as an article — else a linked page whose fetch FAILED (said so), else the first link."""
    source = link_content_source(item)
    if source is not None:
        domain = urlsplit(source.url).hostname
        return {
            "url": source.url,
            "domain": domain,
            "title": source.title,
            "kind": source.kind,
            "failed": False,
        }
    failed = _failed_source(item, LINK_CONTENT_KINDS)
    if failed is not None:
        domain = urlsplit(failed.url).hostname
        return {
            "url": failed.url,
            "domain": domain,
            "title": None,
            "kind": failed.kind,
            "failed": True,
        }
    if item.links:
        link = item.links[0]
        return {
            "url": link.url,
            "domain": link.domain,
            "title": None,
            "kind": None,
            "failed": False,
        }
    return None


def _topic_rows(
    comparison: ItemComparison, membership: dict[str, float], slugs: set[str]
) -> list[dict[str, Any]]:
    """One row per topic either side holds — enrich's in enrich's order, then Jev's additions
    strongest first — each verdict read off the comparison's own buckets. Jev's primary gets
    a row of its own when no other row names it (and it is a topic, not the fallback), so a
    topic filter finds the post; it is not a disagreement of its own — the primary line is."""
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
    primary = comparison.jev_primary
    if primary in slugs and all(row["slug"] != primary for row in rows):
        rows.append(
            {
                "slug": primary,
                "enrich": False,
                "p": membership.get(primary),
                "verdict": "primario_jev",
            }
        )
    return rows


def _surfaces(item: Item, char_limit: int) -> list[dict[str, Any]]:
    """What Jev read, surface by surface (`assess.state_surfaces`), with its full size and how
    much of it the state's cut kept. A surface the card already shows ships no text — only
    `same_as`, the card part to look at; the rest ship cut to `PAGE_SURFACE_CHARS`."""
    return [
        {
            "key": part.key,
            "label": SURFACE_LABELS.get(part.key, part.label),
            "chars": part.chars,
            "kept": part.kept,
            "text": None if part.key in _SHOWN_ON_CARD else part.text[:PAGE_SURFACE_CHARS],
            "same_as": _SHOWN_ON_CARD.get(part.key),
        }
        for part in state_surfaces(item, char_limit)
    ]


def _answer(item: Item, assessment: TopicAssessment, char_limit: int) -> dict[str, Any]:
    """What ONE stored answer is, whatever it is compared against: who, when, what it cost
    (`report.assessment_cost_usd`: `None` for unknown usage or an unpriced provider), and
    what Jev read."""
    return {
        "jev_primary": assessment.primary.choice,
        "jev_confidence": assessment.primary.confidence,
        "tokens": assessment.input_tokens,
        "cost_usd": assessment_cost_usd(assessment),
        "unpriced": bool(unpriced([assessment.provider])),
        "truncated": assessment.truncated,
        "model": assessment.model,
        "asked_at": assessment.asked_at.isoformat(),
        "state_chars": assessment.state_chars,
        "surfaces": _surfaces(item, char_limit),
    }


def _compared_view(
    item: Item,
    assessment: TopicAssessment,
    comparison: ItemComparison,
    slugs: set[str],
    char_limit: int,
) -> dict[str, Any]:
    """Jev vs enrich on ONE post, RESHAPED from `report`: the rows, both primaries and the
    disagreement count (`ItemComparison.disagreements`, the one definition)."""
    return {
        **_answer(item, assessment, char_limit),
        "compared": True,
        "topics": _topic_rows(comparison, assessment.membership, slugs),
        "primary": comparison.primary_topic,
        "jev_fallback": chose_fallback(comparison, slugs),
        "primary_differs": comparison.primary_differs,
        "disagreements": comparison.disagreements,
    }


def _uncompared_view(
    item: Item, assessment: TopicAssessment, threshold: float, slugs: set[str], char_limit: int
) -> dict[str, Any]:
    """A paid answer for a post enrich never enriched: Jev's own topics at the threshold
    (`report.jev_assigned`) and its primary. Nothing to disagree with, so nothing counted."""
    membership = assessment.membership
    return {
        **_answer(item, assessment, char_limit),
        "compared": False,
        "topics": [
            {"slug": slug, "enrich": False, "p": membership[slug], "verdict": "jev"}
            for slug in jev_assigned(membership, threshold)
        ],
        "primary": None,
        "jev_fallback": assessment.primary.choice not in slugs,
        "primary_differs": False,
        "disagreements": 0,
    }


def _filter_keys(status: str, comparison: ItemComparison | None, slugs: set[str]) -> list[str]:
    """The page's filters this card belongs to — each key the SAME predicate the report's
    matching count uses (`posts_with_disagreement`, `posts_enrich_only`, `posts_jev_only`,
    `posts_primary_differs`, `primary_fallback`, `items_unassessed`)."""
    if comparison is None:
        return ["uneval"] if status in ("stale", "unevaluated") else []
    flags = (
        ("disc", comparison.disagreements > 0),
        ("enrich_only", bool(comparison.enrich_only)),
        ("adds", bool(comparison.jev_only)),
        ("prim", comparison.primary_differs),
        ("fallback", chose_fallback(comparison, slugs)),
    )
    return [key for key, on in flags if on]


@dataclass(frozen=True)
class _Corpus:
    """What every card is built against, gathered once."""

    current: dict[str, TopicAssessment]
    compared: dict[str, ItemComparison]
    stored: frozenset[str]
    slugs: set[str]
    threshold: float
    char_limit: int
    id2note: dict[str, str]
    media: MediaFiles


def _status(item: Item, corpus: _Corpus) -> str:
    if item.id in corpus.compared:
        return "compared"
    if item.id in corpus.current:
        return "not_enriched"
    return "stale" if item.id in corpus.stored else "unevaluated"


def _jev(item: Item, status: str, corpus: _Corpus) -> dict[str, Any] | None:
    if status == "compared":
        answer, comparison = corpus.current[item.id], corpus.compared[item.id]
        return _compared_view(item, answer, comparison, corpus.slugs, corpus.char_limit)
    if status == "not_enriched":
        answer = corpus.current[item.id]
        return _uncompared_view(item, answer, corpus.threshold, corpus.slugs, corpus.char_limit)
    return None


def _card(item: Item, corpus: _Corpus) -> dict[str, Any]:
    """One post as the browser shows it: the share card, what enrich said, the filters it is
    in, the topics a topic filter matches, and — for a post with a current answer — Jev."""
    status = _status(item, corpus)
    jev = _jev(item, status, corpus)
    enriched = item.enriched
    enrich_topics = list(enriched.topics) if enriched else []
    jev_topics = [row["slug"] for row in jev["topics"]] if jev else []
    return {
        "id": item.id,
        "status": status,
        "in": _filter_keys(status, corpus.compared.get(item.id), corpus.slugs),
        "slugs": list(dict.fromkeys(enrich_topics + jev_topics)),
        "url": item.url,
        "note": corpus.id2note.get(item.id),
        "created": item.created_at.isoformat(),
        "author": {"handle": item.author.handle, "name": item.author.name},
        "text": item.text,
        "media": _card_media(item, corpus.media),
        "quoted": _quoted_card(item),
        "link": _link_card(item),
        "enrich": (
            {"topics": enrich_topics, "primary": enriched.primary_topic} if enriched else None
        ),
        # `jev topics` skips a post with no evidence (`select_items`: `state_chars == 0`), so
        # the page must not offer a command for it.
        "no_evidence": status in ("stale", "unevaluated")
        and build_topic_state(item, corpus.char_limit)[1] == 0,
        "jev": jev,
    }


def _order(card: dict[str, Any]) -> tuple[int, int, str]:
    """Most disagreement first, ties by status then id, so two renders are identical."""
    disagreements = card["jev"]["disagreements"] if card["jev"] else 0
    return (-disagreements, _STATUS_RANK[card["status"]], card["id"])


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
    notes_dir: str | None = None,
    media: MediaFiles | None = None,
) -> dict[str, Any]:
    """Pure: items + side-car + run log + vocabulary in, the JSON blob the template reads.

    EVERY post is a card. Currency is `assess.current_pairs`, the call `jev report` goes
    through: a stale answer is not compared — its post ships as a `stale` card, excluded from
    the numbers and counted in `items_unassessed` / `totals`; an orphaned answer has no post
    and is only counted. `current` is that decision handed in by a caller that already holds
    it (`load.JevPairs.current`); one computed under other options is REFUSED, because the
    counts look the same whatever produced them.

    `assessments` is the RAW side-car on purpose: the cost history prices every record that
    was paid for, stale or not (`report.run_history`).

    `runs_error` is why the run log could not be read (a corrupt line). The page then shows
    that message in place of the cost strip and keeps everything else.

    `now` is the clock, threaded from the caller so `summary["generated_at"]` is a function
    of the arguments. `id2note` maps a post to its note's file name inside `notes_dir` (the
    directory ships once). `media` is `collect_jev_media`'s answer; without it no file is
    known to exist and every picture is a placeholder.
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
        pairs,
        vocab,
        threshold,
        now=now,
        stale=current.stale,
        orphans=current.orphans,
        unassessed=len(items) - len(pairs),
    )
    corpus = _Corpus(
        current={item.id: assessment for item, assessment in pairs},
        compared={comparison.item_id: comparison for comparison in comparisons},
        stored=frozenset(assessments),
        slugs={topic.slug for topic in vocab},
        threshold=threshold,
        char_limit=char_limit,
        id2note=id2note,
        media=media or MediaFiles(),
    )
    return {
        "updated": updated,
        "threshold": threshold,
        "fallback": fallback,
        "docs_url": DOCS_URL,
        "ask_command": ASK_COMMAND,
        "surface_chars": PAGE_SURFACE_CHARS,
        "char_limit": char_limit,
        "notes_dir": notes_dir,
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
        "posts": sorted((_card(item, corpus) for item in items), key=_order),
    }


def note_links(items: Sequence[Item], items_dir: Path) -> tuple[str, dict[str, str]]:
    """The notes directory (absolute: `obsidian://open?path=` needs it) and the file name of
    each note that EXISTS — `jev dashboard` runs independently of `xbrain generate`, and a
    deep link to a note nobody wrote sends a reader to an Obsidian error."""
    names = {item.id: note_filename(item) for item in items}
    return str(items_dir.resolve()), {
        item_id: name for item_id, name in names.items() if (items_dir / name).exists()
    }


def build_page_data(cfg: Config, *, now: datetime, jev: JevPairs | None = None) -> dict[str, Any]:
    """Everything `jev.html` needs, loaded from `cfg` and computed: THE call `jev dashboard`
    makes (and a local server would), so the page's inputs are assembled in one place.

    `jev` is the loader's result when the caller already holds it (the CLI loads first to
    refuse an empty side-car). A run log with a corrupt line does not cost the page: its
    error rides in `cost.error`, and the caller decides how to announce it.
    """
    jev = jev or load_jev_pairs(cfg)
    items = list(jev.store.values())
    runs_error: str | None = None
    try:
        runs = load_runs(cfg.jev_runs_path)
    except JevError as exc:
        runs, runs_error = [], str(exc)
    notes_dir, id2note = note_links(items, cfg.output_dir / "items")
    return compute_jev_dashboard_data(
        items,
        jev.assessments,
        jev.vocab,
        threshold=cfg.jev_threshold,
        fallback=cfg.jev_fallback_option,
        char_limit=cfg.jev_state_char_limit,
        id2note=id2note,
        notes_dir=notes_dir,
        updated=f"{now:%b} {now.day}, {now.year}".upper(),
        runs=runs,
        runs_error=runs_error,
        now=now,
        # The loader already decided currency; recomputing it would be a second
        # `build_topic_state` and sha256 over the whole corpus.
        current=jev.current(),
        media=collect_jev_media(items, cfg.output_dir, cfg.media_dir),
    )


def render_jev_dashboard_html(data: dict[str, Any]) -> str:
    """Inject the blob into `jev.template.html` (same sentinel as the dashboard; no ECharts)."""
    return render_dashboard_html(data, template=_resource("jev.template.html"), echarts="")
