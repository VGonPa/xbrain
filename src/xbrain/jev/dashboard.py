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
* What was asked («Preguntar», `asks`) is the query history (`data/jev/asks/index.json`), each
  query's answers recomputed now by `ask.saved_results` over its own filters — RANKED, every
  current answer, never cut at `[jev].threshold` nor at its `last_min`, which travels only as
  the default of the page's free refine — each through `ask.answer_view`, as compact columns
  (`answer_columns`) with each post's refine keys once (`asks.keys`), its cost
  `report.ask_cost_by_query`, and each topic's post count (`ask.topic_counts`) for the form —
  so the static page shows what the server's `/api/asks` does.

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

import unicodedata
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from xbrain.config import Config
from xbrain.dashboard import _resource, humanize_topic, render_dashboard_html
from xbrain.executors.api import quoted_source
from xbrain.generate import VAULT_MEDIA_SUBDIR
from xbrain.jev.ask import (
    AnswerView,
    JevFilterRefused,
    AskFilters,
    AskQuery,
    answer_view,
    load_history,
    saved_results,
    topic_counts,
)
from xbrain.jev.assess import (
    CUT_MARKER,
    STATE_SURFACE_KEYS,
    CurrentPairs,
    Selection,
    build_topic_state,
    current_pairs,
    questions_digest,
    select_items,
    state_surfaces,
)
from xbrain.jev.client import ChoiceQuestion, JevError, NoulQuestion, Question
from xbrain.jev.defaults import INPUT_USD_PER_MTOK, JEV_DEFAULTS, unpriced
from xbrain.jev.load import JevPairs, load_jev_pairs
from xbrain.jev.models import AskAssessment, AskHistoryEntry, JevRun, TopicAssessment
from xbrain.jev.report import (
    ItemComparison,
    ask_cost,
    ask_cost_by_query,
    assessment_cost_usd,
    bill,
    build_report,
    chose_fallback,
    jev_assigned,
    estimate_selection,
    post_cost_view,
    post_sets,
    report_paths,
    run_history,
)
from xbrain.jev.questions import STATE_KEY, build_topic_questions
from xbrain.jev.store import (
    load_asks,
    load_runs,
)
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
from xbrain.snapshot import is_snapshotted
from xbrain.worksheet import link_content_source

#: How much of each evidence surface, and of a quoted post, the page ships, in characters.
#: Enough to recognise what Jev read; the whole corpus of them must stay a fraction of the page.
PAGE_SURFACE_CHARS = 600
#: How much of an X Article's body the saved copy on a card ships, in characters, cut at a
#: boundary (`_cut_at_boundary`). X's embed of an Article is its bare link, so the page opens
#: it on the saved copy; the whole body is in the vault note («nota ↗») and on X. 201 Articles
#: × this cap is ~0.4 MB of page; whole, they were 2.3 MB.
PAGE_ARTICLE_CHARS = 2000
#: What a cut body ends with.
CUT_MARK = " …"
#: A photo's vision caption, as the page's alt text and tooltip, in characters.
_CAPTION_CHARS = 280
#: Topics enrich put on fewer posts than this sort after the rest in the Topics index, and are
#: marked «pocos datos»: an agreement rate over one or two posts is noise. ONE definition, for
#: every tab that ranks topics (shipped as `topic_min`).
TOPIC_MIN = 5
#: Photos and videos per card: the X grid shows four.
_MEDIA_PER_CARD = 4
#: Where the page's "docs" link points: the operator guide, which explains every number here.
DOCS_URL = "https://github.com/VGonPa/xbrain/blob/develop/docs/jev.md"
#: The guide's section on the Configuración tab (GitHub's anchor for its heading), where the
#: tab's "documentación" link lands.
DOCS_CONFIG_URL = DOCS_URL + "#the-configuración-tab"
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


def _cut_at_boundary(text: str, width: int) -> str:
    """`text` whole when it fits; else at most `width` characters ending in `CUT_MARK`, cut
    at the last paragraph end in the second half of the window, else the last sentence end
    there, else the last space there, else hard (marked with a bare ellipsis)."""
    if len(text) <= width:
        return text
    head = text[: width - len(CUT_MARK)]
    floor = width // 2
    # (the ends a rule cuts at, how many of their characters the kept text ends with)
    rules: tuple[tuple[tuple[str, ...], int], ...] = (
        (("\n\n",), 0),
        ((". ", "! ", "? ", ".\n", "!\n", "?\n"), 1),  # keeps the full stop
        ((" ", "\n"), 0),
    )
    for ends, kept in rules:
        at = max(head.rfind(end) for end in ends)
        if at >= floor:
            return head[: at + kept].rstrip() + CUT_MARK
    return text[: _grapheme_start(text, width - 1)] + "…"


#: Code points that belong to the grapheme before them: joiner, variation selectors, skin
#: tones, emoji tag characters (combining marks are asked of `unicodedata`).
_ZWJ = "\u200d"
_EXTENDS = (("\ufe00", "\ufe0f"), ("\U0001f3fb", "\U0001f3ff"), ("\U000e0020", "\U000e007f"))
_REGIONAL = ("\U0001f1e6", "\U0001f1ff")


def _extends(char: str) -> bool:
    return (
        char == _ZWJ
        or unicodedata.combining(char) > 0
        or any(lo <= char <= hi for lo, hi in _EXTENDS)
    )


def _grapheme_start(text: str, at: int) -> int:
    """`at`, moved back to where the grapheme it falls inside starts, so a hard cut there
    keeps no half of one: not before a mark or a joiner that belongs to what precedes it, not
    after a joiner, not between the two letters of a flag. Close enough to UAX #29 for a cut
    that is only a fallback (the corpus has none)."""
    while 0 < at < len(text) and (_extends(text[at]) or text[at - 1] == _ZWJ):
        at -= 1
    lo, hi = _REGIONAL
    run = 0
    while run < at and lo <= text[at - 1 - run] <= hi:
        run += 1
    if run % 2 and at < len(text) and lo <= text[at] <= hi:
        at -= 1
    return at


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
    for a quote-tweet whose quoted post could not be read, a `missing` card linking to X,
    with `why`: `failed` (a fetch was tried and failed) or `not_fetched` (never tried)."""
    source = quoted_source(item)
    if source is not None:
        return {
            "handle": source.author.handle if source.author else None,
            "name": source.author.name if source.author else None,
            "url": source.url,
            "text": source.text[:PAGE_SURFACE_CHARS],
            "cut": len(source.text) > PAGE_SURFACE_CHARS,
            "missing": False,
            "why": None,
        }
    failed = _failed_source(item, QUOTED_CONTENT_KINDS)
    if failed is None and item.quoted_id is None:
        return None
    url = failed.url if failed else f"https://x.com/i/status/{item.quoted_id}"
    why = "failed" if failed else "not_fetched"
    return {
        "handle": None,
        "name": None,
        "url": url,
        "text": "",
        "cut": False,
        "missing": True,
        "why": why,
    }


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


#: The hosts an X link can have (as `xbrain.fetch`'s, which this page does not import: it
#: loads the article extractor).
_X_HOSTS = frozenset({"x.com", "www.x.com", "twitter.com", "www.twitter.com", "mobile.twitter.com"})


def _is_x_article_url(url: str) -> bool:
    """`x.com/i/article/<id>` (or twitter.com's)."""
    parts = urlsplit(url)
    return (parts.hostname or "").lower() in _X_HOSTS and parts.path.startswith("/i/article/")


def _is_x_article(item: Item) -> bool:
    """The post links an X Article (`x.com/i/article/<id>`). X's embed of such a post shows
    only that link, so the page opens it on the saved copy."""
    return any(_is_x_article_url(link.url) for link in item.links)


def _article_body(item: Item) -> dict[str, Any] | None:
    """An X Article's fetched body for the saved copy, cut to `PAGE_ARTICLE_CHARS` at a
    boundary, with its full size and URL; None for any other post, or when the body was never
    fetched."""
    if not _is_x_article(item) or item.content is None:
        return None
    for source in item.content.sources:
        if (
            isinstance(source, ContentSourceSuccess)
            and source.kind == "x_article"
            and source.text
            and _is_x_article_url(source.url)
        ):
            text = _cut_at_boundary(source.text, PAGE_ARTICLE_CHARS)
            return {
                "text": text,
                "cut": text != source.text,
                "chars": len(source.text),
                # «sigue en X» goes over https, whatever scheme the fetch stored.
                "url": urlsplit(source.url)._replace(scheme="https").geturl(),
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
        "x_article": _is_x_article(item),
        "article": _article_body(item),
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
    per_post: dict[str, Any],
    runs_error: str | None,
) -> dict[str, Any]:
    """The cost strip's data: the run history (or the error that kept it out), the mean per
    post, and the CURRENT answers — counted and priced as one set, the set `summary` covers
    (`bill` is unrounded; the summary's `cost_usd` is rounded for the JSON report)."""
    block: dict[str, Any] = {
        "per_post": per_post,
        "current": bill(current),
    }
    if runs_error is not None:
        block["error"] = runs_error
    else:
        block.update(run_history(runs, assessments))
    return block


def _question_row(key: str, question: Question) -> dict[str, Any]:
    """One question as it goes on the wire: its key, kind, instructions and criteria in wire
    order (pairs, not an object: the order of a Choice's options reaches the model). A kind
    this page has no words for is refused, never shown as a yes/no question."""
    if isinstance(question, ChoiceQuestion):
        kind = "choice"
    elif isinstance(question, NoulQuestion):
        kind = "yes_no"
    else:
        raise TypeError(f"unknown question type for {key!r}: {type(question).__name__}")
    return {
        "key": key,
        "type": kind,
        "instructions": question.instructions,
        "criteria": [[option, text] for option, text in (question.criteria or {}).items()],
    }


def config_view(
    vocab: list[Topic],
    current: list[TopicAssessment],
    *,
    settings: dict[str, Any],
    selection: Selection,
    per_post: dict[str, Any],
    files: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """The Configuración tab's data: every `[jev]` setting beside its default, the models that
    answered the current answers, the price table, the questions EXACTLY as
    `build_topic_questions` sends them and their digest (the contract's half that is not the
    post), the surfaces the state carries in its order with the cut's marker, the files, and
    what another pass would cost (`report.estimate_selection` over `selection`, the posts
    `jev topics` would ask now, at the means `per_post` — the cost strip's own).

    `settings` is `Config.jev_settings()`: `{config key: value in effect}`, keyed like
    `JEV_DEFAULTS`. `models_answered` is `[model, answers]` pairs, most answers first, then
    by name — an order the page shows as shipped.
    """
    questions = build_topic_questions(vocab, settings["fallback_option"])
    models = Counter(a.model for a in current)
    return {
        "settings": [
            {"key": key, "value": settings[key], "default": default}
            for key, default in JEV_DEFAULTS.items()
        ],
        "models_answered": [
            [model, n] for model, n in sorted(models.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "prices": dict(INPUT_USD_PER_MTOK),
        "questions": [_question_row(key, q) for key, q in questions.items()],
        "questions_digest": questions_digest(questions),
        "surfaces": [{"key": k, "label": SURFACE_LABELS[k]} for k in STATE_SURFACE_KEYS],
        "cut_marker": CUT_MARKER.format(dropped="N"),
        "docs_url": DOCS_CONFIG_URL,
        "files": files,
        "estimate": {"per_post": per_post, **estimate_selection(selection, per_post)},
    }


def _pending(
    items: list[Item],
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    fallback: str,
    char_limit: int,
) -> Selection:
    """What `xbrain jev topics` would ask now — `assess.select_items` itself, the `--dry-run`
    count. A SECOND currency pass over the corpus (one `build_topic_state` and sha256 per
    post) beside `current_pairs`'s, on purpose: the estimate must count exactly what the
    command would ask, and only the command's own selection says that (it also skips the
    posts with no evidence). ~0.01 s for the corpus; a test holds the two passes to agree."""
    return select_items(
        {item.id: item for item in items},
        assessments,
        vocab,
        ids=None,
        limit=None,
        force=False,
        fallback=fallback,
        char_limit=char_limit,
    )


def _currency(
    items: list[Item],
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    settings: dict[str, Any],
    current: CurrentPairs | None,
) -> CurrentPairs:
    """`settings` checked to carry exactly the `[jev]` keys, and the currency decision: the
    one handed in (refused if it was made under another fallback or char limit, because the
    counts look the same whatever produced them) or `assess.current_pairs` made here."""
    if set(settings) != set(JEV_DEFAULTS):
        raise ValueError(
            f"settings must carry exactly the [jev] keys {sorted(JEV_DEFAULTS)}, "
            f"got {sorted(settings)}"
        )
    fallback, char_limit = settings["fallback_option"], settings["state_char_limit"]
    if current is None:
        return current_pairs(items, assessments, vocab, fallback=fallback, char_limit=char_limit)
    if (current.fallback, current.char_limit) != (fallback, char_limit):
        raise ValueError(
            f"`current` se calculó con fallback={current.fallback!r} y "
            f"char_limit={current.char_limit}, pero el blob se construye con "
            f"fallback={fallback!r} y char_limit={char_limit}"
        )
    return current


def compute_jev_dashboard_data(
    items: list[Item],
    assessments: dict[str, TopicAssessment],
    vocab: list[Topic],
    *,
    settings: dict[str, Any],
    id2note: dict[str, str],
    updated: str,
    runs: Sequence[JevRun],
    files: list[dict[str, Any]] | None = None,
    runs_error: str | None = None,
    now: datetime | None = None,
    current: CurrentPairs | None = None,
    notes_dir: str | None = None,
    media: MediaFiles | None = None,
    asks: dict[str, Any] | None = None,
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

    `settings` is `Config.jev_settings()`, every `[jev]` key keyed like `JEV_DEFAULTS`: the
    threshold, fallback and char limit build the comparison, and the tab states them all.
    `files` is where the side-car, the run log, the vocabulary, the reports and the page live
    (`page_files`), `None` from a pure caller. `asks` is `asks_view`'s answer (the «Preguntar»
    tab); `NO_ASKS` when none is handed in.
    """
    current = _currency(items, assessments, vocab, settings, current)
    threshold = settings["threshold"]
    fallback, char_limit = settings["fallback_option"], settings["state_char_limit"]
    pairs = list(current.pairs)
    answers = [assessment for _, assessment in pairs]
    per_post = post_cost_view(answers)
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
        # `xbrain jev serve` sets this (its token and `[jev].serve_max_usd`); the file `jev
        # dashboard` writes is not served, and its page keeps «copiar comando».
        "serve": None,
        "surface_chars": PAGE_SURFACE_CHARS,
        "char_limit": char_limit,
        "notes_dir": notes_dir,
        "topic_min": TOPIC_MIN,
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
        "cost": _cost_block(runs, assessments, answers, per_post, runs_error),
        "config": config_view(
            vocab,
            answers,
            settings=settings,
            selection=_pending(items, assessments, vocab, fallback, char_limit),
            per_post=per_post,
            files=files,
        ),
        # The posts behind every pair, primary diagonal and confidence band, for the Topics and
        # Comparar tabs to open. Page-only: the summary (and so `topics-report.json`) keeps
        # the counts.
        "post_sets": post_sets(comparisons),
        "posts": sorted((_card(item, corpus) for item in items), key=_order),
        "asks": asks if asks is not None else NO_ASKS,
    }


#: `asks` when nothing was ever asked (and what a pure caller gets).
NO_ASKS: dict[str, Any] = {
    "history": [],
    "surfaces": {},
    "keys": {},
    "topic_counts": {},
    "error": None,
}
#: The cost of a query no logged pass paid for (every answer came from the cache, or the
#: passes that paid were never logged): `report.ask_cost`'s zero.
_NO_COST = ask_cost([], "")


@dataclass(frozen=True)
class SavedAsk:
    """One query of the history, with its answers as read from its file (`store.load_asks`),
    or why they could not be read."""

    entry: AskHistoryEntry
    records: dict[str, AskAssessment]
    error: str | None = None


@dataclass
class _AskPage:
    """What every row of one build shares: the corpus, THE topic bar, the run log's costs (or
    why it could not be read), and each post's state built at most once."""

    jev: JevPairs
    topic_threshold: float
    char_limit: int
    costs: dict[str, dict[str, Any]] | None
    runs_error: str | None
    #: Posts whose card already carries what Jev read (a current topics answer).
    carded: frozenset[str]
    #: Each post's state as sent and its size before the cut, built at most once.
    states: dict[str, tuple[str, int]] = field(default_factory=dict)
    surfaces: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    #: Per result post, once: `ask.answer_view`'s refine keys plus `n`, the size of the state
    #: Jev read (the «what Jev read» line; cut when above `char_limit`).
    keys: dict[str, dict[str, Any]] = field(default_factory=dict)

    def state(self, item: Item) -> tuple[str, int]:
        if item.id not in self.states:
            state, chars = build_topic_state(item, self.char_limit)
            self.states[item.id] = (state[STATE_KEY], chars)
        return self.states[item.id]

    def state_text(self, item: Item) -> str:
        return self.state(item)[0]


def _ask_cost(page: _AskPage, sha: str) -> dict[str, Any]:
    """A query's bill from the run log, or `{error}` when the log could not be read — never a
    zero that reads as «nothing was paid»."""
    if page.costs is None:
        return {"error": page.runs_error or "no se pudo leer runs.jsonl"}
    return page.costs.get(sha, _NO_COST)


def _minute(moment: datetime) -> str:
    """An instant to the minute, in UTC: as precise as the page shows it."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _most_common(values: list[str]) -> str | None:
    """The value most answers share (the first seen on a tie), or None for no answers."""
    return Counter(values).most_common(1)[0][0] if values else None


def _differs(own: dict[str, str], model: str | None, asked_at: str | None) -> dict[str, str]:
    """What of one answer's model and minute differs from its query's."""
    query = {"model": model, "asked_at": asked_at}
    return {key: value for key, value in own.items() if value != query[key]}


def answer_columns(views: Sequence[AnswerView]) -> dict[str, Any]:
    """A query's answers as the page's blob carries them, in their ranked order: two columns
    (`ids`, `p` — full floats) and the model and minute asked ONCE, the most common of each;
    an answer that differs is in `exceptions` with what differs. ~45 B per answer, where one
    object per answer took ~130 B: a whole-corpus query is ~3,000 answers, and the blob is
    re-sent after every job."""
    minutes = [_minute(view.asked_at) for view in views]
    model = _most_common([view.model for view in views])
    asked_at = _most_common(minutes)
    exceptions: dict[str, dict[str, str]] = {}
    for view, minute in zip(views, minutes, strict=True):
        differs = _differs({"model": view.model, "asked_at": minute}, model, asked_at)
        if differs:
            exceptions[view.id] = differs
    return {
        "ids": [view.id for view in views],
        "p": [view.p for view in views],
        "model": model,
        "asked_at": asked_at,
        "exceptions": exceptions,
    }


def _ask_answers(saved: SavedAsk, page: _AskPage) -> tuple[dict[str, Any], int]:
    """The answers of one query: EVERY current answer over its last filters, ranked by
    `ask.saved_results` (the one order; `[jev].threshold` is the topic bar only), each through
    `ask.answer_view`, as `answer_columns`. The use's minimum is not applied here: it is the
    default of the page's free refine (`row["min"]`), which filters these by `page.keys`
    without asking anything."""
    entry = saved.entry
    found = saved_results(
        page.jev.store,
        page.jev,
        AskQuery.of(entry.query),
        AskFilters.from_json(entry.last_filters),
        saved.records,
        topic_threshold=page.topic_threshold,
        minimum=0.0,
        state_text=page.state_text,
    )
    current = page.jev.current_by_id()
    views = []
    for item, record in found.ranked:
        view = answer_view(item, record, current.get(item.id), page.topic_threshold)
        views.append(view)
        if item.id not in page.keys:
            page.keys[item.id] = {**view.keys, "n": page.state(item)[1]}
        # A post whose card has a Jev block already shows what Jev read: sent once.
        if item.id not in page.carded and item.id not in page.surfaces:
            page.surfaces[item.id] = _surfaces(item, page.char_limit)
    return answer_columns(views), found.answered


def _filters_view(stored: dict[str, Any]) -> dict[str, Any]:
    """A use's filters as the page reads them: `AskFilters.as_json`, so an old entry's single
    `topic` arrives as `topics` like every other. Unreadable, they go as stored (the row
    carries the error)."""
    try:
        return AskFilters.from_json(stored).as_json()
    except ValueError:
        return stored


def _ask_row(saved: SavedAsk, page: _AskPage) -> dict[str, Any]:
    """One query as the «Preguntar» tab lists it. Its results are recomputed NOW: an answer
    whose post changed since is not a result. A row the history rebuilt from the answer file
    (`AskHistoryEntry.rebuilt`) says so; a row that cannot be computed says why."""
    entry = saved.entry
    row: dict[str, Any] = {
        "sha": entry.query_sha,
        "query": entry.query,
        "first_asked_at": entry.first_asked_at.isoformat(),
        "last_asked_at": entry.last_asked_at.isoformat(),
        "times": entry.times,
        "min": entry.last_min,
        "filters": _filters_view(entry.last_filters),
        "rebuilt": entry.rebuilt,
        "answered": 0,
        "answers": answer_columns([]),
        "cost": _ask_cost(page, entry.query_sha),
    }
    if saved.error is not None:
        row["error"] = f"No se pudieron leer las respuestas de esta consulta: {saved.error}"
        return row
    try:
        row["answers"], row["answered"] = _ask_answers(saved, page)
    except JevFilterRefused as exc:
        # A filter the corpus no longer supports (a topic removed from the vocabulary).
        row["error"] = f"El filtro con el que se preguntó ya no aplica: {exc}"
    except JevError as exc:
        row["error"] = f"No se pudieron calcular los resultados de esta consulta: {exc}"
    except ValueError as exc:
        row["error"] = f"No se pudieron leer los filtros de esta consulta: {exc}"
    return row


def asks_view(
    saved: Sequence[SavedAsk],
    jev: JevPairs,
    runs: Sequence[JevRun] | None,
    *,
    topic_threshold: float,
    char_limit: int,
    error: str | None = None,
    runs_error: str | None = None,
) -> dict[str, Any]:
    """The «Preguntar» tab's data: every query asked, the last asked first, each with its
    results and cost. `runs` is `None` when the run log could not be read (`runs_error`);
    `error` is why the history itself could not be read."""
    page = _AskPage(
        jev=jev,
        topic_threshold=topic_threshold,
        char_limit=char_limit,
        costs=ask_cost_by_query(runs) if runs is not None else None,
        runs_error=runs_error,
        carded=frozenset(jev.current_by_id()),
    )
    ordered = sorted(saved, key=lambda s: (s.entry.last_asked_at, s.entry.query_sha), reverse=True)
    history = [_ask_row(one, page) for one in ordered]
    counts = topic_counts(jev.store, AskFilters(), jev=jev, threshold=topic_threshold)
    return {
        "history": history,
        "surfaces": page.surfaces,
        "keys": page.keys,
        "topic_counts": counts,
        "error": error,
    }


def load_saved_asks(cfg: Config) -> tuple[list[SavedAsk], str | None]:
    """The history as `jev ask` reads it (`ask.load_history`: an entry the index lost is rebuilt
    from its answer file, so paid answers are never hidden), and each query's answers. A query
    whose file cannot be read costs its own row; a history that cannot be read is the tab's
    `error`."""
    try:
        # An answer file it cannot read costs the page only that query: `load_history` leaves
        # it out and names it; every other lost query is rebuilt, its answers calibrated.
        index, skipped = load_history(cfg, skip_unreadable=True)
    except JevError as exc:
        return [], str(exc)
    error = "Faltan en la lista, por ilegibles: " + "; ".join(skipped) if skipped else None
    saved: list[SavedAsk] = []
    for entry in index.queries.values():
        try:
            query = AskQuery.of(entry.query)
            if query.sha != entry.query_sha:
                raise JevError(
                    f"el historial guarda {entry.query!r} bajo {entry.query_sha}, que no es su sha"
                )
            records = load_asks(cfg.jev_asks_dir / f"{entry.query_sha}.json", query)
        except (JevError, ValueError) as exc:
            saved.append(SavedAsk(entry, {}, str(exc)))
        else:
            saved.append(SavedAsk(entry, records))
    return saved, error


def ask_page_data(
    cfg: Config, jev: JevPairs, runs: Sequence[JevRun] | None, runs_error: str | None = None
) -> dict[str, Any]:
    """`asks` for a page built from `cfg`: the history loaded, viewed at `[jev].threshold`."""
    saved, error = load_saved_asks(cfg)
    return asks_view(
        saved,
        jev,
        runs,
        topic_threshold=cfg.jev_threshold,
        char_limit=cfg.jev_state_char_limit,
        error=error,
        runs_error=runs_error,
    )


def note_links(items: Sequence[Item], items_dir: Path) -> tuple[str, dict[str, str]]:
    """The notes directory (absolute: `obsidian://open?path=` needs it) and the file name of
    each note that EXISTS — `jev dashboard` runs independently of `xbrain generate`, and a
    deep link to a note nobody wrote sends a reader to an Obsidian error."""
    names = {item.id: note_filename(item) for item in items}
    return str(items_dir.resolve()), {
        item_id: name for item_id, name in names.items() if (items_dir / name).exists()
    }


def page_files(cfg: Config, *, served: bool = False) -> list[dict[str, Any]]:
    """Every file the page reads or is written to, each path from the place its writer or
    reader takes it (`Config`, `report.report_paths`): absolute `path`, a `label` that tells
    `data/topics.json` from `data/jev/topics.json` (relative to `data/`, else the file name),
    whether it `exists` yet (a run log appears with the first paid pass, the reports with
    the first `jev report`), and whether a snapshot of `data/` carries it
    (`snapshot.is_snapshotted`). The page's own row says whether this copy is `served` live
    rather than read from the static file."""
    report_json, report_md = report_paths(cfg.jev_dir)
    paths = (
        ("topics", cfg.jev_topics_path),
        ("runs", cfg.jev_runs_path),
        ("vocab", cfg.vocab_path),
        ("report_json", report_json),
        ("report_md", report_md),
        ("page", cfg.jev_page_path),
    )
    rows: list[dict[str, Any]] = []
    for key, path in paths:
        absolute = path.resolve()
        data_dir = cfg.data_dir.resolve()
        label = (
            absolute.relative_to(data_dir).as_posix()
            if absolute.is_relative_to(data_dir)
            else path.name
        )
        row: dict[str, Any] = {
            "key": key,
            "label": label,
            "path": str(absolute),
            "exists": path.exists(),
            "snapshotted": is_snapshotted(path, cfg.data_dir),
        }
        if key == "page":
            row["served"] = served
        rows.append(row)
    return rows


def build_page_data(
    cfg: Config,
    *,
    now: datetime,
    jev: JevPairs | None = None,
    served: bool = False,
    media: MediaFiles | None = None,
) -> dict[str, Any]:
    """Everything `jev.html` needs, loaded from `cfg` and computed: THE call `jev dashboard`
    makes (and a local server would), so the page's inputs are assembled in one place.

    `jev` is the loader's result when the caller already holds it (the CLI loads first to
    refuse an empty side-car). A run log with a corrupt line does not cost the page: its
    error rides in `cost.error`, and the caller decides how to announce it. `served` is True
    for a page a server renders live, so the page does not call itself the static file.
    `media` is `collect_jev_media`'s answer when the caller keeps one (a server pays for its
    ~thousands of `stat`s once, not per request); looked up here otherwise.
    """
    jev = jev or load_jev_pairs(cfg)
    items = list(jev.store.values())
    runs_error: str | None = None
    try:
        runs = load_runs(cfg.jev_runs_path)
    except JevError as exc:
        runs, runs_error = [], str(exc)
    notes_dir, id2note = note_links(items, cfg.output_dir / "items")
    asks = ask_page_data(cfg, jev, None if runs_error else runs, runs_error)
    return compute_jev_dashboard_data(
        items,
        jev.assessments,
        jev.vocab,
        settings=cfg.jev_settings(),
        id2note=id2note,
        notes_dir=notes_dir,
        updated=f"{now:%b} {now.day}, {now.year}".upper(),
        runs=runs,
        files=page_files(cfg, served=served),
        runs_error=runs_error,
        now=now,
        # The loader already decided which answers are current; handing that in saves
        # `current_pairs` a pass. The Configuración tab's estimate still makes its own
        # (`_pending`, the command's selection) — deliberately, see there.
        current=jev.current(),
        media=collect_jev_media(items, cfg.output_dir, cfg.media_dir) if media is None else media,
        asks=asks,
    )


def render_jev_dashboard_html(data: dict[str, Any]) -> str:
    """Inject the blob into `jev.template.html` (same sentinel as the dashboard; no ECharts)."""
    return render_dashboard_html(data, template=_resource("jev.template.html"), echarts="")
