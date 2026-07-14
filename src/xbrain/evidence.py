"""What counts as EVIDENCE for a generated output — one definition, four consumers.

THE PROBLEM THIS EXISTS TO KILL. Four components each need to know what may support a
claim in a generated `summary` / `digest` / `topics`:

  1. the GENERATOR — what the worksheet (or the `api` prompt) actually hands the agent
  2. the RUBRIC    — what the judge is told may support a claim
  3. the JUDGE     — what `verification._source_text` actually puts in front of it
  4. the CHECKER   — what the deterministic entity-grounding check searches for a name

Each used to keep its OWN hand-written list, and nothing bound them. The suite was green
while the four contradicted one another, because every change tested only its own side.
The contradictions were real and measured in the corpus: the judge was handed the linked
article for a DIGEST whose generator never receives it (so it excused inventions the
generator had no way to source), and neither generator shipped the author display name
that the rubric promised the judge.

THE INVARIANT. `evidence_surfaces(item, target)` is the single source of truth:

    generator fields  ⊇  evidence_surfaces(item, target)
    judge source      ==  evidence_surfaces(item, target)
    checker evidence  ==  evidence_surfaces(item, target)
    verify rubric     declares every surface it admits

`tests/test_evidence_contract.py` asserts exactly that, per target, by identity against
this module. Add a surface to one component and forget the others → red.

EVIDENCE IS TARGET-DEPENDENT, and getting it wrong is a bug in BOTH directions. Judge a
digest against the article and you excuse an invention it could not have sourced. Judge a
summary against the digest's narrower set and you flag the generator for using evidence
it was correctly given. So:

* `digest` — the video only, plus the post it arrived in: author metadata · tweet text ·
  video title · transcript · frame descriptions. `export_video_digest_worksheet` ships
  exactly these. (The video TITLE is admitted deliberately: the digest worksheet has
  always shipped it, and the judge's source carries it, so a digest that names the talk
  is grounded, not invented.)
* `summary` / `topics` — the same, PLUS the surfaces the enrich worksheet also ships:
  the poster's own thread · the fetched article's title and body · the image
  descriptions.

A URL IS NOT A SURFACE. A link's URL/domain is topic signal — never a name, never
content. It is deliberately absent from every surface set, so no name can be grounded in
it. This is not pedantry: a summary in the corpus reconstructed a whole article — its
publication ("Axios") and a company ("Anthropic") — out of the slug
`axios.com/2025/05/28/ai-jobs-white-collar-unemployment-anthropic`, of a link that was
never fetched. The judge could not flag it, because its own rubric carved the URL out of
"unsupported". `rubric-verify.md` now says what the generator's rubric says: a domain is
topic signal, never a name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from xbrain.executors.api import (
    _content_image_descriptions,
    _video_frame_descriptions,
    quoted_attribution,
    quoted_source,
    thread_text,
)
from xbrain.models import Item, VerifyTarget
from xbrain.rubrics import ARTICLE_CHAR_LIMIT
from xbrain.worksheet import _link_content_source, _video_source, _video_transcript


@dataclass(frozen=True)
class Surface:
    """One labelled evidence surface present on an item.

    `values` are the ATOMIC pieces of evidence — the handle and the display name, each
    frame description, the article body. `text` is only how the JUDGE renders them.

    The distinction is load-bearing. A generator ships the author's handle and display
    name as two JSON fields and the frame descriptions as a list; the judge renders them
    as `@handle (Name)` and as bullets. Comparing the two by their rendered text would
    make the contract check blind to exactly the surfaces that are shipped as parts —
    which is how the missing display name survived. The contract compares `values`.

    A surface with no values is never built: an empty labelled block would tell the judge
    evidence exists where there is none.
    """

    key: str
    label: str
    values: tuple[str, ...]

    @property
    def text(self) -> str:
        """How the judge's source renders this surface."""
        if self.key == "author":
            handle, *rest = self.values
            name = rest[0] if rest else ""
            return f"@{handle} ({name})" if name else f"@{handle}"
        if self.key in _BULLETED:
            return "\n".join(f"- {value}" for value in self.values)
        if self.key in _ATTRIBUTION_IN_LABEL:
            return self.values[-1]  # the body; the `@handle (Name)` rides in the label
        return "\n".join(self.values)


# Surfaces whose evidence is a LIST of independent items; the judge reads them as bullets.
_BULLETED = frozenset({"video_frames", "images"})

# Surfaces whose attribution rides in a DYNAMIC label (`[Quoted post — @h (N)]`, #98), so
# the label — not the text — names the account. The handle and name stay in `values`
# regardless: the entity CHECKER strips labels, and a quoted author present only in the
# label would leave a CORRECT attribution ("Karpathy announces…") ungrounded and flagged.
# Values last = the body, so `text` can hand the judge exactly what #98 pinned.
_ATTRIBUTION_IN_LABEL = frozenset({"quoted"})


def _author(item: Item) -> tuple[str, ...]:
    """The handle and the display name. Empty when there is no handle — an empty
    `[Author]` block would present a garbage anchor as trusted metadata (#92)."""
    if not item.author.handle:
        return ()
    return tuple(value for value in (item.author.handle, item.author.name) if value)


def _video_title(item: Item) -> tuple[str, ...]:
    source = _video_source(item)
    return (source.title,) if source and source.title else ()


def _article_title(item: Item) -> tuple[str, ...]:
    source = _link_content_source(item)
    return (source.title,) if source and source.title else ()


def _article_body(item: Item) -> tuple[str, ...]:
    source = _link_content_source(item)
    return (source.text[:ARTICLE_CHAR_LIMIT],) if source else ()


def _one(value: str | None) -> tuple[str, ...]:
    """A single-valued surface, dropped when the item has nothing there."""
    return (value,) if value and value.strip() else ()


def _quoted(item: Item) -> tuple[str, ...]:
    """The quoted post (#98): the account that WROTE it, then its body.

    The handle and the display name are evidence in their own right, not decoration —
    they are the grounding for naming the third party a quote-tweet is sharing, and #86's
    attribution rule (the poster is NOT that author) is only enforceable if the judge and
    the checker can both see WHO wrote the quoted words. The body goes LAST: the judge
    renders the attribution in the label (see `_ATTRIBUTION_IN_LABEL`).
    """
    source = quoted_source(item)
    if source is None:
        return ()
    author = source.author
    handle_and_name = (author.handle, author.name) if author else ()
    return tuple(value for value in (*handle_and_name, source.text) if value)


def _quoted_label(item: Item) -> str:
    """`[Quoted post — @handle (Name)]` — the label #98 pinned across every surface, built
    by the SAME `quoted_attribution` the generators and `rubric-verify` read."""
    return f"[{quoted_attribution(item)}]"


# key → (judge-source label, extractor, the phrase `rubric-verify` must use to declare it).
# The rubric phrase is part of the contract: a surface the rubric has no word for cannot
# be bound to it, so `test_evidence_contract` requires one for every key.
#
# The label is a plain string for every surface whose block is fixed, or a `(Item) -> str`
# builder for one whose label carries per-item evidence — today only the quoted post,
# whose label NAMES the account that wrote it (#98).
_SURFACES: dict[str, tuple[str | Callable[[Item], str], Callable[[Item], tuple[str, ...]], str]] = {
    "author": ("[Author]", _author, "author metadata"),
    "video_title": ("[Video title]", _video_title, "video title"),
    "video_transcript": (
        "[Video transcript]",
        lambda item: _one(_video_transcript(item)),
        "video transcript",
    ),
    "video_frames": (
        "[Video frames shown]",
        lambda item: tuple(_video_frame_descriptions(item)),
        "video frame descriptions",
    ),
    "images": (
        "[Images in the post]",
        lambda item: tuple(_content_image_descriptions(item)),
        "image descriptions",
    ),
    "article_title": ("[Linked article title]", _article_title, "fetched article title"),
    "article": ("[Linked article]", _article_body, "fetched article body"),
    "thread": (
        "[Thread — full text, same author]",
        lambda item: _one(thread_text(item)),
        "poster's own thread",
    ),
    "quoted": (_quoted_label, _quoted, "quoted post"),
    "tweet": ("[Tweet]", lambda item: _one(item.text), "tweet text"),
}

# The surfaces each target's GENERATOR is handed — in the order the judge reads them.
# `digest` is the video and the post it arrived in; `summary`/`topics` add everything the
# enrich worksheet also ships. Declared per TARGET (no item needed), because the rubric
# binding and the contract fingerprint need the set before any item is in hand.
_DIGEST_KEYS: tuple[str, ...] = (
    "author",
    "video_title",
    "video_transcript",
    "video_frames",
    # 45 of the 235 video items are ALSO quote-tweets (#87). The digest worksheet ships
    # the quoted post, so the judge must admit it — and on a quote-tweet the clip is very
    # often the QUOTED account's, which makes it the attribution evidence that keeps
    # "posted by the speaker's own account" from naming the wrong person.
    "quoted",
    "tweet",
)
_ENRICH_KEYS: tuple[str, ...] = (
    "author",
    "video_title",
    "video_transcript",
    "video_frames",
    "images",
    "article_title",
    "article",
    "thread",
    "quoted",
    "tweet",
)

SURFACE_KEYS: dict[VerifyTarget, tuple[str, ...]] = {
    "digest": _DIGEST_KEYS,
    "summary": _ENRICH_KEYS,
    "topics": _ENRICH_KEYS,
}

SURFACE_RUBRIC_PHRASES: dict[str, str] = {key: spec[2] for key, spec in _SURFACES.items()}


def evidence_surfaces(item: Item, target: VerifyTarget) -> list[Surface]:
    """The labelled surfaces that may support a claim in `item`'s `target` output.

    Only surfaces the item actually carries are returned: an absent one contributes
    nothing rather than an empty block. This is the ONE definition — the generators, the
    judge, the rubric and the entity checker all resolve "is this evidence?" through it.
    """
    surfaces: list[Surface] = []
    for key in SURFACE_KEYS[target]:
        label_spec, extract, _phrase = _SURFACES[key]
        values = tuple(value for value in extract(item) if value and value.strip())
        if values:
            label = label_spec(item) if callable(label_spec) else label_spec
            surfaces.append(Surface(key=key, label=label, values=values))
    return surfaces


def evidence_text(item: Item, target: VerifyTarget) -> str:
    """Every admitted surface's atomic VALUES as one blob — what the deterministic entity
    check searches when it asks "does this name appear on ANY evidence surface?".

    Built from `values`, not from `text`. The two differ for exactly the surfaces whose
    rendering hides an atom: the quoted post's author is rendered into the judge's LABEL
    (`[Quoted post — @karpathy (Andrej Karpathy)]`), and the checker strips labels — so a
    text-based blob would omit the quoted author, and the checker would flag "Karpathy
    announces he is leaving OpenAI" as an ungrounded name on the very item that grounds
    it. Searching the values is what "grounded in what the item SAYS" actually means.

    Labels stay excluded: they are the judge's scaffolding, not content.
    """
    return "\n".join(
        value for surface in evidence_surfaces(item, target) for value in surface.values
    )
