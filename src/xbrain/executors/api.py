"""The `api` executor — produces enrichment judgment via the Anthropic API.

One API call per item: simple, robust, easy to retry. The Anthropic client is
injected (defaults to a real one) so tests run offline. The user prompt always
carries the link URLs/domains and the bookmark folder — topic signal even when
the article body was not fetched (design §15.2).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Collection

from xbrain.executors.base import EnrichmentJudgment
from xbrain.llm_json import json_from_response
from xbrain.models import (
    LINK_CONTENT_KINDS,
    ContentSourceSuccess,
    Item,
    MediaPhotoDescribed,
    Topic,
)
from xbrain.rubrics import (
    ARTICLE_CHAR_LIMIT,
    FRAME_DESC_CHAR_LIMIT,
    TRANSCRIPT_CHAR_LIMIT,
    load_rubric,
    truncate_transcript,
)

_MAX_TOKENS = 600


def _recoverable_errors() -> tuple[type[Exception], ...]:
    """Exception classes a per-item failure should swallow + log + continue on.

    `anthropic.APIError` covers auth, rate-limit, server-side and network
    errors the SDK normalises. `ValueError` covers validator rejections and
    `pydantic.ValidationError` (a `ValueError` subclass in pydantic v2).
    `json.JSONDecodeError` covers a malformed LLM response. `KeyError` covers
    a response missing an expected field.

    Lazy-imported because `anthropic` is an optional dependency in the test
    environment (the client is faked).
    """
    try:
        from anthropic import APIError

        return (APIError, ValueError, json.JSONDecodeError, KeyError)
    except ImportError:
        return (ValueError, json.JSONDecodeError, KeyError)


def _vocab_block(vocab: list[Topic]) -> str:
    return "\n".join(f"- {t.slug}: {t.description}" for t in vocab)


def _system_prompt(language: str) -> str:
    """The rubrics are the system prompt — the declarative source of truth.

    `language` substitutes the `{language}` placeholder in `rubric-summary.md`.
    `rubric-topics.md` has no placeholder; passed for consistency.
    """
    return (
        load_rubric("summary", language=language)
        + "\n\n---\n\n"
        + load_rubric("topics", language=language)
        + "\n\n---\n\n"
        "Respond with a single JSON object and nothing else:\n"
        '{"summary": "...", "primary_topic": "<slug>", '
        '"topics": ["<slug>", ...]}'
    )


def _content_image_descriptions(item: Item) -> list[str]:
    """Return non-decorative image descriptions on the item, in media order.

    Decorative photos (`is_decorative=True`) are filtered out at this
    seam so they introduce no topic noise — an avatar or a reaction
    meme would otherwise drag the assigned topics toward whatever the
    image happened to depict. Items without described photos return an
    empty list.
    """
    return [
        entry.description
        for entry in item.media
        if isinstance(entry, MediaPhotoDescribed) and not entry.is_decorative and entry.description
    ]


def _video_frame_descriptions(item: Item) -> list[str]:
    """Return the key-frame descriptions of every `x_video` source, in order.

    These are the slides/screens a video SHOWS — visual topic signal present even
    when the video has no speech (`has_speech=False`, no transcript). They live on
    the `x_video` `ContentSourceSuccess.frames` list, a DIFFERENT field from the
    `MediaPhotoDescribed` photo descriptions on `item.media`
    (`_content_image_descriptions`). Empty-description frames (the VLM found nothing
    to say, or the frame was unreadable) are skipped.
    """
    if item.content is None:
        return []
    descriptions: list[str] = []
    for src in item.content.sources:
        if isinstance(src, ContentSourceSuccess) and src.kind == "x_video":
            descriptions += [frame.description for frame in src.frames if frame.description]
    return descriptions


def _video_frames_section(item: Item) -> list[str]:
    """Build the `Video frames` block bounded to `FRAME_DESC_CHAR_LIMIT`, or [].

    A slide-heavy or screen-share talk can dedup to dozens of distinct frames; the
    block is capped so it can't crowd the transcript out of the per-item prompt.
    When the cap clips frames, an explicit `[… N further frames omitted …]` marker
    signposts the cut so the LLM does not read it as the end of the deck. At least
    one frame is always kept, even if a single description exceeds the cap.
    """
    descriptions = _video_frame_descriptions(item)
    if not descriptions:
        return []
    lines = ["", "Video frames (slides/screens shown in the video):"]
    used = 0
    kept = 0
    for description in descriptions:
        line = f"- {description}"
        if kept > 0 and used + len(line) > FRAME_DESC_CHAR_LIMIT:
            break
        lines.append(line)
        used += len(line)
        kept += 1
    omitted = len(descriptions) - kept
    if omitted > 0:
        lines.append(f"[… {omitted} further frames omitted …]")
    return lines


def _images_section(item: Item) -> list[str]:
    """Build the `Images in this post:` block, or an empty list when not applicable.

    Visual content carries topic signal too. The describe stage
    already filtered decoratives — this just splices the prose in
    right before the article body so the LLM reads the post + the
    image evidence + the article in natural order.
    """
    image_descriptions = _content_image_descriptions(item)
    if not image_descriptions:
        return []
    lines = ["", "Images in this post:"]
    lines += [f"- {description}" for description in image_descriptions]
    return lines


# The rule half of every "this was never downloaded" guardrail note. One wording,
# shared by every LLM surface (api prompt, enrich worksheet, verify source), so the
# generator and the judge read the SAME contract and the judge can hold the
# generator to it.
_UNFETCHED_RULE = (
    "Beyond the URL/domain itself, nothing about it is known — never describe, "
    "reconstruct or guess it from the URL, the domain or world knowledge."
)

# The note stamped when an item quotes a post whose content is not on the item.
# `quoted_id` is captured at extract time but NO fetcher downloads the quoted post,
# so without this marker the generator is ordered (rubric-summary) to summarise
# content that is not in its inputs — an invitation to invent.
QUOTED_CONTENT_UNFETCHED_NOTE = f"The quoted post's content was NOT fetched. {_UNFETCHED_RULE}"


def fetched_link_sources(item: Item) -> int:
    """How many fetched LINK-content bodies the item carries (`LINK_CONTENT_KINDS`).

    A `thread` / `quoted_tweet` / `x_video` source is NOT a fetched link, so it can
    never mask a link nobody downloaded.
    """
    if item.content is None:
        return 0
    return sum(
        1
        for src in item.content.sources
        if isinstance(src, ContentSourceSuccess) and src.kind in LINK_CONTENT_KINDS and src.text
    )


def links_content_unfetched(item: Item) -> bool:
    """True when the item links out and at least one link's content is missing.

    A COUNT comparison, not a per-link URL match: pairing `item.links` to fetched
    sources by URL is unreliable (a `t.co` shortlink vs the resolved URL), but
    counting is exact — fewer fetched link bodies than links means some link's
    content is missing, whether that is all of them or one of two.
    """
    return bool(item.links) and fetched_link_sources(item) < len(item.links)


def unfetched_links_note(item: Item) -> str | None:
    """The guardrail note for an item whose linked content is missing, or None.

    A PARTIAL fetch states the counts: a `Linked article` block IS present then, and
    would otherwise lend its credibility to a claim about the link nobody fetched.
    """
    if not links_content_unfetched(item):
        return None
    total, fetched = len(item.links), fetched_link_sources(item)
    if fetched:
        headline = (
            f"Only {fetched} of {total} linked pages were fetched — the content of "
            f"the other {total - fetched} was NOT fetched, and the fetched article "
            "is no evidence for it."
        )
    else:
        headline = "The linked content was NOT fetched."
    return f"{headline} {_UNFETCHED_RULE}"


def quoted_content_unfetched(item: Item) -> bool:
    """True when the item quotes a post whose content is not on the item."""
    if not item.quoted_id:
        return False
    if item.content is None:
        return True
    return not any(
        isinstance(src, ContentSourceSuccess) and src.kind == "quoted_tweet" and src.text
        for src in item.content.sources
    )


def first_source_text(item: Item, kinds: Collection[str]) -> str | None:
    """The first success source of one of `kinds` that carries text, truncated, or None.

    The one reader every surface shares (api prompt, worksheet, judge source), so a
    thread / quoted post / linked article is never picked up under another's label.
    """
    if item.content is None:
        return None
    for src in item.content.sources:
        if isinstance(src, ContentSourceSuccess) and src.kind in kinds and src.text:
            return src.text[:ARTICLE_CHAR_LIMIT]
    return None


def thread_text(item: Item) -> str | None:
    """The item's own expanded thread body, truncated, or None."""
    return first_source_text(item, _THREAD_KINDS)


def quoted_text(item: Item) -> str | None:
    """The fetched quoted-post body, truncated, or None — today always None."""
    return first_source_text(item, _QUOTED_KINDS)


_THREAD_KINDS: frozenset[str] = frozenset({"thread"})
_QUOTED_KINDS: frozenset[str] = frozenset({"quoted_tweet"})


def _links_section(item: Item) -> list[str]:
    """Build the `Links in the post:` block, or an empty list when not applicable."""
    if not item.links:
        return []
    lines = [
        "",
        "Links in the post (the domain is topic signal even when the article body is unavailable):",
    ]
    lines += [f"- {ln.url}  (domain: {ln.domain})" for ln in item.links]
    note = unfetched_links_note(item)
    if note:
        lines += ["", note]
    return lines


def _thread_section(item: Item) -> list[str]:
    """Build the `Thread:` block — the poster's OWN expanded thread text.

    Labelled as a thread, never as a `Linked article`: it is real evidence (full
    text, by the same author), but it is not the content of any link the post points
    at, and passing it off as one would tell the LLM a page was downloaded when none
    was.
    """
    text = thread_text(item)
    if not text:
        return []
    return ["", "Thread (full text by the same author):", text]


def _quoted_section(item: Item) -> list[str]:
    """Build the quoted-post block: its fetched body, else the NOT-fetched marker.

    No fetcher produces a `quoted_tweet` source today, so in practice this stamps the
    marker. The fetched branch exists so that when one lands, the quoted body reaches
    the LLM under its OWN label instead of being dropped by the `LINK_CONTENT_KINDS`
    narrowing in `_article_sections`.
    """
    if not item.quoted_id:
        return []
    body = quoted_text(item)
    if body:
        return ["", "Quoted post (the content this post is sharing):", body]
    return ["", "Quoted post — content NOT fetched:", QUOTED_CONTENT_UNFETCHED_NOTE]


def _article_sections(item: Item) -> list[str]:
    """Build one block per successfully-fetched LINKED article. Empty if no content.

    Only `LINK_CONTENT_KINDS` are rendered here. A video transcript, a thread and a
    quoted post are manufactured or own-authored text, not a linked article; each has
    its own labelled block (`_video_transcript_section`, `_thread_section`,
    `_quoted_section`). Rendering any of them as a "Linked article" would mislabel the
    content type to the LLM — and tell it a link was fetched when it was not.
    """
    if item.content is None or not item.content.sources:
        return []
    lines: list[str] = []
    for src in item.content.sources:
        # Narrow to the success variant — only those carry `title`/`text`.
        if isinstance(src, ContentSourceSuccess) and src.kind in LINK_CONTENT_KINDS and src.text:
            lines += [
                "",
                f"Linked article ({src.title or src.url}):",
                src.text[:ARTICLE_CHAR_LIMIT],
            ]
    return lines


def _video_transcript_section(item: Item) -> list[str]:
    """Build the `Video transcript:` block(s) for `x_video` sources with speech.

    A no-speech source (`has_speech=False`, empty text) contributes nothing —
    it carries no topic signal and would only add noise. Long transcripts are
    truncated to `TRANSCRIPT_CHAR_LIMIT` so one 72-min talk can't blow the
    per-item prompt (#44).
    """
    if item.content is None:
        return []
    lines: list[str] = []
    for src in item.content.sources:
        if (
            isinstance(src, ContentSourceSuccess)
            and src.kind == "x_video"
            and src.has_speech
            and src.text
        ):
            lines += ["", "Video transcript:", truncate_transcript(src.text, TRANSCRIPT_CHAR_LIMIT)]
    return lines


def _user_prompt(item: Item, vocab: list[Topic]) -> str:
    parts = [
        "Controlled vocabulary (use only these slugs):",
        _vocab_block(vocab),
        "",
        f"Post author: @{item.author.handle}",
        f"Post text:\n{item.text}",
    ]
    if item.bookmark_folder:
        parts += ["", f"Saved by the user in the bookmark folder: {item.bookmark_folder}"]
    parts += _thread_section(item)
    parts += _images_section(item)
    parts += _video_transcript_section(item)
    parts += _video_frames_section(item)
    parts += _quoted_section(item)
    parts += _links_section(item)
    parts += _article_sections(item)
    return "\n".join(parts)


class ApiExecutor:
    """Enrichment executor backed by the Anthropic API."""

    def __init__(self, model: str, output_language: str, client=None):
        if client is None:
            from anthropic import Anthropic  # lazy: tests inject a fake

            client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self._client = client
        self._model = model
        self._output_language = output_language

    def enrich_items(self, items: list[Item], vocab: list[Topic]) -> list[EnrichmentJudgment]:
        system = _system_prompt(self._output_language)
        recoverable = _recoverable_errors()
        results: list[EnrichmentJudgment] = []
        failures = 0
        for item in items:
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": _user_prompt(item, vocab)}],
                )
                judgment = json_from_response(response, context=f"item {item.id}")
                if not {"summary", "primary_topic", "topics"} <= judgment.keys():
                    raise ValueError(
                        f"item {item.id}: response is not a judgment object, "
                        f"keys={sorted(judgment)}"
                    )
                results.append(
                    EnrichmentJudgment(
                        item_id=item.id,
                        summary=str(judgment["summary"]),
                        primary_topic=str(judgment["primary_topic"]),
                        topics=list(judgment["topics"]),
                    )
                )
            except recoverable as exc:
                # One transient/malformed response must not abort the batch:
                # the item stays pending and is retried on the next run. Note:
                # programmer bugs (`AttributeError`, …) and `KeyboardInterrupt`
                # are NOT in `recoverable` — they propagate so the developer
                # sees the traceback and Ctrl-C still works.
                failures += 1
                print(
                    f"warn: enrichment failed for item {item.id}: {exc}",
                    file=sys.stderr,
                )
                continue
        if items and not results and failures > 0:
            raise RuntimeError(
                f"All {failures} items failed enrichment; see warnings above for details."
            )
        if failures > 0:
            # SUMMARY prefix so the line is distinguishable from the per-item
            # `warn:` lines that precede it in a partial-failure batch.
            print(
                f"SUMMARY: enriched: {len(results)}, failed: {failures}",
                file=sys.stderr,
            )
        return results
