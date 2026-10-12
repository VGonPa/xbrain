"""The `api` executor — produces enrichment judgments via the configured LLM API.

One API call per item: simple, robust, easy to retry. The LLM client is
injected (defaults to the configured real one) so tests run offline. The user prompt always
carries the link URLs/domains and the bookmark folder — topic signal even when
the article body was not fetched (design §15.2).
"""

from __future__ import annotations

import sys

from xbrain.executors.base import EnrichmentJudgment
from xbrain.llm_json import json_from_response
from xbrain.llm_client import LlmProvider, build_llm_client, recoverable_llm_errors
from xbrain.models import ArticleImageBlock, ContentSourceSuccess, Item, MediaPhotoDescribed, Topic
from xbrain.rubrics import (
    ARTICLE_CHAR_LIMIT,
    TRANSCRIPT_CHAR_LIMIT,
    load_rubric,
)
from xbrain.video_content import video_content_text

_MAX_TOKENS = 600


def _recoverable_errors() -> tuple[type[Exception], ...]:
    """Exception classes a per-item failure should swallow + log + continue on.

    Provider-specific API errors cover auth, rate-limit, server-side and
    network failures. `ValueError` covers validator rejections and
    `pydantic.ValidationError` (a `ValueError` subclass in pydantic v2);
    JSON/key errors cover malformed or incomplete LLM responses.
    """
    return recoverable_llm_errors()


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
        '"topics": ["<slug>", ...], "topic_confidence": "high|medium|low", '
        '"suggested_new_topics": ["optional-new-kebab-slug"]}'
    )


def _usable_image_description(entry: object) -> str | None:
    """Return a content-bearing vision caption, filtering decorative images."""
    if isinstance(entry, MediaPhotoDescribed) and not entry.is_decorative and entry.description:
        return entry.description
    return None


def _post_image_descriptions(item: Item) -> list[str]:
    """Return non-decorative image descriptions from the post media list."""
    return [
        description
        for entry in item.media
        if (description := _usable_image_description(entry)) is not None
    ]


def _article_image_descriptions(source: ContentSourceSuccess) -> list[str]:
    """Return non-decorative image descriptions from one structured X Article."""
    return [
        description
        for block in source.blocks
        if isinstance(block, ArticleImageBlock)
        and (description := _usable_image_description(block.media)) is not None
    ]


def _content_image_descriptions(item: Item) -> list[str]:
    """Return non-decorative image descriptions on the item, in reading order.

    Decorative photos (`is_decorative=True`) are filtered out at this
    seam so they introduce no topic noise — an avatar or a reaction
    meme would otherwise drag the assigned topics toward whatever the image
    happened to depict. Post photos come first, followed by described inline
    images from X Articles.
    """
    descriptions = _post_image_descriptions(item)
    if item.content is None:
        return descriptions
    for source in item.content.sources:
        if isinstance(source, ContentSourceSuccess) and source.kind == "x_article":
            descriptions.extend(_article_image_descriptions(source))
    return descriptions


def _images_section(item: Item) -> list[str]:
    """Build the `Images in this post:` block, or an empty list when not applicable.

    Visual content carries topic signal too. The describe stage
    already filtered decoratives — this just splices the prose in
    right before the article body so the LLM reads the post + the
    image evidence + the article in natural order.
    """
    image_descriptions = _post_image_descriptions(item)
    if not image_descriptions:
        return []
    lines = ["", "Images in this post:"]
    lines += [f"- {description}" for description in image_descriptions]
    return lines


def _links_section(item: Item) -> list[str]:
    """Build the `Links in the post:` block, or an empty list when not applicable."""
    if not item.links:
        return []
    lines = [
        "",
        "Links in the post (the domain is topic signal even when the article body is unavailable):",
    ]
    lines += [f"- {ln.url}  (domain: {ln.domain})" for ln in item.links]
    return lines


def _article_sections(item: Item) -> list[str]:
    """Build one block per successfully-fetched article. Empty if no content.

    `x_video` sources are excluded here — a video transcript is manufactured
    text, not a linked article, and gets its own labelled `Video summary:`
    block (`_video_transcript_section`). Rendering it as a "Linked article"
    would mislabel the content type to the LLM.
    """
    if item.content is None or not item.content.sources:
        return []
    lines: list[str] = []
    for src in item.content.sources:
        # Narrow to the success variant — only those carry `title`/`text`.
        if isinstance(src, ContentSourceSuccess) and src.kind != "x_video":
            image_descriptions = _article_image_descriptions(src) if src.kind == "x_article" else []
            if not src.text and not image_descriptions:
                continue
            lines += [
                "",
                f"Linked article ({src.title or src.url}):",
                src.text[:ARTICLE_CHAR_LIMIT] if src.text else "(Image-only article.)",
            ]
            if image_descriptions:
                lines += ["", "Images in linked article:"]
                lines += [f"- {description}" for description in image_descriptions]
    return lines


def _video_transcript_section(item: Item) -> list[str]:
    """Build the video-content block(s) for usable `x_video` sources.

    Audio transcripts are included. Visual-only slide/screen videos contribute
    described key frames, so the API executor can still classify them when no ASR
    transcript exists. Empty no-speech sources continue to contribute nothing.
    """
    if item.content is None:
        return []
    lines: list[str] = []
    for src in item.content.sources:
        if isinstance(src, ContentSourceSuccess):
            text = video_content_text(src, TRANSCRIPT_CHAR_LIMIT)
            if text:
                lines += ["", "Video summary:", text]
    return lines


def _topic_hints_section(item: Item, topic_hints: dict[str, list[str]] | None) -> list[str]:
    """Optional operator hints for one-item taxonomy regeneration."""
    hints = list(dict.fromkeys(topic_hints.get(item.id, []) if topic_hints else []))
    if not hints:
        return []
    lines = [
        "",
        "Operator-prioritized taxonomy hints:",
        (
            "Use these slugs as strong review signals. If a hinted slug exists in the "
            "controlled vocabulary above, prefer it when it accurately fits the item. "
            "If a hinted slug is not in the vocabulary, do not use it as primary_topic "
            "or topics; return it in suggested_new_topics only if it still looks useful."
        ),
    ]
    lines += [f"- {slug}" for slug in hints]
    return lines


def _user_prompt(
    item: Item,
    vocab: list[Topic],
    topic_hints: dict[str, list[str]] | None = None,
) -> str:
    parts = [
        "Controlled vocabulary (use only these slugs):",
        _vocab_block(vocab),
        "",
        f"Post author: @{item.author.handle}",
        f"Post text:\n{item.text}",
    ]
    if item.bookmark_folder:
        parts += ["", f"Saved by the user in the bookmark folder: {item.bookmark_folder}"]
    parts += _images_section(item)
    parts += _video_transcript_section(item)
    parts += _links_section(item)
    parts += _article_sections(item)
    parts += _topic_hints_section(item, topic_hints)
    return "\n".join(parts)


class ApiExecutor:
    """Enrichment executor backed by the configured text LLM API."""

    def __init__(
        self,
        model: str,
        output_language: str,
        client=None,
        provider: LlmProvider = "nanogpt",
        base_url: str | None = None,
        topic_hints: dict[str, list[str]] | None = None,
    ):
        if client is None:
            client = build_llm_client(provider, base_url=base_url)
        self._client = client
        self._model = model
        self._output_language = output_language
        self._topic_hints = topic_hints or {}

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
                    messages=[
                        {"role": "user", "content": _user_prompt(item, vocab, self._topic_hints)}
                    ],
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
                        topic_confidence=judgment.get("topic_confidence"),
                        suggested_new_topics=list(judgment.get("suggested_new_topics") or []),
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
