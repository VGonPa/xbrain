"""The `api` executor — produces enrichment judgment via the Anthropic API.

One API call per item: simple, robust, easy to retry. The Anthropic client is
injected (defaults to a real one) so tests run offline. The user prompt always
carries the link URLs/domains and the bookmark folder — topic signal even when
the article body was not fetched (design §15.2).
"""

from __future__ import annotations

import sys

from xbrain.executors.base import EnrichmentJudgment
from xbrain.llm_json import json_from_response
from xbrain.models import ContentSourceSuccess, Item, Topic
from xbrain.rubrics import ARTICLE_CHAR_LIMIT, load_rubric

_MAX_TOKENS = 600


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
    if item.links:
        parts += [
            "",
            "Links in the post (the domain is topic signal even when "
            "the article body is unavailable):",
        ]
        parts += [f"- {ln.url}  (domain: {ln.domain})" for ln in item.links]
    if item.content and item.content.sources:
        # Narrow to the success variant — only those carry `title`/`text`.
        for src in item.content.sources:
            if isinstance(src, ContentSourceSuccess) and src.text:
                parts += [
                    "",
                    f"Linked article ({src.title or src.url}):",
                    src.text[:ARTICLE_CHAR_LIMIT],
                ]
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
        results: list[EnrichmentJudgment] = []
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
            except Exception as exc:  # noqa: BLE001
                # One transient/malformed response must not abort the batch:
                # the item stays pending and is retried on the next run.
                print(
                    f"warn: enrichment failed for item {item.id}: {exc}",
                    file=sys.stderr,
                )
                continue
        return results
