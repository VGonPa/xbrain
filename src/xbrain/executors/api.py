"""The `api` executor — produces enrichment judgment via the Anthropic API.

One API call per item: simple, robust, easy to retry. The Anthropic client is
injected (defaults to a real one) so tests run offline. The user prompt always
carries the link URLs/domains and the bookmark folder — topic signal even when
the article body was not fetched (design §15.2).
"""

from __future__ import annotations

import json
import sys

from xbrain.executors.base import EnrichmentJudgment
from xbrain.llm_json import json_from_response
from xbrain.models import ContentSourceSuccess, Item, Topic
from xbrain.rubrics import ARTICLE_CHAR_LIMIT, load_rubric

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
