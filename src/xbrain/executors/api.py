"""The `api` executor — produces enrichment judgment via the Anthropic API.

One API call per item: simple, robust, easy to retry. The Anthropic client is
injected (defaults to a real one) so tests run offline. The user prompt always
carries the link URLs/domains and the bookmark folder — topic signal even when
the article body was not fetched (design §15.2).
"""
from __future__ import annotations

from xbrain.executors.base import EnrichmentJudgment
from xbrain.llm_json import extract_json
from xbrain.models import Item, Topic
from xbrain.rubrics import load_rubric

_MAX_TOKENS = 600


def _vocab_block(vocab: list[Topic]) -> str:
    return "\n".join(f"- {t.slug}: {t.description}" for t in vocab)


def _system_prompt() -> str:
    """The rubrics are the system prompt — the declarative source of truth."""
    return (
        load_rubric("summary")
        + "\n\n---\n\n"
        + load_rubric("topics")
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
        parts += ["", f"Saved by the user in the bookmark folder: "
                      f"{item.bookmark_folder}"]
    if item.links:
        parts += ["", "Links in the post (the domain is topic signal even when "
                      "the article body is unavailable):"]
        parts += [f"- {ln.url}  (domain: {ln.domain})" for ln in item.links]
    if item.content and item.content.sources:
        for src in item.content.sources:
            if src.ok and src.text:
                parts += ["", f"Linked article ({src.title or src.url}):",
                          src.text[:4000]]
    return "\n".join(parts)


class ApiExecutor:
    """Enrichment executor backed by the Anthropic API."""

    def __init__(self, model: str, client=None):
        if client is None:
            from anthropic import Anthropic  # lazy: tests inject a fake

            client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
        self._client = client
        self._model = model

    def enrich_items(
        self, items: list[Item], vocab: list[Topic]
    ) -> list[EnrichmentJudgment]:
        system = _system_prompt()
        results: list[EnrichmentJudgment] = []
        for item in items:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user",
                           "content": _user_prompt(item, vocab)}],
            )
            blocks = [
                b for b in response.content
                if getattr(b, "type", None) == "text"
            ]
            if not blocks:
                raise ValueError(
                    f"no text block in model response for item {item.id}"
                )
            judgment = extract_json(blocks[0].text)
            results.append(EnrichmentJudgment(
                item_id=item.id,
                summary=str(judgment.get("summary", "")),
                primary_topic=str(judgment.get("primary_topic", "")),
                topics=list(judgment.get("topics", [])),
            ))
        return results
