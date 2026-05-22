"""Topic-overview synthesis — the two executor tracks for the `topics` stage.

Mirrors the enrich pattern: an `api` track (one Anthropic call per topic) and a
worksheet track (export / import for the `manual` / `claude-code` executors).
Both consume the same declarative `rubric-topic-page.md`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from xbrain.llm_json import json_from_response
from xbrain.rubrics import load_rubric
from xbrain.validate import validate_overview

_MAX_TOKENS = 2000


class OverviewJudgment(BaseModel):
    """One synthesized topic overview.

    The LLM emits only the judgment ``{overview, notes}``; `slug` is the
    caller-supplied topic identifier stitched in after validation, never
    produced by the LLM.
    """

    slug: str
    overview: str
    notes: list[str] = Field(default_factory=list)


@dataclass
class TopicInput:
    """Everything an executor needs to synthesize one topic's overview."""

    slug: str
    description: str
    summaries: list[str]


def _system_prompt(language: str) -> str:
    """The rubric is the system prompt — the declarative source of truth.

    `language` substitutes the `{language}` placeholder in `rubric-topic-page.md`.
    """
    return (
        load_rubric("topic-page", language=language) + "\n\n---\n\n"
        "Respond with a single JSON object and nothing else:\n"
        '{"overview": "...", "notes": ["...", ...]}'
    )


def _user_prompt(topic: TopicInput) -> str:
    lines = [
        f"Topic slug: {topic.slug}",
        f"Topic description: {topic.description}",
        "",
        f"Summaries of the {len(topic.summaries)} posts in this topic:",
    ]
    lines += [f"- {summary}" for summary in topic.summaries]
    return "\n".join(lines)


def _judgment_from_response(response, slug: str) -> OverviewJudgment:
    judgment = json_from_response(response, context=f"topic {slug}")
    errors = validate_overview(judgment)
    if errors:
        raise ValueError(f"overview rechazado: {'; '.join(errors)}")
    return OverviewJudgment(
        slug=slug, overview=str(judgment["overview"]), notes=list(judgment["notes"])
    )


def synthesize_overviews_api(
    inputs: list[TopicInput],
    model: str,
    output_language: str,
    client=None,
) -> list[OverviewJudgment]:
    """Synthesize topic overviews via the Anthropic API — one call per topic."""
    if client is None:
        from anthropic import Anthropic  # lazy: tests inject a fake

        client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    system = _system_prompt(output_language)
    results: list[OverviewJudgment] = []
    for topic in inputs:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": _user_prompt(topic)}],
            )
            results.append(_judgment_from_response(response, topic.slug))
        except Exception as exc:  # noqa: BLE001
            # One failed topic must not abort the batch — it stays unsynthesized
            # and is retried on the next run.
            print(
                f"warn: topic synthesis failed for {topic.slug}: {exc}",
                file=sys.stderr,
            )
            continue
    return results


def export_topic_worksheet(inputs: list[TopicInput], path: Path, output_language: str) -> None:
    """Write a worksheet a Claude Code session (or a person) fills with overviews."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instructions": (
            f"For each entry in `topics`, append one object to `judgments` with "
            f"keys {{slug, overview, notes}}. `overview` is plain prose in "
            f"{output_language}; `notes` is a list of plain-prose strings in "
            f"{output_language}. No wikilinks, no filenames. Then run: "
            f"xbrain topics --apply <this file>."
        ),
        "rubric": load_rubric("topic-page", language=output_language),
        "topics": [
            {"slug": t.slug, "description": t.description, "summaries": t.summaries} for t in inputs
        ],
        "judgments": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_topic_worksheet(path: Path) -> list[dict]:
    """Read the `judgments` list from a filled topic worksheet."""
    if not path.exists():
        raise FileNotFoundError(f"Worksheet not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("worksheet must be a JSON object")
    judgments = data.get("judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("worksheet `judgments` must be a list")
    return judgments


def apply_overview_judgments(
    judgments: list[dict],
) -> tuple[list[OverviewJudgment], list[tuple[str, list[str]]]]:
    """Validate worksheet judgments. Returns `(valid, invalid)`."""
    valid: list[OverviewJudgment] = []
    invalid: list[tuple[str, list[str]]] = []
    for judgment in judgments:
        if not isinstance(judgment, dict):
            invalid.append(("", ["worksheet judgment is not a JSON object"]))
            continue
        slug = str(judgment.get("slug", ""))
        candidate = {"overview": judgment.get("overview"), "notes": judgment.get("notes")}
        errors = validate_overview(candidate)
        if errors:
            invalid.append((slug, errors))
            continue
        valid.append(
            OverviewJudgment(
                slug=slug,
                overview=str(judgment["overview"]),
                notes=list(judgment["notes"]),
            )
        )
    return valid, invalid
