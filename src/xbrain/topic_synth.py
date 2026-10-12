"""Topic-overview synthesis — the two executor tracks for the `topics` stage.

Mirrors the enrich pattern: an `api` track (one LLM call per topic) and a
worksheet track (export / import for the `manual` / `claude-code` executors).
Both consume the same declarative `rubric-topic-page.md`.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from xbrain.llm_json import json_from_response
from xbrain.llm_client import LlmProvider, build_llm_client, recoverable_llm_errors
from xbrain.rubrics import load_rubric
from xbrain.validate import validate_overview

_MAX_TOKENS = 2000
_OVERVIEW_JUDGMENT_KEYS = ("overview", "notes")


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
    """Everything an executor needs to synthesize one topic's overview.

    `summaries` is the per-post text already enrichment produced.
    `image_descriptions` is the vision-LLM-produced prose for every
    non-decorative photo in the topic's posts — fed to the
    topic-synthesis LLM alongside the summaries so the topic page
    reflects visual evidence that the per-post enrichment may have
    only partially captured. Decorative photos are excluded at the
    seam (mirrors `xbrain.executors.api._content_image_descriptions`).
    `video_transcripts` is a legacy field name: it now carries bounded
    per-video executive summaries from `x_video` sources. Raw transcripts are
    retained only as vault reference artifacts and never enter this prompt.
    """

    slug: str
    description: str
    summaries: list[str]
    image_descriptions: list[str] = field(default_factory=list)
    video_transcripts: list[str] = field(default_factory=list)


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
    if topic.image_descriptions:
        # Visual evidence supplements the per-post summaries: a chart
        # or screenshot in a topic carries signal the summary may have
        # only hinted at. Decorative photos are filtered out by the
        # caller (`build_topic_inputs`) so they introduce no noise.
        lines += [
            "",
            f"Images across the {len(topic.image_descriptions)} content-bearing photos in this topic:",
        ]
        lines += [f"- {description}" for description in topic.image_descriptions]
    if topic.video_transcripts:
        # Video executive summaries supplement the per-post summaries the same
        # way image descriptions do. Each excerpt is already bounded by the
        # caller (`build_topic_inputs`) so a topic full of videos stays within
        # the token budget.
        lines += [
            "",
            f"Video summaries across the {len(topic.video_transcripts)} videos in this topic (truncated):",
        ]
        lines += [f"- {summary}" for summary in topic.video_transcripts]
    return "\n".join(lines)


def _validated_overview_judgment(judgment: dict) -> dict:
    """Return a clean overview judgment, tolerating harmless extra keys.

    Some LLMs occasionally append an accidental prose-shaped key while still
    providing valid `overview` and `notes`. The worksheet path stays strict via
    `validate_overview`; the API path can safely project the allowed keys and
    keep the batch moving.
    """
    errors = validate_overview(judgment)
    if not errors:
        return judgment

    if any(error.startswith("unexpected keys") for error in errors):
        candidate = {key: judgment.get(key) for key in _OVERVIEW_JUDGMENT_KEYS}
        if not validate_overview(candidate):
            return candidate

    raise ValueError(f"overview rechazado: {'; '.join(errors)}")


def _judgment_from_response(response, slug: str) -> OverviewJudgment:
    judgment = json_from_response(response, context=f"topic {slug}")
    judgment = _validated_overview_judgment(judgment)
    return OverviewJudgment(
        slug=slug, overview=str(judgment["overview"]), notes=list(judgment["notes"])
    )


def _recoverable_errors() -> tuple[type[Exception], ...]:
    """Exception classes a per-topic failure should swallow + log + continue on.

    Mirrors `xbrain.executors.api._recoverable_errors`. See that helper for rationale.
    """
    return recoverable_llm_errors()


def synthesize_overviews_api(
    inputs: list[TopicInput],
    model: str,
    output_language: str,
    client=None,
    provider: LlmProvider = "nanogpt",
    base_url: str | None = None,
) -> list[OverviewJudgment]:
    """Synthesize topic overviews via the configured text LLM API.

    A `RuntimeError` is raised when EVERY input topic fails — exposed as a
    non-zero exit through the CLI's `_handle_cli_errors` wrapper. Partial
    success (some topics synthesise, some fail) returns the successes and
    prints a summary line to stderr.

    Programmer bugs (`AttributeError`, ...) and `KeyboardInterrupt` are not
    in the recoverable set — they propagate so the developer sees the
    traceback and Ctrl-C still works.
    """
    if client is None:
        client = build_llm_client(provider, base_url=base_url)
    system = _system_prompt(output_language)
    recoverable = _recoverable_errors()
    results: list[OverviewJudgment] = []
    failures = 0
    for topic in inputs:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=_MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": _user_prompt(topic)}],
            )
            results.append(_judgment_from_response(response, topic.slug))
        except recoverable as exc:
            # One failed topic must not abort the batch — it stays
            # unsynthesized and is retried on the next run.
            failures += 1
            print(
                f"warn: topic synthesis failed for {topic.slug}: {exc}",
                file=sys.stderr,
            )
            continue
    if inputs and not results and failures > 0:
        raise RuntimeError(
            f"All {failures} topics failed synthesis; see warnings above for details."
        )
    if failures > 0:
        # SUMMARY prefix so the line stands out from the per-topic `warn:`
        # lines that precede it in a partial-failure batch.
        print(
            f"SUMMARY: synthesized: {len(results)}, failed: {failures}",
            file=sys.stderr,
        )
    return results


def export_topic_worksheet(inputs: list[TopicInput], path: Path, output_language: str) -> None:
    """Write a worksheet a Claude Code session (or a person) fills with overviews."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instructions": (
            f"For each entry in `topics`, append one object to `judgments` with "
            f"keys {{slug, overview, notes}}. `overview` is plain prose in "
            f"{output_language}; `notes` is a list of plain-prose strings in "
            f"{output_language}. A topic may also carry `image_descriptions` "
            f"(content-bearing photos) and `video_transcripts` (legacy field name "
            f"for bounded video executive summaries) — weigh them as evidence "
            f"alongside the summaries. "
            f"No wikilinks, no filenames. Then run: "
            f"xbrain topics --apply <this file>."
        ),
        "rubric": load_rubric("topic-page", language=output_language),
        "topics": [
            {
                "slug": t.slug,
                "description": t.description,
                "summaries": t.summaries,
                # Content-bearing photo descriptions (#34): mirrors the `api` track's
                # topic block ("Images across the N content-bearing photos in this
                # topic"). Already collected + decorative-filtered by `build_topic_inputs`
                # into `TopicInput.image_descriptions`; empty list when the topic has
                # none. No truncation — the api path doesn't bound these either.
                "image_descriptions": t.image_descriptions,
                # Bounded video executive summaries: mirrors the `api` track's
                # topic block ("Video summaries across the N videos in this topic").
                # Already collected + per-video bounded (`TOPIC_TRANSCRIPT_CHAR_LIMIT`)
                # by `build_topic_inputs` into the legacy field
                # `TopicInput.video_transcripts`; empty list when the topic has none.
                "video_transcripts": t.video_transcripts,
            }
            for t in inputs
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
