"""Wiki UI strings keyed by output language.

The LLM output language is parameterised via a `{language}` substitution in
`rubric-summary.md` and `rubric-topic-page.md`. This module covers the OTHER
half: the section headers that the wiki generators write directly (`Topics:`,
`Content:`, `Summary`, `Primary posts`, ...). They must agree with the LLM
output language so the wiki reads in one voice.

To add a language: append an entry to `_STRINGS`. `SUPPORTED_LANGUAGES` is
derived from the dict — no second list to keep in sync.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strings:
    """Localised section headers and labels written by the wiki generators.

    `topics_label` covers both the per-item line ("**Topics:** ...") and the
    index topics section ("## Topics"). These are identical within each
    language, so a single field carries both — adding a second variant would
    suggest variance that does not exist.
    """

    topics_label: str  # "Temas" / "Topics"
    content_header: str  # "Contenido" / "Content"
    summary_header: str  # "Resumen" / "Summary"
    primary_posts: str  # "Posts primarios" / "Primary posts"
    also_relevant: str  # "También relevante" / "Also relevant"
    video_digest_header: str  # "Resumen del vídeo" / "Video digest" (#44)
    silent_video: str  # the one-line no-speech marker (#44)
    video_evidence_header: str  # collapsible label for the raw frames + transcript


_STRINGS: dict[str, Strings] = {
    "English": Strings(
        topics_label="Topics",
        content_header="Content",
        summary_header="Summary",
        primary_posts="Primary posts",
        also_relevant="Also relevant",
        video_digest_header="Video digest",
        silent_video="🔇 Silent video (no speech detected).",
        video_evidence_header="Frames + transcript",
    ),
    "Spanish": Strings(
        topics_label="Temas",
        content_header="Contenido",
        summary_header="Resumen",
        primary_posts="Posts primarios",
        also_relevant="También relevante",
        video_digest_header="Resumen del vídeo",
        silent_video="🔇 Vídeo sin voz (sin transcripción).",
        video_evidence_header="Frames y transcripción",
    ),
}

SUPPORTED_LANGUAGES: tuple[str, ...] = tuple(_STRINGS.keys())


def strings_for(language: str) -> Strings:
    """Return the wiki UI strings for the given output language.

    Raises ValueError listing supported languages if `language` is unknown.
    """
    if language not in _STRINGS:
        raise ValueError(
            f"Unsupported output language: {language!r}. Supported: {SUPPORTED_LANGUAGES}"
        )
    return _STRINGS[language]
