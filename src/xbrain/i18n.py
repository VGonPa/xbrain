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
    """Localised section headers and labels written by the wiki generators."""

    topics_label: str  # "Temas" / "Topics" — per-item line listing topics
    content_header: str  # "Contenido" / "Content" — linked-article body section
    tweet_header: str  # "Tweet" / "Tweet" — the post body section
    summary_header: str  # "Resumen" / "Summary" — index summary section
    index_topics_header: str  # "Temas" / "Topics" — index topics list
    primary_posts: str  # "Posts primarios" / "Primary posts" — topic page
    also_relevant: str  # "También relevante" / "Also relevant" — topic page


_STRINGS: dict[str, Strings] = {
    "English": Strings(
        topics_label="Topics",
        content_header="Content",
        tweet_header="Tweet",
        summary_header="Summary",
        index_topics_header="Topics",
        primary_posts="Primary posts",
        also_relevant="Also relevant",
    ),
    "Spanish": Strings(
        topics_label="Temas",
        content_header="Contenido",
        tweet_header="Tweet",
        summary_header="Resumen",
        index_topics_header="Temas",
        primary_posts="Posts primarios",
        also_relevant="También relevante",
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
