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

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Strings:
    """Localised section headers and labels written by the wiki generators.

    `topics_label` covers both the per-item line ("**Topics:** ...") and the
    index topics section ("## Topics"). These are identical within each
    language, so a single field carries both — adding a second variant would
    suggest variance that does not exist.
    """

    language: str  # the output language these strings belong to — the rubric the judge
    # was handed is language-substituted, so the verdict's contract hash needs it (PR-D).
    topics_label: str  # "Temas" / "Topics"
    content_header: str  # "Contenido" / "Content"
    summary_header: str  # "Resumen" / "Summary"
    primary_posts: str  # "Posts primarios" / "Primary posts"
    also_relevant: str  # "También relevante" / "Also relevant"
    video_digest_header: str  # "Resumen del vídeo" / "Video digest" (#44)
    silent_video: str  # the one-line no-speech marker (#44)
    video_evidence_header: str  # collapsible label for the raw frames + transcript
    verify_badge_fail: str  # "Verification: FAIL" / "Verificación: FALLA" (#79 badge)
    verify_badge_review: str  # "Verification: REVIEW" / "Verificación: REVISAR" (#79 badge)
    # The quoted post a quote-tweet is sharing. The header carries the quoted account's
    # `@handle (Name)`, so the reader can never take a third party's words for the
    # poster's — the same conflation the attribution rule stops on the LLM side.
    quoted_post_header: str  # "Quoted post" / "Post citado"
    # When we could not read the quoted post. Written down, never silently omitted: a
    # reader who sees nothing concludes there was nothing. The reason is a fact.
    quoted_post_unavailable: str  # "Quoted post unavailable" / "Post citado no disponible"
    quoted_unavailable_deleted: str  # `not_found`  — X tombstoned it
    quoted_unavailable_protected: str  # `forbidden`  — protected / suspended
    quoted_unavailable_unknown: str  # anything else — X served us nothing usable
    # --- Note body lines written by `generate` (were hard-coded Spanish until v0.2.1) ---
    links_header: str  # "## Links" section of an item note
    view_original: str  # permalink label under the links
    broken_link: str  # "⚠ Broken link: <url> — detail (checked DATE)"
    broken_link_checked: str
    broken_link_unknown: str  # detail when there is neither an HTTP status nor a reason
    content_failure_reasons: Mapping[str, str]  # `FailureReason` slug -> reader text
    media_failure_reasons: Mapping[str, str]  # `MediaFailureReason` slug -> reader text
    photo_unavailable: str
    image_unavailable: str  # an X Article inline image
    video_unavailable: str
    watch_video: str  # link label of a playable stream not downloaded yet
    pending_download: str
    # --- The user tail appended to a freshly created page (item notes + topic pages) ---
    my_notes_header: str
    my_notes_hint: str
    # --- `_index.md` and `log.md` ---
    generated_label: str
    total_items: str
    own_posts: str
    with_own_note: str
    enriched_label: str
    indexes_header: str
    full_log_link: str
    dashboard_link: str
    dashboard_hint: str
    log_title: str
    note_link: str  # wikilink alias from a log line to its note
    # --- Topic pages ---
    overview_pending: str
    overview_stale: str  # format string: `{delta}` is the signed post delta (e.g. "+2")
    key_notes_header: str


_STRINGS: dict[str, Strings] = {
    "English": Strings(
        language="English",
        topics_label="Topics",
        content_header="Content",
        summary_header="Summary",
        primary_posts="Primary posts",
        also_relevant="Also relevant",
        video_digest_header="Video digest",
        silent_video="🔇 Silent video (no speech detected).",
        video_evidence_header="Frames + transcript",
        verify_badge_fail="Verification: FAIL",
        verify_badge_review="Verification: REVIEW",
        quoted_post_header="Quoted post",
        quoted_post_unavailable="Quoted post unavailable",
        quoted_unavailable_deleted="deleted or never existed",
        quoted_unavailable_protected="protected or suspended account",
        quoted_unavailable_unknown="X served no content for it",
        links_header="Links",
        view_original="View original post",
        broken_link="Broken link",
        broken_link_checked="checked",
        broken_link_unknown="could not be retrieved",
        content_failure_reasons={
            "not_found": "not found",
            "forbidden": "access denied",
            "paywall": "paywall",
            "timeout": "timed out",
            "dns_error": "domain did not resolve",
            "js_required": "requires JavaScript",
            "empty_content": "no extractable content",
            "blocked_interstitial": "cookie or login wall (no article)",
            "unknown_error": "unknown error",
        },
        media_failure_reasons={
            "http_4xx": "URL not found (HTTP 4xx)",
            "http_5xx": "server error (HTTP 5xx)",
            "timeout": "timed out",
            "format_error": "unrecognised format",
            "unknown_error": "unknown error",
        },
        photo_unavailable="Photo unavailable",
        image_unavailable="Image unavailable",
        video_unavailable="Video unavailable",
        watch_video="Watch video",
        pending_download="not downloaded yet",
        my_notes_header="My notes",
        my_notes_hint=(
            "*(Write below. Everything above this point is regenerated "
            "automatically; do not edit it.)*"
        ),
        generated_label="Generated",
        total_items="Total items",
        own_posts="Own posts",
        with_own_note="With a note",
        enriched_label="Enriched",
        indexes_header="Indexes",
        full_log_link="Full chronological log",
        dashboard_link="Interactive dashboard",
        dashboard_hint="metrics, drill-down and links (opens in the browser)",
        log_title="Chronological log",
        note_link="note",
        overview_pending="*(Overview pending — run `xbrain topics`.)*",
        overview_stale=(
            "> ⚠ Overview out of date: {delta} posts since the last synthesis. "
            "Run `xbrain topics --resynth`."
        ),
        key_notes_header="Key notes",
    ),
    "Spanish": Strings(
        language="Spanish",
        topics_label="Temas",
        content_header="Contenido",
        summary_header="Resumen",
        primary_posts="Posts primarios",
        also_relevant="También relevante",
        video_digest_header="Resumen del vídeo",
        silent_video="🔇 Vídeo sin voz (sin transcripción).",
        video_evidence_header="Frames y transcripción",
        verify_badge_fail="Verificación: FALLA",
        verify_badge_review="Verificación: REVISAR",
        quoted_post_header="Post citado",
        quoted_post_unavailable="Post citado no disponible",
        quoted_unavailable_deleted="borrado o inexistente",
        quoted_unavailable_protected="cuenta protegida o suspendida",
        quoted_unavailable_unknown="X no sirvió su contenido",
        links_header="Enlaces",
        view_original="Ver tweet original",
        broken_link="Enlace roto",
        broken_link_checked="verificado",
        broken_link_unknown="no se pudo recuperar",
        content_failure_reasons={
            "not_found": "no encontrado",
            "forbidden": "acceso denegado",
            "paywall": "muro de pago",
            "timeout": "tiempo de espera agotado",
            "dns_error": "dominio no resuelto",
            "js_required": "requiere JavaScript",
            "empty_content": "sin contenido extraíble",
            "blocked_interstitial": "muro de cookies o login (sin artículo)",
            "unknown_error": "error desconocido",
        },
        media_failure_reasons={
            "http_4xx": "URL no encontrada (HTTP 4xx)",
            "http_5xx": "error del servidor (HTTP 5xx)",
            "timeout": "tiempo de espera agotado",
            "format_error": "formato no reconocido",
            "unknown_error": "error desconocido",
        },
        photo_unavailable="Foto no disponible",
        image_unavailable="Imagen no disponible",
        video_unavailable="Vídeo no disponible",
        watch_video="Ver vídeo",
        pending_download="pendiente de descarga",
        my_notes_header="Mis notas",
        my_notes_hint=(
            "*(Escribe debajo. El bloque por encima de este punto se regenera "
            "automáticamente; no lo edites.)*"
        ),
        generated_label="Generado",
        total_items="Items totales",
        own_posts="Tweets propios",
        with_own_note="Con nota propia",
        enriched_label="Enriquecidos",
        indexes_header="Índices",
        full_log_link="Log cronológico completo",
        dashboard_link="Dashboard interactivo",
        dashboard_hint="métricas, drill-down y enlaces (se abre en el navegador)",
        log_title="Log cronológico",
        note_link="nota",
        overview_pending="*(Overview pendiente — ejecuta `xbrain topics`.)*",
        overview_stale=(
            "> ⚠ Overview desactualizado: {delta} posts desde la última "
            "síntesis. Ejecuta `xbrain topics --resynth`."
        ),
        key_notes_header="Notas importantes",
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
