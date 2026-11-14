"""The generated wiki speaks ONE language: the one `[output].language` names.

Section headers already went through `xbrain.i18n`, but a set of note strings was
hard-coded Spanish in `generate.py`, `topics.py` and `notes_io.py` — a broken-link
line, the media failure/pending lines, the index and log headings, the topic-page
overview notices, the "write below" tail — so a vault configured in English (the
default) still came out half Spanish. These tests render a whole vault through the
PUBLIC entry points, with one item in every state that used to leak, and read the
files actually written.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from xbrain.generate import generate
from xbrain.i18n import SUPPORTED_LANGUAGES, strings_for
from xbrain.models import (
    ArticleImageBlock,
    ArticleTextBlock,
    ArticleVideoBlock,
    Author,
    Content,
    ContentSourceFailure,
    ContentSourceSuccess,
    Enrichment,
    Item,
    Link,
    MediaPhotoFailed,
    MediaVideoFailed,
    MediaVideoPending,
    Topic,
    TopicPage,
)
from xbrain.topics import compute_topic_posts, write_topic_pages

_T = datetime(2026, 5, 10, tzinfo=timezone.utc)

# Accented vowels / ñ / inverted marks, plus Spanish words that carry no accent
# ("Enlace roto", "Ver tweet original", "Mis notas", "Escribe debajo", "Enriquecidos").
_SPANISH = re.compile(
    r"[áéíóúñÁÉÍÓÚÑ¿¡]|\b(Enlace|roto|verificado|Ver|Enlaces|Mis notas|Escribe|debajo|"
    r"Generado|totales|propios|Enriquecidos|pendiente|desactualizado|Notas|nota|"
    r"encontrado|denegado|agotado|desconocido|servidor|Foto|Imagen|disponible|Ejecuta)\b"
)


def _item(item_id: str, **extra) -> Item:
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/a/status/{item_id}",
        author=Author(handle="alice", name="Alice"),
        text=f"Post {item_id}",
        created_at=_T,
        captured_at=_T,
        links=[Link(url=f"https://example.com/{item_id}", domain="example.com")],
        enriched=Enrichment(
            summary="A summary.",
            topics=["agents"],
            primary_topic="agents",
            enriched_at=_T,
            executor="manual",
        ),
        **extra,
    )


def _failed_photo() -> MediaPhotoFailed:
    return MediaPhotoFailed(
        url="https://pbs.twimg.com/media/dead.png",
        failure_reason="http_4xx",
        attempts=1,
        last_attempt_at=_T,
    )


def _store() -> dict[str, Item]:
    broken = _item(
        "1",
        content=Content(
            fetched_at=_T,
            sources=[
                ContentSourceFailure(
                    kind="external_article",
                    url="https://example.com/1",
                    http_status=404,
                    failure_reason="not_found",
                )
            ],
        ),
    )
    media = _item(
        "2",
        media=[
            _failed_photo(),
            MediaVideoFailed(
                url="https://video.twimg.com/dead.mp4",
                failure_reason="http_5xx",
                attempts=1,
                last_attempt_at=_T,
            ),
            MediaVideoPending(url="https://video.twimg.com/live.mp4"),
        ],
    )
    blocks = [
        ArticleTextBlock(text="Body."),
        ArticleImageBlock(media=_failed_photo()),
        ArticleVideoBlock(media=MediaVideoPending(url="https://video.twimg.com/a.mp4")),
    ]
    article = _item(
        "3",
        content=Content(
            fetched_at=_T,
            sources=[
                ContentSourceSuccess(
                    kind="x_article",
                    url="https://x.com/i/article/3",
                    title="An Article",
                    text="Body.",
                    blocks=blocks,
                )
            ],
        ),
    )
    return {i.id: i for i in (broken, media, article)}


def _render_vault(tmp_path: Path, language: str) -> dict[str, str]:
    store = _store()
    generate(store, tmp_path, output_language=language)
    vocab = [Topic(slug="agents", description="AI agents."), Topic(slug="evals", description="E")]
    posts = compute_topic_posts(store, vocab)
    # `agents` has a STALE overview (synthesised at 1 post, now 3) carrying notes;
    # `evals` has posts only if any item names it — give it a pending page instead.
    store["2"].enriched.topics.append("evals")  # type: ignore[union-attr]
    posts = compute_topic_posts(store, vocab)
    pages = {
        "agents": TopicPage(
            slug="agents",
            overview="An overview.",
            notes=["A key note."],
            synthesized_at=_T,
            post_count_at_synth=1,
        )
    }
    write_topic_pages(tmp_path, vocab, posts, pages, output_language=language)
    return {
        str(p.relative_to(tmp_path)): p.read_text(encoding="utf-8")
        for p in sorted(tmp_path.rglob("*.md"))
    }


def test_the_fixture_reaches_every_leaking_state(tmp_path: Path):
    """Without this, a fixture that stopped producing (say) the stale-overview notice
    would leave the language test green for the wrong reason (rule 1)."""
    vault = _render_vault(tmp_path, "English")
    text = "\n".join(vault.values())
    for marker in ("dead.png", "dead.mp4", "live.mp4", "a.mp4", "HTTP 404", "+2"):
        assert marker in text, marker
    assert "topics/evals.md" in vault and "topics/agents.md" in vault
    assert "_index.md" in vault and "log.md" in vault


def test_an_english_vault_carries_no_spanish(tmp_path: Path):
    vault = _render_vault(tmp_path, "English")
    leaks = {
        name: sorted({m.group(0) for m in _SPANISH.finditer(body)})
        for name, body in vault.items()
        if _SPANISH.search(body)
    }
    # The fixture's own content (post text, summaries, topic descriptions) is English,
    # so ANY Spanish token in a written file came from the renderer.
    assert leaks == {}


def test_a_spanish_vault_keeps_its_exact_wording(tmp_path: Path):
    """Moving the strings into `i18n` must not change a single word for a vault already
    configured in Spanish: each line is pinned where it lives."""
    vault = _render_vault(tmp_path, "Spanish")
    note1 = next(v for k, v in vault.items() if k.startswith("items/") and k.endswith("-1.md"))
    note2 = next(v for k, v in vault.items() if k.startswith("items/") and k.endswith("-2.md"))
    note3 = next(v for k, v in vault.items() if k.startswith("items/") and k.endswith("-3.md"))
    assert (
        "> ⚠ Enlace roto: <https://example.com/1> — HTTP 404 · no encontrado "
        "(verificado 2026-05-10)" in note1.splitlines()
    )
    assert "## Enlaces" in note1.splitlines()
    assert "[Ver tweet original](https://x.com/a/status/1)" in note1.splitlines()
    lines2 = note2.splitlines()
    assert (
        "> ⚠ Foto no disponible (URL no encontrada (HTTP 4xx)): "
        "<https://pbs.twimg.com/media/dead.png>" in lines2
    )
    assert (
        "> ⚠ Vídeo no disponible (error del servidor (HTTP 5xx)): "
        "<https://video.twimg.com/dead.mp4>" in lines2
    )
    assert "> 🎥 [Ver vídeo](https://video.twimg.com/live.mp4) (pendiente de descarga)" in lines2
    lines3 = note3.splitlines()
    assert (
        "> ⚠ Imagen no disponible (URL no encontrada (HTTP 4xx)): "
        "<https://pbs.twimg.com/media/dead.png>" in lines3
    )
    assert "> 🎥 [Ver vídeo](https://video.twimg.com/a.mp4) (pendiente de descarga)" in lines3
    assert "## Mis notas" in note1.splitlines()
    index = vault["_index.md"].splitlines()
    assert "## Índices" in index
    assert "- [[log|Log cronológico completo]]" in index
    assert "- Items totales: 3" in index
    assert vault["log.md"].splitlines()[0] == "# Log cronológico"
    agents = vault["topics/agents.md"].splitlines()
    assert (
        "> ⚠ Overview desactualizado: +2 posts desde la última síntesis. "
        "Ejecuta `xbrain topics --resynth`." in agents
    )
    assert "## Notas importantes" in agents
    assert (
        "*(Overview pendiente — ejecuta `xbrain topics`.)*" in vault["topics/evals.md"].splitlines()
    )


@pytest.mark.parametrize("language", SUPPORTED_LANGUAGES)
def test_every_language_fills_every_note_string(language: str):
    """A language added to `_STRINGS` without one of the new fields must fail here,
    not render an empty line into someone's vault."""
    strings = strings_for(language)
    empty = [name for name, value in vars(strings).items() if isinstance(value, str) and not value]
    assert empty == []
