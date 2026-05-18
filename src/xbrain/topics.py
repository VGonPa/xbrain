"""The `topics` stage — topic-page post lists, rendering and staleness.

Post lists are mechanical (computed from item enrichments). Overview synthesis
lives in `xbrain.topic_synth`; this module renders the pages and decides which
topics need (re)synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from xbrain import notes_io
from xbrain.models import Item, Topic, TopicPage
from xbrain.topic_synth import OverviewJudgment, TopicInput


@dataclass
class TopicPosts:
    """The two post blocks of one topic page."""

    primary: list[Item] = field(default_factory=list)
    also: list[Item] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.primary) + len(self.also)


def compute_topic_posts(store: dict[str, Item], vocab: list[Topic]) -> dict[str, TopicPosts]:
    """Group enriched items into per-topic primary / also-relevant lists.

    A post is *primary* under its `primary_topic` and *also-relevant* under each
    of its other topics. Lists are sorted newest-first. An item whose
    `primary_topic` is not in the current vocabulary is skipped from the primary
    block (its also-relevant placements still apply).
    """
    result: dict[str, TopicPosts] = {topic.slug: TopicPosts() for topic in vocab}
    for item in store.values():
        enriched = item.enriched
        if enriched is None or not enriched.primary_topic:
            continue
        if enriched.primary_topic in result:
            result[enriched.primary_topic].primary.append(item)
        for slug in dict.fromkeys(enriched.topics):
            if slug != enriched.primary_topic and slug in result:
                result[slug].also.append(item)
    for posts in result.values():
        posts.primary.sort(key=lambda i: i.created_at, reverse=True)
        posts.also.sort(key=lambda i: i.created_at, reverse=True)
    return result


def _post_block(heading: str, items: list[Item]) -> list[str]:
    """Render one post block as markdown lines (empty when the block is empty)."""
    if not items:
        return []
    lines = [f"## {heading} ({len(items)})", ""]
    for item in items:
        date = item.created_at.date().isoformat()
        stem = Path(notes_io.note_filename(item)).stem
        title = item.text.replace("\n", " ")[:80] or item.id
        lines.append(f"- `{date}` · @{item.author.handle} · [[items/{stem}|{title}]]")
    lines.append("")
    return lines


def _topic_frontmatter(topic: Topic, posts: TopicPosts) -> str:
    return "\n".join(
        [
            "---",
            f"topic: {topic.slug}",
            f"tags: [x-knowledge-topic, {topic.slug}]",
            f"posts: {posts.total}",
            f"primary_posts: {len(posts.primary)}",
            "---",
        ]
    )


def render_topic_page(topic: Topic, posts: TopicPosts, page: TopicPage | None) -> str:
    """Render one topic page's generated block (frontmatter, overview, lists)."""
    lines = [
        _topic_frontmatter(topic, posts),
        "",
        f"# {topic.slug}",
        "",
        f"> {topic.description}",
        "",
        "## Overview",
        "",
    ]
    if page is None:
        lines += ["*(Overview pendiente — ejecuta `xbrain topics`.)*", ""]
    else:
        if posts.total > page.post_count_at_synth:
            delta = posts.total - page.post_count_at_synth
            lines += [
                f"> ⚠ Overview desactualizado: {delta:+d} posts desde la última "
                "síntesis. Ejecuta `xbrain topics --resynth`.",
                "",
            ]
        lines += [page.overview, ""]
        if page.notes:
            lines += ["## Notas importantes", ""]
            lines += [f"- {note}" for note in page.notes]
            lines += [""]
    lines += _post_block("Posts primarios", posts.primary)
    lines += _post_block("También relevante", posts.also)
    return notes_io.wrap("\n".join(lines).rstrip())


def write_topic_pages(
    output_dir: Path,
    vocab: list[Topic],
    all_posts: dict[str, TopicPosts],
    pages: dict[str, TopicPage],
) -> int:
    """Write `topics/<slug>.md` for every non-empty topic. Returns the count."""
    topics_dir = output_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for topic in vocab:
        posts = all_posts.get(topic.slug, TopicPosts())
        if posts.total == 0:
            continue
        block = render_topic_page(topic, posts, pages.get(topic.slug))
        path = topics_dir / f"{topic.slug}.md"
        tail = (
            notes_io.user_tail(path.read_text(encoding="utf-8"), notes_io.DEFAULT_TAIL)
            if path.exists()
            else notes_io.DEFAULT_TAIL
        )
        path.write_text(block + tail, encoding="utf-8")
        written += 1
    return written


def topics_needing_synth(
    vocab: list[Topic],
    all_posts: dict[str, TopicPosts],
    pages: dict[str, TopicPage],
    threshold: int,
    resynth: bool,
) -> list[str]:
    """The slugs whose overview must be (re)synthesized.

    A topic with no page is always included. With `resynth`, any topic whose
    post count changed is included; otherwise only topics that grew by at least
    `threshold` posts. Empty topics are never synthesized.
    """
    needing: list[str] = []
    for topic in vocab:
        posts = all_posts.get(topic.slug, TopicPosts())
        if posts.total == 0:
            continue
        page = pages.get(topic.slug)
        if page is None:
            needing.append(topic.slug)
            continue
        delta = posts.total - page.post_count_at_synth
        if resynth and delta != 0:
            needing.append(topic.slug)
        elif not resynth and delta >= threshold:
            needing.append(topic.slug)
    return needing


def build_topic_inputs(
    slugs: list[str], vocab: list[Topic], all_posts: dict[str, TopicPosts]
) -> list[TopicInput]:
    """Build the synthesis input for each slug — its description + post summaries."""
    by_slug = {topic.slug: topic for topic in vocab}
    inputs: list[TopicInput] = []
    for slug in slugs:
        topic = by_slug.get(slug)
        if topic is None:
            raise ValueError(f"slug '{slug}' is not in the vocabulary")
        posts = all_posts.get(slug, TopicPosts())
        summaries = [
            item.enriched.summary
            for item in (posts.primary + posts.also)
            if item.enriched and item.enriched.summary
        ]
        inputs.append(TopicInput(slug=slug, description=topic.description, summaries=summaries))
    return inputs


def merge_overviews(
    pages: dict[str, TopicPage],
    judgments: list[OverviewJudgment],
    all_posts: dict[str, TopicPosts],
) -> None:
    """Fold synthesized overviews into the topic-page store (in place)."""
    now = datetime.now(timezone.utc)
    for judgment in judgments:
        posts = all_posts.get(judgment.slug, TopicPosts())
        pages[judgment.slug] = TopicPage(
            slug=judgment.slug,
            overview=judgment.overview,
            notes=judgment.notes,
            synthesized_at=now,
            post_count_at_synth=posts.total,
        )
