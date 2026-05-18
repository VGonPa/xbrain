"""The `topics` stage — topic-page post lists, rendering and staleness.

Post lists are mechanical (computed from item enrichments). Overview synthesis
lives in `xbrain.topic_synth`; this module renders the pages and decides which
topics need (re)synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from xbrain.models import Item, Topic


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
    of its other topics. Lists are sorted newest-first.
    """
    result: dict[str, TopicPosts] = {topic.slug: TopicPosts() for topic in vocab}
    for item in store.values():
        enriched = item.enriched
        if enriched is None or not enriched.primary_topic:
            continue
        if enriched.primary_topic in result:
            result[enriched.primary_topic].primary.append(item)
        for slug in enriched.topics:
            if slug != enriched.primary_topic and slug in result:
                result[slug].also.append(item)
    for posts in result.values():
        posts.primary.sort(key=lambda i: i.created_at, reverse=True)
        posts.also.sort(key=lambda i: i.created_at, reverse=True)
    return result
