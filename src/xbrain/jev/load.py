"""Load what every reader of the Jev side-car needs, once, counted.

`jev report`, `jev dashboard` (and, later, a local server) read the store, the vocabulary
and the side-car through ONE loader, so they can never disagree about which stored
assessments are still current.
"""

from __future__ import annotations

from dataclasses import dataclass

from xbrain.config import Config
from xbrain.jev.assess import CurrentPairs, current_pairs
from xbrain.jev.models import TopicAssessment
from xbrain.jev.store import load_assessments
from xbrain.models import Item, Topic
from xbrain.rubrics import load_vocab
from xbrain.store import load_store


@dataclass(frozen=True)
class JevPairs:
    """Everything a reader of the side-car needs, loaded once and counted.

    `assessments` is the RAW side-car and `pairs` the current subset, with `stale` and
    `orphans` accounting for the difference: `len(assessments) == len(pairs) + stale +
    orphans`. Returning only `pairs` — as this loader first did — throws away the one number
    that distinguishes "nobody has run `xbrain jev topics`" from "a vocabulary edit just
    retired every paid record in the corpus", and those two readings differ by the price of
    re-assessing ~3,000 posts.
    """

    store: dict[str, Item]
    vocab: list[Topic]
    assessments: dict[str, TopicAssessment]
    pairs: list[tuple[Item, TopicAssessment]]
    stale: int
    orphans: int
    #: The two options `pairs` was decided under, carried so a consumer that rebuilds a
    #: `CurrentPairs` from this (the dashboard) hands on the real provenance instead of
    #: re-asserting its own `cfg` and calling that agreement.
    fallback: str
    char_limit: int

    @property
    def unassessed(self) -> int:
        """Posts with no current answer — never asked, or stale: `len(store) - len(pairs)`."""
        return len(self.store) - len(self.pairs)

    def current(self) -> CurrentPairs:
        """The currency verdict as `assess.current_pairs` returns it, rebuilt from what was
        loaded — with the options it was decided under, not a caller's `cfg` again."""
        return CurrentPairs(
            pairs=tuple(self.pairs),
            stale=self.stale,
            orphans=self.orphans,
            fallback=self.fallback,
            char_limit=self.char_limit,
        )


def load_jev_pairs(cfg: Config) -> JevPairs:
    """Store, vocabulary and the CURRENT (item, assessment) pairs, with the drop counts.

    ONE loader for every reader of the side-car — `jev report` and `jev dashboard`. They
    must never disagree about which stored assessments are still current, and two call sites
    each opening the three files their own way is exactly how they would: `assess.current_pairs`
    decides currency from the vocabulary and the fallback, so a reader that loaded a different
    `vocab.yaml` would silently compare a different set.

    It LOADS and COUNTS; it does not judge. Whether an empty result is an error belongs to the
    command (`_refuse_empty_report`), and today BOTH commands say yes: `jev report` and
    `jev dashboard` refuse over an empty side-car, each naming the artifact it protects. A
    future reader that genuinely wants to render the empty state simply does not call it.

    THE ONE EXCEPTION is an empty vocabulary, and it is not a judgement: there is nothing to
    compare WITH, so no comparison is attempted and `stale`/`orphans` stay 0 because nothing
    was examined. Calling `current_pairs` here would raise out of `build_topic_questions` with
    a message about running `xbrain vocab` "antes de `xbrain jev topics`" — the wrong command
    for this caller, and silent about the report it did not overwrite. A reader that finds an
    empty `vocab` must say so itself rather than read the two zeros as "nothing was dropped".
    """
    store = load_store(cfg.items_path)
    vocab = load_vocab(cfg.data_dir / "vocab.yaml")
    assessments = load_assessments(cfg.jev_topics_path)
    if not vocab:
        return JevPairs(
            store=store,
            vocab=vocab,
            assessments=assessments,
            pairs=[],
            stale=0,
            orphans=0,
            fallback=cfg.jev_fallback_option,
            char_limit=cfg.jev_state_char_limit,
        )
    current = current_pairs(
        list(store.values()),
        assessments,
        vocab,
        fallback=cfg.jev_fallback_option,
        char_limit=cfg.jev_state_char_limit,
    )
    return JevPairs(
        store=store,
        vocab=vocab,
        assessments=assessments,
        pairs=list(current.pairs),
        stale=current.stale,
        orphans=current.orphans,
        fallback=current.fallback,
        char_limit=current.char_limit,
    )
