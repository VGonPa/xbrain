# tests/jev_ask_fixtures.py — a repo shaped like the first real `jev ask` (2026-09-27).
"""Víctor asked «En qué topics hablo de agentic engineering?» over the posts since 2026-05-07:
808 answers, max 0.80, 35 at or above 0.7, 156 at or above 0.5, median 0.15. Under the old rule
(results = answers ≥ `[jev].threshold` = 0.85) that was «0 de 808». The history entry here is
written in the shape that use left on disk (`last_threshold` 0.85, `last_results` 0, the
filter `since`), so a test can reopen it exactly as his page does.

Answers are built by `ask.assess_post` with a fake client, so each carries the real contract
of its post's state — current, reopened for free. No run log: nothing here was paid.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.jev_fakes import FakeJevClient
from xbrain.config import Config, load_config
from xbrain.jev.ask import AskQuery, assess_post
from xbrain.jev.assess import assess_topics
from xbrain.jev.client import JevResult, NoulAnswer, Question
from xbrain.jev.questions import ASK_KEY, build_topic_questions
from xbrain.jev.store import ASK_INDEX, save_asks, save_assessments
from xbrain.models import Author, Enrichment, Item, Topic
from xbrain.rubrics import save_vocab
from xbrain.store import save_store

VICTOR_QUERY = "En qué topics hablo de agentic engineering?"
VICTOR_SINCE = "2026-05-07"
VOCAB = [
    Topic(slug="agentic-engineering", description="Construir con agentes."),
    Topic(slug="startups", description="Fundar empresas."),
    Topic(slug="nutricion", description="Comer bien."),
]
_START = datetime(2026, 5, 7, 12, tzinfo=timezone.utc)


def victor_probabilities() -> list[float]:
    """808 probabilities with the real distribution's landmarks, ties included."""
    top = [0.80, 0.79, 0.79, 0.77] + [0.76] * 6 + [0.75] + [0.74] * 4 + [0.73] * 3
    top += [0.72] * 6 + [0.71] * 5 + [0.70] * 6  # 35 at or above 0.7
    middle = [round(0.69 - (i % 20) * 0.01, 2) for i in range(121)]  # 0.50 … 0.69
    low = [0.05] * 200 + [0.15] * 250 + [0.30] * 202
    values = top + middle + low
    assert len(values) == 808
    return values


class _Planned(FakeJevClient):
    """Answers the ask's Noul with the probability planned for the post (`pNNN` in its text)."""

    def __init__(self, planned: dict[str, float]) -> None:
        super().__init__(provider="typesafe", input_tokens=1_172)
        self.planned = planned

    def ask(self, state: dict[str, str], questions: dict[str, Question]) -> JevResult:
        result = super().ask(state, questions)
        key = state["post"].split()[0]
        return JevResult(
            provider=result.provider,
            model=result.model,
            answers={ASK_KEY: NoulAnswer(noul=self.planned[key])},
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )


def victor_item(
    n: int,
    *,
    topic: str,
    created: datetime,
    handle: str = "someone",
    topics: list[str] | None = None,
) -> Item:
    item_id = f"{1_000_000 + n}"
    return Item(
        id=item_id,
        source="bookmark",
        url=f"https://x.com/{handle}/status/{item_id}",
        author=Author(handle=handle, name=handle.title()),
        text=f"p{n:04d} un post sobre {topic}",
        created_at=created,
        captured_at=created,
        enriched=Enrichment(
            enriched_at=created,
            executor="claude-code",
            summary="s",
            primary_topic=topic,
            topics=topics or [topic],
        ),
    )


#: With `topics=True`: Jev's `agentic-engineering` noul for the enrich-`startups` posts, by
#: `n % 7` — 0 puts the post in the topic at the default bar (0.85), 1 falls just short
#: (0.84), anything else is far below. So the Jev side of a topic refine is exercised on both
#: sides of the bar, on posts enrich alone would never put there.
JEV_AGENTIC = {0: 0.85, 1: 0.84}


def victor_topic_side_car(cfg: Config, store: dict[str, Item]) -> None:
    """A CURRENT topics side-car over every other post: Jev puts some enrich-`startups` posts
    in `agentic-engineering` at exactly 0.85, some at 0.84 (`JEV_AGENTIC`)."""
    questions = build_topic_questions(VOCAB, cfg.jev_fallback_option)
    records = {}
    for n, item in enumerate(store.values()):
        if n % 2:
            continue
        startups = item.enriched is not None and item.enriched.primary_topic == "startups"
        noul = JEV_AGENTIC.get(n % 7, 0.05) if startups else 0.9
        client = FakeJevClient(nouls={"agentic-engineering": noul})
        records[item.id] = assess_topics(
            item, questions, client, char_limit=cfg.jev_state_char_limit
        )
    save_assessments(records, cfg.jev_topics_path)


#: With `varied`: five authors (one handle in mixed case, one with several leading «@» typed
#: in the tests), a post's author by `n % 5`.
VARIED_HANDLES = ["someone", "Ana_B", "carlos", "Dana", "eve"]


def _varied(n: int, p: float) -> dict:
    """`varied`: the post's author, day and topics spread so the list view has something to
    sort and group — posts over ~2 months in an order unrelated to the ranking (a permutation
    of n), two posts at each moment (ties the base order breaks), a third with a second topic
    (a post under two topics), every eleventh with `nutricion` as its primary."""
    primary = "nutricion" if n % 11 == 0 else ("agentic-engineering" if p >= 0.5 else "startups")
    topics = [primary] + (["nutricion"] if n % 3 == 0 and primary != "nutricion" else [])
    return {
        "handle": VARIED_HANDLES[n % 5],
        "created": _START + timedelta(hours=4 * (((n * 389) % 808) // 2)),
        "topic": primary,
        "topics": topics,
    }


def victor_shaped_repo(
    root: Path, *, older: int = 5, jev: str = "", topics: bool = False, varied: bool = False
) -> Config:
    """A repo whose one query is Víctor's, answered on the 808 posts since 2026-05-07, with
    `older` posts before that day (outside the filter, never asked). Enrich puts the posts
    with the higher probabilities in `agentic-engineering`, the rest in `startups`. With
    `topics`, a current topics side-car too (`victor_topic_side_car`); with `varied`, several
    authors, days and topics (`_varied`)."""
    vault = root / "vault"
    vault.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(
        f'[paths]\nvault = "{vault}"\noutput_subdir = "x"\ndata_dir = "data"\n'
        f'[x]\nhandle = "v"\n[jev]\nconcurrency = 2\n{jev}',
        encoding="utf-8",
    )
    (root / "data").mkdir(exist_ok=True)
    probabilities = victor_probabilities()
    store: dict[str, Item] = {}
    planned: dict[str, float] = {}
    for n, p in enumerate(probabilities):
        if varied:
            item = victor_item(n, **_varied(n, p))
        else:
            item = victor_item(
                n,
                topic="agentic-engineering" if p >= 0.5 else "startups",
                created=_START + timedelta(hours=n),
            )
        store[item.id] = item
        planned[f"p{n:04d}"] = p
    for k in range(older):
        n = len(probabilities) + k
        item = victor_item(n, topic="startups", created=_START - timedelta(days=1 + k))
        store[item.id] = item
    save_store(store, root / "data" / "items.json")
    save_vocab(VOCAB, root / "data" / "vocab.yaml")
    cfg = load_config(root)
    if topics:
        victor_topic_side_car(cfg, store)
    query = AskQuery.of(VICTOR_QUERY)
    client = _Planned(planned)
    asked_at = datetime(2026, 9, 27, 7, 16, 32, tzinfo=timezone.utc)
    records = {
        item.id: assess_post(item, query, client, char_limit=cfg.jev_state_char_limit, now=asked_at)
        for item in list(store.values())[: len(probabilities)]
    }
    cfg.jev_asks_dir.mkdir(parents=True)
    save_asks(query, records, cfg.jev_asks_dir / f"{query.sha}.json")
    moment = "2026-09-27T07:16:58.345477Z"
    legacy = {
        "calibration": {
            "answers": 808,
            "chars": 2997616.0,
            "chars_sq": 90863471684.0,
            "chars_tokens": 21768022920.0,
            "tokens": 946824.0,
        },
        "queries": {
            query.sha: {
                "first_asked_at": moment,
                "last_asked_at": moment,
                "last_evaluated": 808,
                "last_filters": {"since": VICTOR_SINCE},
                "last_results": 0,
                "last_threshold": 0.85,
                "query": query.text,
                "query_sha": query.sha,
                "rebuilt": False,
                "times": 1,
            }
        },
    }
    (cfg.jev_asks_dir / ASK_INDEX).write_text(json.dumps(legacy), encoding="utf-8")
    return cfg
