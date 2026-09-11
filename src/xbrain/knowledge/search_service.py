"""`search` as a SERVICE (spec §5.3, §5.4, §7.1, §7.2) — CLI and MCP are adapters over it.

Spec §7.1: *CLI JSON and MCP are adapters over the same application services. They do not
implement two searches.* The way two searches appear is never a decision — the second adapter
is written months later against whatever the first happened to emit — so the service returns
a `SearchResponse` and the adapters render it. Plan 04's MCP tool is then a consumer of this
module, not an author.

THE PIPELINE, five steps, each in its own function because Plan 02 §14 named this file the
natural candidate for a radon D:

    validate -> score (two planes) -> verify fingerprints -> group by item -> hydrate

**Validation refuses, it does not guess.** An empty query is an error, not an empty result:
an empty result set makes a claim about the corpus when the truth is that nothing was asked
(spec §9.3). An unknown topic is an error LISTING the valid slugs, because spec §3.7.5 says
topics and filters come from the store and are never invented from the query text — and
answering with zero results would be exactly that invention, silently.

**Two planes, not fused.** `chunks_fts` produces citable matches; `profiles_fts` produces
item candidates (spec §5.1). Their bm25 scores are computed over different corpora and have
no common scale, so ranking them in one sorted list would invent one. Chunk-matched items
come first — they carry evidence — and profile-only candidates follow. Fusion is Plan 03's
decision, with RRF and the golden set in front of it.

**A topic match expands to the topic's items.** `SearchResult` is item-shaped: it requires an
`item_id`, a `url`, an `author` and a `created_at`, none of which a `topic_note` has. So a
match on a topic surface is attached to that topic's SUPPORTING ITEMS, primary first — which
is what Plan 02 §4 asks for (*el resultado incluye los `supporting_item_ids` del topic, para
que el consumidor pueda saltar a items reales con `get`*) and is stronger than a list of ids,
because the items arrive hydrated.

**Verification comes from the LIVE STORE (M5).** Never from a column: `surface_fingerprint`
does not depend on the verdict, so a stored copy could not be invalidated when the verdict
changed and a `FAIL` revoked by `verify --audit` would keep being served as the old `PASS`.
The freshness check is `verification.verdict_is_current`, the same one
`generate._verdict_badge` applies.

**`verify_with` empty means `no_underlying_source`, and that is the whole statement.** Spec
§3.5: *si una superficie derivada no permite llegar a material sustentante, el resultado debe
decirlo*. A match on a primary surface names that surface; a match on a derived surface names
the item's primary surfaces; and an item that has none gets `()`. `verify_with == ()` is
therefore reachable in exactly one case, which makes the empty tuple the structural form of
the warning. `no_underlying_source()` is the predicate — the word is printed by whichever
adapter Plan 02 §12 lands for the CLI, which does not exist in this package yet — and the
gap — that the FROZEN `SearchResult` has no `warnings` field for a JSON consumer to read it
from — is recorded in the execution report rather than closed by amending a contract Plan 01
froze on purpose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from xbrain.knowledge.chunking import DEFAULT_CHUNKER_PARAMS, ChunkerParams, fragment_locator
from xbrain.knowledge.contracts import (
    SearchFilters,
    SearchMatch,
    SearchResponse,
    SearchResult,
    Strategy,
    resolve_strategy,
)
from xbrain.knowledge.index_schema import REBUILD_ADVICE, IndexIncompatibleError
from xbrain.knowledge.index_store import (
    OpenIndex,
    open_for_query,
    resolvable_hits,
    verify_fingerprints,
)
from xbrain.knowledge.lexical import LexicalHit, OwnerKey, distinct_owners
from xbrain.knowledge.models import DerivedText, SurfaceType
from xbrain.knowledge.provenance import DEFAULT_EVIDENCE_CLASSES, ORIGIN_TRUST
from xbrain.knowledge.surfaces import (
    SURFACE_ORIGIN,
    hydrate_verification,
    item_topics,
    knowledge_item,
)
from xbrain.models import Item, Topic, TopicPage

# The search cursor is an OFFSET into the ranking (M-4, round 08): `s:<offset>`. The other
# two cursor shapes in this package (`get`'s positional `<surface>:<chunk>` and `q:<offset>`)
# index one item's chunks; this one indexes a ranking of items, and each decoder refuses the
# others' shape by name rather than restarting from zero.
_SEARCH_CURSOR_PREFIX = "s"

# WHICH SURFACE TO VERIFY A DERIVED CLAIM AGAINST, most decisive first.
#
# A summary is best checked against the source it summarised, so the content sources lead and
# the post trails: the post is what the item IS, but for a claim the summary made about a
# linked article, the article settles it and the tweet does not. One ordered list, in one
# place, rather than an ordering that emerges from whatever order the emitter happened to
# walk `content.sources` in.
VERIFY_PREFERENCE: tuple[SurfaceType, ...] = (
    "external_article",
    "x_article",
    "thread",
    "quoted_post",
    "video_transcript",
    "video_frame",
    "image_description",
    "post",
    "user_note",
)


@dataclass(frozen=True)
class QueryContext:
    """Everything a query needs that is not the query. Built once by the adapter.

    The STORE is in here, and it is not an optimisation detail: spec §3.7.7 makes `get` read
    the live store, and M5 makes `search` hydrate verification from it. A service that read
    metadata from the index and verdicts from the store would have two answers to *who wrote
    this*, which is the divergence rule 5 is about.

    The configured transcribe/vision commands are NOT here any more (F7-7, round 08). A-4
    (round 02) threaded them through so `get` would serve the same `producer` the build
    had stamped — and what both stamped was the command configured at the time, not the
    one that wrote the text (measured: `[transcribe].command` changed, `producer` changed,
    text and fingerprints identical). The ASR/VLM surfaces now declare no producer, so
    there is nothing for the two doors to disagree on.

    THE THREE INPUT PATHS ARE THREE FIELDS WITH THREE DEFAULTS, never two of them optional
    (P1a). The cheap signal covers `vocab.yaml` and `topics.json` as well as `items.json`, and
    a `None` there does not mean "skip this one" — it means the input is absent, which compares
    EQUAL forever however the file moves. `open_for_query` takes them as a unit for the same
    reason.

    `params` are the chunker parameters the CODE would cut with (M-1, round 05), and they
    come from the same `IndexOptions` too. `open_for_query` compares them against the
    manifest only when handed them, and the one production path never did: a manifest whose
    `chunker_params` had moved was refused by `update` and declared unusable by `status`
    while `search` answered over it — chunks cut differently under identical ids, the case
    the check exists for, reachable by editing the manifest or by changing
    `DEFAULT_CHUNKER_PARAMS` without a version bump.
    """

    store: Mapping[str, Item]
    vocab: Sequence[Topic] = ()
    topic_pages: Mapping[str, TopicPage] = field(default_factory=dict)
    index_dir: Path = Path("data/index")
    items_path: Path = Path("data/items.json")
    vocab_path: Path = Path("data/vocab.yaml")
    topics_path: Path = Path("data/topics.json")
    vault_dir: Path | None = None
    language: str = "English"
    max_matches_per_item: int = 3
    params: ChunkerParams = DEFAULT_CHUNKER_PARAMS


def search(
    query: str,
    context: QueryContext,
    *,
    filters: SearchFilters | None = None,
    limit: int = 10,
    strategy: Strategy = "lexical",
    cursor: str | None = None,
) -> SearchResponse:
    """Run one query end to end and return the frozen envelope (spec §7.2).

    Read-only throughout: the index is opened `mode=ro` and the store is never written. No
    network, no model call — spec §13.12 requires that `search` work without a single LLM.

    `strategy` is what the CALLER asked for; `SearchResponse.strategy` is what RAN. They
    differ whenever the requested retriever has no backend, and the difference is declared as
    a `<requested>_not_implemented` degradation rather than hidden — spec §9.3: *lexical
    sigue operativo y el response declara estrategia degradada; no finge resultados
    vectoriales.* Echoing the request back was that pretence in the one field that names the
    retriever (F-2).

    A PAGE SHORTER THAN THE RANKING IS DECLARED, AND CAN BE CONTINUED (M-4, round 08).
    `truncated` and `cursor` were declared in the frozen envelope and never set: `--limit 2`
    cut a fifty-item ranking to two with `truncated: false` — the silent cut spec §9.3
    forbids, on a field that could not come out any other way (rule 2). Now the candidate
    window is materialised until it holds ONE OWNER MORE than the page needs
    (`LexicalIndex.search_owners`, whose ONLY caller this is — the evaluation harness was
    named here as a second consumer and never called it, and the two therefore score
    different retrievals; see `lexical.OWNER_CHUNK_MULTIPLIER`), so `truncated` is a
    measurement of the ranking against the
    page; the cursor is the offset of the next page (`s:<offset>`), and the pages are
    disjoint and reassemble the ranking in order, because every window is a prefix of the
    same ranking and a topic hit expands to the same sorted members under any page. The
    one truncation that has no cursor is a window that reached `MAX_CHUNK_DEPTH` short of
    owners: more may exist and cannot be paged to, and the response says so instead of
    calling the page complete.
    """
    filters = filters or SearchFilters()
    _validate(query, filters, limit, context)
    executed, strategy_degradation = resolve_strategy(strategy)
    offset = _decode_search_cursor(cursor)
    index = open_for_query(
        context.index_dir,
        context.items_path,
        context.vocab_path,
        context.topics_path,
        params=context.params,
    )
    try:
        # One owner beyond the page: what decides `truncated` without guessing.
        beyond = offset + limit + 1
        ordered, excluded, exhausted = _materialise(index, query, filters, context, needed=beyond)
        page, evidence_excluded = _settle_evidence(
            index, query, filters, context, ordered[offset : offset + limit]
        )
        excluded |= evidence_excluded
        corrupt_chunks_excluded = len(excluded)
        results = tuple(
            _hydrate(rank, item_id, matches, context)
            for rank, (item_id, matches) in enumerate(page, start=offset + 1)
        )
        truncated = len(ordered) > offset + limit or exhausted
        return SearchResponse(
            query=query,
            strategy=executed,
            filters=filters,
            index=index.status_ref(
                corrupt_chunks_excluded=corrupt_chunks_excluded,
                strategy_degradation=strategy_degradation,
            ),
            results=results,
            truncated=truncated,
            cursor=_encode_search_cursor(offset + limit) if len(ordered) > offset + limit else None,
        )
    finally:
        index.close()


def _materialise(
    index: OpenIndex,
    query: str,
    filters: SearchFilters,
    context: QueryContext,
    *,
    needed: int,
) -> tuple[list[tuple[str, list[LexicalHit]]], set[str], bool]:
    """The candidate window, DEEPENED until it can answer — `(ordered, excluded, exhausted)`.

    THE WINDOW IS MEASURED IN CANDIDATES AND THE PAGE IN ANSWERS, and three stages sit between
    the two dropping rows: `resolvable_hits` (no locator), `verify_fingerprints` (evidence that
    does not recompute), and `_group_by_item` (an owner the store no longer holds). Sizing the
    window ONCE and treating the survivors as the ranking is what let every exclusion silently
    shorten the corpus.

    Measured before this loop existed, on two long documents monopolising the eight chunks a
    `--limit 1` page materialises: corrupt the pair and `search` answered `results: []`,
    `truncated: false`, `cursor: null` — a COMPLETE «the corpus has nothing» over a corpus
    that still matched twice. Not a short page: a false claim about the corpus, the same shape
    as answering an unknown topic with zero results. Both say *nothing is there* when the
    truth is *I did not look*.

    BOTH PLANES DEEPEN, EACH ON ITS OWN TERMINATION. The first version of this loop
    doubled the CHUNK window and asked `search_profiles` for a fixed `needed`, then stopped
    when the chunk window stopped growing — so the profile plane could never be looked at
    harder, and for a query answered by that plane ALONE (a handle lives in every profile and
    in no chunk) the chunk window is empty at every depth and the very first shortfall ended
    the search. Measured: delete the three top-ranked owners without reindexing and `limit=1`
    answered `results: []`, `truncated: false`, `cursor: null` over nine owners still present;
    paging the same state walked seven of ten and then handed back no cursor, so three valid
    items were unreachable by paging at all. One plane deepening is not a refill, it is a
    refill for half the corpus.

    So both windows double until the response can serve `needed` owners, or until there is
    nothing deeper to find. THREE TERMINATIONS, and the third is the one that keeps this
    finite: `exhausted` is `search_owners` reporting `MAX_CHUNK_DEPTH` reached, and a pair of
    windows that did not GROW means both rankings are fully materialised — without that, a
    corpus smaller than the windows would double forever asking for owners that do not exist.
    The pair is compared, not the chunk count alone, because a profile plane that can still
    grow is a reason to keep looking even when the chunk plane is finished.

    THE COUNT IS RECOMPUTED PER WINDOW, NEVER ACCUMULATED. Every window is a PREFIX of the same
    ranking (`search_owners`: a deeper window only appends), so the deepest one already
    contains every exclusion the shallower ones saw; adding them up would count the same rows
    once per doubling. `corrupt_chunks_excluded` is what THIS response's candidate set held,
    which is why a response that looked deeper honestly reports more.

    Determinism survives (spec §3.7.8), and it is now a property of BOTH rankings. Each window
    is a prefix of its own plane, chunk-matched owners always precede profile-only ones, and
    `_append_profile_candidates` skips an id already grouped — so a deeper pass only APPENDS,
    and the ordering a page slices is the same ordering whatever depth was reached. That is
    what makes the cursor a position in one ranking rather than an offset into a set that was
    re-mixed per page: the round trip is asserted by walking a cursor to exhaustion and
    requiring the concatenation to EQUAL a single unpaged call.
    """
    grouped, excluded, exhausted = _chunk_owners(index, query, filters, context, needed=needed)
    _fill_from_profiles(index, query, filters, context, grouped, needed=needed)
    return list(grouped.items()), excluded, exhausted


def _chunk_owners(
    index: OpenIndex,
    query: str,
    filters: SearchFilters,
    context: QueryContext,
    *,
    needed: int,
) -> tuple[dict[str, list[LexicalHit]], set[str], bool]:
    """Deepen the CHUNK window until it yields `needed` owners the response can serve.

    THIS RUNS TO COMPLETION BEFORE A SINGLE PROFILE CANDIDATE IS CONSIDERED, and that ordering
    IS the fix. The previous loop appended profile candidates and then tested `len(grouped) >=
    needed`, so a matching profile could satisfy the quota while deeper VALID chunk owners were
    still unreached — the weaker plane answering while the stronger one had an unexamined
    result.

    Measured on thirty items whose text is one marker, so both planes rank the same owners,
    with the eight raw hits a `limit=1` page materialises corrupted: the first page served
    `p000`, a profile-only candidate, when the ranking's head was `p008`, a real chunk match.

    AND IT IS WHAT MAKES THE CURSOR A POSITION IN ONE RANKING. A larger offset materialises a
    deeper window, so while the chunk section could be cut short by profile fill its SIZE was a
    function of the page — every page sliced a different ranking, `p008` was unreachable and
    `p000` came back twice. Deciding the section by `needed` alone makes it grow monotonically
    and freeze once the plane is exhausted, which is the prefix-stability paging needs.

    A hit without a resolvable surface locator is excluded and counted FIRST (B-k): the
    alternative was a locator invented from the chunk's own columns, and since U-5 the
    fingerprint is recomputed over the narrowed locator, so a hit that has none cannot be
    verified at all. Then every survivor's evidence must recompute.

    THE TERMINATION ASKS FOR OWNERS, NOT FOR ROWS, AND THE ROW VERSION WAS WRONG. This loop
    used to stop on `len(candidates) == seen` — «a window that did not GROW means the ranking
    is fully materialised». That premise is FALSE: `search_owners(q, d)` doubles its own query
    window internally and returns the first slice holding `d` distinct owners, so its ROW COUNT
    IS NOT MONOTONE IN `d`. Measured on a corpus of one dense article and fourteen thin ones:

        C(n) = {1: 4, 2: 16, 3: 12, 4: 16, 8: 24}   rows — rises, falls, and ties at 2 and 4
        owners(n) = {1: 1, 2: 7, 3: 3, 4: 7, 8: 15}

    `C(2) == C(4)` while the ranking holds 24 rows over 15 owners, so the equality fired on a
    coincidence and the plane was declared finished with rows never examined. With the leading
    owners deleted from the live store, `limit=1` then answered `results: []`,
    `truncated: false`, `cursor: null` over eight items the same service returns at
    `limit=200` — spec §9.3's silent cut in its worst form, a false claim about the corpus.

    `distinct_owners(candidates) < depth` is the honest question, and it is monotone because it
    counts the thing the depth is expressed in: `search_owners` guarantees `depth` distinct
    owners unless the ranking cannot supply them, so falling short IS «fully materialised».
    The rows were never the quantity being asked about.

    THE SIBLING LOOP DOES NOT SHARE THIS, and that was measured rather than assumed.
    `_fill_from_profiles` terminates on its own row count, but `search_profiles` issues a plain
    `LIMIT ?`, so its count is `min(n, total)` — monotone, measured
    `P(n) = {1: 1, 2: 2, 3: 3, 4: 4, 8: 8, 12: 12, 16: 12, 32: 12}` — and a tie there can only
    mean saturation. One loop needed the change, not two.

    The exclusions are returned as IDS, not as a count, and the response reports the size of
    their UNION with the evidence phase's. A row rejected by two candidate sets is one row that
    could not be served; counting visits read 2, 4, 6 and then 7 for one, two, three and four
    forged rows — not a consistent multiple, so not a count of anything. Within this loop the
    final window's set is the whole answer anyway: every window is a prefix of the same ranking,
    so the deepest already holds every exclusion the shallower ones saw.
    """
    depth = needed
    while True:
        candidates, exhausted = index.lexical.search_owners(query, depth, filters=filters)
        hits, _unresolvable = resolvable_hits(candidates)
        hits, _corrupt = verify_fingerprints(hits)
        excluded = {hit.chunk_id for hit in candidates} - {hit.chunk_id for hit in hits}
        grouped = _group_by_item(hits, context, limit=needed)
        if len(grouped) >= needed or exhausted or distinct_owners(candidates) < depth:
            return grouped, excluded, exhausted
        depth *= 2


def _verified_top(
    index: OpenIndex,
    query: str,
    filters: SearchFilters,
    owners: tuple[OwnerKey, ...],
    wanted: int,
    excluded: set[str],
) -> list[LexicalHit]:
    """The best `wanted` chunks of `owners` that SURVIVE verification, deepening to find them.

    EXCLUSION MUST COST THE ROW, NEVER THE SLOT, and that was the defect. This asked for exactly
    `wanted` rows and verified afterwards, so a forged row consumed a citation slot a VALID chunk
    of the same owner could have filled. Measured on one item with six citable chunks at `cap=3`:
    one forged row left 2 matches, two left 1, three left NONE — the item served at rank 1 with
    no citations while three valid chunks of its own sat in the index unasked, `verify_with`
    flipped to the derived branch, and `no_underlying_source` could not warn because it tests
    `bool(result.matches)`.

    So the window doubles until `wanted` rows survive or the owners' rows run out. It is bounded
    by the owners' own chunk count, which is why this is affordable where deepening the GLOBAL
    window was not (round 7 measured that at 5-56x on the real corpus).

    The ids it rejects go into `excluded`, a SET: `corrupt_chunks_excluded` answers how many rows
    this response could not serve, and a row rejected by two candidate sets is one row. Counting
    visits instead read 2, 4, 6 and then 7 for one, two, three and four forged rows — not a
    consistent multiple, so not a count of anything.

    `owners` are `OwnerKey` PAIRS, and they have to be: an id alone does not say which plane
    it names, and the two planes share an alphabet (C1, `lexical.owner_key`).
    """
    if not owners or wanted <= 0:
        return []
    depth = wanted
    seen = -1
    while True:
        scoped = index.lexical.search(query, depth, filters=filters, owners=owners)
        kept, _ = resolvable_hits(scoped)
        kept, _ = verify_fingerprints(kept)
        survivors = {hit.chunk_id for hit in kept}
        excluded |= {hit.chunk_id for hit in scoped} - survivors
        if len(kept) >= wanted or len(scoped) == seen:
            return list(kept[:wanted])
        seen = len(scoped)
        depth *= 2


def _settle_evidence(
    index: OpenIndex,
    query: str,
    filters: SearchFilters,
    context: QueryContext,
    page: list[tuple[str, list[LexicalHit]]],
) -> tuple[list[tuple[str, list[LexicalHit]]], set[str]]:
    """Re-derive each SERVED item's matches, its OWN chunks first, then its topics'.

    THE EVIDENCE A RESULT CARRIES MUST NOT DEPEND ON THE PAGE SIZE, and it did. The candidate
    window is sized in OWNERS and `_group_by_item` filled each bucket from whatever that window
    happened to hold, so an item's second-best chunk was included or not according to `limit`:

        --limit  5: X000 rank=1 matches=['summary']                    verify_with=('external_article','post')
        --limit 10: X000 rank=1 matches=['summary','external_article'] verify_with=('external_article',)

    Same corpus, same query, same item, same rank — and the consequence is not a shorter list.
    With only a machine-written `summary` to cite, `_verify_with` takes the DERIVED branch and
    says *go check the article and the post*; with the article matched it says *check the
    article*. Two verification instructions for one claim, and spec §3.5 makes `verify_with` an
    instruction rather than a display detail.

    THE OWNER UNIVERSE COMES FROM THE STORE, NOT FROM THE BUCKET. Re-deriving from the hits
    already collected reproduced the page-dependence one layer down: an item whose TOPIC hit fell
    outside the smaller window offered only its own owner, a wider window offered the topic too.
    `item_topics` reads the assignment off the ITEM, so the universe is a property of the corpus.

    AND THE ITEM'S OWN CHUNKS FILL THE SLOTS FIRST. Widening that universe let a topic's
    description and overview beat the item's second article chunk under `cap=3` — page-independent
    and consistently less primary evidence. A topic note is synthesized prose about a GROUP; the
    item's article is the thing a reader can check, which is the distinction `verify_with` exists
    to draw, so it decides slots too. Topics fill only what the item's own chunks leave.

    A profile-only candidate has no chunk evidence by construction and is left alone: an empty
    bucket means the chunk plane never selected this item, and giving it evidence here would
    invent a citation the ranking never made.

    ONE SCOPED QUERY FAMILY PER SERVED ITEM, bounded by that item's own rows. `LexicalIndex.search`
    narrows by `owners`, internal narrowing deliberately kept off the FROZEN `SearchFilters`; this
    is its only caller in the package (it was documented as also serving `get` and the evaluation
    harness, and `get` does not exist yet).

    WHAT THIS STAGE COSTS, measured rather than remembered (B1, round 11). The figure here read
    "0.7-2.0x" for three rounds and no measurement ever reproduced it. Re-measured at round 11
    on the population this sentence names — the real 2,474-item corpus, 45 topics, 22,933 chunks,
    `items.json` sha256 `4fed54a0…` — against the IDENTICAL query with this function replaced by
    a passthrough, so the ratio isolates the stage rather than the whole response. Best of five
    after a warm-up:

        de 5.06x · la 3.89x · openai 4.02x · anthropic 4.14x
        video 3.31x · modelo 2.98x · agente 2.88x · retrieval 2.12x

    **2.1x-5.1x, median 3.6x.** The round-10 gate measured the same stage independently and got
    2.08x-7.48x, median 4.11x: different cells, same band and the same direction.

    And 0.7 was never reachable. This stage only ADDS queries to the work the passthrough already
    did, so a ratio below 1.0 could not come out however the corpus fell — a number that cannot
    come out another way is not a measurement (rule 2), which is what carried a wrong figure
    through three rounds of green.

    It is still the cheaper of the two designs by a wide margin: deepening the GLOBAL window to
    the same end was measured at 5-56x on this corpus (`de` 67 ms -> 3,723 ms). The comparison
    survives the correction; the band that made it look free did not.

    THE TWO PHASES NARROW BY OWNER KEY, not by id. A topic slug may be all digits and so is every
    tweet id, so `(item, X)` and `(topic, X)` are both representable and were both answered by an
    id-only clause: the own-chunks phase served a same-named topic's prose in the PRIMARY slots,
    and the top-up phase served another item's article under this item's name (C1, round 10).
    """
    cap = context.max_matches_per_item
    settled: list[tuple[str, list[LexicalHit]]] = []
    excluded: set[str] = set()
    for item_id, hits in page:
        if not hits:
            settled.append((item_id, hits))
            continue
        own = _verified_top(index, query, filters, (("item", item_id),), cap, excluded)
        if len(own) < cap:
            slugs = tuple(item_topics(context.store[item_id])) if item_id in context.store else ()
            topics: tuple[OwnerKey, ...] = tuple(("topic", slug) for slug in slugs)
            own += _verified_top(index, query, filters, topics, cap - len(own), excluded)
        settled.append((item_id, own))
    return settled, excluded


def _fill_from_profiles(
    index: OpenIndex,
    query: str,
    filters: SearchFilters,
    context: QueryContext,
    grouped: dict[str, list[LexicalHit]],
    *,
    needed: int,
) -> None:
    """Top the ranking up from the PROFILE plane, deepening it the way the chunk plane deepens.

    Only what the chunk plane could not supply, and only after it has been asked to exhaustion
    — so a profile candidate can never displace a chunk owner that was simply not looked at
    yet. Deepening matters here for its own reason: `_append_profile_candidates` skips an id
    the store no longer holds, so a fixed depth let deleted owners consume the quota and end
    the search with a short page (the round-2 finding, on the plane round 1 did not touch).

    The termination is the profile ranking refusing to grow, which is what keeps a corpus
    smaller than the window from doubling forever asking for owners that do not exist.
    """
    depth = needed
    seen = -1
    while len(grouped) < needed:
        profiles = index.lexical.search_profiles(query, depth, filters=filters)
        _append_profile_candidates(
            grouped, [hit.item_id for hit in profiles], context, limit=needed
        )
        if len(profiles) == seen:
            return
        seen = len(profiles)
        depth *= 2


def _encode_search_cursor(offset: int) -> str:
    """The cursor of the page starting at `offset` in the ranking. Opaque to the caller."""
    return f"{_SEARCH_CURSOR_PREFIX}:{offset}"


def _decode_search_cursor(cursor: str | None) -> int:
    """The offset a search cursor names — refusing `get`'s two shapes and anything malformed.

    Restarting from zero on a bad cursor would loop a paginating consumer forever while
    looking like progress; the refusal names which sequence the cursor belongs to.
    """
    if not cursor:
        return 0
    head, _, tail = cursor.partition(":")
    if head != _SEARCH_CURSOR_PREFIX:
        raise ValueError(
            f"Cursor inválido: {cursor!r} no es un cursor de `search` (esos son `s:<offset>`); "
            "los cursores de `get` no indexan esta secuencia."
        )
    try:
        offset = int(tail)
    except ValueError as error:
        raise ValueError(
            f"Cursor inválido: {cursor!r}. Usa el que devolvió la respuesta anterior."
        ) from error
    if offset < 0:
        raise ValueError(f"Cursor inválido: {cursor!r}. Usa el que devolvió la respuesta anterior.")
    return offset


def no_underlying_source(result: SearchResult) -> bool:
    """True when a DERIVED match on this item leads to no recoverable source (spec §3.5).

    The predicate behind the `no_underlying_source` warning. `verify_with` is empty in
    exactly this case — a primary match names its own surface and a derived match names the
    item's primary surfaces — so the empty tuple is the structural statement and this is its
    name.
    """
    return bool(result.matches) and not result.verify_with


# ---------------------------------------------------------------------------
# 1 — validate
# ---------------------------------------------------------------------------


def _validate(query: str, filters: SearchFilters, limit: int, context: QueryContext) -> None:
    """Refuse what cannot be answered honestly, naming what would have been valid.

    An unknown topic is the interesting one. Answering it with zero results would be a claim
    about the corpus produced by a typo, and it would be indistinguishable from a topic that
    genuinely has no matches — so spec §3.7.5's *los topics no se inventan desde el texto del
    query* is enforced by listing the vocabulary instead.

    AND AN EMPTY VOCABULARY IS NOT A REASON TO STOP CHECKING. The guard read `if filters.topics
    and context.vocab`, so the one state in which EVERY topic is unknown was the one state it
    skipped: a filter naming a topic that exists nowhere came back `0 results`, exit 0. The
    sentence names that case rather than inviting the operator to read an empty list of valid
    slugs as "none matched".
    """
    if not query.strip():
        raise ValueError("La consulta está vacía. Escribe algo que buscar.")
    if limit <= 0:
        raise ValueError(f"--limit debe ser >= 1, recibido {limit}.")
    if filters.topics:
        known = {topic.slug for topic in context.vocab}
        unknown = [slug for slug in filters.topics if slug not in known]
        if unknown:
            valid = (
                f"Los válidos son: {', '.join(sorted(known))}."
                if known
                else ("El vocabulario está vacío: no hay ningún topic válido todavía.")
            )
            raise ValueError(f"Topic(s) desconocido(s): {', '.join(unknown)}. {valid}")


# ---------------------------------------------------------------------------
# 4 — group by item (spec §5.4)
# ---------------------------------------------------------------------------


def _group_by_item(
    hits: Sequence[LexicalHit], context: QueryContext, *, limit: int
) -> dict[str, list[LexicalHit]]:
    """Collapse chunk hits into item buckets, capped at `max_matches_per_item`.

    Spec §5.4: grouping is what stops a long transcript filling the top ten with ten adjacent
    windows of itself. The cap is per item and the hits arrive already ranked, so the ones
    kept are the best ones; the item's own rank is its best chunk's, and the rest order
    inside it.

    A TOPIC-owned hit has no item to be grouped under, so it expands to the topic's
    supporting items (see `_topic_owners`).
    """
    grouped: dict[str, list[LexicalHit]] = {}
    for hit in hits:
        for item_id in _owners_of(hit, context, limit=limit):
            bucket = grouped.setdefault(item_id, [])
            if len(bucket) < context.max_matches_per_item:
                bucket.append(hit)
    return grouped


def _owners_of(hit: LexicalHit, context: QueryContext, *, limit: int) -> list[str]:
    """The item ids this hit belongs to. One for an item chunk, several for a topic chunk."""
    if hit.owner_type == "item":
        return [hit.owner_id] if hit.owner_id in context.store else []
    return _topic_owners(hit.owner_id, context, limit=limit)


def _topic_owners(slug: str, context: QueryContext, *, limit: int) -> list[str]:
    """The items a topic surface points at, PRIMARY first, capped at the response limit.

    Capped because a topic with 173 primary items would otherwise flood the answer with one
    note's worth of evidence. Primary first because a primary assignment is the stronger
    claim that the item is about the topic, and sorted within each group so two runs over the
    same store agree (spec §3.7.8).
    """
    primary = sorted(
        item_id
        for item_id, item in context.store.items()
        if item.enriched and item.enriched.primary_topic == slug
    )
    secondary = sorted(
        item_id
        for item_id, item in context.store.items()
        if item.enriched and slug in item.enriched.topics and item_id not in primary
    )
    return (primary + secondary)[:limit]


def _append_profile_candidates(
    grouped: dict[str, list[LexicalHit]],
    profile_ids: Sequence[str],
    context: QueryContext,
    *,
    limit: int,
) -> None:
    """Add profile-only items AFTER the chunk-matched ones, never interleaved.

    The two planes' bm25 scores are computed over different corpora, so there is no scale on
    which to compare them; appending is a declared ordering rather than an invented one.
    Chunk matches come first because they carry a citable excerpt, which is what a consumer
    can actually verify — a profile match says *this item is about that*, and the profile
    itself may never be quoted (spec §5.1.A).
    """
    for item_id in profile_ids:
        if len(grouped) >= limit:
            return
        if item_id not in grouped and item_id in context.store:
            grouped[item_id] = []


# ---------------------------------------------------------------------------
# 5 — hydrate (spec §5.4)
# ---------------------------------------------------------------------------


def _hydrate(
    rank: int, item_id: str, hits: Sequence[LexicalHit], context: QueryContext
) -> SearchResult:
    """One item's result: metadata, labelled context, matches, and what to ask `get` for.

    `summary` and `topics` travel as CONTEXT, and the summary carries its `origin` in the
    same object (`DerivedText`), which is invariant 2 of spec §3.7 made structural: the text
    cannot be read without the provenance that qualifies it.
    """
    item = context.store[item_id]
    projection = knowledge_item(item, vault_dir=context.vault_dir)
    verdicts = hydrate_verification(item, context.language)
    summary = None
    if item.enriched is not None and item.enriched.summary:
        summary = DerivedText(
            text=item.enriched.summary,
            origin=SURFACE_ORIGIN["summary"],
            verification_status=(verdicts["summary"].verdict if "summary" in verdicts else None),
        )
    matches = tuple(_match(position, hit) for position, hit in enumerate(hits, start=1))
    return SearchResult(
        rank=rank,
        item_id=item.id,
        url=item.url,
        author=item.author,
        created_at=item.created_at,
        summary=summary,
        topics=item_topics(item),
        matches=matches,
        available_surfaces=projection.available_surfaces,
        verify_with=_verify_with(matches, projection.available_surfaces),
    )


def _match(position: int, hit: LexicalHit) -> SearchMatch:
    """One chunk, with WHY it matched and where it came from (spec §5.3).

    `score` is bm25's, and bm25 is negative-is-better; it is carried unchanged and documented
    as a RANKING SIGNAL rather than a probability, which spec §5.3 requires — a fused rank
    has no calibrated scale, and neither does an unfused one.

    THE TITLE IS THE SURFACE'S TOO, AND LEAVING IT OUT WAS NOT A DEFAULT — IT WAS A DROP.
    `chunks` stores it, `LexicalIndex` reads it back into `LexicalHit.title`, and this is the
    only production `SearchMatch` constructor in the tree; omitting it here meant every article
    fragment reached a consumer as an orphan paragraph, with `SearchResponse.schema_version`
    already bumped to `"2"` FOR this field. Spec §4 asks the title to accompany its chunk, and
    a chunk that cannot say which work it came from is the fragment problem the locator exists
    to solve, one field over. `None` stays the honest value where the surface has none: a post
    body, a summary and an image description have no title, and inventing the item's URL or a
    neighbour's title for them is what this must not become.

    THE ATTRIBUTION AND THE LOCATOR ARE THE SURFACE'S (A-1). The first version left
    `attribution` at its default and fabricated a locator from the chunk's own columns —
    `source_index: null`, `content_kind: null`, the ITEM's url — so a quoted post's match
    was served under the poster's name and pointed at the poster's tweet: the exact defect
    CLAUDE.md lists as paid for in blood, on a new LLM surface (spec §3.7 invariant 3, §3.8).
    The surface's locator says where the surface lives; the chunk's range narrows it to the
    match — through `fragment_locator`, the SAME function that builds the locator of the
    chunk `get` serves (seam b, round 06), so the two services cannot disagree on where a
    fragment lives. And no fallback (B-k): a hit with no resolvable surface locator was
    excluded by `resolvable_hits` upstream; the guard below is what keeps a fabrication
    from ever being reachable again, not a path a caller takes.
    """
    if hit.surface_locator is None:
        raise IndexIncompatibleError(
            f"El chunk {hit.chunk_id} no resuelve a su superficie. {REBUILD_ADVICE}"
        )
    return SearchMatch(
        chunk_id=hit.chunk_id,
        surface_type=hit.surface_type,
        origin=hit.origin,  # type: ignore[arg-type]
        trust_class=hit.trust_class,  # type: ignore[arg-type]
        derived=hit.derived,
        excerpt=hit.excerpt,
        title=hit.title,
        attribution=hit.attribution,
        matched_by=("lexical",),
        lexical_rank=position,
        score=hit.score,
        locator=fragment_locator(hit.surface_locator, hit.char_start, hit.char_end),
    )


def _verify_with(
    matches: Sequence[SearchMatch], available: Sequence[SurfaceType]
) -> tuple[SurfaceType, ...]:
    """What to ask `get` for in order to check these matches (spec §5.4, §3.5).

    A PRIMARY match names its own surface: you verify a quoted post by fetching the quoted
    post. A DERIVED match — summary, digest, topic overview or note — names the item's
    primary surfaces instead, because spec §3.5 says a derived result is a discovery signal
    and the underlying source is what settles the claim.

    An item with no primary surface at all yields `()`, and that is the `no_underlying_source`
    state: it is reachable in exactly this one case, which is what makes the empty tuple an
    unambiguous statement rather than an absence.
    """
    primary_matched = [
        match.surface_type
        for match in matches
        if ORIGIN_TRUST[SURFACE_ORIGIN[match.surface_type]] in DEFAULT_EVIDENCE_CLASSES
    ]
    if primary_matched:
        return tuple(dict.fromkeys(primary_matched))
    candidates = {
        surface
        for surface in available
        if ORIGIN_TRUST[SURFACE_ORIGIN[surface]] in DEFAULT_EVIDENCE_CLASSES
    }
    return tuple(surface for surface in VERIFY_PREFERENCE if surface in candidates)


def supporting_item_ids(slug: str, context: QueryContext, *, limit: int = 10) -> tuple[str, ...]:
    """The items a topic surface is supported by — public, for Plan 02 §12's CLI and Plan 04.

    Exposed rather than left private because Plan 02 §4 requires the consumer to be able to
    jump from a topic match to real items, and the CLI's human output has to name them.
    """
    return tuple(_topic_owners(slug, context, limit=limit))
