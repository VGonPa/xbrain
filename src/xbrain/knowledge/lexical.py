"""The lexical retriever over the persisted schema — the SAME scorer, a different home.

This replaces `lexical_memory.InMemoryLexicalIndex` (Plan 02 §9). Plan 01 §5.3 justified
building the baseline on FTS5 rather than a hand-written BM25 with the claim that *what dies
in Plan 02 is where the database lives, not how it scores*; this module is where that claim
is either kept or quietly broken. It is kept: the tokenizer, the indexed column set, the
connective and the tie-break all come from `lexical_fts.py`, and the only thing that changed
is the connection — `sqlite3(":memory:")` for the evaluation harness,
`data/index/knowledge.db` for the real index. The characterization fixture
(`tests/fixtures/knowledge_ranking.json`) is NOT regenerated, and its assertion moved here
with the module (m-vii).

FILTERS ARE `WHERE` CLAUSES, NOT QUERY TEXT (spec §5.3). All eight of spec §7.2 are pushed
into SQL BEFORE the `MATCH` runs, over columns denormalised onto `chunks` (`created_at`,
`source`, `primary_topic`) and `EXISTS` sub-queries over the metadata tables. The alternative
— appending the filter's words to the query string — lets the filter compete for bm25 weight
against the user's terms, and a filter that changes the RANKING is not a filter.

TWO PLANES, TWO RETURN TYPES (spec §5.1). `search` returns `LexicalHit`s, which carry an
excerpt and resolve back to a surface: they are citable. `search_profiles` returns
`ProfileHit`s, which carry an item id and a score and NOTHING ELSE. The profile is a string
nobody wrote — a post, a summary and three topic descriptions glued together — so a shape
that could not carry an excerpt is what stops it ever being presented as a quotation.

ITEM-SCOPED FILTERS FAIL CLOSED ON TOPIC-OWNED CHUNKS. A `topic_note` has no author, no date
and no source; `created_at`, `source` and `primary_topic` are NULL on its row, so a date or
author filter excludes it by construction. The one exception is the topic filter itself,
which matches a topic surface against its OWN slug — otherwise filtering by a topic would
exclude exactly the surfaces that ARE that topic.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from xbrain.knowledge.contracts import SearchFilters
from xbrain.knowledge.index_schema import REBUILD_ADVICE, IndexIncompatibleError
from xbrain.knowledge.lexical_fts import match_expression, rank_order
from xbrain.knowledge.models import KnowledgeChunk, Locator, OwnerType, SurfaceType
from xbrain.models import Author

# How much of a matching chunk is shown back. Long enough to recognise the hit, short enough
# that a log or a report never carries an article (spec §10.8).
EXCERPT_CHARS = 300

# The candidate window of a query is counted in OWNERS, never in chunks (U-6, round 07): a
# transcript matching in seven windows at the top of the ranking held ten chunks and two
# owners, so a depth of ten CHUNKS answered a question about ten ITEMS with two. The window
# starts at `OWNER_CHUNK_MULTIPLIER` chunks per owner asked for and DOUBLES while the result
# set came back full and still holds fewer owners than asked — a set shorter than its limit
# is the whole ranking, and there is nothing deeper to find — up to `MAX_CHUNK_DEPTH`.
# Reaching the bound short of owners is DECLARED to the caller, never absorbed.
#
# TWO CONSUMERS, AND THAT IS THE RULE-5 BINDING: `search_service._chunk_owners` (M-4, round
# 08), which decides `truncated` over this window, and `evaluation._search`, which scores the
# published baseline over the SAME window. One function, so what the harness measures at
# depth N is what the service pages at depth N.
#
# THE BINDING WAS ABSENT FROM THIS TREE, and a round-10 comment here recorded the absence as
# though it were the design: `evaluation.py` contained neither `search_owners` nor
# `exhausted`, its `_search` called `LexicalIndex.search(q, limit)`, and the only caller was
# the service. What that cost is measurable, on a 31-item corpus where one item monopolises
# the head of the chunk ranking:
#
#     k= 5   chunk-limited search(q,k):  5 chunks /  1 owner    owner window:  20 chunks / 15 owners
#     k=10   chunk-limited search(q,k): 10 chunks /  5 owners   owner window:  36 chunks / 31 owners
#
# A `recall@5` of 1/1 owners and one of 15/15 are not the same measurement, so while the two
# diverged a baseline figure was not a statement about what `search` returns. Unifying them
# is what the evaluator-parity child restored; the numbers above are the size of the gap it
# closed, kept here because they are the reason the binding is not optional.
OWNER_CHUNK_MULTIPLIER = 4
MAX_CHUNK_DEPTH = 10_000

# The fixed SELECT skeletons. Module constants rather than inline literals so the only thing
# the query builder concatenates at call time is a WHERE clause of bound `?` placeholders
# plus the `rank_order` string — which is what makes the `# nosec B608` below a statement of
# fact.
# The surface columns ride along on a LEFT JOIN by primary key (A-1): the index STORES each
# surface's attribution and locator, and the first version threw both away between the row
# and the response — a quoted post came back with `attribution: null` under the poster's
# name. LEFT, not INNER, because the retriever is also driven bare (`add` without a surface
# row) by the characterization fixture and the evaluation harness, and a chunk whose
# surface row is missing must still rank; it simply carries no attribution.
#
# The three aliases appear as LITERALS here and in `fetch_chunk`, rather than through one
# shared f-string: bandit reads any f-string that starts with `SELECT` as B608, and a
# suppression is a request to stop looking. `_hit` is the one reader of the alias names.
_SELECT_CHUNKS = (
    "SELECT chunks.*, "
    "surfaces.attribution_handle AS surface_attribution_handle, "
    "surfaces.attribution_name AS surface_attribution_name, "
    "surfaces.locator_json AS surface_locator_json, "
    "bm25(chunks_fts) AS score FROM chunks_fts "
    "JOIN chunks ON chunks.rowid = chunks_fts.rowid "
    "LEFT JOIN surfaces ON surfaces.surface_id = chunks.surface_id"
)
_COUNT_CHUNKS = "SELECT COUNT(*) FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid"
_SELECT_PROFILES = (
    "SELECT profiles.item_id, bm25(profiles_fts) AS score FROM profiles_fts "
    "JOIN profiles ON profiles.rowid = profiles_fts.rowid"
)

_CHUNK_RANK_ORDER = rank_order("chunks_fts", "chunks")
_PROFILE_RANK_ORDER = "ORDER BY bm25(profiles_fts) ASC, profiles.item_id ASC"


@dataclass(frozen=True)
class LexicalHit:
    """One ranked chunk, resolvable back to its surface and owner.

    Carries the owner and the surface, not just a score: spec §3.3 requires reverse
    resolution from a chunk to its surface and item, and a ranked id nobody can resolve is a
    number rather than evidence.

    `attribution` is the SURFACE's author — the quoted author of a quoted post, never the
    poster (spec §3.7 invariant 3) — and `surface_locator` is where the surface lives in the
    original data; the chunk's `char_start`/`char_end` narrow it to the match (A-1). Both are
    `None` for a chunk indexed without its surface row.
    """

    chunk_id: str
    surface_id: str
    owner_type: OwnerType
    owner_id: str
    surface_type: SurfaceType
    origin: str
    trust_class: str
    derived: bool
    chunk_index: int
    char_start: int
    char_end: int
    title: str | None
    url: str | None
    language: str | None
    fingerprint: str
    text: str
    excerpt: str
    score: float
    attribution: Author | None = None
    surface_locator: Locator | None = None


# An owner's IDENTITY. The PAIR, never the id alone: `surface_id` has carried
# `<owner_type>:<owner_id>:…` since spec §3.3 because the two namespaces OVERLAP —
# `Topic.slug` is `^[a-z0-9]+(?:-[a-z0-9]+)*$`, which admits an all-digit slug, and every
# real tweet id is all digits. One alias, so the counting and the narrowing cannot drift.
#
# The type half is `OwnerType`, the SAME closed set `KnowledgeChunk` is written from, so a
# caller cannot ask for `("items", id)` and get a silent empty answer, and so `_owner_clause`
# can bound its expression tree by a set size it can name rather than by a caller's length.
OwnerKey = tuple[OwnerType, str]


def owner_key(hit: LexicalHit) -> OwnerKey:
    """Which owner this hit belongs to — THE definition, for both readers (C1, round 10).

    `distinct_owners` counted the pair; the owner narrowing matched the id alone. Two answers
    to one question is the drift rule 5 exists to stop, and it survived nine rounds because
    no corpus in the suite held two owner types sharing an id. When one does, half a
    composite key is indistinguishable from all of it: a topic's synthesized prose is served
    under an item's name, and another item's article is served with that item's locator.
    """
    return (hit.owner_type, hit.owner_id)


def distinct_owners(hits: Sequence[LexicalHit]) -> int:
    """How many distinct owners a ranking prefix holds.

    Public because `search_owners` STOPS on it and `search_service` has to read the same
    number to know whether a window shorter than it asked for is the whole ranking or just
    a shallow one (B1). Two readings of «how deep did we get» is how the window and its
    consumer drift apart, and the drift is invisible until the exclusions bite (rule 5).
    """
    return len({owner_key(hit) for hit in hits})


@dataclass(frozen=True)
class ProfileHit:
    """One ranked ITEM, from the profile plane. Deliberately carries no text.

    Spec §5.1.A: the profile is *a retrieval representation, not a new source returned as a
    citation*. Two fields, and neither of them can hold a quotation.
    """

    item_id: str
    score: float


class LexicalIndex:
    """FTS5 over the persisted schema. Read or write, depending on how the connection opened.

    Takes a `sqlite3.Connection` rather than a path so the caller decides — and so `search`
    can be handed a `file:…?mode=ro` connection on which a write RAISES rather than silently
    repairing the index (spec §5.6). That is a property of the object, not a promise about
    the code.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    # -- writing -----------------------------------------------------------

    def add(
        self,
        chunks: Sequence[KnowledgeChunk],
        *,
        created_at: datetime | None = None,
        source: str | None = None,
    ) -> int:
        """Index a batch of chunks. Returns how many were stored.

        `created_at` and `source` are the item's, DENORMALISED onto every chunk of the batch
        so the date and source filters run before the `MATCH` with no join. They are
        arguments rather than fields of `KnowledgeChunk` because the chunk is part of the
        frozen contract and a retrieval-layout copy has no business on it — and because a
        batch is always one owner's chunks, so one value per call is the honest shape. A
        topic's chunks pass neither, which is what makes an item-scoped filter exclude them.

        A chunk already present (same `chunk_id`) is skipped rather than duplicated: the
        emitter can legitimately produce the same chunk twice across two calls, and a
        duplicate would let one body occupy two ranks.

        A blank body is skipped for the reason the emitters drop a blank surface — there is
        nothing to retrieve — and `index build` counts the skip so the omission is visible.
        """
        stored = 0
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO chunks (chunk_id, surface_id, owner_type, owner_id, "
                "surface_type, origin, trust_class, derived, chunk_index, char_start, "
                "char_end, text, title, url, language, fingerprint, created_at, source, "
                "primary_topic) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    chunk.chunk_id,
                    chunk.surface_id,
                    chunk.owner_type,
                    chunk.owner_id,
                    chunk.surface_type,
                    chunk.origin,
                    chunk.trust_class,
                    int(chunk.derived),
                    chunk.chunk_index,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.text,
                    chunk.title,
                    chunk.url,
                    chunk.language,
                    chunk.fingerprint,
                    _iso(created_at),
                    source,
                    chunk.topics[0] if chunk.topics else None,
                ),
            )
            if cursor.rowcount:
                # The FTS row carries the SAME rowid as its metadata row, which is what lets
                # the join be a plain equality — and is why `rowid INTEGER PRIMARY KEY` is
                # declared explicitly in `index_schema` (VACUUM renumbers an implicit one).
                self.connection.execute(
                    "INSERT INTO chunks_fts (rowid, text, title) VALUES (?,?,?)",
                    (cursor.lastrowid, chunk.text, chunk.title or ""),
                )
                stored += 1
        return stored

    def add_profile(self, item_id: str, text: str, fingerprint: str) -> bool:
        """Store one item profile on its own plane. Returns whether it was written."""
        if not text.strip():
            return False
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO profiles (item_id, profile_text, fingerprint) VALUES (?,?,?)",
            (item_id, text, fingerprint),
        )
        if not cursor.rowcount:
            return False
        self.connection.execute(
            "INSERT INTO profiles_fts (rowid, profile_text) VALUES (?,?)",
            (cursor.lastrowid, text),
        )
        return True

    def set_item_metadata(
        self,
        item_id: str,
        *,
        source: str = "bookmark",
        url: str = "",
        author_handle: str = "",
        author_name: str = "",
        created_at: datetime | None = None,
        captured_at: datetime | None = None,
        primary_topic: str | None = None,
        note_path: str | None = None,
        bookmark_folder: str | None = None,
        store_fingerprint: str = "",
        topics: tuple[str, ...] = (),
        content_kinds: tuple[str, ...] = (),
        surface_types: tuple[str, ...] = (),
    ) -> None:
        """Write the filterable metadata of one item, replacing whatever was there.

        The `surface_types` argument writes MINIMAL rows into `surfaces` — enough for the
        `has_surfaces` filter and no more. The full surface records come from `index_build`,
        which has the emitter's output; this keeps the retriever testable on its own without
        a second, divergent way of populating the table.
        """
        self.connection.execute(
            "INSERT OR REPLACE INTO items (item_id, source, url, author_handle, author_name, "
            "created_at, captured_at, primary_topic, note_path, bookmark_folder, "
            "store_fingerprint) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                item_id,
                source,
                url,
                author_handle,
                author_name,
                _iso(created_at),
                _iso(captured_at),
                primary_topic,
                note_path,
                bookmark_folder,
                store_fingerprint,
            ),
        )
        self.connection.execute("DELETE FROM item_topics WHERE item_id = ?", (item_id,))
        for position, slug in enumerate(topics):
            self.connection.execute(
                "INSERT OR REPLACE INTO item_topics (item_id, slug, is_primary) VALUES (?,?,?)",
                (item_id, slug, int(position == 0 and slug == (primary_topic or topics[0]))),
            )
        self.connection.execute("DELETE FROM item_content_kinds WHERE item_id = ?", (item_id,))
        for kind in content_kinds:
            self.connection.execute(
                "INSERT OR REPLACE INTO item_content_kinds (item_id, kind) VALUES (?,?)",
                (item_id, kind),
            )
        for surface_type in surface_types:
            self.connection.execute(
                "INSERT OR REPLACE INTO surfaces (surface_id, owner_type, owner_id, "
                "surface_type, origin, trust_class, derived, locator_json, fingerprint, "
                "char_length) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    f"item:{item_id}:{surface_type}:0",
                    "item",
                    item_id,
                    surface_type,
                    "source",
                    "primary_source",
                    0,
                    "{}",
                    "0" * 64,
                    0,
                ),
            )

    # -- reading -----------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int,
        *,
        filters: SearchFilters | None = None,
        surface_types: tuple[SurfaceType, ...] = (),
        owners: tuple[OwnerKey, ...] = (),
    ) -> tuple[LexicalHit, ...]:
        """The top `limit` chunks for `query`, best first, deterministic under ties.

        `surface_types` and `owners` are NOT among spec §7.2's eight — they are internal
        narrowing, kept off `SearchFilters` because that model is the FROZEN external
        contract and adding to it would be the incompatible change the freeze prevents.
        ONE consumer today, `search_service._verified_top`, which re-derives a served item's
        citations from its own plane before its topics'.

        `owners` takes `OwnerKey` PAIRS, and that is the whole of C1. It took bare ids, and
        an id is not an identifier here: a topic slug may be all digits and so is every tweet
        id, so `chunks.owner_id IN (…)` answered with whatever carried that id in EITHER
        namespace. Measured at `804b341`, an item with no `content` at all was served citing
        two chunks of another item's article, under its title and its locator; and a topic
        whose slug equalled an item's own id took the first two of that item's three
        citation slots, ahead of its own summary.

        An empty query is a validation error, not an empty result (spec §9.3): an empty
        result set claims something about the corpus, when the truth is that nothing was
        asked. A non-positive limit is the same mistake wearing a different hat.
        """
        expression = self._expression(query, limit)
        if expression is None:
            return ()
        clauses, params = self._where(expression, filters, surface_types, owners)
        sql = f"{_SELECT_CHUNKS} WHERE {' AND '.join(clauses)} {_CHUNK_RANK_ORDER} LIMIT ?"  # nosec B608
        rows = self._fetch(sql, (*params, limit))
        return tuple(_hit(row) for row in rows)

    def search_owners(
        self, query: str, owners: int, *, filters: SearchFilters | None = None
    ) -> tuple[tuple[LexicalHit, ...], bool]:
        """The ranking's prefix deep enough to hold `owners` DISTINCT owners, in rank order.

        Returns `(hits, depth_exhausted)`. `hits` is a prefix of the full ranking — the
        same rows `search` returns for the chunk limit reached — so a caller that groups
        by owner sees a list that is prefix-consistent across depths: a deeper window only
        appends. `depth_exhausted` is True when `MAX_CHUNK_DEPTH` was reached with fewer
        owners than asked, and the service declares it as a truncation it cannot page — it
        says so rather than guessing. The evaluation harness calls it too, for the same
        window; see the note on `OWNER_CHUNK_MULTIPLIER`.

        THE FIRST WINDOW IS CAPPED TOO, AND IT WAS NOT. `MAX_CHUNK_DEPTH` bounded only the
        DOUBLING path, so `owners * OWNER_CHUNK_MULTIPLIER` could open larger than the bound on
        its very first query: the limit that exists to cap the work applied only to small
        pages. Measured on 11,200 matching rows over 16 items — `--limit 500` read 2,008 rows
        and stopped at the cap, while `--limit 2763` read 11,056, i.e. PAST a bound of 10,000,
        and the two pages therefore answered from different amounts of the corpus. Capping
        here makes the bound mean one thing for every page size.

        WHAT IT COSTS DID CHANGE, and this docstring said it did not (round 10). Before the
        cap, an owner past the bound was reachable at a page size large enough to open a
        first window past it; after it, that owner is out of the chunk plane at EVERY page
        size. Measured on the same corpus: `--limit 2763` returned 16 results before and 15
        after, the sixteenth missing at 2763 and at 5000 alike. Consistency was the point and
        it was bought with a capability, which is a trade worth naming rather than a cost
        that stayed still.

        Nor is such an owner simply gone: with no chunk of its own inside the window it is
        usually re-admitted from the PROFILE plane, carrying zero citations and the derived
        `verify_with` branch. That is pre-existing behaviour and not a consequence of the
        cap, but "unreachable" was the wrong word for it. What the bound reaching its end
        means is DECLARED through `depth_exhausted`, and is the price of bounding work.
        """
        chunk_limit = min(max(owners * OWNER_CHUNK_MULTIPLIER, 1), MAX_CHUNK_DEPTH)
        while True:
            hits = self.search(query, chunk_limit, filters=filters)
            if distinct_owners(hits) >= owners or len(hits) < chunk_limit:
                return hits, False
            if chunk_limit >= MAX_CHUNK_DEPTH:
                return hits, True
            chunk_limit = min(chunk_limit * 2, MAX_CHUNK_DEPTH)

    def search_profiles(
        self, query: str, limit: int, *, filters: SearchFilters | None = None
    ) -> tuple[ProfileHit, ...]:
        """The top `limit` ITEMS for `query` on the profile plane (spec §5.1.A).

        The two planes are NOT fused here and their bm25 scores are not comparable: they are
        computed over different corpora, so a single sorted merge would invent a scale.
        Fusion is Plan 03's decision, with RRF and the golden set in front of it.

        AN ORIGIN FILTER YIELDS NOTHING ON THIS PLANE (G-1). `origins` is a property of the
        TEXT that matched — spec §7.2's `--origin vlm` asks for the figure *seen in an image*
        — and a profile is a string nobody wrote, composed from a post, a summary and three
        topic descriptions: it has no origin to satisfy. Until round 04 the seven item-scoped
        filters reached this plane and `origins` did not, so `search "Forecasting" --origin
        asr` on the real corpus answered 8 items, 6 of them without one ASR surface and 7
        without one citable match. Failing closed here is the same shape as a topic chunk
        under an author filter: what cannot satisfy the question is not a candidate for it.
        """
        expression = self._expression(query, limit)
        if expression is None:
            return ()
        if filters is not None and filters.origins:
            return ()
        clauses = ["profiles_fts MATCH ?"]
        params: list[object] = [expression]
        if filters is not None:
            item_clauses, item_params = _item_clauses(filters, "profiles.item_id")
            clauses += item_clauses
            params += item_params
        sql = f"{_SELECT_PROFILES} WHERE {' AND '.join(clauses)} {_PROFILE_RANK_ORDER} LIMIT ?"  # nosec B608
        rows = self._fetch(sql, (*params, limit))
        return tuple(ProfileHit(item_id=row["item_id"], score=float(row["score"])) for row in rows)

    def explain(self, query: str, filters: SearchFilters | None = None) -> tuple[str, ...]:
        """`EXPLAIN QUERY PLAN` for the filtered search, as sqlite's own `detail` strings.

        Exposed so a test can assert that the backend USES an index for the filtered column
        rather than scanning the corpus (m3). Asserting the SQL string would only prove that
        we wrote that string.
        """
        expression = match_expression(query) or ""
        clauses, params = self._where(expression, filters, (), ())
        sql = f"{_SELECT_CHUNKS} WHERE {' AND '.join(clauses)} {_CHUNK_RANK_ORDER}"  # nosec B608
        return tuple(
            row["detail"] for row in self.connection.execute(f"EXPLAIN QUERY PLAN {sql}", params)
        )

    def scored_row_count(self, query: str, filters: SearchFilters | None = None) -> int:
        """How many rows the filtered query hands the scorer — the second half of m3.

        A filter that runs shows up as FEWER rows reaching bm25. A filter applied afterwards
        would leave this number unchanged and only shorten the output.
        """
        expression = match_expression(query)
        if expression is None:
            return 0
        clauses, params = self._where(expression, filters, (), ())
        sql = f"{_COUNT_CHUNKS} WHERE {' AND '.join(clauses)}"  # nosec B608
        row = self.connection.execute(sql, params).fetchone()
        return int(row[0])

    def fetch_chunk(self, chunk_id: str) -> LexicalHit | None:
        """One chunk by id, with everything needed to verify its fingerprint."""
        row = self.connection.execute(
            "SELECT chunks.*, "
            "surfaces.attribution_handle AS surface_attribution_handle, "
            "surfaces.attribution_name AS surface_attribution_name, "
            "surfaces.locator_json AS surface_locator_json, "
            "0.0 AS score FROM chunks "
            "LEFT JOIN surfaces ON surfaces.surface_id = chunks.surface_id "
            "WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return _hit(row) if row is not None else None

    def __len__(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    # -- internals ---------------------------------------------------------

    def _expression(self, query: str, limit: int) -> str | None:
        if not query.strip():
            raise ValueError("la consulta está vacía")
        if limit <= 0:
            raise ValueError(f"limit debe ser >= 1, recibido {limit}")
        return match_expression(query)

    def _where(
        self,
        expression: str,
        filters: SearchFilters | None,
        surface_types: tuple[SurfaceType, ...],
        owners: tuple[OwnerKey, ...],
    ) -> tuple[list[str], list[object]]:
        """The WHERE clauses and their bound parameters — never an interpolated value."""
        clauses = ["chunks_fts MATCH ?"]
        params: list[object] = [expression]
        if filters is not None:
            chunk_clauses, chunk_params = _chunk_clauses(filters)
            clauses += chunk_clauses
            params += chunk_params
            item_clauses, item_params = _item_clauses(filters, "chunks.owner_id")
            clauses += item_clauses
            params += item_params
            topic_clause, topic_params = _topic_clause(filters)
            if topic_clause:
                clauses.append(topic_clause)
                params += topic_params
        if surface_types:
            clauses.append(f"chunks.surface_type IN ({_placeholders(len(surface_types))})")
            params += list(surface_types)
        if owners:
            owner_clause, owner_params = _owner_clause(owners)
            clauses.append(owner_clause)
            params += owner_params
        return clauses, params

    def _fetch(self, sql: str, params: tuple[object, ...]) -> list[sqlite3.Row]:
        try:
            return list(self.connection.execute(sql, params))
        except sqlite3.OperationalError as error:
            # A MATCH expression FTS5's parser rejects degrades to "no results", HERE AND
            # ONLY HERE: the query was understood as data and simply held nothing FTS5
            # could parse. The first version caught every `OperationalError` except a
            # read-only one — chosen by substring, rule 9 in miniature — so `no such table`,
            # `database is locked` and `disk I/O error` all came back as "the corpus holds
            # nothing" (C-2). The parser's own error family is the closed set the branch was
            # written for, and it is matched positively. Everything else is a base this code
            # cannot read and is CONVERTED like the rest of the family below (U-4): the C-2
            # fix re-raised it raw, and `no such column: surfaces.attribution_name` reached
            # the operator as a traceback naming no command.
            if _is_fts_parse_error(error):
                return []
            raise IndexIncompatibleError(
                f"La base del índice no se puede consultar ({error}). {REBUILD_ADVICE}"
            ) from error
        except sqlite3.DatabaseError as error:
            # The REST of the family — `fts5: corruption found reading blob…`, `file is not
            # a database`, `database disk image is malformed` — is what a corrupt or
            # amputated base raises at QUERY time, past the page-1 and schema checks of the
            # open door (G-4). It is neither an `IndexError_` nor an `OSError`, so it
            # reached the operator as a 68-line traceback naming no command; Plan 02 §11
            # tabulates it as *base corrupta -> error accionable con `index build --force`*.
            raise IndexIncompatibleError(
                f"La base del índice no se puede consultar ({error}). {REBUILD_ADVICE}"
            ) from error


# What FTS5's expression parser says when it cannot parse — measured on sqlite 3.51.2 with
# `NEAR(`, `"unterminated` and `*`. A closed, positive list: an error that is not one of these
# is not a parse error and must not be read as an empty corpus.
_FTS_PARSE_ERRORS: tuple[str, ...] = (
    "fts5: syntax error",
    "unterminated string",
    "unknown special query",
)


def _is_fts_parse_error(error: sqlite3.OperationalError) -> bool:
    message = str(error)
    return any(message.startswith(prefix) for prefix in _FTS_PARSE_ERRORS)


def _chunk_clauses(filters: SearchFilters) -> tuple[list[str], list[object]]:
    """Filters that live on the DENORMALISED columns of `chunks` — no join at all.

    `created_at`, `source` and `origin` are copied onto every chunk precisely so these run
    before the `MATCH` without a per-query join. A topic-owned chunk has NULL in the first
    two, so a date or source filter excludes it by construction — which is the right answer,
    since a topic note has no date and no source of its own.
    """
    clauses: list[str] = []
    params: list[object] = []
    if filters.created_from is not None:
        clauses.append("chunks.created_at >= ?")
        params.append(filters.created_from.isoformat())
    if filters.created_to is not None:
        clauses.append("chunks.created_at <= ?")
        params.append(filters.created_to.isoformat())
    if filters.source is not None:
        clauses.append("chunks.source = ?")
        params.append(filters.source)
    if filters.origins:
        clauses.append(f"chunks.origin IN ({_placeholders(len(filters.origins))})")
        params += list(filters.origins)
    return clauses, params


def _item_clauses(filters: SearchFilters, owner_column: str) -> tuple[list[str], list[object]]:
    """Filters that need the item tables, as `EXISTS` sub-queries on the owner id.

    `owner_column` is a CALLER-CHOSEN identifier from a closed set of two (`chunks.owner_id`,
    `profiles.item_id`), never a user string — the two planes ask the same questions of the
    same tables and a second copy of these clauses would be the divergence rule 5 is about.

    What is NOT here, and where it is: `origins` is a CHUNK property (`_chunk_clauses`), and
    on the profile plane it is answered by `search_profiles` returning nothing at all, because
    a profile has no origin (G-1). The totality test over both planes
    (`test_every_declared_filter_is_pushed_on_the_profile_plane_too`) is what keeps the two
    planes agreeing on all eight, this docstring only says where each one lives.

    AN ITEM-SCOPED FILTER REQUIRES AN ITEM-OWNED ROW, ONCE, HERE (M2, round 12). Every clause
    below joins an item table to `{owner_column}`, and on the chunk plane that column holds
    BOTH namespaces: a topic whose slug equals an item's id inherited that item's author, its
    content kinds and its surfaces, so `--author vgonpa` returned an item written by someone
    else and `--has-surfaces external_article` returned an item with no `content` at all.

    This is the module's own documented intent, not a new rule — *ITEM-SCOPED FILTERS FAIL
    CLOSED ON TOPIC-OWNED CHUNKS*. Date and source already did, and only by luck: their columns
    are NULL on a topic row, so the comparison fails. The three that route through `EXISTS`
    joins had no such accident behind them.

    ONE guard for all three rather than a predicate bolted onto each: they combine with `AND`,
    so one clause is equivalent, and the sentence «an item filter means the owner is an item»
    then exists in one place instead of three that can drift (rule 5). It is added only when an
    item-scoped filter is actually present — an unfiltered query still sees both planes — and
    only on the chunk plane, because `profiles` holds nothing but items and has no such column.
    """
    clauses: list[str] = []
    params: list[object] = []
    if owner_column == "chunks.owner_id" and (
        filters.author is not None or filters.content_kinds or filters.has_surfaces
    ):
        clauses.append("chunks.owner_type = ?")
        params.append("item")
    if filters.author is not None:
        clauses.append(
            f"EXISTS (SELECT 1 FROM items WHERE items.item_id = {owner_column} "  # nosec B608
            "AND items.author_handle = ?)"
        )
        params.append(filters.author)
    if filters.content_kinds:
        clauses.append(
            "EXISTS (SELECT 1 FROM item_content_kinds WHERE item_content_kinds.item_id = "  # nosec B608
            f"{owner_column} AND item_content_kinds.kind IN "
            f"({_placeholders(len(filters.content_kinds))}))"
        )
        params += list(filters.content_kinds)
    if filters.has_surfaces:
        clauses.append(
            f"EXISTS (SELECT 1 FROM surfaces WHERE surfaces.owner_id = {owner_column} "  # nosec B608
            f"AND surfaces.owner_type = 'item' AND surfaces.surface_type IN "
            f"({_placeholders(len(filters.has_surfaces))}))"
        )
        params += list(filters.has_surfaces)
    if owner_column != "chunks.owner_id":
        # The profile plane has no chunk row to read `source`/`created_at` off, so the two
        # denormalised filters are answered from `items` instead. Same question, same table
        # family, and the chunk plane keeps its faster path.
        if filters.source is not None:
            clauses.append(
                f"EXISTS (SELECT 1 FROM items WHERE items.item_id = {owner_column} "  # nosec B608
                "AND items.source = ?)"
            )
            params.append(filters.source)
        if filters.created_from is not None:
            clauses.append(
                f"EXISTS (SELECT 1 FROM items WHERE items.item_id = {owner_column} "  # nosec B608
                "AND items.created_at >= ?)"
            )
            params.append(filters.created_from.isoformat())
        if filters.created_to is not None:
            clauses.append(
                f"EXISTS (SELECT 1 FROM items WHERE items.item_id = {owner_column} "  # nosec B608
                "AND items.created_at <= ?)"
            )
            params.append(filters.created_to.isoformat())
        if filters.topics:
            clauses.append(
                f"EXISTS (SELECT 1 FROM item_topics WHERE item_topics.item_id = {owner_column} "  # nosec B608
                f"AND item_topics.slug IN ({_placeholders(len(filters.topics))}))"
            )
            params += list(filters.topics)
    return clauses, params


def _topic_clause(filters: SearchFilters) -> tuple[str, list[object]]:
    """The topic filter on the CHUNK plane, with its one exception.

    An item chunk matches through `item_topics`. A TOPIC-owned chunk matches against its own
    slug: without that branch, filtering by a topic would exclude exactly the surfaces that
    ARE the topic — a `topic_note` about `ai-policy` would vanish from `--topic ai-policy`,
    which reads as "the topic has no notes" rather than as a shape of the filter.

    EACH ARM NAMES ITS OWN OWNER TYPE (M2, round 12). Arm 2 always did. Arm 1 asked only
    whether the owner id belonged to an item in the topic, and asked it of every row — so a
    chunk owned by topic `7001` matched `--topic kestrel` whenever ITEM `7001` was in
    `kestrel`, which is one topic answering for another through a name it merely shares.

    The fix is arm 1, never the clause: closing the whole thing to topics would pass a
    regression about the collision and silently delete the feature arm 2 exists for, which is
    why both arms are asserted together.
    """
    if not filters.topics:
        return "", []
    placeholders = _placeholders(len(filters.topics))
    clause = (
        "((chunks.owner_type = 'item' AND EXISTS (SELECT 1 FROM item_topics "  # nosec B608
        f"WHERE item_topics.item_id = chunks.owner_id AND item_topics.slug IN ({placeholders}))) "
        f"OR (chunks.owner_type = 'topic' AND chunks.owner_id IN ({placeholders})))"
    )
    return clause, [*filters.topics, *filters.topics]


def _placeholders(count: int) -> str:
    """`?,?,?` for a variadic `IN` clause — derived from a COUNT, never from a caller string.

    The one SQL fragment this module builds dynamically, and it is built from an integer, so
    it cannot carry a caller's text into the statement no matter what was passed. Extracted
    as a function so that claim is structurally true and checkable in one line, rather than a
    promise attached to an f-string.
    """
    return ",".join("?" * count)


def _owner_clause(owners: tuple[OwnerKey, ...]) -> tuple[str, list[object]]:
    """The owner narrowing: one `IN` PER TYPE, not one equality per owner (C1, then M1).

    Both columns are constrained, because an owner is the pair — that is C1, and an id-only
    clause served one owner's text under another owner's name.

    ONE DISJUNCT PER TYPE, and the first version wrote one per OWNER. A left-deep `OR` tree of
    equalities meets `SQLITE_MAX_EXPR_DEPTH` (1000) long before anything else: measured on
    sqlite 3.50.4, 993 owners answered and 995 raised, where the `IN (…)` it replaced was a
    single node and answered past 16,384. Grouping by type bounds the tree at the number of
    distinct owner TYPES — two, and `OwnerType` makes that a closed set — so the width of the
    request is carried by the `IN` list, whose ceiling is the variable budget that bounded the
    old clause too. Correctness and the ceiling were never actually in tension; the first
    shape just paid for one with the other.

    Ids are deduplicated per type, preserving first-seen order: `IN` does not care, the
    variable budget does, and a caller that asks for the same owner twice should not lose
    headroom for it.

    WHAT REMAINS, declared rather than discovered. Past roughly 32k owners the statement runs
    out of bound parameters, and `_fetch` converts that into `IndexIncompatibleError` — telling
    an operator to rebuild an index that is perfectly healthy when the truth is that the
    REQUEST was too wide (rule 9, the instrument naming the wrong surface). That is unchanged
    from the id-only clause this replaces and is not repaired here; it belongs with the filter
    joins in `_item_clauses`, which have their own half-key defect and are umbrella code. It is
    far outside anything the callers reach: measured on the live corpus, the top-up phase asks
    for one item's topics, and that is at most 4 against a 45-topic vocabulary.

    The fragment is still built from COUNTS alone (`_placeholders`), so no caller text reaches
    the statement; the types and ids are bound values.
    """
    grouped: dict[OwnerType, list[str]] = {}
    for owner_type, owner_id in owners:
        grouped.setdefault(owner_type, []).append(owner_id)
    parts: list[str] = []
    params: list[object] = []
    for owner_type, ids in grouped.items():
        unique = list(dict.fromkeys(ids))
        parts.append(
            f"(chunks.owner_type = ? AND chunks.owner_id IN ({_placeholders(len(unique))}))"
        )
        params.append(owner_type)
        params += unique
    return f"({' OR '.join(parts)})", params


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _surface_locator(raw: str | None) -> Locator | None:
    """The surface's locator, or None when the row carries none a `Locator` can hold.

    `set_item_metadata` writes `{}` for the minimal surface rows the retriever's own tests
    use, and a chunk indexed bare has no row at all; neither is a locator, and inventing one
    would be the fabrication A-1 removed.
    """
    if not raw:
        return None
    try:
        return Locator.model_validate_json(raw)
    except ValidationError:
        return None


def _attribution(row: sqlite3.Row) -> Author | None:
    handle = row["surface_attribution_handle"]
    if handle is None:
        return None
    return Author(handle=handle, name=row["surface_attribution_name"] or "")


def _hit(row: sqlite3.Row) -> LexicalHit:
    return LexicalHit(
        chunk_id=row["chunk_id"],
        surface_id=row["surface_id"],
        owner_type=row["owner_type"],
        owner_id=row["owner_id"],
        surface_type=row["surface_type"],
        origin=row["origin"],
        trust_class=row["trust_class"],
        derived=bool(row["derived"]),
        chunk_index=row["chunk_index"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        title=row["title"],
        url=row["url"],
        language=row["language"],
        fingerprint=row["fingerprint"],
        text=row["text"],
        excerpt=row["text"][:EXCERPT_CHARS],
        score=float(row["score"]),
        attribution=_attribution(row),
        surface_locator=_surface_locator(row["surface_locator_json"]),
    )
