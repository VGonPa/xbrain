# Atlas Knowledge Graph

Atlas Knowledge Graph is the topic-first exploration surface inside the Atlas
workspace. It is generated from `items.json` enrichment data through
`compute_dashboard_data()` and remains a derived view: Markdown notes, topic
pages, and the dashboard never become a second source of truth.

## Data Contract

The dashboard payload includes `graph`:

- `topics`: topic nodes with counts, related topic count, recent activity,
  confidence buckets, content type buckets, overview, and optional topic note.
- `items`: compact document nodes with title label, 180-character summary,
  acquisition date, publication date, primary/all topics, content type,
  confidence, author, domain, note path, original URL, review signal, and
  suggested topics.
- `topic_edges`: co-occurrence edges with both shared-document volume and
  normalized affinity.
- `membership_edges`: topic-to-item edges, marked `primary` or `secondary`.
- `emerging_topics`: suggested topics that have not been promoted to vocabulary.
- `facets`: content, confidence, and month buckets.
- `insights`: simple explainable signals such as most connected, isolated,
  needs review, recently active, and fastest growing.
- `version`: deterministic schema/taxonomy hash. Adding documents no longer
  discards a carefully arranged local layout.

## Atlas Views

Atlas exposes three internal views:

- `Map`: ECharts graph surface with topic nodes shown first. Documents appear
  only after search, explicit document toggle, or topic expansion.
- `Timeline`: time scatter plot grouped by primary topic, sharing the same
  search, date, content type, and confidence filters.
- `Documents`: accessible topic-and-document fallback for keyboard, mobile, and
  dense-corpus scanning. It displays the actual filtered notes, not only topics.

## Shared Selection and Controls

The Library filter bar is shared by Library, Atlas, and Analytics. Topic,
content, confidence, author, domain, full-text query, date basis, and date range
therefore describe the same selection in every view. Filter state is reflected
in URL parameters and persisted locally.

- Search covers topic labels and slugs, item labels, summaries, authors,
  domains, item ids, and topic lists.
- Date presets are `All time`, `12 months`, `6 months`, `90 days`, `30 days`,
  and `Custom`.
- Content filters are `All`, `Articles`, `Videos`, `Failed`, and `Posts`.
- Confidence filters are `All`, `High`, `Medium`, `Low`, `Unknown`,
  `Needs review`, and `Misc`.
- `Documents` toggles document nodes globally.
- `Affinity / Volume` switches edge width between normalized relatedness and
  raw shared-document count.
- Double-clicking a topic expands or collapses its documents.
- Double-clicking a document opens its note via `/notes?path=...` when served
  and `obsidian://open?path=...` when opened locally.
- `Emerging` toggles suggested topic nodes.
- `Pause` switches the map between force and fixed layout modes.
- `Reset layout` clears local graph preferences.

## Local Layout Persistence

Visual layout state is stored in browser `localStorage` under:

```text
xbrain.graph.layout.v1
```

The stored payload records `graph.version` for compatibility and can include:

- pinned node positions,
- zoom and center,
- date/content/confidence filters,
- document and emerging topic toggles,
- the focused topic and hop depth,
- locally hidden topics.

This does not modify `items.json`, generated Markdown, topic pages, or the
Obsidian vault. `Reset layout` clears hidden topics and focus state. Layouts
survive ordinary ingestion and change version only when the graph schema or
taxonomy changes incompatibly.

Topic drawers expose graph-only controls for local exploration:

- `Show documents` expands representative notes around the topic.
- `Focus graph` narrows the map to one or two topic-relation hops.
- `Hide topic` removes the topic locally until reset or a new graph version.
- `Pin` stores the current node position locally.

The search box filters as the user types. Pressing Enter selects the first
matching topic or document, applies a focused graph context, and opens the
matching drawer.

## Served Actions

When the dashboard is served with `xbrain serve-dashboard`, document drawers and
the note viewer expose the same topic-review decisions with consequence-based
labels:

- `Confirm current topics` keeps the assignment, sets confidence to high, and
  clears pending suggestions.
- `Move to Misc` replaces all assigned topics with `misc` and completes the
  review. This is the explicit replacement for the ambiguous `Reject` label.
- `Reclassify automatically` runs classification again against the current
  controlled vocabulary.
- `Try selected suggestions` uses checked suggestions as hints without changing
  the shared vocabulary.
- `Add to vocabulary & reclassify` adds checked suggestions to the global
  vocabulary and then reclassifies only the current document. It requires an
  additional confirmation.
- Closing the topic-review modal makes no changes.

These decisions use `POST /api/topic-action`. The wider note-viewer modal groups
confirmation and AI actions separately, explains each consequence, keeps status
feedback visible, and traps keyboard focus while open.

Document drawers also reuse the existing local endpoints:

- `POST /api/reprocess-note` for single-note reprocessing.
- `GET /api/status` to wait for the background operation and reload the
  regenerated dashboard.

Served POST actions require the dashboard's same-origin custom action header;
direct browser forms and cross-site requests are rejected.

When the dashboard is opened as a local HTML file, these write controls are
disabled and the drawer states that topic editing requires the served dashboard.

## Obsidian Canvas Export

`xbrain generate` also writes:

```text
maps/xbrain-map.generated.canvas
```

This file is derived from the same `graph` payload as the dashboard. It includes
topic nodes, topic co-occurrence edges, and a small set of representative item
note nodes per topic. Topic nodes link to `topics/<slug>.md` when that topic
page exists; otherwise they are emitted as text nodes.

For manual Canvas editing, duplicate the generated file in Obsidian as:

```text
maps/xbrain-map.canvas
```

XBrain never writes that editable copy. Regeneration updates only the
`.generated.canvas` file.

## Validation

Before merging graph changes, run:

```bash
uv run pytest tests/test_dashboard.py
uv run pytest
```

For visual checks, generate a dashboard and verify:

- `#atlas` renders a nonblank `#c-graph` canvas on desktop and mobile.
- `Map`, `Timeline`, and `Documents` switch by click and arrow/Home/End keys.
- Shared filters and acquisition/publication date choice update all three views.
- Document drawers open notes locally and when served.
- Served topic/reprocess actions start jobs and reload after completion.
- `maps/xbrain-map.generated.canvas` is regenerated while
  `maps/xbrain-map.canvas` remains untouched.
