# Jev topic assessment (`xbrain jev`)

XBrain assigns topics to a post in the `enrich` stage: one LLM reads the post and picks
slugs out of `vocab.yaml`. Nothing checks that work. `xbrain jev` asks a **different model
family the same question** and puts the two answers side by side.

Jev (TypeSafe AI) is a typed-question model: you give it a `state` and a map of questions,
and it answers each one with a calibrated probability rather than with prose. For every
item XBrain asks one yes/no question per vocabulary topic ("is this post about
*ai-coding*?") plus one pick-one question for the primary topic.

**It is a second opinion, and it never changes the wiki.** The answers land in a side-car,
`data/jev/topics.json`. Nothing under `xbrain jev` opens `items.json` for writing, no
verdict is derived, no topic assignment moves, and no note is re-rendered. You can run it,
read it, and throw it away without touching the corpus.

- [Setup](#setup)
- [Daily use](#daily-use)
- [`xbrain jev topics` — what a run does](#xbrain-jev-topics--what-a-run-does)
- [`xbrain jev report` — reading the comparison](#xbrain-jev-report--reading-the-comparison)
- [`xbrain jev dashboard` — reading the page](#xbrain-jev-dashboard--reading-the-page)
- [Staleness: when an assessment stops counting](#staleness-when-an-assessment-stops-counting)
- [Where the files live, and what protects them](#where-the-files-live-and-what-protects-them)
- [Vendor facts, with their dates](#vendor-facts-with-their-dates)
- [Troubleshooting](#troubleshooting)

---

## Setup

**Before anything else:** `xbrain jev` compares Jev against `enrich`, so it needs a
vocabulary and an enriched corpus to compare against. Run the normal pipeline
(`extract → fetch → vocab → enrich`) first — if XBrain is not set up at all, start with
[the tutorial](tutorial.md). Provisioning a paid credential before that is wasted: nothing
here can run until the corpus is enriched.

With that in place:

1. Create an API key at <https://console.typesafe.ai/>.
2. Put it where XBrain looks — the environment first, then `<repo>/.env`:

   ```bash
   cp .env.example .env     # then paste the key after TYPESAFE_API_KEY=
   ```

   `.env` is gitignored. Only `TYPESAFE_API_KEY` is read from it; a line may carry an
   `export` prefix, single or double quotes and a trailing `#` comment, and the **last**
   assignment wins, as `source` would. An `export TYPESAFE_API_KEY=…` in your shell takes
   precedence over the file. A blank value counts as no key, so a `.env.example` copied and
   left unfilled fails as "no key" instead of as an empty string rejected by the API later.

3. Optionally tune `[jev]` in `config.toml`. Every key is optional; these are the defaults,
   and an unknown key under `[jev]` is refused at load time rather than ignored:

   ```toml
   [jev]
   model = "jev-latest"        # a moving alias; each assessment records the model that answered
   threshold = 0.85            # "backed by Jev" = probability >= threshold
   fallback_option = "otro"    # the escape option of the primary question
   concurrency = 8             # requests in flight
   state_char_limit = 100000   # evidence is cut here; assessments record the pre-cut length
   ```

Every `[jev]` key is also documented inline in
[`config.toml.example`](../config.toml.example), next to the defaults.

## Daily use

```bash
uv run xbrain jev topics --dry-run          # how many items are pending; calls nothing, needs no key
uv run xbrain jev topics --limit 20         # a small paid smoke run
uv run xbrain jev topics                    # the whole backlog
uv run xbrain jev report                    # data/jev/topics-report.{json,md} at [jev].threshold
uv run xbrain jev report --threshold 0.95   # the same side-car, read more strictly
uv run xbrain jev dashboard                 # <output_dir>/jev.html — open the printed URI
```

`jev topics` is the only command that spends money. `report` and `dashboard` re-read the
side-car it already paid for: no API call, no key needed, no cost, any number of times.

The full option list:

| Command | Options |
|---|---|
| `xbrain jev topics` | `--id TEXT` (repeatable — only these items) · `--limit INTEGER` · `--force` · `--dry-run` |
| `xbrain jev report` | `--threshold FLOAT` (default `[jev].threshold`) |
| `xbrain jev dashboard` | `--threshold FLOAT` (default `[jev].threshold`) |

Exit codes: **0** normal · **1** operator error (no key, a refusal, every item failed) ·
**130** interrupted with Ctrl-C.

## `xbrain jev topics` — what a run does

### It tells you what it selected, before it spends anything

Every run opens with one line that accounts for every item it considered, not just the part
it is about to ask:

```
20 items por evaluar · 1169 vigentes · 8 sin evidencia · 1412 fuera del límite (1169 evaluaciones guardadas)
```

Four of the five counts partition the **candidate set** — `20 + 1169 + 8 + 1412 = 2609` here,
which is the whole corpus because no `--id` was given — so an empty selection caused by a
regression in the evidence layer can never look like a clean "everything is up to date".
With `--id a --id b` the candidate set is those two items and the line sums to 2.

- **por evaluar** — items this run will ask about.
- **vigentes** — items whose stored assessment still describes today's question (skipped).
- **sin evidencia** — items with nothing to send. `evidence_surfaces` found no text.
- **forzados** *(a sub-count of `por evaluar`, not a fifth segment)* — appears only under
  `--force`: selected items whose assessment was *still current* and is being re-asked
  anyway. This is the segment that says "you are about to re-pay for work you already had",
  and it is why a `--force` line can look like it sums to more than the corpus.
- **fuera del límite** — evaluable items `--limit` left for a later run. Without it a
  nightly `--limit 200` cannot tell you whether the backlog is draining or growing.

`items por evaluar` and the parenthetical side-car total are always printed; the other
segments appear only when they are non-zero, so the common line stays short.

`--dry-run` stops there and adds one line — including whether a key is configured, so a
green dry-run is not followed by a real run that dies on the first thing it checks:

```
--dry-run: no se llama a Jev · clave TYPESAFE_API_KEY: configurada
```

`--dry-run` and an empty backlog never build a client, so neither needs a key. The key is
checked **before** the SDK is imported, and the SDK is imported inside the command rather
than at module top: `xbrain --help` never loads the vendor's HTTP stack.

### It asks once per item, in parallel

One call carries every question about one post: one yes/no (`topic__<slug>`) per vocabulary
topic, in slug order, plus one pick-one (`primary`) over every slug with the fallback
offered last. `[jev].concurrency` calls are in flight at a time. Progress prints every 50
items and at the end:

```
  50/2609
```

### It never loses what it paid for

- **A per-item failure is recorded, never dropped.** The run continues; the first ten
  failures are echoed on stderr as `FALLO <id>: <motivo>`, the rest as `… y N fallos más`.
  Re-running asks only the ones still pending.
- **A run where every item failed raises** instead of reporting an empty success — a dead
  key or a dead API is an error.
- **Records are flushed to disk every 25**, and again at the end. The dict in memory is not
  durability: a SIGTERM or an unexpected exception would take everything in it.
- **Ctrl-C saves what was banked.** Queued calls are cancelled deliberately (draining the
  queue would pay the full bill the interrupt was meant to stop), the banked records are
  written, and the command exits **130**:

  ```
  Interrumpido: 43 evaluaciones nuevas guardadas (1212 en total) en data/jev/topics.json
    258000 tokens de entrada (~0.0108 $)
  ```

  Under `--force` the noun changes — `Interrumpido: 43 evaluaciones re-evaluadas
  guardadas (1212 en total) …` — because those records are re-bills of work you already
  had, not new work. It is the run most likely to be interrupted, for the same reason.

  **43 is this run's new records; 1212 is the file total.** An interrupt that rescued
  nothing writes nothing at all (`Interrumpido: nada nuevo que guardar`) — saving there
  would write the unchanged map over the side-car, which on a first run is `{}`.
- **A save failure names the path and the paid count**, because `[Errno 28] No space left
  on device` is true and useless on its own.

### It closes with the bill

```
2583 evaluadas · 18 fallidas · 15498000 tokens de entrada (~0.6509 $) · modelo jev-1.13.0 → data/jev/topics.json
```

The cost sentence is formatted with a decimal point and no digit grouping — the page's
own numbers use es-ES formatting, the bill does not, because it is one string produced
in exactly one place and printed identically by
`jev topics`, `jev report`, `topics-report.md` and the dashboard, so a recap can never quote a
different figure from the bill it recaps. Two markers exist because a bare `~0,0000 $`
cannot say which zero it is:

- `(+K sin recuento)` — K records whose provider reported no token usage. They contribute
  nothing they cannot prove, so without this a fully paid run reports itself as free.
- `· proveedor sin tarifa: X` — **inside** the cost parentheses, after the figure
  (`N tokens de entrada (~X $ · proveedor sin tarifa: a, b)`): a provider absent from the
  price table. It contributes `0.0` rather than borrowing another vendor's rate, and is
  **named** rather than counted. `proveedores sin tarifa` for more than one.

The figure is an **estimate**, not an invoice: the rate is a list price for a concrete
model version while `[jev].model` defaults to a moving alias. See [Vendor facts](#vendor-facts-with-their-dates).

### What a pass actually costs

Measured on this repo's corpus on 2026-09-22 — 2,609 items, 45 topics, `state_char_limit =
100000`:

| | chars |
|---|---|
| Question set, per call (constant) | 20,305 |
| Evidence per item — median | 788 |
| Evidence per item — mean / p95 / max | 2,476 / 4,944 / 100,054 |
| Items cut at the limit | 7 of 2,609 |
| **One full pass** | **59.4 M** (≈ 15–17 M input tokens, ≈ 6k per item) |

At `0.042 $/MTok` that is roughly **0.63–0.71 $** for the whole corpus, in a few minutes at
8 concurrent requests. The token figure is a character-count conversion; the authoritative
number is the one the run itself reports from the provider's usage.

The sample outputs above are built on that ≈ 6k-tokens-per-item figure, so they can be
re-derived rather than taken on trust: 2,583 assessed + 18 failed + 8 without evidence =
2,609, and 2,583 × ≈ 6k ≈ 15.5 M tokens ≈ 0.65 $.

**The questions are 89% of that bill, not the evidence.** They are constant per call and
scale with `[vocab].target_count`, so the cost lever is the size of the vocabulary —
lowering `state_char_limit` barely moves it, and would cost evidence.

## `xbrain jev report` — reading the comparison

Writes two files under `<data_dir>/jev/` (default `data/jev/`), overwritten on every run:

- `topics-report.md` — what a person reads. Headline numbers, then the tables. `Por topic`
  carries **every** vocabulary row, worst-backed first; the four queue tables (doubtful,
  missing candidates, unjudged, primary mismatches) are **deliberately cut** at 20 rows
  each — the primary mismatches at 20 **per reason**, so that section can carry up to 100
  rows across its five sub-tables, as its own heading says (`top 20 por motivo`). Every cut
  table announces the cut (`_… y N filas más (el JSON las lleva todas)._`, italicised in
  the raw file).
- `topics-report.json` — what a program reads: `{"summary": {…}, "items": [{…}]}`, one
  record per compared item, no post text (join on `item_id`).

Both are written atomically, and the JSON first — a program comparing `generated_at` can
see a mismatched pair, a person reading a stale markdown cannot.

stdout is one line, the same one `jev dashboard` prints:

```
Umbral 0.85 · 0 caducadas · items comparados 2565/2583 · enrich respaldado 78.4 % · Jev respaldado 61.2 % · dudosas 1204 · sin juzgar 0 · candidatas 3310 · primario coincide 64.1 % · 15498000 tokens de entrada (~0.6509 $)
→ data/jev/topics-report.md
→ data/jev/topics-report.json
```

### `items comparados 2565/2583` — two different populations

The second number is how many stored assessments are **current**. The first is how many of
those had something to compare against: `compare_item` returns nothing for an item with no
`Enrichment`, so it is counted as assessed and not as compared.

The gap is therefore the population **Jev has an opinion about that `enrich` has not
enriched** — normally items extracted since the last `xbrain enrich` run. It is not an
error and nothing is lost: those items have no JSON record in the report and contribute to
no bucket. Run `xbrain enrich` and the gap closes on the next report.

Collapsing the two into one number would hide that population entirely, which is why the
line prints both. A gap that keeps growing across runs means `enrich` is falling behind
`extract`; a gap that appears suddenly usually means a `vocab --regenerate` cleared the
enrichments.

### The signals

| Signal | Meaning |
|---|---|
| **dudosa** | a topic `enrich` assigned that Jev scored **below** the threshold |
| **candidata que falta** | a topic Jev scored **at or above** the threshold that `enrich` did not assign |
| **asignación de Jev** | every topic at or above the threshold — Jev's own proposal, strongest first |
| **respaldada (enrich → Jev)** | `assigned_backed / assigned_pairs`: how much of `enrich`'s work Jev backs |
| **respaldada (Jev → enrich)** | `jev_backed / jev_pairs`: how much of Jev's own proposal `enrich` already has |
| **sin juzgar** | an assigned topic Jev was never **asked** about — a slug that left `vocab.yaml` after the item was enriched |
| **primario coincide** | Jev's pick-one answer equals `enriched.primary_topic` |
| **rango del primario** | the 1-based position of `enrich`'s primary in Jev's distribution, ties broken by option name |
| **primario = fallback** | Jev answered "none of these": the vocabulary is missing a topic, not a verdict on `enrich` |
| **primario sin juzgar** | `enrich`'s primary left the vocabulary |
| **primario sin rango** | Jev was asked about it and then left it out of its own distribution |
| **truncado** | the evidence exceeded `state_char_limit`; Jev saw a marked prefix |

Three rules worth stating plainly, because each one is a place the number could have lied:

- **`backed + doubtful + unjudged` partition `assigned_pairs`.** Backed is what is left
  once the other two are removed, never `assigned − doubtful`.
- **An assigned topic absent from `membership` is `unjudged`, never scored `0.0`.** A
  fabricated zero would put a topic Jev was never asked about at the top of the "most
  doubtful" table as the strongest disagreement in the corpus.
- **The denominators keep counting the awkward cases.** An item with no primary at all
  counts as not agreeing ("there is nothing to agree with" is not agreement) and stays in
  the denominator, so a corpus cannot improve its agreement rate by losing vocabulary.

In the markdown, the primary mismatches are **split by reason** — `Sin primario en
enrich`, `Primario sin juzgar (salió del vocabulario)`, `Jev eligió el fallback`,
`Primario ausente de la distribución de Jev`, `Desacuerdo real` — and each item appears
**once**, under the first reason that applies. The headline counters are independent and
count the same item in two of them if both are true, so the section counts can sum to
**less** than the headline. That is not a discrepancy, and
the report says so above the tables.

### The JSON summary

`generated_at` (ISO 8601 UTC) · `threshold` · `items_assessed` · `items_compared` ·
`assessments_stored` · `assessments_stale` · `assessments_orphaned` · `models` ·
`providers` · `truncated` · `input_tokens` · `input_tokens_unknown` · `cost_usd` ·
`unpriced_providers` · `assigned_pairs` · `assigned_backed` · `assigned_unjudged` ·
`enrich_backed_pct` · `jev_pairs` · `jev_backed` · `jev_backed_pct` · `doubtful_pairs` ·
`missing_pairs` · `primary_agree` · `primary_agree_pct` · `primary_fallback` ·
`primary_unjudged` · `primary_unranked` · `per_topic`.

`assessments_stored == items_assessed + assessments_stale + assessments_orphaned` — a real
partition of the side-car, so "0 vigentes" can always be told apart from "0 guardadas".

Two name pairs collide and are worth reading carefully: `summary.primary_unjudged` is a
**count of items** while `items[].primary_unjudged` is a **boolean about one item**; and
the membership-side count is `summary.assigned_unjudged` while the per-item field is
`items[].unjudged`.

`report.THRESHOLD_DEPENDENT_KEYS` names the summary keys that move when the threshold
moves. The dashboard reads it to know which numbers its slider invalidates; a test asserts
it and its complement together cover every key the summary emits.

### It refuses rather than overwrite a good report with zeros

A comparison over nothing is not a comparison of zeros — it is a plausible file of zeros
written over the last good one. So the command checks first, names the missing input, the
command that fixes it, and the artifact it left alone. `jev dashboard` refuses on the same
five conditions, naming `jev.html` instead:

```text
Error: el vocabulario está vacío o falta <data_dir>/vocab.yaml: ejecuta `xbrain vocab`. No se sobrescribe <artefacto>
Error: no hay items que comparar en <items.json>: ejecuta `xbrain extract`. No se sobrescribe <artefacto>
Error: no hay evaluaciones guardadas en <topics.json>: ejecuta `xbrain jev topics`. No se sobrescribe <artefacto>
Error: 0 evaluaciones vigentes de N guardadas (S caducadas, H huérfanas): ejecuta `xbrain jev topics` (o revisa <vocab.yaml> si acabas de cambiarlo). No se sobrescribe <artefacto>
Error: ninguna evaluación vigente tiene con qué compararse: los N items evaluados no están enriquecidos. Ejecuta `xbrain enrich`. No se sobrescribe <artefacto>
Error: ninguna evaluación vigente tiene con qué compararse: el 1 item evaluado no está enriquecido. Ejecuta `xbrain enrich`. No se sobrescribe <artefacto>
```

`<artefacto>` is a full path, and it is the one the command being run would have written:
`data/jev/topics-report.json` for `jev report`, `<output_dir>/jev.html` for `jev dashboard`.

The fifth refusal is shown twice because the whole phrase agrees in number — article, noun
and verb — so at one item it reads `el 1 item evaluado no está enriquecido`, not `los 1
item evaluado no están enriquecidos`.

The fourth is the one that costs money to misread, which is why `caducadas` is printed even
at zero in every summary line: it is the number that tells "nobody has run `xbrain jev
topics` yet" apart from "a vocabulary edit just retired every paid record you have".

### What the report does not carry

- **No wall-clock.** Time a pass with `time uv run xbrain jev topics`.
- **No language split.** To compare agreement on Spanish and English posts, join
  `topics-report.json`'s `items[].item_id` against `items.json` for `author.handle` and
  `text`. Nothing in the report carries a language field.

## `xbrain jev dashboard` — reading the page

Writes `jev.html` into the vault's output directory, next to `dashboard.html`. It prints
the same line `jev report` prints, then how many items reached the page and where it is:

```text
2583 items en el dashboard → file:///…/x-knowledge/jev.html
```

Open it with `open <uri>`.

**That count is not the `2565` in the line above it, and the difference is the same one
[§ items comparados](#items-comparados-25652583--two-different-populations) explains.** The
page carries a row for every **current assessment** — 2,583 — because an item Jev has an
opinion about is worth showing whether or not `enrich` has reached it. The report's ratio
counts the **comparable** ones, the 2,565 that carry an enrichment. Same side-car, two
questions.

It is **one self-contained file**: the data as a JSON blob, ECharts vendored into the page.
Nothing is fetched at runtime except the Google Fonts stylesheet, so it renders offline in
system fonts. Measured on this corpus on 2026-09-22 (2,609 items × 45 topics):
**4,531,327 bytes, about 4.53 MB** — ECharts 1.03 MB and the JSON blob 3.43 MB, of which
the memberships are 1.16 MB, the `obsidian://` deep links 0.45 MB and the post text
0.42 MB. Rendered with synthetic six-decimal probabilities, so the membership term is an
upper bound; a provider answering in two or three decimals ships less.

**The membership probabilities ship unrounded, on purpose.** Rounding them for size would
be a real divergence: `0.8496` renders as `0.85` and then clears a `0.85` slider, so the
page would call *backed* what the report calls *doubtful*.

What it shows: the KPI band and the side-car line (`N vigentes de M guardadas` — the
number that says whether a vocabulary edit just retired paid work), a per-topic backing
chart worst-first, a noul histogram on a log axis, three queues (doubtful, missing
candidates, primary mismatches) and a per-item drawer with every membership as a bar and
the five most probable options of the Choice.

**The page applies two cuts, and only one of them is visible.** The queues stop at **200
rows** and say so with a cut note. The **post text is cut at 240 characters** in the blob
itself and says nothing — it is enough to recognise a post in a queue row, and carrying the
whole corpus of full texts would add megabytes to a page that is already large (the 240
characters are the 0.42 MB above). Open the item's own note or its `X ↗` link for the full
post; neither the drawer nor the queue has it. The Choice distribution is cut too, to the
five most probable options, but the drawer labels that one (`5 más probables de 46`).

**The threshold slider recomputes in the browser.** `jev report` prints one threshold; the
page lets you move it, so every threshold-dependent number is re-derived client-side from
the raw probabilities, by the same rules `jev/report.py` uses.

> **If a red *"Inconsistencia entre el informe y el dashboard"* banner appears, do not
> trust the page.** On load, the page recomputes the numbers at the report's own threshold
> and compares them against the summary shipped inside it. The banner means its arithmetic
> and `jev/report.py` have diverged. Trust `xbrain jev report`, and report the banner as a
> bug. The banner also says so itself, and names the first few keys that disagreed.

Two more things the page cannot tell you itself:

- The **`nota ↗` deep link appears only for notes that exist** on disk. `jev dashboard`
  runs independently of `xbrain generate`, so run `generate` first or expect only `X ↗`.
- It is **static**. It goes stale the moment the side-car, the corpus or `vocab.yaml`
  moves. `dashboard.html` and `jev.html` are siblings — same directory, same template
  mechanism, same visual language — but different commands write them and neither
  regenerates the other. Re-run `xbrain jev dashboard` to refresh.

`xbrain generate` links `jev.html` from `_index.md` by absolute `file://` URI, exactly as
it links `dashboard.html`, **but only when the page is already on disk**: the link appears
after your first `xbrain jev dashboard`, on the next `generate`.

## Staleness: when an assessment stops counting

Every stored assessment carries a `contract`: a sha256 over the contract version, the state
as sent, and a digest of the **questions that went with it** — each question's type,
instructions and criteria. A stored assessment is *current* while recomputing that hash
today gives the same value.

| Change | Assessment | Why |
|---|---|---|
| Re-enriching the item | **still current** | the comparison against `enrich` is recomputed at report time — this is the event the report exists to look at |
| Reordering `vocab.yaml` | **still current** | the question set is canonical: one vocabulary, one wire form |
| Changing `[jev].model` or the provider | **still current** | the contract binds what Jev was *asked*, never who answered |
| New or edited evidence text | **stale** | a different state was sent |
| Adding, removing or re-describing a topic | **stale** | the questions changed |
| A different `[jev].fallback_option` | **stale** | the Choice offers different options |
| Changing `state_char_limit` so an item's cut moves | **stale, for those items only** | a different prefix was sent. An item shorter than both limits is untouched |

Stale records are **excluded** from the report and the dashboard, never compared as if they
were current, and **counted** (`caducadas`) so the exclusion is visible. An *orphaned*
record — one whose item is no longer in the store — gets its own counter (`huérfanas`): a
different event with the same symptom. `xbrain jev topics` re-asks exactly the stale ones.

`output_fingerprint` on each record says which `enrich` assignment existed at ask time. It
is **informational only** and never consulted for currency: comparing it would retire an
assessment the moment the item was re-enriched.

## Where the files live, and what protects them

| Path | What it is |
|---|---|
| `data/jev/topics.json` | the side-car: one `TopicAssessment` per item id |
| `data/jev/topics-report.json` · `.md` | the comparison, rewritten on every `jev report` |
| `<output_dir>/jev.html` | the page, rewritten on every `jev dashboard` |

`<output_dir>` is the path the CLI prints: your vault root joined with
`[paths].output_subdir` from `config.toml` (`learnings/x-knowledge/` in the examples here).
The config key is the subdirectory; `<output_dir>` is the absolute path it resolves to.

> **`data/topics.json` and `data/jev/topics.json` are different files.** The first holds the
> synthesised topic pages and is part of the store. The second holds Jev's assessments and
> is a side-car. Only the first is snapshotted.

**This file costs money to regenerate and there is no undo.** Three consequences:

- **It is not snapshotted.** `xbrain snapshot create` copies the four flat store artifacts
  from `data/`; `data/jev/topics.json` is one level down and is not among them. `xbrain
  snapshot restore` therefore rolls the store back and leaves the side-car at its newer
  state. A restore reverts **`vocab.yaml` as well as `items.json`**, and the contract hashes
  the vocabulary-derived questions digest — so a restore from before a `vocab --regenerate`
  moves the digest and retires **every record at once**, a full re-bill; only a restore that
  leaves both the item's evidence and the vocabulary untouched leaves an assessment current.
  Retired records are reported as `caducadas`, and `xbrain jev topics` re-asks — and re-pays
  for — them. Either way staleness is **detected, never consumed**: a reverted item is never
  compared against an answer about its newer text.
- **It is not in git.** `data/` is gitignored in full, so there is no `git checkout` back to
  a good copy. `--force` overwrites a paid record with no recovery, and a corrupt file is
  repaired by hand or paid for again — which is why a malformed side-car raises instead of
  quietly starting from `{}`.
- **Two runs at once are last-write-wins.** The file is rewritten wholesale on every save.

It is written atomically and dumped sorted and pretty, so an unchanged corpus re-dumps
byte-identically and a hand `diff` between two runs shows only what moved. That copy is
also the only backup there is.

## Vendor facts, with their dates

These are the only dated third-party claims in the Jev layer, and this is the one place
they are recorded. Everything else that needs them points here. Re-check them when the
`jev-latest` alias advances.

| Fact | Value | Source, dated |
|---|---|---|
| Request budget | 64k tokens for `state` **plus all questions**, and 32k for `state` plus the single longest question | docs.typesafe.ai/models, 2026-09-22 |
| Input price | `0.042 $` per million **input** tokens for `jev-1.13.0`; output tokens are free | docs.typesafe.ai/models, 2026-09-22 |
| Rate limits | 250k tokens/second and 1,200 requests/minute, `429` over either | docs.typesafe.ai/models, 2026-09-22 |
| Concurrency | throughput saturates around **8** concurrent requests | PriorBench, 2026-09-20 — an independent benchmark, **not** a TypeSafe figure |

Two readings that follow from the first row and are easy to get backwards:

- **The 64k budget is the one that binds here**, not the 32k one, because a call sends one
  question per vocabulary topic plus the Choice. It tightens as `[vocab].target_count`
  grows.
- **`state_char_limit` is a bound on the evidence, not a defence of that budget.** Cutting
  the state shrinks only the state half of it; the question half does not move.

The price is per *version* while `[jev].model` defaults to the moving `jev-latest` alias,
so every figure derived from it is an estimate. Each stored assessment records the concrete
model that answered, so a report can always say what it priced.

## Troubleshooting

Operator-facing failures print as `Error: <mensaje>` and exit 1.

```text
Error: TYPESAFE_API_KEY no encontrada: expórtala o pégala en <repo>/.env (ver .env.example)
```

No key in the environment and none in `<repo>/.env`. Nothing was called and nothing was
written. `xbrain jev topics --dry-run` reports whether a key is visible without spending.

```text
Error: el SDK de TypeSafe no está disponible (…): instala las dependencias con `uv sync` y vuelve a lanzar el comando
```

The key was accepted but the vendor SDK is not importable — usually a half-finished
`uv sync`. No call was made.

```text
Error: Jev: configuración inválida (…)
Error: TYPESAFE_API_KEY vacía: ponla en el entorno o en <repo>/.env
```

The key reached the SDK and was rejected before any request — typically a stray whitespace
or non-ASCII character that survived `.env` parsing.

```text
  FALLO <id>: Jev API: …          (on stderr, the first 10; then `  … y N fallos más`)
```

Those items failed after the SDK's retries; everything else was saved. Re-run `xbrain jev
topics` — only the failures are pending. Sustained `Jev API: … 429` means the rate limit;
lower `[jev].concurrency`.

```text
Error: ninguna de las N evaluaciones terminó; primer error: …
```

Every call failed: a wrong key, no network, or the API is down. **Nothing was written** —
the side-car is exactly as it was.

```text
Interrumpido: N evaluaciones nuevas guardadas (M en total) en <path>     (exit 130)
```

Ctrl-C. N is this run's new records, M is the file total. Re-run to continue; the banked
records count as `vigentes` and are not re-billed. Under `--force` the noun is
`evaluaciones re-evaluadas` instead of `evaluaciones nuevas`, because those records are
re-bills rather than new work. `Interrumpido: nada nuevo que guardar` means the interrupt
arrived before the first answer and nothing was written.

```text
Error: no se pudo guardar <path> (N evaluaciones pagadas sin guardar): …
```

The run completed and the write failed — a full disk, a permission, a read-only mount. **N
records were billed and are lost.** Fix the path and re-run.

```text
Error: el vocabulario está vacío: ejecuta `xbrain vocab` antes de `xbrain jev topics`
```

No vocabulary to ask about. Raised before the first call, so it costs nothing.

```text
Error: [jev].fallback_option 'otro' choca con un slug del vocabulario
```

The escape option and a real topic are the same string, which would make "none of these"
and that topic the same option. Pick another name in `config.toml`. Also raised before any
call — as are a duplicate slug and a topic with a blank description.

```text
Error: <path>: side-car ilegible (…)
```

`data/jev/topics.json` is unparseable, is not a JSON object, or holds a record this build
refuses. It is **not** silently discarded: returning an empty map would re-ask and re-pay
for the whole corpus and then overwrite whatever was still readable. Repair the file by
hand from a copy, or delete it and pay again.

```text
Error: ids desconocidos: a, b
Error: --limit debe ser >= 1
Error: --threshold debe estar en [0.0, 1.0]
```

Operator errors, caught before anything is asked or written. A threshold above 1.0 would
make every assignment doubtful and below 0.0 would make everything backed — a plausible
file of noise over the last good one, which is why the bound is checked before the write.

```text
Error: 0 evaluaciones vigentes de N guardadas (S caducadas, H huérfanas): …
```

The vocabulary or the evidence moved and retired the stored contracts. See
[Staleness](#staleness-when-an-assessment-stops-counting). The remedy is `xbrain jev
topics`, and it is a re-bill.

**A red banner on `jev.html`**
The page's own arithmetic disagrees with the report embedded in it. Trust `xbrain jev
report` and report it as a bug — see [the dashboard](#xbrain-jev-dashboard--reading-the-page).

**The dashboard numbers look old**
It is a static page. Re-run `xbrain jev dashboard`.

---

## See also

- [ARCHITECTURE.md § jev](../ARCHITECTURE.md#jev) — how the layer is built: the call shape,
  the contract, the seams, and why the side-car is not the store.
- [docs/troubleshooting.md](troubleshooting.md) — everything in XBrain that is not Jev.
- [docs/tutorial.md](tutorial.md) — the pipeline this compares against, end to end.
- [`config.toml.example`](../config.toml.example) — the `[jev]` block, annotated.
