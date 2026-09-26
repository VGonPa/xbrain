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
uv run xbrain jev dashboard                 # <output_dir>/jev.html at [jev].threshold — open the printed URI
```

`jev topics` is the only command that spends money. `report` and `dashboard` re-read the
side-car it already paid for: no API call, no key needed, no cost, any number of times.

The full option list:

| Command | Options |
|---|---|
| `xbrain jev topics` | `--id TEXT` (repeatable — only these items) · `--limit INTEGER` · `--force` · `--dry-run` |
| `xbrain jev report` | `--threshold FLOAT` (default `[jev].threshold`) |
| `xbrain jev dashboard` | none — it always compares at `[jev].threshold` |

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
  and it is why a `--force` line can look like it sums to more than the corpus. When it is
  non-zero the run also copies the side-car first and prints
  `Copia de seguridad: data/jev/topics.<UTC stamp>.bak`
  ([the side-car section](#where-the-files-live-and-what-protects-them) has the details).
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
different figure from the bill it recaps. Two markers exist because a bare `~0.0000 $`
cannot say which zero it is:

- `(+K sin recuento)` — K records whose provider reported no token usage. They contribute
  nothing they cannot prove, so without this a fully paid run reports itself as free.
- `· proveedor sin tarifa: X` — **inside** the cost parentheses, after the figure
  (`N tokens de entrada (~X $ · proveedor sin tarifa: a, b)`): a provider absent from the
  price table. It contributes `0.0` rather than borrowing another vendor's rate, and is
  **named** rather than counted. `proveedores sin tarifa` for more than one.

The figure is an **estimate**, not an invoice: the rate is a list price for a concrete
model version while `[jev].model` defaults to a moving alias. See [Vendor facts](#vendor-facts-with-their-dates).

### It logs the pass: `data/jev/runs.jsonl`

The side-car keeps only the **latest** answer per item, so it cannot say what Jev has cost
over time. Every pass that **sent at least one request** appends one JSON line to
`data/jev/runs.jsonl` and says so on its last line, after the side-car is saved:

```
pasada registrada → /…/data/jev/runs.jsonl
```

A `--dry-run`, a pass with nothing to evaluate, or a Ctrl-C before the first request writes
nothing. Every count is taken where the calls are made, so an answer counts the moment Jev
returns it, whatever xbrain does with it next:

| Field | Meaning |
|---|---|
| `kind` | `topics` (the only kind today; a line without it reads as `topics`) |
| `started_at` · `finished_at` | UTC. The page's history table shows the start, in local time |
| `requests` | calls xbrain **sent**, each item once. Retries inside the vendor SDK are invisible to xbrain and are not counted |
| `ok` | answers kept in the side-car |
| `failed` | calls that raised (a provider error, a 402) plus answers xbrain refused (a malformed answer set) |
| `unsaved` | only after Ctrl-C: answers that came back but were not yet saved when the interrupt landed. Paid, not kept, not failed |
| `input_tokens_by_provider` · `input_tokens` | tokens of **every** answer that came back — refused and unsaved ones included, because each was billed — per provider, and their sum. Empty and 0 when nothing answered |
| `input_tokens_unknown` | answers that reported no usage |
| `models` | the distinct models that answered, sorted |
| `interrupted` | Ctrl-C. `requests - ok - failed - unsaved` is then the number of calls still in flight |

It stores **tokens, never dollars**: every report prices the history at read time with the
same formula the side-car uses, so a price correction reprices every past pass.

The line is written on **every** exit path of a pass that sent something: success, partial
failure, the all-failed error (a 402 on every call is 20 requests made, and that is history
too), a side-car that could not be written, and Ctrl-C (exit 130). Writing it can never
change how the pass ends: if the line cannot be built or appended, or the terminal is gone
(`| head`), the pass keeps its exit code and its saved records, and the line is printed to
stderr under `no se pudo registrar la pasada en …; añádela a mano:` so you can append it by
hand.

**What the log cannot see**, and how it still shows up: a worker that picks up its item
just after Ctrl-C can send one more call that is never logged, and a pass killed by SIGTERM
or `kill -9` logs nothing at all. The answers those calls stored are in the side-car with an
`asked_at` that no logged pass covers, so the reports count and price them as **fuera del
registro** (see below) instead of dropping them.

**The file protects itself against a torn line.** A crash or a full disk in the middle of a
write can leave the last line cut short. The next append starts on a new line instead of
gluing its record onto the fragment, and an append whose write fails is rolled back to the
file's size before it. A line that does not parse is refused by `jev report` and `jev
dashboard` with its path and line number, never skipped.

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
`items_unassessed` ·
`assessments_stored` · `assessments_stale` · `assessments_orphaned` · `models` ·
`providers` · `truncated` · `input_tokens` · `input_tokens_unknown` · `cost_usd` ·
`unpriced_providers` · `assigned_pairs` · `assigned_backed` · `assigned_unjudged` ·
`enrich_backed_pct` · `jev_pairs` · `jev_backed` · `jev_backed_pct` · `doubtful_pairs` ·
`missing_pairs` · `primary_agree` · `primary_agree_pct` · `primary_fallback` ·
`primary_unjudged` · `primary_unranked` · `posts_with_disagreement` · `posts_enrich_only` ·
`posts_jev_only` · `posts_primary_differs` · `per_topic` · `topic_confusion` ·
`primary_confusion`.

`assessments_stored == items_assessed + assessments_stale + assessments_orphaned` — a real
partition of the side-car, so "0 vigentes" can always be told apart from "0 guardadas".
`items_unassessed` counts the posts of the corpus with no current answer — never asked, or
stale — which is what `xbrain jev topics` would ask next (it still skips a post with no
evidence at all).

Two name pairs collide and are worth reading carefully: `summary.primary_unjudged` is a
**count of items** while `items[].primary_unjudged` is a **boolean about one item**; and
the membership-side count is `summary.assigned_unjudged` while the per-item field is
`items[].unjudged`.

`posts_with_disagreement` counts the compared posts with at least one disagreement — a
topic enrich assigned that Jev does not back, a topic Jev backs that enrich did not assign,
or a primary that differs (a post enrich left without a primary counts). The recap line
prints it as `N posts con desacuerdo`, and the dashboard's "Con discrepancias" count is the same
number. `posts_enrich_only`, `posts_jev_only` and `posts_primary_differs` split the same posts by
KIND of disagreement (a post can count in more than one). Each `per_topic` row carries
`disagreeing` = its `doubtful` + its `missing`: the posts that disagree about that topic.
Separately, each row also carries `enrich_primary` and `jev_primary`: on how many compared posts
each side picked it as THE topic (these are not part of `disagreeing`).

`topic_confusion` pairs what each side put INSTEAD. On every post, each topic only enrich has
(`doubtful`) is paired with each topic only Jev has (`missing`); a post where only one side has
something pairs it with `null` (enrich put a topic Jev does not back and Jev put nothing in its
place, or Jev added one without replacing anything). Each row is `{enrich, jev, posts}`, most
posts first. It is a PRODUCT: a post with two enrich-only topics and two Jev-only topics is in
four rows. `primary_confusion` is the same shape for the primary: enrich's primary × Jev's
choice on every post where they differ (`enrich: null` = enrich left no primary; the fallback
appears as Jev answered it), and its rows' `posts` add up to `posts_primary_differs`.

The JSON report carries these COUNTS only. The posts behind each row live in one index,
`report.post_sets`, which only the page ships (to open a pair's posts): the lists grow with
every evaluated post, and a report file is for numbers.

Under the recap line, `jev report` prints the run history in the shared cost sentence:

```text
Histórico: 3 pasadas · 2640 peticiones · 15520000 tokens de entrada (~0.6518 $)
Histórico: sin pasadas registradas · 20 evaluaciones fuera del registro de pasadas: 125548 tokens de entrada (~0.0053 $)
```

The recap line prices the side-car (the latest answer per item); this one prices every pass
the log recorded, re-asks included. A stored answer counts as logged only when its
`asked_at` falls inside some logged pass. Everything else — answers from before the log
existed, from a copy of xbrain that does not write it, from a pass whose line could not be
appended or that was killed — is named and priced at the end
(`· N evaluaciones fuera del registro de pasadas: …`) instead of being silently left out.
With no logged pass at all the line says `sin pasadas registradas` rather than quoting a
`~0.0000 $` that reads as free.

`jev report` reads the run log **before** writing either report: a corrupt line refuses the
command with the line number, and the previous reports are left alone.

### It refuses rather than overwrite a good report with zeros

A comparison over nothing is not a comparison of zeros — it is a plausible file of zeros
written over the last good one. So the command checks first, names the missing input, the
command that fixes it, and the artifact it left alone. `jev dashboard` refuses on the same
five conditions, naming `jev.html` instead — six lines below, because the last condition is
shown in both of its number forms:

```text
Error: el vocabulario está vacío o falta <data_dir>/vocab.yaml: ejecuta `xbrain vocab`. No se sobrescribe <artefacto>
Error: no hay items que comparar en <items.json>: ejecuta `xbrain extract`. No se sobrescribe <artefacto>
Error: no hay evaluaciones guardadas en <topics.json>: ejecuta `xbrain jev topics`. No se sobrescribe <artefacto>
Error: 0 evaluaciones vigentes de N guardadas (S caducadas, H huérfanas): ejecuta `xbrain jev topics` (o revisa <vocab.yaml> si acabas de cambiarlo). No se sobrescribe <artefacto>
Error: ninguna evaluación vigente tiene con qué compararse: los N items evaluados no están enriquecidos. Ejecuta `xbrain enrich`. No se sobrescribe <artefacto>
Error: ninguna evaluación vigente tiene con qué compararse: el 1 item evaluado no está enriquecido. Ejecuta `xbrain enrich`. No se sobrescribe <artefacto>
```

The last refusal appears twice because the whole phrase agrees in number — article, noun and
verb — so at one item it reads `el 1 item evaluado no está enriquecido`, never `los 1 item
evaluado no están enriquecidos`.

`<artefacto>` is a full path, and it is the one the command being run would have written:
`data/jev/topics-report.json` for `jev report`, `<output_dir>/jev.html` for `jev dashboard`.

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
the same line `jev report` prints, then how many posts the page carries and where it is:

```text
2609 posts en el dashboard (293 comparados con Jev) → file:///…/XBrain/jev.html
```

Open it with `open <uri>`.

**The page is a browser of the whole corpus that shows, post by post, what enrich put and
what Jev sees, so you can find and fix the wrong topics, and what finding out cost.** It
compares at `[jev].threshold` (0,85 by default), fixed: there is no control that moves it,
and the page recomputes nothing. Every number comes from the same `build_report` call
`xbrain jev report` makes, so the two surfaces always agree; the costs come from the same
price formula (`report.run_history`, `report.post_cost_view`). The browser only filters,
searches and orders the posts. To read the side-car at another threshold, use
`xbrain jev report --threshold`.

Top to bottom:

1. **Header** — when the page was written, the model that answered, the threshold with
   `(config)` next to it, and a link to this document.
2. **Qué es esto** — a short paragraph on what the page is for.
3. **Coste y peticiones** — three figures and a folded table:
   - **Total**: requests, input tokens and dollars across every logged pass; `—` and
     `sin pasadas registradas` when none is logged yet.
   - **Media por post**: what one current stored answer cost on average, labelled
     `media de K de las N evaluaciones vigentes` — K is how many could be priced (the
     provider reported usage and has a price). Unpriced providers are named, not averaged in.
   - **Evaluaciones vigentes**: how many current answers the side-car holds and what exactly
     those cost.
   - **Histórico por pasada** (click to unfold): one row per `xbrain jev topics` pass,
     newest first — start date, requests, ok, failed, tokens, cost, model, and for an
     interrupted pass how many calls were in flight and how many answers were not saved.

   Answers no logged pass covers are named in one line
   (`N evaluaciones fuera del registro de pasadas (~X $ según sus propios tokens) …`) and are
   not added to the total: the side-car keeps only the latest answer per item, so it cannot
   reconstruct the passes the log missed. If `runs.jsonl` has a line that cannot be read,
   the strip shows that error (with the line number) in place of the total and the history,
   `jev dashboard` repeats it on stderr, and the rest of the page renders as usual.
4. **Three numbers**, each with a one-line explanation:
   - *Jev confirma X de Y topics de enrich (Z %)* — enrich's topics that Jev also sees at or
     above the threshold (`assigned_backed` of `assigned_pairs`, `enrich_backed_pct`).
   - *Jev añadiría N topics que enrich no puso* — `missing_pairs`.
   - *El topic principal coincide en P %* — `primary_agree_pct`.
5. **Four tabs**: **Posts**, **Topics**, **Comparar Jev vs enrich** and **Configuración**.
   Posts and Topics are described below; Comparar and Configuración show a one-line
   placeholder (they arrive in later PRs).

### The Posts tab

**Left, the filters.** Seven views of the corpus. *Todos* shows every post (the page's own
count of them, `totals.items`); the other six are report counts, and each view lists exactly
the posts its number counts — which posts belong to which view is decided in Python, card by
card, from the same comparison the report counts:

| View | Lists | Number beside it |
|---|---|---|
| *Con discrepancias* (default) | posts with any disagreement | `posts_with_disagreement` |
| *Enrich asigna y Jev no* | posts with a topic enrich put and Jev does not back | `posts_enrich_only` |
| *Jev añadiría topic* | posts with a topic Jev backs and enrich did not put | `posts_jev_only` |
| *Primario distinto* | posts whose primary topic is not Jev's choice | `posts_primary_differs` |
| *Jev eligió «otro»* | posts where Jev answered "none of these" | `primary_fallback` |
| *Sin evaluar por Jev* | posts with no current answer, never asked or stale | `items_unassessed` |

Under them, the vocabulary's topics, most disagreement first, each with three numbers counted
over the **compared** posts only: how many enrich put there (`assigned`), how many of those
Jev confirms (`backed`) and how many disagree about it (`disagreeing` = enrich puts it and Jev
does not back it, plus Jev backs it and enrich did not put it). Clicking a topic narrows the
current view, and a line under it says what the list now counts:

- under *Con discrepancias*, the posts that disagree about that topic — exactly its
  `disagreeing` number;
- under *Enrich asigna y Jev no* / *Jev añadiría topic*, that one direction — its `doubtful`
  / `missing` number;
- under the other views, every post where enrich or Jev has the topic, including Jev's
  primary choice, and including posts Jev has not evaluated (enrich's topics). Here the list
  count (`mostrando N`) is the number to read.

Click the topic again, or its chip above the list, to drop it. On a narrow screen the topic
list starts folded.

**Right, the posts**, fifty at a time and more as you scroll (or with *Mostrar más*). Each
card is a share-style preview built from data XBrain already has, with nothing fetched from
X:

- the author, `@handle` and date, the whole text (a long one starts folded behind *ver
  todo*), `X ↗` and `nota ↗`;
- up to four photos, from the vault's `_media/` folder (the same files the notes embed) by a
  path relative to the page. A video shows the first extracted frame of **its own** video
  source with ▶. A picture the page cannot show says why: *falta en _media/: corre xbrain
  generate* (downloaded, not mirrored yet), *imagen sin descargar*, *la descarga falló*,
  *el fichero ya no está*, or *vídeo sin fotograma extraído*;
- the quoted post as a nested card (the same quoted post Jev read, cut at 600 characters),
  or a *Post citado no disponible* box linking to it on X — saying *no se pudo leer* when a
  fetch failed, or *sin leer todavía: corre xbrain refresh-quoted* when none was tried;
- the fetched linked page as a mini card with its domain and kind (`artículo`, or
  `x_article · página de X` — some of those hold scraped replies rather than an article),
  marked *no se pudo leer* when the fetch failed, or else the first link in the post.

Under the preview, **Jev vs enrich**: one row per topic either side has — enrich ✓ or —,
Jev's probability as a bar (the tick is the threshold) and a number, and the verdict
*coinciden* / *solo enrich* / *solo Jev* (or *sin juzgar* for a topic that left the
vocabulary). When Jev's primary choice is a topic no other row names, it gets a row marked
*primario de Jev*; it is not a separate disagreement (the primary line counts it). Then both
primary topics, highlighted when they differ, with Jev's probability for its choice; the
number of discrepancies, the model, when it was asked, and what the answer cost.
`recortado` means the evidence was longer than `[jev].state_char_limit` and was cut before
sending: the tweet always goes whole (it leads the state), what was dropped is the tail of
the other sources.

*Lo que vio Jev* (folded) lists each evidence surface Jev was sent — tweet, author, video
title, video transcript, video frame descriptions, image descriptions, linked article title
and body, thread, quoted post — with its size and whether the cut reached it, from the same
`state_surfaces` split of the state `jev topics` sends. The tweet, the author and the quoted
post are already on the card, so they point there instead of repeating it; every other
surface shows at most **600 characters**, and the page says so when one is longer.

A post Jev evaluated but enrich never enriched shows what Jev sees (its topics at the
threshold and its primary), with nothing to compare. A post Jev has not answered for shows
enrich's topics, a *sin evaluar por Jev* mark (plus *evaluación caducada* when its answer is
stale), and a **copiar comando** button with the exact line that asks for it:
`xbrain jev topics --id <id>`. A post with no evidence at all says *sin evidencia* instead,
because `jev topics` skips it. The page never asks Jev anything.

Search matches text, author, id and topics (slug or label); sort is *más discrepancias*
(default), *más recientes* or *más caros*. Keys: **j** / **k** next / previous post, **n** /
**p** next / previous post with a discrepancy. The filter, topic, search and sort live in the
URL (`#posts?f=uneval&t=ai-coding&q=…&s=recent`), so a view can be bookmarked and survives a
reload, including searches with `&`, `+`, `%` or `?`. The page follows the system's light or
dark theme. If one post cannot be drawn, its card says so in one line and the rest draw.

Below the list, one line names what the numbers leave out: stale and orphaned answers, and
evaluated posts with no enrichment to compare against.

It is **one file**: the data as a JSON blob in the page, no charting library, no external
scripts. Photos are files next to it in `_media/`, not embedded, so moving `jev.html` out of
the vault loses the pictures and nothing else. The only network reference is the Google
Fonts stylesheet. Measured 2026-09-26 on the real vault (2,609 posts, 293 evaluated):
**3,762,648 bytes**, about **1.4 KB per post** — ~2.7 KB for an evaluated post (its topic
rows and evidence) and ~1.2 KB for the rest. The Topics tab's data is the two confusion lists
(~18 KB of counts) and `post_sets` (~23 KB, ~80 bytes per evaluated post). With every post
evaluated the page would be about **7.2 MB**. JavaScript draws everything; without it the page says so.

### The Topics tab

**The index** (`#topics`) lists the vocabulary's topics with, per topic, the `per_topic` row of
the report: *Enrich lo pone* (`assigned`), *Jev confirma* (`backed`, at the threshold, which the
header shows), *Acuerdo* (`backed_pct`, `—` when enrich never put it), *Jev lo añadiría*
(`missing`), *Discrepancias* (`disagreeing`), and *Principal según enrich / según Jev*
(`enrich_primary` / `jev_primary`). Each header says what it counts in one line (also the
tooltip of every cell). Under the table, a line names the posts where Jev chose «otro»: they
are not in the *Principal según Jev* column.

By default the worst agreement comes first — the report's own order, by the exact ratio — among
the topics enrich put on **at least 5 posts** (`TOPIC_MIN`, shipped as `topic_min`); the rest
follow, marked *pocos datos*, since an agreement rate over one or two posts is noise. Clicking a
header sorts by it (click again to reverse). *Acuerdo* sorts by the exact ratio, never the
rounded percentage, and topics enrich never put go last in both directions. The URL keeps the
order (`#topics?o=disagreeing&d=desc`) without adding a history entry per click, and the back
link from a topic returns to it. On a phone the headers fold into each row and a drop-down
offers the same orders.

**A topic's page** (`#topics?t=<slug>`, bookmarkable) shows its description and numbers, then:

- **Con qué se confunde**: from `topic_confusion`, what Jev put in its place where enrich put
  this topic and Jev does not back it, and what enrich had where Jev adds it, with how many
  posts. *nada en su lugar* is a post where the other side put nothing instead. The first eight
  show; *ver todos* opens the rest.
- **Topic principal**: from `primary_confusion`, what Jev chose where enrich picked this topic as
  primary, and what enrich had where Jev picked it. *sin principal* is a post enrich left
  without one; *(ninguno del vocabulario)* is Jev's «otro»; *(ya no está en el vocabulario)* is a
  primary enrich chose that has since left `vocab.yaml`.
- When a side is empty, it says what the row says: *Enrich no lo pone en ningún post comparado*,
  *Enrich / Jev nunca lo elige como principal*, or that the two always agree.
- Each row opens exactly its posts, as cards (`#topics?t=<slug>&cx=<enrich>~<jev>` or `&px=…`,
  with `-` for "nothing"), scrolled into view; *quitar este cruce* goes back. A pair that does
  not involve the topic, or no longer exists in these data, says *Ese cruce ya no existe en
  estos datos* and shows the topic's groups; a post of the pair missing from the page is
  counted (*N de M posts no están en esta página*). An unknown topic says it is not in the
  current vocabulary.
- Otherwise its posts, as the same cards as the Posts tab, in three groups — **Coinciden**
  (`backed`), **Solo enrich** (`doubtful`), **Solo Jev** (`missing`) — twenty at a time, with
  buttons that jump to each group.

Links: *ver en Posts* opens the Posts tab on every post with this topic (`#posts?f=all&t=…`);
*sus discrepancias en Posts* on the ones that disagree about it (`#posts?t=…`). On any card, a
topic's name in the Jev vs enrich rows opens its page, and the Posts rail offers *ficha del
topic* for the topic it is filtering by. Back and forward move between the index, a topic and a
pair. If the tab ever fails to draw, it says so inside the tab.

Two more things the page cannot tell you itself:

- The **`nota ↗` deep link appears only for notes that exist** on disk, and photos only for
  files already mirrored into `_media/`. `jev dashboard` runs independently of `xbrain
  generate`, so run `generate` first or expect only `X ↗` and placeholders that say
  *falta en _media/*.
- It is **static**. It goes stale the moment the side-car, the run log, the corpus or
  `vocab.yaml` moves. `dashboard.html` and `jev.html` are siblings — same directory, same
  template mechanism, same visual language — but different commands write them and neither
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

Stale records are **excluded from every number** in the report and the dashboard, never
compared as if they were current, and **counted** (`caducadas`) so the exclusion is visible.
On the dashboard their post is still a card, marked *evaluación caducada*, under *Sin evaluar
por Jev*. An *orphaned*
record — one whose item is no longer in the store — gets its own counter (`huérfanas`): a
different event with the same symptom. `xbrain jev topics` re-asks exactly the stale ones.

`output_fingerprint` on each record says which `enrich` assignment existed at ask time. It
is **informational only** and never consulted for currency: comparing it would retire an
assessment the moment the item was re-enriched.

**`xbrain vocab` says what it just retired.** Any write of `vocab.yaml` moves the questions
digest, so it expires the whole side-car at once — a re-worded description does it as surely
as a new topic. Rather than leave that to be discovered by the next report's `0 vigentes`,
`vocab --apply` and `vocab --executor api` print it:

```text
2583 evaluaciones de Jev quedan caducadas: `xbrain jev topics` las vuelve a pedir (y a facturar).
```

A plain `jev topics` is the whole remedy — a retired record is not current, so it is selected
without `--force`, and `--force` would additionally re-bill whatever is still current.

A worksheet export (`vocab --executor claude-code` or `manual`) writes no vocabulary, so it
retires nothing and says nothing.

## Where the files live, and what protects them

| Path | What it is |
|---|---|
| `data/jev/topics.json` | the side-car: one `TopicAssessment` per item id |
| `data/jev/topics.<UTC stamp>.bak` | a copy of the side-car, written before a `--force` run re-asks a current record. Never pruned |
| `data/jev/runs.jsonl` | the run log: one line per `jev topics` pass that sent a request. Append-only |
| `data/jev/topics-report.json` · `.md` | the comparison, rewritten on every `jev report` |
| `<output_dir>/jev.html` | the page, rewritten on every `jev dashboard` |

`<output_dir>` is the path the CLI prints: your vault root joined with
`[paths].output_subdir` from `config.toml` (`learnings/x-knowledge/` in the examples here).
The config key is the subdirectory; `<output_dir>` is the absolute path it resolves to.

> **`data/topics.json` and `data/jev/topics.json` are different files.** The first holds the
> synthesised topic pages and is part of the store. The second holds Jev's assessments and
> is a side-car. Only the first is snapshotted.

The run log `data/jev/runs.jsonl` lives beside the side-car and shares its standing: not
snapshotted, not in git (`data/` is gitignored), never rewritten — only appended to.

**This file costs money to regenerate and `snapshot restore` will not bring it back.** Four
consequences:

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
  a good copy. A corrupt file is repaired by hand or paid for again — which is why a
  malformed side-car raises instead of quietly starting from `{}`.
- **`--force` keeps a copy, and it is the only automatic one.** A run that actually re-asks
  a current assessment copies the side-car to `data/jev/topics.<UTC stamp>.bak` first and
  says so:

  ```text
  Copia de seguridad: data/jev/topics.2026-09-22T18-30-05-123Z.bak
  ```

  Taken before the client is built — so a copy that cannot be written stops a run before it
  is billed — and before the first checkpoint, so it is the file as it was. A `--force` that
  re-asks nothing current writes no copy: the trigger is the re-ask (`N forzados`), not the
  flag. To restore one, stop any running `jev` command and move the `.bak` back over
  `data/jev/topics.json`.

  **They are never pruned.** Nothing deletes them — not `jev topics`, not
  `snapshot restore`, not a retention rule — because the copy an operator wants is the one
  from before the run they regret, which the tool cannot know. Delete them by hand; each is
  the size of the side-car.
- **Two runs at once are last-write-wins.** The file is rewritten wholesale on every save.

It is written atomically and dumped sorted and pretty, so an unchanged corpus re-dumps
byte-identically and a hand `diff` between two runs shows only what moved.

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

And one that follows from the third row, for the same reason:

- **Of the two rate ceilings, the REQUEST one binds on this corpus.** At ~6k input tokens
  per call, 1,200 req/min is 20 req/s, which is ~120k tok/s — 48 % of the 250k tok/s limit
  while the request rate is at 100 % of its own. The token ceiling would bind first only
  above ~12.5k tokens per call, roughly double what this corpus sends. So a sustained `429`
  is answered by lowering `[jev].concurrency`, not by cutting `state_char_limit` (which, by
  the line above, moves almost nothing).

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
Error: el SDK de TypeSafe no está disponible (…): instala las dependencias con `uv sync --extra dev --locked` y vuelve a lanzar el comando
```

The key was accepted but the vendor SDK is not importable — usually a half-finished
install. Re-run the install command from [the README](../README.md#installation). Note the
`--extra dev --locked`: a bare `uv sync` prunes the environment to the resolved set, and
`dev` is an *extra*, so it would uninstall `pytest`, `ruff`, `mypy`, `poe`, `bandit` and
`detect-secrets` — repairing the SDK and silently removing `uv run poe check`. No call was
made.

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
Error: ninguna de las N evaluaciones terminó: N fallos, K motivos distintos; primero: …
```

Every call failed: a wrong key, no network, or the API is down. **Nothing was written** —
the side-car is exactly as it was. Read `K` before the quoted reason: **one** distinct
motive is a single cause — a key, a quota, an outage — so fixing that one thing and
re-running is the whole remedy, while **many** means the failures are per item and the
quoted one does not cover the rest.

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

**`Error: <path>/runs.jsonl: registro de pasadas ilegible en la línea N (…)`**
One line of the run log does not parse: a line cut short by a crash mid-write, or a hand
edit. **Split it, never delete it.** A file written before the torn-line guard existed can
hold a fragment with a complete, valid record glued straight after it on the same line —
that record is a paid pass. Put the record on its own line and delete only the fragment.
`jev report` refuses until it is fixed (and writes nothing); `jev dashboard` still renders,
with the error in place of the cost strip.

**`no se pudo registrar la pasada en …; añádela a mano:`**
`jev topics` could not append to the run log (disk full, permissions). The pass itself is
fine: its answers are saved and its exit code is unchanged. The next line on stderr is the
JSON record; append it to `data/jev/runs.jsonl` once the disk is fixed.

**A red *"La página no pudo dibujarse"* banner on `jev.html`**
The page's script failed while loading. The numbers are in `xbrain jev report`; report the
message in the banner as a bug. (A post that fails to draw later shows *no se pudo dibujar*
on its own card instead, and the rest of the page keeps working.)

**Cards show *falta en _media/: corre xbrain generate* instead of photos**
The photos are downloaded (`data/media/`) but not yet mirrored into the vault, which
`xbrain generate` does. Run it, then `xbrain jev dashboard` again.

**The dashboard shows `—` as the total cost**
No pass has been logged yet: the side-car was filled before the run log existed (or by a
copy of xbrain that does not write it). The `fuera del registro` line under the figures
gives those answers' cost from their own tokens.

**The dashboard numbers look old**
It is a static page. Re-run `xbrain jev dashboard`.

---

## See also

- [ARCHITECTURE.md § jev](../ARCHITECTURE.md#jev) — how the layer is built: the call shape,
  the contract, the seams, and why the side-car is not the store.
- [docs/troubleshooting.md](troubleshooting.md) — everything in XBrain that is not Jev.
- [docs/tutorial.md](tutorial.md) — the pipeline this compares against, end to end.
- [`config.toml.example`](../config.toml.example) — the `[jev]` block, annotated.
