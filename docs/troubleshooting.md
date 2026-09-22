# Troubleshooting & FAQ

Common failures and how to fix them. Most are environment issues (auth, PATH,
external tools), not bugs.

## X session expired / auth fails

Symptoms: `extract`/`sync` scrapes 0 posts, or `status` says it can't
authenticate. X sessions are short-lived.

Fix — re-import cookies from a browser you're logged in to:

```bash
# Chrome — log in to x.com in Chrome first, then:
.venv/bin/python scripts/import_chrome_session.py
# → "auth_token: OK"

# Safari — log in in Safari, grant your terminal "Full Disk Access"
# (System Settings → Privacy & Security), then:
.venv/bin/python scripts/import_safari_session.py
```

`xbrain login` (in-app Playwright login) exists but is unreliable with
Google/SSO accounts — the automated browser gets blocked. Cookie import is the
recommended path.

## "Re-saw 0 known items on a non-empty store" — the run aborts without saving

A safety tripwire: extraction saw none of the items it already has, which almost
always means an **expired session** or an X GraphQL change, not that your
bookmarks vanished. It aborts rather than overwrite good data. Re-authenticate
(above) and re-run. If you're sure the store is stale, `--force` overrides it.

## `extract` captured nothing — "0 respuestas de … en toda la timeline"

Symptom: `extract --source tweets` (or `sync`) aborts with

```
own_tweet: 0 respuestas de UserOriginalsTimeline/UserTweets en toda la timeline —
no es que no haya items nuevos, es que no se capturó NADA. Lo normal es que X haya
renombrado la operación: mira las operaciones GraphQL reales de la página y añade
el nombre nuevo a `_OPERATIONS`.
```

Cause: X renames its internal GraphQL timeline operations without notice. The
own-tweets timeline answered to `UserTweets` until X moved it to
`UserOriginalsTimeline` (measured 30-ago-2026). The *parser* survives a rename —
it anchors on the `tweet_results` key, not on a path — but the **capture filter**
matches by operation name, so a name it doesn't know means every response is
filtered out and nothing is ever collected.

**This is not "no new posts", and the difference is not a judgement call.** A
healthy timeline always answers its operation at least once; even an account with
zero posts gets one response carrying an empty instruction list. Zero responses
therefore has exactly one meaning: the filter matched nothing. That is why the run
now fails closed instead of reporting a total.

Fix — find the name X is using and add it:

1. Open `x.com` in a browser, DevTools → Network, filter on `graphql`.
2. Scroll the timeline that's failing (your profile for `tweets`, `/i/bookmarks`
   for `bookmarks`) and read the operation name out of the request path.
3. Add it to `_OPERATIONS` in `src/xbrain/extract/extractor.py`, **newest first**,
   keeping the old names — X A/B-tests these and rolls them back:

   ```python
   _OPERATIONS: dict[str, tuple[str, ...]] = {
       "bookmark": ("Bookmarks",),
       "own_tweet": ("UserOriginalsTimeline", "UserTweets"),
   }
   ```

If instead the run reports **"0 nuevos items" and exits 0**, you are on a build
from before this was fixed, where the filter held a single literal per source and
a rename was silent — indistinguishable from an empty timeline, with the cursor
advancing over the gap. Update, then re-run.

## Getting rate-limited / the browser stalls

`extract` runs **headful** (visible Chromium) by default to look human, paces
itself, and backs off on `429`. If you still hit limits, wait and re-run — the
store is incremental, so you lose nothing. Don't run many extracts back-to-back.

## `parakeet-mlx` / `ffmpeg` not found (digest-video)

```
transcriber '.../xbrain-transcribe' exited 1: FileNotFoundError: 'parakeet-mlx'
```

The external tools aren't on `PATH`. Two cases:

- **Interactive shell:** install them (`brew install ffmpeg openai-whisper`,
  `uv tool install parakeet-mlx mlx-vlm`) and make sure `~/.local/bin` +
  `/opt/homebrew/bin` are on your `PATH`.
- **cron / launchd / a scheduled job:** these run with a **minimal PATH** that
  excludes `~/.local/bin` and `/opt/homebrew/bin`. Set the job's environment
  explicitly — e.g. in a launchd plist:

  ```xml
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/Users/you/.local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
  </dict>
  ```

  When testing a job, reproduce its env (`env -i HOME=$HOME PATH=... your-cmd`),
  not your shell — your shell's full PATH hides the bug.

If you use `scripts/xbrain-transcribe-auto`, it needs `ffmpeg` **and** `whisper`
on `PATH` to detect the language at all, plus `uv` for the GPU backend. A minimal
job environment that hides them makes the run fail outright rather than fall
through to parakeet — which is the safe direction, but it does mean a cron job
that used to "work" on an English corpus starts erroring once you switch to the
router.

## `digest-video` is slow or times out

Local vision (`--frames`) is the bottleneck: a slide-heavy talk can have up to
40 key-frames, and a local VLM reloads the model per frame. On a 16 GB Mac,
`qwen-7b` is ~2 min/frame → a long talk takes over an hour.

- **First run of a large model** can exceed the 300 s per-frame timeout while it
  *downloads* — pre-pull once: `~/.local/share/uv/tools/mlx-vlm/bin/python -c
  "from mlx_vlm import load; load('mlx-community/Qwen2.5-VL-7B-Instruct-4bit')"`.
- **Too slow overall?** Use a smaller model (`--vision-model qwen-3b`), or
  transcript-only (drop `--frames`), or cloud (`--vision-model opus`, needs
  `ANTHROPIC_API_KEY`).
- Frame extraction never hangs the run — ffmpeg is bounded by its own timeout.

## Every video comes back `fallidos` / `sin voz`

- `sin voz` (silent): the video has **no audio track** at the source (GIFs,
  muted screencasts). This is expected — it attaches as `has_speech=false`
  ("silent video"), not an error. Verify with `yt-dlp -f bestaudio <tweet-url>`
  (errors = no audio exists).
- `fallidos` (real failures): usually `parakeet-mlx` not found (see the PATH
  section above) — the fix is almost always the environment, not the video.

## A digest reads perfectly but says nothing that was said

Symptom: a video note's transcript and digest are fluent, confident English —
and bear no relation to the video. Nothing failed: `digest-video` reported the
video under **transcritos**, `enrich` summarised it, `video-digest` wrote a
readable digest of it, and every stage exited 0.

Cause: `parakeet-mlx` transcribed non-English audio. `parakeet-tdt` (v2 and v3)
is English-only and **does not fail** on other languages — handed Spanish audio
it invents fluent English and exits 0. There is no error anywhere in the run,
and by the time the text reaches your vault it is rendered as a quotation of the
speaker.

Fix: switch `[transcribe].command` to the router, then re-transcribe the affected
videos with `--force`:

```toml
[transcribe]
command = "/abs/path/to/xbrain/scripts/xbrain-transcribe-auto"
```

```bash
brew install openai-whisper       # the router's detector + its multilingual backend
uv run xbrain digest-video --ids <affected-ids> --force
uv run xbrain video-digest --executor claude-code    # re-digest against the real transcript
uv run xbrain enrich                                 # re-summarise it
```

The router detects the language on the first 30 s and sends English to parakeet,
everything else — and anything it cannot identify — to whisper. See
[Picking the transcriber](digest-video.md#picking-the-transcriber) for the
backends, the tuning env vars, and why it fails towards whisper.

There is no way to detect this after the fact from the transcript alone: a
fabricated transcript is well-formed. If you have run `digest-video` over a
multilingual corpus with plain `parakeet-mlx`, treat every non-English video's
text as unverified rather than trying to spot the bad ones.

## `generate` hangs or takes very long

If your vault is on **iCloud** with "Optimize Mac Storage" on, files can be
evicted to the cloud (dataless), and reading/writing them blocks on
re-download — worst at night with no activity. Run `generate` while the machine
is active, or keep the vault folder materialized (turn off Optimize Storage for
it). `data/items.json` already holds every digest, so a slow `generate` never
loses data — just re-run it.

## Do I need an API key?

No. The default execution mode (`vocab`/`enrich`/`topics`/`describe`) uses a
**Claude Code session** — no key, no cost — and `video-digest`/`verify` run **only**
on that keyless `claude-code` (or `manual`) track; they have no `api` track at all.
`ANTHROPIC_API_KEY` is only for `--executor api` on the first four stages (unattended
LLM runs) and cloud vision (`--vision-model opus`). `FIRECRAWL_API_KEY` is an optional
fallback fetcher for JavaScript-heavy pages.

`TYPESAFE_API_KEY` is the third, and the only one attached to a command that bills
per run: `xbrain jev topics`, the second opinion on topic assignment. Nothing in the
pipeline needs it — `jev` is a side-car you can ignore entirely — and unlike the other
two it may also be read from `<repo>/.env` (gitignored; `.env.example` is the committed
template). `xbrain jev report` and `xbrain jev dashboard` need no key at all: they
re-read what `jev topics` already paid for. See [jev.md](jev.md).

## `video-digest` / `verify` say "no pending" or "nothing to verify"

Both are **worksheet** stages (like `enrich`): the first run *exports* a worksheet,
you fill it, and a second `--apply` run consumes it. Common cases:

- **`video-digest` → "No hay vídeos pendientes de digest."** Every video already has
  a `digest`, or none has a transcript yet. Run `digest-video` first — a digest is a
  synthesis *of* the transcript, so there is nothing to digest without one.
- **`verify` → "No hay outputs que verificar."** There are no enrichment outputs for
  the chosen `--target`. Run `enrich` (and, for `--target digest`, `video-digest`)
  first — `verify` judges *existing* `summary` / `digest` / `topics`, it never
  generates them.
- **`--apply` did nothing / an empty report.** You applied the *exported* worksheet
  without filling its `judgments` array. Fill it (Claude Code session or by hand)
  between the export and the `--apply`. `verify --apply` takes **one worksheet per
  judge** — pass several with repeated `--apply` flags to aggregate them.

Both default to `[enrich].executor` and support only `--executor claude-code|manual`
(no `api` track).

## A note shows no verification badge, but I know it was judged

Symptom: you ran `verify … --write-verdicts`, the verdict is in `items.json`, and
`generate` renders the note with no ❌ / ⚠️ line under it.

Two causes, both deliberate:

- **The verdict was PASS.** A PASS is never badged — it would put a green line
  under most of the corpus and train you to ignore all of them. Only FAIL and
  REVIEW paint.
- **The verdict went stale.** A badge paints only while the stored
  `contract_fingerprint` still matches what would be computed today, and that
  fingerprint hashes **three arms**: the *output* under judgment, the *source the
  judge actually read* for that target (the evidence surfaces plus the
  not-fetched markers), and the *rubrics* it was judged by. Change any one and
  the verdict is retired silently.

The staleness rule is the point, not a bug: a verdict is not a property of the
output alone, it is the result of judging *that* output against *that* source
under *those* rules. So **an output fixed after a FAIL never shows a ❌** — which
is exactly what you want — but so does a FAIL whose article body arrived later,
or whose frame descriptions landed, or that was judged before the rubrics were
rewritten. A verdict stored before contract fingerprinting existed carries no
fingerprint at all and is stale by construction: it is retired, not
grandfathered.

Fix: nothing is broken, so there is nothing to repair — re-run `verify` to judge
the output under the contract in force now. When it exports a worksheet it tells
you how much of the layer has been retired, which is the number to watch after a
rubric change:

```
⚠️  N de M verdicts almacenados quedaron OBSOLETOS: se juzgaron bajo otro contrato
(otro output, otra fuente u otra rúbrica). No pintan badge; hay que re-verificarlos.
```

## `xbrain eval` fails to load or to resolve the golden set — and they are different faults

The golden-set loader has **two stages**, and the error tells you which one tripped. Reading
the wrong one sends you to the wrong file.

**`caso X: …` from `load_cases` — the FILE is wrong.** A structural problem, visible in
`eval/golden-set.yaml` itself without opening the corpus: an unknown stratum or filter, an
unfilled `<X>` template, a scorable case with an empty relevant set, an `expected_text` over
the 300-char ceiling, a topic pseudo-id left in `relevant_items`, a duplicate id. Fix the
YAML. This stage runs in CI, so a broken golden set is caught before it reaches anyone.

**`caso X: id relevante que no existe en el store` from `resolve_cases` — the CORPUS moved.**
The file is fine; an id it names is no longer in `data/items.json`. That is an error and not
a case scoring zero, deliberately: scoring it zero would blame retrieval for a stale ground
truth, and the number would look like a permanent regression that no change to retrieval
could ever fix. Either the item was removed, or the id was mistyped when the case was
enumerated. Re-verify it against the store and update the case — and record in `notes:` how
you verified it.

**"golden set no encontrado".** `xbrain eval` resolves `--golden-set` relative to the repo
root. Running it against another checkout's corpus needs both `XBRAIN_REPO_ROOT` pointing at
that checkout and `--golden-set` pointing at this one.

---

## `eval` reports a stratum as *sin cobertura* — is that a failure?

No, and the distinction is the point. Three different things are NOT a score of 0.0:

- **a stratum with no cases** (`expansion` has no mechanism until the graph exists);
- **a surface with no data** (`thread` and `user_note` have zero instances in the corpus, so
  no case can be written and none is invented);
- **a case whose filters the strategy cannot apply** — it is reported under *casos NO
  medidos*, naming the filters that blocked it, because a zero there would blame retrieval
  for an instrument that is not there.

Reporting any of these as 0.0 would say retrieval failed where nobody asked it anything. If
you want a gate, pass `--min-recall`. A bucket without coverage can never be NAMED as the one
that failed — that would be a verdict on a population nobody measured — but the gate is not
vacuous either: it counts the `(bucket, metric)` comparisons it actually made, and if that
count is **zero** it fails with *«el umbral … no se comparó contra nada»* instead of passing.
A threshold of 1.0 used to exit 0 over a golden set the baseline could not score at all.

**The third of those — a filter the strategy cannot push — has no live instance today.** The
harness builds its baseline through the same writer `xbrain index build` drives, so `lexical`
pushes all eight filters and both `filtros` cases are scored. It *used to* push only
`has_surfaces` and `origins`, and those two cases *were* the unmeasured ones — read that as
history, not as a current limit. The rule stays for Plan 03's vector strategy, which arrives
with no filter columns of its own, so a case reported unmeasured again is naming a real gap.
See [the filter question below](#xbrain-eval-reports-a-case-as-unmeasured-for-a-filter-that-search-applies).

You will also see a **`vacíos`** column. It counts the cases in that bucket whose query
retrieved **no chunk at all**, which is a different fault from "the right item ranked below
k" even though both leave `recall@k` at 0.0. When `vacíos == casos`, the retriever was never
given anything to rank; `precision@k` is then reported as *sin cobertura* rather than 0.0,
because its numerator is zero by construction and the figure would restate the empty set.

---

## The knowledge index

Everything below is about `data/index/` — the SQLite database `xbrain search`
reads. It is **derived and reconstructible**: nothing in it is yours, deleting it
loses nothing, and `xbrain index build` puts it back. Operating it day to day is
[docs/knowledge-index.md](knowledge-index.md).

### `Error: No hay índice en …/data/index. Constrúyelo con xbrain index build.`

The index has not been built, or you deleted it. Build it:

```bash
uv run xbrain index build
```

If the message instead says **`No hay base de datos en … pero su manifest sigue en
pie: el índice quedó incompleto`**, the database was removed and the manifest was
left behind — plain `build` would refuse (*Ya existe un índice*), so the command
it names is the forced one:

```bash
uv run xbrain index build --force
```

`xbrain get` is unaffected by any of this: it reads the live store, so it keeps
working with `data/index/` deleted.

### `Reconstruye el índice con xbrain index build --force`

Every incompatibility ends with that one sentence, and they are all the same
operator situation — *what is on disk is not what this code can read*. The line
before it says which:

| What the message says | What happened |
|---|---|
| `El índice fue construido con otra versión: schema_version '3' != '4'` — or a `surface_version`, `chunker_version` **or `chunker_params`** mismatch | you upgraded xbrain, or the chunker's parameters moved; the stored rows were cut and hashed by different code. **All four are checked, by every door**: a parameter change re-cuts every chunk under *identical* ids, so an index that only compared the version strings would answer over a corpus fingerprinted differently from the one it claims |
| `faltan las tablas …` / `faltan las columnas …` | the database is from an older schema, or was edited |
| `El manifest no declara …` / `El manifest declara …, que este código no conoce` | `manifest.json` was hand-edited, or written by another version |
| `El manifest no es un objeto JSON, es list` | `manifest.json` is corrupt |
| `La base del índice en … no se puede consultar (…)` | SQLite itself refused a read — a torn page, or not a database at all |

The fix is the same in every row, and it is cheap: a rebuild costs seconds to
minutes, not a re-fetch. **Nothing is queried partially** — a partial answer over
a schema this code no longer matches is a wrong answer wearing a right one's
shape, so the query refuses instead.

### `search` warns that the index is behind the store

```
⚠ El índice va por detrás del store: `items.json`, `vocab.yaml` o `topics.json`
  cambió después de construirlo. La evidencia puede estar obsoleta — actualiza
  con `xbrain index update`.
```

That is the expected warning after any stage that writes the store — `enrich`,
`topics`, `vocab`, `fetch`, `digest-video`. Indexing is manual, so run:

```bash
uv run xbrain index update
```

Two things worth knowing about the signal behind it. It compares `mtime` and size
of the three inputs, so **a `touch` with no edit trips it**: that is a false
positive by design, because the alternative failure — serving stale evidence as
fresh — is the one that matters. And it is blind in exactly one direction: a
replacement of the same size with the mtime preserved (`cp -p`, `rsync -a`,
`unzip`, a restored backup) is invisible to it, deterministically. After a
restore, run `xbrain index status`, which compares fingerprints rather than
timestamps.

### `index status` says *señal barata DESFASADA* but `0 cambiados`

Exactly the false positive above: a file was rewritten with identical content, or
merely touched. `index update` then reports `+0 nuevos · 0 cambiados · -0
borrados` and reseals the manifest, which is the cheapest way to silence it.

### `search` reports `corrupt_chunks_excluded: N`

`N` rows were **withheld, not served**. A chunk is dropped when its fingerprint
does not recompute over the row served beside it, or when its surface no longer
resolves to a locator. Both mean the same thing — this code cannot serve that row
honestly — and both are repaired by `xbrain index build --force`. Results still
come back; the count is how you know some did not.

### Results come back, but not the ones I expected

Three causes, in the order they actually bite.

**There is no stemming.** Singular and plural are different words. On one corpus
the top ten for `agente` and for `agentes` shared **zero** items. Search for the
form you expect to be written, or search for both.

**It is lexical, not semantic.** It matches proper nouns, figures and exact
phrases — not meaning. The response says so on every call
(`degraded: ["no_embeddings"]`). Conceptual similarity arrives with the vector
layer; until then, query with the words the author would have used.

**Accents are not the problem.** The tokenizer folds diacritics, so `atencion`
and `atención` return the same ranking. If the results changed, something else in
the query did.

### `--strategy vector` returned results — are they vector results?

No, and the response says so twice: `strategy` comes back as `lexical`, and
`degraded` carries `vector_not_implemented` beside a warning line —
*Estos resultados NO son de `vector`*. The vector backend is Plan 03. A
**misspelled** strategy is refused instead of degraded, listing the valid names,
because answering a typo with lexical results would turn it into a measurement.

### `xbrain eval` reports a case as unmeasured for a filter that `search` applies

**It no longer does, for the `lexical` strategy.** The harness used to walk the
corpus its own way and write chunks with no metadata, so it could push only
`has_surfaces` and `origins` and a case declaring a date, author, source or
content-kind filter came back unmeasured. It now builds through the same writer
`xbrain index build` drives, so it pushes the same eight filters `search` does.

If you still see it, the case is declaring a filter that the strategy you asked
for cannot push — which is the rule working, not a bug. **Unmeasured is never
`0.0`**: a zero from a filter nobody applied reads as "retrieval failed at
filtering" when the instrument was not there. See
[the stratum question above](#eval-reports-a-stratum-as-sin-cobertura--is-that-a-failure).

### An index error prints a second, empty `Error:` line

**Fixed — if you still see this, you are on an old build.** Both CLI error handlers
used to fire on an index error, so the real message was followed by a blank
`Error:`. `_handle_cli_errors` now re-raises `typer.Exit` instead of catching it, so
exactly one line is printed; the exit code was `1` before and after, which is why
nothing failed while it was wrong. Pinned by
`test_an_index_error_prints_exactly_one_error_line`.

The same change makes `download-videos` print Click's `Aborted!` when you answer `n`
to its size gate, where it used to print a bare `Error:`.

---

## `xbrain jev` — the second opinion on topics

Every failure of `xbrain jev topics|report|dashboard` is covered, message by message,
in **[jev.md § Troubleshooting](jev.md#troubleshooting)** — a missing or empty
`TYPESAFE_API_KEY`, an unimportable SDK, per-item `FALLO` lines, a run where every call
failed, the Ctrl-C checkpoint, a failed save that names what it cost you, an unreadable
side-car, and the five refusals that protect an existing report or page from being
overwritten with zeros.

Three that send people here first, because the symptom does not name Jev:

- **`0 evaluaciones vigentes de N guardadas` with a full side-car.** The vocabulary or
  the evidence moved, so the stored contracts no longer describe today's question. The
  records are not lost, they are retired; re-running `xbrain jev topics` re-asks them,
  and that is a re-bill. [Why](jev.md#staleness-when-an-assessment-stops-counting).
- **`xbrain snapshot restore` did not roll back my assessments.** It cannot: the
  side-car lives at `data/jev/topics.json`, one level below the four flat files a
  snapshot covers. A restore reverts **`vocab.yaml` as well as `items.json`**, and the
  contract hashes the vocabulary-derived questions digest — so a restore from before a
  `vocab --regenerate` moves the digest and retires **every record at once**, a full
  re-bill; only a restore that leaves both the item's evidence and the vocabulary
  untouched leaves an assessment current. Retired records report as `caducadas` on the
  next run. [Why](jev.md#where-the-files-live-and-what-protects-them).
- **A red banner on `jev.html`.** The page's own arithmetic disagrees with the report
  embedded in it. Trust `xbrain jev report`, not the page, and report it as a bug.

`data/jev/topics.json` is **paid, gitignored and never snapshotted**. There is no
`git checkout` and no `snapshot restore` back to a good copy — `--force` overwrites a
paid record with no recovery, and a corrupt file is repaired by hand or paid for again.

---

## Where's the source of truth? Can I delete the vault notes?

`data/items.json` is the hub — the markdown is **derived and disposable**.
Delete `items/`, `topics/`, `_index.md` and re-run `generate` any time. Every
destructive command auto-snapshots `items.json` first (see
[Snapshots & safety](../README.md#snapshots--safety)); restore from
`data/snapshots/` if needed.
