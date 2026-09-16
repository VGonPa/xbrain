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

**The third of those — a filter the strategy cannot push — has a live instance again, under
`vector` and `hybrid`.** `lexical` builds through the same writer `xbrain index build` drives,
pushes all eight filters, and scores both `filtros` cases (it *used to* push only
`has_surfaces` and `origins` — read that as history, not as a current limit). The vector plane
has **no filter columns**, so under `--strategy vector|hybrid` those two cases are reported
unmeasured. That is a real gap being named, not a zero.
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
(`degraded: ["no_embeddings"]` on an index without a vector plane). An optional
vector plane exists (`--strategy hybrid`), but it is not the default and no
candidate has been measured to beat lexical — [the bake-off](embeddings-bakeoff.md)
is incomplete. Query with the words the author would have used.

**Accents are not the problem.** The tokenizer folds diacritics, so `atencion`
and `atención` return the same ranking. If the results changed, something else in
the query did.

### I asked for `hybrid` and the response says `lexical` — are those hybrid results?

No, and the response says so: `strategy` is what RAN, not what you asked for. A
response names `hybrid` only when the vector channel actually ran; otherwise
`degraded` names why and not one match carries `vector` in `matched_by`:

| `degraded` carries | Cause | Fix |
|---|---|---|
| `embeddings_not_configured` | `[embeddings].command` is empty in `config.toml` | configure the embedder that built the plane |
| `embedder_unavailable` | the command did not answer: missing, not executable, non-zero exit, timeout or unusable output | run it by hand (below) |
| `no_embeddings` | the index was built without `--embeddings` | `xbrain index build --embeddings --force` |
| `vector_filters_unsupported` | you passed a filter; the vector plane has no filter columns | drop the filters, or read the lexical answer as lexical |

The first two also print a line starting *⚠ Pediste `hybrid` y ha respondido
`lexical`*. The last one prints only the bare flag. The full matrix, including what
`index build --embeddings` does in each case, is in
[docs/knowledge-index.md](knowledge-index.md#when-the-vector-channel-cannot-run).
A **misspelled** strategy is refused instead of degraded, because answering a typo
with lexical results would turn it into a measurement.

### `--strategy vector` is an error, not a result

Deliberately: you asked for vectors by name, and a lexical answer under that request
would be a lie about it. The message names what is missing:

- `` `--strategy vector` necesita vectores y este índice no tiene plano vectorial… `` —
  the index was built without a plane. Configure `[embeddings].command`, then
  `uv run xbrain index build --embeddings --force`.
- `` `--strategy vector` necesita embeber la consulta y `[embeddings].command` está vacío… `` —
  the plane exists and the command does not. Set it to the embedder that built the plane.
- an `embedder '…'` message — the backend is down; see the sections below.

The one case where `vector` does answer lexically is a **filtered** request
(`--topic`, `--from`, …): `degraded` carries `vector_filters_unsupported`.

### `embedder '…' not found` / `could not be executed`

```
embedder '/path/to/xbrain-embed' not found — install it or set a valid [embeddings].command in config.toml (…)
```

The path in `[embeddings].command` does not exist, or is not executable. The command is
split with `shlex` and run without a shell, so `~`, `$VAR` and pipes are not expanded:
use absolute paths. `index build --embeddings` fails here, on its probe batch, before
writing anything; `search --strategy hybrid` answers lexically with
`embedder_unavailable`.

A common variant: the command names `scripts/xbrain-embed` alone. Its shebang is
`#!/usr/bin/env python3`, so it runs whichever `python3` comes first on `PATH`, which is
usually not the one with `sentence-transformers`. Name the interpreter explicitly —
`command = "/path/to/embedder-env/bin/python /path/to/xbrain/scripts/xbrain-embed"` —
as the [bake-off's §9](embeddings-bakeoff.md#9-cómo-re-derivarlo) does.

### `embedder '…' exited N — its stderr is not repeated here`

The backend crashed, and xbrain **deliberately withholds its stderr**: a Python
traceback prints the `repr` of the text it failed on, and during a build that text is
your corpus — which would end up in a terminal, a log or a pasted issue. Reproduce it
with a request you wrote yourself, so the output is safe to read:

```bash
printf '%s' '{"schema_version": "1", "model": null, "texts": ["hola"]}' \
  | /path/to/embedder-env/bin/python /path/to/xbrain/scripts/xbrain-embed
```

The reference wrapper exits with a one-line reason: no `sentence-transformers` in that
interpreter, a request that is not JSON, a schema version it does not speak.

### `embedder '…' timed out after Ns`

A single subprocess call exceeded `[embeddings].timeout_seconds` (default 600). The
first call of a model that is not cached yet can download its weights inside that
window; so can a machine that is swapping, which is what stopped the bake-off. Raise
`timeout_seconds`, or lower `batch_size` so each call does less.

During `index build --embeddings` a timeout is fatal to the build (next section). At
query time, `hybrid` answers lexically with `embedder_unavailable` and `vector` is an
error.

### After a failed `index build --embeddings`, even lexical `search` refuses

Expected, and the refusal names the fix. There is **no vector-only rebuild**:
`--embeddings --force` deletes the previous manifest and database first and re-derives
the lexical plane from the store, so once the embedder fails mid-build there is no index
left to answer from. A fresh build that fails leaves the lexical rows committed and no
manifest, and an index with no manifest is refused by every door. Get lexical search back
at once with

```bash
uv run xbrain index build --force
```

and retry `--embeddings` once the embedder answers a hand-made request.

### `models are never mixed` / another model, same dimension

```
embedder returned dimension 384, but this index holds 768-dimensional vectors — models are never mixed in one matrix; …
el embedder sirvió el modelo '…' a mitad del build, y el sondeo declaró '…': jamás se mezclan vectores de dos modelos …
el embedder de `[embeddings].command` sirve el modelo '…' y este plano vectorial se escribió con '…' …
```

The backend is serving a different model from the one that wrote the plane (or, mid-build,
from the one its own probe declared). A matching dimension is not accepted as proof: two
models of one width produce cosines that look exactly as healthy as real ones. So this is a
hard error under **both** `vector` and `hybrid` — never a lexical fallback, because a backend
serving another model is not a backend that is down.

`search` asks the backend for the model **recorded in the manifest**, not the one in
`config.toml`. Either make the backend serve that model, or rebuild with the new one:
`uv run xbrain index build --embeddings --force`.

### `no hay plano vectorial en …` / `El manifest declara un plano vectorial que no está en …`

The manifest declares a vector plane, and `vectors.f32` or `vectors.meta.json` is gone. A
`vector` or `hybrid` query refuses rather than answering over half an index, and does not pay
to embed the query first; `index status` reports the state as `missing`. The mirror case —
`Hay ficheros de plano vectorial … que el manifest no declara` — is files left beside a
manifest that declares none (`undeclared`). Both are fixed by
`uv run xbrain index build --embeddings --force`.

### `El plano vectorial no cubre el corpus indexado: N fragmentos sin el vector de su texto actual…`

You ran `index update` over an index with a vector plane. `update` rewrites the chunks that
moved and **never re-embeds** them, so the plane is `behind`. Lexical search is fully current.
`vector` and `hybrid` still run over the chunks the plane does cover, skip every stale vector,
and declare `vector_plane_behind`. Restoring full coverage is a rebuild of both planes:
`uv run xbrain index build --embeddings --force`.

### A query needing vectors names `uv pip install 'xbrain[embeddings]'`

`numpy` is not installed: it is the optional `[embeddings]` extra, not a dependency. From a
checkout, install it with `uv sync --extra embeddings` (add `--extra dev` if you also run the
quality gate). `uv sync` removes what its flags did not ask for, so a later `uv sync` without
`--extra embeddings` uninstalls it again — the usual reason this error comes back. `index
status` does not refuse over a missing `numpy`; it reports the plane as `unreadable` with this
same sentence.

### `xbrain eval --strategy vector` refuses before measuring anything

`--strategy vector|hybrid` measures one embedding model and needs `--embeddings-model <model>`;
`--embeddings-model` beside `lexical` is refused too, since no number in that report would come
from it. `--sweep-fusion` is `hybrid`-only. All three refusals happen before anything is loaded or
embedded. The bake-off procedure — one `XBRAIN_REPO_ROOT` per candidate, what the run builds under
`data/eval-index/<model>/` — is [docs/embeddings-bakeoff.md §9](embeddings-bakeoff.md#9-cómo-re-derivarlo).

### `xbrain eval` reports a case as unmeasured for a filter that `search` applies

**It no longer does, for the `lexical` strategy.** The harness used to walk the
corpus its own way and write chunks with no metadata, so it could push only
`has_surfaces` and `origins` and a case declaring a date, author, source or
content-kind filter came back unmeasured. It now builds through the same writer
`xbrain index build` drives, so it pushes the same eight filters `search` does.

If you still see it, the case is declaring a filter that the strategy you asked
for cannot push — today that means `--strategy vector` or `hybrid`, whose plane has no
filter columns — which is the rule working, not a bug. **Unmeasured is never
`0.0`**: a zero from a filter nobody applied reads as "retrieval failed at
filtering" when the instrument was not there. See
[the stratum question above](#eval-reports-a-stratum-as-sin-cobertura--is-that-a-failure).

### The graph: an index built before it existed is refused

The graph plane (`graph_edges`) changed the database layout, so an index sealed by
an earlier xbrain is not updated in place: `index update` and the query doors end
with `xbrain index build --force` ([above](#reconstruye-el-índice-con-xbrain-index-build---force)).
Rebuild once and the graph is there. `index status --json` shows it in the
manifest's `graph` block (`algorithm_version`, the three thresholds, `edges`).

### `graph-expand` refuses: `El índice va por detrás del store (index_behind_store)`

`search` answers over an index behind the store and declares it; `graph-expand`
refuses, because its edges may describe a corpus that no longer exists. Run
`uv run xbrain index update`. The same fix applies to
`La arista … se apoya en items que ya no están en el store`: an edge whose
listed support left the store makes the whole expansion refuse rather than serve a
path nothing supports.

### `graph-expand` returns one node and no paths

Either the item has no topics (`enrich` never assigned any), or **the id does not
exist**: today both come back as exit 0 with a single `item:<id>` node, and the two
cannot be told apart from the output (backlog). Check the id with
`uv run xbrain get <id>`, which refuses an unknown one. If the item exists and has
topics, `--max-hops 2` is what reaches the co-occurring topics.

### I changed a `[index].graph_*` threshold and nothing happened

The edges are written by `index build` / `index update`, not read from
`config.toml` at query time. Run `uv run xbrain index update`: it rewrites the
graph plane when the thresholds differ from the ones the manifest sealed, and
leaves the rest alone. **`index status` does not flag this state yet** — it reports
nothing behind while `update` would rewrite the edges (backlog).

### I asked for `hybrid_graph` and the response says `lexical`

```
$ uv run xbrain search "harness engineering" --strategy hybrid_graph
"harness engineering" · estrategia lexical
⚠ La estrategia `hybrid_graph` no tiene backend todavía: ha respondido `lexical`. Estos resultados NO son de `hybrid_graph`.
```

Expected: the graph re-ranking is **off by default**, and neither `xbrain search`
nor MCP can switch it on — nor can `xbrain eval --strategy hybrid_graph`, which
answers the same way (`strategy: lexical`, `hybrid_graph_not_implemented`). It is
not missing: the Python API runs it (`search(..., graph_enabled=True)`), and so
does the one command built on that call, the threshold sweep
`xbrain eval --strategy hybrid_graph --sweep-graph …` — a single cell
(`--sweep-graph "min_shared_items=5 min_weight=0.05"`) measures the applied
threshold. It is off because the measured result is negative: in all 16 threshold cells
it made `recall@10` worse and lifted none of the 33 results only the graph could
reach ([graph-threshold-sweep.md](graph-threshold-sweep.md)). The sentence says
"no tiene backend todavía", which is stale wording for "switched off" (backlog).

### `xbrain mcp-serve` does not start, or the client sees no tools

- **`Error: … necesita el extra opcional [mcp]`** — the SDK is not installed in
  the environment the client launches. Launch through
  `uv run --directory /path/to/xbrain --extra mcp xbrain mcp-serve`, or install the
  extra ([docs/mcp.md](mcp.md#before-you-connect)).
- **The client reports the server failed or exited.** Run the exact command the
  client runs in a terminal: it should start and wait silently on stdin (Ctrl-C to
  stop). Anything printed before that is the cause.
- **Tools listed, `search` and `graph_expand` error with `No hay índice en …`** —
  the server is reading a checkout without an index (`get` still answers: it reads
  the store, never the index). It reads `config.toml` and `data/` from the
  checkout it runs from (or from `XBRAIN_REPO_ROOT`); build the index there.
- **Results from the wrong corpus** — same cause: check `--directory` and
  `XBRAIN_REPO_ROOT` in the client's configuration.

### An MCP tool answers `Error executing tool xbrain.… : …`

That is an operator error, and the text after the colon is the same message the
CLI prints for the same request — follow it as you would on the command line
(`xbrain index build`, `xbrain index update`, a valid item id, a declared
strategy). A bare `Error executing tool xbrain.search` with nothing after it is a
bug, not a configuration problem: report it with the arguments you sent.

### An index error prints a second, empty `Error:` line

Cosmetic, and known. Both CLI error handlers fire on an index error, so the clean
message is followed by a blank one. The exit code is still `1` and the first line
is the real one.

---

## Where's the source of truth? Can I delete the vault notes?

`data/items.json` is the hub — the markdown is **derived and disposable**.
Delete `items/`, `topics/`, `_index.md` and re-run `generate` any time. Every
destructive command auto-snapshots `items.json` first (see
[Snapshots & safety](../README.md#snapshots--safety)); restore from
`data/snapshots/` if needed.
