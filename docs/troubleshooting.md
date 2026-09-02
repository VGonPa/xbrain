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
- **a case whose filters the strategy cannot apply** — the lexical baseline has no date,
  author, source or content-kind columns, so the two `filtros` cases are listed under *casos
  NO medidos* with the filters that blocked them.

Reporting any of these as 0.0 would say retrieval failed where nobody asked it anything. If
you want a gate, pass `--min-recall`. A bucket without coverage can never be NAMED as the one
that failed — that would be a verdict on a population nobody measured — but the gate is not
vacuous either: it counts the `(bucket, metric)` comparisons it actually made, and if that
count is **zero** it fails with *«el umbral … no se comparó contra nada»* instead of passing.
A threshold of 1.0 used to exit 0 over a golden set the baseline could not score at all.

You will also see a **`vacíos`** column. It counts the cases in that bucket whose query
retrieved **no chunk at all**, which is a different fault from "the right item ranked below
k" even though both leave `recall@k` at 0.0. When `vacíos == casos`, the retriever was never
given anything to rank; `precision@k` is then reported as *sin cobertura* rather than 0.0,
because its numerator is zero by construction and the figure would restate the empty set.

---

## The knowledge index

Every error below is deliberately actionable: spec §9.3 requires the message to name the
command that fixes it, and none of them ever repairs anything on its own — a query that
silently repaired the index would be a query whose results depend on when you ran it.

**`No hay índice en data/index. Constrúyelo con xbrain index build`** — `search` needs the
index; `get` does not. The index is derived and reconstructible, so this is never data loss:
`xbrain index build` takes under two seconds on a 2,400-item corpus.

**`El índice fue construido con otra versión: … Reconstruye el índice con xbrain index build
--force`** — you upgraded xbrain and the surface emitter, the chunker or its parameters moved.
The query is refused **entirely** rather than answered partially, because a partial answer over
the wrong version is a wrong answer wearing a right one's shape. `xbrain index update` refuses
for the same reason: an incremental update under a new chunker would leave the corpus half cut
one way and half the other, with every id still resolving, so nothing would raise and the
ranking would quietly become a blend of two chunkers.

**`⚠ El índice va por detrás del store`** (`degraded: ["index_behind_store"]`) — you ran
`enrich`, `topics`, `vocab`, `digest-video` or `fetch` and did not reindex. The answer is still
given — possibly-stale evidence is usable as long as it says so — but run `xbrain index update`.
This is the failure that actually happens, because indexing is manual by decision. `xbrain index
status` tells you **how many** items changed. The signal is the mtime+size of **all three** inputs
— `items.json`, `vocab.yaml`, `topics.json` — since round 05 (P1a): until then only the first was
watched, so a `topics` or `vocab` run, which never touches `items.json`, left `search` answering
over the old topic plane with nothing declared. A `touch` on any of the three with no edit also
trips it: a false positive costs one warning, a false negative costs serving stale evidence as
fresh. **After upgrading across round 05 every `search` says this once**: a manifest written
before it carries no vocab/topics signal, which compares unequal until the next `index update`
re-seals it.

**`N topics con miembros desfasados`** from `index status` (`topics_changed` in `--json`) — the
topic rows in the base do not list the members the store implies now, or their `stale` bit is
wrong. Two ways in: an `enrich` moved items between topics and nobody reindexed (then
`items_changed` is non-zero too), or the index was last updated by a version before round 04,
which rewrote `item_topics` on a topic move and never the `topics` rows (then every item
fingerprint matches and only this count is non-zero — H1). Either way `xbrain index update`:
it rewrites exactly those rows and reports them as `N topics con miembros recalculados`.

**The continuation `get` printed returned an empty page, or `Cursor inválido: 'q:N' es un cursor
de --query`.** A cursor is an offset into a sequence, and the sequence is defined by the
`--surface`s and the `--query` of the call that produced it: resuming without them lands
inside the default selection (an empty page) or is refused by the decoder. Since round 04 the
printed line repeats both (`xbrain get ID --surface … --query '…' --cursor C`, H2); if you typed
the continuation by hand, pass the same `--surface` and `--query` as the first call.

**`⚠ N chunk(s) excluido(s): su fingerprint no cuadra con su texto, procedencia, autor o
localizador`** — a row whose fingerprint does not recompute over what the index serves about it:
its text, or — since round 07 (U-5) — its `origin`/`trust_class`/`surface_type`, its owner, its
position, or the attribution and locator of its surface row; or — since round 06 (B-k) — a row
whose surface holds no locator at all. A hand-edited database, or a row written by a different
chunker. Those chunks are **not returned** and are counted in `corrupt_chunks_excluded`, never
silently dropped and never repaired mid-query. `xbrain index build --force`. **After upgrading
across round 07 every door refuses the old base by name** (`schema_version 2 != 3`), because
every fingerprint of a v2 base was computed over the text alone: one forced rebuild.

**`Índice: INCOMPLETO o inexistente`** from `index status` — a build was interrupted. The
manifest is written last, precisely so an interrupted build leaves nothing a query will trust —
and a forced rebuild removes the previous manifest **first**, so an interrupted `--force` cannot
leave the old manifest standing over an empty base (C-1). Rebuild with `xbrain index build`.

**`⚠ Índice INCOMPLETO o inutilizable`** from `index status`, with a manifest present — the base
does not hold what its manifest declares (`topics 0 != 45`), or the manifest is from another
schema/emitter/chunker version, or — since round 05 (B-1) — `PRAGMA quick_check` found damage
(`La base del índice … está dañada (quick_check: …)`): `status` is the explicit command that pays
for the whole-file check (155–167 ms by the gate's measurement, a 425 ms median at load 8.6 in
round 05, on the 52 MB real index), so a corrupt page no query has
touched yet is declared here before a query hits it and fails closed. `index update` refuses the
same states instead of rewriting every item over them (C-3) — and since round 06 it runs the
same `quick_check` before re-sealing the manifest (D-1). The advice names `xbrain index build
--force`, and it has to: plain `index build` refuses while a manifest exists.

**`El manifest tiene el campo 'counts' malformado: …`** (also `skipped`, `chunker_params`,
`store_signal`, `embeddings`, `failed`, `built_at`) — the manifest parses as JSON but does not
carry the whole nested schema: a plane, cause or parameter missing, an unknown key, a count that
is a string, a boolean or negative. Until round 06 (B1) such a document loaded as compatible and
a manifest with `counts: {}` compared nothing, so `status`, `search` and `update` all agreed an
amputated base was healthy. `xbrain index build --force`; nothing is repaired in place.

**`La base del índice en … no se puede leer (database disk image is malformed)`** from `index
status`, `search` or `index update` — a damaged page that a maintenance read touched (the root
page of `items`, `chunks` or `topics`). Until round 06 (D-1) this was a raw `sqlite3.DatabaseError`
traceback on the three commands. `xbrain index build --force`.

**A search does not find an accent.** It should: the tokenizer is `unicode61 remove_diacritics
2`, so `evaluacion` reaches `evaluación`. What it will **not** do is match `agent` to `agents` —
there is no stemming, deliberately, because FTS5 has no multilingual stemmer and the English one
would wreck the Spanish half of the corpus. See
[`docs/knowledge-index.md`](./knowledge-index.md#what-the-lexical-baseline-cannot-do).

**`search` returns a summary with `no_underlying_source`.** Not a bug: the match landed on
derived text (a summary, a digest, a topic note) and that item keeps no primary source to check
it against. It is NOT common: measured 2026-09-01 on the real corpus (2,404 items, sha256
`f76341a3…`), **0 items** have no evidence-class surface — every item keeps its `post`, and a post
is a primary source — so the warning is a contract guarantee, not a frequent state. *(This
paragraph used to say "40 % of items have no `content`, so this is common": that is the F-5
confusion of two populations, items without a `content` block (960) and items without any
primary surface (0), left uncorrected here in round 01.)* The JSON says it as `verify_with: []`,
which is reachable in exactly that one case.

## Where's the source of truth? Can I delete the vault notes?

`data/items.json` is the hub — the markdown is **derived and disposable**.
Delete `items/`, `topics/`, `_index.md` and re-run `generate` any time. Every
destructive command auto-snapshots `items.json` first (see
[Snapshots & safety](../README.md#snapshots--safety)); restore from
`data/snapshots/` if needed.
