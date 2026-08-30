# `digest-video` — turn bookmarked talks into readable notes

`digest-video` manufactures **text** from a video so it flows through the normal
enrich → topics → generate pipeline like any other post. For each selected video
it does an **ephemeral** fetch, transcribes the audio with an external local
transcriber, attaches the transcript as an `x_video` content source, and
**discards the bytes** (the corpus never lands on disk). `--frames` adds a visual
layer: it extracts the slide key-frames and describes each with a vision model.

## Prerequisites

The heavy lifting is **external** — xbrain core carries no ML/ffmpeg dependency.
Install once (see [Local models for `digest-video`](../README.md#local-models-for-digest-video-apple-silicon)):

```bash
brew install ffmpeg                # frame extraction, audio probe, language detection
brew install openai-whisper        # multilingual ASR + the router's language detector
uv tool install parakeet-mlx       # fast English-only ASR (Apple Silicon)
uv tool install mlx-vlm            # vision, only for --frames
```

and point `config.toml` at the wrappers:

```toml
[transcribe]
# Detects the language, then picks the ASR. See "Picking the transcriber" below.
command = "/abs/path/to/xbrain/scripts/xbrain-transcribe-auto"

[vision]
command = "/abs/path/to/xbrain/scripts/xbrain-vision"       # local + cloud selector
model   = "qwen-7b"
```

Leave `[transcribe].model` unset unless you know which backend will run: the
router picks the backend, and each one resolves its own default model.

If your corpus is **English only**, you can skip `openai-whisper` and point
`[transcribe].command` straight at `scripts/xbrain-transcribe` (the parakeet
wrapper). It is the faster path. On anything else, read the next section before
you run: the wrong choice here does not fail, it fabricates.

## Picking the transcriber

xbrain never transcribes anything itself: it shells out to whatever
`[transcribe].command` names. The repo ships three backends and a router to
choose between them, and the choice matters more than it looks.

### Why `parakeet-mlx` on its own is a data-loss path

`parakeet-tdt` (v2 and v3) is **English-only, and it does not fail on other
languages.** Handed Spanish audio it exits 0 and emits fluent, broken English
that never reproduces what was said — verified 17-jul-2026 against an es-ES TV
clip, and the reason `scripts/xbrain-transcribe-auto` exists.

This is the worst failure mode available here. A crash shows up in a log; this
one passes the entire pipeline. The invented transcript attaches as an `x_video`
source, `enrich` summarises it, `topics` files it, `video-digest` writes a
readable digest *of* it, and it lands in your vault rendered as a quotation of
the speaker. By the time you read it, nothing on the page distinguishes "the
video said that" from "the ASR made it up".

So the backend has to be chosen **before** transcription, not audited after.

### The router — `scripts/xbrain-transcribe-auto`

Point `[transcribe].command` at it and it decides per video:

1. `ffprobe` short-circuits a genuinely **silent** clip (no audio stream) into
   the empty-speech JSON — `has_speech=false`, not a failure — before anything
   else runs.
2. `ffmpeg` slices the first **30 s** of audio; `whisper` transcribes that slice
   with **no** `--language`, so it reports the language it detected.
3. **English** (`en`) goes to `xbrain-transcribe` → parakeet-mlx, the fast path.
   **Anything else, and anything it could not identify**, goes to whisper: it
   tries `xbrain-transcribe-mlx` (mlx-whisper on the Apple GPU, pulled on demand
   through `uv run --with`) and falls back to `xbrain-transcribe-whisper` (the
   brew `whisper` CLI, CPU, portable) when the GPU backend cannot run.

| Backend | Wrapper | Languages | Notes |
|---------|---------|-----------|-------|
| parakeet-mlx | `xbrain-transcribe` | **English only** | fastest; fabricates on anything else |
| mlx-whisper | `xbrain-transcribe-mlx` | multilingual | Apple GPU; needs `uv` |
| whisper (brew) | `xbrain-transcribe-whisper` | multilingual | CPU; works anywhere, slow |

The two whisper backends were verified to produce a **character-identical**
transcript on a real clip, so the GPU→CPU fallback costs accuracy nothing and
only costs time. The wrapper records the measurement behind that ordering: on
one 68-second Spanish clip (M-series Mac, 12-ago-2026) the CPU CLI took
**7 min 25 s** and mlx-whisper **17.7 s**.

**It fails towards whisper, always.** Missing `ffmpeg`, a `whisper` that errors,
an unreadable result — every one of those returns "undetected", and undetected
routes to whisper. The costs are asymmetric in exactly one direction: whisper on
English is slower than it needed to be, while parakeet on non-English is a
fabricated transcript.

One consequence worth knowing: **detection needs the `whisper` CLI on `PATH`.**
Without it nothing can ever be detected, so every video takes the multilingual
path — you lose the parakeet fast path silently rather than noisily. That still
transcribes correctly *provided the mlx backend can run*, because the CPU
fallback is that same missing `whisper` binary: with neither `uv`/mlx-whisper nor
`whisper` available the run fails outright. Which is the right direction to fail
in, but it is a failure, not a slow success.

Both whisper wrappers also discard whisper's canned **no-speech artefacts** —
when the *whole* transcript is one of the subtitle-boilerplate lines the model
emits on silence (`Subtítulos realizados por la comunidad de Amara.org`,
`Thanks for watching!`, `[Música]`), it is recorded as no speech instead of as
something a person said. Matching is on the entire transcript, so a video that
merely *mentions* Amara or thanks its viewers mid-sentence is untouched.

### Tuning

All optional environment variables, all read by the router:

| Variable | Default | What it does |
|----------|---------|--------------|
| `XBRAIN_ASR_DETECT_MODEL` | `base` | whisper model used for the detection pass |
| `XBRAIN_ASR_DETECT_SECONDS` | `30` | seconds of audio sampled to detect |
| `XBRAIN_ASR_FORCE` | *(unset)* | `parakeet` or `whisper` — skip detection entirely |
| `XBRAIN_TRANSCRIBE_PARAKEET` | sibling script | path to the parakeet wrapper |
| `XBRAIN_TRANSCRIBE_MLX` | sibling script | path to the mlx-whisper wrapper |
| `XBRAIN_TRANSCRIBE_WHISPER` | sibling script | path to the CPU whisper wrapper |

Detection costs one small-model pass over 30 seconds per video. The wrapper's
own timing (M-series, 12-ago-2026, one es-ES and one en-US clip) puts `base` at
~19 s per clip and `tiny` at ~10 s, with both models identifying both clips
correctly. `base` stays the default: two clips are not enough evidence to trade
accuracy for nine seconds on the one axis where being wrong means a fabricated
transcript. Reach for `XBRAIN_ASR_DETECT_MODEL=tiny` on a large backlog, where
the halving actually adds up.

`XBRAIN_ASR_FORCE=parakeet` skips detection for a run you *know* is English —
and re-introduces the fabrication risk for every clip in it that isn't.

## Run it

```bash
# Transcript only (no vision, no ffmpeg-frames) — fast:
uv run xbrain digest-video --all-pending

# → Vídeos: transcritos 6, sin voz 2, ya digeridos 0, fallidos 0, sin vídeo 1, ...
#   Dedup: 9 items ← 9 vídeos (6 transcritos este run).
```

Read the summary: **transcritos** = had speech, **sin voz** = silent (no audio
track — GIFs, muted clips; attached as `has_speech=false`, not a failure),
**fallidos** = a real transcribe failure, **sin vídeo** = the video couldn't be
fetched (deleted / unavailable). Videos are **deduped by identity** — N bookmarks
of the same clip are fetched + transcribed once.

Add `--frames` for slide-heavy talks:

```bash
uv run xbrain digest-video --all-pending --frames
# → ... Visual: 5 con slides, 4 talking-head (saltados).
```

`--frames` extracts key frames (ffmpeg scene-detection + interval sampling),
classifies the video as **slides** vs **talking-head** (talking-heads are skipped
— no vision calls wasted), and describes each slide of a slide video. The slide
images are embedded in the note like downloaded photos.

Then build the readable digest and render:

```bash
# Turn the transcript (+ frames) into a long-form readable digest — a worksheet
# flow, like enrich: export a worksheet, fill it (Claude Code or by hand), apply.
uv run xbrain video-digest --executor claude-code
uv run xbrain video-digest --apply data/video-digest-worksheet.json

uv run xbrain generate
```

## What you get

Once you've run [`video-digest`](#run-it), the item's note **leads with the readable
digest** as the headline of its `## Video digest` section; the raw transcript + slide
frames are demoted into a collapsible block below it:

```markdown
## Video digest: Elon Musk on the first thing to do when starting a company

Musk's one rule for a new company: build something people love enough to tell
their friends — advertising can't rescue a product nobody recommends. He traces it
to Tesla's early bet on proving what electric cars could actually do… (readable digest)

<details>
<summary>Frames + transcript</summary>

Uh, the goal with Tesla was really to try to show what electric cars can do,
because people had the wrong impression… (full transcript, rendered raw)

![[_media/1874.../frames/0.png]]
> Slide: a line chart of Model S range vs. price, 2012–2015.

</details>
```

The readable digest is produced by [`video-digest`](#run-it) — not `digest-video`,
which only attaches the raw transcript + frames. **Before** you run it (or for a
video with no digest yet) the section falls back to the **old inline layout**
(transcript then frames, no `<details>`), so the render is safe either way. The
transcript + slide descriptions are plain note text, so they feed `enrich` (summary
+ topics) and are **searchable** in Obsidian. A silent video with no slides degrades
gracefully to a one-line "silent video" note.

## Choosing the model, per run

`config.toml` `[vision].model` is the default; `--vision-model` overrides it for
one run. The `scripts/xbrain-vision` selector routes the name:

| `--vision-model` | Backend | Notes |
|------------------|---------|-------|
| `qwen-3b` / `qwen-7b` / `qwen-32b` / `<hf/repo>` | local (mlx-vlm) | free, offline; `qwen-32b` needs ~20 GB RAM |
| `opus` / `sonnet` / `haiku` / `claude-<id>` | cloud (Claude) | best quality; needs `ANTHROPIC_API_KEY`; frames leave the machine |

```bash
uv run xbrain digest-video --ids <slide-heavy-id> --frames --vision-model opus
uv run xbrain digest-video --topic ai-coding      --frames --vision-model qwen-7b
```

## Selecting which videos

```bash
--ids a,b,c        # specific item ids
--topic ai-coding  # every video whose post is in that topic
--all-pending      # every not-yet-digested video (idempotent; re-runs skip done ones)
--source bookmarks|tweets|all   --limit N   --language en
```

`digest-video` is destructive (rewrites `items.json`) → it auto-snapshots first.
Re-running skips videos already carrying an `x_video` source unless `--force`.

Slow? See [Troubleshooting → digest-video](troubleshooting.md#digest-video-is-slow-or-times-out).
Digest reads fluently but doesn't match the video? →
[Troubleshooting](troubleshooting.md#a-digest-reads-perfectly-but-says-nothing-that-was-said).

## Fixing stale captions — `redescribe-frames`

The rubric behind `--frames`' captions can itself change after you've already
captioned a corpus — it did once: captions were found to be **translating**
on-screen text (slide labels, code identifiers, chart axis labels) into the
output language instead of transcribing it verbatim. Whoever later cites a
label needs to match it against what is actually printed on screen, and a
translated label breaks that match.

When the rubric changes like that, you do **not** need to re-fetch the
videos. The frames it would re-describe are already on disk at
`data/media/<id>/frames/`, so `redescribe-frames` re-describes those bytes
directly — **zero network, zero ffmpeg, zero X:**

```bash
uv run xbrain redescribe-frames --dry-run   # free: no model call, no [vision].command needed
uv run xbrain redescribe-frames             # the real re-caption pass
```

`--dry-run` costs nothing at all: it doesn't require `[vision].command` to be
configured, and it never calls the vision model — it just checks which frame
images actually exist on disk, so it can tell you a frame whose PNG is gone
will fail without spending a single vision call to find that out. It cannot
predict which captions will *change*, since only the model calling on the real
text can tell you that.

**It skips anything already re-captioned.** Every video's frames remember the
caption contract they were described under, so re-running the command on an
already-current corpus costs zero vision calls; `--force` re-describes
everything regardless. Select a narrower slice the same way as `digest-video`:
`--ids a,b,c` (an id absent from the store is warned about, not silently
dropped), `--topic <t>`, `--source bookmarks|tweets|all`, `--limit N`. With
**none** of those set, the command targets every stale video in the corpus —
deliberately, unlike `digest-video`'s selectors, because this is a one-shot
corpus-wide repair rather than an incremental pipeline stage you run
regularly.

**A permanently-missing frame image is a real, recurring cost — know this
before you rely on it for a large backfill.** A video's frames are marked
"re-captioned" only once *every one* of them re-describes successfully. If
even one frame's PNG is gone for good, that video never gets marked done:
every future `redescribe-frames` run re-describes its surviving frames all
over again — and because a real vision model is not deterministic, those
captions come back reworded even though the pixels never changed, which
re-triggers `enrich` and `video-digest` for that item on **every single run**,
not just once. This is the price of the all-or-nothing design (it is what
lets a partially-broken video keep retrying instead of being marked "done"
with a stale caption baked in), but it means a video stuck this way is not a
one-time cost — it is an ongoing one, for as long as it stays selected. On the
corpus that motivated this command, that was 2077 frames across 142 videos,
some of whose source videos are no longer downloadable at all.

**This fixes translation, not an OCR misread.** `redescribe-frames` re-runs
the *rubric* over the same pixels — it fixes a mistranslated or paraphrased
label. It cannot fix a label the model genuinely could not read: the stored
frames are downscaled to 640px wide at extraction time, and that resolution
is not recoverable from the PNGs on disk. If a caption is wrong because the
frame itself is illegible, the fix is re-extracting the frame — `digest-video
--force --frames` — not `redescribe-frames`.

It is destructive (rewrites `items.json`) → auto-snapshots first, but only
when at least one caption actually changed.

### The photo half needs a manual step

`redescribe-frames` only covers **video key frames**. The same on-screen-text
rule was also added to the rubric `xbrain describe` uses for tweet photos, and
that half does **not** self-invalidate: `describe` decides whether a photo is
stale by comparing its stored `description_version` against the hand-set
`[describe].version` tag in `config.toml` — never by looking at whether the
rubric's *content* changed. So after upgrading, if you want already-described
photos to lose their old, possibly mistranslated captions, you have to say so
explicitly:

```toml
[describe]
version = "v2"   # was "v1" (or whatever you had it at)
```

```bash
uv run xbrain describe
```

Without that bump, every photo `describe` already ran stays exactly as it
was, forever — the rubric changed under it and nothing re-reads it. The
default in `config.py` is deliberately **not** bumped for you: doing that
automatically would silently re-describe every photo in every user's corpus
on their next upgrade, at full vision-API cost, with no warning. That decision
is yours to make, on your own schedule.
