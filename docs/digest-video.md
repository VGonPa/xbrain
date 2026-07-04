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
brew install ffmpeg                # frame extraction + audio probe
uv tool install parakeet-mlx       # ASR (Apple Silicon)
uv tool install mlx-vlm            # vision, only for --frames
```

and point `config.toml` at the wrappers:

```toml
[transcribe]
command = "/abs/path/to/xbrain/scripts/xbrain-transcribe"   # wraps parakeet-mlx

[vision]
command = "/abs/path/to/xbrain/scripts/xbrain-vision"       # local + cloud selector
model   = "qwen-7b"
```

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

Then render:

```bash
uv run xbrain generate
```

## What you get

The item's note gains a `## Video digest` section:

```markdown
## Video digest: Elon Musk on the first thing to do when starting a company

> Uh, the goal with Tesla was really to try to show what electric cars can do,
> because people had the wrong impression… (full transcript)

![[_media/1874.../frames/0.png]]
> Slide: a line chart of Model S range vs. price, 2012–2015.
```

The transcript + slide descriptions are plain note text, so they feed `enrich`
(summary + topics) and are **searchable** in Obsidian. A silent video with no
slides degrades gracefully to a one-line "silent video" note.

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
