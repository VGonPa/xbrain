# `digest-video` - videos without storing video bytes

`digest-video` turns bookmarked X videos into text for the normal
`enrich -> topics -> generate` pipeline without storing MP4, audio, frames or
thumbnails as local video artifacts.

The preferred input is a caption/text-track URL already exposed by X and captured
in `items.json`. If X does not expose captions, XBrain can stream the MP4 into a
private temporary directory, transcribe it with `[transcribe].command`, store only
the resulting text, and delete the temporary media bytes.

## What It Stores

For each video:

1. XBrain fetches the small caption/text file when X exposes one.
2. If captions are absent, it streams the MP4 only inside a temporary directory.
   The `--max-size` cap is enforced against `Content-Length` when present and
   again while bytes are streaming. `list-videos` may show a conservative
   `bitrate × duration_millis` estimate, but the fallback uses the real response
   size whenever possible.
3. It normalizes the caption or ASR output into the original-language raw transcript.
4. With the faster-whisper wrapper, it extracts a temporary mono 16k WAV and
   transcribes that smaller audio file.
5. It deletes temporary media bytes before persisting anything.
6. It asks the configured text LLM for a medium-depth executive summary.
7. It stores the executive summary as the `x_video` content source used by
   `enrich`, `topics`, the dashboard and Ask XBrain.
8. It renders two nearby vault files under `videos/<video>/`:
   - `summary.md`: dashboard-ready executive summary.
   - `transcript.md`: raw transcript reference, marked `xbrain_exclude: true`.

The raw transcript is retained for audit/reading, but it is not indexed by the
dashboard or Ask XBrain and is not fed into topic synthesis.

## Run It

```bash
uv run xbrain digest-video --all-pending
uv run xbrain generate
```

Useful selectors:

```bash
uv run xbrain digest-video --ids 123,456
uv run xbrain digest-video --topic ai-coding
uv run xbrain digest-video --all-pending --limit 10
uv run xbrain digest-video --ids 123 --force
```

Output example:

```text
Videos: resumidos 6, ya digeridos 2, sin transcript 4, fallidos 0, ...
Dedup: 12 items <- 9 videos (6 procesados este run).
```

## Disabled Paths

These commands/options are intentionally disabled by storage policy:

```bash
uv run xbrain download-videos
uv run xbrain fetch-video
uv run xbrain digest-video --frames
uv run xbrain digest-video --vision-model xiaomi/mimo-v2.5
```

They do not write MP4, audio or frame files. `--max-size` on `digest-video` is
used only by the temporary MP4 fallback; captions bypass it because no video
bytes are downloaded.

## Limitations

The ASR fallback needs an external transcriber installed on the server and
configured in `config.toml` under `[transcribe].command`. Without it, videos that
lack captions are reported as failed and no summary is written.

On a Linux VPS, the included `scripts/xbrain-transcribe-faster-whisper` wrapper
can be used with a separate faster-whisper venv. `model = "small"` is the
recommended CPU default; larger models may improve accuracy but raise time and
memory risk on long videos. `XBRAIN_WHISPER_BEAM_SIZE=1` is the recommended VPS
default for long videos; raise it to 3/5 if you prefer accuracy over runtime.
Use `[transcribe].timeout_seconds` to allow longer ASR runs explicitly. Example:

```toml
[transcribe]
command = "env XBRAIN_WHISPER_CACHE=/opt/xbrain-asr/models XBRAIN_WHISPER_CPU_THREADS=6 XBRAIN_WHISPER_BEAM_SIZE=1 XBRAIN_WHISPER_DELETE_SOURCE_AFTER_AUDIO=1 /opt/xbrain-asr/bin/python /workspace/projects/xbrain/scripts/xbrain-transcribe-faster-whisper"
model = "small"
timeout_seconds = 7200
```
