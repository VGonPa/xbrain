"""Shell out to an external local transcriber and read its transcript.

The thin, ML-free wrapper at the heart of the `digest-video` stage (#44). xbrain
stays **mechanical**: the heavy ASR (Parakeet TDT via `parakeet-mlx`, or a
whisper fallback) runs as an EXTERNAL subprocess located via config/PATH, so the
CLI carries **no** MLX / CoreML dependency. This module imports no ML library — a
test asserts it.

The transcriber contract xbrain expects (matches the real `parakeet-mlx`):

- It is invoked as ``<command> [--model M] --output-format json --output-dir
  <TMPDIR> <media-path>`` where ``<command>`` comes from ``[transcribe].command``
  (default ``parakeet-mlx``) and may itself be a multi-token command (a wrapper
  script), split with ``shlex`` and run WITHOUT a shell. Note: ``--language`` is
  **not** passed — parakeet auto-detects and rejects it; the `language` argument
  is only recorded on the result as a fallback when the tool omits it.
- `parakeet-mlx` writes its transcript to a **file** ``<TMPDIR>/<stem>.json`` (it
  does NOT emit JSON on stdout), so xbrain reads the produced JSON file from the
  temp ``--output-dir``. A user-provided wrapper that emits JSON on **stdout** is
  also supported — stdout is the fallback source. The JSON shape::

      {"text": "...", "language": "en",
       "segments": [{"start": 0.0, "end": 3.2, "text": "..."}]}

  ``has_speech`` may be reported explicitly; otherwise it is derived (non-empty
  text or any segment ⇒ speech).
- **No-speech is a JSON signal, never an ABSENCE of output.** A valid JSON doc
  with empty text (``{"text": ""}``) / empty segments / ``has_speech: false``
  yields ``has_speech=False`` + empty text — recorded data, not a failure. But a
  transcriber that exits 0 and produces **no usable output** (no file / empty
  file AND empty stdout) raises `TranscriberFailed`: inferring no-speech there
  would SILENTLY LOSE the transcript (the real parakeet writes a file, so an
  empty stdout is not "silence").

Failures surface as clear operator errors (subclasses of `RuntimeError`, which
the CLI's `_handle_cli_errors` turns into a clean exit-1): a **missing / non-
executable binary** (`TranscriberNotFound`), a **non-zero exit**, **no output**,
or **malformed output** (`TranscriberFailed`). The `runner` (a `subprocess.run`
stand-in) is injectable so tests run offline against a fake — no real
transcriber, ever.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess  # nosec B404 - the transcriber is an external subprocess by design (#44)
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The default external transcriber (Parakeet TDT 0.6b via MLX/Metal). Override
# with `[transcribe].command`; whisper / faster-whisper is the portable fallback.
DEFAULT_TRANSCRIBE_COMMAND = "parakeet-mlx"

# A generous default wall-clock cap: a 72-minute talk transcribes in minutes on
# Apple Silicon, but a wedged process must not hang the run forever.
_DEFAULT_TIMEOUT_SECONDS = 1800

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class TranscribeError(RuntimeError):
    """Base class for every transcriber failure (a clean CLI exit-1)."""


class TranscriberNotFound(TranscribeError):
    """The configured transcriber binary is missing / not executable."""


class TranscriberFailed(TranscribeError):
    """The transcriber ran but failed: non-zero exit, no output, or unparseable output."""


@dataclass(frozen=True)
class Segment:
    """One timestamped transcript segment (seconds)."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    """The parsed result of a transcription run.

    `has_speech=False` (with empty `text` + no `segments`) is the graceful
    no-audio / no-speech outcome — carried, not raised. `language` is the
    detected language when the transcriber reports it, else the requested one.
    """

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    has_speech: bool = False
    # An optional talk/video title the transcriber may surface (`None` when the
    # ASR does not derive one — the PR2 default contract). Carried onto the
    # `x_video` content source so PR3 can render the digest title.
    title: str | None = None


def _build_argv(command: str, model: str | None, output_dir: Path, path: Path) -> list[str]:
    """Assemble the subprocess argv from config + the media path.

    `command` is `shlex`-split so a multi-token wrapper (`python -m my_asr`)
    works, and the whole thing runs WITHOUT a shell. The transcript is written to
    `--output-dir` (file-based, matching the real parakeet-mlx). `--language` is
    deliberately omitted — parakeet auto-detects and rejects it.
    """
    argv = shlex.split(command)
    if model:
        argv += ["--model", model]
    argv += ["--output-format", "json", "--output-dir", str(output_dir), str(path)]
    return argv


def _run_transcriber(argv: list[str], runner: Runner, timeout_seconds: int) -> str:
    """Run the transcriber; return its stdout, or raise a clear operator error."""
    try:
        completed = runner(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise TranscriberNotFound(
            f"transcriber {argv[0]!r} not found — install it or set a valid "
            f"[transcribe].command in config.toml ({exc})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriberFailed(
            f"transcriber {argv[0]!r} timed out after {timeout_seconds}s"
        ) from exc
    except OSError as exc:  # not-executable, permission denied, etc.
        raise TranscriberNotFound(
            f"transcriber {argv[0]!r} could not be executed — check "
            f"[transcribe].command in config.toml ({exc})"
        ) from exc
    except UnicodeDecodeError as exc:  # subprocess.run(text=True) on non-UTF-8 stdout
        raise TranscriberFailed(
            f"transcriber {argv[0]!r} produced non-UTF-8 stdout: {exc}"
        ) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise TranscriberFailed(
            f"transcriber {argv[0]!r} exited {completed.returncode}: {stderr or '(no stderr)'}"
        )
    return completed.stdout or ""


def _read_raw_output(output_dir: Path, stdout: str) -> str | None:
    """The transcriber's raw JSON: the produced `*.json` file (preferred, matching
    parakeet-mlx), else stdout (a wrapper), else None when NEITHER exists.

    Returning None is the no-output signal the caller turns into a hard error — it
    is NOT treated as no-speech, because absence of output means a lost transcript,
    not silence. An unreadable or non-UTF-8 output file is malformed output
    (`TranscriberFailed`), so — like every other per-video failure — it is recorded
    and the batch continues, never a raw `UnicodeDecodeError`/`OSError` traceback.
    """
    try:
        for json_file in sorted(output_dir.glob("*.json")):
            content = json_file.read_text(encoding="utf-8")
            if content.strip():
                return content
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise TranscriberFailed(
            f"transcriber output file in {output_dir} could not be read as UTF-8: {exc}"
        ) from exc
    return stdout if stdout.strip() else None


def _parse_segment(raw: object) -> Segment:
    """Parse one segment dict into a `Segment`, or raise on a malformed shape."""
    if not isinstance(raw, dict):
        raise TranscriberFailed(f"transcriber segment is not an object: {raw!r}")
    try:
        return Segment(
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
            text=str(raw.get("text", "")),
        )
    except (TypeError, ValueError) as exc:
        raise TranscriberFailed(f"transcriber segment has non-numeric bounds: {raw!r}") from exc


def _parse_segments(data: dict[str, Any]) -> list[Segment]:
    """Parse the `segments` list. `null`/absent → empty; a non-list is malformed."""
    raw_segments = data.get("segments") or []
    if not isinstance(raw_segments, list):
        raise TranscriberFailed(f"transcriber 'segments' is not a list: {raw_segments!r}")
    return [_parse_segment(segment) for segment in raw_segments]


def _resolve_text(raw_text: str, segments: list[Segment]) -> str:
    """The transcript text: the top-level `text` when present, else backfilled from
    the segment texts.

    A segments-only transcriber (some whisper wrappers) emits an empty top-level
    `text` with the spoken content ONLY in `segments`. Persisting `text=""` there
    would silently drop the transcript (with `has_speech=True`), so join the
    segment texts to keep `text` populated and consistent.
    """
    if raw_text.strip():
        return raw_text
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def _derive_has_speech(data: dict[str, Any], text: str, segments: list[Segment]) -> bool:
    """Trust an explicit `has_speech`, else infer from non-empty text or segments."""
    reported = data.get("has_speech")
    if reported is not None:
        return bool(reported)
    return bool(text.strip() or segments)


def _parse_output(raw: str, requested_language: str | None) -> Transcript:
    """Parse the transcriber's raw JSON into a `Transcript`.

    `raw` is guaranteed non-empty by the caller (empty output is a hard error, not
    a parse target). It MUST be a JSON object; a list/scalar or invalid JSON is
    malformed output (`TranscriberFailed`).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TranscriberFailed(f"transcriber output was not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise TranscriberFailed(f"transcriber output was not a JSON object: {type(data).__name__}")

    segments = _parse_segments(data)
    text = _resolve_text(str(data.get("text") or ""), segments)
    language = data.get("language") or requested_language
    has_speech = _derive_has_speech(data, text, segments)
    raw_title = data.get("title")
    title = str(raw_title) if raw_title else None
    return Transcript(
        text=text, segments=segments, language=language, has_speech=has_speech, title=title
    )


def transcribe_media(
    path: Path | str,
    *,
    command: str = DEFAULT_TRANSCRIBE_COMMAND,
    model: str | None = None,
    language: str | None = None,
    runner: Runner | None = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
) -> Transcript:
    """Transcribe one audio/video file via the external transcriber subprocess.

    Shells out to `command` (default `parakeet-mlx`; a multi-token wrapper is
    supported and run WITHOUT a shell), writing the transcript to a temp
    `--output-dir` INSIDE the media's directory, then reads the produced JSON file
    (or stdout, for a wrapper) into a `Transcript`. The no-speech case is graceful
    ONLY via a valid JSON signal (`has_speech=False`, empty `text`); a run that
    produces no usable output raises `TranscriberFailed` (never silently lost). A
    missing / non-executable binary raises `TranscriberNotFound`; a non-zero exit
    or malformed output raises `TranscriberFailed` — all clean CLI exit-1s.

    `language` is recorded on the result as a fallback when the transcriber omits
    it; it is NOT passed to the (auto-detecting) transcriber. `runner` (a
    `subprocess.run` stand-in) is injectable for tests; it defaults to
    `subprocess.run`, resolved at call time so it stays monkeypatchable. The temp
    `--output-dir` is always removed, even on failure.
    """
    audio = Path(path)
    active_runner: Runner = runner if runner is not None else subprocess.run
    output_dir = Path(tempfile.mkdtemp(prefix="xbrain-asr-", dir=audio.parent))
    try:
        argv = _build_argv(command, model, output_dir, audio)
        stdout = _run_transcriber(argv, active_runner, timeout_seconds)
        raw = _read_raw_output(output_dir, stdout)
        if raw is None:
            raise TranscriberFailed(
                f"transcriber {argv[0]!r} exited 0 but produced no output "
                f"(no JSON file in --output-dir, empty stdout) — a transcript would be lost"
            )
        return _parse_output(raw, language)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
