"""Tests for `xbrain.transcribe` — the external-transcriber subprocess wrapper.

`transcribe_media` shells out to an EXTERNAL local transcriber (config
`[transcribe].command`, default `parakeet-mlx`) and reads its transcript. It
imports NO ML library — the heavy ASR lives outside xbrain core (the locked #44
architecture).

The REAL `parakeet-mlx` writes its JSON transcript to a **file**
`<--output-dir>/<audiostem>.json` (it does NOT emit JSON on stdout and does NOT
accept `--language`), so the default invocation targets a temp `--output-dir` and
reads the produced file. A stdout-emitting wrapper is still supported (stdout is
the fallback source). The **no-output** case (exit 0, no file / empty file, empty
stdout) is a hard error — inferring no-speech from ABSENCE of output would
silently lose transcripts. The only legitimate no-speech signal is a valid JSON
doc (`{"text": ""}` / empty segments / `has_speech: false`).

Every test injects a fake `runner` (a `subprocess.run` stand-in) so NO real
subprocess runs; the file-writing fake mirrors parakeet's real behaviour.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from xbrain.transcribe import (
    DEFAULT_TRANSCRIBE_COMMAND,
    Segment,
    Transcript,
    TranscriberFailed,
    TranscriberNotFound,
    transcribe_media,
)


def _completed(
    stdout: str, *, returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["x"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _file_runner(payload: dict | None, *, returncode: int = 0, stderr: str = ""):
    """A fake `subprocess.run` mirroring the REAL parakeet-mlx: on a zero exit it
    writes `<--output-dir>/<audiostem>.json` (empty stdout). `payload=None`
    simulates a tool that exits 0 but writes nothing (the silent-loss trap)."""
    calls: list[list[str]] = []

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        audio = Path(argv[-1])
        if returncode == 0 and payload is not None:
            (out_dir / f"{audio.stem}.json").write_text(json.dumps(payload), encoding="utf-8")
        return _completed("", returncode=returncode, stderr=stderr)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def _stdout_runner(stdout: str, *, returncode: int = 0, stderr: str = ""):
    """A fake for a stdout-emitting wrapper (the flexible override path)."""
    calls: list[list[str]] = []

    def _run(argv, **_kwargs):
        calls.append(list(argv))
        return _completed(stdout, returncode=returncode, stderr=stderr)

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def _raw_bytes_runner(raw: bytes, *, returncode: int = 0):
    """A fake that writes RAW bytes to the output file (e.g. non-UTF-8), to
    exercise the read/decode guard."""

    def _run(argv, **_kwargs):
        out_dir = Path(argv[argv.index("--output-dir") + 1])
        audio = Path(argv[-1])
        (out_dir / f"{audio.stem}.json").write_bytes(raw)
        return _completed("", returncode=returncode)

    return _run


# ------------------------------------------------------------ file-based (default) path


def test_reads_produced_json_file_into_transcript(tmp_path: Path):
    """The REAL contract: parakeet writes a JSON file to --output-dir; xbrain reads
    THAT file (not stdout) and parses it."""
    payload = {
        "text": "hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "hello"},
            {"start": 1.5, "end": 2.7, "text": "world"},
        ],
    }
    result = transcribe_media(tmp_path / "a.mp4", runner=_file_runner(payload))
    assert isinstance(result, Transcript)
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.has_speech is True
    assert result.segments == [
        Segment(start=0.0, end=1.5, text="hello"),
        Segment(start=1.5, end=2.7, text="world"),
    ]


def test_argv_targets_output_dir_and_omits_language(tmp_path: Path):
    """The default argv passes `--output-dir` (file-based) + the model + the audio
    path, and does NOT pass `--language` (parakeet auto-detects and rejects it)."""
    runner = _file_runner({"text": "hi"})
    media = tmp_path / "clip.mp4"
    transcribe_media(
        media, command="parakeet-mlx", model="parakeet-tdt-0.6b", language="es", runner=runner
    )
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[0] == "parakeet-mlx"
    assert "parakeet-tdt-0.6b" in argv
    assert "--output-dir" in argv
    assert "--language" not in argv  # C1: not passed to the auto-detecting tool
    assert "es" not in argv
    assert argv[-1] == str(media)


def test_multi_token_command_is_split(tmp_path: Path):
    """A `command` with args (a wrapper `python -m foo`) is split into argv
    tokens, not run through a shell."""
    runner = _file_runner({"text": "hi"})
    transcribe_media(tmp_path / "a.mp4", command="python -m my_asr", runner=runner)
    argv = runner.calls[0]  # type: ignore[attr-defined]
    assert argv[:3] == ["python", "-m", "my_asr"]


def test_default_command_is_parakeet_mlx(tmp_path: Path):
    runner = _file_runner({"text": "hi"})
    transcribe_media(tmp_path / "a.mp4", runner=runner)
    assert runner.calls[0][0] == DEFAULT_TRANSCRIBE_COMMAND  # type: ignore[attr-defined]
    assert DEFAULT_TRANSCRIBE_COMMAND == "parakeet-mlx"


def test_output_dir_is_cleaned_up(tmp_path: Path):
    """The temp `--output-dir` lives inside the audio's dir and is removed after —
    no transcript files or dirs linger."""
    transcribe_media(tmp_path / "a.mp4", runner=_file_runner({"text": "hi"}))
    assert list(tmp_path.glob("xbrain-asr-*")) == []
    assert list(tmp_path.rglob("*.json")) == []


def test_stdout_wrapper_output_is_parsed(tmp_path: Path):
    """A user-provided wrapper that emits JSON on STDOUT (no file) is still
    supported — stdout is the fallback source."""
    result = transcribe_media(
        tmp_path / "a.mp4", runner=_stdout_runner(json.dumps({"text": "from stdout"}))
    )
    assert result.text == "from stdout"
    assert result.has_speech is True


# ------------------------------------------------------------ no-output / silent-loss guard


def test_no_output_raises_transcriber_failed(tmp_path: Path):
    """C1(b): exit 0 but NO file and empty stdout → hard error. Inferring
    no-speech from absence would silently LOSE the transcript."""
    with pytest.raises(TranscriberFailed) as excinfo:
        transcribe_media(tmp_path / "a.mp4", runner=_file_runner(None))
    assert "no output" in str(excinfo.value).lower()


def test_empty_json_file_yields_no_speech(tmp_path: Path):
    """A LEGITIMATE no-speech signal: a valid JSON doc with empty text → graceful
    has_speech=False (many X videos are silent/screen-only)."""
    result = transcribe_media(
        tmp_path / "silent.mp4", runner=_file_runner({"text": "", "segments": []})
    )
    assert result.has_speech is False
    assert result.text == ""
    assert result.segments == []


def test_whitespace_only_text_is_no_speech(tmp_path: Path):
    result = transcribe_media(
        tmp_path / "s.mp4", runner=_file_runner({"text": "   ", "segments": []})
    )
    assert result.has_speech is False


def test_explicit_has_speech_flag_is_honored(tmp_path: Path):
    result = transcribe_media(
        tmp_path / "s.mp4", runner=_file_runner({"text": "", "has_speech": True, "segments": []})
    )
    assert result.has_speech is True


def test_has_speech_derived_from_segments_when_text_empty(tmp_path: Path):
    """Top-level text empty but a real segment present → has_speech=True (item 4c)."""
    result = transcribe_media(
        tmp_path / "s.mp4",
        runner=_file_runner({"text": "", "segments": [{"start": 0, "end": 1, "text": "hi"}]}),
    )
    assert result.has_speech is True


def test_segments_only_transcript_backfills_text(tmp_path: Path):
    """A segments-only transcriber (empty top-level text) must NOT lose the spoken
    content: `text` is backfilled by joining the segment texts, so the persisted
    `x_video` source carries the transcript (consistent with has_speech=True)."""
    result = transcribe_media(
        tmp_path / "s.mp4",
        runner=_file_runner(
            {
                "text": "",
                "segments": [
                    {"start": 0, "end": 1, "text": "hello"},
                    {"start": 1, "end": 2, "text": "world"},
                ],
            }
        ),
    )
    assert result.text == "hello world"
    assert result.has_speech is True


def test_top_level_text_wins_over_segments_when_present(tmp_path: Path):
    """When the top-level text IS present it is kept verbatim (no double-join)."""
    result = transcribe_media(
        tmp_path / "s.mp4",
        runner=_file_runner(
            {"text": "full text", "segments": [{"start": 0, "end": 1, "text": "frag"}]}
        ),
    )
    assert result.text == "full text"


# ------------------------------------------------------------ errors


def test_missing_binary_raises_clear_operator_error(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "parakeet-mlx")

    with pytest.raises(TranscriberNotFound) as excinfo:
        transcribe_media(tmp_path / "a.mp4", runner=_run)
    assert "parakeet-mlx" in str(excinfo.value)
    assert "transcribe" in str(excinfo.value).lower()


def test_permission_denied_raises_transcriber_not_found(tmp_path: Path):
    """A non-FileNotFound OSError (e.g. the binary is not executable) is still a
    clean 'cannot run the transcriber' operator error, not a raw traceback."""

    def _run(_argv, **_kwargs):
        raise PermissionError(13, "Permission denied", "parakeet-mlx")

    with pytest.raises(TranscriberNotFound):
        transcribe_media(tmp_path / "a.mp4", runner=_run)


def test_nonzero_exit_raises_with_stderr(tmp_path: Path):
    runner = _file_runner(None, returncode=3, stderr="model weights not found")
    with pytest.raises(TranscriberFailed) as excinfo:
        transcribe_media(tmp_path / "a.mp4", runner=runner)
    assert "model weights not found" in str(excinfo.value)


def test_malformed_json_raises(tmp_path: Path):
    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=_stdout_runner("this is not json {"))


def test_non_object_json_raises(tmp_path: Path):
    """Valid JSON that is not an object (a list/scalar) is malformed output."""
    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=_stdout_runner("[1, 2, 3]"))


def test_malformed_segment_raises(tmp_path: Path):
    runner = _file_runner({"text": "hi", "segments": [{"start": "nope", "end": 1.0, "text": "x"}]})
    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=runner)


def test_non_list_segments_raises(tmp_path: Path):
    """A `segments` value that is present but not a list is malformed output."""
    with pytest.raises(TranscriberFailed):
        transcribe_media(
            tmp_path / "a.mp4", runner=_file_runner({"text": "hi", "segments": "nope"})
        )


def test_null_segments_treated_as_empty(tmp_path: Path):
    """`segments: null` is tolerated as 'no segments' (not a crash) — has_speech
    still derives from the non-empty text."""
    result = transcribe_media(
        tmp_path / "a.mp4", runner=_file_runner({"text": "hi", "segments": None})
    )
    assert result.segments == []
    assert result.has_speech is True


def test_timeout_raises_transcriber_failed(tmp_path: Path):
    def _run(_argv, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="parakeet-mlx", timeout=1)

    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=_run)


def test_non_utf8_output_file_is_per_video_transcriber_failed(tmp_path: Path):
    """A `.json` output file with invalid UTF-8 bytes must surface as a per-video
    `TranscriberFailed` (which `digest-video` records + continues the batch), NOT a
    raw `UnicodeDecodeError` that aborts the whole run."""
    with pytest.raises(TranscriberFailed):
        transcribe_media(tmp_path / "a.mp4", runner=_raw_bytes_runner(b"\xff\xfe\x00not-utf8"))


def test_non_utf8_stdout_is_per_video_transcriber_failed(tmp_path: Path):
    """The STDOUT-arm mirror of the output-file guard: `subprocess.run(text=True)`
    itself raises `UnicodeDecodeError` while decoding non-UTF-8 stdout. That must
    surface as a per-video `TranscriberFailed` (recorded, batch continues), NOT a raw
    traceback that aborts the whole run."""

    def _run(_argv, **_kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid start byte")

    with pytest.raises(TranscriberFailed) as excinfo:
        transcribe_media(tmp_path / "a.mp4", runner=_run)
    assert "non-UTF-8 stdout" in str(excinfo.value)


def test_language_falls_back_to_requested_when_payload_omits_it(tmp_path: Path):
    """The payload has no `language` → the Transcript records the requested hint."""
    result = transcribe_media(
        tmp_path / "a.mp4", language="es", runner=_file_runner({"text": "hola"})
    )
    assert result.language == "es"


def test_title_passed_through_when_present(tmp_path: Path):
    """An optional `title` in the transcriber JSON flows onto the Transcript (item
    14) — future-proofs PR3's digest-title rendering."""
    result = transcribe_media(
        tmp_path / "a.mp4", runner=_file_runner({"text": "hi", "title": "A Great Talk"})
    )
    assert result.title == "A Great Talk"


def test_title_is_none_when_absent(tmp_path: Path):
    result = transcribe_media(tmp_path / "a.mp4", runner=_file_runner({"text": "hi"}))
    assert result.title is None


def test_transcribe_imports_no_ml_library():
    """The locked #44 architecture: the CLI carries NO MLX/CoreML/torch/whisper
    dependency — the transcriber is an external subprocess. Guard it so a future
    edit can't quietly `import mlx`."""
    import xbrain.transcribe as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import mlx",
        "import torch",
        "import whisper",
        "parakeet_mlx",
        "coremltools",
    ):
        assert forbidden not in source
