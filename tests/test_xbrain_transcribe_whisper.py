# tests/test_xbrain_transcribe_whisper.py — the scripts/xbrain-transcribe-whisper wrapper.
"""The multilingual backend. It lived untested outside the repo for a month.

That is not incidental to the bug it was written to fix: it was authored on
17-jul-2026 after parakeet was measured fabricating Spanish, and then never wired
into `config.toml`, so every Spanish video in the nightly kept going to parakeet.
A backend that is not in the repo is a backend nobody can see is unused.
"""

import importlib.util
import json
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-transcribe-whisper"
_LOADER = SourceFileLoader("xbrain_transcribe_whisper", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_transcribe_whisper", _LOADER)
xtw = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xtw)

_WHISPER_JSON = {
    "text": "  Hola, esto es una prueba.  ",
    "language": "spanish",
    "segments": [{"start": 0.0, "end": 2.0, "text": " Hola, esto es una prueba. ", "extra": 1}],
}


def _drive(monkeypatch, tmp_path, *, whisper_json=_WHISPER_JSON, rc=0, no_audio=False, env=None):
    """Run main() with the `whisper` CLI mocked; return (exit_code, argv, out_path)."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    seen: dict = {}
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)

    def fake_run(cmd, *a, **k):
        seen["argv"] = cmd
        if rc == 0 and whisper_json is not None:
            (tmp_path / "clip.json").write_text(json.dumps(whisper_json))
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(xtw.subprocess, "run", fake_run)
    monkeypatch.setattr(xtw, "_confirmed_no_audio", lambda m: no_audio)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xbrain-transcribe-whisper",
            "--output-format",
            "json",
            "--output-dir",
            str(tmp_path),
            str(media),
        ],
    )
    return xtw.main(), seen.get("argv"), tmp_path / "clip.json"


def test_output_is_normalised_to_the_shape_xbrain_reads(monkeypatch, tmp_path):
    """xbrain reads {text, language, segments, has_speech} — whisper emits its own."""
    code, _argv, out = _drive(monkeypatch, tmp_path)
    assert code == 0
    written = json.loads(out.read_text())
    assert written["text"] == "Hola, esto es una prueba."  # stripped
    assert written["language"] == "spanish"
    assert written["has_speech"] is True
    assert written["segments"] == [{"start": 0.0, "end": 2.0, "text": "Hola, esto es una prueba."}]


def test_language_defaults_to_spanish(monkeypatch, tmp_path):
    monkeypatch.delenv("WHISPER_LANGUAGE", raising=False)
    _code, argv, _out = _drive(monkeypatch, tmp_path)
    assert argv is not None
    assert argv[argv.index("--language") + 1] == "Spanish"


@pytest.mark.parametrize("value", ["auto", "AUTO", ""])
def test_auto_language_omits_the_flag_so_whisper_detects(monkeypatch, tmp_path, value):
    """The router passes "auto" when it could not identify the clip.

    Forcing the Spanish default onto an UNKNOWN language would be the same class
    of error as sending it to parakeet — an assumption presented as a transcript.
    """
    _code, argv, _out = _drive(monkeypatch, tmp_path, env={"WHISPER_LANGUAGE": value})
    assert argv is not None
    assert "--language" not in argv


def test_a_parakeet_model_id_is_ignored_not_passed_to_whisper(monkeypatch, tmp_path):
    """`--model mlx-community/...` is meaningless here; fall back to WHISPER_MODEL."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    seen: dict = {}

    def fake_run(cmd, *a, **k):
        seen["argv"] = cmd
        (tmp_path / "clip.json").write_text(json.dumps(_WHISPER_JSON))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(xtw.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xbrain-transcribe-whisper",
            "--model",
            "mlx-community/parakeet-tdt-0.6b-v2",
            "--output-dir",
            str(tmp_path),
            str(media),
        ],
    )
    assert xtw.main() == 0
    assert seen["argv"][seen["argv"].index("--model") + 1] == "small"


def test_a_real_whisper_model_id_is_honoured(monkeypatch, tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    seen: dict = {}

    def fake_run(cmd, *a, **k):
        seen["argv"] = cmd
        (tmp_path / "clip.json").write_text(json.dumps(_WHISPER_JSON))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(xtw.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "xbrain-transcribe-whisper",
            "--model",
            "large-v3-turbo",
            "--output-dir",
            str(tmp_path),
            str(media),
        ],
    )
    assert xtw.main() == 0
    assert seen["argv"][seen["argv"].index("--model") + 1] == "large-v3-turbo"


def test_confirmed_silent_video_yields_empty_speech_not_a_failure(monkeypatch, tmp_path):
    code, _argv, out = _drive(monkeypatch, tmp_path, whisper_json=None, no_audio=True)
    assert code == 0
    assert json.loads(out.read_text())["has_speech"] is False


def test_no_output_with_audio_present_is_a_real_failure(monkeypatch, tmp_path):
    """Unverifiable silence must NOT be masked as a silent video."""
    code, _argv, _out = _drive(monkeypatch, tmp_path, whisper_json=None, no_audio=False)
    assert code == 1


def test_whisper_nonzero_exit_propagates(monkeypatch, tmp_path):
    code, _argv, _out = _drive(monkeypatch, tmp_path, rc=3)
    assert code == 3


def test_empty_transcript_reports_no_speech(monkeypatch, tmp_path):
    _code, _argv, out = _drive(
        monkeypatch, tmp_path, whisper_json={"text": "   ", "language": "es", "segments": []}
    )
    assert json.loads(out.read_text())["has_speech"] is False


def test_missing_media_is_a_clean_usage_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["xbrain-transcribe-whisper", "--output-dir", "/tmp"])
    assert xtw.main() == 2


@pytest.mark.parametrize(
    "artefact",
    [
        "Subtítulos realizados por la comunidad de Amara.org",  # the REAL store case
        "subtitles by the Amara.org community",
        "Thanks for watching!",
        "  ♪♪♪  ",
        "[Música]",
    ],
)
def test_whisper_no_speech_hallucinations_are_discarded(monkeypatch, tmp_path, artefact):
    """Whisper invents subtitle boilerplate on silent audio — it is not speech.

    `ffprobe` cannot catch this: the file HAS an audio stream, it just carries no
    speech. Item 2082467662735507767 reached the vault as a "transcript" reading
    only the Amara credit line, recorded `has_speech=true`.
    """
    _code, _argv, out = _drive(
        monkeypatch,
        tmp_path,
        whisper_json={
            "text": artefact,
            "language": "spanish",
            "segments": [{"start": 0.0, "end": 1.0, "text": artefact}],
        },
    )
    written = json.loads(out.read_text())
    assert written["text"] == ""
    assert written["segments"] == []
    assert written["has_speech"] is False


@pytest.mark.parametrize(
    "real",
    [
        "Thanks for watching! Now let me show you the actual demo.",
        "En este vídeo hablo de Amara.org y de cómo funciona la comunidad.",
        "Music theory is what this whole talk is about.",
    ],
)
def test_real_speech_mentioning_an_artefact_phrase_survives(monkeypatch, tmp_path, real):
    """Whole-transcript match only — a video that MENTIONS these is a real video."""
    _code, _argv, out = _drive(
        monkeypatch,
        tmp_path,
        whisper_json={"text": real, "language": "en", "segments": []},
    )
    written = json.loads(out.read_text())
    assert written["text"] == real
    assert written["has_speech"] is True
