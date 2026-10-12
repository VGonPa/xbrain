import importlib.util
import json
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-transcribe-faster-whisper"
_LOADER = SourceFileLoader("xbrain_transcribe_faster_whisper", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_transcribe_faster_whisper", _LOADER)
xtfw = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xtfw)


def _media(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fake")
    return path


def _json_payload(output_dir: Path) -> dict:
    return json.loads(next(output_dir.glob("*.json")).read_text(encoding="utf-8"))


def test_confirmed_no_audio_writes_no_speech_json(monkeypatch, tmp_path):
    media = _media(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(xtfw, "_confirmed_no_audio", lambda _media: True)

    rc = xtfw.main(["--output-format", "json", "--output-dir", str(out), str(media)])

    assert rc == 0
    payload = _json_payload(out)
    assert payload["has_speech"] is False
    assert payload["text"] == ""


def test_faster_whisper_segments_are_written_as_xbrain_json(monkeypatch, tmp_path):
    media = _media(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(xtfw, "_confirmed_no_audio", lambda _media: False)
    monkeypatch.setattr(xtfw, "_audio_input", lambda media, _out: media)

    class _Info:
        language = "es"

    class _Model:
        def __init__(self, *args, **kwargs):
            assert args == ("small",)
            assert kwargs["device"] == "cpu"
            assert kwargs["compute_type"] == "int8"

        def transcribe(self, path, *, language, vad_filter, beam_size):
            assert Path(path) == media
            assert language is None
            assert vad_filter is True
            assert beam_size == 1
            segments = [
                types.SimpleNamespace(start=0, end=1.5, text=" Hola "),
                types.SimpleNamespace(start=1.5, end=2.0, text=""),
                types.SimpleNamespace(start=2.0, end=4.0, text="mundo"),
            ]
            return iter(segments), _Info()

    fake_module = types.SimpleNamespace(WhisperModel=_Model)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

    rc = xtfw.main(
        ["--model", "small", "--output-format", "json", "--output-dir", str(out), str(media)]
    )

    assert rc == 0
    payload = _json_payload(out)
    assert payload["text"] == "Hola mundo"
    assert payload["language"] == "es"
    assert payload["has_speech"] is True
    assert payload["segments"] == [
        {"start": 0.0, "end": 1.5, "text": "Hola"},
        {"start": 2.0, "end": 4.0, "text": "mundo"},
    ]


def test_faster_whisper_beam_size_is_configurable(monkeypatch, tmp_path):
    media = _media(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(xtfw, "_confirmed_no_audio", lambda _media: False)
    monkeypatch.setattr(xtfw, "_audio_input", lambda media, _out: media)
    monkeypatch.setenv("XBRAIN_WHISPER_BEAM_SIZE", "3")

    class _Info:
        language = "es"

    class _Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path, *, language, vad_filter, beam_size):
            assert beam_size == 3
            segments = [types.SimpleNamespace(start=0, end=1, text=" Hola ")]
            return iter(segments), _Info()

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_Model))

    rc = xtfw.main(["--output-format", "json", "--output-dir", str(out), str(media)])

    assert rc == 0
    assert _json_payload(out)["text"] == "Hola"


def test_missing_faster_whisper_is_clean_error(monkeypatch, tmp_path):
    media = _media(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(xtfw, "_confirmed_no_audio", lambda _media: False)
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)

    class _BlockImport:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "faster_whisper":
                return None
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockImport()])

    rc = xtfw.main(["--output-format", "json", "--output-dir", str(out), str(media)])

    assert rc == 127
    assert not out.exists()


def test_audio_input_extracts_mono_wav_with_ffmpeg(monkeypatch, tmp_path):
    media = _media(tmp_path)
    out = tmp_path / "out"
    calls: list[list[str]] = []

    monkeypatch.setattr(
        xtfw.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    )

    def _run(cmd, **_kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(b"RIFF" + (b"\0" * 100))
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(xtfw.subprocess, "run", _run)

    audio = xtfw._audio_input(media, out)

    assert audio == out / "clip.wav"
    assert audio.exists()
    assert "-vn" in calls[0]
    assert calls[0][-1] == str(audio)
    assert media.exists()


def test_audio_input_can_delete_ephemeral_source_after_extract(monkeypatch, tmp_path):
    media = _media(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setenv("XBRAIN_WHISPER_DELETE_SOURCE_AFTER_AUDIO", "1")
    monkeypatch.setattr(
        xtfw.shutil, "which", lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None
    )

    def _run(cmd, **_kwargs):
        Path(cmd[-1]).write_bytes(b"RIFF" + (b"\0" * 100))
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(xtfw.subprocess, "run", _run)

    audio = xtfw._audio_input(media, out)

    assert audio.exists()
    assert not media.exists()
