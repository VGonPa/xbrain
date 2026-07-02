# tests/test_xbrain_transcribe.py — the scripts/xbrain-transcribe parakeet wrapper.
import importlib.util
import json
import sys
import types
from importlib.machinery import SourceFileLoader
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-transcribe"
_LOADER = SourceFileLoader("xbrain_transcribe", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_transcribe", _LOADER)
xt = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xt)


def _run(monkeypatch, tmp_path, *, parakeet_rc, writes_file, has_audio):
    """Drive main() with parakeet + ffprobe mocked; return (rc, expected_json_path)."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"x")
    expected = tmp_path / "clip.json"

    def fake_run(cmd, *a, **k):
        if writes_file and parakeet_rc == 0:  # parakeet "wrote" a real transcript
            expected.write_text(
                '{"text": "hola", "segments": [{"start": 0, "end": 1, "text": "hola"}]}'
            )
        return types.SimpleNamespace(returncode=parakeet_rc, stdout="", stderr="")

    monkeypatch.setattr(xt.subprocess, "run", fake_run)
    monkeypatch.setattr(xt, "_has_audio_stream", lambda m: has_audio)
    monkeypatch.setattr(
        sys,
        "argv",
        ["xbrain-transcribe", "--output-format", "json", "--output-dir", str(tmp_path), str(media)],
    )
    return xt.main(), expected


def test_speech_passes_parakeet_output_through(monkeypatch, tmp_path):
    rc, expected = _run(monkeypatch, tmp_path, parakeet_rc=0, writes_file=True, has_audio=True)
    assert rc == 0
    assert json.loads(expected.read_text())["text"] == "hola"  # not overwritten


def test_silent_video_gets_empty_speech_json(monkeypatch, tmp_path):
    rc, expected = _run(monkeypatch, tmp_path, parakeet_rc=0, writes_file=False, has_audio=False)
    assert rc == 0
    data = json.loads(expected.read_text())
    assert data["has_speech"] is False and data["text"] == ""


def test_audio_but_no_output_is_a_real_failure(monkeypatch, tmp_path):
    # exit 0, no file, but the media HAS audio → parakeet choked → surface as failure.
    rc, expected = _run(monkeypatch, tmp_path, parakeet_rc=0, writes_file=False, has_audio=True)
    assert rc == 1
    assert not expected.exists()  # never masked as silent


def test_parakeet_nonzero_exit_propagates(monkeypatch, tmp_path):
    rc, expected = _run(monkeypatch, tmp_path, parakeet_rc=2, writes_file=False, has_audio=False)
    assert rc == 2
    assert not expected.exists()


def test_has_audio_stream_false_without_ffprobe(monkeypatch):
    monkeypatch.setattr(xt.shutil, "which", lambda _: None)  # ffprobe unavailable
    assert xt._has_audio_stream("/some/clip.mp4") is False
    assert xt._has_audio_stream("") is False
