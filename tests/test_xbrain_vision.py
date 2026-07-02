# tests/test_xbrain_vision.py — the scripts/xbrain-vision model-selector wrapper.
import importlib.util
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

# The wrapper is a bare script (no .py suffix), so give importlib an explicit
# source loader. Top-level imports are stdlib only → safe without mlx/anthropic.
_PATH = Path(__file__).resolve().parent.parent / "scripts" / "xbrain-vision"
_LOADER = SourceFileLoader("xbrain_vision", str(_PATH))
_SPEC = importlib.util.spec_from_loader("xbrain_vision", _LOADER)
xv = importlib.util.module_from_spec(_SPEC)
_LOADER.exec_module(xv)


def test_resolve_local_aliases():
    assert xv._resolve("qwen-3b") == ("local", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    assert xv._resolve("qwen-7b") == ("local", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
    assert xv._resolve("qwen-32b")[0] == "local"


def test_resolve_cloud_aliases_use_current_model_ids():
    assert xv._resolve("opus") == ("cloud", "claude-opus-4-8")
    assert xv._resolve("sonnet") == ("cloud", "claude-sonnet-4-6")
    assert xv._resolve("haiku") == ("cloud", "claude-haiku-4-5")


def test_resolve_claude_prefix_passthrough():
    assert xv._resolve("claude-opus-4-8") == ("cloud", "claude-opus-4-8")


def test_resolve_hf_repo_is_local():
    assert xv._resolve("mlx-community/Some-VLM-4bit") == ("local", "mlx-community/Some-VLM-4bit")


def test_resolve_unknown_model_exits():
    with pytest.raises(SystemExit):
        xv._resolve("gpt-9")


def test_default_model_is_local_qwen3b():
    assert xv.DEFAULT_MODEL == "qwen-3b"
    assert xv._resolve(xv.DEFAULT_MODEL)[0] == "local"


def test_main_returns_1_on_empty_description(monkeypatch, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(xv, "_describe_local", lambda model, image: "")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", "--model", "qwen-3b", str(img)])
    assert xv.main() == 1  # empty output is a failure, per the vision contract


def test_main_prints_description_and_returns_0(monkeypatch, tmp_path, capsys):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(xv, "_describe_cloud", lambda model, image: "Un gráfico de barras.")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", "--model", "opus", str(img)])
    assert xv.main() == 0
    assert "Un gráfico de barras." in capsys.readouterr().out
