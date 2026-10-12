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
    assert xv._resolve("opus", "anthropic") == ("cloud", "claude-opus-4-8")
    assert xv._resolve("sonnet", "anthropic") == ("cloud", "claude-sonnet-4-6")
    assert xv._resolve("haiku", "anthropic") == ("cloud", "claude-haiku-4-5")


def test_resolve_claude_prefix_passthrough():
    assert xv._resolve("claude-opus-4-8", "anthropic") == ("cloud", "claude-opus-4-8")


def test_resolve_nanogpt_model_is_cloud():
    assert xv._resolve("zai-org/glm-5.2", "nanogpt") == ("cloud", "zai-org/glm-5.2")


def test_resolve_nanogpt_rejects_anthropic_native_model():
    with pytest.raises(SystemExit):
        xv._resolve("claude-sonnet-4-6", "nanogpt")


def test_resolve_hf_repo_is_local():
    assert xv._resolve("local:mlx-community/Some-VLM-4bit") == (
        "local",
        "mlx-community/Some-VLM-4bit",
    )


def test_resolve_provider_qualified_model_under_anthropic_errors():
    with pytest.raises(SystemExit):
        xv._resolve("zai-org/glm-5.2", "anthropic")


def test_resolve_unknown_model_exits():
    with pytest.raises(SystemExit):
        xv._resolve("gpt-9", "anthropic")


def test_default_model_is_nanogpt_model():
    assert xv.DEFAULT_NANOGPT_VISION_MODEL == "xiaomi/mimo-v2.5"
    assert xv._resolve(xv.DEFAULT_NANOGPT_VISION_MODEL, "nanogpt") == (
        "cloud",
        "xiaomi/mimo-v2.5",
    )


def test_main_returns_1_on_empty_description(monkeypatch, tmp_path):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(xv, "_describe_local", lambda model, image: "")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", "--model", "qwen-3b", str(img)])
    assert xv.main() == 1  # empty output is a failure, per the vision contract


def test_main_prints_description_and_returns_0(monkeypatch, tmp_path, capsys):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.setenv("XBRAIN_LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(xv, "_describe_anthropic", lambda model, image: "Un gráfico de barras.")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", "--model", "opus", str(img)])
    assert xv.main() == 0
    assert "Un gráfico de barras." in capsys.readouterr().out


def test_main_uses_nanogpt_by_default(monkeypatch, tmp_path, capsys):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.delenv("XBRAIN_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("XBRAIN_LLM_MODEL", raising=False)
    monkeypatch.delenv("XBRAIN_LLM_VISION_MODEL", raising=False)
    monkeypatch.delenv("NANOGPT_MODEL", raising=False)
    monkeypatch.delenv("NANOGPT_VISION_MODEL", raising=False)
    monkeypatch.setattr(xv, "_describe_nanogpt", lambda model, image: f"nanogpt:{model}")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", str(img)])
    assert xv.main() == 0
    assert "nanogpt:xiaomi/mimo-v2.5" in capsys.readouterr().out


def test_main_prefers_vision_model_environment(monkeypatch, tmp_path, capsys):
    img = tmp_path / "f.png"
    img.write_bytes(b"x")
    monkeypatch.delenv("XBRAIN_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("XBRAIN_LLM_MODEL", "zai-org/glm-5.2")
    monkeypatch.setenv("XBRAIN_LLM_VISION_MODEL", "xiaomi/mimo-v2.5-pro")
    monkeypatch.setattr(xv, "_describe_nanogpt", lambda model, image: f"nanogpt:{model}")
    monkeypatch.setattr(sys, "argv", ["xbrain-vision", str(img)])
    assert xv.main() == 0
    assert "nanogpt:xiaomi/mimo-v2.5-pro" in capsys.readouterr().out
