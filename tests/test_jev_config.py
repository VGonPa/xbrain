# tests/test_jev_config.py
import re
from pathlib import Path

import pytest

from xbrain.config import load_config


def _write_repo(root: Path, jev: str = "") -> None:
    (root / "config.toml").write_text(
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "learnings/x-knowledge"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n' + jev,
        encoding="utf-8",
    )


def test_jev_defaults(tmp_path: Path):
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.jev_model == "jev-latest"
    assert cfg.jev_threshold == 0.85
    assert cfg.jev_fallback_option == "otro"
    assert cfg.jev_concurrency == 8
    assert cfg.jev_state_char_limit == 100_000
    assert cfg.jev_dir == tmp_path / "data" / "jev"
    assert cfg.jev_topics_path == tmp_path / "data" / "jev" / "topics.json"


def test_jev_section_round_trips(tmp_path: Path):
    _write_repo(
        tmp_path,
        "[jev]\n"
        'model = "jev-1.13.0"\n'
        "threshold = 0.9\n"
        'fallback_option = "ninguno"\n'
        "concurrency = 2\n"
        "state_char_limit = 5000\n",
    )
    cfg = load_config(tmp_path)
    assert cfg.jev_model == "jev-1.13.0"
    assert cfg.jev_threshold == 0.9
    assert cfg.jev_fallback_option == "ninguno"
    assert cfg.jev_concurrency == 2
    assert cfg.jev_state_char_limit == 5000


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("threshold = 1.5", "[jev].threshold must be in [0.0, 1.0]"),
        ("concurrency = 0", "[jev].concurrency must be >= 1"),
        ("state_char_limit = 0", "[jev].state_char_limit must be >= 1"),
        ('fallback_option = "  "', "[jev].fallback_option is empty"),
    ],
)
def test_jev_section_rejects_bad_values(tmp_path: Path, bad: str, message: str):
    _write_repo(tmp_path, f"[jev]\n{bad}\n")
    with pytest.raises(ValueError, match=re.escape(message)):
        load_config(tmp_path)
