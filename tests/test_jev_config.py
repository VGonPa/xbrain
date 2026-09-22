# tests/test_jev_config.py — the `[jev]` section of config.toml.
import re
import shutil
from pathlib import Path

import pytest

from xbrain.config import load_config
from xbrain.jev import defaults

REPO_ROOT = Path(__file__).resolve().parent.parent


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


def test_jev_defaults_are_the_owning_modules_constants(tmp_path: Path, monkeypatch):
    """The defaults are IMPORTED, never retyped — rule 5, asserted so it can FAIL.

    `assert cfg.jev_threshold == DEFAULT_THRESHOLD` would pass just as happily against a
    literal typed into `config.py`, because both sides read 0.85 either way. So the owning
    constants are MOVED here and the configured defaults must move with them.
    """
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_MODEL", "moved-model")
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_THRESHOLD", 0.11)
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_FALLBACK_OPTION", "moved-otro")
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_CONCURRENCY", 3)
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_STATE_CHAR_LIMIT", 4242)
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.jev_model == "moved-model"
    assert cfg.jev_threshold == 0.11
    assert cfg.jev_fallback_option == "moved-otro"
    assert cfg.jev_concurrency == 3
    assert cfg.jev_state_char_limit == 4242


def test_config_example_jev_block_is_the_documented_default(tmp_path: Path):
    """`config.toml.example` is the operator's copy of the pinned defaults: it must parse,
    and it must parse to exactly the constants."""
    shutil.copy(REPO_ROOT / "config.toml.example", tmp_path / "config.toml")
    cfg = load_config(tmp_path)
    assert cfg.jev_model == defaults.DEFAULT_MODEL
    assert cfg.jev_threshold == defaults.DEFAULT_THRESHOLD
    assert cfg.jev_fallback_option == defaults.DEFAULT_FALLBACK_OPTION
    assert cfg.jev_concurrency == defaults.DEFAULT_CONCURRENCY
    assert cfg.jev_state_char_limit == defaults.DEFAULT_STATE_CHAR_LIMIT


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
    ("jev", "attr", "expected"),
    [
        ("threshold = 0.0", "jev_threshold", 0.0),
        ("threshold = 1.0", "jev_threshold", 1.0),
        ("threshold = 1", "jev_threshold", 1.0),
        ("concurrency = 1", "jev_concurrency", 1),
        ("state_char_limit = 1", "jev_state_char_limit", 1),
    ],
)
def test_jev_accepts_the_edges_of_every_range(tmp_path: Path, jev: str, attr: str, expected):
    """The bounds are inclusive; `<=` for `<` would silently refuse a legal config."""
    _write_repo(tmp_path, f"[jev]\n{jev}\n")
    assert getattr(load_config(tmp_path), attr) == expected


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("threshold = 1.5", "[jev].threshold must be a number in [0.0, 1.0]"),
        ("threshold = 1.1", "[jev].threshold must be a number in [0.0, 1.0]"),
        ("threshold = -0.1", "[jev].threshold must be a number in [0.0, 1.0]"),
        ("threshold = true", "[jev].threshold must be a number in [0.0, 1.0]"),
        ('threshold = "alto"', "[jev].threshold must be a number in [0.0, 1.0]"),
        ('threshold = "0.9"', "[jev].threshold must be a number in [0.0, 1.0]"),
        ("concurrency = 0", "[jev].concurrency must be an integer >= 1"),
        ("concurrency = -1", "[jev].concurrency must be an integer >= 1"),
        ("concurrency = true", "[jev].concurrency must be an integer >= 1"),
        ("concurrency = 8.9", "[jev].concurrency must be an integer >= 1"),
        ("state_char_limit = 0", "[jev].state_char_limit must be an integer >= 1"),
        ("state_char_limit = -1", "[jev].state_char_limit must be an integer >= 1"),
        ("state_char_limit = true", "[jev].state_char_limit must be an integer >= 1"),
        ('fallback_option = "  "', "[jev].fallback_option must be a non-empty string"),
        ("fallback_option = 0", "[jev].fallback_option must be a non-empty string"),
        ('model = ""', "[jev].model must be a non-empty string"),
        ('model = "   "', "[jev].model must be a non-empty string"),
        ("model = 113", "[jev].model must be a non-empty string"),
    ],
)
def test_jev_section_rejects_bad_values(tmp_path: Path, bad: str, message: str):
    """Every bad value fails when the config LOADS, naming the section and the key —
    never mid-run, and never as a bare CPython coercion error."""
    _write_repo(tmp_path, f"[jev]\n{bad}\n")
    with pytest.raises(ValueError, match=re.escape(f"config.toml: {message}")):
        load_config(tmp_path)


@pytest.mark.parametrize("typo", ["threshhold = 0.99", 'modl = "jev-9"', "concurrancy = 4"])
def test_jev_section_rejects_an_unknown_key(tmp_path: Path, typo: str):
    """A silently ignored typo means the operator reads a report computed at the default
    while looking at a config that says otherwise — every number plausible and wrong."""
    _write_repo(tmp_path, f"[jev]\n{typo}\n")
    with pytest.raises(ValueError, match=re.escape("config.toml: [jev] unknown key")) as excinfo:
        load_config(tmp_path)
    assert typo.split(" =")[0] in str(excinfo.value)
    assert "allowed:" in str(excinfo.value)


def test_a_jev_key_that_is_not_a_table_is_refused_clearly(tmp_path: Path):
    """`jev = "yes"` would otherwise raise a bare AttributeError from inside the loader."""
    (tmp_path / "config.toml").write_text(
        'jev = "yes"\n'
        "[paths]\n"
        'vault = "/tmp/vault"\n'
        'output_subdir = "x"\n'
        'data_dir = "data"\n'
        "[x]\n"
        'handle = "vgonpa"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=re.escape("config.toml: [jev] must be a table")):
        load_config(tmp_path)
