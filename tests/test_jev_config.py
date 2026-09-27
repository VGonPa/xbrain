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
    assert cfg.jev_serve_max_usd == 1.0
    assert cfg.jev_ask_max_usd == 0.25
    assert cfg.jev_ask_top == 20
    # `jev ask`'s answers, one file per query plus the history, beside the side-car.
    assert cfg.jev_asks_dir == tmp_path / "data" / "jev" / "asks"
    assert cfg.jev_dir == tmp_path / "data" / "jev"
    assert cfg.jev_topics_path == tmp_path / "data" / "jev" / "topics.json"
    # The run log sits beside the side-car: same directory, same gitignored `data/`.
    assert cfg.jev_runs_path == tmp_path / "data" / "jev" / "runs.jsonl"
    # The pass lock sits beside the side-car it protects (`jev.lock.pass_lock`).
    assert cfg.jev_lock_path == tmp_path / "data" / "jev" / ".lock"


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
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_SERVE_MAX_USD", 0.33)
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_ASK_MAX_USD", 0.07)
    monkeypatch.setattr("xbrain.jev.defaults.DEFAULT_ASK_TOP", 13)
    _write_repo(tmp_path)
    cfg = load_config(tmp_path)
    assert cfg.jev_model == "moved-model"
    assert cfg.jev_threshold == 0.11
    assert cfg.jev_fallback_option == "moved-otro"
    assert cfg.jev_concurrency == 3
    assert cfg.jev_state_char_limit == 4242
    assert cfg.jev_serve_max_usd == 0.33
    assert cfg.jev_ask_max_usd == 0.07
    assert cfg.jev_ask_top == 13


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
    assert cfg.jev_serve_max_usd == defaults.DEFAULT_SERVE_MAX_USD
    assert cfg.jev_ask_max_usd == defaults.DEFAULT_ASK_MAX_USD
    assert cfg.jev_ask_top == defaults.DEFAULT_ASK_TOP


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
        ("serve_max_usd = 0.0001", "jev_serve_max_usd", 0.0001),
        ("serve_max_usd = 5", "jev_serve_max_usd", 5.0),
        ("ask_max_usd = 0.0001", "jev_ask_max_usd", 0.0001),
        ("ask_max_usd = 2", "jev_ask_max_usd", 2.0),
        ("ask_top = 1", "jev_ask_top", 1),
        ("ask_top = 500", "jev_ask_top", 500),
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
        ("serve_max_usd = 0", "[jev].serve_max_usd must be a number > 0"),
        ("serve_max_usd = -1", "[jev].serve_max_usd must be a number > 0"),
        ("serve_max_usd = true", "[jev].serve_max_usd must be a number > 0"),
        ('serve_max_usd = "1"', "[jev].serve_max_usd must be a number > 0"),
        ("serve_max_usd = inf", "[jev].serve_max_usd must be a number > 0"),
        ("serve_max_usd = nan", "[jev].serve_max_usd must be a number > 0"),
        ("ask_max_usd = 0", "[jev].ask_max_usd must be a number > 0"),
        ("ask_max_usd = true", "[jev].ask_max_usd must be a number > 0"),
        ("ask_max_usd = inf", "[jev].ask_max_usd must be a number > 0"),
        ("ask_max_usd = nan", "[jev].ask_max_usd must be a number > 0"),
        ("ask_top = 0", "[jev].ask_top must be an integer >= 1"),
        ("ask_top = true", "[jev].ask_top must be an integer >= 1"),
        ("ask_top = 2.5", "[jev].ask_top must be an integer >= 1"),
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


def test_config_toml_accepts_exactly_the_keys_the_defaults_name(tmp_path: Path):
    """The `[jev]` keys are listed once (`defaults.JEV_DEFAULTS`, which the Configuración tab
    reads too); the loader refuses any other and names the allowed ones from that list."""
    _write_repo(tmp_path, "[jev]\nnope = 1\n")

    with pytest.raises(ValueError) as exc:
        load_config(tmp_path)

    assert f"(allowed: {', '.join(sorted(defaults.JEV_DEFAULTS))})" in str(exc.value)
    assert set(defaults.JEV_DEFAULTS) == {
        "model",
        "threshold",
        "fallback_option",
        "concurrency",
        "state_char_limit",
        "serve_max_usd",
        "ask_max_usd",
        "ask_top",
    }


#: A value other than the default for every `[jev]` key, as config.toml would carry it.
_NON_DEFAULT = {
    "threshold": 0.875,
    "model": "jev-9.9.9",
    "fallback_option": "ninguno",
    "concurrency": 3,
    "state_char_limit": 5000,
    "serve_max_usd": 0.25,
    "ask_max_usd": 0.5,
    "ask_top": 7,
}


def test_every_jev_key_config_toml_accepts_is_parsed_into_the_config(tmp_path: Path):
    """A key `JEV_DEFAULTS` lists (so the loader accepts it) but nobody parses would be read
    as the default forever. Every key, a non-default value, read back from `cfg.jev_<key>`
    and from `jev_settings()`, the one dict the dashboard is built from."""
    assert set(_NON_DEFAULT) == set(defaults.JEV_DEFAULTS)
    for key, value in _NON_DEFAULT.items():
        assert value != defaults.JEV_DEFAULTS[key], key
    body = "".join(
        f'{k} = "{v}"\n' if isinstance(v, str) else f"{k} = {v}\n" for k, v in _NON_DEFAULT.items()
    )
    _write_repo(tmp_path, "[jev]\n" + body)

    cfg = load_config(tmp_path)

    for key, value in _NON_DEFAULT.items():
        assert getattr(cfg, f"jev_{key}") == value, key
    assert cfg.jev_settings() == _NON_DEFAULT
    assert list(cfg.jev_settings()) == list(defaults.JEV_DEFAULTS)


def test_the_vocabulary_and_the_jev_page_have_one_path_each(tmp_path: Path):
    _write_repo(tmp_path)

    cfg = load_config(tmp_path)

    assert cfg.vocab_path == cfg.data_dir / "vocab.yaml"
    assert cfg.jev_page_path == cfg.output_dir / "jev.html"
