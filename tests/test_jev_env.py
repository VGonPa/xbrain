# tests/test_jev_env.py — where the TypeSafe key comes from.
from pathlib import Path

import pytest

from xbrain.jev import env as env_module
from xbrain.jev.env import typesafe_api_key

#: Captured at import time, BEFORE `conftest._isolate_typesafe_credentials` redirects it.
#: This module tests the lookup itself, so it restores the real seam — and still goes
#: THROUGH `dotenv_path`, with a `tmp_path` root, so no developer's `.env` is ever read.
_REAL_DOTENV_PATH = env_module.dotenv_path


@pytest.fixture(autouse=True)
def _use_the_real_dotenv_lookup(monkeypatch):
    monkeypatch.setattr("xbrain.jev.env.dotenv_path", _REAL_DOTENV_PATH)


def _write_dotenv(root: Path, text: str, encoding: str = "utf-8") -> None:
    (root / ".env").write_text(text, encoding=encoding)


def test_env_var_wins_over_dotenv(tmp_path: Path):
    _write_dotenv(tmp_path, "TYPESAFE_API_KEY=from-file\n")
    environ = {"TYPESAFE_API_KEY": "from-env"}  # pragma: allowlist secret
    assert typesafe_api_key(tmp_path, environ) == "from-env"


def test_dotenv_is_read_when_env_is_empty(tmp_path: Path):
    dotenv = '# comment\nOTHER=1\nTYPESAFE_API_KEY="ts-123"\n'  # pragma: allowlist secret
    _write_dotenv(tmp_path, dotenv)
    assert typesafe_api_key(tmp_path, {}) == "ts-123"


def test_blank_values_count_as_absent(tmp_path: Path):
    _write_dotenv(tmp_path, "TYPESAFE_API_KEY=\n")
    assert typesafe_api_key(tmp_path, {"TYPESAFE_API_KEY": "  "}) is None


def test_missing_dotenv_is_none(tmp_path: Path):
    assert typesafe_api_key(tmp_path, {}) is None


def test_blank_env_var_falls_through_to_a_populated_dotenv(tmp_path: Path):
    """`export TYPESAFE_API_KEY=` in a shell leaves the var set but empty: the file wins."""
    _write_dotenv(tmp_path, "TYPESAFE_API_KEY=ts-from-file\n")
    assert typesafe_api_key(tmp_path, {"TYPESAFE_API_KEY": "  "}) == "ts-from-file"


def test_the_process_environment_is_read_when_no_mapping_is_given(tmp_path: Path, monkeypatch):
    """`environ=None` is the CLI's actual call shape."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-from-os-environ")
    assert typesafe_api_key(tmp_path) == "ts-from-os-environ"


def test_the_dotenv_location_is_a_seam_the_suite_can_redirect(tmp_path: Path, monkeypatch):
    """`conftest` points this at an empty dir so no test can read a developer's real key."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("TYPESAFE_API_KEY=ts-elsewhere\n", encoding="utf-8")
    monkeypatch.setattr("xbrain.jev.env.dotenv_path", lambda root: elsewhere / ".env")
    assert typesafe_api_key(tmp_path, {}) == "ts-elsewhere"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("TYPESAFE_API_KEY=ts-plain", "ts-plain"),
        ("export TYPESAFE_API_KEY=ts-exported", "ts-exported"),
        ("export   TYPESAFE_API_KEY=ts-spaced", "ts-spaced"),
        ("export\tTYPESAFE_API_KEY=ts-tabbed", "ts-tabbed"),
        ("  TYPESAFE_API_KEY = ts-padded", "ts-padded"),
        ("TYPESAFE_API_KEY=ts-abc # prod key", "ts-abc"),
        ("TYPESAFE_API_KEY=ts-abc\t# prod key", "ts-abc"),
        ("TYPESAFE_API_KEY=ts-abc#nospace", "ts-abc#nospace"),
        ("TYPESAFE_API_KEY='ts-single'", "ts-single"),  # pragma: allowlist secret
        ('TYPESAFE_API_KEY="ts-double"', "ts-double"),  # pragma: allowlist secret
        ('TYPESAFE_API_KEY="ts-abc # inside"', "ts-abc # inside"),  # pragma: allowlist secret
        ("TYPESAFE_API_KEY='ts-unmatched\"", "'ts-unmatched\""),
        ('export TYPESAFE_API_KEY="ts-both" # comment', "ts-both"),  # pragma: allowlist secret
    ],
)
def test_dotenv_line_forms(tmp_path: Path, line: str, expected: str):
    """`export` prefixes, padding, trailing comments after ANY whitespace, and quotes
    stripped only as a matched pair."""
    _write_dotenv(tmp_path, f"{line}\n")
    assert typesafe_api_key(tmp_path, {}) == expected


@pytest.mark.parametrize(
    "line",
    [
        "TYPESAFE_API_KEY=",
        "TYPESAFE_API_KEY=   ",
        "TYPESAFE_API_KEY=  # paste your key here",
        "TYPESAFE_API_KEY= # TODO",
        "TYPESAFE_API_KEY=\t# TODO",
        "TYPESAFE_API_KEY=#nothing-yet",
        'TYPESAFE_API_KEY=""',
        "TYPESAFE_API_KEY=''",
    ],
)
def test_a_value_that_is_blank_or_only_a_comment_counts_as_absent(tmp_path: Path, line: str):
    """`.env.example` says "paste your key here"; that placeholder must never become the
    key — the failure would be a remote auth error pointing at the wrong thing."""
    _write_dotenv(tmp_path, f"{line}\n")
    assert typesafe_api_key(tmp_path, {}) is None


def test_the_last_assignment_wins_like_source(tmp_path: Path):
    """Rotating a key means pasting the new one at the bottom. Taking the first would give
    a permanent 401 against a `.env` that visibly contains the right key."""
    old = "TYPESAFE_API_KEY=ts-OLD-revoked\n"  # pragma: allowlist secret
    new = "TYPESAFE_API_KEY=ts-NEW-current\n"  # pragma: allowlist secret
    _write_dotenv(tmp_path, old + new)
    assert typesafe_api_key(tmp_path, {}) == "ts-NEW-current"


def test_a_later_blank_assignment_also_wins(tmp_path: Path):
    _write_dotenv(tmp_path, "TYPESAFE_API_KEY=ts-old\nTYPESAFE_API_KEY=\n")
    assert typesafe_api_key(tmp_path, {}) is None


def test_crlf_line_endings_are_tolerated(tmp_path: Path):
    """A `.env` written on Windows, or pasted through one."""
    (tmp_path / ".env").write_bytes(b"OTHER=1\r\nTYPESAFE_API_KEY=ts-crlf\r\n")
    assert typesafe_api_key(tmp_path, {}) == "ts-crlf"


def test_a_utf8_bom_is_tolerated(tmp_path: Path):
    """A BOM is not whitespace, so without `utf-8-sig` the operator is told there is no key
    while looking at a `.env` that plainly has one."""
    _write_dotenv(tmp_path, "TYPESAFE_API_KEY=ts-bom\n", encoding="utf-8-sig")
    assert typesafe_api_key(tmp_path, {}) == "ts-bom"


@pytest.mark.parametrize(
    "line", ["MY_TYPESAFE_API_KEY=ts-other", "TYPESAFE_API_KEY_2=ts-other", "OTHER=1"]
)
def test_a_near_miss_key_name_is_not_matched(tmp_path: Path, line: str):
    _write_dotenv(tmp_path, f"{line}\n")
    assert typesafe_api_key(tmp_path, {}) is None
