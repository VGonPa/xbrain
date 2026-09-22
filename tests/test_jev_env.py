# tests/test_jev_env.py
from pathlib import Path

import pytest

from xbrain.jev.env import typesafe_api_key


def test_env_var_wins_over_dotenv(tmp_path: Path):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from-file\n", encoding="utf-8")
    environ = {"TYPESAFE_API_KEY": "from-env"}  # pragma: allowlist secret
    assert typesafe_api_key(tmp_path, environ) == "from-env"


def test_dotenv_is_read_when_env_is_empty(tmp_path: Path):
    dotenv = '# comment\nOTHER=1\nTYPESAFE_API_KEY="ts-123"\n'  # pragma: allowlist secret
    (tmp_path / ".env").write_text(dotenv, encoding="utf-8")
    assert typesafe_api_key(tmp_path, {}) == "ts-123"


def test_blank_values_count_as_absent(tmp_path: Path):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=\n", encoding="utf-8")
    assert typesafe_api_key(tmp_path, {"TYPESAFE_API_KEY": "  "}) is None


def test_missing_dotenv_is_none(tmp_path: Path):
    assert typesafe_api_key(tmp_path, {}) is None


def test_dotenv_without_the_key_is_none(tmp_path: Path):
    """A `.env` that carries other keys but not this one reads as absent, not as ''."""
    (tmp_path / ".env").write_text("OTHER=1\n# TYPESAFE_API_KEY=commented\n", encoding="utf-8")
    assert typesafe_api_key(tmp_path, {}) is None


def test_blank_env_var_falls_through_to_a_populated_dotenv(tmp_path: Path):
    """`export TYPESAFE_API_KEY=` in a shell leaves the var set but empty: the file wins."""
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=ts-from-file\n", encoding="utf-8")
    assert typesafe_api_key(tmp_path, {"TYPESAFE_API_KEY": "  "}) == "ts-from-file"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("export TYPESAFE_API_KEY=ts-exported", "ts-exported"),
        ("export   TYPESAFE_API_KEY=ts-spaced", "ts-spaced"),
        ("TYPESAFE_API_KEY=ts-abc # prod key", "ts-abc"),
        ("TYPESAFE_API_KEY=ts-abc#nospace", "ts-abc#nospace"),
        ("TYPESAFE_API_KEY='ts-single'", "ts-single"),  # pragma: allowlist secret
        ('TYPESAFE_API_KEY="ts-double"', "ts-double"),  # pragma: allowlist secret
        ('TYPESAFE_API_KEY="ts-abc # inside"', "ts-abc # inside"),  # pragma: allowlist secret
        ("TYPESAFE_API_KEY='ts-unmatched\"", "'ts-unmatched\""),
        ('export TYPESAFE_API_KEY="ts-both" # comment', "ts-both"),  # pragma: allowlist secret
    ],
)
def test_dotenv_line_forms(tmp_path: Path, line: str, expected: str):
    """`export` prefixes, trailing comments and quotes are handled, quotes only in pairs."""
    (tmp_path / ".env").write_text(f"{line}\n", encoding="utf-8")
    assert typesafe_api_key(tmp_path, {}) == expected
