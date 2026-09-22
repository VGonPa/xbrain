# tests/test_jev_env.py
from pathlib import Path

from xbrain.jev.env import typesafe_api_key


def test_env_var_wins_over_dotenv(tmp_path: Path):
    (tmp_path / ".env").write_text("TYPESAFE_API_KEY=from-file\n", encoding="utf-8")
    assert typesafe_api_key(tmp_path, {"TYPESAFE_API_KEY": "from-env"}) == "from-env"


def test_dotenv_is_read_when_env_is_empty(tmp_path: Path):
    (tmp_path / ".env").write_text(
        '# comment\nOTHER=1\nTYPESAFE_API_KEY="ts-123"\n', encoding="utf-8"
    )
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
