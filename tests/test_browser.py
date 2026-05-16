# tests/test_browser.py
from pathlib import Path

import pytest

from xkb.extract.browser import is_logged_out, x_context


def test_is_logged_out_detects_login_pages():
    assert is_logged_out("https://x.com/login")
    assert is_logged_out("https://x.com/i/flow/login")


def test_is_logged_out_false_for_normal_pages():
    assert not is_logged_out("https://x.com/i/bookmarks")
    assert not is_logged_out("https://x.com/vgonpa")


def test_x_context_requires_saved_session(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="xkb login"):
        with x_context(tmp_path / "missing.json"):
            pass
