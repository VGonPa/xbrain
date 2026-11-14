# tests/test_browser.py
from pathlib import Path

import pytest

from xbrain.extract.browser import is_logged_out, x_context


def test_is_logged_out_detects_login_pages():
    assert is_logged_out("https://x.com/login")
    assert is_logged_out("https://x.com/i/flow/login")


def test_is_logged_out_false_for_normal_pages():
    assert not is_logged_out("https://x.com/i/bookmarks")
    assert not is_logged_out("https://x.com/vgonpa")


def test_x_context_requires_saved_session(tmp_path: Path):
    """The README's recommended route is importing the browser session, not
    `xbrain login`; the error must name both, or a new user follows the one the
    README calls unreliable."""
    with pytest.raises(FileNotFoundError) as excinfo:
        with x_context(tmp_path / "missing.json"):
            pass

    message = str(excinfo.value)
    assert "scripts/import_chrome_session.py" in message
    assert "xbrain login" in message
