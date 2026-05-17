"""Playwright browser session management for X extraction."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from playwright.sync_api import BrowserContext, sync_playwright

X_LOGIN_URL = "https://x.com/login"


def login(storage_state_path: Path) -> None:
    """Open a visible browser so the user can log in to X by hand.

    The session (cookies + localStorage) is saved to `storage_state_path`
    once the user confirms they have reached their timeline.
    """
    storage_state_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context()
        context.new_page().goto(X_LOGIN_URL)
        print("Inicia sesión en X en la ventana del navegador.")
        print("Cuando veas tu timeline, vuelve aquí y pulsa Enter.")
        input()
        context.storage_state(path=str(storage_state_path))
        browser.close()
    print(f"Sesión guardada en {storage_state_path}")


@contextmanager
def x_context(
    storage_state_path: Path, headless: bool = True
) -> Iterator[BrowserContext]:
    """Yield a Playwright context authenticated with the saved X session."""
    if not storage_state_path.exists():
        raise FileNotFoundError(
            f"No hay sesión guardada en {storage_state_path}. Ejecuta `xbrain login`."
        )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(storage_state=str(storage_state_path))
        try:
            yield context
        finally:
            browser.close()


def is_logged_out(page_url: str) -> bool:
    """True if a navigation landed on a login page (session expired)."""
    return "/login" in page_url or "/i/flow/login" in page_url
