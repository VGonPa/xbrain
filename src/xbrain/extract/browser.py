"""Playwright browser session management for X extraction."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

from playwright.async_api import BrowserContext as AsyncBrowserContext
from playwright.async_api import async_playwright
from playwright.sync_api import BrowserContext, sync_playwright

X_LOGIN_URL = "https://x.com/login"

# One place for the launch flags, so the sync and async contexts cannot drift into
# presenting two different browsers to X from the same account.
_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]


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
def x_context(storage_state_path: Path, headless: bool = False) -> Iterator[BrowserContext]:
    """Yield a Playwright context authenticated with the saved X session.

    Defaults to a *visible* (headful) browser: headless Chromium is the most
    easily fingerprinted automation mode, so headful lowers the detection
    surface when scraping your own account. Pass `headless=True` (CLI:
    `--headless`) for unattended runs where no display is available.
    """
    if not storage_state_path.exists():
        raise FileNotFoundError(
            f"No hay sesión guardada en {storage_state_path}. Ejecuta `xbrain login`."
        )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
        context = browser.new_context(storage_state=str(storage_state_path))
        try:
            yield context
        finally:
            browser.close()


@asynccontextmanager
async def x_context_async(
    storage_state_path: Path, headless: bool = False
) -> AsyncIterator[AsyncBrowserContext]:
    """`x_context`, on the async driver — same session, same flags, same headful default.

    Exists for the one workload that needs several tabs making progress at once
    (`refetch_pool`). Playwright's sync API is single-threaded by contract, so the only
    way to hold N tabs open under ONE browser is the async driver; driving N sync
    Playwright instances from N threads would mean N browsers, which is precisely the
    "stop opening browsers" this was built to fix.
    """
    if not storage_state_path.exists():
        raise FileNotFoundError(
            f"No hay sesión guardada en {storage_state_path}. Ejecuta `xbrain login`."
        )
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless, args=_LAUNCH_ARGS)
        context = await browser.new_context(storage_state=str(storage_state_path))
        try:
            yield context
        finally:
            await browser.close()


def is_logged_out(page_url: str) -> bool:
    """True if a navigation landed on a login page (session expired)."""
    return "/login" in page_url or "/i/flow/login" in page_url
