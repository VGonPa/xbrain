"""Import an X (Twitter) session into XBrain from your real Safari browser.

XBrain needs a logged-in X session to read your bookmarks and tweets. The
built-in ``xbrain login`` opens an automated Chromium window, but X often
blocks automated browsers when the account signs in through Google/SSO.

This helper sidesteps that: it reads the X cookies from the Safari browser you
already use and writes them into ``auth/storage_state.json`` in the Playwright
storage-state format that XBrain expects.

Full Disk Access
----------------
macOS protects Safari's cookie store. The terminal app you run this script
from (Terminal, iTerm, Ghostty, ...) must have *Full Disk Access* granted in
``System Settings -> Privacy & Security -> Full Disk Access``. Without it,
``browser_cookie3.safari(...)`` cannot read the cookie file and this script
prints actionable instructions and exits non-zero.

Usage
-----
1. Log in to X (https://x.com) in Safari -- the normal browser you use.
2. Install the one extra dependency::

       uv pip install browser-cookie3 --index-url https://pypi.org/simple

3. Grant your terminal app Full Disk Access (see above), then restart it.
4. Run this script from anywhere in the repo::

       python scripts/import_safari_session.py

It writes ``auth/storage_state.json`` at the repo root. Re-run it whenever the
session expires.

Notes
-----
- ``auth/storage_state.json`` is gitignored -- the session never leaves your
  machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# auth/ sits at the repo root; this script lives in repo-root/scripts/.
AUTH_PATH = Path(__file__).resolve().parent.parent / "auth" / "storage_state.json"

# X serves the same session under both its current and legacy domains.
X_DOMAINS = ("x.com", "twitter.com")

# Printed when no Safari cookies could be read at all -- almost always a
# missing Full Disk Access grant on the terminal app.
_FDA_MESSAGE = """No se pudieron leer las cookies de Safari.

macOS protege las cookies de Safari tras "Acceso a todo el disco" (Full Disk Access).
Para arreglarlo:
  1. Ajustes del Sistema -> Privacidad y seguridad -> Acceso a todo el disco
  2. Añade tu app de terminal (Terminal, iTerm, Ghostty, ...) y actívala
  3. Reinicia la terminal (o abre una ventana nueva)
  4. Vuelve a ejecutar: uv run python scripts/import_safari_session.py"""


def import_session(output: Path) -> int:
    """Read X cookies from Safari and write a Playwright storage-state file.

    Returns 0 on success, 1 if no ``auth_token`` cookie was found (the signal
    that the user is not logged in to X in Safari), and 1 if no Safari cookies
    could be read at all (almost always a missing Full Disk Access grant).
    """
    try:
        import browser_cookie3
    except ImportError:
        sys.exit(
            "browser-cookie3 is not installed. Run:\n"
            "  uv pip install browser-cookie3 --index-url https://pypi.org/simple"
        )

    # Key by (name, domain, path) so a cookie seen on both domains is not duplicated.
    collected: dict[tuple[str, str, str], dict] = {}
    names: set[str] = set()

    # Track whether *any* safari() call succeeded. If every call raises, the
    # cookie store is unreadable -- almost always a missing Full Disk Access
    # grant rather than a real "not logged in" state.
    any_domain_read = False

    for domain in X_DOMAINS:
        try:
            jar = browser_cookie3.safari(domain_name=domain)
        except Exception as exc:  # noqa: BLE001 - report and try the next domain
            print(f"warn: could not read {domain} cookies: {exc!r}", file=sys.stderr)
            continue
        any_domain_read = True
        for c in jar:
            rest = getattr(c, "_rest", {}) or {}
            http_only = any(k.lower() == "httponly" for k in rest)
            collected[(c.name, c.domain, c.path)] = {
                "name": c.name,
                "value": c.value or "",
                "domain": c.domain,
                "path": c.path or "/",
                "expires": float(c.expires) if c.expires else -1,
                "httpOnly": http_only,
                "secure": bool(c.secure),
                "sameSite": "Lax",
            }
            names.add(c.name)

    # Every safari() call failed -- the cookie store is unreadable. This is the
    # Full Disk Access case; bail out before writing an empty storage state.
    if not any_domain_read:
        print(f"\n{_FDA_MESSAGE}", file=sys.stderr)
        return 1

    cookies = list(collected.values())
    state = {"cookies": cookies, "origins": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2), encoding="utf-8")

    values = {c["name"]: c["value"] for c in cookies}
    auth_token = values.get("auth_token", "")
    print(f"Wrote {len(cookies)} cookies to {output}")
    print(f"  auth_token: {f'OK ({len(auth_token)} chars)' if auth_token else 'MISSING / EMPTY'}")
    print(f"  ct0:        {'OK' if values.get('ct0') else 'MISSING / EMPTY'}")
    print(f"  cookie names: {', '.join(sorted(names)) or '(none)'}")

    if not auth_token:
        print(
            "\nNo `auth_token` cookie found. Log in to X in Safari first, then re-run this script.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import an X session from Safari into XBrain's auth/storage_state.json.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=AUTH_PATH,
        help=f"Where to write the storage-state JSON (default: {AUTH_PATH}).",
    )
    args = parser.parse_args()
    sys.exit(import_session(args.output))


if __name__ == "__main__":
    main()
