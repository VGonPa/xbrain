"""Small HTTP security primitives used by the local dashboard server."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from urllib.parse import urlparse

ACTION_HEADER = "x-xbrain-action"
ACTION_HEADER_VALUE = "1"


class SlidingWindowRateLimiter:
    """In-memory per-key limiter suitable for XBrain's single-process server."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, *, limit: int, window_seconds: float) -> tuple[bool, int]:
        """Return whether an event is allowed and the retry delay when rejected."""
        now = self._clock()
        cutoff = now - window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(1, int(window_seconds - (now - events[0]) + 0.999))
                return False, retry_after
            events.append(now)
            return True, 0


def is_same_origin_request(*, origin: str | None, host: str | None, fetch_site: str | None) -> bool:
    """Reject browser cross-site requests while preserving non-browser CLI access."""
    if fetch_site and fetch_site.lower() in {"cross-site", "same-site"}:
        return False
    if not origin:
        return True
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not host:
        return False
    return parsed.netloc.casefold() == host.casefold()


def has_action_header(value: str | None) -> bool:
    """The custom header forces browsers to preflight mutating requests."""
    return value == ACTION_HEADER_VALUE
