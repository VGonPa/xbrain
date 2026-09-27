"""How `xbrain jev serve` refuses a request: an HTTP status and a Spanish message for the page.

Raised by the routes (`jev.serve`), the service (`jev.service`) and the pick parsers
(`jev.picks`) alike, so it lives apart from all three and imports none of them.
"""

from __future__ import annotations


class ServeError(Exception):
    """A request refused with an HTTP status and a Spanish message for the page."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def refuse(message: str) -> ServeError:
    """A 400: the request itself is wrong."""
    return ServeError(400, message)
