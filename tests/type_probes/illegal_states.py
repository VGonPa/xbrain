"""Mypy probe: every annotation below MUST be a static type error.

Verified at runtime by ``tests/test_type_safety.py``, which shells out to
``mypy`` on this file and asserts the exact errors fire. The point of the
#20 refactor is *illegal states unrepresentable*; this probe is the only
test that proves the property keeps holding under future edits.

DO NOT add ``# type: ignore`` to these — that would defeat the test.
"""

from xbrain.models import ContentSourceFailure, ContentSourceSuccess


def missing_text_on_success() -> ContentSourceSuccess:
    """`ContentSourceSuccess` requires `text` — mypy must flag the omission."""
    return ContentSourceSuccess(kind="external_article", url="x")


def missing_failure_reason_on_failure() -> ContentSourceFailure:
    """`ContentSourceFailure` requires `failure_reason` — mypy must flag the omission."""
    return ContentSourceFailure(kind="external_article", url="x")


def cannot_read_failure_reason_off_success(src: ContentSourceSuccess) -> str:
    """A `failure_reason` field does not exist on the success variant."""
    return src.failure_reason


def cannot_read_text_off_failure(src: ContentSourceFailure) -> str:
    """A `text` field does not exist on the failure variant."""
    return src.text
