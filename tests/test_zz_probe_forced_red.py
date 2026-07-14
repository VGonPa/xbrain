"""Throwaway: forces check.sh to exit 1 to measure STEP-level continue-on-error."""


def test_forced_failure() -> None:
    """Deliberately fails so the gate exits non-zero."""
    assert False, 'forced red: step-level continue-on-error probe'
