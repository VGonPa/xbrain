"""Throwaway: forces the gate red to measure how continue-on-error reports."""


def test_forced_failure() -> None:
    """Deliberately fails."""
    assert False, 'forced red for the continue-on-error probe'
