"""Mypy probes — fixtures that exist only to be checked by mypy.

These files are NOT picked up by pytest's normal discovery; they are
imported / type-checked by tests in `tests/test_type_safety.py` which
shells out to mypy and asserts the expected errors are reported.
"""
