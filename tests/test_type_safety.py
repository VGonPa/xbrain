"""Static-type-safety regression test for the #20 tagged-union refactor.

The whole point of the #20 refactor is *illegal states unrepresentable at
the type level*. This test shells out to ``mypy`` against the
``tests/type_probes/illegal_states.py`` probe and asserts the four
intended errors are reported.

If a future edit weakens the type contract (e.g. a Union[..] sneaks in,
or a required field becomes Optional), one of these checks stops failing
under mypy and this test goes red — that is exactly the regression we want
to catch, because it would silently allow the class of bugs the refactor
is meant to prevent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROBE = Path(__file__).parent / "type_probes" / "illegal_states.py"
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def test_illegal_states_probe_is_rejected_by_mypy():
    """Every annotation in the probe must produce one mypy error.

    Four illegal states, four errors expected:
      1. `ContentSourceSuccess(kind=..., url=...)` missing `text`.
      2. `ContentSourceFailure(kind=..., url=...)` missing `failure_reason`.
      3. Reading `.failure_reason` off a `ContentSourceSuccess`.
      4. Reading `.text` off a `ContentSourceFailure`.
    """
    import pytest

    # Use the SAME Python interpreter that runs pytest — guarantees the
    # mypy executable matches the project's venv and the pydantic plugin
    # resolves. `shutil.which("mypy")` is unreliable: under `uv run` the
    # venv's bin is on PATH but under bare `pytest` it might not be.
    try:
        import mypy  # noqa: F401 - import probe only
    except ImportError:
        pytest.skip("mypy not installed — install dev deps")

    env = {
        **os.environ,
        "MYPYPATH": str(SRC_DIR),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--no-incremental",
            "--show-error-codes",
            "--config-file",
            str(REPO_ROOT / "pyproject.toml"),
            str(PROBE),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
    )

    error_lines = [line for line in result.stdout.splitlines() if "error:" in line]
    expected_errors = {
        # call-arg: missing required field on construction
        'Missing named argument "text" for "ContentSourceSuccess"',
        'Missing named argument "failure_reason" for "ContentSourceFailure"',
        # attr-defined: the field does not exist on the narrowed variant
        '"ContentSourceSuccess" has no attribute "failure_reason"',
        '"ContentSourceFailure" has no attribute "text"',
    }
    matched = {expected for expected in expected_errors if any(expected in e for e in error_lines)}

    assert matched == expected_errors, (
        f"mypy did not report the expected errors.\n"
        f"matched: {matched}\n"
        f"missing: {expected_errors - matched}\n"
        f"actual stdout:\n{result.stdout}\n"
        f"actual stderr:\n{result.stderr}"
    )
    # And mypy must exit non-zero overall.
    assert result.returncode != 0, "mypy returned 0 — illegal states slipped through"
