"""Standard pytest exit codes and category filtering, exercised in isolated processes."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from .resources import REQUEST_SECONDS, child_process

pytestmark = pytest.mark.unit


async def run_pytest(log: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, str]:
    async with child_process(sys.executable, "-m", "pytest", *args, log=log, env=env) as child:
        async with asyncio.timeout(REQUEST_SECONDS):
            code = await child.process.wait()
    return code, log.read_text()


@pytest.mark.parametrize("selection", ["unit", "e2e"])
async def test_unselected_import_errors_are_not_imported_and_selected_errors_fail(
    tmp_path: Path, selection: str
) -> None:
    local = tmp_path / "test_local.py"
    local.write_text(
        "import pytest\npytestmark = pytest.mark.unit\ndef test_local():\n    assert True\n"
    )
    live = tmp_path / "test_live.py"
    category = "e2e"
    live.write_text(
        f"import pytest\npytestmark = pytest.mark.{category}\n"
        "raise ImportError('selected live module imported')\ndef test_live(): pass\n"
    )
    code, output = await run_pytest(
        tmp_path / "selection.log", "-p", "tests.conftest", str(tmp_path), f"--{selection}", "-q"
    )
    if selection == "unit":
        assert code == 0, output
        assert "1 passed" in output
        assert "selected live module imported" not in output
    else:
        assert code == 2, output
        assert "selected live module imported" in output


async def test_empty_selection_uses_pytest_no_tests_exit_code(tmp_path: Path) -> None:
    code, output = await run_pytest(
        tmp_path / "empty.log",
        "tests/e2e/test_harness.py",
        "--unit",
        "-k",
        "no_such_e2e_journey",
        "-q",
    )
    assert code == 5, output


async def test_selected_e2e_missing_credentials_fails_instead_of_skipping(tmp_path: Path) -> None:
    env = dict(os.environ)
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        env.pop(name, None)
    code, output = await run_pytest(
        tmp_path / "missing-credentials.log",
        "tests/e2e/test_text.py::test_threads_create_update_list_delete",
        "--e2e",
        "-q",
        env=env,
    )
    assert code == 1, output
    assert "selected E2E requires" in output
    assert "skipped" not in output
