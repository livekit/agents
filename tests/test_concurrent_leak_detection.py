"""End-to-end check that leaked-task detection is attributed *per test* under concurrency.

Concurrent group members share one event loop, so ``fail_on_leaked_tasks`` can't diff
``asyncio.all_tasks()`` -- it relies on the contextvar task-ownership in tests/concurrency.py to
know which still-pending task belongs to which test. This module pins that down with pytester
sub-runs: a test that leaks a task must be blamed, a test that cleans up must not, and the
outcome must be identical whether the module runs concurrently or is forced sequential with
``--no-concurrent``. (It is not itself a concurrent module: the test body is sync, and the
machinery is exercised in the nested run.)
"""

from __future__ import annotations

import os
import pathlib

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[1])

# A clean test (awaits its task) and a leaky one (spawns a task it never awaits/cancels).
_TESTS = """
import asyncio

async def test_clean():
    await asyncio.create_task(asyncio.sleep(0.01))

async def test_leaky():
    asyncio.create_task(asyncio.sleep(3600))  # never awaited or cancelled -> leaked
    await asyncio.sleep(0.05)
"""

# Mirrors tests/conftest.py::fail_on_leaked_tasks, exercising the real ownership helpers from
# tests.concurrency (only the thin fixture wrapper is duplicated here).
_CONFTEST = """
import asyncio, pytest
from tests import concurrency

def _ignorable(task):
    try:
        coro = task.get_coro()
        name = getattr(coro, "__qualname__", "") or type(coro).__name__
        if "async_generator_athrow" in name:
            return True
        if "pytest" in (getattr(coro, "__module__", "") or ""):
            return True
        for frame in task.get_stack():
            if "pytest" in frame.f_code.co_filename:
                return True
    except Exception:
        pass
    return False

@pytest.fixture(autouse=True)
async def fail_on_leaked_tasks(request):
    if concurrency.is_concurrent_member(request.node):
        yield
        leaked = [t for t in concurrency.owned_pending_tasks(request.node.nodeid)
                  if not _ignorable(t)]
    else:
        before = set(asyncio.all_tasks())
        yield
        leaked = [t for t in asyncio.all_tasks() - before if not t.done() and not _ignorable(t)]
    if leaked:
        pytest.fail(f"Test leaked {len(leaked)} tasks")
"""


@pytest.mark.parametrize(
    "mode_args", [["--concurrent"], ["--no-concurrent"]], ids=["concurrent", "sequential"]
)
def test_leaked_task_is_attributed_per_test(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, mode_args: list[str]
) -> None:
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([_REPO_ROOT, os.environ.get("PYTHONPATH", "")])
    )
    # The sub-run's mode must come from mode_args alone, not an inherited LK_TEST_CONCURRENCY.
    monkeypatch.delenv("LK_TEST_CONCURRENCY", raising=False)
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(_TESTS)

    result = pytester.runpytest_subprocess(
        "-p", "tests.concurrency", "-o", "asyncio_mode=auto", "-p", "no:cacheprovider", *mode_args
    )

    # test_clean's task is awaited (no leak); test_leaky's task is detected -> its teardown
    # errors. Same outcome whether concurrent or sequential.
    result.assert_outcomes(passed=2, errors=1)
    result.stdout.fnmatch_lines(["*ERROR at teardown of test_leaky*", "*Test leaked 1 tasks*"])
