"""Smoke test for the concurrent-execution machinery (see tests/concurrency.py).

This module is marked ``@pytest.mark.concurrent`` so it always runs concurrently (its whole point
is to exercise the machinery), independent of the ``--concurrent`` switch. It checks the two
things the machinery adds on top of pytest-asyncio-concurrent -- per-test output capture and
per-test caplog -- in a way that would fail if they leaked across the concurrently-running tests.

Everything here is written to pass both concurrently and sequentially (``--no-concurrent``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys

import pytest

# `unit` is the test category (selected by --unit); `concurrent` forces concurrent execution.
pytestmark = [pytest.mark.unit, pytest.mark.concurrent]

_log = logging.getLogger("livekit.agents.smoke")
_SHARED_LOG_LINE = "shared-smoke-log-line"


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
async def test_sleeps_independently(idx: int) -> None:
    # Independent awaits; run concurrently the whole module collapses to ~one sleep.
    # Also confirms parametrized tests are handled as concurrent group members.
    await asyncio.sleep(0.2)
    assert idx in (0, 1, 2, 3)


async def test_stdout_and_stderr_are_captured() -> None:
    # Output is routed to this test's own buffer; on success it stays hidden, and it never
    # bleeds onto another concurrently-running test's report.
    print("smoke-stdout-should-stay-hidden-on-success")
    await asyncio.sleep(0.1)
    print("smoke-stderr-should-stay-hidden-on-success", file=sys.stderr)


async def test_caplog_is_isolated_a(caplog: pytest.LogCaptureFixture) -> None:
    # This test logs the shared line twice; with per-test capture it must see exactly its own
    # two records even though test_caplog_is_isolated_b logs the same line at the same time.
    caplog.set_level(logging.INFO, logger="livekit.agents.smoke")
    _log.info(_SHARED_LOG_LINE)
    await asyncio.sleep(0.2)
    _log.info(_SHARED_LOG_LINE)
    assert [r.message for r in caplog.records].count(_SHARED_LOG_LINE) == 2


async def test_caplog_is_isolated_b(caplog: pytest.LogCaptureFixture) -> None:
    # ...and this one logs it exactly once. If caplog leaked across the group these counts
    # would inflate (a -> 3, b -> 3) and both assertions would fail.
    caplog.set_level(logging.INFO, logger="livekit.agents.smoke")
    await asyncio.sleep(0.1)
    _log.info(_SHARED_LOG_LINE)
    await asyncio.sleep(0.1)
    assert [r.message for r in caplog.records].count(_SHARED_LOG_LINE) == 1


async def test_awaited_task_is_not_a_leak() -> None:
    # A background task that is awaited must not trip fail_on_leaked_tasks, even though other
    # group members have their own tasks in flight on the shared loop at the same time.
    results = await asyncio.gather(*(asyncio.create_task(asyncio.sleep(0.1)) for _ in range(3)))
    assert results == [None, None, None]


async def test_cancelled_task_is_not_a_leak() -> None:
    # A task that is created and then cancelled (a common cleanup pattern) is also not a leak.
    task = asyncio.create_task(asyncio.sleep(3600))
    await asyncio.sleep(0.1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
