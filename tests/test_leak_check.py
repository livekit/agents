from __future__ import annotations

import asyncio

import pytest

from .conftest import _still_pending_after_settle

pytestmark = pytest.mark.unit


async def test_settle_drops_a_task_that_finishes_on_its_own() -> None:
    # Garbage-collector finalization schedules short-lived close tasks onto whichever loop
    # is running (e.g. google-genai's AsyncClient.__del__ -> aclose()). Those are pending
    # the instant the leak check samples the loop, but they finish immediately after, so
    # they are not leaks.
    async def finishes_soon() -> None:
        for _ in range(3):
            await asyncio.sleep(0)

    task = asyncio.create_task(finishes_soon())
    assert not task.done()

    assert await _still_pending_after_settle([task]) == []


async def test_settle_keeps_a_task_that_never_finishes() -> None:
    # a real leak stays pending, and must still be reported
    async def never_finishes() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(never_finishes())

    assert await _still_pending_after_settle([task]) == [task]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
