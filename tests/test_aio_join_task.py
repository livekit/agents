"""Owned cleanup finishes before cancellation reaches its observer."""

import asyncio

import pytest

from livekit.agents.utils.aio import join_task

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_join_task_preserves_result() -> None:
    assert await join_task(asyncio.create_task(asyncio.sleep(0, result=11))) == 11


@pytest.mark.asyncio
async def test_join_task_repeated_cancellation_waits_for_owner() -> None:
    release = asyncio.Event()
    settled = asyncio.Event()

    async def owned() -> int:
        await release.wait()
        settled.set()
        return 7

    task = asyncio.create_task(owned())
    observer = asyncio.create_task(join_task(task))
    try:
        await asyncio.sleep(0)
        for _ in range(3):
            observer.cancel()
            await asyncio.sleep(0)
            assert not observer.done()
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert settled.is_set()
        assert task.result() == 7
    finally:
        release.set()
        await asyncio.gather(task, observer, return_exceptions=True)


@pytest.mark.asyncio
async def test_join_task_preserves_cleanup_failure_after_cancellation() -> None:
    release = asyncio.Event()

    async def owned() -> None:
        await release.wait()
        raise RuntimeError("expected cleanup failure")

    task = asyncio.create_task(owned())
    observer = asyncio.create_task(join_task(task))
    try:
        await asyncio.sleep(0)
        observer.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(RuntimeError, match="expected cleanup failure") as error:
            await observer
        assert isinstance(error.value.__cause__, asyncio.CancelledError)
    finally:
        release.set()
        await asyncio.gather(task, observer, return_exceptions=True)


@pytest.mark.asyncio
async def test_join_task_preserves_owner_cancellation() -> None:
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await join_task(task)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_join_task_rejects_another_event_loop() -> None:
    other_loop = asyncio.new_event_loop()
    task = other_loop.create_task(asyncio.sleep(0))
    try:
        with pytest.raises(ValueError, match="another event loop"):
            await join_task(task)
    finally:
        await asyncio.to_thread(other_loop.run_until_complete, task)
        other_loop.close()
