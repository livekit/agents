"""Owned async task cleanup observes terminal failures."""

import asyncio
import gc

import pytest

from livekit.agents.utils import aio

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("already_finished", [True, False])
async def test_cleanup_observes_failure_even_when_cancellation_loses_the_race(
    already_finished: bool,
) -> None:
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unobserved: list[str] = []
    loop.set_exception_handler(lambda _, context: unobserved.append(context["message"]))

    async def failing_task() -> None:
        try:
            if not already_finished:
                await asyncio.Future[None]()
        finally:
            raise RuntimeError("provider task failed while closing")

    try:
        task = asyncio.create_task(failing_task())
        await asyncio.sleep(0)
        await aio.cancel_and_wait(task)
        del task
        gc.collect()
        await asyncio.sleep(0)
        assert unobserved == []
    finally:
        loop.set_exception_handler(previous_handler)
