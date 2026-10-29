import asyncio
from typing import assert_type

from livekit.agents.utils import aio, log_exceptions


@log_exceptions()
def format_value(value: int, *, prefix: str = "") -> str:
    return prefix + str(value)


@log_exceptions()
async def fetch_value(key: str) -> int:
    return len(key)


async def decorated_results() -> None:
    assert_type(format_value(2, prefix="item-"), str)
    assert_type(await fetch_value("key"), int)


async def heterogeneous_cleanup(number: asyncio.Future[int], text: asyncio.Future[str]) -> None:
    await aio.cancel_and_wait(number, text)


async def task_results() -> None:
    tasks = aio.TaskSet()
    assert_type(tasks.create_task(fetch_value("key")), asyncio.Task[int])
    assert_type(tasks.tasks, set[asyncio.Task[object]])
