import asyncio
import functools
from typing import Any


async def cancel_and_wait(*futures: asyncio.Future[Any]) -> None:
    loop = asyncio.get_running_loop()
    waiters = []

    for fut in futures:
        waiter = loop.create_future()
        cb = functools.partial(_release_waiter, waiter)
        waiters.append((waiter, cb))
        fut.add_done_callback(cb)
        fut.cancel()

    try:
        for waiter, _ in waiters:
            await waiter
    finally:
        for i, fut in enumerate(futures):
            _, cb = waiters[i]
            fut.remove_done_callback(cb)


def _release_waiter(waiter: asyncio.Future[Any], *_: Any) -> None:
    if not waiter.done():
        waiter.set_result(None)


gracefully_cancel = cancel_and_wait
