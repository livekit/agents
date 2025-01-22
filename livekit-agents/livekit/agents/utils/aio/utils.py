import asyncio
import functools


async def gracefully_cancel(*futures: asyncio.Future):
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


def _release_waiter(waiter, *_):
    if not waiter.done():
        waiter.set_result(None)
