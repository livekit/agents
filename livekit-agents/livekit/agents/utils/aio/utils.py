import asyncio
import functools


async def cancel_and_wait(*futures: asyncio.Future):
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

        # ✅ NEW: Safely retrieve exceptions to silence warnings
        for fut in futures:
            if fut.done():
                try:
                    _ = fut.exception()
                except Exception:
                    pass  # Exception already retrieved or not present

def _release_waiter(waiter, *_):
    if not waiter.done():
        waiter.set_result(None)


gracefully_cancel = cancel_and_wait
