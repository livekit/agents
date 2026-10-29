import asyncio
import functools
from typing import Any, TypeVar

_T = TypeVar("_T")


async def join_task(task: asyncio.Task[_T]) -> _T:
    """Join owned work before propagating cancellation of its observer."""
    if task.get_loop() is not asyncio.get_running_loop():
        raise ValueError("owned cleanup task belongs to another event loop")
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
        except BaseException:
            if not task.done():
                raise  # Preserve observer failures without querying an unsettled result.
            break

    try:
        result = task.result()
    except BaseException as exc:
        if cancellation is not None:
            raise exc from cancellation
        raise
    if cancellation is not None:
        raise cancellation
    return result


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
