"""Asyncio task retention and loop cleanup, without SDK initialization dependencies."""

from __future__ import annotations

import asyncio

# A close nobody awaits (a dropped stream's ``aclose()``), by loop: held until it finishes, since
# asyncio holds a task only weakly, and awaited by ``close_loop``.
_closing: dict[asyncio.AbstractEventLoop, set[asyncio.Task[object]]] = {}
# How long ``close_loop`` gives those closes, the core's teardown bound.
_CLOSING_BOUND = 5.0


def _keep_closing(task: asyncio.Task[object]) -> None:
    loop = task.get_loop()
    held = _closing.setdefault(loop, set())
    held.add(task)

    def done(task: asyncio.Task[object]) -> None:
        held.discard(task)
        if not held and _closing.get(loop) is held:
            del _closing[loop]

    task.add_done_callback(done)


def close_loop(loop: asyncio.AbstractEventLoop) -> list[Exception]:
    """``asyncio.run()``'s teardown of ``loop``, then close it, returning what its tasks raised.

    The closes still running for dropped streams finish first, bounded. Cancelling runs again
    after the generators and the default executor shut down: a task they started (a host call
    the framework is waiting on) is cancelled, never left pending in a closed loop with its
    waiter asleep.
    """
    failures: list[Exception] = []

    def cancel_all() -> None:
        while tasks := [task for task in asyncio.all_tasks(loop) if not task.done()]:
            for task in tasks:
                task.cancel()
            results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            failures.extend(result for result in results if isinstance(result, Exception))

    try:
        if closing := _closing.pop(loop, None):
            loop.run_until_complete(asyncio.wait(closing, timeout=_CLOSING_BOUND))
        cancel_all()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(loop.shutdown_default_executor())
        cancel_all()
    finally:
        loop.close()
    return failures
