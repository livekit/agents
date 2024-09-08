import asyncio
import functools

from . import debug, duplex_unix, itertools
from .channel import Chan, ChanClosed, ChanReceiver, ChanSender
from .interval import Interval, interval
from .sleep import Sleep, SleepFinished, sleep
from .task_set import TaskSet


async def gracefully_cancel(*futures: asyncio.Future):
    loop = asyncio.get_running_loop()

    waiters = []

    for fut in futures:
        waiter = loop.create_future()
        waiter.add_done_callback(waiters.remove)
        waiters.append(waiter)
        cb = functools.partial(_release_waiter, waiter)
        fut.add_done_callback(cb)
        fut.cancel()

    try:
        for waiter in waiters:
            await waiter
    finally:
        for waiter in waiters:
            waiter.remove_done_callback(_release_waiter)


def _release_waiter(waiter, *args):
    if not waiter.done():
        waiter.set_result(None)


__all__ = [
    "ChanClosed",
    "Chan",
    "ChanSender",
    "ChanReceiver",
    "channel",
    "Interval",
    "interval",
    "Sleep",
    "SleepFinished",
    "sleep",
    "TaskSet",
    "debug",
    "gracefully_cancel",
    "duplex_unix",
    "itertools",
]
