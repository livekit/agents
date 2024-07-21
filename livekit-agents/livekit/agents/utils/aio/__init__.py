import asyncio
from . import debug
from .channel import Chan, ChanClosed, ChanReceiver, ChanSender
from .interval import Interval, interval
from .sleep import Sleep, SleepFinished, sleep
from .task_set import TaskSet


async def gracefully_cancel(*futures: asyncio.Future):
    for f in futures:
        f.cancel()

    await asyncio.gather(*futures, return_exceptions=True)


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
]
