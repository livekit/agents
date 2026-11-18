from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable

import pytest

from tests.fakes import ReplyTimer


def capture_reply_timers(monkeypatch: pytest.MonkeyPatch, seconds: float) -> list[ReplyTimer]:
    loop = asyncio.get_running_loop()
    original = loop.call_later
    original_at = loop.call_at
    timers: list[ReplyTimer] = []

    def schedule(
        delay: float,
        callback: Callable[..., object],
        *args: object,
        context: contextvars.Context | None = None,
    ) -> asyncio.TimerHandle:
        if delay != seconds:
            return original(delay, callback, *args, context=context)
        timer = ReplyTimer(delay, callback, args)
        timers.append(timer)
        return timer

    def schedule_at(
        when: float,
        callback: Callable[..., object],
        *args: object,
        context: contextvars.Context | None = None,
    ) -> asyncio.TimerHandle:
        # asyncio.wait_for uses asyncio.timeout/call_at on Python 3.12+, but call_later on 3.11.
        if abs(when - loop.time() - seconds) < 0.001:
            timer = ReplyTimer(seconds, callback, args)
            timers.append(timer)
            return timer
        return original_at(when, callback, *args, context=context)

    monkeypatch.setattr(loop, "call_later", schedule)
    monkeypatch.setattr(loop, "call_at", schedule_at)
    return timers
