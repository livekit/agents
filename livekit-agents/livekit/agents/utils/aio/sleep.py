"""
Resettable sleep implementation for async timeout management.

Provides a cancellable and resettable alternative to asyncio.sleep.
"""

from __future__ import annotations

import asyncio
from typing import Any


def _finish_fut(fut: asyncio.Future[Any]):
    """Internal helper to complete a future if not already done."""
    if fut.cancelled():
        return
    fut.set_result(None)


class SleepFinished(Exception):
    """Raised when attempting to reset a completed/cancelled sleep."""


class Sleep:
    """Resettable sleep implementation with timeout management.
    
    Features:
    - Reset timer while running
    - Cancel mid-sleep
    - Reuse for multiple timeouts
    
    Usage:
        timer = Sleep(5.0)
        try:
            await timer
        except asyncio.CancelledError:
            print("Timer cancelled")
            
        # Reset and reuse
        timer.reset(3.0)
        await timer
    """

    def __init__(self, delay: float) -> None:
        """
        Args:
            delay: Initial timeout in seconds
        """
        self._delay = delay
        self._handler: asyncio.TimerHandle | None = None

    def reset(self, new_delay: float | None = None) -> None:
        """Reset the timer with optional new duration.
        
        Args:
            new_delay: New timeout (uses original if None)
            
        Raises:
            SleepFinished: If sleep has already completed
        """
        if new_delay is None:
            new_delay = self._delay

        self._delay = new_delay

        if self._handler is None:
            return

        if self._handler.cancelled() or self._fut.done():
            raise SleepFinished("Cannot reset completed sleep")

        self._handler.cancel()
        loop = asyncio.get_event_loop()
        self._handler = loop.call_later(new_delay, _finish_fut, self._fut)

    def cancel(self) -> None:
        """Cancel the current sleep immediately."""
        if self._handler is None:
            return

        self._handler.cancel()
        self._fut.cancel()

    async def _sleep(self) -> None:
        """Internal sleep implementation."""
        if self._delay <= 0:
            self._fut = asyncio.Future[None]()
            self._fut.set_result(None)
            return

        loop = asyncio.get_event_loop()
        self._fut = loop.create_future()
        self._handler = loop.call_later(self._delay, _finish_fut, self._fut)

        try:
            await self._fut
        finally:
            self._handler.cancel()

    def __await__(self):
        """Allow await syntax on Sleep instances."""
        return self._sleep().__await__()


def sleep(delay: float) -> Sleep:
    """Create a resettable sleep timer.
    
    Equivalent to Sleep(delay) but matches asyncio.sleep interface.
    """
    return Sleep(delay)
