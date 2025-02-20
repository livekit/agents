"""
Asynchronous interval timer for periodic task execution.

Provides precise timing control for recurring async operations.
"""

from __future__ import annotations

import asyncio
from typing import Any


def _finish_fut(fut: asyncio.Future[Any]):
    """Internal helper to complete a future if not already done."""
    if fut.cancelled():
        return
    fut.set_result(None)


class Interval:
    """Precision interval timer for async operations.
    
    Features:
    - Maintains fixed interval between ticks
    - Resettable timing
    - Missed interval compensation (delay strategy)
    
    Usage:
        timer = Interval(1.0)  # 1 second interval
        async for tick in timer:
            print(f"Tick {tick} at {time.time()}")
            await process_data()
    """

    def __init__(self, interval: float) -> None:
        """
        Args:
            interval: Time between ticks in seconds
        """
        self._interval = interval
        self._last_sleep = 0.0  # Last sleep completion time
        self._i = 0  # Tick counter
        self._handler: asyncio.TimerHandle | None = None  # Current timer handle

    def reset(self) -> None:
        """Reset the interval timer.
        
        If timer is running, reschedules next tick. If not running,
        resets internal state for next use.
        """
        if self._fut and self._handler and not self._handler.cancelled():
            # Reschedule existing timer
            loop = asyncio.get_event_loop()
            self._handler.cancel()
            self._handler = loop.call_later(self._interval, _finish_fut, self._fut)
        else:
            # Reset internal state
            self._last_sleep = 0

    async def tick(self) -> int:
        """Wait for next interval tick.
        
        Returns:
            int: Number of completed ticks since start
            
        Note:
            Maintains fixed interval even if processing time exceeds interval
        """
        loop = asyncio.get_event_loop()

        if self._last_sleep:
            # Schedule next tick based on remaining time
            self._fut = loop.create_future()
            delay = self._last_sleep - loop.time() + self._interval
            self._handler = loop.call_later(delay, _finish_fut, self._fut)
            try:
                await self._fut
            finally:
                self._handler.cancel()
            self._i += 1

        # Record completion time of this tick
        self._last_sleep = loop.time()
        return self._i

    def __aiter__(self) -> "Interval":
        """Allow async iteration over interval ticks."""
        return self

    async def __anext__(self) -> int:
        """Get next interval tick count."""
        return await self.tick()


def interval(interval: float) -> Interval:
    """Create a new interval timer.
    
    Args:
        interval: Time between ticks in seconds
        
    Returns:
        Interval: Configured timer instance
    """
    return Interval(interval)
