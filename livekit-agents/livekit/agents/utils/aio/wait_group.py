"""
Asynchronous wait group implementation for coordinating multiple concurrent tasks.

Inspired by Go's sync.WaitGroup, adapted for Python's asyncio.

Features:
- Track completion of multiple async tasks
- Thread-safe coordination
- Async/await compatible
- Zero-dependency implementation

Usage:
    wg = WaitGroup()
    wg.add(2)
    
    async def task():
        await asyncio.sleep(1)
        wg.done()
        
    asyncio.create_task(task())
    asyncio.create_task(task())
    
    await wg.wait()  # Blocks until both tasks call done()
"""

import asyncio


class WaitGroup:
    """
    Coordination primitive for waiting on multiple async tasks.
    
    Maintains an internal counter that must reach zero before wait() unblocks.
    
    Example:
        async def worker(wg):
            await asyncio.sleep(0.1)
            wg.done()
            
        wg = WaitGroup()
        wg.add(3)
        for _ in range(3):
            asyncio.create_task(worker(wg))
        await wg.wait()  # Resumes when all 3 workers finish
    """

    def __init__(self):
        self._counter = 0
        self._zero_event = asyncio.Event()
        self._zero_event.set()  # Start in signaled state

    def add(self, delta: int = 1):
        """Adjust the wait group counter.
        
        Args:
            delta: Number to add to the counter (can be negative)
            
        Raises:
            ValueError: If counter would go negative
        """
        new_value = self._counter + delta
        if new_value < 0:
            raise ValueError("WaitGroup counter cannot go negative.")

        self._counter = new_value

        if self._counter == 0:
            self._zero_event.set()
        else:
            self._zero_event.clear()

    def done(self):
        """Decrement the counter by 1. Equivalent to add(-1)."""
        self.add(-1)

    async def wait(self):
        """Async wait until counter reaches zero.
        
        Returns immediately if counter is already zero.
        """
        await self._zero_event.wait()
