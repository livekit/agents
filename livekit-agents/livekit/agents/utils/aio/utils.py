"""
Asynchronous task utilities for graceful cancellation and cleanup.

Provides safe patterns for managing async task lifecycles in LiveKit agents.
"""

import asyncio
import functools


async def cancel_and_wait(*futures: asyncio.Future):
    """Cancel multiple async futures and wait for their completion.
    
    Ensures proper cleanup of cancelled tasks by waiting until they finish.
    
    Args:
        *futures: Async futures to cancel and wait for
        
    Usage:
        task1 = asyncio.create_task(long_running_op())
        task2 = asyncio.create_task(another_op())
        
        # Later...
        await cancel_and_wait(task1, task2)
        
    Note:
        - Will wait even if futures ignore cancellation
        - Safe to call on already completed/cancelled tasks
    """
    loop = asyncio.get_running_loop()
    waiters = []

    # Create waiters for each future to track completion
    for fut in futures:
        waiter = loop.create_future()
        cb = functools.partial(_release_waiter, waiter)
        waiters.append((waiter, cb))
        fut.add_done_callback(cb)
        fut.cancel()  # Initiate cancellation

    try:
        # Wait for all futures to complete (either normally or via cancellation)
        for waiter, _ in waiters:
            await waiter
    finally:
        # Clean up callbacks to prevent memory leaks
        for i, fut in enumerate(futures):
            _, cb = waiters[i]
            fut.remove_done_callback(cb)


def _release_waiter(waiter, *_):
    """Internal helper to mark waiter as complete when future finishes."""
    if not waiter.done():
        waiter.set_result(None)


# Alias for clearer intent in different contexts
gracefully_cancel = cancel_and_wait
