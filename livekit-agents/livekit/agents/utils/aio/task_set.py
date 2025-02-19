"""
Managed collection of async tasks for coordinated lifecycle management.

Provides safe creation and tracking of related async tasks, ensuring proper cleanup.
"""

from __future__ import annotations

import asyncio
from typing import Any, Coroutine, TypeVar

_T = TypeVar("_T")


class TaskSet:
    """Managed collection of async tasks with automatic cleanup.
    
    Features:
    - Automatic task tracking
    - Fire-and-forget task creation
    - Safe cancellation patterns
    
    Usage:
        tasks = TaskSet()
        tasks.create_task(long_running_operation())
        await tasks.close()  # Cancels all pending tasks
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialize TaskSet with optional event loop.
        
        Args:
            loop: Event loop to use (defaults to current)
        """
        self._loop = loop or asyncio.get_event_loop()
        self._set = set[asyncio.Task[Any]]()
        self._closed = False  # Prevent new tasks after closing

    def create_task(
        self, coro: Coroutine[Any, Any, _T], name: str | None = None
    ) -> asyncio.Task[_T]:
        """Create and track a new async task.
        
        Args:
            coro: Coroutine to execute
            name: Optional task name for debugging
            
        Returns:
            asyncio.Task: The created task
            
        Raises:
            RuntimeError: If TaskSet is closed
        """
        if self._closed:
            raise RuntimeError("Cannot create tasks on closed TaskSet")

        task = self._loop.create_task(coro, name=name)
        self._set.add(task)
        task.add_done_callback(self._set.remove)  # Auto-remove completed tasks
        return task

    @property
    def tasks(self) -> set[asyncio.Task[Any]]:
        """Get a copy of currently tracked tasks."""
        return self._set.copy()

    # TODO: should ther eis a close like this?
    # async def close(self):
    #     """Cancel all tasks and wait for completion."""
    #     if self._closed:
    #         return

    #     self._closed = True
    #     tasks = self.tasks
    #     for task in tasks:
    #         task.cancel()

    #     await asyncio.gather(*tasks, return_exceptions=True)
