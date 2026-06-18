import asyncio


class AsyncAtomicCounter:
    """Async atomic counter implementation."""

    def __init__(self, initial: int = 0):
        self._value = initial
        self._lock = asyncio.Lock()

    async def increment(self, n: int = 1) -> int:
        async with self._lock:
            self._value += n
            return self._value

    async def decrement(self, n: int = 1) -> int:
        async with self._lock:
            self._value -= n
            return self._value

    async def get(self) -> int:
        async with self._lock:
            return self._value

    async def set(self, value: int) -> None:
        async with self._lock:
            self._value = value

    async def compare_and_swap(self, expected: int, new: int) -> bool:
        async with self._lock:
            if self._value == expected:
                self._value = new
                return True
            return False

    async def get_and_reset(self, reset_value: int = 0) -> int:
        """Atomically read the current value and reset it."""
        async with self._lock:
            prev = self._value
            self._value = reset_value
            return prev

    def get_nowait(self) -> int:
        """Best-effort non-async read — safe if no await between read and use."""
        return self._value
