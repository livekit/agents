import asyncio
import time
from asyncio.base_events import _format_handle  # type: ignore

from ..log import logger


def hook_slow_callbacks(slow_duration: float) -> None:
    _run = asyncio.events.Handle._run

    def instrumented(self):
        start = time.monotonic()
        val = _run(self)
        dt = time.monotonic() - start
        if dt >= slow_duration:
            logger.warning(
                "Running %s took too long: %.2f seconds", _format_handle(self), dt
            )
        return val

    asyncio.events.Handle._run = instrumented
