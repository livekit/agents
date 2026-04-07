from __future__ import annotations

import asyncio
import os
import time
from asyncio.base_events import _format_handle  # type: ignore
from typing import Any

from ...log import logger


def hook_slow_callbacks(slow_duration: float) -> None:
    if not os.environ.get("LIVEKIT_AGENTS_DEBUG"):
        logger.debug(
            "hook_slow_callbacks is disabled; set LIVEKIT_AGENTS_DEBUG=1 to enable"
        )
        return

    _run = asyncio.events.Handle._run

    def instrumented(self: Any) -> Any:
        start = time.monotonic()
        val = _run(self)
        dt = time.monotonic() - start
        if dt >= slow_duration:
            logger.warning("Running %s took too long: %.2f seconds", _format_handle(self), dt)
        return val

    asyncio.events.Handle._run = instrumented  # type: ignore
