from __future__ import annotations

import asyncio

from ...log import logger


def hook_slow_callbacks(slow_duration: float) -> None:
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = slow_duration
    if not loop.get_debug():
        loop.set_debug(True)
        logger.debug("enabled event loop debug mode for slow callback detection")
