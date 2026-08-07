from __future__ import annotations

import asyncio


def hook_slow_callbacks(slow_duration: float) -> None:
    loop = asyncio.get_event_loop()
    loop.slow_callback_duration = slow_duration
    loop.set_debug(True)
