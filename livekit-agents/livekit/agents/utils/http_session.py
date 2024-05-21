from __future__ import annotations
import asyncio
import contextvars

import aiohttp

from ..log import logger

_g_session: aiohttp.ClientSession | None = None
_g_loop: asyncio.AbstractEventLoop | None = None

def http_session() -> aiohttp.ClientSession:
    """Optional utility function to avoid having to manually manage an aiohttp.ClientSession lifetime.
    On job processes, this http session will be bound to the main event loop."""

    global _g_session, _g_loop

    if _g_session is not None:
        if _g_loop != asyncio.get_running_loop():
            raise ValueError("http_session is bound to a different event loop")

        return _g_session

    _g_loop = asyncio.get_running_loop()
    _g_session = aiohttp.ClientSession()
    return _g_session
