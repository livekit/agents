import asyncio
import contextvars

import aiohttp

from ..log import logger

_ContextVar = contextvars.ContextVar("agent_http_session")


def http_session() -> aiohttp.ClientSession:
    """Optional utility function to avoid having to manually manage an aiohttp.ClientSession lifetime.
    On job processes, this http session will be bound to the main event loop."""

    if asyncio.current_task() is None:
        raise Exception("http_session() must be called within an asyncio task")

    val = _ContextVar.get(None)
    if val is None:
        logger.debug("http_session(): creating a new http client session")
        val = aiohttp.ClientSession()
        _ContextVar.set(val)

    return val
