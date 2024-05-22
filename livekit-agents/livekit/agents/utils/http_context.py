from __future__ import annotations

import contextvars
from typing import Callable

import aiohttp

from ..log import logger

_ClientFactory = Callable[[], aiohttp.ClientSession]
_ContextVar = contextvars.ContextVar("agent_http_session")


def _new_session_ctx() -> _ClientFactory:
    g_session: aiohttp.ClientSession | None = None

    def _new_session() -> aiohttp.ClientSession:
        nonlocal g_session
        if g_session is None:
            logger.debug("http_session(): creating a new httpclient ctx")
            g_session = aiohttp.ClientSession()
        return g_session

    _ContextVar.set(_new_session)
    return _new_session


def http_session() -> aiohttp.ClientSession:
    """Optional utility function to avoid having to manually manage an aiohttp.ClientSession lifetime.
    On job processes, this http session will be bound to the main event loop.
    """

    val = _ContextVar.get(None)
    if val is None:
        raise RuntimeError("no http_session() context available")

    return val()


async def _close_http_ctx():
    val = _ContextVar.get(None)
    if val is not None:
        logger.debug("http_session(): closing the httpclient ctx")
        await val().close()
        _ContextVar.set(None)
