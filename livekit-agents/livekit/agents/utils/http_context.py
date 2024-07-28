from __future__ import annotations

import contextvars
from typing import Callable

import aiohttp

from ..log import logger

_ClientFactory = Callable[[], aiohttp.ClientSession]
_ContextVar = contextvars.ContextVar("agent_http_session")  # type: ignore


def _new_session_ctx() -> _ClientFactory:
    g_session: aiohttp.ClientSession | None = None

    def _new_session() -> aiohttp.ClientSession:
        nonlocal g_session
        if g_session is None:
            logger.debug("http_session(): creating a new httpclient ctx")
            g_session = aiohttp.ClientSession()
        return g_session

    _ContextVar.set(_new_session)  # type: ignore
    return _new_session


def http_session() -> aiohttp.ClientSession:
    """Optional utility function to avoid having to manually manage an aiohttp.ClientSession lifetime.
    On job processes, this http session will be bound to the main event loop.
    """

    val = _ContextVar.get(None)  # type: ignore
    if val is None:
        raise RuntimeError(
            "Attempted to use an http session outside of a job context. This is probably because you are trying to use a plugin without using the agent worker api. You may need to create your own aiohttp.ClientSession, pass it into the plugin constructor as a kwarg, and manage its lifecycle."
        )

    return val()  # type: ignore


async def _close_http_ctx():
    val = _ContextVar.get(None)  # type: ignore
    if val is not None:
        logger.debug("http_session(): closing the httpclient ctx")
        await val().close()  # type: ignore
        _ContextVar.set(None)  # type: ignore
