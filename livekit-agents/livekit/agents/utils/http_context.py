from __future__ import annotations

import contextvars
from typing import Callable, Optional

import aiohttp

from ..log import logger

_ClientFactory = Callable[[], aiohttp.ClientSession]
_ContextVar = contextvars.ContextVar[Optional[_ClientFactory]]("agent_http_session")


def _new_session_ctx() -> _ClientFactory:
    g_session: aiohttp.ClientSession | None = None

    def _new_session() -> aiohttp.ClientSession:
        nonlocal g_session
        if g_session is None:
            logger.debug("http_session(): creating a new httpclient ctx")

            from ..job import get_job_context

            try:
                http_proxy = get_job_context().proc.http_proxy
            except RuntimeError:
                http_proxy = None

            connector = aiohttp.TCPConnector(
                limit_per_host=50,
                keepalive_timeout=120,  # the default is only 15s
            )
            g_session = aiohttp.ClientSession(proxy=http_proxy, connector=connector)
        return g_session

    _ContextVar.set(_new_session)
    return _new_session


def http_session() -> aiohttp.ClientSession:
    """Optional utility function to avoid having to manually manage an aiohttp.ClientSession lifetime.
    On job processes, this http session will be bound to the main event loop.
    """  # noqa: E501

    val = _ContextVar.get(None)
    if val is None:
        raise RuntimeError(
            "Attempted to use an http session outside of a job context. This is probably because you are trying to use a plugin without using the agent worker api. You may need to create your own aiohttp.ClientSession, pass it into the plugin constructor as a kwarg, and manage its lifecycle."  # noqa: E501
        )

    return val()


async def _close_http_ctx() -> None:
    val = _ContextVar.get(None)
    if val is not None:
        logger.debug("http_session(): closing the httpclient ctx")
        await val().close()
        _ContextVar.set(None)
