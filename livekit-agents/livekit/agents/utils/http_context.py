from __future__ import annotations

import contextvars
import threading
from typing import Callable, Optional

import aiohttp

from ..log import logger

_ClientFactory = Callable[[], aiohttp.ClientSession]
_ContextVar = contextvars.ContextVar[Optional[_ClientFactory]]("agent_http_session")


class _SessionManager:
    g_session: aiohttp.ClientSession | None = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def _new_session(cls) -> aiohttp.ClientSession:
        with cls._lock:
            if cls.g_session is None:
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
                cls.g_session = aiohttp.ClientSession(proxy=http_proxy, connector=connector)
            return cls.g_session

    @classmethod
    async def _close_session(cls) -> None:
        session = None
        with cls._lock:
            if cls.g_session is not None:
                session = cls.g_session
                cls.g_session = None  # Set to None first to prevent reuse

        # Close outside the lock to avoid blocking
        if session is not None:
            await session.close()


def _new_session_ctx() -> Callable[[], aiohttp.ClientSession]:
    _ContextVar.set(_SessionManager._new_session)
    return _SessionManager._new_session


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
        await _SessionManager._close_session()
        _ContextVar.set(None)
