from __future__ import annotations

import contextlib
import contextvars
import os
import ssl
from collections.abc import AsyncIterator, Callable

import aiohttp
import certifi

from ..log import logger

_ClientFactory = Callable[[], aiohttp.ClientSession]
_ContextVar = contextvars.ContextVar[_ClientFactory | None]("agent_http_session")


def _create_ssl_context() -> ssl.SSLContext:
    """Build the TLS context used by the shared http session.

    Honors the ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` environment overrides, then
    prefers the host's system trust store. When that store is missing or
    unresolvable (e.g. minimal containers without a ca-certificates package),
    certifi's CA bundle is loaded as a fallback so TLS still verifies. This
    mirrors the certifi-backed behavior of the httpx client used by the
    inference LLM, giving consistent TLS trust roots across LLM, STT, and TTS.
    """
    cafile = os.environ.get("SSL_CERT_FILE")
    capath = os.environ.get("SSL_CERT_DIR")
    if cafile or capath:
        return ssl.create_default_context(cafile=cafile, capath=capath)

    ctx = ssl.create_default_context()

    # `create_default_context()` configures the system trust store (eagerly for
    # a cafile, lazily for a hashed capath dir), so cert_store_stats() can't be
    # trusted cross-platform. Check whether the default verify paths actually
    # resolve on disk; if not, the host has no usable system store.
    paths = ssl.get_default_verify_paths()
    has_system_store = bool(
        (paths.cafile and os.path.exists(paths.cafile))
        or (paths.capath and os.path.isdir(paths.capath))
    )
    if not has_system_store:
        ctx.load_verify_locations(cafile=certifi.where())
    return ctx


def _new_session_ctx() -> _ClientFactory:
    g_session: aiohttp.ClientSession | None = None

    def _new_session() -> aiohttp.ClientSession:
        nonlocal g_session
        if g_session is None or g_session.closed:
            logger.debug("http_session(): creating a new httpclient ctx")

            from ..job import get_job_context

            try:
                http_proxy = get_job_context().proc.http_proxy
            except RuntimeError:
                http_proxy = None

            connector = aiohttp.TCPConnector(
                limit_per_host=50,
                keepalive_timeout=120,  # the default is only 15s
                ssl=_create_ssl_context(),
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
            "Attempted to use an http session outside of a job context. This is probably because you are trying to use a plugin without using the agent worker api. "  # noqa: E501
            "If you're running plugins outside the agent worker (e.g. tests or scripts), wrap your code with `async with livekit.agents.utils.http_context.open(): ...`. "  # noqa: E501
            "Alternatively, create your own aiohttp.ClientSession, pass it into the plugin constructor as a kwarg, and manage its lifecycle."  # noqa: E501
        )

    return val()


async def _close_http_ctx() -> None:
    val = _ContextVar.get(None)
    if val is not None:
        logger.debug("http_session(): closing the httpclient ctx")
        await val().close()
        _ContextVar.set(None)


@contextlib.asynccontextmanager
async def open() -> AsyncIterator[aiohttp.ClientSession]:  # noqa: A001
    """Bind a process-local aiohttp.ClientSession to the current asyncio context.

    Use this when running plugins outside a job worker (e.g. tests, scripts,
    notebooks) so that ``http_session()`` returns a usable session inside the
    ``async with`` block. The session is closed and the context is reset on exit.

    If an http session context is already bound (nested call, or already set up
    by the worker), this is a no-op pass-through — the existing session is
    yielded and left untouched on exit.

    Example::

        async with utils.http_context.open():
            async with AgentSession() as session:
                await session.start(MyAgent())
    """
    if _ContextVar.get(None) is not None:
        yield _ContextVar.get()()  # type: ignore[misc]
        return

    factory = _new_session_ctx()
    try:
        yield factory()
    finally:
        await _close_http_ctx()
