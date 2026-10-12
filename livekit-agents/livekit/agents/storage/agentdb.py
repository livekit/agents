"""``AgentDBStorage`` (plan 5.7): threads on agent-db through ``_core``'s storage surface.

One connection per ``open``; ``GetThread`` cached per process for ``GET_THREAD_TTL`` seconds.
"""

from __future__ import annotations

import functools
import os
import time
from datetime import timedelta
from typing import Any

from .. import _core
from .._core import Streams, ThreadDb, Threads
from .._generated.storage_threads import ThreadInfo

GET_THREAD_TTL = 30.0

_thread_cache: dict[str, tuple[float, ThreadInfo]] = {}


def _seconds(value: timedelta | None, variable: str) -> float | None:
    """``value`` in seconds, else the environment variable as float seconds, else ``None``."""
    if value is not None:
        return value.total_seconds()

    raw = os.environ.get(variable)
    return float(raw) if raw else None


class AgentDBStorage:
    """Sessions on agent-db (plan 5.7), the default under LiveKit Cloud and against a local agent-db.

    Args:
        url: The data plane, ``ws://host:port/db``; ``LIVEKIT_AGENTDB_URL`` when omitted.
        streams_url: ``/v1/streams``; ``LIVEKIT_AGENTDB_STREAMS_URL``, else the data plane's
            host over HTTP.
        management_url: The ``AgentDB`` gRPC service; ``LIVEKIT_AGENTDB_MANAGEMENT_URL``, else
            ``streams_url``.
        threads_url: The ``AgentThreads`` gRPC service; ``LIVEKIT_AGENT_THREADS_URL``, else
            ``management_url``. Beside a local agent-db the four are ports 7781, 7780, 7783 and
            7793; in-cluster each is a service name.
        api_key, api_secret: ``LIVEKIT_AGENTDB_API_KEY`` and ``LIVEKIT_AGENTDB_API_SECRET`` when
            omitted, falling back to ``LIVEKIT_API_KEY`` and ``LIVEKIT_API_SECRET``; the worker
            mints its own tokens from them. Under LiveKit Cloud the two pairs are one; a local
            agent-db beside a local livekit-server has keys of its own.
        lease_ttl: Lease lifetime, rounded up to whole seconds; ``LIVEKIT_AGENTDB_LEASE_TTL``
            (float seconds) when omitted, otherwise the Rust default.
        renew_interval: Renewal interval; ``LIVEKIT_AGENTDB_RENEW_INTERVAL`` (float seconds)
            when omitted, otherwise the Rust default.
        holder: This worker's lease holder id; the worker sets its id once registered.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        streams_url: str | None = None,
        management_url: str | None = None,
        threads_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: timedelta | None = None,
        renew_interval: timedelta | None = None,
        holder: str = "",
    ) -> None:
        url = url or os.environ.get("LIVEKIT_AGENTDB_URL")
        if not url:
            raise ValueError("AgentDBStorage needs a url or LIVEKIT_AGENTDB_URL")

        api_key = (
            api_key
            or os.environ.get("LIVEKIT_AGENTDB_API_KEY")
            or os.environ.get("LIVEKIT_API_KEY")
        )
        api_secret = (
            api_secret
            or os.environ.get("LIVEKIT_AGENTDB_API_SECRET")
            or os.environ.get("LIVEKIT_API_SECRET")
        )
        if not api_key or not api_secret:
            raise ValueError(
                "AgentDBStorage needs an API key and secret: LIVEKIT_AGENTDB_API_KEY and "
                "LIVEKIT_AGENTDB_API_SECRET, or LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
            )

        # what a job process rebuilds the storage from (``__reduce__``)
        self._args: dict[str, Any] = {
            "url": url,
            "streams_url": streams_url,
            "management_url": management_url,
            "threads_url": threads_url,
            "api_key": api_key,
            "api_secret": api_secret,
            "lease_ttl": lease_ttl,
            "renew_interval": renew_interval,
        }
        self._core = _core.Storage(
            url,
            streams_url=streams_url or os.environ.get("LIVEKIT_AGENTDB_STREAMS_URL"),
            management_url=management_url or os.environ.get("LIVEKIT_AGENTDB_MANAGEMENT_URL"),
            threads_url=threads_url or os.environ.get("LIVEKIT_AGENT_THREADS_URL"),
            api_key=api_key,
            api_secret=api_secret,
            lease_ttl=_seconds(lease_ttl, "LIVEKIT_AGENTDB_LEASE_TTL"),
            renew_interval=_seconds(renew_interval, "LIVEKIT_AGENTDB_RENEW_INTERVAL"),
        )
        self.holder = holder
        """This worker's lease holder id: the worker's id, set once it is registered."""

    def __reduce__(self) -> tuple[functools.partial[AgentDBStorage], tuple[()]]:
        return functools.partial(AgentDBStorage, **self._args, holder=self.holder), ()

    @property
    def threads(self) -> Threads:
        return self._core.threads

    @property
    def streams(self) -> Streams:
        return self._core.streams

    async def get_thread(self, thread_id: str, *, fresh: bool = False) -> ThreadInfo:
        """``GetThread``, cached per process for ``GET_THREAD_TTL`` seconds."""
        cached = _thread_cache.get(thread_id)
        if cached is not None and not fresh and cached[0] > time.monotonic():
            return cached[1]

        return self._remember(await self._core.threads.get_thread(thread_id))

    async def open(self, thread_id: str) -> ThreadDb:
        """One connection to the thread's database."""
        info = await self.get_thread(thread_id)
        return await self._core.open(info.database_id)

    async def open_database(self, database_id: str) -> ThreadDb:
        """One connection to a database by its id, skipping the threads service."""
        return await self._core.open(database_id)

    def _remember(self, info: ThreadInfo) -> ThreadInfo:
        _thread_cache[info.thread_id] = (time.monotonic() + GET_THREAD_TTL, info)
        return info


__all__ = ["GET_THREAD_TTL", "AgentDBStorage"]
