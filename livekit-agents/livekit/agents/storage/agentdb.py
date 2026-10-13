"""``AgentDBStorage`` (plan 5.7): threads on agentstore through ``_core``'s storage surface.

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
    """Threads on agentstore (plan 5.7), reached through the public API of ``LIVEKIT_URL``'s
    project.

    Args:
        api_key, api_secret: The project's key pair, ``LIVEKIT_API_KEY`` and
            ``LIVEKIT_API_SECRET`` when omitted; the worker mints its own tokens from them.
        lease_ttl: Lease lifetime, rounded up to whole seconds; ``LIVEKIT_AGENTDB_LEASE_TTL``
            (float seconds) when omitted, otherwise the Rust default.
        renew_interval: Renewal interval; ``LIVEKIT_AGENTDB_RENEW_INTERVAL`` (float seconds)
            when omitted, otherwise the Rust default.
        holder: This worker's lease holder id; the worker sets its id once registered.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: timedelta | None = None,
        renew_interval: timedelta | None = None,
        holder: str = "",
    ) -> None:
        api_key = api_key or os.environ.get("LIVEKIT_API_KEY")
        api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        if not api_key or not api_secret:
            raise ValueError(
                "AgentDBStorage needs the project's LIVEKIT_API_KEY and LIVEKIT_API_SECRET"
            )

        self._core = _core.Storage(
            api_key=api_key,
            api_secret=api_secret,
            lease_ttl=_seconds(lease_ttl, "LIVEKIT_AGENTDB_LEASE_TTL"),
            renew_interval=_seconds(renew_interval, "LIVEKIT_AGENTDB_RENEW_INTERVAL"),
        )
        # what a job process rebuilds the storage from (``__reduce__``)
        self._args: dict[str, Any] = {
            "api_key": api_key,
            "api_secret": api_secret,
            "lease_ttl": lease_ttl,
            "renew_interval": renew_interval,
        }
        self.holder = holder
        """This worker's lease holder id: the worker's id, set once it is registered."""

    def __reduce__(self) -> tuple[functools.partial[AgentDBStorage], tuple[()]]:
        return functools.partial(AgentDBStorage, **self._args, holder=self.holder), ()

    @property
    def url(self) -> str:
        """The public API agentstore is reached through."""
        return self._core.url

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
