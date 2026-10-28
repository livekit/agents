"""``Storage`` and its two configurations (docs/internals/storage.md#configuration):
``AgentDBStorage``, LiveKit Cloud's agentstore, and ``SQLiteStorage``, a local directory.

Both configure the one Rust ``_core.Storage``; Python inheritance is not the backend extension
mechanism. Thread metadata is loaded from its backend for every request.
"""

from __future__ import annotations

import functools
import os
import socket
from datetime import timedelta
from pathlib import Path
from typing import Any

from .. import _core
from .._core import Streams, Thread


class Storage:
    """Where threads live: the contract ``AgentServer(storage=)`` and ``a2a_session(storage=)``
    take. Construct ``AgentDBStorage`` or ``SQLiteStorage``."""

    def __init__(self, args: dict[str, Any], holder: str | None) -> None:
        self._core = _core.Storage(**args)
        # what a job process rebuilds the storage from (``__reduce__``)
        self._args = args
        self._holder = holder

    @property
    def holder(self) -> str:
        """This process's lease holder id: the explicit one, else ``<hostname>-<pid>``."""
        return self._holder or f"{socket.gethostname()}-{os.getpid()}"

    def __reduce__(self) -> tuple[functools.partial[Storage], tuple[()]]:
        # a job process computes its own default holder: its pid is not the parent's
        return functools.partial(_rebuild, type(self), self._args, self._holder), ()

    @property
    def threads(self) -> _core.Storage:
        """Thread lifecycle: ``create_thread``, ``get_thread``, ``list_threads``,
        ``update_thread``, ``delete_thread``, ``get_thread_items``, ``get_thread_tasks``."""
        return self._core

    async def streams(self) -> Streams:
        """The response streams a task's subscribers read."""
        return await self._core.streams()

    async def open(self, thread_id: str) -> Thread:
        """The thread's handle, its writes under this worker's holder. Takes no lease."""
        return await self._core.open(thread_id, self.holder)

    async def aclose(self) -> None:
        """Releases the backend's resources; repeat-safe."""
        await self._core.aclose()


def _rebuild(cls: type[Storage], args: dict[str, Any], holder: str | None) -> Storage:
    storage = cls.__new__(cls)
    Storage.__init__(storage, args, holder)
    return storage


def _seconds(value: timedelta | None, variable: str) -> float | None:
    """``value`` in seconds, else the environment variable as float seconds, else ``None``."""
    if value is not None:
        return value.total_seconds()

    raw = os.environ.get(variable)
    return float(raw) if raw else None


class AgentDBStorage(Storage):
    """Threads on LiveKit Cloud's agentstore, reached through the public API of ``LIVEKIT_URL``'s
    project.

    Args:
        url: The project URL, ``LIVEKIT_URL`` when omitted.
        api_key, api_secret: The project's key pair, ``LIVEKIT_API_KEY`` and
            ``LIVEKIT_API_SECRET`` when omitted; the worker mints its own tokens from them.
        lease_ttl: Lease lifetime; ``LIVEKIT_AGENTDB_LEASE_TTL`` (float seconds) when omitted,
            else 10 seconds.
        renew_interval: Renewal interval; ``LIVEKIT_AGENTDB_RENEW_INTERVAL`` (float seconds)
            when omitted, else 3 seconds.
        holder: This worker's lease holder id; ``<hostname>-<pid>`` of the process that opens
            threads when omitted.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: timedelta | None = None,
        renew_interval: timedelta | None = None,
        holder: str | None = None,
    ) -> None:
        url = url or os.environ.get("LIVEKIT_URL")
        api_key = api_key or os.environ.get("LIVEKIT_API_KEY")
        api_secret = api_secret or os.environ.get("LIVEKIT_API_SECRET")
        if not url or not api_key or not api_secret:
            raise ValueError(
                "AgentDBStorage needs the project's LIVEKIT_URL, LIVEKIT_API_KEY and "
                "LIVEKIT_API_SECRET"
            )

        args = {
            "url": url,
            "api_key": api_key,
            "api_secret": api_secret,
            "lease_ttl": _seconds(lease_ttl, "LIVEKIT_AGENTDB_LEASE_TTL"),
            "renew_interval": _seconds(renew_interval, "LIVEKIT_AGENTDB_RENEW_INTERVAL"),
        }
        super().__init__(args, holder)

    @property
    def url(self) -> str:
        """The public API agentstore is reached through."""
        return self._core.location


class SQLiteStorage(Storage):
    """Threads in a local directory: ``registry.sqlite`` and ``threads/<id>.sqlite``.

    Args:
        directory: Required. A relative path resolves against the working directory once, here;
            job processes receive the absolute path unchanged.
        lease_ttl, renew_interval: As for ``AgentDBStorage``, for processes sharing the
            directory.
        holder: This worker's lease holder id; ``<hostname>-<pid>`` when omitted.
    """

    def __init__(
        self,
        *,
        directory: str | os.PathLike[str],
        lease_ttl: timedelta | None = None,
        renew_interval: timedelta | None = None,
        holder: str | None = None,
    ) -> None:
        args = {
            "directory": str(Path(directory).absolute()),
            "lease_ttl": lease_ttl.total_seconds() if lease_ttl is not None else None,
            "renew_interval": (
                renew_interval.total_seconds() if renew_interval is not None else None
            ),
        }
        super().__init__(args, holder)

    @property
    def directory(self) -> str:
        """The absolute directory the threads live in."""
        return self._core.location


__all__ = ["AgentDBStorage", "SQLiteStorage", "Storage"]
