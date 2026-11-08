"""Session storage: ``AgentDBStorage`` and ``SQLiteStorage``, durable thread and lane handles, and
the errors a door's user catches."""

from __future__ import annotations

import functools
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .. import _core
from .._generated import facades_storage as _facades, public_storage as _public
from .._generated.public_storage import *  # noqa: F403
from .._generated.storage_class import LocalStorageOptions
from ..llm import ChatContext
from ..log import logger
from .context import (
    ContextStore,
    Row,
    SqlValue,
    load_context,
    persisted,
    save_context,
)

if TYPE_CHECKING:
    from ..job import JobContext


class Storage:
    """Where threads live: the abstract type ``AgentServer(storage=)`` and
    ``a2a_session(storage=)`` take. Construct ``AgentDBStorage`` or ``SQLiteStorage``."""

    def __init__(self, args: dict[str, Any]) -> None:
        if type(self) is Storage:
            raise TypeError("Storage is abstract: construct AgentDBStorage or SQLiteStorage")
        self._core = _core.Storage(**args)
        # what a job process rebuilds the storage from (``__reduce__``); an omitted holder
        # stays omitted, so a job process takes its own pid's default
        self._args = args

    @property
    def holder(self) -> str:
        """This process's lease holder id: the explicit one, else ``<hostname>-<pid>``."""
        return self._core.holder

    def __reduce__(self) -> tuple[functools.partial[Storage], tuple[()]]:
        return functools.partial(_rebuild, type(self), self._args), ()

    @property
    def threads(self) -> _core.Storage:
        """Thread lifecycle: ``create_thread``, ``get_thread``, ``list_threads``,
        ``update_thread``, ``delete_thread``, ``get_thread_items``, ``get_thread_tasks``."""
        return self._core

    async def _open(self, thread_id: str) -> _core.Thread:
        """The thread's handle, its writes under this worker's holder. Takes no lease."""
        return await self._core.open(thread_id, self.holder)

    async def aclose(self) -> None:
        """Releases the backend's resources; repeat-safe."""
        await self._core.aclose()


def _rebuild(cls: type[Storage], args: dict[str, Any]) -> Storage:
    storage = cls.__new__(cls)
    Storage.__init__(storage, args)
    return storage


class _StorageHandle(Storage):
    """A borrowed native namespace hydrated by the runtime."""

    def __init__(self, native: _core.Storage) -> None:
        self._core = native
        self._args = {}

    def __reduce__(self) -> tuple[functools.partial[Storage], tuple[()]]:
        raise TypeError("borrowed storage handles cannot be copied")


class AgentDBStorage(Storage):
    """Threads on LiveKit Cloud's agentstore, reached through the public API of ``LIVEKIT_URL``'s
    project. It is built before the server runs: ``AgentServer(ws_url=, api_key=, api_secret=)``
    does not configure it.

    Args:
        api_key, api_secret: The project's key pair, ``LIVEKIT_API_KEY`` and
            ``LIVEKIT_API_SECRET`` when omitted; the worker mints its own tokens from them.
        lease_ttl: Lease lifetime; ``LIVEKIT_AGENTDB_LEASE_TTL`` (float seconds) when omitted,
            else 10 seconds.
        renew_interval: Renewal interval; ``LIVEKIT_AGENTDB_RENEW_INTERVAL`` (float seconds)
            when omitted, else 3 seconds. Must be positive and less than half ``lease_ttl``.
        holder: This worker's lease holder id; ``<hostname>-<pid>`` of the process that opens
            threads when omitted.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        lease_ttl: timedelta | None = None,
        renew_interval: timedelta | None = None,
        holder: str | None = None,
    ) -> None:
        args = {
            "api_key": api_key or None,
            "api_secret": api_secret or None,
            "lease_ttl": lease_ttl,
            "renew_interval": renew_interval,
            "holder": holder,
        }
        super().__init__(args)


class SQLiteStorage(Storage):
    """Threads in a local directory: ``registry.sqlite`` and ``threads/<id>.sqlite``.

    Args:
        directory: Defaults to ``~/.livekit/agent-threads/<project-id>``. A relative path
            resolves against the working directory once; job processes receive the absolute path.
        project_id: Local project namespace; defaults to ``"default"``.
        lease_ttl, renew_interval: As for ``AgentDBStorage``, for processes sharing the
            directory.
        holder: This worker's lease holder id; ``<hostname>-<pid>`` when omitted.
    """

    def __init__(
        self,
        *,
        directory: str | os.PathLike[str] | None = None,
        project_id: str = "default",
        lease_ttl: timedelta | None = None,
        renew_interval: timedelta | None = None,
        holder: str | None = None,
    ) -> None:
        args = {
            "local": LocalStorageOptions(
                directory=os.fspath(directory) if directory is not None else None,
                project_id=project_id,
            ),
            "lease_ttl": lease_ttl,
            "renew_interval": renew_interval,
            "holder": holder,
        }
        super().__init__(args)
        self._args["local"] = LocalStorageOptions(
            directory=self._core.location, project_id=project_id
        )


class Thread(_facades.Thread):
    """``ctx.thread``: the durable thread (docs/threads.md). Each read opens the thread's
    database for the call; there is nothing to open or close."""

    def __init__(self, storage: Storage, thread_id: str, *, agent_name: str) -> None:
        self._storage = storage
        self._agent_name = agent_name
        self._native = _core._core_ThreadHandle(
            storage.threads, thread_id, storage.holder, agent_name
        )

    def lane(self, agent: str) -> Lane:
        """An agent's lane of this thread: its history and its tasks."""
        return Lane(self._native, agent)

    async def chat_history(self, agent: str | None = None) -> ChatContext:
        """This agent's history, or another agent's."""
        return await self.lane(agent or self._agent_name).history()


@dataclass(frozen=True)
class _CallThreadEntrypoint:
    """The ``rtc_session`` entrypoint of a server with storage: the call's thread is resolved
    first, the ``thread_id`` of the dispatch metadata, else a new thread, as ``ctx.thread``.
    An unknown or deleted id, or a storage failure, is logged and the call runs with no
    thread."""

    entrypoint: Callable[[JobContext], Awaitable[None]]
    storage: Storage

    async def __call__(self, ctx: JobContext) -> None:
        try:
            thread = await self.storage.threads._core_call_thread(ctx.job.metadata)
            ctx._thread = Thread(self.storage, thread.thread_id, agent_name=ctx.job.agent_name)
        except StorageError as e:
            logger.warning(
                "the call's thread could not be resolved; the call is kept in memory",
                extra={"error": str(e)},
            )
        await self.entrypoint(ctx)


__all__ = [
    "ContextStore",
    "Row",
    "SqlValue",
    "persisted",
    "load_context",
    "save_context",
    "AgentDBStorage",
    "SQLiteStorage",
    "Storage",
    "Thread",
]
__all__ += _public.__all__
