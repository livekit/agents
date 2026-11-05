"""Session storage: ``AgentDBStorage`` and ``SQLiteStorage``, durable thread and lane handles, and
the errors a door's user catches."""

from __future__ import annotations

import functools
import json
import os
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from .. import _core
from .._generated.storage_class import (
    LocalStorageOptions,
    TaskRecord,
    TaskState,
    ThreadInfo,
)
from .._generated.storage_errors import (
    LeaseLost as LeaseLost,
    NotSerializable as NotSerializable,
    RunRejected as RunRejected,
    StorageError as StorageError,
    StorageUnavailable as StorageUnavailable,
    ThreadNotFound as ThreadNotFound,
    TooLarge as TooLarge,
    VersionConflict as VersionConflict,
)
from .._generated.thread_reads import ThreadReads
from ..llm import ChatContext
from ..log import logger
from .context import (
    ContextStore,
    Row,
    SqlValue,
    Transaction,
    load_context,
    persisted,
    save_context,
)

if TYPE_CHECKING:
    from ..job import JobContext
from collections.abc import Iterable

from .._generated.storage_class import ItemRow
from .._proto import payload_to_item


class Storage:
    """Where threads live: the abstract type ``AgentServer(storage=)`` and
    ``a2a_session(storage=)`` take. Construct ``AgentDBStorage`` or ``SQLiteStorage``."""

    def __init__(self, args: dict[str, Any], holder: str | None) -> None:
        if type(self) is Storage:
            raise TypeError("Storage is abstract: construct AgentDBStorage or SQLiteStorage")
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

    async def _open(self, thread_id: str) -> _core.Thread:
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


class _StorageHandle(Storage):
    """A borrowed native namespace hydrated by the runtime."""

    def __init__(self, native: _core.Storage) -> None:
        self._core = native
        self._args = {}
        self._holder = None

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
        url = os.environ.get("LIVEKIT_URL")
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
            "lease_ttl": lease_ttl.total_seconds() if lease_ttl is not None else None,
            "renew_interval": (
                renew_interval.total_seconds() if renew_interval is not None else None
            ),
        }
        super().__init__(args, holder)
        self._args["local"] = LocalStorageOptions(
            directory=self._core.location, project_id=project_id
        )


@dataclass
class _ResolvedThread:
    thread: ThreadInfo
    created: bool

    @property
    def id(self) -> str:
        return self.thread.thread_id


async def _resolve_thread(
    threads: _core.Storage, *, context_id: str | None = None, scope: str = ""
) -> _ResolvedThread:
    """Exactly one thread for a request (plan 4.4): the one ``context_id`` names, else a new one
    in ``scope``. A thread is the project's, not one agent's: any agent picks it up."""
    if context_id:
        return _ResolvedThread(thread=await threads.get_thread(context_id), created=False)

    return _ResolvedThread(thread=await threads.create_thread(scope=scope), created=True)


class Thread(ThreadReads):
    """``ctx.thread``: the durable thread (docs/threads.md). Each read opens the thread's
    database for the call; there is nothing to open or close."""

    def __init__(self, storage: Storage, thread_id: str, *, agent_name: str) -> None:
        self._storage = storage
        self._agent_name = agent_name
        self._handle = _core._core_ThreadHandle(
            storage.threads, thread_id, storage.holder, agent_name
        )

    @property
    def _native(self) -> _core._core_ThreadHandle:
        return self._handle

    def lane(self, agent: str) -> Lane:
        """An agent's lane of this thread: its history and its tasks."""
        return Lane(self, agent)

    async def chat_history(self, agent: str | None = None) -> ChatContext:
        """This agent's history, or another agent's."""
        return await self.lane(agent or self._agent_name).history()


class Lane:
    """One agent's lane of a thread (docs/threads.md): ``thread.lane(agent)``."""

    def __init__(self, thread: Thread, agent: str) -> None:
        self._native = thread._native
        self._agent = agent

    async def history(self) -> ChatContext:
        """The lane's whole history, in order."""
        return history_of(await self._native.history(self._agent))

    async def tasks(self, *, status: TaskState | None = None, limit: int = 50) -> list[TaskRecord]:
        """The lane's A2A task records, most recent first; only those in ``status`` when given."""
        return await self._native.tasks(self._agent, status=status, limit=limit)


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
            metadata = json.loads(ctx.job.metadata or "{}")
        except ValueError:
            metadata = None
        thread_id = metadata.get("thread_id") if isinstance(metadata, dict) else None
        if not isinstance(thread_id, str):
            thread_id = None
        try:
            thread_id = (await _resolve_thread(self.storage.threads, context_id=thread_id)).id
            ctx._thread = Thread(self.storage, thread_id, agent_name=ctx.job.agent_name)
        except StorageError as e:
            logger.warning(
                "the call's thread could not be resolved; the call is kept in memory",
                extra={"threadId": thread_id, "error": str(e)},
            )
        await self.entrypoint(ctx)


def history_of(rows: Iterable[ItemRow]) -> ChatContext:
    """The stored items as the chat context the model reads."""
    return ChatContext([payload_to_item(row.payload) for row in rows])


__all__ = [
    "ContextStore",
    "Row",
    "SqlValue",
    "Transaction",
    "persisted",
    "load_context",
    "save_context",
    "AgentDBStorage",
    "SQLiteStorage",
    "Storage",
    "LeaseLost",
    "NotSerializable",
    "RunRejected",
    "StorageUnavailable",
    "ThreadNotFound",
    "StorageError",
    "TooLarge",
    "VersionConflict",
    "Lane",
    "Thread",
]
