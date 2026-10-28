"""Thread resolution and the application-facing thread handle."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue

from .. import _core
from .._core import Threads
from .._generated.storage import EventRow
from .._generated.storage_run import TaskPage, TaskRow
from .._generated.storage_threads import ThreadInfo
from ..llm import ChatContext
from ..log import logger
from .agentdb import AgentDBStorage
from .protocol import StorageError, history_of

if TYPE_CHECKING:
    from ..job import JobContext


@dataclass
class ResolvedThread:
    thread: ThreadInfo
    created: bool

    @property
    def id(self) -> str:
        return self.thread.thread_id


async def resolve_thread(
    threads: Threads, *, context_id: str | None = None, scope: str = ""
) -> ResolvedThread:
    """Exactly one thread for a request (plan 4.4): the one ``context_id`` names, else a new one
    in ``scope``. A thread is the project's, not one agent's: any agent picks it up."""
    if context_id:
        return ResolvedThread(thread=await threads.get_thread(context_id), created=False)

    return ResolvedThread(thread=await threads.create_thread(scope=scope), created=True)


class Thread:
    """``ctx.thread``: the durable thread, with the verbs that reach outside the turn. Reads
    open the thread's database for the call."""

    def __init__(self, storage: AgentDBStorage, thread_id: str, *, agent_name: str) -> None:
        self._storage = storage
        self._id = thread_id
        self._agent_name = agent_name

    @property
    def id(self) -> str:
        return self._id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def storage(self) -> AgentDBStorage:
        return self._storage

    async def info(self) -> ThreadInfo:
        return await self._storage.get_thread(self._id, fresh=True)

    async def chat_history(self, agent: str | None = None) -> ChatContext:
        """The whole history of this agent's lane, or another agent's."""
        core = await self._open()
        try:
            return history_of(await core.history(agent or self._agent_name))
        finally:
            core.close()

    async def tasks(self, agent: str | None = None) -> list[TaskRow]:
        """The lane's task rows, newest first."""
        core = await self._open()
        try:
            return await core.tasks(agent or self._agent_name, limit=500)
        finally:
            core.close()

    async def task(self, task_id: str) -> TaskRow | None:
        """One task, whichever agent's it is."""
        core = await self._open()
        try:
            return await core.task(task_id)
        finally:
            core.close()

    async def list_tasks(
        self,
        status: str | None = None,
        updated_after: int = 0,
        page_size: int = 50,
        page_token: str | None = None,
    ) -> TaskPage:
        """One page of every agent's tasks, last updated first (A2A ``tasks/list``)."""
        core = await self._open()
        try:
            return await core.list_tasks(status, updated_after, page_size, page_token)
        finally:
            core.close()

    async def all_tasks(self) -> list[TaskRow]:
        """Every lane's task rows, oldest first."""
        core = await self._open()
        try:
            records: list[TaskRow] = []
            for lane in await core.lanes():
                records.extend(await core.tasks(lane.lane, limit=500))
        finally:
            core.close()

        records.sort(key=lambda r: (r.updated_at, r.seq))
        return records

    async def background_tasks(self, agent: str | None = None) -> list[TaskRow]:
        """The session's submitted tasks, rows with a name rather than turns: every lane's, or
        one agent's."""
        records = await self.tasks(agent) if agent else await self.all_tasks()
        return [record for record in records if record.name]

    async def cancel(self, record: TaskRow, metadata: dict[str, JsonValue]) -> TaskRow | None:
        """``tasks/cancel``: a row not yet terminal goes ``canceled`` with ``metadata`` merged in;
        a running turn yields at its next checkpoint and writes nothing over it. The row after:
        unchanged when it became terminal first."""
        core = await self._open()
        try:
            return await core.cancel_task(self._agent_name, record.task_id, metadata)
        finally:
            core.close()

    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]:
        """One page of the session's journal (docs/internals/storage.md#journal), every lane, in order; pass the
        last ``id`` to read the next."""
        core = await self._open()
        try:
            return await core.events(after_id, limit)
        finally:
            core.close()

    async def update(
        self, *, attributes: Mapping[str, str] | None = None, idle_ttl_seconds: int | None = None
    ) -> ThreadInfo:
        """What a sidebar shows: the attributes, and the idle TTL."""
        return await self._storage.threads.update_thread(
            self._id,
            attributes=dict(attributes) if attributes is not None else None,
            idle_ttl_seconds=idle_ttl_seconds,
        )

    async def _open(self) -> _core.Thread:
        return await self._storage.open(self._id)


@dataclass(frozen=True)
class CallThreadEntrypoint:
    """The ``rtc_session`` entrypoint of a server with storage: the call's thread is resolved
    first, the ``thread_id`` of the dispatch metadata, else a new thread, as ``ctx.thread``.
    A storage failure is logged and the call runs with no thread."""

    entrypoint: Callable[[JobContext], Awaitable[None]]
    storage: AgentDBStorage

    async def __call__(self, ctx: JobContext) -> None:
        if not self.storage.holder:
            self.storage.holder = ctx.worker_id
        try:
            metadata = json.loads(ctx.job.metadata or "{}")
        except ValueError:
            metadata = None
        thread_id = metadata.get("thread_id") if isinstance(metadata, dict) else None
        try:
            if not isinstance(thread_id, str) or not thread_id:
                thread_id = (await resolve_thread(self.storage.threads)).id
            ctx._thread = Thread(self.storage, thread_id, agent_name=ctx.job.agent_name)
        except StorageError as e:
            logger.warning(
                "the call's thread could not be resolved; the call is kept in memory",
                extra={"threadId": thread_id, "error": str(e)},
            )
        await self.entrypoint(ctx)


__all__ = [
    "CallThreadEntrypoint",
    "ResolvedThread",
    "Thread",
    "resolve_thread",
]
