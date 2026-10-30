"""Thread resolution and the application-facing thread handle."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .. import _core
from .._generated.storage_class import (
    EventRow,
    TaskPage,
    TaskRecord,
    TaskState,
    ThreadInfo,
    ThreadItemPage,
)
from ..llm import ChatContext
from ..log import logger
from ._protocol import history_of
from .backends import Storage
from .errors import StorageError

if TYPE_CHECKING:
    from ..job import JobContext


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


class Thread:
    """``ctx.thread``: the durable thread (docs/threads.md). Reads open the thread's database for
    the call."""

    def __init__(self, storage: Storage, thread_id: str, *, agent_name: str) -> None:
        self._storage = storage
        self._id = thread_id
        self._agent_name = agent_name

    @property
    def id(self) -> str:
        return self._id

    async def info(self) -> ThreadInfo:
        """The platform's record: scope, attributes, created, last active, expiry."""
        return await self._storage.threads.get_thread(self._id)

    async def update(
        self, *, attributes: Mapping[str, str] | None = None, idle_ttl_seconds: int | None = None
    ) -> ThreadInfo:
        """Change the attributes a listing returns, or the idle period."""
        return await self._storage.threads.update_thread(
            self._id,
            attributes=dict(attributes) if attributes is not None else None,
            idle_ttl_seconds=idle_ttl_seconds,
        )

    def lane(self, agent: str) -> Lane:
        """An agent's lane of this thread: its history and its tasks."""
        return Lane(self, agent)

    async def chat_history(self, agent: str | None = None) -> ChatContext:
        """This agent's history, or another agent's."""
        return await self.lane(agent or self._agent_name).history()

    async def list_items(
        self, *, agent: str | None = None, after: str | None = None, limit: int = 100
    ) -> ThreadItemPage:
        """One page of the thread's committed chat items in insertion order, every lane's or
        ``agent``'s; pass the page's ``after`` to read the next (empty on the last page). The
        same records Cloud's threads API answers, for an application's own read routes."""
        core = await self._open()
        try:
            return await core.list_items(agent, after, limit)
        finally:
            core.close()

    async def list_tasks(
        self,
        *,
        agent: str | None = None,
        status: TaskState | None = None,
        order: Literal["inserted", "updated"] = "inserted",
        page_token: str | None = None,
        limit: int = 50,
    ) -> TaskPage:
        """One page of the thread's A2A task records: in insertion order, or last updated first
        as A2A ``tasks/list`` lists them; of ``agent``'s lane and ``status`` when given. Pass
        the page's ``next_page_token`` to read the next; a token is bound to its listing."""
        core = await self._open()
        try:
            return await core.list_tasks(agent, status, order, limit, page_token)
        finally:
            core.close()

    async def task(self, task_id: str) -> TaskRecord | None:
        """One task record, ``None`` when this thread has none of that id."""
        core = await self._open()
        try:
            return await core.task(task_id)
        finally:
            core.close()

    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]:
        """One page of the journal (docs/internals/storage.md#journal), every lane, in order;
        pass the last ``id`` to read the next."""
        core = await self._open()
        try:
            return await core.journal(after_id, limit)
        finally:
            core.close()

    async def _open(self) -> _core.Thread:
        return await self._storage._open(self._id)


class Lane:
    """One agent's lane of a thread (docs/threads.md): ``thread.lane(agent)``."""

    def __init__(self, thread: Thread, agent: str) -> None:
        self._thread = thread
        self._agent = agent

    async def history(self) -> ChatContext:
        """The lane's whole history, in order."""
        core = await self._thread._open()
        try:
            return history_of(await core.history(self._agent))
        finally:
            core.close()

    async def tasks(self, *, status: TaskState | None = None, limit: int = 50) -> list[TaskRecord]:
        """The lane's A2A task records, most recent first; only those in ``status`` when given."""
        core = await self._thread._open()
        try:
            if status is None:
                return await core.tasks(self._agent, limit=limit)
            page = await core.list_tasks(self._agent, status, "updated", limit)
            return page.tasks
        finally:
            core.close()


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


__all__ = ["Lane", "Thread"]
