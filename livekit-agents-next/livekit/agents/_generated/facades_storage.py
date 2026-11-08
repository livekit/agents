# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] traits in lk-agents-abi.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from .. import _bindings, _core
from ..llm import ChatContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..storage.context import Row, SqlValue
    from .storage_class import (
        EventRow,
        ItemRow,
        TaskOrder,
        TaskPage,
        TaskRecord,
        TaskState,
        ThreadInfo,
        ThreadItemPage,
    )


class Transaction:
    """`storage.Transaction`: application SQL in one transaction
    (docs/context.md#application-records-and-transactions).

    Reads observe committed records. Writes are staged and applied together when the
    transaction ends, so read before the first write: a read after one is refused.
    """

    def __init__(self, native: _core._core_AppTransaction) -> None:
        self._native = native

    async def execute(self, sql: str, params: Sequence[SqlValue] | None = None) -> None:
        """Runs one statement when the transaction commits."""
        await self._native.execute(sql, params)

    async def fetch(self, sql: str, params: Sequence[SqlValue] | None = None) -> list[Row]:
        """The rows of one read."""
        return cast("list[Row]", await self._native.fetch(sql, params))

    async def fetch_one(self, sql: str, params: Sequence[SqlValue] | None = None) -> Row | None:
        """The first row of one read, `None` when it has none."""
        return cast("Row | None", await self._native.fetch_one(sql, params))


class Thread:
    """`ctx.thread`'s reads (docs/threads.md): each opens the thread's database for the call."""

    if TYPE_CHECKING:
        _native: _core._core_ThreadHandle

    @property
    def id(self) -> str:
        """The thread id."""
        return self._native.id

    async def info(self) -> ThreadInfo:
        """The platform's record: scope, attributes, created, last active, expiry."""
        return await self._native.info()

    async def update(
        self,
        *,
        attributes: Mapping[str, str] | None = None,
        idle_ttl_seconds: int | None = None,
    ) -> ThreadInfo:
        """Change the attributes a listing returns, or the idle period."""
        return await self._native.update(attributes=attributes, idle_ttl_seconds=idle_ttl_seconds)

    async def list_items(
        self,
        *,
        agent: str | None = None,
        after: str | None = None,
        limit: int = 100,
    ) -> ThreadItemPage:
        """One page of the thread's committed chat items in insertion order, every lane's or
        `agent`'s; pass the page's `after` to read the next (empty on the last page). The same
        records Cloud's threads API answers, for an application's own read routes.
        """
        return await self._native.list_items(agent=agent, after=after, limit=limit)

    async def list_tasks(
        self,
        *,
        agent: str | None = None,
        status: TaskState | None = None,
        order: TaskOrder = "inserted",
        page_token: str | None = None,
        limit: int = 50,
    ) -> TaskPage:
        """One page of the thread's A2A task records: in insertion order (`order="inserted"`), or
        last updated first (`"updated"`) as A2A `tasks/list` lists them; of `agent`'s lane and
        `status` when given. Pass the page's `next_page_token` to read the next; a token is bound
        to its listing.
        """
        return await self._native.list_tasks(
            agent=agent,
            status=status,
            order=order,
            page_token=page_token,
            limit=limit,
        )

    async def task(self, task_id: str) -> TaskRecord | None:
        """One task record, `None` when this thread has none of that id."""
        return await self._native.task(task_id)

    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]:
        """One page of the journal (docs/internals/storage.md#journal), every lane, in order; pass
        the last `id` to read the next.
        """
        return await self._native.journal(after_id, limit)


class Lane:
    """`thread.lane(agent)`: one agent's lane of a thread (docs/threads.md), its history and its
    tasks.
    """

    def __init__(self, native: _core._core_ThreadHandle, agent: str) -> None:
        self._native = native
        self._agent = agent

    async def history(self) -> ChatContext:
        """The lane's whole history, in order."""
        return ChatContext([x.payload for x in await self._native.history(self._agent)])

    async def tasks(self, *, status: TaskState | None = None, limit: int = 50) -> list[TaskRecord]:
        """The lane's A2A task records, most recent first; only those in `status` when given."""
        return await self._native.tasks(self._agent, status=status, limit=limit)


Transaction.__module__ = "livekit.agents.storage"
Lane.__module__ = "livekit.agents.storage"
