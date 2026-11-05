# GENERATED FILE -- DO NOT EDIT.
# From #[host(facade)] declarations in lk-agents-abi.

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from .. import _core
from .storage_class import ThreadInfo
from collections.abc import Mapping
from .storage_class import ThreadItemPage
from .storage_class import TaskPage
from .storage_class import TaskState
from .storage_class import TaskRecord
from .storage_class import EventRow


class ThreadReads:
    if TYPE_CHECKING:

        @property
        def _native(self) -> _core._core_ThreadHandle: ...

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
        order: str = "inserted",
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
