"""Thread resolution (plan 3.3.6, 4.4), ``Ref`` resolution (5.3) and the ``Thread`` handle (7.8).

``resolve_thread`` asks the threads service for the A2A ``contextId``, else creates a thread.
``AgentRegistry`` rebuilds an ``Agent`` by name and an ``AgentTask`` from its class name and
arguments; a ``Ref`` that cannot be rebuilt drops itself and everything above it with a
warning. ``Thread`` is ``ctx.thread``: the durable session with the verbs that reach outside
the turn.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import JsonValue

from .. import _core, utils
from .._core import Threads
from .._generated.storage import CommitRequest, EventRow, EventWrite, TaskWrite
from .._generated.storage_run import Ref, TaskRecord
from .._generated.storage_threads import ThreadInfo
from ..llm import ChatContext
from ..log import logger
from .agentdb import AgentDBStorage
from .protocol import history_of

if TYPE_CHECKING:
    from ..voice.agent import Agent, AgentTask

TaskClass = type["AgentTask[Any, Any]"]  # the task's result and state types, erased here
"""An ``AgentTask`` subclass, as the registry rebuilds one from a ``Ref``."""


class AgentRegistry:
    """The server's registered agents and task classes: what a ``Ref`` rebuilds.

    ``AgentServer`` registers every ``Agent`` it is given and ``server.register(agent)`` adds the
    rest; ``AgentTask`` subclasses are found by class name among ``AgentTask``'s subclasses, which
    exist from import.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._task_classes: dict[str, TaskClass] = {}

    def register(self, agent: Agent | TaskClass) -> None:
        """An ``Agent`` by its name, or an ``AgentTask`` class by its class name."""
        if isinstance(agent, type):
            self._task_classes[agent.__name__] = agent
            return

        name = agent.id
        existing = self._agents.get(name)
        if existing is not None and existing is not agent:
            logger.warning("two agents registered under one name", extra={"agentName": name})
        self._agents[name] = agent

    def register_all(self, agents: Iterable[Agent | TaskClass]) -> None:
        for agent in agents:
            self.register(agent)

    def include(self, other: AgentRegistry) -> None:
        self.register_all([*other._agents.values(), *other._task_classes.values()])

    def rebuild(self, ref: Ref) -> Agent | None:
        """One ``Ref`` as an ``Agent``: by name, or its task class called with its arguments."""
        if ref.args is None and ref.name in self._agents:
            return self._agents[ref.name]

        task_class = self._task_classes.get(ref.name) or self._find_task_class(ref.name)
        if task_class is None:
            return self._agents.get(ref.name)

        try:
            return task_class._rebuild(ref.args)
        except Exception as e:
            logger.warning(
                "an AgentTask could not be rebuilt from its arguments",
                extra={"taskClass": ref.name, "error": str(e)},
            )
            return None

    def _find_task_class(self, name: str) -> TaskClass | None:
        """The ``AgentTask`` subclass of that name, searched depth first."""
        from ..voice.agent import AgentTask

        pending = AgentTask.__subclasses__()
        while pending:
            candidate = pending.pop(0)
            if candidate.__name__ == name:
                return candidate
            pending = candidate.__subclasses__() + pending

        return None

    def resolve(self, refs: Iterable[Ref]) -> list[Agent]:
        """The stack as agents, bottom first; the first failure drops it and everything above."""
        agents: list[Agent] = []

        for ref in refs:
            agent = self.rebuild(ref)
            if agent is None:
                logger.warning(
                    "dropping an agent stack entry the server cannot rebuild, and everything "
                    "above it; the nearest rebuildable ancestor speaks",
                    extra={"refName": ref.name, "hasArgs": ref.args is not None},
                )
                break
            agents.append(agent)

        return agents

    def speaking(self, refs: Iterable[Ref], default: Agent | None = None) -> Agent | None:
        """Who speaks: the top of the rebuilt stack, else ``default``."""
        resolved = self.resolve(refs)
        return resolved[-1] if resolved else default


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

    async def tasks(self, agent: str | None = None) -> list[TaskRecord]:
        """The lane's task rows, newest first."""
        core = await self._open()
        try:
            return await core.tasks(agent or self._agent_name, limit=500)
        finally:
            core.close()

    async def all_tasks(self) -> list[TaskRecord]:
        """Every lane's task rows, oldest first."""
        core = await self._open()
        try:
            records: list[TaskRecord] = []
            for lane in await core.lanes():
                records.extend(await core.tasks(lane.name, limit=500))
        finally:
            core.close()

        records.sort(key=lambda r: (r.updated_at, r.seq))
        return records

    async def background_tasks(self, agent: str | None = None) -> list[TaskRecord]:
        """The session's submitted tasks, rows with a name rather than turns: every lane's, or
        one agent's."""
        records = await self.tasks(agent) if agent else await self.all_tasks()
        return [record for record in records if record.name]

    async def cancel(self, record: TaskRecord, metadata: dict[str, JsonValue]) -> None:
        """``tasks/cancel``: an unfenced commit of the row ``canceled`` under ``metadata`` (plan
        5.5); a running turn yields at its next boundary and writes nothing over it."""
        db = await self._storage.open(self._id)
        try:
            await db.commit(
                CommitRequest(
                    lane=self._agent_name,
                    run_id=f"cancel_{record.id}_{utils.shortuuid()}",
                    tasks=[
                        TaskWrite(
                            task_id=record.id,
                            status="canceled",
                            message_id=record.message_id,
                            name=record.name,
                            status_message=record.status_message,
                            data=record.data.model_copy(update={"metadata": metadata}),
                            internal=record.internal,
                        )
                    ],
                    events=[
                        self._event("run_started", {"task_id": record.id, "channel": "cancel"})
                    ],
                    run_ended=self._event("run_ended", {"outcome": "completed", "reason": None}),
                )
            )
        finally:
            db.close()

    async def journal(self, after_id: int = 0, limit: int = 100) -> list[EventRow]:
        """One page of the session's journal (docs/journal.md), every lane, in order; pass the
        last ``id`` to read the next."""
        db = await self._storage.open(self._id)
        try:
            return await db.events(after_id, limit)
        finally:
            db.close()

    async def update(
        self, *, attributes: Mapping[str, str] | None = None, idle_ttl_seconds: int | None = None
    ) -> ThreadInfo:
        """What a sidebar shows: the attributes, and the idle TTL."""
        return await self._storage.threads.update_thread(
            self._id,
            attributes=dict(attributes) if attributes is not None else None,
            idle_ttl_seconds=idle_ttl_seconds,
        )

    def _event(self, type: str, payload: dict[str, JsonValue]) -> EventWrite:
        return EventWrite(ts_ms=int(time.time() * 1000), type=type, payload=payload)

    async def _open(self) -> _core.Thread:
        return _core.Thread(await self._storage.open(self._id), self._storage.holder)


__all__ = [
    "AgentRegistry",
    "ResolvedThread",
    "Thread",
    "TaskClass",
    "resolve_thread",
]
