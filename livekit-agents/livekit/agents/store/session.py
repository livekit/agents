"""One session's rows in its conversation's database: what it loads, and what each save writes.

A save writes the items gained, changed or lost since the last one and rewrites the small
mutable part, in one batch; the last save wins.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from ..llm.chat_context import ChatContext, ChatItem
from ..log import logger
from .executor import Executor, Row, Statement, StoreError, Value
from .schema import migrate

if TYPE_CHECKING:
    from ..delegation.delegate import Delegate

SESSION_OWNER = "session"
"""The ``chat_items.owner`` of the session's own history, as opposed to an agent's context."""

_ITEM_ADAPTER: TypeAdapter[ChatItem] = TypeAdapter(ChatItem)


@dataclass
class AgentRecord:
    """One agent of a session: its class by name, and what rebuilds it."""

    agent_id: str
    cls: str
    """``module:qualname``."""
    parent_agent_id: str | None = None
    """The agent an ``AgentTask`` returns to."""
    state: dict[str, Any] | None = None
    """What ``_snapshot_state`` returned, or none when the class cannot be rebuilt."""
    durable_state: bytes = b""
    """The agent's durable tools, pickled; empty when none runs."""
    chat_items: Sequence[ChatItem] = field(default_factory=list)
    """The agent's own context."""


@dataclass
class StoredSession:
    """What a session had written when it was last saved, read back."""

    current_agent_id: str | None
    userdata: Any
    """Plain JSON; the persistence loads it into the handler's userdata type."""
    history: list[ChatItem]
    agents: dict[str, AgentRecord]
    children: dict[str | None, str]
    """Per endpoint, the latest child session this one reached there, to resume on."""


def item_json(item: ChatItem) -> str:
    """One chat item as its row stores it: ``ChatContext.to_dict()``, no audio or images."""
    data = ChatContext([item]).to_dict(exclude_timestamp=False)["items"][0]
    return json.dumps(data)


class _Database:
    """One database's connection, opened on first use and closed when its last session goes."""

    def __init__(self, database_id: str, *, connect: Callable[[], Awaitable[Executor]]) -> None:
        self.database_id = database_id
        self._connect = connect
        self._executor: Executor | None = None
        self._lock = asyncio.Lock()
        self.sessions = 0

    @property
    def executor(self) -> Executor:
        if self._executor is None:
            raise StoreError("closed", f"database {self.database_id} is not open")
        return self._executor

    async def open(self) -> Executor:
        async with self._lock:
            if self._executor is None:
                executor = await self._connect()
                try:
                    await migrate(executor)
                except BaseException:
                    await executor.aclose()
                    raise
                self._executor = executor
            return self._executor

    async def aclose(self) -> None:
        async with self._lock:
            if self._executor is not None:
                executor, self._executor = self._executor, None
                await executor.aclose()


class _Store:
    """What both backends share: one connection per database, one handle per session."""

    def __init__(self) -> None:
        self._databases: dict[str, _Database] = {}

    async def _connect(self, database_id: str) -> Executor:
        raise NotImplementedError

    def session(
        self,
        database_id: str,
        session_id: str,
        *,
        parent: str | None = None,
        endpoint: str | None = None,
    ) -> PersistedSession:
        """A handle on one session's rows, for ``start(persist=)``; ``parent`` is its caller's."""
        if (database := self._databases.get(database_id)) is None:
            database = self._databases[database_id] = _Database(
                database_id, connect=lambda: self._connect(database_id)
            )
        database.sessions += 1
        return PersistedSession(database, session_id, parent=parent, endpoint=endpoint)

    async def aclose(self) -> None:
        for database in self._databases.values():
            await database.aclose()
        self._databases.clear()


class PersistedSession:
    """A handle on one session's rows, from ``session()``, for ``AgentSession.start(persist=)``."""

    def __init__(
        self, database: _Database, session_id: str, *, parent: str | None, endpoint: str | None
    ) -> None:
        self._database = database
        self._session_id = session_id
        self._parent = parent
        self._endpoint = endpoint
        self._released = False
        self._children: dict[str | None, str] = {}
        # per owner, each item's row as last saved, which the next save compares against
        self._saved: dict[str, dict[str, str]] = {}

    @property
    def database_id(self) -> str:
        return self._database.database_id

    @property
    def session_id(self) -> str:
        return self._session_id

    async def load(self) -> StoredSession | None:
        """Read the session back, or create it and return None when it is new."""
        executor = await self._database.open()
        now = time.time()
        created = await executor.exec(
            "INSERT INTO sessions (session_id, parent_session_id, endpoint, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT (session_id) DO NOTHING",
            self._session_id,
            self._parent,
            self._endpoint,
            now,
            now,
        )
        if created.rows_affected == 1:
            return None
        await executor.exec(
            "UPDATE sessions SET closed_at = NULL WHERE session_id = ?", self._session_id
        )

        session: Row = {}
        async for row in executor.query(
            "SELECT * FROM sessions WHERE session_id = ?", self._session_id
        ):
            session = row

        history: list[ChatItem] = []
        agents: dict[str, AgentRecord] = {}
        async for row in executor.query(
            "SELECT * FROM agents WHERE session_id = ? ORDER BY rowid", self._session_id
        ):
            agent_id = str(row["agent_id"])
            durable_state = row["durable_state"]
            agents[agent_id] = AgentRecord(
                agent_id=agent_id,
                cls=str(row["cls"]),
                parent_agent_id=_text(row["parent_agent_id"]),
                state=_json(row["state"]),
                durable_state=durable_state if isinstance(durable_state, bytes) else b"",
                chat_items=[],
            )
        owned: dict[str, list[ChatItem]] = {}
        async for row in executor.query(
            "SELECT owner, item FROM chat_items WHERE session_id = ? ORDER BY created_at, rowid",
            self._session_id,
        ):
            owner, data = str(row["owner"]), str(row["item"])
            item = _ITEM_ADAPTER.validate_json(data)
            owned.setdefault(owner, []).append(item)
            self._saved.setdefault(owner, {})[item.id] = data
        for owner, items in owned.items():
            if owner == SESSION_OWNER:
                history = items
            else:
                agents.setdefault(owner, AgentRecord(agent_id=owner, cls="")).chat_items = items

        async for row in executor.query(
            "SELECT session_id, endpoint FROM sessions WHERE parent_session_id = ? "
            "ORDER BY created_at",
            self._session_id,
        ):
            self._children[_text(row["endpoint"])] = str(row["session_id"])

        return StoredSession(
            current_agent_id=_text(session.get("current_agent_id")),
            userdata=_json(session.get("userdata")),
            history=history,
            agents=agents,
            children=dict(self._children),
        )

    async def save(
        self,
        *,
        current_agent_id: str | None,
        userdata: Any,
        history: Sequence[ChatItem],
        agents: list[AgentRecord],
    ) -> None:
        """Write the items the history and each agent's context gained, changed or lost since
        the last save, and the mutable part, in one batch. ``None`` userdata leaves it as is."""
        statements: list[Statement] = []
        saved: dict[str, dict[str, str]] = {}
        owners = [(SESSION_OWNER, history)] + [(a.agent_id, a.chat_items) for a in agents]
        for owner, items in owners:
            # an item is rewritten whole whenever its row would differ, whatever changed in it
            rows = {item.id: (item_json(item), item.created_at) for item in items}
            base = self._saved.get(owner, {})
            for item_id in base.keys() - rows.keys():
                statements.append(
                    (
                        "DELETE FROM chat_items WHERE session_id = ? AND owner = ? AND item_id = ?",
                        (self._session_id, owner, item_id),
                    )
                )
            for item_id, (data, created_at) in rows.items():
                if base.get(item_id) != data:
                    statements.append(
                        (
                            "INSERT OR REPLACE INTO chat_items (session_id, owner, item_id, item, "
                            "created_at) VALUES (?, ?, ?, ?, ?)",
                            (self._session_id, owner, item_id, data, created_at),
                        )
                    )
            saved[owner] = {item_id: data for item_id, (data, _) in rows.items()}

        statements.append(
            (
                "UPDATE sessions SET current_agent_id = ?, userdata = COALESCE(?, userdata), "
                "updated_at = ? WHERE session_id = ?",
                (
                    current_agent_id,
                    json.dumps(userdata) if userdata is not None else None,
                    time.time(),
                    self._session_id,
                ),
            )
        )
        for agent in agents:
            statements.append(
                (
                    "INSERT OR REPLACE INTO agents (session_id, agent_id, cls, parent_agent_id, "
                    "state, durable_state) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        self._session_id,
                        agent.agent_id,
                        agent.cls,
                        agent.parent_agent_id,
                        json.dumps(agent.state) if agent.state is not None else None,
                        agent.durable_state,
                    ),
                )
            )
        await self._database.executor.batch(*statements)
        # a failed batch keeps the old base, so the next save writes the same difference again
        self._saved.update(saved)

    def resume_delegate(self, delegate: Delegate) -> None:
        """Point a delegate back at the child session this one last had on its endpoint."""
        endpoint = delegate.endpoint
        if (child := self._children.get(endpoint)) is not None and delegate.resume(child):
            logger.debug(
                "resuming the delegate's earlier context",
                extra={"endpoint": endpoint, "context_id": child},
            )

    async def release(self) -> None:
        """Mark the session closed and let it go, closing the database after its last."""
        if self._released:
            return
        self._released = True
        try:
            await self._database.executor.exec(
                "UPDATE sessions SET closed_at = ? WHERE session_id = ?",
                time.time(),
                self._session_id,
            )
        finally:
            database = self._database
            database.sessions = max(database.sessions - 1, 0)
            if database.sessions == 0:
                await database.aclose()


def _text(value: Value | None) -> str | None:
    return None if value is None else str(value)


def _json(value: Value | None) -> Any:
    return None if value is None else json.loads(str(value))


__all__ = [
    "SESSION_OWNER",
    "AgentRecord",
    "PersistedSession",
    "StoredSession",
    "item_json",
]
