"""One session's rows in its conversation's database: what it loads, appends and checkpoints.

Chat items are appended through a queue the session never waits on; the small mutable
part is rewritten at checkpoints, each fenced by the session's lease.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from ..llm.chat_context import ChatContext, ChatItem
from ..log import logger
from ..utils import shortuuid
from .executor import Executor, Row, Statement, StoreError, Value
from .schema import migrate

if TYPE_CHECKING:
    from ..delegation.delegate import Delegate

SESSION_OWNER = "session"
"""The ``chat_items.owner`` of the session's own history, as opposed to an agent's context."""

LEASE_TTL = 30.0
"""How long a session stays claimed without a checkpoint, and so the longest a restart waits."""

_ITEM_ADAPTER: TypeAdapter[ChatItem] = TypeAdapter(ChatItem)


class LeaseLostError(Exception):
    """Another worker holds this session now, so this one must stop writing it."""


class LeaseHeldError(Exception):
    """Another worker held this session for longer than the wait allowed."""


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
    durable_state: bytes | None = None
    """The agent's durable tools, pickled; empty when none runs, None to leave the row's as is."""
    chat_items: list[ChatItem] = field(default_factory=list)
    """Filled on load; a checkpoint writes items through ``sync`` instead."""


@dataclass
class StoredSession:
    """What a session had written when it was last checkpointed, read back."""

    current_agent_id: str | None
    userdata: Any
    """Plain JSON; the persistence loads it into the handler's userdata type."""
    history: list[ChatItem]
    agents: dict[str, AgentRecord]
    children: dict[str | None, str]
    """Per endpoint, the latest child session this one reached there, to resume on."""


def qualified_name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def import_qualified(name: str) -> Any:
    """The class a ``module:qualname`` names. Raises ``ImportError`` when it is gone."""
    module_name, _, qualname = name.partition(":")
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            raise ImportError(f"{name} does not import") from None
    return target


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

    def __init__(self, *, lease_ttl: float) -> None:
        self._lease_ttl = lease_ttl
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
        return PersistedSession(
            database, session_id, parent=parent, endpoint=endpoint, lease_ttl=self._lease_ttl
        )

    async def aclose(self) -> None:
        for database in self._databases.values():
            await database.aclose()
        self._databases.clear()


class PersistedSession:
    """A handle on one session's rows, from ``session()``, for ``AgentSession.start(persist=)``."""

    def __init__(
        self,
        database: _Database,
        session_id: str,
        *,
        parent: str | None,
        endpoint: str | None,
        lease_ttl: float,
    ) -> None:
        self._database = database
        self._session_id = session_id
        self._parent = parent
        self._endpoint = endpoint
        self._lease_ttl = lease_ttl
        self._lease_owner = shortuuid("lease_")
        self._pending: list[Statement] = []
        self._writer: asyncio.Task[None] | None = None
        self._released = False
        self._children: dict[str | None, str] = {}
        # per owner, each item's fingerprint as last queued, so only a changed item is rewritten
        self._written: dict[str, dict[str, int]] = {}

    @property
    def database_id(self) -> str:
        return self._database.database_id

    @property
    def session_id(self) -> str:
        return self._session_id

    async def load(self) -> StoredSession | None:
        """Claim the session, waiting out a previous owner's lease, and read it back.

        ``None`` means the session is new, and has been created.
        """
        executor = await self._database.open()
        now = time.time()
        created = await executor.exec(
            "INSERT INTO sessions (session_id, parent_session_id, endpoint, created_at, "
            "updated_at, lease_owner, lease_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (session_id) DO NOTHING",
            self._session_id,
            self._parent,
            self._endpoint,
            now,
            now,
            self._lease_owner,
            now + self._lease_ttl,
        )
        if created.rows_affected == 1:
            return None

        deadline = time.monotonic() + self._lease_ttl + 1.0
        while True:
            now = time.time()
            claimed = await executor.exec(
                "UPDATE sessions SET lease_owner = ?, lease_expires_at = ?, closed_at = NULL "
                "WHERE session_id = ? AND (lease_owner IS NULL OR lease_owner = ? "
                "OR lease_expires_at < ?)",
                self._lease_owner,
                now + self._lease_ttl,
                self._session_id,
                self._lease_owner,
                now,
            )
            if claimed.rows_affected == 1:
                break
            expires_at = now
            async for row in executor.query(
                "SELECT lease_expires_at FROM sessions WHERE session_id = ?", self._session_id
            ):
                expires_at = float(row["lease_expires_at"] or now)
            if time.monotonic() > deadline:
                raise LeaseHeldError(f"session {self._session_id} is held by another worker")
            wait = min(max(expires_at - now, 0.0) + 0.05, max(deadline - time.monotonic(), 0.05))
            logger.info(
                "waiting for the previous owner's lease on the session to expire",
                extra={"session_id": self._session_id, "wait": round(wait, 1)},
            )
            await asyncio.sleep(wait)

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
                durable_state=durable_state if isinstance(durable_state, bytes) else None,
            )
        async for row in executor.query(
            "SELECT owner, item FROM chat_items WHERE session_id = ? ORDER BY created_at, rowid",
            self._session_id,
        ):
            item = _ITEM_ADAPTER.validate_json(str(row["item"]))
            owner = str(row["owner"])
            self._written.setdefault(owner, {})[item.id] = hash(item_json(item))
            if owner == SESSION_OWNER:
                history.append(item)
            else:
                agents.setdefault(owner, AgentRecord(agent_id=owner, cls="")).chat_items.append(
                    item
                )

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

    def sync(
        self, items: list[ChatItem], *, owner: str = SESSION_OWNER, prune: bool = False
    ) -> None:
        """Write each item that is new or changed since last queued, again on its id; ``prune``
        also drops the owner's rows no longer in ``items``. Queued; nothing waits on it."""
        written = self._written.setdefault(owner, {})
        for item in items:
            data = item_json(item)
            if written.get(item.id) == hash(data):
                continue
            written[item.id] = hash(data)
            self._enqueue(
                "INSERT OR REPLACE INTO chat_items (session_id, owner, item_id, item, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._session_id, owner, item.id, data, item.created_at),
            )
        if prune:
            for item_id in written.keys() - {item.id for item in items}:
                del written[item_id]
                self._enqueue(
                    "DELETE FROM chat_items WHERE session_id = ? AND owner = ? AND item_id = ?",
                    (self._session_id, owner, item_id),
                )

    def resume_delegate(self, delegate: Delegate) -> None:
        """Point a delegate back at the child session this one last had on its endpoint."""
        endpoint = delegate.endpoint
        if (child := self._children.get(endpoint)) is not None and delegate.resume(child):
            logger.debug(
                "resuming the delegate's earlier context",
                extra={"endpoint": endpoint, "context_id": child},
            )

    async def _fenced_batch(self, statements: list[Statement]) -> None:
        # a stale owner writes held = 0 into _lease_check, which aborts the whole batch
        fence: Statement = (
            "INSERT OR REPLACE INTO _lease_check (id, held) VALUES (1, (SELECT COUNT(*) "
            "FROM sessions WHERE session_id = ? AND lease_owner = ?))",
            (self._session_id, self._lease_owner),
        )
        try:
            await self._database.executor.batch(fence, *statements)
        except StoreError as e:
            if "lease_held" in e.message:
                raise LeaseLostError(
                    f"session {self._session_id} is held by another worker now"
                ) from None
            raise

    async def checkpoint(
        self, *, current_agent_id: str | None, userdata: Any, agents: list[AgentRecord]
    ) -> None:
        """Rewrite the mutable part in one batch and renew the lease, or write nothing and
        raise ``LeaseLostError`` when another worker holds the session. ``None`` userdata
        leaves the stored value as is."""
        await self.flush()
        now = time.time()
        statements: list[Statement] = [
            (
                "UPDATE sessions SET current_agent_id = ?, userdata = COALESCE(?, userdata), "
                "updated_at = ?, lease_expires_at = ? WHERE session_id = ? AND lease_owner = ?",
                (
                    current_agent_id,
                    json.dumps(userdata) if userdata is not None else None,
                    now,
                    now + self._lease_ttl,
                    self._session_id,
                    self._lease_owner,
                ),
            )
        ]
        for agent in agents:
            statements.append(
                (
                    "INSERT INTO agents (session_id, agent_id, cls, parent_agent_id, state, "
                    "durable_state) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (session_id, agent_id) "
                    "DO UPDATE SET cls = excluded.cls, parent_agent_id = excluded.parent_agent_id, "
                    "state = excluded.state, "
                    "durable_state = COALESCE(excluded.durable_state, agents.durable_state)",
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
        await self._fenced_batch(statements)

    async def write_durable_state(self, agent_id: str, *, cls: str, durable_state: bytes) -> None:
        """Write one agent's durable tools, fenced like a checkpoint, once the queue has landed."""
        await self.flush()
        await self._fenced_batch(
            [
                (
                    "INSERT INTO agents (session_id, agent_id, cls, durable_state) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT (session_id, agent_id) "
                    "DO UPDATE SET durable_state = excluded.durable_state",
                    (self._session_id, agent_id, cls, durable_state),
                )
            ]
        )

    async def release(self) -> None:
        """Flush what is queued and let the session go, closing the database after its last."""
        if self._released:
            return
        self._released = True
        try:
            await self.flush()
            await self._database.executor.exec(
                "UPDATE sessions SET lease_owner = NULL, lease_expires_at = NULL, closed_at = ? "
                "WHERE session_id = ? AND lease_owner = ?",
                time.time(),
                self._session_id,
                self._lease_owner,
            )
        finally:
            database = self._database
            database.sessions = max(database.sessions - 1, 0)
            if database.sessions == 0:
                await database.aclose()

    async def flush(self) -> None:
        """Wait for every queued write to land."""
        while self._writer is not None and not self._writer.done():
            await asyncio.shield(self._writer)

    def _enqueue(self, sql: str, params: tuple[Value, ...]) -> None:
        self._pending.append((sql, params))
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(self._write(), name="persisted_session_write")

    async def _write(self) -> None:
        # everything queued since the last write goes as one batch, in order
        while self._pending:
            statements, self._pending = self._pending, []
            try:
                await self._database.executor.batch(*statements)
            except Exception:
                logger.warning(
                    "could not write to the session store",
                    extra={"session_id": self._session_id, "statements": len(statements)},
                    exc_info=True,
                )
                # the owners in the lost batch are rewritten whole at the next sync, and a lost
                # delete is kept under a fingerprint no item has, so the next prune retries it
                chat_rows = [params for sql, params in statements if "INTO chat_items" in sql]
                deletes = [params for sql, params in statements if "FROM chat_items" in sql]
                for params in chat_rows + deletes:
                    self._written.pop(str(params[1]), None)
                for params in deletes:
                    self._written.setdefault(str(params[1]), {})[str(params[2])] = 0


def _text(value: Value | None) -> str | None:
    return None if value is None else str(value)


def _json(value: Value | None) -> Any:
    return None if value is None else json.loads(str(value))


__all__ = [
    "LEASE_TTL",
    "SESSION_OWNER",
    "AgentRecord",
    "LeaseHeldError",
    "LeaseLostError",
    "PersistedSession",
    "StoredSession",
    "import_qualified",
    "item_json",
    "qualified_name",
]
