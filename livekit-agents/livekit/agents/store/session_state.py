"""One session's rows in a conversation database: what it loads, appends and checkpoints.

A ``SessionState`` is a handle bound to a conversation and a session id, not a snapshot. Chat
items are appended as they land, through an ordered queue the conversation never waits on;
the small mutable part is rewritten at checkpoints, each fenced by the session's lease so a
worker that lost the session cannot overwrite the one that took it.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import pickle
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import TypeAdapter

from ..llm.chat_context import ChatContext, ChatItem
from ..log import logger
from ..utils import shortuuid
from .executor import Row, Statement, StoreError, Value

if TYPE_CHECKING:
    from .conversation import Conversation

SessionKind = Literal["voice", "text", "a2a"]
TaskStatus = Literal["running", "done", "error", "cancelled", "interrupted"]
TaskOrigin = Literal["llm", "code"]

SESSION_OWNER = "session"
"""The ``chat_items.owner`` of the session's own history, as opposed to an agent's context."""

LEASE_TTL = 30.0
"""How long a session stays claimed without a checkpoint renewing it. A worker restarted
after a crash waits at most this long before it can take the session back."""

INTERRUPTED_OUTPUT = "the call was interrupted before it finished; its outcome is unknown"

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
    tools: list[str] | None = None
    chat_items: list[ChatItem] = field(default_factory=list)
    """Filled on load; a checkpoint writes items through ``append`` instead."""


@dataclass
class TaskRecord:
    call_id: str
    name: str
    arguments: str | None
    status: TaskStatus
    started_at: float
    ended_at: float | None = None
    output: str | None = None
    is_error: bool = False
    origin: TaskOrigin = "llm"


@dataclass
class StoredSession:
    """What a session had written when it was last checkpointed, read back."""

    session_id: str
    kind: SessionKind
    parent_session_id: str | None
    endpoint: str | None
    current_agent_id: str | None
    userdata: Any
    """Decoded into its class when that class still imports, else plain JSON. Pickled
    userdata stays bytes here: it may name agents, which the session rebuilds first."""
    userdata_encoding: str | None
    tools: list[str] | None
    history: list[ChatItem]
    agents: dict[str, AgentRecord]
    interrupted: list[TaskRecord]
    """Calls that were running when the previous owner stopped; now ``interrupted``."""
    created_at: float


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


class SessionState:
    """One session in a conversation database: load it, append to it, checkpoint it.

    Made by ``Conversation.session()`` and handed to ``AgentSession.start(state=...)``.
    """

    def __init__(
        self,
        conversation: Conversation,
        session_id: str,
        *,
        kind: SessionKind,
        parent: str | None = None,
        endpoint: str | None = None,
        lease_ttl: float,
    ) -> None:
        self._conversation = conversation
        self._session_id = session_id
        self._kind: SessionKind = kind
        self._parent = parent
        self._endpoint = endpoint
        self._lease_ttl = lease_ttl
        self._lease_owner = shortuuid("lease_")
        self._pending: list[Statement] = []
        self._writer: asyncio.Task[None] | None = None
        self._pickle_warned = False
        self._released = False

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def kind(self) -> SessionKind:
        return self._kind

    @property
    def lease_owner(self) -> str:
        return self._lease_owner

    async def load(self) -> StoredSession | None:
        """Claim the session and read it back, or create it. ``None`` means it is new.

        Waits out a previous owner's lease, which is what a restart after a crash meets.
        """
        executor = await self._conversation.open()
        now = time.time()
        created = await executor.exec(
            "INSERT INTO sessions (session_id, parent_session_id, kind, endpoint, created_at, "
            "updated_at, lease_owner, lease_expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (session_id) DO NOTHING",
            self._session_id,
            self._parent,
            self._kind,
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

        return await self._read()

    async def _read(self) -> StoredSession:
        executor = self._conversation.executor
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
            agents[agent_id] = AgentRecord(
                agent_id=agent_id,
                cls=str(row["cls"]),
                parent_agent_id=_text(row["parent_agent_id"]),
                state=_json(row["state_json"]),
                tools=_json(row["tools_json"]),
            )
        async for row in executor.query(
            "SELECT owner, item_json FROM chat_items WHERE session_id = ? "
            "ORDER BY created_at, rowid",
            self._session_id,
        ):
            item = _ITEM_ADAPTER.validate_json(str(row["item_json"]))
            if row["owner"] == SESSION_OWNER:
                history.append(item)
            else:
                owner = str(row["owner"])
                agents.setdefault(owner, AgentRecord(agent_id=owner, cls="")).chat_items.append(
                    item
                )

        interrupted: list[TaskRecord] = []
        async for row in executor.query(
            "SELECT * FROM tasks WHERE session_id = ? AND status = 'running' ORDER BY started_at",
            self._session_id,
        ):
            interrupted.append(
                TaskRecord(
                    call_id=str(row["call_id"]),
                    name=str(row["name"]),
                    arguments=_text(row["arguments"]),
                    status="interrupted",
                    started_at=float(row["started_at"]),  # type: ignore[arg-type]
                    origin=row["origin"],  # type: ignore[arg-type]
                )
            )
        if interrupted:
            # the lease is ours now, so whatever is still running belonged to a worker that died
            now = time.time()
            await executor.exec(
                "UPDATE tasks SET status = 'interrupted', ended_at = ?, output = ?, is_error = 1 "
                "WHERE session_id = ? AND status = 'running'",
                now,
                INTERRUPTED_OUTPUT,
                self._session_id,
            )
            for task in interrupted:
                task.ended_at, task.output, task.is_error = now, INTERRUPTED_OUTPUT, True

        userdata, has_userdata = self._decode_userdata(session)
        return StoredSession(
            session_id=self._session_id,
            kind=session.get("kind") or self._kind,  # type: ignore[arg-type]
            parent_session_id=_text(session.get("parent_session_id")),
            endpoint=_text(session.get("endpoint")),
            current_agent_id=_text(session.get("current_agent_id")),
            userdata=userdata,
            userdata_encoding=_text(session.get("userdata_encoding")) if has_userdata else None,
            tools=_json(session.get("tools_json")),
            history=history,
            agents=agents,
            interrupted=interrupted,
            created_at=float(session.get("created_at") or 0.0),
        )

    def _decode_userdata(self, session: Row) -> tuple[Any, bool]:
        raw = session.get("userdata")
        if raw is None:
            return None, False
        if session.get("userdata_encoding") == "pickle":
            return raw, True

        data = json.loads(str(raw))
        cls_name = (_json(session.get("extra")) or {}).get("userdata_cls")
        if not cls_name:
            return data, True
        try:
            cls = import_qualified(cls_name)
            return TypeAdapter(cls).validate_python(data), True
        except Exception:
            logger.warning(
                "the stored userdata's class did not rebuild, restoring it as plain JSON",
                extra={"session_id": self._session_id, "cls": cls_name},
                exc_info=True,
            )
            return data, True

    def _encode_userdata(self, userdata: Any) -> tuple[Value, str | None, str | None]:
        if userdata is None:
            return None, None, None
        cls = type(userdata)
        try:
            adapter = TypeAdapter(cls)
            data = adapter.dump_python(userdata, mode="json")
            # JSON that does not read back as the same value, such as a dict keyed by tuples,
            # would restore something else, so it is stored pickled instead
            if adapter.validate_python(data) != userdata:
                raise ValueError("userdata does not round-trip through JSON")
            return json.dumps(data), "json", qualified_name(cls)
        except Exception:
            if not self._pickle_warned:
                self._pickle_warned = True
                logger.warning(
                    "userdata is not JSON-serializable, so it is stored pickled; make it a "
                    "dataclass or pydantic model of JSON fields to keep it readable",
                    extra={"session_id": self._session_id, "cls": qualified_name(cls)},
                )
            return pickle.dumps(userdata), "pickle", qualified_name(cls)

    def append(self, item: ChatItem, *, owner: str = SESSION_OWNER) -> None:
        """Write one chat item, again on its id. Queued; the caller never waits on it."""
        self._enqueue(
            "INSERT OR REPLACE INTO chat_items (session_id, owner, item_id, item_json, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (self._session_id, owner, item.id, item_json(item), item.created_at),
        )

    def remove(self, item_id: str, *, owner: str) -> None:
        """Drop one chat item from an agent's context. Queued like ``append``."""
        self._enqueue(
            "DELETE FROM chat_items WHERE session_id = ? AND owner = ? AND item_id = ?",
            (self._session_id, owner, item_id),
        )

    async def task_started(
        self, call_id: str, *, name: str, arguments: str, origin: TaskOrigin = "llm"
    ) -> None:
        """Record a call before its body runs, so a crash mid-call leaves it ``running``."""
        self._enqueue(
            "INSERT OR REPLACE INTO tasks (session_id, call_id, name, arguments, status, "
            "started_at, idempotency_key, origin) VALUES (?, ?, ?, ?, 'running', ?, ?, ?)",
            (self._session_id, call_id, name, arguments, time.time(), call_id, origin),
        )
        await self.flush()

    def task_ended(
        self, call_id: str, *, status: TaskStatus, output: str | None, is_error: bool
    ) -> None:
        self._enqueue(
            "UPDATE tasks SET status = ?, ended_at = ?, output = ?, is_error = ? "
            "WHERE session_id = ? AND call_id = ?",
            (status, time.time(), output, int(is_error), self._session_id, call_id),
        )

    def delegation_started(
        self, call_id: str, *, endpoint: str | None, child_session_id: str, task_id: str
    ) -> None:
        """Link a delegate call to the expert session and task that answer it."""
        self._enqueue(
            "INSERT OR REPLACE INTO delegations (session_id, call_id, child_session_id, task_id, "
            "endpoint, status, created_at) VALUES (?, ?, ?, ?, ?, 'working', ?)",
            (self._session_id, call_id, child_session_id, task_id, endpoint, time.time()),
        )

    def delegation_ended(self, call_id: str, *, status: str) -> None:
        self._enqueue(
            "UPDATE delegations SET status = ?, ended_at = ? WHERE session_id = ? AND call_id = ?",
            (status, time.time(), self._session_id, call_id),
        )

    async def child_session(self, endpoint: str | None) -> str | None:
        """The expert session this one last delegated to on ``endpoint``, to reuse on resume."""
        await self.flush()
        child: str | None = None
        async for row in self._conversation.executor.query(
            "SELECT child_session_id FROM delegations WHERE session_id = ? AND endpoint IS ? "
            "AND child_session_id IS NOT NULL ORDER BY created_at DESC LIMIT 1",
            self._session_id,
            endpoint,
        ):
            child = _text(row["child_session_id"])
        return child

    async def checkpoint(
        self,
        *,
        current_agent_id: str | None,
        userdata: Any,
        agents: list[AgentRecord],
        tools: list[str] | None = None,
    ) -> None:
        """Rewrite the mutable part in one batch, and renew the lease with it.

        Raises ``LeaseLostError``, having written nothing, when another worker holds the
        session now.
        """
        await self.flush()
        encoded, encoding, userdata_cls = self._encode_userdata(userdata)
        now = time.time()
        statements: list[Statement] = [
            (
                "INSERT OR REPLACE INTO _lease_check (id, held) VALUES (1, (SELECT COUNT(*) "
                "FROM sessions WHERE session_id = ? AND lease_owner = ?))",
                (self._session_id, self._lease_owner),
            ),
            (
                "UPDATE sessions SET current_agent_id = ?, userdata = ?, userdata_encoding = ?, "
                "tools_json = ?, extra = ?, updated_at = ?, lease_expires_at = ? "
                "WHERE session_id = ? AND lease_owner = ?",
                (
                    current_agent_id,
                    encoded,
                    encoding,
                    json.dumps(tools) if tools is not None else None,
                    json.dumps({"userdata_cls": userdata_cls}) if userdata_cls else None,
                    now,
                    now + self._lease_ttl,
                    self._session_id,
                    self._lease_owner,
                ),
            ),
        ]
        for agent in agents:
            statements.append(
                (
                    "INSERT INTO agents (session_id, agent_id, cls, parent_agent_id, state_json, "
                    "tools_json) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (session_id, agent_id) "
                    "DO UPDATE SET cls = excluded.cls, parent_agent_id = excluded.parent_agent_id, "
                    "state_json = excluded.state_json, tools_json = excluded.tools_json",
                    (
                        self._session_id,
                        agent.agent_id,
                        agent.cls,
                        agent.parent_agent_id,
                        json.dumps(agent.state) if agent.state is not None else None,
                        json.dumps(agent.tools) if agent.tools is not None else None,
                    ),
                )
            )
        try:
            await self._conversation.executor.batch(*statements)
        except StoreError as e:
            if "lease_held" in e.message:
                raise LeaseLostError(
                    f"session {self._session_id} is held by another worker now"
                ) from None
            raise

    async def release(self) -> None:
        """Flush what is queued and let the session go, so the next worker need not wait.

        The last session of a conversation to be released closes its connection.
        """
        if self._released:
            return
        self._released = True
        try:
            await self.flush()
            await self._conversation.executor.exec(
                "UPDATE sessions SET lease_owner = NULL, lease_expires_at = NULL, closed_at = ? "
                "WHERE session_id = ? AND lease_owner = ?",
                time.time(),
                self._session_id,
                self._lease_owner,
            )
        finally:
            await self._conversation._session_released()

    async def flush(self) -> None:
        """Wait for every queued write to land."""
        while self._writer is not None and not self._writer.done():
            await asyncio.shield(self._writer)

    def _enqueue(self, sql: str, params: tuple[Value, ...]) -> None:
        self._pending.append((sql, params))
        if self._writer is None or self._writer.done():
            self._writer = asyncio.create_task(self._write(), name="session_state_write")

    async def _write(self) -> None:
        # everything queued since the last write goes as one batch, in order
        while self._pending:
            statements, self._pending = self._pending, []
            try:
                await self._conversation.executor.batch(*statements)
            except Exception:
                logger.warning(
                    "could not write to the session store",
                    extra={"session_id": self._session_id, "statements": len(statements)},
                    exc_info=True,
                )


def _text(value: Value | None) -> str | None:
    return None if value is None else str(value)


def _json(value: Value | None) -> Any:
    return None if value is None else json.loads(str(value))


__all__ = [
    "INTERRUPTED_OUTPUT",
    "LEASE_TTL",
    "SESSION_OWNER",
    "AgentRecord",
    "LeaseHeldError",
    "LeaseLostError",
    "SessionKind",
    "SessionState",
    "StoredSession",
    "TaskOrigin",
    "TaskRecord",
    "TaskStatus",
    "import_qualified",
    "item_json",
    "qualified_name",
]
