"""One session's rows in a conversation database: what it loads, appends and checkpoints.

Chat items are appended through a queue the conversation never waits on; the small mutable
part is rewritten at checkpoints, each fenced by the session's lease.
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
    from ..delegation.delegate import Delegate
    from .conversation import Conversation

SessionKind = Literal["voice", "text", "a2a"]
TaskStatus = Literal["running", "done", "error", "cancelled", "interrupted"]

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
    tools: list[str] | None = None
    chat_items: list[ChatItem] = field(default_factory=list)
    """Filled on load; a checkpoint writes items through ``append`` instead."""


@dataclass
class TaskRecord:
    """A call the history does not answer: left running, or ended with its output unwritten."""

    call_id: str
    name: str
    arguments: str | None
    output: str | None = None
    is_error: bool = False


@dataclass
class StoredSession:
    """What a session had written when it was last checkpointed, read back."""

    current_agent_id: str | None
    userdata: Any
    """Decoded into its class when that still imports, else plain JSON; pickled stays bytes,
    since it may name agents the session rebuilds first."""
    userdata_encoding: str | None
    history: list[ChatItem]
    agents: dict[str, AgentRecord]
    interrupted: list[TaskRecord]
    """Calls still ``running`` under a previous owner, which died before they ended."""
    ended: list[TaskRecord]
    """Calls that ended, whose output the previous owner died before writing to the history."""


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
    """A handle on one session's rows, from ``Conversation.session()``, for ``start(state=)``."""

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
        self._resumed: dict[Delegate, asyncio.Task[None]] = {}
        # per owner, each item's fingerprint as last queued, so only a changed item is rewritten
        self._written: dict[str, dict[str, int]] = {}

    @property
    def conversation(self) -> Conversation:
        return self._conversation

    @property
    def session_id(self) -> str:
        return self._session_id

    async def load(self) -> StoredSession | None:
        """Claim the session, waiting out a previous owner's lease, and read it back.

        ``None`` means the session is new, and has been created.
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
            )
        async for row in executor.query(
            "SELECT owner, item_json FROM chat_items WHERE session_id = ? "
            "ORDER BY created_at, rowid",
            self._session_id,
        ):
            item = _ITEM_ADAPTER.validate_json(str(row["item_json"]))
            owner = str(row["owner"])
            self._written.setdefault(owner, {})[item.id] = hash(item_json(item))
            if owner == SESSION_OWNER:
                history.append(item)
            else:
                agents.setdefault(owner, AgentRecord(agent_id=owner, cls="")).chat_items.append(
                    item
                )

        interrupted: list[TaskRecord] = []
        async for row in executor.query(
            "SELECT call_id, name, arguments FROM tasks WHERE session_id = ? "
            "AND status = 'running' ORDER BY started_at",
            self._session_id,
        ):
            interrupted.append(
                TaskRecord(
                    call_id=str(row["call_id"]),
                    name=str(row["name"]),
                    arguments=_text(row["arguments"]),
                )
            )

        answered = {item.call_id for item in history if item.type == "function_call_output"}
        ended: list[TaskRecord] = []
        async for row in executor.query(
            "SELECT call_id, name, arguments, output, is_error FROM tasks WHERE session_id = ? "
            "AND status != 'running' ORDER BY started_at",
            self._session_id,
        ):
            if str(row["call_id"]) not in answered:
                ended.append(
                    TaskRecord(
                        call_id=str(row["call_id"]),
                        name=str(row["name"]),
                        arguments=_text(row["arguments"]),
                        output=_text(row["output"]),
                        is_error=bool(row["is_error"]),
                    )
                )

        userdata: Any = session.get("userdata")
        encoding = _text(session.get("userdata_encoding"))
        if encoding == "json":
            userdata = json.loads(str(userdata))
            if cls_name := (_json(session.get("extra")) or {}).get("userdata_cls"):
                try:
                    userdata = TypeAdapter(import_qualified(cls_name)).validate_python(userdata)
                except Exception:
                    logger.warning(
                        "the stored userdata's class did not rebuild, restoring it as plain JSON",
                        extra={"session_id": self._session_id, "cls": cls_name},
                        exc_info=True,
                    )
        return StoredSession(
            current_agent_id=_text(session.get("current_agent_id")),
            userdata=userdata,
            userdata_encoding=encoding,
            history=history,
            agents=agents,
            interrupted=interrupted,
            ended=ended,
        )

    def append(self, item: ChatItem, *, owner: str = SESSION_OWNER) -> None:
        """Write one chat item, again on its id when it changed. Queued; nothing waits on it."""
        data = item_json(item)
        written = self._written.setdefault(owner, {})
        if written.get(item.id) == hash(data):
            return
        written[item.id] = hash(data)
        self._enqueue(
            "INSERT OR REPLACE INTO chat_items (session_id, owner, item_id, item_json, "
            "created_at) VALUES (?, ?, ?, ?, ?)",
            (self._session_id, owner, item.id, data, item.created_at),
        )

    def sync(self, items: list[ChatItem], *, owner: str, prune: bool) -> None:
        """Append whatever changed in ``items``; ``prune`` also drops rows no longer in them."""
        for item in items:
            self.append(item, owner=owner)
        if prune:
            written = self._written.setdefault(owner, {})
            for item_id in written.keys() - {item.id for item in items}:
                del written[item_id]
                self._enqueue(
                    "DELETE FROM chat_items WHERE session_id = ? AND owner = ? AND item_id = ?",
                    (self._session_id, owner, item_id),
                )

    async def task_started(self, call_id: str, *, name: str, arguments: str) -> None:
        """Record a call before its body runs, so a crash mid-call leaves it ``running``."""
        self._enqueue(
            "INSERT OR REPLACE INTO tasks (session_id, call_id, name, arguments, status, "
            "started_at, idempotency_key, origin) VALUES (?, ?, ?, ?, 'running', ?, ?, 'llm')",
            (self._session_id, call_id, name, arguments, time.time(), call_id),
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

    async def resume_delegate(self, delegate: Delegate) -> None:
        """Point a delegate back at the expert session this one last had on its endpoint, once."""
        if (lookup := self._resumed.get(delegate)) is None:

            async def resume() -> None:
                if (endpoint := delegate.endpoint) is None:
                    return
                if (child := await self.child_session(endpoint)) is not None and delegate.resume(
                    child
                ):
                    logger.debug(
                        "resuming the delegate's earlier conversation",
                        extra={"endpoint": endpoint, "context_id": child},
                    )

            lookup = self._resumed[delegate] = asyncio.create_task(resume())
        await asyncio.shield(lookup)

    async def checkpoint(
        self,
        *,
        current_agent_id: str | None,
        userdata: Any,
        agents: list[AgentRecord],
        tools: list[str] | None = None,
    ) -> None:
        """Rewrite the mutable part in one batch and renew the lease, or write nothing and
        raise ``LeaseLostError`` when another worker holds the session.
        """
        await self.flush()
        encoded: Value = None
        encoding = userdata_cls = None
        if userdata is not None:
            cls = type(userdata)
            userdata_cls = qualified_name(cls)
            try:
                adapter = TypeAdapter(cls)
                data = adapter.dump_python(userdata, mode="json")
                # JSON that reads back as something else, such as a tuple-keyed dict, is pickled
                if adapter.validate_python(data) != userdata:
                    raise ValueError("userdata does not round-trip through JSON")
                encoded, encoding = json.dumps(data), "json"
            except Exception:
                if not self._pickle_warned:
                    self._pickle_warned = True
                    logger.warning(
                        "userdata is not JSON-serializable, so it is stored pickled; make it a "
                        "dataclass or pydantic model of JSON fields to keep it readable",
                        extra={"session_id": self._session_id, "cls": userdata_cls},
                    )
                encoded, encoding = pickle.dumps(userdata), "pickle"
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
        """Flush what is queued and let the session go, closing the conversation after its last."""
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
            conversation = self._conversation
            conversation._sessions -= 1
            if conversation._sessions <= 0:
                conversation._sessions = 0
                await conversation.aclose()

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
    "SessionKind",
    "SessionState",
    "StoredSession",
    "TaskRecord",
    "TaskStatus",
    "import_qualified",
    "item_json",
    "qualified_name",
]
