"""Where conversations live: agent-db in production, a directory of SQLite files offline.

Every session of one conversation, the voice agent and each expert it delegated to, is rows in
one database, so the conversation is one socket for the agent and one query for a dashboard.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..utils import shortuuid
from .agentdb import TOKEN_TTL, AgentDBExecutor, AgentDBService, access_token
from .executor import Executor, SQLiteExecutor, StoreError
from .schema import migrate
from .session_state import LEASE_TTL, SessionKind, SessionState


class Conversation:
    """One conversation database, whose connection is open only while a session holds it."""

    def __init__(
        self,
        database_id: str,
        *,
        connect: Callable[[], Awaitable[Executor]],
        lease_ttl: float,
    ) -> None:
        self._database_id = database_id
        self._connect = connect
        self._lease_ttl = lease_ttl
        self._executor: Executor | None = None
        self._open_lock = asyncio.Lock()
        self._sessions = 0

    @property
    def database_id(self) -> str:
        return self._database_id

    @property
    def executor(self) -> Executor:
        """The open connection. Raises while closed; ``open()`` reopens it."""
        if self._executor is None:
            raise StoreError("closed", f"conversation {self._database_id} is not open")
        return self._executor

    async def open(self) -> Executor:
        """Connect and migrate, unless already open."""
        async with self._open_lock:
            if self._executor is None:
                executor = await self._connect()
                try:
                    await migrate(executor)
                except BaseException:
                    await executor.aclose()
                    raise
                self._executor = executor
            return self._executor

    def session(
        self,
        session_id: str,
        *,
        kind: SessionKind = "text",
        parent: str | None = None,
        endpoint: str | None = None,
    ) -> SessionState:
        """A handle on one session, for ``AgentSession.start``; ``parent`` is its delegator."""
        self._sessions += 1
        return SessionState(
            self,
            session_id,
            kind=kind,
            parent=parent,
            endpoint=endpoint,
            lease_ttl=self._lease_ttl,
        )

    async def aclose(self) -> None:
        async with self._open_lock:
            if self._executor is not None:
                executor, self._executor = self._executor, None
                await executor.aclose()


class SQLite:
    """Conversations as SQLite files under a directory, one file each. No server needed."""

    def __init__(self, directory: str | os.PathLike[str], *, lease_ttl: float = LEASE_TTL):
        self._directory = Path(directory)
        self._lease_ttl = lease_ttl
        self._open: dict[str, Conversation] = {}

    async def conversation(self, database_id: str) -> Conversation:
        path = self._directory / f"{database_id}.sqlite"
        if database_id not in self._open:
            if not path.exists():
                raise StoreError("not_found", f"no conversation database at {path}")
            self._open[database_id] = self._conversation(database_id, path)
        conversation = self._open[database_id]
        await conversation.open()
        return conversation

    async def create_conversation(self) -> Conversation:
        self._directory.mkdir(parents=True, exist_ok=True)
        database_id = shortuuid("DB_")
        path = self._directory / f"{database_id}.sqlite"
        conversation = self._open[database_id] = self._conversation(database_id, path)
        await conversation.open()
        return conversation

    def _conversation(self, database_id: str, path: Path) -> Conversation:
        async def connect() -> Executor:
            return SQLiteExecutor(str(path))

        return Conversation(database_id, connect=connect, lease_ttl=self._lease_ttl)

    async def aclose(self) -> None:
        for conversation in self._open.values():
            await conversation.aclose()
        self._open.clear()


class AgentDB:
    """Conversations in agent-db: its management API mints them, its data plane serves them."""

    def __init__(
        self,
        *,
        url: str,
        ws_url: str,
        api_key: str,
        api_secret: str,
        lease_ttl: float = LEASE_TTL,
    ) -> None:
        self._url = url
        self._ws_url = ws_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._lease_ttl = lease_ttl
        self._service: AgentDBService | None = None
        self._access_token = ""
        self._token_expires_at = 0.0
        self._open: dict[str, Conversation] = {}

    @classmethod
    def from_env(cls, *, lease_ttl: float = LEASE_TTL) -> AgentDB:
        """Read the agent-db URLs, and its key from ``LIVEKIT_AGENTDB_API_KEY``/``_SECRET``
        or else the project's ``LIVEKIT_API_KEY``/``_SECRET``.
        """
        env = {
            "url": os.environ.get("LIVEKIT_AGENTDB_URL"),
            "ws_url": os.environ.get("LIVEKIT_AGENTDB_WS_URL"),
            "api_key": os.environ.get("LIVEKIT_AGENTDB_API_KEY")
            or os.environ.get("LIVEKIT_API_KEY"),
            "api_secret": os.environ.get("LIVEKIT_AGENTDB_API_SECRET")
            or os.environ.get("LIVEKIT_API_SECRET"),
        }
        missing = [name for name, value in env.items() if not value]
        if missing:
            raise ValueError(
                "agent-db is not configured: set LIVEKIT_AGENTDB_URL, LIVEKIT_AGENTDB_WS_URL "
                f"and the API key and secret (missing {', '.join(missing)})"
            )
        return cls(
            url=env["url"] or "",
            ws_url=env["ws_url"] or "",
            api_key=env["api_key"] or "",
            api_secret=env["api_secret"] or "",
            lease_ttl=lease_ttl,
        )

    def _token(self) -> str:
        # one token serves every socket and call until it is near expiry
        if time.time() > self._token_expires_at:
            self._access_token = access_token(
                self._api_key, self._api_secret, identity="livekit-agents", ttl=TOKEN_TTL
            )
            self._token_expires_at = time.time() + TOKEN_TTL / 2
        return self._access_token

    @property
    def service(self) -> AgentDBService:
        """The management API, for listing and deleting conversations."""
        if self._service is None:
            self._service = AgentDBService(self._url, token=self._token)
        return self._service

    async def conversation(self, database_id: str) -> Conversation:
        if database_id not in self._open:

            async def connect() -> Executor:
                executor = AgentDBExecutor(
                    ws_url=self._ws_url, database_id=database_id, token=self._token
                )
                await executor.connect()
                return executor

            self._open[database_id] = Conversation(
                database_id, connect=connect, lease_ttl=self._lease_ttl
            )
        conversation = self._open[database_id]
        await conversation.open()
        return conversation

    async def create_conversation(self, *, ttl_seconds: int = 0) -> Conversation:
        """Mint a new database. ``ttl_seconds`` unset means it never expires."""
        created = await self.service.create_database(ttl_seconds=ttl_seconds)
        return await self.conversation(created.database_id)

    async def aclose(self) -> None:
        for conversation in self._open.values():
            await conversation.aclose()
        self._open.clear()
        if self._service is not None:
            await self._service.aclose()
            self._service = None


__all__ = ["AgentDB", "Conversation", "SQLite"]
