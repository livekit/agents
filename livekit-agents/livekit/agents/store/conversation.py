"""Where conversations live: agent-db in production, a directory of SQLite files offline.

A conversation is one database. Every session of it, the voice agent and each expert it
delegated to, is rows in that database keyed by session id, so the whole of one user's
conversation is one socket for the agent and one query for a dashboard.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol

from ..utils import shortuuid
from .agentdb import TOKEN_TTL, AgentDBExecutor, AgentDBService, access_token
from .executor import Executor, SQLiteExecutor, StoreError
from .schema import migrate
from .session_state import SessionKind, SessionState

LEASE_TTL = 30.0
"""How long a session stays claimed without a checkpoint renewing it. A worker restarted
after a crash waits at most this long before it can take the session back."""


class Conversation:
    """One conversation database, migrated and open. Sessions are made from it."""

    def __init__(self, database_id: str, executor: Executor, *, lease_ttl: float) -> None:
        self._database_id = database_id
        self._executor = executor
        self._lease_ttl = lease_ttl

    @property
    def database_id(self) -> str:
        return self._database_id

    @property
    def executor(self) -> Executor:
        return self._executor

    def session(
        self,
        session_id: str,
        *,
        kind: SessionKind = "text",
        parent: str | None = None,
        endpoint: str | None = None,
    ) -> SessionState:
        """A handle on one session of this conversation; ``AgentSession.start`` loads it.

        ``parent`` is the session that delegated to this one, when there is one.
        """
        return SessionState(
            self,
            session_id,
            kind=kind,
            parent=parent,
            endpoint=endpoint,
            lease_ttl=self._lease_ttl,
        )

    async def aclose(self) -> None:
        await self._executor.aclose()


class Store(Protocol):
    """Opens conversations by id, and makes new ones."""

    async def conversation(self, database_id: str) -> Conversation: ...

    async def create_conversation(self) -> Conversation: ...

    async def aclose(self) -> None: ...


class SQLite:
    """Conversations as SQLite files under a directory, one file each. No server needed."""

    def __init__(self, directory: str | os.PathLike[str], *, lease_ttl: float = LEASE_TTL):
        self._directory = Path(directory)
        self._lease_ttl = lease_ttl
        self._open: dict[str, Conversation] = {}

    async def conversation(self, database_id: str) -> Conversation:
        if database_id not in self._open:
            path = self._directory / f"{database_id}.sqlite"
            if not path.exists():
                raise StoreError("not_found", f"no conversation database at {path}")
            await self._attach(database_id, path)
        return self._open[database_id]

    async def create_conversation(self) -> Conversation:
        self._directory.mkdir(parents=True, exist_ok=True)
        database_id = shortuuid("DB_")
        await self._attach(database_id, self._directory / f"{database_id}.sqlite")
        return self._open[database_id]

    async def _attach(self, database_id: str, path: Path) -> None:
        executor = SQLiteExecutor(str(path))
        await migrate(executor)
        self._open[database_id] = Conversation(database_id, executor, lease_ttl=self._lease_ttl)

    async def aclose(self) -> None:
        for conversation in self._open.values():
            await conversation.aclose()
        self._open.clear()


class AgentDB:
    """Conversations in agent-db: its management API mints them, its data plane serves them.

    Keeps one socket per conversation it has opened, until ``aclose``.
    """

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
        """Read ``LIVEKIT_AGENTDB_URL`` and ``LIVEKIT_AGENTDB_WS_URL``, and the project's key.

        ``LIVEKIT_AGENTDB_API_KEY``/``_SECRET`` win over ``LIVEKIT_API_KEY``/``_SECRET``, for a
        database served apart from the LiveKit project, as a local agent-db is.
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
            executor = AgentDBExecutor(
                ws_url=self._ws_url, database_id=database_id, token=self._token
            )
            await executor.connect()
            try:
                await migrate(executor)
            except BaseException:
                await executor.aclose()
                raise
            self._open[database_id] = Conversation(database_id, executor, lease_ttl=self._lease_ttl)
        return self._open[database_id]

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


__all__ = ["LEASE_TTL", "AgentDB", "Conversation", "SQLite", "Store"]
