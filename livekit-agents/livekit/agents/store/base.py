"""What both backends share: the SQL they run, and the store over them.

agent-db serves SQLite, so the store's statements run unchanged on a local ``sqlite3`` file,
and only the wire tests need a server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .session import StoredSession, _Database

Value = int | float | str | bytes | None
"""One of SQLite's five storage classes."""

Row = dict[str, Value]
"""One result row, keyed by column name."""

Statement = tuple[str, Sequence[Value]]
"""One SQL statement and its positional parameters."""


@dataclass
class ExecResult:
    rows_affected: int = 0
    last_insert_id: int = 0
    tip: int = 0
    """The durable commit sequence the write landed at, where the backend reports one."""


class StoreError(Exception):
    """A statement the backend refused, with its machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Executor(Protocol):
    """Runs SQL against one database. Concurrent calls are not ordered against each other."""

    async def exec(self, sql: str, *params: Value) -> ExecResult: ...

    async def batch(self, *statements: Statement) -> ExecResult:
        """Apply every statement in order, atomically."""
        ...

    def query(self, sql: str, *params: Value) -> AsyncIterator[Row]: ...

    async def aclose(self) -> None: ...


class SessionStore(ABC):
    """A store holds the sessions of conversations. A backend pickles as its configuration only,
    so each job process opens its own connections."""

    def __init__(self) -> None:
        self._databases: dict[str, _Database] = {}

    @abstractmethod
    async def _connect(self, database_id: str) -> Executor:
        """Open one database, which exists."""

    @abstractmethod
    async def create_database(self) -> str:
        """Create an empty database and return its id."""

    def session(
        self,
        conversation_id: str,
        session_id: str | None = None,
        *,
        parent: str | None = None,
        endpoint: str | None = None,
    ) -> StoredSession:
        """A conversation's session, for ``start(persist=)``: with no ``session_id`` the front
        session, whose id is the conversation id. ``parent`` is the caller's session."""
        from .session import StoredSession, _Database

        if (database := self._databases.get(conversation_id)) is None:
            database = self._databases[conversation_id] = _Database(
                conversation_id, connect=lambda: self._connect(conversation_id)
            )
        return StoredSession(
            database,
            session_id if session_id is not None else conversation_id,
            parent=parent,
            endpoint=endpoint,
        )

    async def aclose(self) -> None:
        for database in self._databases.values():
            await database.aclose()
        self._databases.clear()


__all__ = [
    "ExecResult",
    "Executor",
    "Row",
    "SessionStore",
    "Statement",
    "StoreError",
    "Value",
]
