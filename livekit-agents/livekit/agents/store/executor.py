"""SQL over two backends: agent-db serves SQLite, so the store's statements run unchanged on
a local ``sqlite3`` file, and only the wire tests need a server.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

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


class SQLiteExecutor:
    """An ``Executor`` on a local SQLite file, or on one ``:memory:`` database per instance."""

    def __init__(self, path: str = ":memory:") -> None:
        self._path = path
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lk_store_sqlite")
        self._db: sqlite3.Connection | None = None

    async def _run(self, fn: Any, *args: Any) -> Any:
        def call() -> Any:
            if self._db is None:
                # autocommit, so a batch is the only place a transaction is opened
                self._db = sqlite3.connect(self._path, isolation_level=None)
            try:
                return fn(self._db, *args)
            except sqlite3.Error as e:
                raise StoreError(getattr(e, "sqlite_errorname", "exec"), str(e)) from e

        return await asyncio.get_running_loop().run_in_executor(self._thread, call)

    async def exec(self, sql: str, *params: Value) -> ExecResult:
        def run(db: sqlite3.Connection) -> ExecResult:
            cursor = db.execute(sql, params)
            return ExecResult(rows_affected=cursor.rowcount, last_insert_id=cursor.lastrowid or 0)

        result: ExecResult = await self._run(run)
        return result

    async def batch(self, *statements: Statement) -> ExecResult:
        def run(db: sqlite3.Connection) -> ExecResult:
            db.execute("BEGIN IMMEDIATE")
            try:
                for sql, params in statements:
                    db.execute(sql, tuple(params))
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
            return ExecResult()

        result: ExecResult = await self._run(run)
        return result

    async def query(self, sql: str, *params: Value) -> AsyncIterator[Row]:
        def run(db: sqlite3.Connection) -> list[Row]:
            cursor = db.execute(sql, params)
            names = [column[0] for column in cursor.description or ()]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

        rows: list[Row] = await self._run(run)
        for row in rows:
            yield row

    async def aclose(self) -> None:
        def close(db: sqlite3.Connection) -> None:
            db.close()

        if self._db is not None:
            await self._run(close)
            self._db = None
        self._thread.shutdown(wait=False)


__all__ = [
    "ExecResult",
    "Executor",
    "Row",
    "SQLiteExecutor",
    "Statement",
    "StoreError",
    "Value",
]
