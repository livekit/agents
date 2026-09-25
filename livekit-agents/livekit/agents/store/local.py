"""Sessions in a directory of SQLite files, one per database, for tests and offline runs."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..utils import shortuuid
from .base import ExecResult, Executor, Row, Statement, Store, StoreError, Value
from .schema import migrate


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


class LocalStore(Store):
    """Databases as SQLite files under ``directory``. No server needed."""

    def __init__(self, directory: str | os.PathLike[str]):
        super().__init__()
        self._directory = Path(directory)

    def __reduce__(self) -> tuple[type[LocalStore], tuple[Path]]:
        return (LocalStore, (self._directory,))

    async def _connect(self, database_id: str) -> Executor:
        path = self._directory / f"{database_id}.sqlite"
        if not path.exists():
            raise StoreError("not_found", f"no database at {path}")
        return SQLiteExecutor(str(path))

    async def create_database(self) -> str:
        """Create an empty database and return its id."""
        self._directory.mkdir(parents=True, exist_ok=True)
        database_id = shortuuid("DB_")
        executor = SQLiteExecutor(str(self._directory / f"{database_id}.sqlite"))
        try:
            await migrate(executor)
        finally:
            await executor.aclose()
        return database_id


__all__ = ["LocalStore", "SQLiteExecutor"]
