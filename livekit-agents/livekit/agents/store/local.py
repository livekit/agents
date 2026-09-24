"""Sessions in a directory of SQLite files, one per database, for tests and offline runs."""

from __future__ import annotations

import os
from pathlib import Path

from ..utils import shortuuid
from .executor import Executor, SQLiteExecutor, StoreError
from .schema import migrate
from .session import LEASE_TTL, _Store


class LocalStore(_Store):
    """Databases as SQLite files under ``directory``. No server needed."""

    def __init__(self, directory: str | os.PathLike[str], *, lease_ttl: float = LEASE_TTL):
        super().__init__(lease_ttl=lease_ttl)
        self._directory = Path(directory)

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


__all__ = ["LocalStore"]
