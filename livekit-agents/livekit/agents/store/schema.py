"""The conversation database's schema, versioned in ``_meta`` and migrated forward on open.

A database written by one release is opened by a later one weeks on, so every change to the
tables is a new migration appended below, never an edit to an old one. A framework older than
the database refuses to open it rather than write rows it does not understand.
"""

from __future__ import annotations

from .executor import Executor, Statement, StoreError

SCHEMA_VERSION = 1

MIGRATIONS: dict[int, list[str]] = {
    1: [
        """CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            parent_session_id TEXT,
            kind TEXT NOT NULL,
            endpoint TEXT,
            current_agent_id TEXT,
            userdata BLOB,
            userdata_encoding TEXT,
            tools_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            closed_at REAL,
            lease_owner TEXT,
            lease_expires_at REAL,
            extra TEXT
        )""",
        """CREATE TABLE chat_items (
            session_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            item_id TEXT NOT NULL,
            item_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (session_id, owner, item_id)
        )""",
        """CREATE TABLE agents (
            session_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            cls TEXT NOT NULL,
            parent_agent_id TEXT,
            state_json TEXT,
            tools_json TEXT,
            durable_state BLOB,
            PRIMARY KEY (session_id, agent_id)
        )""",
        """CREATE TABLE tasks (
            session_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            name TEXT NOT NULL,
            arguments TEXT,
            status TEXT NOT NULL,
            started_at REAL NOT NULL,
            ended_at REAL,
            output TEXT,
            is_error INTEGER,
            idempotency_key TEXT,
            origin TEXT NOT NULL,
            PRIMARY KEY (session_id, call_id)
        )""",
        """CREATE TABLE delegations (
            session_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            child_session_id TEXT,
            task_id TEXT,
            endpoint TEXT,
            status TEXT NOT NULL,
            created_at REAL NOT NULL,
            ended_at REAL,
            PRIMARY KEY (session_id, call_id)
        )""",
        # a checkpoint from a worker that lost the lease writes one row here that breaks the
        # check, which aborts its whole batch rather than half of it
        """CREATE TABLE _lease_check (
            id INTEGER PRIMARY KEY,
            held INTEGER NOT NULL CONSTRAINT lease_held CHECK (held = 1)
        )""",
    ],
}


class SchemaVersionError(Exception):
    """The database was written by a newer framework than this one."""


async def migrate(executor: Executor) -> int:
    """Bring a database up to ``SCHEMA_VERSION`` and return the version it was found at."""
    await executor.exec("CREATE TABLE IF NOT EXISTS _meta (key TEXT PRIMARY KEY, value TEXT)")
    found = 0
    async for row in executor.query("SELECT value FROM _meta WHERE key = 'schema_version'"):
        found = int(str(row["value"]))
    if found > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"the conversation database is at schema version {found}, and this framework "
            f"only knows up to {SCHEMA_VERSION}; upgrade livekit-agents to open it"
        )

    for version in range(found + 1, SCHEMA_VERSION + 1):
        statements: list[Statement] = [(sql, ()) for sql in MIGRATIONS[version]]
        statements.append(
            (
                "INSERT OR REPLACE INTO _meta (key, value) VALUES ('schema_version', ?)",
                (str(version),),
            )
        )
        try:
            await executor.batch(*statements)
        except StoreError:
            # two openers raced and the other one migrated first: its batch is whole or absent
            async for row in executor.query("SELECT value FROM _meta WHERE key = 'schema_version'"):
                if int(str(row["value"])) >= version:
                    break
            else:
                raise
    return found


__all__ = ["MIGRATIONS", "SCHEMA_VERSION", "SchemaVersionError", "migrate"]
