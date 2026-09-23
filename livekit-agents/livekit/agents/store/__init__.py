"""Persisting sessions: a conversation is one SQLite database, in agent-db or on disk.

Open a conversation, bind a session of it, and hand that to ``AgentSession.start``::

    STORE = store.AgentDB.from_env()

    conversation = await STORE.conversation(database_id)
    await session.start(agent=FareDesk(), state=conversation.session("fare-desk", kind="a2a"))

``store.SQLite(directory)`` is the same API on local files, for tests and offline runs.
"""

from .agentdb import AgentDBExecutor, AgentDBService, access_token
from .conversation import LEASE_TTL, AgentDB, Conversation, SQLite, Store
from .executor import ExecResult, Executor, Row, SQLiteExecutor, Statement, StoreError, Value
from .schema import SCHEMA_VERSION, SchemaVersionError
from .session_state import (
    INTERRUPTED_OUTPUT,
    AgentRecord,
    LeaseHeldError,
    LeaseLostError,
    SessionKind,
    SessionState,
    StoredSession,
    TaskRecord,
)

__all__ = [
    "INTERRUPTED_OUTPUT",
    "LEASE_TTL",
    "SCHEMA_VERSION",
    "AgentDB",
    "AgentDBExecutor",
    "AgentDBService",
    "AgentRecord",
    "Conversation",
    "ExecResult",
    "Executor",
    "LeaseHeldError",
    "LeaseLostError",
    "Row",
    "SQLite",
    "SQLiteExecutor",
    "SchemaVersionError",
    "SessionKind",
    "SessionState",
    "Statement",
    "Store",
    "StoreError",
    "StoredSession",
    "TaskRecord",
    "Value",
    "access_token",
]
