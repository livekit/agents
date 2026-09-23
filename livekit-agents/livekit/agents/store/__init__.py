"""Persisting sessions: a conversation is one SQLite database, in agent-db or on disk.

Open a conversation, bind a session of it, and hand that to ``AgentSession.start``::

    STORE = store.AgentDB.from_env()

    conversation = await STORE.conversation(database_id)
    await session.start(agent=FareDesk(), state=conversation.session("fare-desk", kind="a2a"))

``store.SQLite(directory)`` is the same API on local files, for tests and offline runs. The
executors and the agent-db clients underneath are importable from their modules.
"""

from .conversation import AgentDB, Conversation, SQLite, Store
from .executor import StoreError
from .schema import SchemaVersionError
from .session_state import LeaseHeldError, LeaseLostError, SessionKind, SessionState

__all__ = [
    "AgentDB",
    "Conversation",
    "LeaseHeldError",
    "LeaseLostError",
    "SQLite",
    "SchemaVersionError",
    "SessionKind",
    "SessionState",
    "Store",
    "StoreError",
]
