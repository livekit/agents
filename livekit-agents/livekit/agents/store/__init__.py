"""Persisting sessions, so a conversation survives the worker that ran it.

A conversation is one SQLite database, in agent-db or a local file; ``SQLite`` runs the same
API offline and in tests.
"""

from .conversation import AgentDB, Conversation, SQLite
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
    "StoreError",
]
