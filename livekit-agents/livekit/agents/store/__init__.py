"""Persisting sessions, so a conversation survives the worker that ran it.

Every session of one conversation is rows in one SQLite database, served by agent-db or kept
in a local file by ``LocalStore`` for tests and offline runs.
"""

from .agentdb import AgentDB
from .executor import StoreError
from .local import LocalStore
from .schema import SchemaVersionError
from .session import LeaseHeldError, LeaseLostError, PersistedSession

__all__ = [
    "AgentDB",
    "LeaseHeldError",
    "LeaseLostError",
    "LocalStore",
    "PersistedSession",
    "SchemaVersionError",
    "StoreError",
]
