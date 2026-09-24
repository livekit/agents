"""Persisting sessions, so a conversation can be loaded again after its session closed.

A conversation is stored as one SQLite database, so a conversation id is a database id: every
session of it is rows in that database, served by agent-db or kept in a local file by
``LocalStore`` for tests and offline runs.
"""

from .agentdb import AgentDB
from .executor import StoreError
from .local import LocalStore
from .schema import SchemaVersionError
from .session import PersistedSession

__all__ = [
    "AgentDB",
    "LocalStore",
    "PersistedSession",
    "SchemaVersionError",
    "StoreError",
]
