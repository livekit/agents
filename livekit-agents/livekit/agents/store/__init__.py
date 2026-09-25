"""Persisting sessions, so a conversation can be loaded again after its session closed.

A conversation is stored as one SQLite database, so a conversation id is a database id: every
session of it is rows in that database, served by agent-db or kept in a local file by
``LocalStore`` for tests and offline runs. The front session, the one every channel of the
conversation resumes, takes the conversation id as its own; an expert's session takes its A2A
context id, under the session that delegated to it.
"""

from .agentdb import AgentDB
from .base import SessionStore, StoreError
from .local import LocalStore
from .schema import SchemaVersionError
from .session import StoredSession

__all__ = [
    "AgentDB",
    "LocalStore",
    "SchemaVersionError",
    "SessionStore",
    "StoreError",
    "StoredSession",
]
