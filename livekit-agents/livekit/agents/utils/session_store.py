"""
Session Store using SQLite Session Extension for efficient delta synchronization.

This module provides a store for AgentSession state that tracks changes incrementally
using SQLite's session extension, enabling efficient delta sync to cloud storage.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import apsw
except ImportError:
    raise ImportError("apsw is required for session_store. Install it with: pip install apsw")

from .. import utils
from ..log import logger


@dataclass
class DBOperation:
    """Represents a single database operation."""

    table: str  # Table name
    operation: Literal["INSERT", "UPDATE", "DELETE"]  # "INSERT", "UPDATE", "DELETE"
    data: dict[str, Any] | None = None  # Data for INSERT/UPDATE (column: value)
    where: dict[str, Any] | None = None  # WHERE clause for UPDATE/DELETE (column: value)


@dataclass
class StoredState:
    """
    Represents the complete state stored in the database.

    Matches the structure from AgentSession.get_state():
    {
        "userdata": dict,
        "tools": list[str],
        "history": dict,  # chat_ctx.to_dict()
        "agent": dict,    # Agent.get_state()
    }
    """

    session_id: str
    version: int
    userdata: dict[str, Any]
    tools: list[str]
    history: dict[str, Any]
    agent: dict[str, Any]


class SessionStore:
    """
    SQLite-based session store with delta tracking using session extension.

    Design Principles:
    - One SQLite DB per session (no session_id in tables)
    - Chat items use ID-based updates (not index-based)
    - Userdata stored as K-V pairs for granular deltas
    - Agent hierarchy via parent_id text references
    - Base64 encoding for encrypted/binary data
    - agent_id (TEXT) as primary key for agents

    Usage:
        # Initialize store (one DB per session)
        store = SessionStore(session_id="session_123", db_path="./session_123.db")

        # Save initial state
        store.save_session_state(
            userdata={"user_id": "123", "credits": 100},
            session_history=[
                {"id": "item_abc", "role": "user", "content": "Hello", "created_at": 1706123456.0}
            ],
            tools=["tool1", "tool2"],
            current_agent_id="main_agent",
            agent_states={
                "main_agent": {  # agent_id as key
                    "instructions": "Be helpful",
                    "chat_items": [...]
                }
            }
        )

        # Track changes for delta sync
        store.begin_tracking()
        # ... make updates ...
        changeset = store.end_tracking()
        upload_to_cloud(changeset)

        # On worker side: reconstruct from base + changesets
        worker_store = SessionStore.from_base_and_changesets(
            session_id="session_123",
            base_db_path="./base.db",
            changesets=[changeset1, changeset2, ...]
        )
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(self, session_id: str, db_path: str | Path, create_schema: bool = True):
        """
        Initialize session store.

        Args:
            session_id: Unique identifier for this session
            db_path: Path to SQLite database file
            create_schema: If True, creates schema if it doesn't exist
        """
        self.session_id = session_id
        self.db_path = Path(db_path)
        self.conn = apsw.Connection(str(self.db_path))
        self._session: apsw.Session | None = None
        self._current_version = 0

        if create_schema:
            self._create_schema()

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()

        # Schema version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS _schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Session metadata (single row since one DB = one session)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_metadata (
                version INTEGER NOT NULL DEFAULT 0,
                current_agent_id TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (current_agent_id) REFERENCES agent_states(agent_id)
            )
        """)

        # User data - stored as key-value pairs for granular delta tracking
        # Each key can be individually updated without affecting others
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS userdata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                is_encrypted BOOLEAN NOT NULL DEFAULT 0
            )
        """)

        # Session-level chat history (AgentSession._chat_ctx)
        # Uses item_id from ChatMessage.id for diff-based updates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                item_id TEXT PRIMARY KEY,
                item_data TEXT NOT NULL,
                is_encrypted BOOLEAN NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            )
        """)

        # Available tools
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                tool_name TEXT PRIMARY KEY
            )
        """)

        # Agent states - metadata and configuration
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT PRIMARY KEY,
                agent_cls BLOB,
                parent_id TEXT,
                init_kwargs_json TEXT,
                custom_state_json TEXT,
                runtime_stack BLOB,
                FOREIGN KEY (parent_id) REFERENCES agent_states(agent_id)
            )
        """)

        # Agent-specific chat contexts (Agent._chat_ctx)
        # Uses item_id from ChatMessage.id for diff-based updates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_chat_items (
                agent_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                item_data TEXT NOT NULL,
                is_encrypted BOOLEAN NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                PRIMARY KEY (agent_id, item_id),
                FOREIGN KEY (agent_id) REFERENCES agent_states(agent_id)
            )
        """)

        # Indices for efficient queries (ordered by created_at for sorting)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_history_created ON session_history(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_chat_created "
            "ON agent_chat_items(agent_id, created_at)"
        )

        # Store schema version
        cursor.execute(
            "INSERT OR REPLACE INTO _schema_metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )

    def begin_tracking(self) -> None:
        """Begin tracking changes for delta generation."""
        if self._session is not None:
            raise RuntimeError("Already tracking changes. Call end_tracking() first.")

        self._session = apsw.Session(self.conn)
        self._session.attach(None)

        logger.debug(
            "began tracking session changes",
            extra={"session_id": self.session_id, "version": self._current_version},
        )

    def end_tracking(self) -> bytes | None:
        """
        End tracking and return changeset.

        Returns:
            Changeset bytes if there were changes, None otherwise
        """
        if self._session is None:
            raise RuntimeError("Not currently tracking. Call begin_tracking() first.")

        try:
            changeset = self._session.changeset()
            return changeset if len(changeset) > 0 else None
        finally:
            self._session = None
            logger.debug(
                "ended tracking session changes",
                extra={"session_id": self.session_id, "version": self._current_version},
            )

    def init_session(
        self, *, current_agent_id: str = "default_agent", passphrase: str | None = None
    ) -> int:
        """
        Initialize a new session with metadata and default agent.

        Creates session_metadata row and initial agent_states entry.

        Args:
            current_agent_id: ID of the initial agent (default: "default_agent")
            passphrase: Optional passphrase for encryption

        Returns:
            Initial version number (0)
        """
        raise NotImplementedError

    def save_session_state(self, new_state: dict[str, Any]) -> int:
        """
        Save session state to database, incrementing version.

        Takes the output from AgentSession.get_state() which has:
        {
            "userdata": dict,
            "tools": list[str],
            "history": dict,  # from chat_ctx.to_dict()
            "agent": dict,    # from Agent.get_state()
        }

        Args:
            new_state: State dict from AgentSession.get_state()

        Returns:
            New version number
        """
        cursor = self.conn.cursor()
        self._current_version += 1
        now = time.time()

        # ===== PHASE 1: GET CURRENT STATE AND COMPUTE DIFF =====
        current_state = self.get_session_state()
        operations = self._compute_state_diff(current_state, new_state)

        # ===== PHASE 2: APPLY ALL OPERATIONS =====
        for op in operations:
            self._apply_operation(cursor, op)

        logger.debug(
            "saved session state",
            extra={
                "session_id": self.session_id,
                "version": self._current_version,
                "num_operations": len(operations),
            },
        )

        return self._current_version

    def _compute_state_diff(
        self, old_state: dict[str, Any], new_state: dict[str, Any]
    ) -> list[DBOperation]:
        """
        Compute database operations needed to transform current_state into new_state.

        Returns list of DBOperation objects.
        """
        from ..llm import ChatContext
        from ..llm.utils import compute_chat_ctx_diff

        operations: list[DBOperation] = []
        now = time.time()

        # 1. Metadata operations
        is_first_save = not old_state.get("agent")
        new_agent_id = new_state["agent"]["id"]

        if is_first_save:
            operations.append(
                DBOperation(
                    table="session_metadata",
                    operation="INSERT",
                    data={
                        "version": self._current_version,
                        "current_agent_id": new_agent_id,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
            )
        else:
            operations.append(
                DBOperation(
                    table="session_metadata",
                    operation="UPDATE",
                    data={
                        "version": self._current_version,
                        "current_agent_id": new_agent_id,
                        "updated_at": now,
                    },
                    where={},  # Single row table, no WHERE needed
                )
            )

        # 2. Userdata operations
        current_userdata = old_state.get("userdata", {})
        new_userdata = new_state.get("userdata", {})

        for key, value in new_userdata.items():
            value_text, is_encrypted = self._serialize_data(value, None)
            if key in current_userdata:
                # Update if changed
                if current_userdata[key] != value:
                    operations.append(
                        DBOperation(
                            table="userdata",
                            operation="UPDATE",
                            data={"value": value_text, "is_encrypted": is_encrypted},
                            where={"key": key},
                        )
                    )
            else:
                # Insert new key
                operations.append(
                    DBOperation(
                        table="userdata",
                        operation="INSERT",
                        data={"key": key, "value": value_text, "is_encrypted": is_encrypted},
                    )
                )

        # Delete removed keys
        for key in current_userdata:
            if key not in new_userdata:
                operations.append(
                    DBOperation(table="userdata", operation="DELETE", where={"key": key})
                )

        # 3. Tools operations
        current_tools = set(old_state.get("tools", []))
        new_tools = set(new_state.get("tools", []))

        for tool_name in new_tools - current_tools:
            operations.append(
                DBOperation(table="tools", operation="INSERT", data={"tool_name": tool_name})
            )

        for tool_name in current_tools - new_tools:
            operations.append(
                DBOperation(table="tools", operation="DELETE", where={"tool_name": tool_name})
            )

        # 4. Session history operations
        current_history = ChatContext.from_dict(old_state.get("history", {"items": []}))
        new_history = ChatContext.from_dict(new_state.get("history", {"items": []}))

        current_history_ids = [item.id for item in current_history.items]
        history_diff = compute_chat_ctx_diff(current_history_ids, new_history)

        # Delete operations
        for item_id in history_diff.to_remove:
            operations.append(
                DBOperation(table="session_history", operation="DELETE", where={"item_id": item_id})
            )

        # Insert/Update operations
        new_history_by_id = {item.id: item for item in new_history.items}
        for _, item_id in history_diff.to_create + history_diff.to_update:
            item = new_history_by_id[item_id]
            item_dict = item.model_dump()
            item_text, is_encrypted = self._serialize_data(item_dict, None)
            operations.append(
                DBOperation(
                    table="session_history",
                    operation="INSERT",  # Use INSERT OR REPLACE
                    data={
                        "item_id": item.id,
                        "item_data": item_text,
                        "is_encrypted": is_encrypted,
                        "created_at": item.created_at,
                    },
                )
            )

        # 5. Agent states operations (recursive)
        current_agent = old_state.get("agent", {})
        new_agent = new_state.get("agent", {})
        agent_ops = self._compute_agent_diff_recursive(current_agent, new_agent)
        operations.extend(agent_ops)

        return operations

    def _compute_agent_diff_recursive(
        self, current_agent: dict[str, Any], new_agent: dict[str, Any]
    ) -> list[DBOperation]:
        """Recursively compute agent state operations."""
        if not new_agent:
            return []

        from ..llm import ChatContext
        from ..llm.utils import compute_chat_ctx_diff

        operations: list[DBOperation] = []

        # Process parent agent first (if exists)
        current_parent = current_agent.get("parent_agent", {})
        new_parent = new_agent.get("parent_agent", {})
        if new_parent:
            operations.extend(self._compute_agent_diff_recursive(current_parent, new_parent))

        # Process current agent
        agent_id = new_agent["id"]
        agent_cls = new_agent["cls"]
        init_kwargs = new_agent.get("init_kwargs", {})
        new_chat_ctx_dict = new_agent.get("chat_ctx", {})

        # Extract custom fields
        standard_fields = {"cls", "id", "init_kwargs", "tools", "chat_ctx", "parent_agent"}
        custom_state = {k: v for k, v in new_agent.items() if k not in standard_fields}

        # Agent metadata operation
        operations.append(
            DBOperation(
                table="agent_states",
                operation="INSERT",  # Use INSERT OR REPLACE
                data={
                    "agent_id": agent_id,
                    "agent_cls": pickle.dumps(agent_cls),
                    "parent_id": new_parent["id"] if new_parent else None,
                    "init_kwargs_json": json.dumps(init_kwargs) if init_kwargs else None,
                    "custom_state_json": json.dumps(custom_state) if custom_state else None,
                    "runtime_stack": None,  # Placeholder for future runtime binary stack
                },
            )
        )

        # Agent chat context operations
        current_chat_ctx = ChatContext.from_dict(current_agent.get("chat_ctx", {"items": []}))
        new_chat_ctx = ChatContext.from_dict(new_chat_ctx_dict)

        current_chat_ids = [item.id for item in current_chat_ctx.items]
        chat_diff = compute_chat_ctx_diff(current_chat_ids, new_chat_ctx)

        # Delete operations
        for item_id in chat_diff.to_remove:
            operations.append(
                DBOperation(
                    table="agent_chat_items",
                    operation="DELETE",
                    where={"agent_id": agent_id, "item_id": item_id},
                )
            )

        # Insert/Update operations
        new_chat_by_id = {item.id: item for item in new_chat_ctx.items}
        for _, item_id in chat_diff.to_create + chat_diff.to_update:
            item = new_chat_by_id[item_id]
            item_dict = item.model_dump()
            item_text, is_encrypted = self._serialize_data(item_dict, None)
            operations.append(
                DBOperation(
                    table="agent_chat_items",
                    operation="INSERT",  # Use INSERT OR REPLACE
                    data={
                        "agent_id": agent_id,
                        "item_id": item.id,
                        "item_data": item_text,
                        "is_encrypted": is_encrypted,
                        "created_at": item.created_at,
                    },
                )
            )

        return operations

    def _apply_operation(self, cursor: apsw.Cursor, op: DBOperation) -> None:
        """Apply a single database operation."""
        if op.operation == "INSERT":
            # Use INSERT OR REPLACE for tables with primary keys
            columns = ", ".join(op.data.keys())
            placeholders = ", ".join("?" * len(op.data))
            values = tuple(op.data.values())
            sql = f"INSERT OR REPLACE INTO {op.table} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, values)

        elif op.operation == "UPDATE":
            set_clause = ", ".join(f"{k} = ?" for k in op.data.keys())
            values = list(op.data.values())

            if op.where:
                where_clause = " AND ".join(f"{k} = ?" for k in op.where.keys())
                values.extend(op.where.values())
                sql = f"UPDATE {op.table} SET {set_clause} WHERE {where_clause}"
            else:
                sql = f"UPDATE {op.table} SET {set_clause}"

            cursor.execute(sql, tuple(values))

        elif op.operation == "DELETE":
            if op.where:
                where_clause = " AND ".join(f"{k} = ?" for k in op.where.keys())
                values = tuple(op.where.values())
                sql = f"DELETE FROM {op.table} WHERE {where_clause}"
                cursor.execute(sql, values)
            else:
                sql = f"DELETE FROM {op.table}"
                cursor.execute(sql)

    def get_session_state(self) -> dict[str, Any]:
        """
        Get current session state from database in AgentSession.get_state() format.

        Returns:
            State dict with keys: userdata, tools, history, agent
            Returns empty state if no data exists.
        """
        cursor = self.conn.cursor()

        # Check if metadata exists
        meta = cursor.execute("SELECT version, current_agent_id FROM session_metadata").fetchone()
        if not meta:
            # No state exists yet
            return {"userdata": {}, "tools": [], "history": {"items": []}, "agent": {}}

        _, current_agent_id = meta

        # Load userdata
        userdata = {}
        for key, value_text, is_encrypted in cursor.execute(
            "SELECT key, value, is_encrypted FROM userdata"
        ):
            userdata[key] = self._deserialize_data(value_text, bool(is_encrypted), None)

        # Load tools
        tools = [row[0] for row in cursor.execute("SELECT tool_name FROM tools ORDER BY tool_name")]

        # Load session history
        history_items = []
        for item_id, item_data, is_encrypted, created_at in cursor.execute(
            "SELECT item_id, item_data, is_encrypted, created_at FROM session_history ORDER BY created_at"
        ):
            item_dict = self._deserialize_data(item_data, bool(is_encrypted), None)
            history_items.append(item_dict)

        history = {"items": history_items}

        # Load agent state (current agent)
        agent = (
            self._load_agent_state_recursive(cursor, current_agent_id) if current_agent_id else {}
        )

        return {
            "userdata": userdata,
            "tools": tools,
            "history": history,
            "agent": agent,
        }

    def _load_agent_state_recursive(
        self, cursor: apsw.Cursor, agent_id: str
    ) -> dict[str, Any] | None:
        """Load agent state recursively including parent agents."""
        row = cursor.execute(
            "SELECT agent_cls, parent_id, init_kwargs_json, custom_state_json, runtime_stack FROM agent_states WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()

        if not row:
            return None

        agent_cls_blob, parent_id, init_kwargs_json, custom_state_json, runtime_stack = row

        # Unpickle agent class
        agent_cls = pickle.loads(agent_cls_blob) if agent_cls_blob else None

        # Parse JSON fields
        init_kwargs = json.loads(init_kwargs_json) if init_kwargs_json else {}
        custom_state = json.loads(custom_state_json) if custom_state_json else {}

        # Load agent's chat context
        chat_items = []
        for item_id, item_data, is_encrypted, created_at in cursor.execute(
            "SELECT item_id, item_data, is_encrypted, created_at FROM agent_chat_items WHERE agent_id = ? ORDER BY created_at",
            (agent_id,),
        ):
            item_dict = self._deserialize_data(item_data, bool(is_encrypted), None)
            chat_items.append(item_dict)

        chat_ctx = {"items": chat_items}

        # Load agent tools (Note: Agent stores its own tools list)
        # We'll get this from custom_state if it was saved there
        tools = custom_state.pop("tools", [])

        # Build agent state
        agent_state = {
            "cls": agent_cls,
            "id": agent_id,
            "init_kwargs": init_kwargs,
            "tools": tools,
            "chat_ctx": chat_ctx,
            **custom_state,  # Add any custom fields
        }

        # Recursively load parent agent
        if parent_id:
            parent_agent = self._load_agent_state_recursive(cursor, parent_id)
            agent_state["parent_agent"] = parent_agent

        return agent_state

    def load_session_state(self, passphrase: str | None = None) -> StoredState:
        """
        Load complete session state from database.

        Args:
            passphrase: Optional passphrase for decrypting encrypted data

        Returns:
            StoredState containing all session data
        """
        state = self.get_session_state()
        return StoredState(
            session_id=self.session_id,
            version=self._current_version,
            **state,
        )

    def apply_changeset(self, changeset: bytes, conflict_strategy: str = "replace") -> None:
        """
        Apply a changeset to the database.

        Args:
            changeset: Changeset bytes from end_tracking()
            conflict_strategy: How to handle conflicts ("replace" or "abort")
                              "replace" = take new value (default for single writer)
                              "abort" = raise error on conflict
        """
        raise NotImplementedError

    def get_version(self) -> int:
        """
        Get current version number.

        Returns:
            Current version number
        """
        return self._current_version

    @classmethod
    def from_base_and_changesets(
        cls,
        session_id: str,
        base_db_path: str | Path,
        changesets: list[bytes],
        output_db_path: str | Path | None = None,
    ) -> SessionStore:
        """
        Create a session store by applying changesets to a base database.

        Typically used on worker side to reconstruct current state from
        base snapshot plus incremental changes.

        Args:
            session_id: Session identifier
            base_db_path: Path to base database
            changesets: List of changesets to apply in order
            output_db_path: Where to create the new DB (default: temp file)

        Returns:
            SessionStore with all changes applied
        """
        raise NotImplementedError

    def create_snapshot(self, output_path: str | Path) -> None:
        """
        Create a snapshot (full copy) of the current database.

        Args:
            output_path: Where to save the snapshot
        """
        raise NotImplementedError

    # Helper methods (internal)

    def _serialize_data(self, data: Any, passphrase: str | None) -> tuple[str, bool]:
        """
        Serialize data to TEXT format, optionally with encryption.

        Uses JSON for simple data, pickle+base64 for complex data.
        Encryption not yet implemented (always returns is_encrypted=False).

        Returns:
            tuple of (serialized_text, is_encrypted)
        """
        import base64

        # TODO: Implement encryption when passphrase is provided
        # For now, always store unencrypted

        # Try JSON first (more readable and efficient)
        try:
            return json.dumps(data), False
        except (TypeError, ValueError):
            # Fall back to pickle + base64 for complex types
            pickled = pickle.dumps(data)
            return base64.b64encode(pickled).decode("ascii"), False

    def _deserialize_data(
        self, data_text: str | None, is_encrypted: bool, passphrase: str | None
    ) -> Any:
        """
        Deserialize data from TEXT format, handling encryption.

        Returns:
            Deserialized Python object
        """
        import base64

        if data_text is None:
            return None

        # TODO: Implement decryption when is_encrypted=True
        if is_encrypted:
            raise NotImplementedError("Encryption not yet implemented")

        # Try JSON first
        try:
            return json.loads(data_text)
        except (json.JSONDecodeError, ValueError):
            # Fall back to pickle (stored as base64)
            try:
                pickled = base64.b64decode(data_text)
                return pickle.loads(pickled)
            except Exception:
                # If all else fails, return the raw text
                return data_text

    def _insert_chat_item(
        self,
        cursor: apsw.Cursor,
        table_name: str,
        agent_id: str | None,
        item: Any,
        passphrase: str | None,
    ) -> None:
        """
        Insert or replace a single chat item into the database.

        Args:
            cursor: Database cursor
            table_name: "session_history" or "agent_chat_items"
            agent_id: Required for agent_chat_items, None for session_history
            item: Chat item (dict or object with id and created_at)
            passphrase: Optional encryption passphrase
        """
        raise NotImplementedError

    def _save_chat_items_diff(
        self,
        cursor: apsw.Cursor,
        table_name: str,
        agent_id: str | None,
        new_items: list[Any],
        passphrase: str | None,
    ) -> None:
        """
        Save chat items using diff-based approach.

        Computes INSERT/UPDATE/DELETE operations based on item IDs.

        Args:
            cursor: Database cursor
            table_name: "session_history" or "agent_chat_items"
            agent_id: Required for agent_chat_items, None for session_history
            new_items: List of chat items with id field
            passphrase: Optional encryption passphrase
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the database connection."""
        if self._session is not None:
            self._session = None
        self.conn.close()
        logger.debug("closed session store", extra={"session_id": self.session_id})

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self.close()
