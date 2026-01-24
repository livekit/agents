from __future__ import annotations

import hashlib
import json
import pickle
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import apsw
import apsw.ext

from ..log import logger


@dataclass
class SessionSnapshot:
    """
    Full database snapshot for initial upload or full sync.

    Used when:
    - First time creating session in cloud
    - Cloud requests full resync
    - Worker needs to download entire state
    """

    version_hash: str  # Content hash of the database
    db_data: bytes  # Complete database file content
    timestamp: float  # When snapshot was created

    def size(self) -> int:
        """Size of the snapshot in bytes."""
        return len(self.db_data)


@dataclass
class SessionDelta:
    """
    Incremental changeset for efficient sync (formerly ChangesetMetadata).

    Used for normal operation after initial snapshot exists.
    """

    base_version: str | None  # Hash of the base version (None for initial)
    new_version: str  # Hash of the new version (content-addressable)
    changeset: bytes  # SQLite changeset bytes (binary delta)
    timestamp: float  # When this delta was created

    def size(self) -> int:
        """Size of the changeset in bytes."""
        return len(self.changeset)


# Tables that are part of session data (used for session diff)
SESSION_TABLES = [
    "session_metadata",
    "userdata",
    "tools",
    "session_history",
    "agent_states",
    "agent_chat_items",
]


class SessionStore:
    """
    SQLite-based session store with git-like hash-based versioning.
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(self, db_file: str | Path | bytes | None, *, create_schema: bool = True):
        """
        Initialize session store.

        Args:
            db_path: Path to SQLite database file, if None, a temporary database will be created
            create_schema: If True, creates schema if it doesn't exist
        """
        self._temp_file: Path | None = None
        if db_file is None or isinstance(db_file, bytes):
            self._temp_file = Path(tempfile.NamedTemporaryFile(suffix=".db", delete=False).name)
            if isinstance(db_file, bytes):
                with open(self._temp_file, "wb") as f:
                    f.write(db_file)
            self._db_path = self._temp_file
        else:
            self._db_path = Path(db_file)

        self.conn = apsw.Connection(str(self._db_path))
        self._current_version: str | None = None  # Hash of current version

        if create_schema:
            self._create_schema()

        self._load_current_version()

    def _load_current_version(self) -> None:
        """Load current version hash from database."""
        cursor = self.conn.cursor()
        result = cursor.execute("SELECT version_hash FROM session_metadata").fetchone()
        if result:
            self._current_version = result[0]

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> SessionStore:
        """
        Create a SessionStore from a state dict.

        Args:
            state: State dict from AgentSession.get_state()

        Returns:
            SessionStore with state written to DB
        """
        store = cls(db_file=None, create_schema=True)

        # Compute version hash from state
        version_hash = hashlib.sha256(pickle.dumps(state)).hexdigest()

        # Write state to DB with version
        store._write_state_to_db(state, version_hash)
        store._current_version = version_hash

        return store

    def compute_changesets(self, target_store: SessionStore) -> SessionDelta:
        """
        Compute changeset from this store to target store using SQLite's session diff.

        Uses sqlite3session_diff to directly compare two databases and generate
        a changeset, which is more efficient than manual row-by-row comparison.

        Args:
            target_store: The target SessionStore to compute diff to

        Returns:
            SessionDelta with base_version (this), new_version (target), and changeset
        """
        # Attach target database to current connection for diff
        # Use a unique alias to avoid conflicts
        attached_name = "_target_diff_db"

        try:
            # Attach target database
            self.conn.execute(
                f"ATTACH DATABASE ? AS {attached_name}", (str(target_store._db_path),)
            )

            # session.diff(from_schema, table) generates changes to make
            # from_schema match the session's database schema.
            #
            # We want a changeset that transforms THIS db (main) into TARGET db.
            # So we create a session on the attached TARGET database and diff FROM main.
            # This gives us: changes needed to make main match target (correct direction!)
            session = apsw.Session(self.conn, attached_name)

            # Attach all tables we want to diff
            for table_name in SESSION_TABLES:
                session.attach(table_name)

            # Diff each table from main -> target
            cursor = self.conn.cursor()
            for table_name in SESSION_TABLES:
                main_exists = cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                ).fetchone()
                target_exists = cursor.execute(
                    f"SELECT name FROM {attached_name}.sqlite_master WHERE type='table' AND name=?",
                    (table_name,),
                ).fetchone()

                if main_exists and target_exists:
                    session.diff("main", table_name)

            # Get the changeset
            changeset = session.changeset()

            return SessionDelta(
                base_version=self._current_version,
                new_version=target_store.get_version(),
                changeset=changeset if len(changeset) > 0 else b"",
                timestamp=time.time(),
            )

        finally:
            # Detach the target database
            try:
                self.conn.execute(f"DETACH DATABASE {attached_name}")
            except Exception:
                pass  # Ignore detach errors

    def apply_changesets(self, changesets: list[SessionDelta]) -> None:
        """
        Apply a list of changesets in order, verifying version hashes.

        Args:
            changesets: List of SessionDelta to apply in order

        Raises:
            ValueError: If version hash mismatch detected
        """
        for cs_meta in changesets:
            # Verify base version matches current
            if cs_meta.base_version != self._current_version:
                raise ValueError(
                    f"Changeset base version {cs_meta.base_version} does not match "
                    f"current version {self._current_version}"
                )

            # Apply changeset
            if cs_meta.changeset:

                def conflict_handler(conflict_reason: int, table_change: Any) -> int:
                    # Always take the new value (single writer scenario)
                    return apsw.SQLITE_CHANGESET_REPLACE

                apsw.Changeset.apply(cs_meta.changeset, self.conn, conflict=conflict_handler)

            # Read version hash from DB and verify
            cursor = self.conn.cursor()
            result = cursor.execute("SELECT version_hash FROM session_metadata").fetchone()
            db_version = result[0] if result else None

            if db_version != cs_meta.new_version:
                raise ValueError(
                    f"Version hash mismatch after applying changeset! "
                    f"Expected {cs_meta.new_version}, got {db_version}"
                )

            self._current_version = cs_meta.new_version

            logger.debug(
                "applied changeset",
                extra={
                    "base_version": cs_meta.base_version,
                    "new_version": cs_meta.new_version,
                    "changeset_size": len(cs_meta.changeset),
                },
            )

    def get_version(self) -> str | None:
        """
        Get current version hash.

        Returns:
            Current version hash (SHA256 hex string) or None if no versions yet
        """
        return self._current_version

    def export_snapshot(self) -> SessionSnapshot:
        """
        Export the entire database as a snapshot for cloud upload.

        Use this for:
        - Initial upload to cloud
        - Full resync when deltas are unavailable

        Returns:
            SessionSnapshot containing full database content
        """
        # Read entire database file
        with open(self._db_path, "rb") as f:
            db_data = f.read()

        version = self.get_version()
        if not version:
            raise ValueError("Database has no version - cannot export snapshot")

        return SessionSnapshot(
            version_hash=version,
            db_data=db_data,
            timestamp=time.time(),
        )

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
                version_hash TEXT NOT NULL,
                current_agent_id TEXT,
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

    def _write_state_to_db(self, state: dict[str, Any], version_hash: str) -> None:
        """
        Write state dict to database (semantic conversion: state → DB).

        This is the only place where we understand the state structure.

        Args:
            state: State dict from AgentSession.get_state()
            version_hash: Pre-computed version hash for this state
        """
        from ..llm import ChatContext

        cursor = self.conn.cursor()

        # 1. Write metadata
        if state.get("agent"):
            cursor.execute(
                "INSERT INTO session_metadata (version_hash, current_agent_id) VALUES (?, ?)",
                (version_hash, state["agent"]["id"]),
            )

        # 2. Write userdata
        user_data = state.get("userdata") or {}
        for key, value in user_data.items():
            value_text, is_encrypted = self._serialize_data(value, None)
            cursor.execute(
                "INSERT INTO userdata (key, value, is_encrypted) VALUES (?, ?, ?)",
                (key, value_text, is_encrypted),
            )

        # 3. Write tools
        for tool in state.get("tools", []):
            cursor.execute("INSERT INTO tools (tool_name) VALUES (?)", (tool,))

        # 4. Write session history
        history_ctx = ChatContext.from_dict(state.get("history", {"items": []}))
        for item in history_ctx.items:
            item_dict = item.model_dump()
            item_text, is_encrypted = self._serialize_data(item_dict, None)
            cursor.execute(
                "INSERT INTO session_history (item_id, item_data, is_encrypted, created_at) VALUES (?, ?, ?, ?)",
                (item.id, item_text, is_encrypted, item.created_at),
            )

        # 5. Write agent states (recursive)
        self._write_agent_to_db_recursive(cursor, state.get("agent", {}))

    def _write_agent_to_db_recursive(
        self, cursor: apsw.Cursor, agent_state: dict[str, Any]
    ) -> None:
        """Recursively write agent state and parents to DB."""
        from ..llm import ChatContext

        if not agent_state:
            return

        # Write parent first (if exists)
        parent_agent = agent_state.get("parent_agent", {})
        if parent_agent:
            self._write_agent_to_db_recursive(cursor, parent_agent)

        # Write current agent
        agent_id = agent_state["id"]
        agent_cls = agent_state["cls"]
        init_kwargs = agent_state.get("init_kwargs", {})
        chat_ctx_dict = agent_state.get("chat_ctx", {})

        # Extract custom fields
        standard_fields = {"cls", "id", "init_kwargs", "tools", "chat_ctx", "parent_agent"}
        custom_state = {k: v for k, v in agent_state.items() if k not in standard_fields}

        # Write agent metadata
        cursor.execute(
            """
            INSERT INTO agent_states
            (agent_id, agent_cls, parent_id, init_kwargs_json, custom_state_json, runtime_stack)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                pickle.dumps(agent_cls),
                parent_agent["id"] if parent_agent else None,
                json.dumps(init_kwargs) if init_kwargs else None,
                json.dumps(custom_state) if custom_state else None,
                None,
            ),
        )

        # Write agent chat items
        chat_ctx = ChatContext.from_dict(chat_ctx_dict)
        for item in chat_ctx.items:
            item_dict = item.model_dump()
            item_text, is_encrypted = self._serialize_data(item_dict, None)
            cursor.execute(
                "INSERT INTO agent_chat_items (agent_id, item_id, item_data, is_encrypted, created_at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, item.id, item_text, is_encrypted, item.created_at),
            )

    def get_session_state(self) -> dict[str, Any]:
        """
        Get current session state from database in AgentSession.get_state() format.

        Returns:
            State dict with keys: userdata, tools, history, agent
            Returns empty state if no data exists.
        """
        cursor = self.conn.cursor()

        # Check if metadata exists
        meta = cursor.execute(
            "SELECT version_hash, current_agent_id FROM session_metadata"
        ).fetchone()
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
        for item_data, is_encrypted in cursor.execute(
            "SELECT item_data, is_encrypted FROM session_history ORDER BY created_at"
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
        for item_data, is_encrypted in cursor.execute(
            "SELECT item_data, is_encrypted FROM agent_chat_items WHERE agent_id = ? ORDER BY created_at",
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

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
        if self._temp_file and self._temp_file.exists():
            self._temp_file.unlink()
            self._temp_file = None

    def __enter__(self) -> SessionStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self.close()
