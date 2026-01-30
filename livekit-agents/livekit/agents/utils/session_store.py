from __future__ import annotations

import base64
import hashlib
import json
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import apsw

from ..log import logger
from . import is_given

# Fixed size for SHA-1 hash (40 hex characters)
VERSION_SIZE = 40


@dataclass
class SessionDelta:
    base_version: str | None  # Hash of the base version (None for initial)
    new_version: str  # Hash of the new version (content-addressable)
    changeset: bytes  # SQLite changeset bytes (binary delta)

    def dumps(self) -> bytes:
        base_ver_bytes = (self.base_version or ("0" * VERSION_SIZE)).encode("ascii")
        new_ver_bytes = self.new_version.encode("ascii")

        if len(base_ver_bytes) != VERSION_SIZE:
            raise ValueError(f"base_version must be {VERSION_SIZE} characters")
        if len(new_ver_bytes) != VERSION_SIZE:
            raise ValueError(f"new_version must be {VERSION_SIZE} characters")

        return base_ver_bytes + new_ver_bytes + self.changeset

    @classmethod
    def load(cls, data: bytes | str | Path) -> SessionDelta:
        if isinstance(data, (str, Path)):
            with open(data, "rb") as f:
                data = f.read()

        if len(data) < VERSION_SIZE * 2:
            raise ValueError(f"Data too short: expected at least {VERSION_SIZE * 2} bytes")

        base_ver_bytes = data[:VERSION_SIZE]
        new_ver_bytes = data[VERSION_SIZE : VERSION_SIZE * 2]
        changeset_bytes = data[VERSION_SIZE * 2 :]

        base_version = base_ver_bytes.decode("ascii")
        # convert "0"*40 back to None
        if base_version == "0" * VERSION_SIZE:
            base_version = None

        new_version = new_ver_bytes.decode("ascii")

        return cls(base_version=base_version, new_version=new_version, changeset=changeset_bytes)


SCHEMA_VERSION = 1
SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS session (
        version TEXT NOT NULL,
        current_agent_id TEXT,
        tools_json TEXT,
        FOREIGN KEY (current_agent_id) REFERENCES agent(id)
    );

    CREATE TABLE IF NOT EXISTS userdata (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        is_encrypted BOOLEAN NOT NULL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS session_history (
        item_id TEXT PRIMARY KEY,
        item_data TEXT NOT NULL,
        is_encrypted BOOLEAN NOT NULL DEFAULT 0,
        created_at REAL NOT NULL
    );

    CREATE TABLE IF NOT EXISTS agent (
        id TEXT PRIMARY KEY,
        runtime TEXT NOT NULL,
        class_type BLOB,
        parent_id TEXT,
        tools_json TEXT,
        init_kwargs_json TEXT,
        custom_state_json TEXT,
        durable_state BLOB,
        FOREIGN KEY (parent_id) REFERENCES agent(id)
    );

    CREATE TABLE IF NOT EXISTS agent_chat_ctx (
        agent_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        item_data TEXT NOT NULL,
        is_encrypted BOOLEAN NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        PRIMARY KEY (agent_id, item_id),
        FOREIGN KEY (agent_id) REFERENCES agent(id)
    );
"""


class SessionStore:
    def __init__(self, db_file: str | Path | bytes | None):
        """
        Initialize session store.

        Args:
            db_path: Path to SQLite database file, if None, a temporary database will be created
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
        self._version: str | None = None  # Hash of current version

        self._create_schema()

        # load version from database if it exists
        cursor = self.conn.cursor()
        result = cursor.execute("SELECT version FROM session").fetchone()
        if result:
            self._version = result[0]

    def get_schema_version(self) -> int | None:
        """Get the schema version from the database."""
        cursor = self.conn.cursor()
        result = cursor.execute(
            "SELECT value FROM _schema_metadata WHERE key = 'schema_version'"
        ).fetchone()
        return int(result[0]) if result else None

    @property
    def version(self) -> str | None:
        return self._version

    @classmethod
    def from_session_state(cls, state: dict[str, Any]) -> SessionStore:
        """
        Write state dict to database (semantic conversion: state → DB).
        Should be called only on a fresh SessionStore instance.
        """
        store = cls(db_file=None)
        version = hashlib.sha1(pickle.dumps(state)).hexdigest()
        cursor = store.conn.cursor()

        # write session metadata (version, current agent, tools)
        session_tools = state.get("tools", [])
        cursor.execute(
            "INSERT INTO session (version, current_agent_id, tools_json) VALUES (?, ?, ?)",
            (
                version,
                state["agent"]["id"] if state.get("agent") else None,
                json.dumps(session_tools) if session_tools else None,
            ),
        )

        # write userdata
        user_data = state.get("userdata") or {}
        for key, value in user_data.items():
            value_text = store._serialize_data(value, passphrase=None)
            cursor.execute(
                "INSERT INTO userdata (key, value, is_encrypted) VALUES (?, ?, ?)",
                (key, value_text, False),
            )

        # write session history
        session_history = state.get("history", {"items": []})
        for item in session_history["items"]:
            item_text = store._serialize_data(item, passphrase=None)
            cursor.execute(
                "INSERT INTO session_history (item_id, item_data, is_encrypted, created_at) VALUES (?, ?, ?, ?)",
                (item["id"], item_text, False, item["created_at"]),
            )

        # write agent state recursively
        store._write_agent_state(cursor, state.get("agent", {}))

        # update version
        store._version = version
        return store

    def export_session_state(self) -> dict[str, Any]:
        """Get current session state from database in AgentSession.get_state() format."""
        cursor = self.conn.cursor()

        meta = cursor.execute(
            "SELECT version, current_agent_id, tools_json FROM session"
        ).fetchone()
        if not meta:
            return {}

        _, current_agent_id, tools_json = meta

        # load tools from session
        tools = json.loads(tools_json) if tools_json else []

        # load userdata
        userdata = {}
        for key, value_text, is_encrypted in cursor.execute(
            "SELECT key, value, is_encrypted FROM userdata"
        ):
            userdata[key] = self._deserialize_data(value_text, bool(is_encrypted), None)

        # load session history
        history_items = []
        for item_data, is_encrypted in cursor.execute(
            "SELECT item_data, is_encrypted FROM session_history ORDER BY created_at"
        ):
            item_dict = self._deserialize_data(item_data, bool(is_encrypted), None)
            history_items.append(item_dict)

        history = {"items": history_items}

        # load agent state (current agent)
        agent = self._load_agent(cursor, current_agent_id) if current_agent_id else {}

        return {
            "userdata": userdata,
            "tools": tools,
            "history": history,
            "agent": agent,
        }

    def compute_delta(self, target_store: SessionStore) -> SessionDelta:
        """
        Compute changeset from this store to target store using SQLite's session diff.

        Uses sqlite3session_diff to directly compare two databases and generate
        a changeset, which is more efficient than manual row-by-row comparison.

        Args:
            target_store: The target SessionStore to compute diff to

        Returns:
            SessionDelta with base_version (this), new_version (target), and changeset

        Raises:
            ValueError: If schema versions don't match
        """
        # Verify schema versions match
        self_version = self.get_schema_version()
        target_version = target_store.get_schema_version()
        if self_version != target_version:
            raise ValueError(
                f"Schema version mismatch: source={self_version}, target={target_version}. "
                "Cannot compute changeset between different schema versions."
            )

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
            session.attach(None)

            # Diff each session table from main -> target
            # Get table names from database, excluding internal tables (prefixed with _)
            cursor = self.conn.cursor()
            tables = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\'"
            ).fetchall()
            for (table_name,) in tables:
                session.diff("main", table_name)

            # Get the changeset
            changeset = session.changeset()

            return SessionDelta(
                base_version=self._version,
                new_version=target_store.version,
                changeset=changeset,
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
            if cs_meta.base_version != self._version:
                raise ValueError(
                    f"Changeset base version {cs_meta.base_version} does not match "
                    f"current version {self._version}"
                )

            # Apply changeset
            if cs_meta.changeset:

                def conflict_handler(conflict_reason: int, table_change: Any) -> int:
                    # Always take the new value (single writer scenario)
                    return apsw.SQLITE_CHANGESET_REPLACE

                apsw.Changeset.apply(cs_meta.changeset, self.conn, conflict=conflict_handler)

            # Read version from DB and verify
            cursor = self.conn.cursor()
            result = cursor.execute("SELECT version FROM session").fetchone()
            db_version = result[0] if result else None

            if db_version != cs_meta.new_version:
                raise ValueError(
                    f"Version hash mismatch after applying changeset! "
                    f"Expected {cs_meta.new_version}, got {db_version}"
                )

            self._version = cs_meta.new_version

            logger.debug(
                "applied changeset",
                extra={
                    "base_version": cs_meta.base_version,
                    "new_version": cs_meta.new_version,
                    "changeset_size": len(cs_meta.changeset),
                },
            )

    def sync_snapshot(
        self, changesets: list[SessionDelta], *, inplace: bool = True
    ) -> SessionStore:
        """
        Sync a session store with a list of changesets to the latest version.

        Builds the correct changeset chain order, finds the starting point that matches
        the store's current version, then applies all subsequent changesets in order.

        Args:
            store: Base SessionStore to sync from
            changesets: List of SessionDelta (order not guaranteed)
            inplace: If True, modify store in place; if False, create a new store

        Returns:
            Updated SessionStore (same as input if inplace=True, new instance otherwise)

        Raises:
            ValueError: If no matching changeset found or chain is broken
        """
        if not changesets:
            return self

        # build a map of base_version -> SessionDelta for quick lookup
        version_map: dict[str | None, SessionDelta] = {}
        for delta in changesets:
            if delta.base_version in version_map:
                raise ValueError(
                    f"Multiple changesets found with same base_version={delta.base_version}"
                )
            version_map[delta.base_version] = delta

        current_version = self.version
        if current_version not in version_map:
            raise ValueError(
                f"No changeset found with base_version={current_version}. "
                f"Cannot sync from current version."
            )

        ordered_changesets: list[SessionDelta] = []
        next_version = current_version

        while next_version in version_map:
            delta = version_map[next_version]
            ordered_changesets.append(delta)
            next_version = delta.new_version

        if not inplace:
            db_data = self.export_database()
            store = SessionStore(db_file=db_data)
        else:
            store = self

        # apply the changesets in order
        store.apply_changesets(ordered_changesets)

        return store

    def export_database(self) -> bytes:
        with open(self._db_path, "rb") as f:
            return f.read()

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()

        # Schema version tracking table (internal, not part of session data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS _schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        for _ in cursor.execute(SCHEMA_SQL):
            pass

        # Store schema version
        cursor.execute(
            "INSERT OR REPLACE INTO _schema_metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )

    def _write_agent_state(self, cursor: apsw.Cursor, agent_state: dict[str, Any]) -> None:
        """Recursively write agent state and parents to DB."""
        if not agent_state:
            return

        # Write parent first (if exists)
        parent_agent = agent_state.get("parent_agent", {})
        if parent_agent:
            self._write_agent_state(cursor, parent_agent)

        # Write current agent
        agent_id = agent_state["id"]
        agent_type = agent_state["cls"]
        agent_tools = agent_state.get("tools", [])
        init_kwargs = agent_state.get("init_kwargs", {})
        chat_ctx_dict = agent_state.get("chat_ctx", {})
        durable_state = agent_state.get("durable_state", None)

        # filter out NOT_GIVEN values
        init_kwargs = {k: v for k, v in init_kwargs.items() if is_given(v)}

        # Extract custom fields
        standard_fields = {
            "cls",
            "id",
            "init_kwargs",
            "tools",
            "chat_ctx",
            "parent_agent",
            "durable_state",
        }
        custom_state = {k: v for k, v in agent_state.items() if k not in standard_fields}

        # Write agent metadata
        cursor.execute(
            """
            INSERT INTO agent
            (id, runtime, class_type, parent_id, tools_json, init_kwargs_json, custom_state_json, durable_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_id,
                "python",
                pickle.dumps(agent_type),
                parent_agent["id"] if parent_agent else None,
                json.dumps(agent_tools) if agent_tools else None,
                json.dumps(init_kwargs) if init_kwargs else None,
                json.dumps(custom_state) if custom_state else None,
                durable_state,
            ),
        )

        # Write agent chat context
        for item in chat_ctx_dict["items"]:
            item_text = self._serialize_data(item, passphrase=None)
            cursor.execute(
                "INSERT INTO agent_chat_ctx (agent_id, item_id, item_data, is_encrypted, created_at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, item["id"], item_text, False, item["created_at"]),
            )

    def _load_agent(self, cursor: apsw.Cursor, agent_id: str) -> dict[str, Any] | None:
        """Load agent state recursively including parent agents."""
        row = cursor.execute(
            "SELECT runtime, class_type, parent_id, tools_json, init_kwargs_json, custom_state_json, durable_state FROM agent WHERE id = ?",
            (agent_id,),
        ).fetchone()

        if not row:
            return None

        (
            runtime,
            class_type_blob,
            parent_id,
            tools_json,
            init_kwargs_json,
            custom_state_json,
            durable_state,
        ) = row

        # unpickle agent class (Python runtime only for now)
        if runtime != "python":
            raise ValueError(f"Unsupported agent runtime: {runtime}")
        agent_class = pickle.loads(class_type_blob) if class_type_blob else None

        # parse JSON fields
        tools = json.loads(tools_json) if tools_json else []
        init_kwargs = json.loads(init_kwargs_json) if init_kwargs_json else {}
        custom_state = json.loads(custom_state_json) if custom_state_json else {}

        # load agent's chat context
        chat_items = []
        for item_data, is_encrypted in cursor.execute(
            "SELECT item_data, is_encrypted FROM agent_chat_ctx WHERE agent_id = ? ORDER BY created_at",
            (agent_id,),
        ):
            item_dict = self._deserialize_data(item_data, bool(is_encrypted), None)
            chat_items.append(item_dict)

        chat_ctx = {"items": chat_items}

        # Build agent state
        agent_state = {
            "cls": agent_class,
            "id": agent_id,
            "init_kwargs": init_kwargs,
            "tools": tools,
            "chat_ctx": chat_ctx,
            "durable_state": durable_state,
            **custom_state,
        }

        # load parent agent
        if parent_id:
            agent_state["parent_agent"] = self._load_agent(cursor, parent_id)

        return agent_state

    # Helper methods (internal)

    def _serialize_data(self, data: Any, passphrase: str | None) -> str:
        # TODO: Implement encryption when passphrase is provided
        try:
            return json.dumps(data)
        except (TypeError, ValueError):
            pickled = pickle.dumps(data)
            return base64.b64encode(pickled).decode("ascii")

    def _deserialize_data(
        self, data_text: str | None, is_encrypted: bool, passphrase: str | None
    ) -> Any:
        if data_text is None:
            return None

        # TODO: Implement decryption when is_encrypted=True
        if is_encrypted:
            raise NotImplementedError("Encryption not yet implemented")

        try:
            return json.loads(data_text)
        except (json.JSONDecodeError, ValueError):
            pickled = base64.b64decode(data_text)
            return pickle.loads(pickled)

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
