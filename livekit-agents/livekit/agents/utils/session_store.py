from __future__ import annotations

import base64
import json
import os
import pickle
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import apsw

from ..log import logger
from . import is_given

if TYPE_CHECKING:
    from ..voice.agent import _AgentState
    from ..voice.agent_session import _AgentSessionState


SCHEMA_VERSION = 1
SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS session (
        version INTEGER NOT NULL PRIMARY KEY,
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


@dataclass
class SessionDelta:
    version: int
    changeset: bytes


class SessionStore:
    def __init__(self, db_file: str | bytes | None, *, create_schema: bool = True):
        """
        Initialize session store.

        Args:
            db_file: path to SQLite database file, or bytes of the database file
                if None, a temporary database will be created
            create_schema: whether to create the schema if it doesn't exist
        """
        self._temp_file: str | None = None
        if db_file is None or isinstance(db_file, bytes):
            self._temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
            if isinstance(db_file, bytes):
                with open(self._temp_file, "wb") as f:
                    f.write(db_file)
            self._db_path = self._temp_file
        else:
            self._db_path = str(db_file)

        self._conn = apsw.Connection(self._db_path)

        if create_schema:
            if (
                schema_version := self.get_schema_version()
            ) is not None and schema_version != SCHEMA_VERSION:
                raise ValueError(
                    f"Schema version mismatch when creating schema: {schema_version} != {SCHEMA_VERSION}"
                )
            self._create_schema()

        # load version from database if it exists
        cursor = self._conn.cursor()
        result = cursor.execute("SELECT version FROM session").fetchone()
        self._version = int(result[0]) if result else 0

    def get_schema_version(self) -> int | None:
        """Get the schema version from the database."""
        cursor = self._conn.cursor()
        try:
            result = cursor.execute(
                "SELECT value FROM _schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
        except apsw.SQLError:
            return None
        return int(result[0]) if result else None

    @property
    def version(self) -> str | None:
        return self._version

    @classmethod
    def from_state(cls, state: _AgentSessionState, *, version: int = 0) -> SessionStore:
        """Write state dict to database."""
        store = cls(db_file=None)
        cursor = store._conn.cursor()

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
        store._write_agent_state(cursor, state["agent"])

        # update version
        store._version = version
        return store

    def export_state(self) -> _AgentSessionState:
        """Export current session state from database."""
        from ..voice.agent_session import _AgentSessionState

        cursor = self._conn.cursor()

        meta = cursor.execute(
            "SELECT version, current_agent_id, tools_json FROM session"
        ).fetchone()
        if not meta:
            raise ValueError("session not initialized")

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

        return _AgentSessionState(
            userdata=userdata,
            tools=tools,
            history=history,
            agent=agent,
        )

    def update_state(self, state: _AgentSessionState) -> SessionDelta:
        """Update session state and return the changeset."""
        with SessionStore.from_state(state, version=self._version + 1) as target:
            delta = self.compute_delta(target)
            self.apply_changesets([delta])
            return delta

    def compute_delta(self, target_store: SessionStore) -> SessionDelta:
        """Compute changeset from this store to target store using SQLite's session diff."""

        # verify schema versions match
        curr_schema_version = self.get_schema_version()
        target_schema_version = target_store.get_schema_version()
        if curr_schema_version != target_schema_version:
            raise ValueError(
                f"Schema version mismatch: source={curr_schema_version}, target={target_schema_version}. "
                "Cannot compute changeset between different schema versions."
            )

        # attach target database to current connection for diff
        attached_name = "_target_diff_db"
        try:
            # attach target database
            self._conn.execute(f"ATTACH DATABASE ? AS {attached_name}", (target_store._db_path,))

            # we want a changeset that transforms THIS db (main) into TARGET db.
            # so we create a session on the attached TARGET database and diff FROM main.
            # this gives us the changes needed to make main match target
            session = apsw.Session(self._conn, attached_name)
            session.attach(None)

            # Get table names from database, excluding internal tables (prefixed with _)
            cursor = self._conn.cursor()
            tables = cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '\\_%' ESCAPE '\\'"
            ).fetchall()
            for (table_name,) in tables:
                session.diff("main", table_name)

            changeset = session.changeset()

            return SessionDelta(version=target_store.version, changeset=changeset)

        finally:
            # detach the target database
            try:
                self._conn.execute(f"DETACH DATABASE {attached_name}")
            except Exception:
                pass

    def apply_changesets(self, changesets: list[SessionDelta]) -> None:
        """Apply a list of changesets in order, verifying versions in the changeset chain."""
        for cs_meta in changesets:
            # verify base version matches current
            if self._version + 1 != cs_meta.version:
                raise ValueError(
                    f"Changeset version {cs_meta.version} does not match current version {self._version} + 1"
                )

            # apply changeset
            if cs_meta.changeset:

                def conflict_handler(conflict_reason: int, table_change: Any) -> int:
                    # Always take the new value (single writer scenario)
                    return apsw.SQLITE_CHANGESET_REPLACE

                apsw.Changeset.apply(cs_meta.changeset, self._conn, conflict=conflict_handler)

            # read version from DB and verify
            cursor = self._conn.cursor()
            result = cursor.execute("SELECT version FROM session").fetchone()
            db_version = result[0] if result else None

            if db_version != cs_meta.version:
                raise ValueError(
                    f"Version mismatch after applying changeset! "
                    f"Expected {cs_meta.version}, got {db_version}"
                )

            self._version = cs_meta.version

            logger.debug(
                "applied changeset",
                extra={"version": cs_meta.version, "changeset_size": len(cs_meta.changeset)},
            )

    def export_snapshot(self) -> bytes:
        with open(self._db_path, "rb") as f:
            return f.read()

    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        cursor = self._conn.cursor()

        # schema version tracking table (internal, not part of session data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS _schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        for _ in cursor.execute(SCHEMA_SQL):
            pass

        # store schema version
        cursor.execute(
            "INSERT OR REPLACE INTO _schema_metadata (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )

    def _write_agent_state(self, cursor: apsw.Cursor, agent_state: _AgentState) -> None:
        """Recursively write agent state and parents to DB."""
        from ..voice.agent import _AgentState

        # write parent first (if exists)
        parent_agent = agent_state.get("parent_agent")
        if parent_agent:
            self._write_agent_state(cursor, parent_agent)

        # write current agent
        agent_id = agent_state["id"]
        agent_type = agent_state["cls"]
        agent_tools = agent_state.get("tools", [])
        init_kwargs = agent_state.get("init_kwargs", {})
        chat_ctx_dict = agent_state.get("chat_ctx", {})
        durable_state = agent_state.get("durable_state", None)

        # filter out NOT_GIVEN values
        init_kwargs = {k: v for k, v in init_kwargs.items() if is_given(v)}

        # extract custom fields
        standard_keys = set(_AgentState.__required_keys__) | set(_AgentState.__optional_keys__)
        custom_state = {k: v for k, v in agent_state.items() if k not in standard_keys}

        # write agent metadata
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

        # write agent chat context
        for item in chat_ctx_dict["items"]:
            item_text = self._serialize_data(item, passphrase=None)
            cursor.execute(
                "INSERT INTO agent_chat_ctx (agent_id, item_id, item_data, is_encrypted, created_at) VALUES (?, ?, ?, ?, ?)",
                (agent_id, item["id"], item_text, False, item["created_at"]),
            )

    def _load_agent(self, cursor: apsw.Cursor, agent_id: str) -> _AgentState | None:
        """Load agent state recursively including parent agents."""
        from ..voice.agent import _AgentState

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

        # build agent state
        agent_state = _AgentState(
            cls=agent_class,
            id=agent_id,
            init_kwargs=init_kwargs,
            tools=tools,
            chat_ctx=chat_ctx,
            durable_state=durable_state,
        )
        agent_state.update(custom_state)

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
        self._conn.close()
        if self._temp_file and os.path.exists(self._temp_file):
            os.unlink(self._temp_file)
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
