from __future__ import annotations

import json
import os
import pickle
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import apsw

from livekit.protocol import agent

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
        userdata_blob BLOB,
        userdata_encrypted BOOLEAN NOT NULL DEFAULT 0,
        FOREIGN KEY (current_agent_id) REFERENCES agent(id)
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
        init_kwargs BLOB,
        extra_state BLOB,
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
    def __init__(self, db_file: str | Path | bytes | None, *, create_schema: bool = True):
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
    def version(self) -> int:
        return self._version

    @classmethod
    def from_state(cls, state: _AgentSessionState, *, version: int = 0) -> SessionStore:
        """Write state dict to database."""
        store = cls(db_file=None)
        cursor = store._conn.cursor()

        # write session metadata (version, current agent, tools)
        userdata_blob = (
            store._serialize_data(state.userdata, passphrase=None, output_type="bytes")
            if state.userdata
            else None
        )
        userdata_encrypted = False
        cursor.execute(
            "INSERT INTO session (version, current_agent_id, tools_json, userdata_blob, userdata_encrypted) VALUES (?, ?, ?, ?, ?)",
            (
                version,
                state.agent.id,
                json.dumps(state.tools) if state.tools else None,
                userdata_blob,
                userdata_encrypted,
            ),
        )

        # write session history
        session_history = state.history or {"items": []}
        for item in session_history["items"]:
            item_text = store._serialize_data(item, passphrase=None, output_type="text")
            cursor.execute(
                "INSERT INTO session_history (item_id, item_data, is_encrypted, created_at) VALUES (?, ?, ?, ?)",
                (item["id"], item_text, False, item["created_at"]),
            )

        # write agent state recursively
        store._write_agent_state(cursor, state.agent)

        # update version
        store._version = version
        return store

    def export_state(self) -> _AgentSessionState:
        """Export current session state from database."""
        from ..voice.agent_session import _AgentSessionState

        cursor = self._conn.cursor()

        meta = cursor.execute(
            "SELECT version, current_agent_id, tools_json, userdata_blob, userdata_encrypted FROM session"
        ).fetchone()
        if not meta:
            raise ValueError("session not initialized")

        _, current_agent_id, tools_json, userdata_blob, userdata_encrypted = meta

        # load tools from session
        tools = json.loads(tools_json) if tools_json else []
        userdata = self._deserialize_data(userdata_blob, userdata_encrypted, None)

        # load session history
        history_items = []
        for item_data, is_encrypted in cursor.execute(
            "SELECT item_data, is_encrypted FROM session_history ORDER BY created_at"
        ):
            item_dict = self._deserialize_data(item_data, bool(is_encrypted), None)
            history_items.append(item_dict)

        history = {"items": history_items}

        # load agent state (current agent)
        agent = self._load_agent(cursor, current_agent_id)
        if agent is None:
            raise ValueError(f"Agent with id {current_agent_id} not found")

        return _AgentSessionState(
            userdata=userdata,
            tools=tools,
            history=history,
            agent=agent,
        )

    def update_state(self, state: _AgentSessionState) -> tuple[int, bytes]:
        """Update session state and return the changeset."""
        with SessionStore.from_state(state, version=self._version + 1) as target:
            changeset = self.compute_delta(target)
            self.apply_changeset(changeset)
            return self._version, changeset

    def compute_delta(self, target_store: SessionStore) -> bytes:
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
                session.diff("main", str(table_name))

            changeset = session.changeset()

            return changeset

        finally:
            # detach the target database
            try:
                self._conn.execute(f"DETACH DATABASE {attached_name}")
            except Exception:
                pass

    def apply_changeset(self, changeset: bytes, *, version: int | None = None) -> None:
        """Apply a changeset to the database, verifying the version if provided."""
        # verify base version matches current
        if version is not None and version != self._version + 1:
            raise ValueError(
                f"Changeset version does not match current version: {version} != {self._version} + 1"
            )

        def conflict_handler(conflict_reason: int, table_change: Any) -> int:
            # Always take the new value (single writer scenario)
            return apsw.SQLITE_CHANGESET_REPLACE

        apsw.Changeset.apply(changeset, self._conn, conflict=conflict_handler)

        # read version from DB and verify
        cursor = self._conn.cursor()
        result = cursor.execute("SELECT version FROM session").fetchone()
        if not result:
            raise ValueError("session version not found after applying changeset")
        db_version = int(result[0])

        if version is not None and db_version != version:
            raise ValueError(
                f"Version mismatch after applying changeset! Expected {version}, got {db_version}"
            )

        self._version = db_version

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

    def _write_agent_state(self, cursor: apsw.Cursor, state: _AgentState) -> None:
        """Recursively write agent state and parents to DB."""
        # write parent first (if exists)
        parent_agent = state.parent_agent
        if parent_agent:
            self._write_agent_state(cursor, parent_agent)

        # filter out NOT_GIVEN values
        init_kwargs: dict[str, Any] = {k: v for k, v in state.init_kwargs.items() if is_given(v)}
        extra_state_blob = (
            self._serialize_data(state.extra_state, passphrase=None, output_type="bytes")
            if state.extra_state
            else None
        )

        # write agent metadata
        cursor.execute(
            """
            INSERT INTO agent
            (id, runtime, class_type, parent_id, tools_json, init_kwargs, extra_state, durable_state)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state.id,
                "python",
                pickle.dumps(state.cls),
                parent_agent.id if parent_agent else None,
                json.dumps(state.tools) if state.tools else None,
                self._serialize_data(init_kwargs, passphrase=None, output_type="bytes")
                if init_kwargs
                else None,
                extra_state_blob,
                state.durable_state,
            ),
        )

        # write agent chat context
        for item in state.chat_ctx["items"]:
            item_text = self._serialize_data(item, passphrase=None, output_type="text")
            cursor.execute(
                "INSERT INTO agent_chat_ctx (agent_id, item_id, item_data, is_encrypted, created_at) VALUES (?, ?, ?, ?, ?)",
                (state.id, item["id"], item_text, False, item["created_at"]),
            )

    def _load_agent(self, cursor: apsw.Cursor, agent_id: str) -> _AgentState | None:
        """Load agent state recursively including parent agents."""
        from ..voice.agent import _AgentState

        row = cursor.execute(
            "SELECT runtime, class_type, parent_id, tools_json, init_kwargs, extra_state, durable_state FROM agent WHERE id = ?",
            (agent_id,),
        ).fetchone()

        if not row:
            return None

        (
            runtime,
            class_type_blob,
            parent_id,
            tools_json,
            init_kwargs_blob,
            extra_state_blob,
            durable_state,
        ) = row

        # unpickle agent class (Python runtime only for now)
        if runtime != "python":
            raise ValueError(f"Unsupported agent runtime: {runtime}")
        agent_class = pickle.loads(class_type_blob)

        # parse JSON fields
        tools = json.loads(tools_json) if tools_json else []
        init_kwargs = self._deserialize_data(init_kwargs_blob, False, None) or {}
        extra_state = self._deserialize_data(extra_state_blob, False, None) or {}

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
            parent_agent=None,
            extra_state=extra_state,
        )

        # load parent agent
        if parent_id:
            agent_state.parent_agent = self._load_agent(cursor, parent_id)

        return agent_state

    # Helper methods (internal)

    def _serialize_data(
        self, data: Any, passphrase: str | None, output_type: Literal["text", "bytes"] = "text"
    ) -> str | bytes:
        # TODO: Implement encryption when passphrase is provided
        if output_type == "text":
            return json.dumps(data)
        elif output_type == "bytes":
            return pickle.dumps(data)
        else:
            raise ValueError(f"Invalid output_type: {output_type}")

    def _deserialize_data(
        self, data: str | bytes | None, is_encrypted: bool, passphrase: str | None
    ) -> Any:
        if data is None:
            return None

        # TODO: Implement decryption when is_encrypted=True
        if is_encrypted:
            raise NotImplementedError("Encryption not yet implemented")

        # Use data type to determine deserialization method
        if isinstance(data, bytes):
            return pickle.loads(data)
        elif isinstance(data, str):
            return json.loads(data)
        else:
            raise TypeError(f"Expected str or bytes, got {type(data)}")

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


@dataclass
class _SessionCacheEntry:
    size: int
    data: bytes

    @staticmethod
    def create(db_file: Path | bytes) -> _SessionCacheEntry:
        if isinstance(db_file, bytes):
            return _SessionCacheEntry(size=len(db_file), data=db_file)
        else:
            return _SessionCacheEntry(size=db_file.stat().st_size, data=b"")


class EphemeralSessionCache:
    """Cache manager for session database files.

    Maintains a temporary directory of cached SQLite database files with LRU eviction.
    The cache is ephemeral - it lives only for the worker's lifetime and is cleaned
    up on shutdown.
    """

    def __init__(
        self,
        *,
        cache_dir: Path | str | None = None,
        max_size_mb: int = 100,
        max_files: int = 100,
        in_memory: bool = False,
    ) -> None:
        """Initialize the session database cache.

        Args:
            cache_dir: Directory to store cached database files.
                If None, a temporary directory will be created.
            max_size_mb: Maximum total size of cached files in megabytes.
            max_files: Maximum number of cached files.
            in_memory: Whether to store session states in memory.
        """
        self._in_memory = in_memory
        if in_memory and cache_dir is not None:
            raise ValueError("cache_dir and in_memory cannot be provided together")

        if in_memory:
            self._cache_dir: Path | None = None
            self._temp_dir: tempfile.TemporaryDirectory | None = None
        elif cache_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="livekit_session_cache_")
            self._cache_dir = Path(self._temp_dir.name)
        else:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._temp_dir = None

        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_files = max_files

        self._cached_sessions = OrderedDict[str, _SessionCacheEntry]()

        logger.debug(
            "initialized session DB cache",
            extra={
                "cache_dir": str(self._cache_dir),
                "max_size_mb": max_size_mb,
                "max_files": max_files,
            },
        )

    def resolve(
        self, session_id: str, session_state: agent.AgentSessionState
    ) -> agent.AgentSessionState:
        """Resolve the session state from the cache or create a new one."""
        db_file = self._get_db_file(session_id)

        target_version = session_state.version
        which_oneof = session_state.WhichOneof("data")
        if which_oneof == "snapshot":
            if isinstance(db_file, bytes):
                db_file = session_state.snapshot
            else:
                db_file.write_bytes(session_state.snapshot)
            store = SessionStore(db_file, create_schema=False)
        else:
            store = SessionStore(db_file)

        with store:
            if store.version != target_version:
                if which_oneof == "delta":
                    store.apply_changeset(session_state.delta, version=target_version)
                else:
                    raise ValueError(
                        f"Version mismatch: expected {target_version}, got {store.version}"
                    )

            self._cached_sessions[session_id] = _SessionCacheEntry.create(db_file)
            self._cached_sessions.move_to_end(session_id)

            self._enforce_limits()

            return agent.AgentSessionState(
                version=store.version,
                snapshot=store.export_snapshot(),
            )

    def save(self, session_id: str, session_state: agent.AgentSessionState) -> None:
        """Update the session state in the cache."""
        db_file = self._get_db_file(session_id)

        updated = False
        which_oneof = session_state.WhichOneof("data")
        if which_oneof == "snapshot":
            if isinstance(db_file, bytes):
                db_file = session_state.snapshot
            else:
                db_file.write_bytes(session_state.snapshot)
            updated = True

        elif which_oneof == "delta":
            if not isinstance(db_file, bytes) and not db_file.exists():
                raise ValueError(f"Session {session_id} not found in cache")

            with SessionStore(db_file) as store:
                store.apply_changeset(session_state.delta, version=session_state.version)
                if isinstance(db_file, bytes):
                    db_file = store.export_snapshot()
                updated = True

        if updated:
            self._cached_sessions[session_id] = _SessionCacheEntry.create(db_file)
            self._cached_sessions.move_to_end(session_id)
            self._enforce_limits()

    def _get_db_file(self, session_id: str) -> Path | bytes:
        if not self._in_memory:
            assert self._cache_dir is not None
            return self._cache_dir / f"{session_id}.db"

        return (
            self._cached_sessions[session_id].data if session_id in self._cached_sessions else b""
        )

    def _enforce_limits(self) -> None:
        """Evict oldest files if cache exceeds size or file count limits."""
        total_size = sum(entry.size for entry in self._cached_sessions.values())
        while len(self._cached_sessions) > self._max_files or total_size > self._max_size_bytes:
            try:
                session_id, entry = self._cached_sessions.popitem(last=False)

                db_path = self._get_db_file(session_id)
                if isinstance(db_path, Path):
                    db_path.unlink(missing_ok=True)

                total_size = max(0, total_size - entry.size)
            except Exception as e:
                logger.warning("failed to evict cache file", exc_info=e)
                break

    def close(self) -> None:
        """Close the cache and clean up resources."""
        if self._temp_dir:
            self._temp_dir.cleanup()
        elif not self._in_memory:
            for session_id in self._cached_sessions.keys():
                db_path = self._get_db_file(session_id)
                if isinstance(db_path, Path):
                    db_path.unlink(missing_ok=True)
        self._cached_sessions.clear()
