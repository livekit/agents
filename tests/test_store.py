"""The session store on a local SQLite file; ``StoreSuite`` runs again over agent-db's wire."""

from __future__ import annotations

import pathlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import aiohttp
import pytest

from livekit.agents import store
from livekit.agents.llm import ChatItem, ChatMessage, FunctionCall
from livekit.agents.store.executor import ExecResult, Executor, SQLiteExecutor, Statement, Value
from livekit.agents.store.schema import SCHEMA_VERSION, migrate
from livekit.agents.store.session import AgentRecord, PersistedSession, _Database, _Store

pytestmark = pytest.mark.unit


@dataclass
class Database:
    """One database of a store, as the suite drives it."""

    store: _Store
    database_id: str

    def session(self, session_id: str, **kwargs: Any) -> PersistedSession:
        return self.store.session(self.database_id, session_id, **kwargs)

    async def executor(self) -> Executor:
        databases = self.store._databases
        if (database := databases.get(self.database_id)) is None:
            database = databases[self.database_id] = _Database(
                self.database_id, connect=lambda: self.store._connect(self.database_id)
            )
        return await database.open()

    async def rows(self, sql: str, *params: Value) -> list[dict]:
        executor = await self.executor()
        return [row async for row in executor.query(sql, *params)]


class StoreSuite:
    async def test_schema_is_created_and_gated(self, database: Database) -> None:
        tables = await database.rows(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
        assert {row["name"] for row in tables} == {"_meta", "sessions", "chat_items", "agents"}
        version = await database.rows("SELECT value FROM _meta WHERE key = 'schema_version'")
        assert version == [{"value": str(SCHEMA_VERSION)}]

        # migrating again is a no-op, and a database from a newer framework is refused
        executor = await database.executor()
        assert await migrate(executor) == SCHEMA_VERSION
        await executor.exec(
            "UPDATE _meta SET value = ? WHERE key = 'schema_version'", str(SCHEMA_VERSION + 1)
        )
        with pytest.raises(store.SchemaVersionError):
            await migrate(executor)

    async def test_a_save_writes_only_what_changed(
        self, database: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        persisted = database.session("s1")
        assert await persisted.load() is None
        kept = ChatMessage(role="user", content=["first"])
        dropped = ChatMessage(role="user", content=["second"])
        await _save(persisted, [kept, dropped], agent_items=[kept])

        executor = await database.executor()
        batches: list[tuple[Statement, ...]] = []
        batch = executor.batch

        async def recording(*statements: Statement) -> list[ExecResult]:
            batches.append(statements)
            return await batch(*statements)

        monkeypatch.setattr(executor, "batch", recording)
        corrected = kept.model_copy(update={"content": ["first, corrected"]})
        later = ChatMessage(role="assistant", content=["third"])
        await _save(persisted, [corrected, later], agent_items=[kept])
        await _save(persisted, [corrected, later], agent_items=[kept])

        chat_writes = [
            [(sql.split()[0], params[2]) for sql, params in statements if "chat_items" in sql]
            for statements in batches
        ]
        # the first save deletes, rewrites and adds one item each; the second has nothing to add
        assert sorted(chat_writes[0]) == sorted(
            [("DELETE", dropped.id), ("INSERT", kept.id), ("INSERT", later.id)]
        )
        assert chat_writes[1] == []
        rows = await database.rows(
            "SELECT owner, item_id, item FROM chat_items ORDER BY owner, created_at"
        )
        assert [(row["owner"], row["item_id"]) for row in rows] == [
            ("agent_1", kept.id),
            ("session", kept.id),
            ("session", later.id),
        ]
        assert "first, corrected" in rows[1]["item"]

    async def test_a_failed_save_is_written_again_by_the_next(
        self, database: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        persisted = database.session("s1")
        await persisted.load()
        first = ChatMessage(role="user", content=["a"])
        await _save(persisted, [first])

        executor = await database.executor()
        batch = executor.batch

        async def failing(*statements: object) -> None:
            raise store.StoreError("unavailable", "the database is moving")

        monkeypatch.setattr(executor, "batch", failing)
        second = ChatMessage(role="user", content=["b"])
        with pytest.raises(store.StoreError):
            await _save(persisted, [second])
        monkeypatch.setattr(executor, "batch", batch)

        await _save(persisted, [second])
        rows = await database.rows("SELECT item_id FROM chat_items")
        assert [row["item_id"] for row in rows] == [second.id]

    async def test_a_save_rewrites_the_mutable_rows(self, database: Database) -> None:
        persisted = database.session("s1", endpoint="fare-desk")
        await persisted.load()
        await persisted.save(
            current_agent_id="agent_1",
            userdata={"step": 1},
            history=[],
            agents=[AgentRecord(agent_id="agent_1", cls="app:FareDesk", state={})],
        )
        await persisted.save(
            current_agent_id="agent_2",
            userdata=None,
            history=[],
            agents=[
                AgentRecord(
                    agent_id="agent_1", cls="app:FareDesk", state={"n": 1}, durable_state=b"frame"
                ),
                AgentRecord(
                    agent_id="agent_2", cls="app:Rebook", parent_agent_id="agent_1", state={}
                ),
            ],
        )

        (session,) = await database.rows("SELECT * FROM sessions")
        assert session["current_agent_id"] == "agent_2"
        # none leaves the stored userdata as is
        assert session["userdata"] == '{"step": 1}'
        assert session["endpoint"] == "fare-desk" and session["parent_session_id"] is None
        agents = await database.rows(
            "SELECT agent_id, parent_agent_id, state, durable_state FROM agents ORDER BY agent_id"
        )
        assert [tuple(agent.values()) for agent in agents] == [
            ("agent_1", None, '{"n": 1}', b"frame"),
            ("agent_2", "agent_1", "{}", b""),
        ]

    async def test_a_load_returns_what_was_saved(self, database: Database) -> None:
        persisted = database.session("s1")
        await persisted.load()
        greeting = ChatMessage(role="assistant", content=["hi, how can I help?"])
        question = ChatMessage(role="user", content=["move my flight"])
        call = FunctionCall(
            call_id="call_1", name="lookup", arguments="{}", extra={"lk.task_id": "task-1"}
        )
        userdata = {"airline": "Northwind", "bookings": [{"reference": "NW812"}]}
        await persisted.save(
            current_agent_id="agent_1",
            userdata=userdata,
            history=[greeting, question, call],
            agents=[
                AgentRecord(
                    agent_id="agent_1",
                    cls="app:FareDesk",
                    state={"tier": "gold"},
                    chat_items=[question],
                )
            ],
        )
        await persisted.release()
        (row,) = await database.rows("SELECT closed_at FROM sessions")
        assert row["closed_at"] is not None

        again = database.session("s1")
        stored = await again.load()
        assert stored is not None
        assert [item.id for item in stored.history] == [greeting.id, question.id, call.id]
        assert stored.history[2] == call
        assert stored.current_agent_id == "agent_1"
        assert stored.userdata == userdata
        agent = stored.agents["agent_1"]
        assert (agent.cls, agent.state) == ("app:FareDesk", {"tier": "gold"})
        assert [item.id for item in agent.chat_items] == [question.id]
        (row,) = await database.rows("SELECT closed_at FROM sessions")
        assert row["closed_at"] is None
        # the delegate call names its expert task, so a dashboard joins through chat_items
        rows = await database.rows(
            "SELECT json_extract(item, '$.extra.\"lk.task_id\"') AS task_id FROM chat_items "
            "WHERE item_id = ?",
            call.id,
        )
        assert rows == [{"task_id": "task-1"}]
        await again.release()

    async def test_children_are_found_by_parent_and_endpoint(self, database: Database) -> None:
        caller = database.session("voice")
        await caller.load()
        for child, endpoint in (("ctx-1", "fare-desk"), ("ctx-2", "baggage")):
            expert = database.session(child, parent="voice", endpoint=endpoint)
            await expert.load()
            await expert.release()
        await caller.release()

        stored = await database.session("voice").load()
        assert stored is not None
        assert stored.children == {"fare-desk": "ctx-1", "baggage": "ctx-2"}
        tree = await database.rows(
            "SELECT session_id, parent_session_id, endpoint FROM sessions ORDER BY 1"
        )
        assert tree == [
            {"session_id": "ctx-1", "parent_session_id": "voice", "endpoint": "fare-desk"},
            {"session_id": "ctx-2", "parent_session_id": "voice", "endpoint": "baggage"},
            {"session_id": "voice", "parent_session_id": None, "endpoint": None},
        ]

    async def test_the_connection_closes_with_its_last_session(self, database: Database) -> None:
        first, second = database.session("a"), database.session("b")
        await first.load()
        await second.load()
        await first.release()
        await first.release()  # a second release is a no-op, not a second let-go
        connection = database.store._databases[database.database_id]
        assert connection.executor is not None
        await second.release()
        with pytest.raises(store.StoreError):
            _ = connection.executor

        # the next session opens it again
        again = database.session("a")
        assert await again.load() is not None
        assert connection.executor is not None
        await again.release()


async def _save(
    persisted: PersistedSession, history: list[ChatItem], agent_items: Sequence[ChatItem] = ()
) -> None:
    await persisted.save(
        current_agent_id="agent_1",
        userdata=None,
        history=history,
        agents=[AgentRecord(agent_id="agent_1", cls="app:FareDesk", chat_items=agent_items)],
    )


class TestLocalStore(StoreSuite):
    @pytest.fixture
    async def database(self, tmp_path: pathlib.Path) -> AsyncIterator[Database]:
        local = store.LocalStore(tmp_path)
        yield Database(local, await local.create_database())
        await local.aclose()


async def test_a_local_database_reopens_by_id(tmp_path: pathlib.Path) -> None:
    local = store.LocalStore(tmp_path)
    database_id = await local.create_database()
    persisted = local.session(database_id, "s1")
    await persisted.load()
    await _save(persisted, [ChatMessage(role="user", content=["hi"])])
    await persisted.release()
    await local.aclose()

    reopened = store.LocalStore(tmp_path)
    stored = await reopened.session(database_id, "s1").load()
    assert stored is not None and [m.text_content for m in stored.history] == ["hi"]  # type: ignore[union-attr]
    await reopened.aclose()
    with pytest.raises(store.StoreError):
        await store.LocalStore(tmp_path).session("DB_missing", "s1").load()


async def test_memory_executor_batch_is_atomic() -> None:
    executor = SQLiteExecutor()
    await executor.exec("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    with pytest.raises(store.StoreError):
        await executor.batch(("INSERT INTO t VALUES (1)", ()), ("INSERT INTO nope VALUES (1)", ()))
    assert [row async for row in executor.query("SELECT * FROM t")] == []
    await executor.aclose()


async def test_an_agentdb_that_cannot_connect_leaves_no_http_session_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from livekit.agents.store import agentdb

    opened: list[aiohttp.ClientSession] = []
    client_session = aiohttp.ClientSession

    def recording(*args: Any, **kwargs: Any) -> aiohttp.ClientSession:
        opened.append(session := client_session(*args, **kwargs))
        return session

    monkeypatch.setattr(agentdb.aiohttp, "ClientSession", recording)
    # nothing listens on port 1, so the dial fails on the spot
    db = store.AgentDB(url="http://127.0.0.1:1", api_key="key", api_secret="secret")
    with pytest.raises(aiohttp.ClientError):
        await db.session("DB_x", "s1").load()
    assert opened and all(session.closed for session in opened)
    await db.aclose()
