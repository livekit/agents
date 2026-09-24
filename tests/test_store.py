"""The session store on a local SQLite file; ``StoreSuite`` runs again over agent-db's wire."""

from __future__ import annotations

import asyncio
import pathlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from livekit.agents import store
from livekit.agents.llm import ChatMessage, FunctionCall
from livekit.agents.store.executor import Executor, SQLiteExecutor, Value
from livekit.agents.store.schema import SCHEMA_VERSION, migrate
from livekit.agents.store.session import AgentRecord, PersistedSession, _Database, _Store

pytestmark = pytest.mark.unit

LEASE_TTL = 0.5


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
        assert {row["name"] for row in tables} == {
            "_meta",
            "_lease_check",
            "sessions",
            "chat_items",
            "agents",
        }
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

    async def test_append_is_idempotent_on_item_id(self, database: Database) -> None:
        persisted = database.session("s1")
        assert await persisted.load() is None
        item = ChatMessage(role="user", content=["first"])
        persisted.append(item)
        persisted.append(item.model_copy(update={"content": ["first, corrected"]}))
        persisted.append(item, owner="agent_1")
        await persisted.flush()

        rows = await database.rows(
            "SELECT owner, item_id, item FROM chat_items WHERE session_id = 's1' ORDER BY owner"
        )
        assert [(row["owner"], row["item_id"]) for row in rows] == [
            ("agent_1", item.id),
            ("session", item.id),
        ]
        assert "first, corrected" in rows[1]["item"]

    async def test_a_lost_write_is_redone_at_the_next_sync(
        self, database: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        persisted = database.session("s1")
        await persisted.load()
        kept, dropped = (
            ChatMessage(role="user", content=["a"]),
            ChatMessage(role="user", content=["b"]),
        )
        persisted.sync([kept, dropped], owner="agent_1", prune=True)
        await persisted.flush()

        executor = await database.executor()
        batch = executor.batch

        async def failing(*statements: object) -> None:
            raise store.StoreError("unavailable", "the database is moving")

        monkeypatch.setattr(executor, "batch", failing)
        later = ChatMessage(role="user", content=["c"])
        persisted.sync([kept, later], owner="agent_1", prune=True)
        await persisted.flush()
        monkeypatch.setattr(executor, "batch", batch)

        # the next sync rewrites what the lost batch held, the delete included
        persisted.sync([kept, later], owner="agent_1", prune=True)
        await persisted.flush()
        rows = await database.rows("SELECT item_id FROM chat_items ORDER BY created_at")
        assert [row["item_id"] for row in rows] == [kept.id, later.id]

    async def test_checkpoint_rewrites_the_mutable_rows_only(self, database: Database) -> None:
        persisted = database.session("s1", endpoint="fare-desk")
        await persisted.load()
        persisted.append(ChatMessage(role="user", content=["hello"]))
        await persisted.checkpoint(
            current_agent_id="agent_1",
            userdata={"step": 1},
            agents=[AgentRecord(agent_id="agent_1", cls="app:FareDesk", state={})],
        )
        executor = await database.executor()
        await executor.exec(
            "UPDATE agents SET durable_state = ? WHERE agent_id = 'agent_1'", b"frame"
        )
        await persisted.checkpoint(
            current_agent_id="agent_2",
            userdata={"step": 2},
            agents=[
                AgentRecord(agent_id="agent_1", cls="app:FareDesk", state={"n": 1}),
                AgentRecord(
                    agent_id="agent_2", cls="app:Rebook", parent_agent_id="agent_1", state={}
                ),
            ],
        )

        (session,) = await database.rows("SELECT * FROM sessions")
        assert session["current_agent_id"] == "agent_2"
        assert session["userdata"] == '{"step": 2}'
        assert session["endpoint"] == "fare-desk" and session["parent_session_id"] is None
        agents = await database.rows(
            "SELECT agent_id, parent_agent_id, state, durable_state FROM agents"
        )
        assert sorted((a["agent_id"], a["parent_agent_id"], a["state"]) for a in agents) == [
            ("agent_1", None, '{"n": 1}'),
            ("agent_2", "agent_1", "{}"),
        ]
        # the checkpoint leaves what it does not own alone: durable frames and the history
        assert {a["agent_id"]: a["durable_state"] for a in agents}["agent_1"] == b"frame"
        assert len(await database.rows("SELECT * FROM chat_items")) == 1

    async def test_rehydrate_returns_what_was_written(self, database: Database) -> None:
        persisted = database.session("s1")
        await persisted.load()
        greeting = ChatMessage(role="assistant", content=["hi, how can I help?"])
        question = ChatMessage(role="user", content=["move my flight"])
        call = FunctionCall(
            call_id="call_1", name="lookup", arguments="{}", extra={"lk.task_id": "task-1"}
        )
        for item in (greeting, question, call):
            persisted.append(item)
        persisted.append(question, owner="agent_1")
        userdata = {"airline": "Northwind", "bookings": [{"reference": "NW812"}]}
        await persisted.checkpoint(
            current_agent_id="agent_1",
            userdata=userdata,
            agents=[AgentRecord(agent_id="agent_1", cls="app:FareDesk", state={"tier": "gold"})],
        )
        await persisted.release()

        stored = await database.session("s1").load()
        assert stored is not None
        assert [item.id for item in stored.history] == [greeting.id, question.id, call.id]
        assert stored.history[2] == call
        assert stored.current_agent_id == "agent_1"
        assert stored.userdata == userdata
        agent = stored.agents["agent_1"]
        assert (agent.cls, agent.state) == ("app:FareDesk", {"tier": "gold"})
        assert [item.id for item in agent.chat_items] == [question.id]
        # the delegate call names its expert task, so a dashboard joins through chat_items
        rows = await database.rows(
            "SELECT json_extract(item, '$.extra.\"lk.task_id\"') AS task_id FROM chat_items "
            "WHERE item_id = ?",
            call.id,
        )
        assert rows == [{"task_id": "task-1"}]

    async def test_lease_fences_a_stale_owner(self, database: Database) -> None:
        first = database.session("s1")
        await first.load()
        await first.checkpoint(current_agent_id="a", userdata=None, agents=[])

        # the first owner stops renewing, so the second waits out its lease and takes over
        second = database.session("s1")
        loop = asyncio.get_running_loop()
        started = loop.time()
        assert await second.load() is not None
        assert loop.time() - started >= LEASE_TTL / 2

        with pytest.raises(store.LeaseLostError):
            await first.checkpoint(
                current_agent_id="stale",
                userdata=None,
                agents=[AgentRecord(agent_id="stale", cls="app:Stale")],
            )
        (session,) = await database.rows("SELECT current_agent_id, lease_owner FROM sessions")
        assert session == {"current_agent_id": "a", "lease_owner": second._lease_owner}
        assert await database.rows("SELECT * FROM agents WHERE agent_id = 'stale'") == []

        await second.checkpoint(current_agent_id="b", userdata=None, agents=[])
        await second.release()
        # a released session is taken at once
        third = database.session("s1")
        started = loop.time()
        await third.load()
        assert loop.time() - started < LEASE_TTL / 2

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


class TestLocalStore(StoreSuite):
    @pytest.fixture
    async def database(self, tmp_path: pathlib.Path) -> AsyncIterator[Database]:
        local = store.LocalStore(tmp_path, lease_ttl=LEASE_TTL)
        yield Database(local, await local.create_database())
        await local.aclose()


async def test_a_local_database_reopens_by_id(tmp_path: pathlib.Path) -> None:
    local = store.LocalStore(tmp_path)
    database_id = await local.create_database()
    persisted = local.session(database_id, "s1")
    await persisted.load()
    persisted.append(ChatMessage(role="user", content=["hi"]))
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
