"""The session store on a local SQLite file: schema, append, checkpoint, rehydrate, lease.

``StoreSuite`` is written against a ``conversation`` fixture only, so the agent-db suite runs
the same tests over the wire.
"""

from __future__ import annotations

import asyncio
import pathlib
import pickle
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest

from livekit.agents import store
from livekit.agents.llm import ChatMessage, FunctionCall
from livekit.agents.store.executor import SQLiteExecutor, Value
from livekit.agents.store.schema import SCHEMA_VERSION, migrate
from livekit.agents.store.session_state import AgentRecord

pytestmark = pytest.mark.unit

LEASE_TTL = 0.5


@dataclass
class Booking:
    reference: str
    seats: list[str] = field(default_factory=list)


@dataclass
class Userdata:
    airline: str
    bookings: list[Booking] = field(default_factory=list)


@dataclass
class Keyed:
    seats: dict[tuple[str, str], str]


async def _rows(conversation: store.Conversation, sql: str, *params: Value) -> list[dict]:
    executor = await conversation.open()
    return [row async for row in executor.query(sql, *params)]


class StoreSuite:
    async def test_schema_is_created_and_gated(self, conversation: store.Conversation) -> None:
        tables = await _rows(
            conversation, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
        names = {row["name"] for row in tables}
        assert {"_meta", "sessions", "chat_items", "agents", "tasks", "delegations"} <= names
        version = await _rows(conversation, "SELECT value FROM _meta WHERE key = 'schema_version'")
        assert version == [{"value": str(SCHEMA_VERSION)}]

        # migrating again is a no-op, and a database from a newer framework is refused
        assert await migrate(conversation.executor) == SCHEMA_VERSION
        await conversation.executor.exec(
            "UPDATE _meta SET value = ? WHERE key = 'schema_version'",
            str(SCHEMA_VERSION + 1),
        )
        with pytest.raises(store.SchemaVersionError):
            await migrate(conversation.executor)

    async def test_append_is_idempotent_on_item_id(self, conversation: store.Conversation) -> None:
        state = conversation.session("s1")
        assert await state.load() is None
        item = ChatMessage(role="user", content=["first"])
        state.append(item)
        state.append(item.model_copy(update={"content": ["first, corrected"]}))
        state.append(item, owner="agent_1")
        await state.flush()

        rows = await _rows(
            conversation,
            "SELECT owner, item_id, item_json FROM chat_items WHERE session_id = 's1' "
            "ORDER BY owner",
        )
        assert [(row["owner"], row["item_id"]) for row in rows] == [
            ("agent_1", item.id),
            ("session", item.id),
        ]
        assert "first, corrected" in rows[1]["item_json"]

    async def test_checkpoint_rewrites_the_mutable_rows_only(
        self, conversation: store.Conversation
    ) -> None:
        state = conversation.session("s1", kind="a2a", endpoint="fare-desk")
        await state.load()
        state.append(ChatMessage(role="user", content=["hello"]))
        await state.checkpoint(
            current_agent_id="agent_1",
            userdata={"step": 1},
            agents=[AgentRecord(agent_id="agent_1", cls="app:FareDesk", state={})],
        )
        await conversation.executor.exec(
            "UPDATE agents SET durable_state = ? WHERE agent_id = 'agent_1'", b"frame"
        )
        await state.checkpoint(
            current_agent_id="agent_2",
            userdata={"step": 2},
            agents=[
                AgentRecord(agent_id="agent_1", cls="app:FareDesk", state={"n": 1}),
                AgentRecord(
                    agent_id="agent_2", cls="app:Rebook", parent_agent_id="agent_1", state={}
                ),
            ],
        )

        (session,) = await _rows(conversation, "SELECT * FROM sessions")
        assert session["current_agent_id"] == "agent_2"
        assert session["userdata"] == '{"step": 2}'
        assert session["userdata_encoding"] == "json"
        assert session["kind"] == "a2a" and session["endpoint"] == "fare-desk"
        agents = await _rows(
            conversation, "SELECT agent_id, parent_agent_id, state_json, durable_state FROM agents"
        )
        assert sorted((a["agent_id"], a["parent_agent_id"], a["state_json"]) for a in agents) == [
            ("agent_1", None, '{"n": 1}'),
            ("agent_2", "agent_1", "{}"),
        ]
        # the checkpoint leaves what it does not own alone: durable frames and the history
        assert {a["agent_id"]: a["durable_state"] for a in agents}["agent_1"] == b"frame"
        assert len(await _rows(conversation, "SELECT * FROM chat_items")) == 1

    async def test_rehydrate_returns_what_was_written(
        self, conversation: store.Conversation
    ) -> None:
        state = conversation.session("s1", kind="voice")
        await state.load()
        greeting = ChatMessage(role="assistant", content=["hi, how can I help?"])
        question = ChatMessage(role="user", content=["move my flight"])
        call = FunctionCall(call_id="call_1", name="lookup", arguments="{}")
        for item in (greeting, question, call):
            state.append(item)
        state.append(question, owner="agent_1")
        userdata = Userdata(airline="Northwind", bookings=[Booking("NW812", ["12A"])])
        await state.checkpoint(
            current_agent_id="agent_1",
            userdata=userdata,
            agents=[
                AgentRecord(
                    agent_id="agent_1", cls="app:FareDesk", state={"tier": "gold"}, tools=["x"]
                )
            ],
            tools=["lookup"],
        )
        await state.release()

        stored = await conversation.session("s1", kind="voice").load()
        assert stored is not None
        assert [item.id for item in stored.history] == [greeting.id, question.id, call.id]
        assert stored.history[2] == call
        assert stored.current_agent_id == "agent_1"
        assert stored.userdata == userdata
        agent = stored.agents["agent_1"]
        assert (agent.cls, agent.state) == ("app:FareDesk", {"tier": "gold"})
        # tool ids are written for a dashboard; a resumed session takes its tools from code
        rows = await _rows(
            conversation, "SELECT tools_json FROM sessions UNION ALL SELECT tools_json FROM agents"
        )
        assert [r["tools_json"] for r in rows] == ['["lookup"]', '["x"]']
        assert [item.id for item in agent.chat_items] == [question.id]
        assert stored.interrupted == []

    async def test_userdata_that_json_cannot_restore_is_pickled(
        self, conversation: store.Conversation
    ) -> None:
        state = conversation.session("s1")
        await state.load()
        # JSON writes the tuple keys as strings and cannot read them back as tuples
        userdata = Keyed(seats={("NW812", "12A"): "held"})
        await state.checkpoint(current_agent_id=None, userdata=userdata, agents=[])
        await state.release()

        (row,) = await _rows(conversation, "SELECT userdata_encoding FROM sessions")
        assert row == {"userdata_encoding": "pickle"}
        stored = await conversation.session("s1").load()
        assert stored is not None
        assert pickle.loads(stored.userdata) == userdata

    async def test_lease_fences_a_stale_owner(self, conversation: store.Conversation) -> None:
        first = conversation.session("s1")
        await first.load()
        await first.checkpoint(current_agent_id="a", userdata=None, agents=[])

        # the first owner stops renewing, so the second waits out its lease and takes over
        second = conversation.session("s1")
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
        (session,) = await _rows(conversation, "SELECT current_agent_id, lease_owner FROM sessions")
        assert session == {"current_agent_id": "a", "lease_owner": second._lease_owner}
        assert await _rows(conversation, "SELECT * FROM agents WHERE agent_id = 'stale'") == []

        await second.checkpoint(current_agent_id="b", userdata=None, agents=[])
        await second.release()
        # a released session is taken at once
        third = conversation.session("s1")
        started = loop.time()
        await third.load()
        assert loop.time() - started < LEASE_TTL / 2

    async def test_running_tasks_are_reported_on_load(
        self, conversation: store.Conversation
    ) -> None:
        state = conversation.session("s1")
        await state.load()
        await state.task_started("call_done", name="lookup", arguments='{"q": 1}')
        state.task_ended("call_done", status="done", output="found", is_error=False)
        await state.task_started("call_hung", name="rebook", arguments='{"flight": "NW812"}')
        await state.release()

        resumed = conversation.session("s1")
        stored = await resumed.load()
        assert stored is not None
        (task,) = stored.interrupted
        assert (task.call_id, task.name, task.arguments) == (
            "call_hung",
            "rebook",
            '{"flight": "NW812"}',
        )
        # the row stays running until the new owner has told the model, then settles it
        (row,) = await _rows(conversation, "SELECT status FROM tasks WHERE call_id = 'call_hung'")
        assert row["status"] == "running"
        resumed.task_ended("call_hung", status="interrupted", output="unknown", is_error=True)
        await resumed.flush()
        rows = await _rows(
            conversation, "SELECT call_id, status, idempotency_key, origin FROM tasks ORDER BY 1"
        )
        assert rows == [
            {
                "call_id": "call_done",
                "status": "done",
                "idempotency_key": "call_done",
                "origin": "llm",
            },
            {
                "call_id": "call_hung",
                "status": "interrupted",
                "idempotency_key": "call_hung",
                "origin": "llm",
            },
        ]

    async def test_delegation_link_round_trips(self, conversation: store.Conversation) -> None:
        caller = conversation.session("voice", kind="voice")
        await caller.load()
        assert await caller.child_session("fare-desk") is None
        caller.delegation_started(
            "call_1", endpoint="fare-desk", child_session_id="ctx-1", task_id="task-1"
        )
        caller.delegation_ended("call_1", status="completed")
        caller.delegation_started(
            "call_2", endpoint="baggage", child_session_id="ctx-2", task_id="task-2"
        )
        expert = conversation.session("ctx-1", kind="a2a", parent="voice", endpoint="fare-desk")
        await expert.load()
        await caller.release()

        resumed = conversation.session("voice", kind="voice")
        await resumed.load()
        assert await resumed.child_session("fare-desk") == "ctx-1"
        assert await resumed.child_session("baggage") == "ctx-2"
        (link,) = await _rows(
            conversation,
            "SELECT child_session_id, task_id, status, ended_at FROM delegations "
            "WHERE call_id = 'call_1'",
        )
        assert (link["child_session_id"], link["task_id"], link["status"]) == (
            "ctx-1",
            "task-1",
            "completed",
        )
        assert link["ended_at"] is not None
        tree = await _rows(
            conversation, "SELECT session_id, parent_session_id FROM sessions ORDER BY 1"
        )
        assert tree == [
            {"session_id": "ctx-1", "parent_session_id": "voice"},
            {"session_id": "voice", "parent_session_id": None},
        ]

    async def test_the_connection_closes_with_its_last_session(
        self, conversation: store.Conversation
    ) -> None:
        first, second = conversation.session("a"), conversation.session("b")
        await first.load()
        await second.load()
        await first.release()
        await first.release()  # a second release is a no-op, not a second let-go
        assert conversation.executor is not None
        await second.release()
        with pytest.raises(store.StoreError):
            _ = conversation.executor

        # the next session opens it again
        again = conversation.session("a")
        assert await again.load() is not None
        assert conversation.executor is not None
        await again.release()


class TestSQLiteStore(StoreSuite):
    @pytest.fixture
    async def conversation(self, tmp_path: pathlib.Path) -> AsyncIterator[store.Conversation]:
        sqlite = store.SQLite(tmp_path, lease_ttl=LEASE_TTL)
        conversation = await sqlite.create_conversation()
        yield conversation
        await sqlite.aclose()


async def test_sqlite_reopens_a_conversation_by_id(tmp_path: pathlib.Path) -> None:
    sqlite = store.SQLite(tmp_path)
    created = await sqlite.create_conversation()
    state = created.session("s1")
    await state.load()
    state.append(ChatMessage(role="user", content=["hi"]))
    await state.release()
    await sqlite.aclose()

    reopened = await store.SQLite(tmp_path).conversation(created.database_id)
    stored = await reopened.session("s1").load()
    assert stored is not None and [m.text_content for m in stored.history] == ["hi"]  # type: ignore[union-attr]
    await reopened.aclose()
    with pytest.raises(store.StoreError):
        await store.SQLite(tmp_path).conversation("DB_missing")


async def test_memory_executor_batch_is_atomic() -> None:
    executor = SQLiteExecutor()
    await executor.exec("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    with pytest.raises(store.StoreError):
        await executor.batch(("INSERT INTO t VALUES (1)", ()), ("INSERT INTO nope VALUES (1)", ()))
    assert [row async for row in executor.query("SELECT * FROM t")] == []
    await executor.aclose()
