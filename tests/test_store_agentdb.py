"""The session store over agent-db's data plane, against a running server.

Start one with ``mage devLocal`` in agent-db, then::

    LIVEKIT_AGENTDB_URL=http://localhost:7780 LIVEKIT_AGENTDB_WS_URL=ws://localhost:7781/db \\
    LIVEKIT_AGENTDB_API_KEY=devkey LIVEKIT_AGENTDB_API_SECRET=secret \\
    uv run pytest tests/test_store_agentdb.py --plugin agentdb
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest

from livekit.agents import store

from .test_store import LEASE_TTL, StoreSuite

pytestmark = [
    pytest.mark.plugin("agentdb"),
    pytest.mark.skipif(
        not os.environ.get("LIVEKIT_AGENTDB_URL"), reason="LIVEKIT_AGENTDB_URL is not set"
    ),
]


@pytest.fixture
async def agentdb() -> AsyncIterator[store.AgentDB]:
    agentdb = store.AgentDB.from_env(lease_ttl=LEASE_TTL)
    yield agentdb
    await agentdb.aclose()


class TestAgentDBStore(StoreSuite):
    @pytest.fixture
    async def conversation(self, agentdb: store.AgentDB) -> AsyncIterator[store.Conversation]:
        conversation = await agentdb.create_conversation(ttl_seconds=3600)
        yield conversation
        await agentdb.service.delete_database(conversation.database_id)


async def test_a_query_streams_many_batches(
    agentdb: store.AgentDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation = await agentdb.create_conversation(ttl_seconds=3600)
    executor = conversation.executor
    await executor.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, body TEXT, raw BLOB, x)")
    rows = 20_000
    await executor.exec(
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) "
        "INSERT INTO t SELECT i, printf('row %d', i), randomblob(i % 7), "
        "CASE i % 3 WHEN 0 THEN NULL WHEN 1 THEN i * 1.5 ELSE 'text' END FROM n",
        rows,
    )
    assert isinstance(executor, store.AgentDBExecutor)

    batches = 0
    original = store.agentdb._decode_batch

    def counting(names: list[str], batch: object) -> list[store.Row]:
        nonlocal batches
        batches += 1
        return original(names, batch)  # type: ignore[arg-type]

    monkeypatch.setattr(store.agentdb, "_decode_batch", counting)
    seen = [row async for row in executor.query("SELECT * FROM t ORDER BY id")]
    assert batches > 1
    assert len(seen) == rows
    assert seen[0] == {"id": 1, "body": "row 1", "raw": seen[0]["raw"], "x": 1.5}
    assert isinstance(seen[0]["raw"], bytes) and len(seen[0]["raw"]) == 1
    assert seen[1]["x"] == "text" and seen[2]["x"] is None
    assert seen[-1]["body"] == f"row {rows}"

    # a reader that stops early cancels its stream, and the socket stays usable
    async for _ in executor.query("SELECT * FROM t"):
        break
    assert [r async for r in executor.query("SELECT COUNT(*) AS n FROM t")] == [{"n": rows}]
    await agentdb.service.delete_database(conversation.database_id)


async def test_reconnects_after_the_socket_is_severed(agentdb: store.AgentDB) -> None:
    conversation = await agentdb.create_conversation(ttl_seconds=3600)
    executor = conversation.executor
    assert isinstance(executor, store.AgentDBExecutor)
    state = conversation.session("s1")
    await state.load()

    assert executor._ws is not None
    await executor._ws.close()
    # issued while the reconnect is in flight: it waits for the new socket rather than failing
    await state.checkpoint(current_agent_id="after", userdata=None, agents=[])
    rows = [r async for r in executor.query("SELECT current_agent_id FROM sessions")]
    assert rows == [{"current_agent_id": "after"}]
    await agentdb.service.delete_database(conversation.database_id)


async def test_a_failing_batch_applies_nothing(agentdb: store.AgentDB) -> None:
    conversation = await agentdb.create_conversation(ttl_seconds=3600)
    executor = conversation.executor
    await executor.exec("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    with pytest.raises(store.StoreError):
        await executor.batch(
            ("INSERT INTO t VALUES (1)", ()),
            ("INSERT INTO t VALUES (1)", ()),  # the primary key refuses the second
        )
    assert [r async for r in executor.query("SELECT * FROM t")] == []
    await agentdb.service.delete_database(conversation.database_id)


async def test_the_service_manages_databases(agentdb: store.AgentDB) -> None:
    created = await agentdb.service.create_database(ttl_seconds=60)
    assert created.database_id.startswith("DB_")
    fetched = await agentdb.service.get_database(created.database_id)
    assert fetched.database_id == created.database_id
    listed = await agentdb.service.list_databases(page_size=1000)
    assert created.database_id in {d.database_id for d in listed.databases}
    await agentdb.service.delete_database(created.database_id)
    with pytest.raises(Exception, match="not_found|not found"):
        await agentdb.service.get_database(created.database_id)
