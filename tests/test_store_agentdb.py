"""The store suite over agent-db's wire, against ``mage devLocal`` and ``LIVEKIT_AGENTDB_URL``."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import store
from livekit.agents.store import agentdb as agentdb_client

from .test_store import LEASE_TTL, Database, StoreSuite

pytestmark = [
    pytest.mark.plugin("agentdb"),
    pytest.mark.skipif(
        not os.environ.get("LIVEKIT_AGENTDB_URL"), reason="LIVEKIT_AGENTDB_URL is not set"
    ),
]


@pytest.fixture
async def agentdb() -> AsyncIterator[store.AgentDB]:
    # devLocal takes its own static key rather than the project's, and serves two ports
    local = "localhost" in os.environ.get("LIVEKIT_AGENTDB_URL", "")
    key = {"api_key": "devkey", "api_secret": "secret"} if local else {}
    ws_url = os.environ.get("LIVEKIT_AGENTDB_WS_URL")
    agentdb = store.AgentDB(ws_url=ws_url, lease_ttl=LEASE_TTL, **key)
    yield agentdb
    await agentdb.aclose()


@pytest.fixture
async def database(agentdb: store.AgentDB) -> AsyncIterator[Database]:
    database = Database(agentdb, await agentdb.create_database(ttl_seconds=3600))
    yield database
    await agentdb.delete_database(database.database_id)


class TestAgentDBStore(StoreSuite):
    pass


async def test_a_query_streams_many_batches(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    executor = await database.executor()
    await executor.exec("CREATE TABLE t (id INTEGER PRIMARY KEY, body TEXT, raw BLOB, x)")
    rows = 20_000
    await executor.exec(
        "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) "
        "INSERT INTO t SELECT i, printf('row %d', i), randomblob(i % 7), "
        "CASE i % 3 WHEN 0 THEN NULL WHEN 1 THEN i * 1.5 ELSE 'text' END FROM n",
        rows,
    )
    assert isinstance(executor, agentdb_client.AgentDBExecutor)

    # query() sends one credit per batch it takes, so the credits count the batches
    batches = 0
    send = executor._send

    async def counting(message: Any) -> None:
        nonlocal batches
        batches += message.WhichOneof("message") == "credit"
        await send(message)

    monkeypatch.setattr(executor, "_send", counting)
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


async def test_reconnects_after_the_socket_is_severed(database: Database) -> None:
    persisted = database.session("s1")
    await persisted.load()
    executor = await database.executor()
    assert isinstance(executor, agentdb_client.AgentDBExecutor)

    assert executor._ws is not None
    await executor._ws.close()
    # issued while the reconnect is in flight: it waits for the new socket rather than failing
    await persisted.checkpoint(current_agent_id="after", userdata=None, agents=[])
    rows = [r async for r in executor.query("SELECT current_agent_id FROM sessions")]
    assert rows == [{"current_agent_id": "after"}]


async def test_a_failing_batch_applies_nothing(database: Database) -> None:
    executor = await database.executor()
    await executor.exec("CREATE TABLE t (id INTEGER PRIMARY KEY)")
    with pytest.raises(store.StoreError):
        await executor.batch(
            ("INSERT INTO t VALUES (1)", ()),
            ("INSERT INTO t VALUES (1)", ()),  # the primary key refuses the second
        )
    assert [r async for r in executor.query("SELECT * FROM t")] == []


async def test_the_store_manages_databases(agentdb: store.AgentDB) -> None:
    database_id = await agentdb.create_database(ttl_seconds=60)
    assert database_id.startswith("DB_")
    fetched = await agentdb.get_database(database_id)
    assert fetched.database_id == database_id
    listed = await agentdb.list_databases(page_size=1000)
    assert database_id in {d.database_id for d in listed.databases}
    await agentdb.delete_database(database_id)
    with pytest.raises(Exception, match="not_found|not found"):
        await agentdb.get_database(database_id)


async def test_a_request_over_the_frame_limit_fails(database: Database) -> None:
    executor = await database.executor()
    await executor.exec("CREATE TABLE t (body BLOB)")
    with pytest.raises(store.StoreError, match="frame"):
        await asyncio.wait_for(
            executor.exec("INSERT INTO t VALUES (?)", b"x" * agentdb_client.MAX_FRAME_BYTES), 10
        )
    assert [r async for r in executor.query("SELECT COUNT(*) AS n FROM t")] == [{"n": 0}]
