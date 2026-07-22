import time

import pytest

from livekit.agents.utils import ConnectionPool

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]


class DummyConnection:
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"DummyConnection({self.id})"


def dummy_connect_factory():
    counter = 0

    async def dummy_connect(timeout: float):
        nonlocal counter
        counter += 1
        return DummyConnection(counter)

    return dummy_connect


@pytest.mark.asyncio
async def test_get_reuses_connection():
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn1 = await pool.get(timeout=10.0)
    # Return the connection to the pool
    pool.put(conn1)

    async with pool.connection(timeout=10.0) as conn:
        assert conn is conn1, "Expected conn to be the same connection as conn1"

    conn2 = await pool.get(timeout=10.0)
    assert conn1 is conn2, "Expected the same connection to be reused when it hasn't expired."


@pytest.mark.asyncio
async def test_get_creates_new_connection_when_none_available():
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn1 = await pool.get(timeout=10.0)
    # Not putting conn1 back means the available pool is empty,
    # so calling get() again should create a new connection.
    conn2 = await pool.get(timeout=10.0)
    assert conn1 is not conn2, "Expected a new connection when no available connection exists."


@pytest.mark.asyncio
async def test_remove_connection():
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn = await pool.get(timeout=10.0)
    pool.put(conn)
    # Reset the connection which should remove it from the pool.
    pool.remove(conn)

    # Even if we try to put it back, it won't be added because it's not tracked anymore.
    pool.put(conn)
    new_conn = await pool.get(timeout=10.0)
    assert new_conn is not conn, "Expected a removed connection to not be reused."


@pytest.mark.asyncio
async def test_get_expired():
    # Use a short max duration to simulate expiration.
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=1, connect_cb=dummy_connect)

    conn = await pool.get(timeout=10.0)
    pool.put(conn)
    # Artificially set the connection's timestamp in the past to simulate expiration.
    pool._connections[conn] = time.time() - 2  # 2 seconds ago (max_session_duration is 1)

    conn2 = await pool.get(timeout=10.0)
    assert conn2 is not conn, "Expected a new connection to be returned."


@pytest.mark.asyncio
async def test_get_discards_invalid_connection():
    # Regression for livekit/agents#6513: a pooled websocket that was closed
    # while idle must be discarded by get() instead of being handed back to the
    # caller (which would burn an outer LLM retry on send).
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(
        max_session_duration=60,
        connect_cb=dummy_connect,
        validate_cb=lambda c: not c.closed,
    )

    stale1 = DummyConnection("stale-1")
    stale1.closed = True
    stale2 = DummyConnection("stale-2")
    stale2.closed = True
    healthy = DummyConnection("healthy")
    healthy.closed = False

    for conn in (stale1, stale2, healthy):
        pool._connections[conn] = time.time()
        pool._available.add(conn)

    acquired = await pool.get(timeout=10.0)
    assert acquired is healthy, (
        "Expected get() to skip stale connections and return the healthy one."
    )
    assert acquired.closed is False
    assert pool.last_connection_reused is True


@pytest.mark.asyncio
async def test_get_creates_new_when_all_invalid():
    # When every pooled connection fails validation, get() must fall back to
    # creating a fresh connection rather than returning a stale one.
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(
        max_session_duration=60,
        connect_cb=dummy_connect,
        validate_cb=lambda c: not c.closed,
    )

    stale = DummyConnection("stale")
    stale.closed = True
    pool._connections[stale] = time.time()
    pool._available.add(stale)

    acquired = await pool.get(timeout=10.0)
    assert acquired is not stale, (
        "Expected a fresh connection when all pooled connections are invalid."
    )
    assert pool.last_connection_reused is False
    assert stale not in pool._connections


@pytest.mark.asyncio
async def test_get_reuses_when_no_validate_cb():
    # Without a validate_cb the pool must keep its previous behavior and reuse
    # available connections regardless of any connection-local state.
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn = await pool.get(timeout=10.0)
    conn.closed = True  # attribute the pool must ignore when no validate_cb is set
    pool.put(conn)

    conn2 = await pool.get(timeout=10.0)
    assert conn2 is conn, "Expected reuse to be unaffected when no validate_cb is provided."
