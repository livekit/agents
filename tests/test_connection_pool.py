import time

import pytest

from livekit.agents.utils import ConnectionPool, ConnectionResult


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
async def test_get_with_timing_new_connection():
    """Test get_with_timing returns correct timing info for new connection."""
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    result = await pool.get_with_timing(timeout=10.0)
    assert isinstance(result, ConnectionResult)
    assert result.connection is not None
    assert result.connect_time >= 0
    assert result.from_pool is False, "Expected from_pool to be False for new connection"


@pytest.mark.asyncio
async def test_get_with_timing_reused_connection():
    """Test get_with_timing returns correct timing info for reused connection."""
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    # Create a connection and return it to the pool
    conn1 = await pool.get(timeout=10.0)
    pool.put(conn1)

    # Get with timing - should reuse the connection
    result = await pool.get_with_timing(timeout=10.0)
    assert isinstance(result, ConnectionResult)
    assert result.connection is conn1, "Expected the same connection to be reused"
    assert result.connect_time >= 0
    assert result.from_pool is True, "Expected from_pool to be True for reused connection"


@pytest.mark.asyncio
async def test_connection_with_timing_context_manager():
    """Test connection_with_timing context manager."""
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    async with pool.connection_with_timing(timeout=10.0) as result:
        assert isinstance(result, ConnectionResult)
        assert result.connection is not None
        assert result.connect_time >= 0
        assert result.from_pool is False

    # Connection should be returned to pool
    async with pool.connection_with_timing(timeout=10.0) as result2:
        assert result2.connection is result.connection
        assert result2.from_pool is True
