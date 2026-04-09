import asyncio
import time

import pytest

from livekit.agents.utils import ConnectionPool


class DummyConnection:
    def __init__(self, id):
        self.id = id
        self.healthy = True

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
async def test_health_check_healthy_connection():
    """Health check passes — pooled connection is reused."""
    dummy_connect = dummy_connect_factory()

    async def health_check(conn: DummyConnection) -> bool:
        return conn.healthy

    pool = ConnectionPool(
        max_session_duration=60,
        connect_cb=dummy_connect,
        health_check_cb=health_check,
    )

    conn1 = await pool.get(timeout=10.0)
    pool.put(conn1)

    conn2 = await pool.get(timeout=10.0)
    assert conn1 is conn2, "Healthy connection should be reused."


@pytest.mark.asyncio
async def test_health_check_unhealthy_connection():
    """Health check fails — pooled connection is removed and a new one is created."""
    dummy_connect = dummy_connect_factory()

    async def health_check(conn: DummyConnection) -> bool:
        return conn.healthy

    pool = ConnectionPool(
        max_session_duration=60,
        connect_cb=dummy_connect,
        health_check_cb=health_check,
    )

    conn1 = await pool.get(timeout=10.0)
    conn1.healthy = False
    pool.put(conn1)

    conn2 = await pool.get(timeout=10.0)
    assert conn2 is not conn1, "Unhealthy connection should not be reused."


@pytest.mark.asyncio
async def test_health_check_timeout():
    """Health check that times out causes connection to be removed."""
    dummy_connect = dummy_connect_factory()

    async def slow_health_check(conn: DummyConnection) -> bool:
        await asyncio.sleep(10)  # longer than the 2s timeout
        return True

    pool = ConnectionPool(
        max_session_duration=60,
        connect_cb=dummy_connect,
        health_check_cb=slow_health_check,
    )

    conn1 = await pool.get(timeout=10.0)
    pool.put(conn1)

    conn2 = await pool.get(timeout=10.0)
    assert conn2 is not conn1, "Connection with timed-out health check should not be reused."


@pytest.mark.asyncio
async def test_health_check_exception():
    """Health check that raises an exception causes connection to be removed."""
    dummy_connect = dummy_connect_factory()

    async def failing_health_check(conn: DummyConnection) -> bool:
        raise RuntimeError("health check failed")

    pool = ConnectionPool(
        max_session_duration=60,
        connect_cb=dummy_connect,
        health_check_cb=failing_health_check,
    )

    conn1 = await pool.get(timeout=10.0)
    pool.put(conn1)

    conn2 = await pool.get(timeout=10.0)
    assert conn2 is not conn1, "Connection with failing health check should not be reused."


@pytest.mark.asyncio
async def test_no_health_check_by_default():
    """Without health_check_cb, connections are reused without checks."""
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn1 = await pool.get(timeout=10.0)
    pool.put(conn1)

    conn2 = await pool.get(timeout=10.0)
    assert conn1 is conn2, "Without health check, connection should be reused."
