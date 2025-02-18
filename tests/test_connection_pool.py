import time

import pytest
from livekit.agents.utils import ConnectionPool


# A simple dummy connection object.
class DummyConnection:
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"DummyConnection({self.id})"


# Factory to produce a dummy async connect callback that returns unique DummyConnection objects.
def dummy_connect_factory():
    counter = 0

    async def dummy_connect():
        nonlocal counter
        counter += 1
        return DummyConnection(counter)

    return dummy_connect


@pytest.mark.asyncio
async def test_get_reuses_connection():
    """
    Test that when a connection is returned to the pool via put(),
    the subsequent call to get() reuses the same connection if it hasn't expired.
    """
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn1 = await pool.get()
    # Return the connection to the pool
    pool.put(conn1)

    conn2 = await pool.get()
    assert conn1 is conn2, (
        "Expected the same connection to be reused when it hasn't expired."
    )


@pytest.mark.asyncio
async def test_get_creates_new_connection_when_none_available():
    """
    Test that get() creates a new connection when there are no available connections.
    """
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn1 = await pool.get()
    # Not putting conn1 back means the available pool is empty,
    # so calling get() again should create a new connection.
    conn2 = await pool.get()
    assert conn1 is not conn2, (
        "Expected a new connection when no available connection exists."
    )


@pytest.mark.asyncio
async def test_reset_connection():
    """
    Test that after resetting a connection (via reset()), the connection is not reused.
    """
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn = await pool.get()
    pool.put(conn)
    # Reset the connection which should remove it from the pool.
    pool.reset(conn)

    # Even if we try to put it back, it won't be added because it's not tracked anymore.
    pool.put(conn)
    new_conn = await pool.get()
    assert new_conn is not conn, "Expected a reset connection to not be reused."


@pytest.mark.asyncio
async def test_maybe_reset_not_expired():
    """
    Test that maybe_reset does not remove a connection that is still within its valid session duration.
    """
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn = await pool.get()
    pool.put(conn)
    # Since the connection is fresh, maybe_reset() should leave it intact.
    pool.maybe_reset(conn)
    assert conn in pool._connections, (
        "Connection should not be reset if within max_session_duration."
    )


@pytest.mark.asyncio
async def test_maybe_reset_expired():
    """
    Test that maybe_reset removes an expired connection.
    """
    # Use a short max duration to simulate expiration.
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=1, connect_cb=dummy_connect)

    conn = await pool.get()
    pool.put(conn)
    # Artificially set the connection's timestamp in the past to simulate expiration.
    pool._connections[conn] = (
        time.time() - 2
    )  # 2 seconds ago (max_session_duration is 1)

    pool.maybe_reset(conn)
    assert conn not in pool._connections, (
        "Expired connection should be removed from _connections."
    )
    assert conn not in pool._available, (
        "Expired connection should be removed from _available."
    )
