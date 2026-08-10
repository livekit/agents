import time

import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

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


def _handshake_error_with_api_key(api_key: str) -> WSServerHandshakeError:
    url = URL("wss://api.cartesia.ai/tts/websocket")
    headers = CIMultiDict({"Host": "api.cartesia.ai", "X-API-Key": api_key})
    request_info = RequestInfo(
        url=url, method="GET", headers=CIMultiDictProxy(headers), real_url=url
    )
    return WSServerHandshakeError(request_info, (), status=401, message="Unauthorized")


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
async def test_prewarm_failure_does_not_leak_api_key_in_logs(caplog):
    secret = "cartesia-secret-api-key-do-not-log"

    async def failing_connect(timeout: float):
        raise _handshake_error_with_api_key(secret)

    pool = ConnectionPool(connect_cb=failing_connect)
    with caplog.at_level("WARNING"):
        pool.prewarm()
        task = pool._prewarm_task()
        assert task is not None
        await task

    assert secret not in repr(task)
    assert all(secret not in record.getMessage() for record in caplog.records)
    warning_records = [
        r for r in caplog.records if "failed to prewarm connection pool" in r.getMessage()
    ]
    assert warning_records
    assert warning_records[0].exception_type == "WSServerHandshakeError"


@pytest.mark.asyncio
async def test_prewarm_failure_does_not_leak_url_credentials_in_logs(caplog):
    secret_key = "url-secret-api-key-do-not-log"
    secret_jwt = "url-secret-jwt-token-do-not-log"

    async def failing_connect(timeout: float):
        raise ConnectionError(f"wss://example.com/ws?api_key={secret_key}&jwt_token={secret_jwt}")

    pool = ConnectionPool(connect_cb=failing_connect)
    with caplog.at_level("WARNING"):
        pool.prewarm()
        task = pool._prewarm_task()
        assert task is not None
        await task

    assert all(secret_key not in record.getMessage() for record in caplog.records)
    assert all(secret_jwt not in record.getMessage() for record in caplog.records)
    warning_records = [
        r for r in caplog.records if "failed to prewarm connection pool" in r.getMessage()
    ]
    assert warning_records
    assert warning_records[0].exception_type == "ConnectionError"
