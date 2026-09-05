import asyncio
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


@pytest.mark.asyncio
async def test_prewarm_retries_after_failure():
    attempts = 0

    async def flaky_connect(timeout: float):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("temporary prewarm failure")
        return DummyConnection(attempts)

    pool = ConnectionPool(connect_cb=flaky_connect)
    pool.prewarm()
    task = pool._prewarm_task()
    assert task is not None
    await task

    assert attempts == 1
    assert not pool._connections

    pool.prewarm()
    task = pool._prewarm_task()
    assert task is not None
    await task

    assert attempts == 2
    assert len(pool._available) == 1


def _closing_pool(max_session_duration: float | None = 60):
    """A pool that records every connection handed to its close callback."""
    closed: list[DummyConnection] = []

    async def close_cb(conn: DummyConnection) -> None:
        closed.append(conn)

    pool = ConnectionPool(
        max_session_duration=max_session_duration,
        connect_cb=dummy_connect_factory(),
        close_cb=close_cb,
    )
    return pool, closed


@pytest.mark.asyncio
async def test_invalidate_does_not_close_a_connection_still_in_use():
    pool, closed = _closing_pool()

    # checked out and never returned: something is streaming on it right now
    in_use = await pool.get(timeout=10.0)
    pool.invalidate()

    # a second acquisition drains the close queue; the in-flight connection must survive
    other = await pool.get(timeout=10.0)
    assert other is not in_use, "Expected a fresh connection after invalidate()."
    assert in_use not in closed, "invalidate() closed a connection that was still checked out."


@pytest.mark.asyncio
async def test_invalidate_closes_idle_connections_immediately():
    pool, closed = _closing_pool()

    idle = await pool.get(timeout=10.0)
    pool.put(idle)
    pool.invalidate()

    await pool.get(timeout=10.0)
    assert idle in closed, "Expected an idle connection to be closed by the next drain."


@pytest.mark.asyncio
async def test_retired_connection_is_closed_once_returned():
    pool, closed = _closing_pool()

    in_use = await pool.get(timeout=10.0)
    pool.invalidate()
    pool.put(in_use)  # the stream finished with it

    assert in_use not in pool._available, "A retired connection must not be reused."
    await pool.get(timeout=10.0)
    assert in_use in closed, "Expected a retired connection to be closed once returned."


@pytest.mark.asyncio
async def test_retired_connection_is_closed_when_removed_after_an_error():
    pool, closed = _closing_pool()

    in_use = await pool.get(timeout=10.0)
    pool.invalidate()
    pool.remove(in_use)  # the stream raised; connection() calls remove()

    await pool.get(timeout=10.0)
    assert in_use in closed, "Expected a retired connection to be closed when removed."


@pytest.mark.asyncio
async def test_aclose_closes_retired_connections_never_returned():
    pool, closed = _closing_pool()

    leaked = await pool.get(timeout=10.0)
    pool.invalidate()
    await pool.aclose()

    assert leaked in closed, "aclose() must close retired connections that were never returned."


@pytest.mark.asyncio
async def test_invalidate_mid_stream_lets_the_stream_finish_then_reconnects():
    """The update_options case: options change while one stream is speaking."""
    pool, closed = _closing_pool()

    speaking = await pool.get(timeout=10.0)
    pool.invalidate()  # e.g. update_options(voice=...) on the TTS

    # a new stream starts and must not reuse the old settings
    fresh = await pool.get(timeout=10.0)
    assert fresh is not speaking
    assert speaking not in closed, "The speaking connection was cut off mid-utterance."

    # the first stream finishes normally and its connection retires
    pool.put(speaking)
    pool.put(fresh)
    reused = await pool.get(timeout=10.0)
    assert reused is fresh, "Expected the post-invalidate connection to be the reusable one."
    assert speaking in closed


@pytest.mark.asyncio
async def test_invalidate_during_a_handshake_retires_the_new_connection():
    """A connection negotiated with the old options must not be pooled for reuse."""
    started = asyncio.Event()
    release = asyncio.Event()
    counter = 0
    closed: list[DummyConnection] = []

    async def slow_connect(timeout: float):
        nonlocal counter
        counter += 1
        started.set()
        await release.wait()
        return DummyConnection(counter)

    async def close_cb(conn: DummyConnection) -> None:
        closed.append(conn)

    pool = ConnectionPool(connect_cb=slow_connect, close_cb=close_cb)

    acquiring = asyncio.create_task(pool.get(timeout=10.0))
    await started.wait()
    pool.invalidate()  # options changed while the socket was still being negotiated
    release.set()
    stale = await acquiring

    # the caller asked before the change and still gets its connection
    assert stale is not None
    pool.put(stale)
    assert stale not in pool._available, "A connection negotiated with stale options was pooled."

    started.clear()
    release.set()
    fresh = await pool.get(timeout=10.0)
    assert fresh is not stale, "Expected a connection negotiated after the option change."
    assert stale in closed, "Expected the stale connection to be closed once returned."


@pytest.mark.asyncio
async def test_prewarm_discards_a_connection_invalidated_mid_handshake():
    started = asyncio.Event()
    release = asyncio.Event()
    counter = 0
    closed: list[DummyConnection] = []

    async def slow_connect(timeout: float):
        nonlocal counter
        counter += 1
        started.set()
        await release.wait()
        return DummyConnection(counter)

    async def close_cb(conn: DummyConnection) -> None:
        closed.append(conn)

    pool = ConnectionPool(connect_cb=slow_connect, close_cb=close_cb)

    pool.prewarm()
    await started.wait()
    pool.invalidate()
    release.set()
    task = pool._prewarm_task()
    assert task is not None
    await task

    assert not pool._available, "A prewarmed connection with stale options stayed available."
    await pool.aclose()
    assert len(closed) == 1, "Expected the discarded prewarm connection to be closed."
