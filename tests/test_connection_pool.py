import asyncio
import time

import pytest
from aiohttp import RequestInfo, WSServerHandshakeError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions, APITimeoutError
from livekit.agents.utils import ConnectionPool, connection_pool

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

NO_POOL_WAIT = APIConnectOptions(pool_timeout=0)


@pytest.mark.asyncio
async def test_pool_wait_expires_without_interrupting_the_connecting_caller():
    connecting = asyncio.Event()
    release = asyncio.Event()

    async def connect(conn_options: APIConnectOptions):
        connecting.set()
        await release.wait()
        return DummyConnection(1)

    pool = ConnectionPool(connect_cb=connect)
    first = asyncio.create_task(pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS))
    try:
        await connecting.wait()
        with pytest.raises(APITimeoutError) as error:
            await pool.get(conn_options=NO_POOL_WAIT)
        assert error.value.phase == "pool"
        assert not first.done()
        release.set()
        connection = await first
        pool.put(connection)
        assert await pool.get(conn_options=NO_POOL_WAIT) is connection
    finally:
        release.set()
        await first
        await pool.aclose()


class DummyConnection:
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return f"DummyConnection({self.id})"


def dummy_connect_factory():
    counter = 0

    async def dummy_connect(conn_options: APIConnectOptions):
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
async def test_remove_connection():
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    conn = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.put(conn)
    # Reset the connection which should remove it from the pool.
    pool.remove(conn)

    # Even if we try to put it back, it won't be added because it's not tracked anymore.
    pool.put(conn)
    new_conn = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert new_conn is not conn, "Expected a removed connection to not be reused."


@pytest.mark.asyncio
async def test_get_expired(monkeypatch: pytest.MonkeyPatch):
    # Use a short max duration to simulate expiration.
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=1, connect_cb=dummy_connect)

    conn = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.put(conn)
    # the pool ages connections on the wall clock: two seconds pass (max_session_duration is 1)
    later = time.time() + 2
    monkeypatch.setattr(connection_pool.time, "time", lambda: later)

    conn2 = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert conn2 is not conn, "Expected a new connection to be returned."


@pytest.mark.asyncio
async def test_prewarm_failure_does_not_leak_api_key_in_logs(caplog):
    secret = "cartesia-secret-api-key-do-not-log"

    async def failing_connect(conn_options: APIConnectOptions):
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

    async def failing_connect(conn_options: APIConnectOptions):
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

    async def flaky_connect(conn_options: APIConnectOptions):
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

    pool.prewarm()
    task = pool._prewarm_task()
    assert task is not None
    await task

    assert attempts == 2
    conn = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert (conn.id, attempts) == (2, 2), "Expected the prewarmed connection to be reused."


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
async def test_invalidate_closes_idle_connections_immediately():
    pool, closed = _closing_pool()

    idle = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.put(idle)
    pool.invalidate()

    await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert idle in closed, "Expected an idle connection to be closed by the next drain."


@pytest.mark.asyncio
async def test_retired_connection_is_closed_when_removed_after_an_error():
    pool, closed = _closing_pool()

    in_use = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.invalidate()
    pool.remove(in_use)  # the stream raised; connection() calls remove()

    await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert in_use in closed, "Expected a retired connection to be closed when removed."


@pytest.mark.asyncio
async def test_aclose_closes_retired_connections_never_returned():
    pool, closed = _closing_pool()

    leaked = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.invalidate()
    await pool.aclose()

    assert leaked in closed, "aclose() must close retired connections that were never returned."


@pytest.mark.asyncio
async def test_invalidate_mid_stream_lets_the_stream_finish_then_reconnects():
    """The update_options case: options change while one stream is speaking."""
    pool, closed = _closing_pool()

    speaking = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.invalidate()  # e.g. update_options(voice=...) on the TTS

    # a new stream starts and must not reuse the old settings
    fresh = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert fresh is not speaking
    assert speaking not in closed, "The speaking connection was cut off mid-utterance."

    # the first stream finishes normally and its connection retires
    pool.put(speaking)
    pool.put(fresh)
    reused = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert reused is fresh, "Expected the post-invalidate connection to be the reusable one."
    assert speaking in closed


@pytest.mark.asyncio
async def test_invalidate_during_a_handshake_discards_the_stale_connection():
    """A socket negotiated with the old options is never handed to a caller."""
    started = asyncio.Event()
    release = asyncio.Event()
    counter = 0
    closed: list[DummyConnection] = []

    async def slow_connect(conn_options: APIConnectOptions):
        nonlocal counter
        counter += 1
        started.set()
        await release.wait()
        return DummyConnection(counter)

    async def close_cb(conn: DummyConnection) -> None:
        closed.append(conn)

    pool = ConnectionPool(connect_cb=slow_connect, close_cb=close_cb)

    acquiring = asyncio.create_task(pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS))
    await started.wait()
    pool.invalidate()  # options changed while the socket was still being negotiated
    release.set()
    conn = await acquiring

    assert conn.id == 2, "Expected the caller to get a connection negotiated after the change."
    # closed on the way out of the handshake, not left queued until some later acquisition
    assert [c.id for c in closed] == [1], "The discarded socket was left open."

    pool.put(conn)
    reused = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert reused is conn, "Expected the fresh connection to be pooled, not the stale one."


@pytest.mark.asyncio
async def test_prewarm_discards_a_connection_invalidated_mid_handshake():
    started = asyncio.Event()
    release = asyncio.Event()
    counter = 0
    closed: list[DummyConnection] = []

    async def slow_connect(conn_options: APIConnectOptions):
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

    # the discarded attempt must not leave the pool cold, and must not be reusable
    warm = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    assert (warm.id, counter) == (2, 2), "Expected prewarm to end with a usable connection."
    pool.put(warm)

    await pool.aclose()
    assert sorted(c.id for c in closed) == [1, 2], (
        "Expected both the discarded and the replacement connection to be closed."
    )


@pytest.mark.asyncio
async def test_cancelling_a_drain_leaves_the_connection_queued_for_a_later_close():
    """A cancelled close must not strand a connection nothing else owns."""
    closing = asyncio.Event()
    finish = asyncio.Event()
    closed: list[DummyConnection] = []

    async def close_cb(conn: DummyConnection) -> None:
        closing.set()
        await finish.wait()
        closed.append(conn)

    pool = ConnectionPool(connect_cb=dummy_connect_factory(), close_cb=close_cb)

    doomed = await pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS)
    pool.put(doomed)
    pool.invalidate()  # idle, so it goes straight to the close queue

    acquiring = asyncio.create_task(pool.get(conn_options=DEFAULT_API_CONNECT_OPTIONS))
    await closing.wait()  # the drain is closing it
    acquiring.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquiring

    assert doomed not in closed

    finish.set()
    await pool.aclose()
    assert doomed in closed, "Expected the requeued connection to be closed later."
