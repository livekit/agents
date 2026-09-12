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


class OrderedPopSet(set):
    def __init__(self, values, pop_order):
        super().__init__(values)
        self._pop_order = iter(pop_order)

    def pop(self):
        for conn in self._pop_order:
            if conn in self:
                self.remove(conn)
                return conn
        return super().pop()


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
async def test_get_reuses_only_a_connection_with_a_matching_key():
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    v2 = await pool.get(timeout=10.0, key="bulbul:v2")
    pool.put(v2)
    v3 = await pool.get(timeout=10.0, key="bulbul:v3")
    pool.put(v3)

    assert await pool.get(timeout=10.0, key="bulbul:v2") is v2
    assert await pool.get(timeout=10.0, key="bulbul:v3") is not v2


@pytest.mark.asyncio
async def test_matching_key_lookup_restores_skipped_connections():
    dummy_connect = dummy_connect_factory()
    pool = ConnectionPool(max_session_duration=60, connect_cb=dummy_connect)

    v2 = await pool.get(timeout=10.0, key="bulbul:v2")
    pool.put(v2)
    v3 = await pool.get(timeout=10.0, key="bulbul:v3")
    pool.put(v3)
    pool._available = OrderedPopSet([v2, v3], [v3, v2])

    assert await pool.get(timeout=10.0, key="bulbul:v2") is v2
    assert pool._available == {v3}
    assert await pool.get(timeout=10.0, key="bulbul:v3") is v3


@pytest.mark.asyncio
async def test_inflight_connection_uses_the_requested_snapshot_callback():
    started = asyncio.Event()
    release = asyncio.Event()
    attempts: list[str] = []

    async def connect_current(timeout: float):
        attempts.append("current")
        return DummyConnection(len(attempts))

    async def connect_snapshot(timeout: float):
        attempts.append("bulbul:v2")
        started.set()
        await release.wait()
        return DummyConnection(len(attempts))

    pool = ConnectionPool(connect_cb=connect_current)
    acquiring = asyncio.create_task(
        pool.get(
            timeout=10.0,
            key="bulbul:v2",
            connect_cb=connect_snapshot,
        )
    )
    await started.wait()
    pool.invalidate()
    release.set()
    await acquiring

    assert attempts == ["bulbul:v2", "bulbul:v2"]


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
async def test_aclose_waits_for_an_inflight_get_without_reconnecting_after_shutdown():
    started = asyncio.Event()
    release = asyncio.Event()
    closed: list[DummyConnection] = []
    counter = 0

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

    closing = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)
    assert not closing.done(), "Shutdown should wait for the in-flight handshake."

    release.set()
    conn = await acquiring
    await closing

    assert conn in closed
    assert not pool._connections
    assert not pool._available
    with pytest.raises(RuntimeError, match="closed"):
        await pool.get(timeout=10.0)


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
async def test_invalidate_during_a_handshake_discards_the_stale_connection():
    """A socket negotiated with the old options is never handed to a caller."""
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
    conn = await acquiring

    assert conn.id == 2, "Expected the caller to get a connection negotiated after the change."
    assert all(c.id != 1 for c in pool._available), "The stale connection was pooled."
    assert conn in pool._connections

    # closed on the way out of _connect, not left queued until some later acquisition
    assert [c.id for c in closed] == [1], "The discarded socket was left open."
    assert not pool._to_close, "The discarded socket is still queued rather than closed."


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

    # the discarded attempt must not leave the pool cold, and must not be reusable
    assert len(pool._available) == 1, "Expected prewarm to end with a usable connection."
    assert next(iter(pool._available)).id == 2

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

    doomed = await pool.get(timeout=10.0)
    pool.put(doomed)
    pool.invalidate()  # idle, so it goes straight to the close queue

    acquiring = asyncio.create_task(pool.get(timeout=10.0))
    await closing.wait()  # inside _maybe_close_connection, popped from _to_close
    acquiring.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquiring

    assert doomed in pool._to_close, "A cancelled drain dropped the connection entirely."
    assert doomed not in closed

    finish.set()
    await pool.aclose()
    assert doomed in closed, "Expected the requeued connection to be closed later."
