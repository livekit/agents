"""Connection-failure propagation tests.

When the worker's connection task exhausts ``max_retry`` it raises, and that
failure must surface out of ``AgentServer.run()`` instead of leaving the worker
hanging forever (https://github.com/livekit/agents/issues/6083).

The retry logs carry a second expectation: losing a connection that had been up and
stable is routine churn, while a connection that never came up, one that keeps
dropping straight away, and any retry that has to back off are all worth a warning
(https://github.com/livekit/agents/issues/6108).
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit.agents.job import JobContext
from livekit.agents.worker import STABLE_CONNECTION_INTERVAL, AgentServer
from livekit.protocol import agent

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


def _make_server(*, max_retry: int = 0) -> AgentServer:
    server = AgentServer(
        ws_url="ws://127.0.0.1:1",  # unreachable: connection refused
        api_key="devkey",
        api_secret="devsecret",
        max_retry=max_retry,
        num_idle_processes=0,
    )

    @server.rtc_session()
    async def _entry(ctx: JobContext) -> None:
        pass

    server._simulation = True  # skip binding the health HTTP server
    return server


def _registered_ws() -> MagicMock:
    """A websocket that answers the register handshake, so the worker comes up."""
    response = agent.ServerMessage()
    response.register.worker_id = "W_test"

    ws = MagicMock()
    ws.send_bytes = AsyncMock()
    ws.receive_bytes = AsyncMock(return_value=response.SerializeToString())
    ws.close = AsyncMock()
    return ws


def _unregistered_ws() -> MagicMock:
    """A websocket that upgrades but never answers the register handshake."""
    ws = MagicMock()
    ws.send_bytes = AsyncMock()
    ws.receive_bytes = AsyncMock(side_effect=ConnectionResetError())
    ws.close = AsyncMock()
    return ws


def _serves_for(seconds: float, error: Exception | None = None) -> AsyncMock:
    """A _run_ws that holds the connection open for `seconds`, then fails with `error`."""

    async def _run(ws) -> None:
        await asyncio.sleep(seconds)
        raise error or ConnectionResetError()

    return AsyncMock(side_effect=_run)


async def _retry_logs(server: AgentServer, connects: list, caplog) -> list[logging.LogRecord]:
    """Run the connection task to exhaustion and return its retry log records."""
    # run() normally clears _closed and builds the process pool; we drive the connection
    # task on its own, so stand both up here.
    server._closed = False
    server._proc_pool = MagicMock(processes=[])
    server._http_session = MagicMock()
    server._http_session.ws_connect = AsyncMock(side_effect=connects)

    with caplog.at_level(logging.DEBUG, logger="livekit.agents"):
        with pytest.raises(RuntimeError, match="failed to connect"):
            await server._connection_task()

    return [
        r
        for r in caplog.records
        if r.getMessage().startswith("failed to connect to livekit, retrying")
    ]


async def test_run_raises_when_connection_exhausts_retries() -> None:
    server = _make_server()

    fake_pool = MagicMock()
    fake_pool.start = AsyncMock()
    fake_pool.aclose = AsyncMock()
    fake_pool.processes = []

    with patch("livekit.agents.ipc.proc_pool.ProcPool", return_value=fake_pool):
        try:
            with pytest.raises(RuntimeError, match="failed to connect"):
                await asyncio.wait_for(server.run(devmode=True), timeout=10)
        finally:
            await server.aclose()
            await server.aclose()  # repeated aclose() stays a no-op


@pytest.mark.virtual_time
async def test_reconnect_after_a_stable_connection_drops_is_reported_at_info(caplog) -> None:
    server = _make_server(max_retry=1)

    with patch.object(server, "_run_ws", _serves_for(STABLE_CONNECTION_INTERVAL * 10)):
        records = await _retry_logs(server, [_registered_ws(), ConnectionRefusedError()], caplog)

    assert [record.getMessage() for record in records] == [
        "failed to connect to livekit, retrying in 0s"
    ]
    assert records[0].levelno == logging.INFO


@pytest.mark.virtual_time
async def test_connection_flapping_below_the_stable_interval_stays_at_warning(caplog) -> None:
    """A worker that registers and loses the connection straight back is not self-healing.

    Every accepted connect resets ``retry_count``, so this never backs off and never
    exhausts ``max_retry``. Demoting it would hide a control plane that is failing
    continuously behind a silent hot loop.
    """
    server = _make_server(max_retry=1)

    with patch.object(server, "_run_ws", _serves_for(STABLE_CONNECTION_INTERVAL / 10)):
        records = await _retry_logs(server, [_registered_ws(), ConnectionRefusedError()], caplog)

    assert [record.getMessage() for record in records] == [
        "failed to connect to livekit, retrying in 0s"
    ]
    assert records[0].levelno == logging.WARNING


async def test_startup_connection_failure_is_reported_at_warning(caplog) -> None:
    server = _make_server(max_retry=1)

    records = await _retry_logs(
        server, [ConnectionRefusedError(), ConnectionRefusedError()], caplog
    )

    assert [record.getMessage() for record in records] == [
        "failed to connect to livekit, retrying in 0s"
    ]
    assert records[0].levelno == logging.WARNING


@pytest.mark.virtual_time
async def test_backed_off_retry_is_reported_at_warning_even_after_a_stable_connection(
    caplog,
) -> None:
    server = _make_server(max_retry=2)

    with patch.object(server, "_run_ws", _serves_for(STABLE_CONNECTION_INTERVAL * 10)):
        records = await _retry_logs(
            server,
            [_registered_ws(), ConnectionRefusedError(), ConnectionRefusedError()],
            caplog,
        )

    assert [record.getMessage() for record in records] == [
        "failed to connect to livekit, retrying in 0s",
        "failed to connect to livekit, retrying in 2s",
    ]
    assert records[0].levelno == logging.INFO
    assert records[1].levelno == logging.WARNING


@pytest.mark.virtual_time
async def test_register_failures_stay_at_warning_after_an_earlier_stable_connection(
    caplog,
) -> None:
    """A control plane that accepts the upgrade but never registers is a real outage.

    ``retry_count`` resets on every accepted upgrade, so each of these attempts computes a
    0s delay. They must stay loud rather than inherit the quiet path from the connection
    that was serving earlier in the loop.
    """
    server = _make_server(max_retry=1)

    with patch.object(server, "_run_ws", _serves_for(STABLE_CONNECTION_INTERVAL * 10)):
        records = await _retry_logs(
            server,
            [_registered_ws(), _unregistered_ws(), ConnectionRefusedError()],
            caplog,
        )

    assert [record.getMessage() for record in records] == [
        "failed to connect to livekit, retrying in 0s",
        "failed to connect to livekit, retrying in 0s",
    ]
    assert records[0].levelno == logging.INFO
    assert records[1].levelno == logging.WARNING


@pytest.mark.virtual_time
async def test_a_fault_that_is_not_the_connection_dropping_stays_at_warning(caplog) -> None:
    """Only losing the connection is routine.

    A bug in a message handler, or a message the worker cannot parse, surfaces through the
    same retry path after an arbitrarily long uptime. Those are worth reporting however
    long the connection had been up.
    """
    server = _make_server(max_retry=1)

    run_ws = _serves_for(STABLE_CONNECTION_INTERVAL * 10, RuntimeError("bad server message"))
    with patch.object(server, "_run_ws", run_ws):
        records = await _retry_logs(server, [_registered_ws(), ConnectionRefusedError()], caplog)

    assert [record.getMessage() for record in records] == [
        "failed to connect to livekit, retrying in 0s"
    ]
    assert records[0].levelno == logging.WARNING
