"""Tests that server.aclose() is always called during shutdown, even when
server.drain() raises TimeoutError.

Previously, _run_worker (cli.py) did not catch TimeoutError from drain(),
which meant aclose() was skipped — stuck child processes were never killed
and became orphaned until the pod was force-terminated.

The drain() docstring explicitly documents this behavior:
  "When timeout isn't None, it will raise asyncio.TimeoutError if the
   processes didn't finish in time."

The fix catches asyncio.TimeoutError around drain() so aclose() always runs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents.cli.cli import _ExitCli, _run_worker
from livekit.agents.cli.proto import CliArgs
from livekit.agents.worker import AgentServer

_CLI_ARGS = CliArgs(log_level="ERROR", url=None, api_key=None, api_secret=None)


def _make_server(drain_timeout: int = 1) -> AgentServer:
    server = AgentServer(drain_timeout=drain_timeout)
    return server


class TestDrainTimeout:
    """Verify that aclose() is called regardless of drain() outcome."""

    def test_aclose_called_when_drain_succeeds(self) -> None:
        """Baseline: when drain completes normally, aclose IS called."""
        server = _make_server()

        with (
            patch.object(server, "drain", new_callable=AsyncMock) as mock_drain,
            patch.object(server, "aclose", new_callable=AsyncMock) as mock_aclose,
            patch.object(server, "run", new_callable=AsyncMock),
        ):
            _run_worker(server, args=_CLI_ARGS)

        mock_drain.assert_awaited_once()
        mock_aclose.assert_awaited_once()

    def test_aclose_called_when_drain_times_out(self) -> None:
        """When drain raises TimeoutError, aclose must still be called.

        The TimeoutError originates from asyncio.wait_for() inside
        AgentServer.drain() (worker.py) when a child process doesn't
        exit within drain_timeout seconds. Without catching it,
        stuck child processes are never sent ShutdownRequest and never
        SIGKILL'd — they become orphaned until the pod is force-killed.
        """
        server = _make_server()

        with (
            patch.object(
                server,
                "drain",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError("drain timed out"),
            ),
            patch.object(server, "aclose", new_callable=AsyncMock) as mock_aclose,
            patch.object(server, "run", new_callable=AsyncMock),
        ):
            _run_worker(server, args=_CLI_ARGS)

        mock_aclose.assert_awaited_once()

    def test_drain_raises_timeout_from_stuck_process(self) -> None:
        """Exercises real AgentServer.drain() with a stuck process,
        confirming TimeoutError originates from asyncio.wait_for in
        worker.py.
        """
        server = _make_server(drain_timeout=1)

        # Set up internal state that drain() needs, without calling run()
        server._draining = False
        server._lock = asyncio.Lock()
        server._job_lifecycle_tasks = set[asyncio.Task[object]]()

        # Create a fake proc whose join() never completes (stuck process)
        stuck_future: asyncio.Future[None] = asyncio.Future()

        class StuckProc:
            running_job = True  # drain checks this to decide whether to wait

            async def join(self) -> None:
                await stuck_future  # never resolves

        class FakeProcPool:
            processes = [StuckProc()]

        server._proc_pool = FakeProcPool()  # type: ignore[assignment]

        # Suppress the _update_worker_status call which needs a websocket
        with patch.object(server, "_update_worker_status", new_callable=AsyncMock):
            with pytest.raises(asyncio.TimeoutError):
                asyncio.get_event_loop().run_until_complete(server.drain())

    def test_aclose_skipped_on_non_exitcli_non_timeout_exception(self) -> None:
        """Other exceptions from drain() still propagate (only TimeoutError
        is caught by the fix).
        """
        server = _make_server()

        with (
            patch.object(
                server,
                "drain",
                new_callable=AsyncMock,
                side_effect=RuntimeError("unexpected"),
            ),
            patch.object(server, "aclose", new_callable=AsyncMock) as mock_aclose,
            patch.object(server, "run", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError):
                _run_worker(server, args=_CLI_ARGS)

        mock_aclose.assert_not_awaited()

    def test_exitcli_during_drain_forces_exit(self) -> None:
        """When drain() raises _ExitCli (second SIGTERM), the handler
        calls os._exit(1) for a forceful shutdown.
        """
        server = _make_server()

        with (
            patch.object(
                server,
                "drain",
                new_callable=AsyncMock,
                side_effect=_ExitCli(),
            ),
            patch.object(server, "aclose", new_callable=AsyncMock) as mock_aclose,
            patch.object(server, "run", new_callable=AsyncMock),
            patch("os._exit") as mock_exit,
        ):
            _run_worker(server, args=_CLI_ARGS)

        mock_aclose.assert_not_awaited()
        mock_exit.assert_called_once_with(1)
