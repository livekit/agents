"""Tests that server.aclose() is always called during shutdown, even when
server.drain() raises TimeoutError.

Currently, _run_worker (cli.py) does not catch TimeoutError from drain(),
which means aclose() is skipped — stuck child processes are never killed
and become orphaned until the pod is force-terminated.

The drain() docstring explicitly documents this behavior:
  "When timeout isn't None, it will raise asyncio.TimeoutError if the
   processes didn't finish in time."

But _run_worker's except clause only catches _ExitCli, not TimeoutError.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents.cli.cli import _ExitCli, _run_worker
from livekit.agents.cli.proto import CliArgs
from livekit.agents.worker import AgentServer


def _make_server(drain_timeout: int = 1) -> AgentServer:
    server = AgentServer(drain_timeout=drain_timeout)
    return server


class TestDrainTimeoutSkipsAclose:
    """When drain() raises TimeoutError, _run_worker does not call aclose()."""

    def test_aclose_called_when_drain_succeeds(self) -> None:
        """Baseline: when drain completes normally, aclose IS called."""
        server = _make_server()

        with (
            patch.object(server, "drain", new_callable=AsyncMock) as mock_drain,
            patch.object(server, "aclose", new_callable=AsyncMock) as mock_aclose,
            patch.object(server, "run", new_callable=AsyncMock),
        ):
            _run_worker(
                server,
                args=CliArgs(log_level="ERROR", url=None, api_key=None, api_secret=None),
            )

        mock_drain.assert_awaited_once()
        mock_aclose.assert_awaited_once()

    def test_aclose_skipped_when_drain_times_out(self) -> None:
        """BUG: when drain raises TimeoutError, aclose is never called.

        This means stuck child processes are never sent ShutdownRequest
        and never SIGKILL'd. They become orphaned until K8s force-kills
        the pod.

        The TimeoutError originates from asyncio.wait_for() inside
        AgentServer.drain() (worker.py) when a child process doesn't
        exit within drain_timeout seconds.
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
            with pytest.raises(asyncio.TimeoutError):
                _run_worker(
                    server,
                    args=CliArgs(log_level="ERROR", url=None, api_key=None, api_secret=None),
                )

        # BUG: aclose was never called because TimeoutError is not caught
        mock_aclose.assert_not_awaited()

    def test_drain_raises_timeout_from_stuck_process(self) -> None:
        """Demonstrates that drain() itself raises TimeoutError via
        asyncio.wait_for when a child process won't exit.

        This exercises the real AgentServer.drain() code path rather than
        mocking it, confirming that the TimeoutError originates from
        worker.py:868.
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

    def test_aclose_skipped_on_any_non_exitcli_exception(self) -> None:
        """The same bug applies to ANY exception type that isn't _ExitCli."""
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
                _run_worker(
                    server,
                    args=CliArgs(log_level="ERROR", url=None, api_key=None, api_secret=None),
                )

        mock_aclose.assert_not_awaited()

    def test_exitcli_is_caught_but_aclose_still_skipped(self) -> None:
        """When drain() raises _ExitCli (second SIGTERM), it is caught
        but aclose() is still not called — the handler calls os._exit(1).
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
            _run_worker(
                server,
                args=CliArgs(log_level="ERROR", url=None, api_key=None, api_secret=None),
            )

        mock_aclose.assert_not_awaited()
        mock_exit.assert_called_once_with(1)
