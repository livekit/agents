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
import multiprocessing as mp
import socket
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents.cli.cli import _ExitCli, _run_worker
from livekit.agents.cli.proto import CliArgs
from livekit.agents.ipc.supervised_proc import SupervisedProc
from livekit.agents.utils import aio
from livekit.agents.worker import AgentServer

_CLI_ARGS = CliArgs(log_level="ERROR", url=None, api_key=None, api_secret=None)


def _make_server(drain_timeout: int = 1) -> AgentServer:
    server = AgentServer(drain_timeout=drain_timeout)
    return server


class _DummySupervisedProc(SupervisedProc):
    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        raise NotImplementedError

    async def _main_task(self, ipc_ch: aio.ChanReceiver[object]) -> None:
        raise NotImplementedError


def _make_supervised_proc() -> _DummySupervisedProc:
    return _DummySupervisedProc(
        initialize_timeout=1.0,
        close_timeout=1.0,
        memory_warn_mb=0.0,
        memory_limit_mb=0.0,
        ping_interval=1.0,
        ping_timeout=1.0,
        high_ping_threshold=1.0,
        http_proxy=None,
        mp_ctx=mp.get_context("spawn"),
        loop=asyncio.get_event_loop(),
    )


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

    def test_memory_monitor_does_not_swallow_exitcli(self) -> None:
        """SIGTERM/SIGINT should not be eaten by broad Exception handlers.

        Issue #4664 showed _ExitCli being raised from a signal handler while
        _memory_monitor_task() was inside psutil. Because _ExitCli inherited
        from Exception, the blanket ``except Exception`` here swallowed the
        shutdown signal and left the worker running instead of draining.
        """
        proc = _make_supervised_proc()
        proc._pid = 123

        async def _fake_sleep(_: float) -> None:
            proc._closing = True

        with (
            patch(
                "livekit.agents.ipc.supervised_proc.psutil.Process",
                side_effect=_ExitCli(),
            ),
            patch("livekit.agents.ipc.supervised_proc.asyncio.sleep", side_effect=_fake_sleep),
            patch("livekit.agents.ipc.supervised_proc.logger.exception") as mock_exception,
        ):
            with pytest.raises(_ExitCli):
                asyncio.get_event_loop().run_until_complete(proc._memory_monitor_task())

        mock_exception.assert_not_called()


class TestSupervisedProcAcloseDeadlock:
    """Verify that aclose() does not deadlock when _shutting_down_fut never resolves.

    In livekit-agents 1.5.2+ (PR #4580), aclose() awaits _shutting_down_fut with
    no timeout after the child acks shutdown. If the child gets stuck (e.g.
    session.aclose() hangs), the parent blocks forever — orphaning the worker.

    The fix wraps that await with wait_for(close_timeout).
    """

    def test_aclose_does_not_deadlock_on_shutting_down_fut(self) -> None:
        """aclose() must complete (not hang) when _shutting_down_fut never resolves."""
        proc = _make_supervised_proc()
        proc._closing = True
        proc._pch = AsyncMock()  # type: ignore[assignment]
        proc._shutdown_ack_fut.set_result(None)
        # _shutting_down_fut intentionally left unresolved (child is stuck)

        supervise_done: asyncio.Future[None] = asyncio.Future()

        async def fake_supervise() -> None:
            await supervise_done

        proc._supervise_atask = asyncio.ensure_future(fake_supervise())

        async def mock_kill() -> None:
            if not supervise_done.done():
                supervise_done.set_result(None)

        async def run_aclose() -> None:
            with (
                patch(
                    "livekit.agents.ipc.supervised_proc.channel.asend_message",
                    new_callable=AsyncMock,
                ),
                patch.object(proc, "_send_kill_signal", side_effect=mock_kill),
                patch.object(proc, "_send_dump_signal", new_callable=AsyncMock),
            ):
                await proc.aclose()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.wait_for(run_aclose(), timeout=10.0))

    def test_aclose_kills_process_when_shutting_down_times_out(self) -> None:
        """When _shutting_down_fut times out, aclose must send kill signal."""
        proc = _make_supervised_proc()
        proc._closing = True
        proc._pch = AsyncMock()  # type: ignore[assignment]
        proc._shutdown_ack_fut.set_result(None)
        # _shutting_down_fut intentionally left unresolved

        kill_call_count = 0
        supervise_done: asyncio.Future[None] = asyncio.Future()

        async def fake_supervise() -> None:
            await supervise_done

        proc._supervise_atask = asyncio.ensure_future(fake_supervise())

        async def mock_kill() -> None:
            nonlocal kill_call_count
            kill_call_count += 1
            if not supervise_done.done():
                supervise_done.set_result(None)

        async def run_aclose() -> None:
            with (
                patch(
                    "livekit.agents.ipc.supervised_proc.channel.asend_message",
                    new_callable=AsyncMock,
                ),
                patch.object(proc, "_send_kill_signal", side_effect=mock_kill),
                patch.object(proc, "_send_dump_signal", new_callable=AsyncMock),
            ):
                await proc.aclose()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.wait_for(run_aclose(), timeout=10.0))

        assert kill_call_count >= 1, (
            "Expected _send_kill_signal to be called after shutting_down timeout"
        )

    def test_aclose_proceeds_normally_when_shutting_down_resolves(self) -> None:
        """When _shutting_down_fut resolves promptly, no kill signal is sent."""
        proc = _make_supervised_proc()
        proc._closing = True
        proc._pch = AsyncMock()  # type: ignore[assignment]
        proc._shutdown_ack_fut.set_result(None)
        proc._shutting_down_fut.set_result(None)  # resolves immediately

        supervise_done: asyncio.Future[None] = asyncio.Future()
        supervise_done.set_result(None)

        async def fake_supervise() -> None:
            await supervise_done

        proc._supervise_atask = asyncio.ensure_future(fake_supervise())

        kill_called = False

        async def mock_kill() -> None:
            nonlocal kill_called
            kill_called = True

        async def run_aclose() -> None:
            with (
                patch(
                    "livekit.agents.ipc.supervised_proc.channel.asend_message",
                    new_callable=AsyncMock,
                ),
                patch.object(proc, "_send_kill_signal", side_effect=mock_kill),
                patch.object(proc, "_send_dump_signal", new_callable=AsyncMock),
            ):
                await proc.aclose()

        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.wait_for(run_aclose(), timeout=5.0))

        assert not kill_called, "Kill should NOT be called when shutting_down resolves in time"
