"""Worker shutdown: aclose() runs after drain() times out, and signals are never swallowed."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import socket
from unittest.mock import AsyncMock, patch

import pytest

from livekit.agents.cli.cli import _ExitCli, _run_worker
from livekit.agents.cli.proto import CliArgs
from livekit.agents.ipc.supervised_proc import SupervisedProc, SupervisedProcKind
from livekit.agents.utils import aio
from livekit.agents.worker import AgentServer

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_CLI_ARGS = CliArgs(log_level="ERROR", url=None, api_key=None, api_secret=None)


class _DummySupervisedProc(SupervisedProc):
    @property
    def process_kind(self) -> SupervisedProcKind:
        return SupervisedProcKind.JOB

    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process:
        raise NotImplementedError

    async def _main_task(self, ipc_ch: aio.ChanReceiver[object]) -> None:
        raise NotImplementedError


class TestDrainTimeout:
    def test_aclose_called_when_drain_succeeds(self) -> None:
        server = AgentServer(drain_timeout=1)

        with (
            patch.object(server, "drain", new_callable=AsyncMock) as mock_drain,
            patch.object(server, "aclose", new_callable=AsyncMock) as mock_aclose,
            patch.object(server, "run", new_callable=AsyncMock),
        ):
            _run_worker(server, args=_CLI_ARGS)

        mock_drain.assert_awaited_once()
        mock_aclose.assert_awaited_once()

    def test_aclose_called_when_drain_times_out(self) -> None:
        """Stuck children must still be killed when drain() times out."""
        server = AgentServer(drain_timeout=1)

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

    async def test_drain_raises_timeout_from_stuck_process(self) -> None:
        server = AgentServer(drain_timeout=1)
        # the state drain() needs, without calling run()
        server._draining = False
        server._lock = asyncio.Lock()
        server._job_lifecycle_tasks = set[asyncio.Task[object]]()

        stuck_future: asyncio.Future[None] = asyncio.Future()

        class StuckProc:
            running_job = True

            async def join(self) -> None:
                await stuck_future

        class FakeProcPool:
            processes = [StuckProc()]

        server._proc_pool = FakeProcPool()  # type: ignore[assignment]

        with patch.object(server, "_update_worker_status", new_callable=AsyncMock):
            with pytest.raises(asyncio.TimeoutError):
                await server.drain()

    def test_other_drain_errors_propagate(self) -> None:
        server = AgentServer(drain_timeout=1)

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

    def test_signal_during_drain_forces_exit(self) -> None:
        import signal as signal_mod

        server = AgentServer(drain_timeout=1)

        async def _signal_while_draining() -> None:
            handler = signal_mod.getsignal(signal_mod.SIGTERM)
            assert callable(handler)
            handler(signal_mod.SIGTERM, None)

        with (
            patch.object(
                server,
                "drain",
                new_callable=AsyncMock,
                side_effect=_signal_while_draining,
            ),
            patch.object(server, "aclose", new_callable=AsyncMock),
            patch.object(server, "run", new_callable=AsyncMock),
            patch("os._exit") as mock_exit,
        ):
            _run_worker(server, args=_CLI_ARGS)

        mock_exit.assert_called_once_with(1)

    # psutil runs in an executor thread, which virtual time would stop waiting for at teardown
    @pytest.mark.real_time
    async def test_memory_monitor_does_not_swallow_exitcli(self) -> None:
        """A shutdown signal raised inside psutil must not be eaten by a broad except."""
        proc = _DummySupervisedProc(
            initialize_timeout=1.0,
            close_timeout=1.0,
            memory_warn_mb=0.0,
            memory_limit_mb=0.0,
            ping_interval=1.0,
            ping_timeout=1.0,
            high_ping_threshold=1.0,
            http_proxy=None,
            mp_ctx=mp.get_context("spawn"),
            loop=asyncio.get_running_loop(),
        )
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
                await proc._memory_monitor_task()

        mock_exception.assert_not_called()
