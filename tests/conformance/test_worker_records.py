"""Typed server updates and CLI command dispatch through the generated native boundary."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import pytest

from livekit.agents import AgentServer, JobContext, _core, cli
from livekit.agents._generated.worker_cli import CliArgsConsole, CliArgsStart
from livekit.agents._generated.worker_options import ServerOptions, WorkerPermissions
from livekit.agents.job import RunningJobInfo
from livekit.agents.worker import _ServerHost

pytestmark = pytest.mark.unit


def test_server_job_generated_record_roundtrip_refuses_partial_configuration_updates() -> None:
    options = ServerOptions(
        ws_url="wss://original.example",
        permissions=WorkerPermissions(hidden=True, can_publish=False, can_publish_sources=[1, 2]),
    )
    config = _core.ServerConfig(options)
    with pytest.raises(ValueError, match="initialize_process_timeout"):
        config.update(
            ServerOptions(ws_url="wss://refused.example", initialize_process_timeout=float("nan"))
        )
    assert config.ws_url == "wss://original.example"
    config.update(ServerOptions(ws_url="wss://accepted.example", drain_timeout=None))
    assert config.ws_url == "wss://accepted.example"
    assert config.drain_timeout == 0


def test_cli_generated_args_matrix_dispatches_every_command_and_preserves_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = AgentServer()
    monkeypatch.setattr(cli, "_load_server", lambda _: server)
    calls: list[tuple[str, Any]] = []

    def run_worker(server: AgentServer, args: CliArgsStart) -> int:
        calls.append(("start", (args.ws_url, args.api_key, args.api_secret, args.log_style)))
        return 17

    def run_console(server: AgentServer, connect_addr: str, record: bool) -> None:
        calls.append(("console", (connect_addr, record)))

    monkeypatch.setattr(cli, "_run_worker", run_worker)
    monkeypatch.setattr(cli, "_run_console", run_console)
    monkeypatch.setattr(cli, "_download_files", lambda: 19)
    assert (
        cli.main(["start", "app.py", "--dev", "--simulation", "--url", "wss://cli.example"]) == 17
    )
    assert calls.pop() == ("start", ("wss://cli.example", None, None, "compact"))
    assert cli.main(["console", "--connect-addr", "127.0.0.1:9", "--record"]) == 0
    assert calls.pop() == ("console", ("127.0.0.1:9", True))
    assert cli.main(["download-files"]) == 19
    console = _core.parse_cli(["console", "--connect-addr", "127.0.0.1:9"])
    assert isinstance(console, CliArgsConsole)
    monkeypatch.setattr(sys, "argv", ["app.py", "download-files"])
    with pytest.raises(SystemExit) as exited:
        cli.run_app(server)
    assert exited.value.code == 19


@pytest.mark.skipif(sys.platform == "linux", reason="Linux supports forkserver directly")
async def test_non_linux_forkserver_fallback_is_reported_after_native_start(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    server = AgentServer(
        ws_url="ws://127.0.0.1:1",
        api_key="test-key",
        api_secret="test-secret",
        multiprocessing_context="forkserver",
        num_idle_processes=0,
    )

    @server.rtc_session()
    async def entrypoint(context: JobContext) -> None:
        pass

    server._config.simulation = True
    started = asyncio.Event()
    server.on("worker_started", started.set)
    running = asyncio.create_task(server.run(devmode=True, unregistered=True))
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        async with asyncio.timeout(3):
            while not any("forkserver is used on Linux only" in r.message for r in caplog.records):
                await asyncio.sleep(0)
    finally:
        await server.aclose()
        await asyncio.wait_for(running, timeout=3)


async def test_native_simulated_job_reaches_python_launch_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched: list[RunningJobInfo] = []

    async def launch(host: _ServerHost, info: RunningJobInfo) -> None:
        launched.append(info)

    monkeypatch.setattr(_ServerHost, "launch", launch)
    server = AgentServer(num_idle_processes=0)

    @server.rtc_session()
    async def entrypoint(context: JobContext) -> None:
        pass

    server._config.simulation = True
    started = asyncio.Event()
    server.on("worker_started", started.set)
    running = asyncio.create_task(server.run(devmode=True, unregistered=True))
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        await server.simulate_job("typed-launch", fake_job=True)
        assert len(launched) == 1
        assert launched[0].fake_job
        assert launched[0].job.room.name == "typed-launch"
    finally:
        await server.aclose()
        await asyncio.wait_for(running, timeout=3)
