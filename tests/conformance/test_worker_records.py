"""Typed server updates and CLI command dispatch through the generated native boundary."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import pytest

from livekit.agents import AgentServer, JobContext, _core, cli
from livekit.agents._generated.worker_cli import CliArgsConsole, CliArgsStart
from livekit.agents.job import RunningJobInfo
from livekit.agents.worker import _ServerCallbacks

pytestmark = pytest.mark.unit


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
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.set_level(logging.WARNING, logger="livekit.agents")
    # the worker exports its credentials for its job processes; monkeypatch restores them
    credentials = {
        "LIVEKIT_URL": "ws://127.0.0.1:1",
        "LIVEKIT_API_KEY": "test-key",
        "LIVEKIT_API_SECRET": "test-secret",
    }
    for name, value in credentials.items():
        monkeypatch.setenv(name, value)
    server = AgentServer(
        ws_url=credentials["LIVEKIT_URL"],
        api_key=credentials["LIVEKIT_API_KEY"],
        api_secret=credentials["LIVEKIT_API_SECRET"],
        num_idle_processes=0,
    )
    server._update_options(multiprocessing_context="forkserver")

    @server.rtc_session()
    async def entrypoint(context: JobContext) -> None:
        pass

    server._config.simulation = True
    started = asyncio.Event()
    server._on_started.append(started.set)
    running = asyncio.create_task(server._run(devmode=True, unregistered=True))
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

    async def launch(host: _ServerCallbacks, info: RunningJobInfo) -> None:
        launched.append(info)

    monkeypatch.setattr(_ServerCallbacks, "launch", launch)
    server = AgentServer(num_idle_processes=0)

    @server.rtc_session()
    async def entrypoint(context: JobContext) -> None:
        pass

    server._config.simulation = True
    started = asyncio.Event()
    server._on_started.append(started.set)
    running = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        await server._simulate_job("typed-launch", fake_job=True)
        assert len(launched) == 1
        assert launched[0].fake_job
        assert launched[0].job.room.name == "typed-launch"
    finally:
        await server.aclose()
        await asyncio.wait_for(running, timeout=3)
