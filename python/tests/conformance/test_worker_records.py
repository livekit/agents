"""Typed server updates and CLI command dispatch through the generated native boundary."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

import pytest

from livekit.agents import AgentServer, JobContext, _core, cli
from livekit.agents._generated.cli import CliArgsConsole, CliArgsStart
from livekit.agents._generated.worker_options import ProcessOptions
from livekit.agents.job import JobExecutorType
from livekit.agents.types import NOT_GIVEN

pytestmark = pytest.mark.unit


def test_cli_generated_args_matrix_dispatches_every_command_and_preserves_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = AgentServer()
    calls: list[tuple[str, Any]] = []

    def run_worker(server: AgentServer, args: CliArgsStart) -> int:
        calls.append(
            (
                "start",
                (
                    args.options.ws_url,
                    args.options.api_key,
                    args.options.api_secret,
                    args.options.shutdown_timeout,
                    args.options.drain_timeout,
                ),
            )
        )
        return 17

    def run_console(server: AgentServer, connect_addr: str, record: bool) -> None:
        calls.append(("console", (connect_addr, record)))

    monkeypatch.setattr(cli, "_run_worker", run_worker)
    monkeypatch.setattr(cli, "_run_console", run_console)
    assert (
        cli._launch(["start", "--dev", "--simulation", "--url", "wss://cli.example"], server) == 17
    )
    assert calls.pop() == (
        "start",
        ("wss://cli.example", NOT_GIVEN, NOT_GIVEN, NOT_GIVEN, NOT_GIVEN),
    )
    assert cli._launch(["console", "--connect-addr", "127.0.0.1:9", "--record"], server) == 0
    assert calls.pop() == ("console", ("127.0.0.1:9", True))
    console = _core.parse_cli(["agent", "console", "--connect-addr", "127.0.0.1:9"]).args
    assert isinstance(console, CliArgsConsole)
    monkeypatch.setattr(sys, "argv", ["app.py", "start"])
    with pytest.raises(SystemExit) as exited:
        cli.run_app(server)
    assert exited.value.code == 17


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
    server._update_options(ProcessOptions(multiprocessing_context="forkserver"))

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


async def test_native_simulated_job_reaches_the_entrypoint() -> None:
    loop = asyncio.get_running_loop()
    seen: asyncio.Future[tuple[str, bool]] = loop.create_future()
    server = AgentServer(num_idle_processes=0)
    server._update_options(ProcessOptions(job_executor_type=JobExecutorType.THREAD))

    @server.rtc_session()
    async def entrypoint(context: JobContext) -> None:
        job = (context.job.room.name, context.is_fake_job())
        loop.call_soon_threadsafe(seen.set_result, job)

    server._config.simulation = True
    started = asyncio.Event()
    server._on_started.append(started.set)
    running = asyncio.create_task(server._run(devmode=True, unregistered=True))
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        await server._simulate_job("typed-launch", fake_job=True)
        assert await asyncio.wait_for(seen, timeout=10) == ("typed-launch", True)
    finally:
        await server.aclose()
        await asyncio.wait_for(running, timeout=3)
