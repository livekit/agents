"""Regression tests for LIVEKIT_DEV_MODE in the thin CLI.

The legacy CLI exports ``LIVEKIT_DEV_MODE=1`` for the ``console`` and ``dev``
commands (see ``cli/_legacy.py``'s typer callback). The thin CLI
(``python -m livekit.agents``) drives the same modes, but only forwarded
``--dev`` to ``AgentServer.run()``: ``utils.is_dev_mode()`` stayed False inside
the worker, so adaptive interruption was disabled by default and
``inference.TurnDetector()`` resolved to the local v1-mini model.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents import utils
from livekit.agents.cli import proto
from livekit.agents.cli.cli import _run_tcp_console, _run_worker

pytestmark = pytest.mark.unit


class _StubServer:
    """Records what ``server.run`` observes; the real run blocks forever."""

    def __init__(self) -> None:
        self.devmode: bool | None = None
        self.dev_mode: bool | None = None

    async def run(self, *, devmode: bool, unregistered: bool) -> None:
        self.devmode = devmode
        self.dev_mode = utils.is_dev_mode()

    async def drain(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def once(self, event: str) -> Any:
        def _register(fn: Any) -> Any:
            return fn

        return _register


class _StubTcpTransport:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    async def close(self) -> None:
        pass


class _StubAudioInput:
    def close(self) -> None:
        pass


class _StubAudioOutput:
    def __init__(self, transport: _StubTcpTransport) -> None:
        self.transport = transport


class _StubConsole:
    def __init__(self) -> None:
        self.enabled = False
        self.record = False


@contextlib.contextmanager
def _cli_test_env() -> Iterator[None]:
    """Both entry points install process-wide signal handlers and env state."""
    saved_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    saved_dev_mode = os.environ.pop("LIVEKIT_DEV_MODE", None)
    try:
        yield
    finally:
        for sig, handler in saved_signals.items():
            signal.signal(sig, handler)
        asyncio.set_event_loop(None)
        if saved_dev_mode is None:
            os.environ.pop("LIVEKIT_DEV_MODE", None)
        else:
            os.environ["LIVEKIT_DEV_MODE"] = saved_dev_mode


def test_run_worker_dev_exports_dev_mode() -> None:
    server = _StubServer()
    with _cli_test_env(), patch("livekit.agents.cli.cli.setup_logging"):
        _run_worker(server, proto.CliArgs(log_level="INFO", dev=True))  # type: ignore[arg-type]

        # what the job process inherits from the env, not just the run() argument
        assert server.dev_mode is True
        assert os.environ["LIVEKIT_DEV_MODE"] == "1"

    assert server.devmode is True


def test_run_worker_start_does_not_export_dev_mode() -> None:
    server = _StubServer()
    with _cli_test_env(), patch("livekit.agents.cli.cli.setup_logging"):
        _run_worker(server, proto.CliArgs(log_level="INFO"))  # type: ignore[arg-type]

        assert server.dev_mode is False
        assert "LIVEKIT_DEV_MODE" not in os.environ

    assert server.devmode is False


def test_run_tcp_console_exports_dev_mode() -> None:
    server = _StubServer()
    console = _StubConsole()
    with (
        _cli_test_env(),
        patch("livekit.agents.cli.cli.setup_logging"),
        patch("livekit.agents.cli.cli.AgentsConsole.get_instance", return_value=console),
        patch("livekit.agents.voice.remote_session.TcpSessionTransport", _StubTcpTransport),
        patch("livekit.agents.cli.tcp_console.TcpAudioInput", _StubAudioInput),
        patch("livekit.agents.cli.tcp_console.TcpAudioOutput", _StubAudioOutput),
    ):
        _run_tcp_console(server=server, connect_addr="127.0.0.1:1234")  # type: ignore[arg-type]

        assert server.dev_mode is True
        assert os.environ["LIVEKIT_DEV_MODE"] == "1"

    assert server.devmode is True
