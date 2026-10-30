"""Regression tests for LIVEKIT_DEV_MODE in the thin CLI.

The legacy CLI exports ``LIVEKIT_DEV_MODE=1`` for the ``console`` and ``dev``
commands (see ``cli/_legacy.py``'s typer callback). The thin CLI
(``python -m livekit.agents``) drives the same modes, but only forwarded
``--dev`` to ``AgentServer.run()``: ``utils.is_dev_mode()`` stayed False inside
the worker, so adaptive interruption was disabled by default and
``inference.TurnDetector()`` resolved to the local v1-mini model.

Two follow-ups are covered as well:

* the variable must be set before the entrypoint module is imported, since
  module-level code may consult ``is_dev_mode()``;
* the override must be scoped to the runner, so a later production run in the
  same process (direct ``_run_worker`` calls from the legacy CLI or Jupyter)
  does not inherit dev mode.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents import utils
from livekit.agents.__main__ import main
from livekit.agents.cli import proto
from livekit.agents.cli.cli import _run_tcp_console, _run_worker

pytestmark = pytest.mark.unit


class _StubServer:
    """Records what ``server.run`` observes; the real run blocks forever."""

    def __init__(self) -> None:
        self.devmode: bool | None = None
        self.dev_mode: bool | None = None
        self.env_dev_mode: str | None = None

    async def run(self, *, devmode: bool, unregistered: bool) -> None:
        self.devmode = devmode
        self.dev_mode = utils.is_dev_mode()
        # the value a job process would inherit when spawned
        self.env_dev_mode = os.environ.get("LIVEKIT_DEV_MODE")

    async def drain(self) -> None:
        pass

    async def aclose(self) -> None:
        pass

    def once(self, event: str) -> Any:
        def _register(fn: Any) -> Any:
            return fn

        return _register


class _FailingCloseServer(_StubServer):
    async def aclose(self) -> None:
        raise RuntimeError("close failed")


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


_AGENT_MODULE = """\
from livekit.agents import utils
from livekit.agents.worker import AgentServer, ServerOptions

IMPORT_DEV_MODE = utils.is_dev_mode()


async def _entrypoint(ctx):
    pass


server = AgentServer.from_server_options(ServerOptions(entrypoint_fnc=_entrypoint))
"""


def _invoke_main(
    tmp_path: Path, module_name: str, argv: list[str], runner: str
) -> tuple[bool, str | None]:
    """Run ``main`` over a probe entrypoint and report what it saw at import."""
    path = tmp_path / f"{module_name}.py"
    path.write_text(_AGENT_MODULE)
    try:
        with (
            _cli_test_env(),
            patch(f"livekit.agents.cli.cli.{runner}", side_effect=lambda *a, **k: None),
        ):
            assert main([argv[0], str(path), *argv[1:]]) == 0
            import_dev_mode = bool(importlib.import_module(module_name).IMPORT_DEV_MODE)
            env_after = os.environ.get("LIVEKIT_DEV_MODE")
        return import_dev_mode, env_after
    finally:
        # entrypoint discovery prepends the module's directory to sys.path
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        sys.modules.pop(module_name, None)


def test_run_worker_dev_exports_dev_mode() -> None:
    server = _StubServer()
    with _cli_test_env(), patch("livekit.agents.cli.cli.setup_logging"):
        _run_worker(server, proto.CliArgs(log_level="INFO", dev=True))  # type: ignore[arg-type]

        # what the job process inherits from the env, not just the run() argument
        assert server.dev_mode is True
        assert server.env_dev_mode == "1"

    assert server.devmode is True


def test_run_worker_start_does_not_export_dev_mode() -> None:
    server = _StubServer()
    with _cli_test_env(), patch("livekit.agents.cli.cli.setup_logging"):
        _run_worker(server, proto.CliArgs(log_level="INFO"))  # type: ignore[arg-type]

        assert server.dev_mode is False
        assert server.env_dev_mode is None
        assert "LIVEKIT_DEV_MODE" not in os.environ

    assert server.devmode is False


def test_run_worker_restores_dev_mode_after_return() -> None:
    server = _StubServer()
    with _cli_test_env(), patch("livekit.agents.cli.cli.setup_logging"):
        _run_worker(server, proto.CliArgs(log_level="INFO", dev=True))  # type: ignore[arg-type]

        # the job processes are gone; a later run must not inherit dev mode
        assert utils.is_dev_mode() is False
        assert "LIVEKIT_DEV_MODE" not in os.environ


def test_run_worker_restores_preexisting_dev_mode_value() -> None:
    server = _StubServer()
    with (
        _cli_test_env(),
        patch("livekit.agents.cli.cli.setup_logging"),
        patch.dict(os.environ, {"LIVEKIT_DEV_MODE": "0"}),
    ):
        _run_worker(server, proto.CliArgs(log_level="INFO", dev=True))  # type: ignore[arg-type]

        assert server.dev_mode is True
        assert os.environ["LIVEKIT_DEV_MODE"] == "0"


def test_run_worker_restores_dev_mode_on_exception() -> None:
    server = _FailingCloseServer()
    with _cli_test_env(), patch("livekit.agents.cli.cli.setup_logging"):
        with pytest.raises(RuntimeError, match="close failed"):
            _run_worker(server, proto.CliArgs(log_level="INFO", dev=True))  # type: ignore[arg-type]

        assert "LIVEKIT_DEV_MODE" not in os.environ


def test_later_production_run_is_not_dev_mode() -> None:
    dev_server = _StubServer()
    prod_server = _StubServer()
    with _cli_test_env(), patch("livekit.agents.cli.cli.setup_logging"):
        _run_worker(dev_server, proto.CliArgs(log_level="INFO", dev=True))  # type: ignore[arg-type]
        _run_worker(prod_server, proto.CliArgs(log_level="INFO"))  # type: ignore[arg-type]

    assert dev_server.dev_mode is True
    assert prod_server.devmode is False
    assert prod_server.dev_mode is False


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
        assert server.env_dev_mode == "1"

        # restored once the console run ends
        assert "LIVEKIT_DEV_MODE" not in os.environ

    assert server.devmode is True


def test_main_sets_dev_mode_before_importing_entrypoint(tmp_path: Path) -> None:
    import_dev_mode, env_after = _invoke_main(
        tmp_path, "dev_mode_probe_start_dev", ["start", "--dev"], "_run_worker"
    )

    assert import_dev_mode is True
    assert env_after is None


def test_main_sets_dev_mode_before_console_imports_entrypoint(tmp_path: Path) -> None:
    import_dev_mode, env_after = _invoke_main(
        tmp_path,
        "dev_mode_probe_console",
        ["console", "--connect-addr", "127.0.0.1:9"],
        "_run_tcp_console",
    )

    assert import_dev_mode is True
    assert env_after is None


def test_main_keeps_production_imports_in_production_mode(tmp_path: Path) -> None:
    import_dev_mode, env_after = _invoke_main(
        tmp_path, "dev_mode_probe_start", ["start"], "_run_worker"
    )

    assert import_dev_mode is False
    assert env_after is None
