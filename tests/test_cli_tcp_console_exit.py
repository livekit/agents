"""A second Ctrl+C in TCP console mode must force-exit on every platform.

``_run_tcp_console`` force-exits on the second signal by killing its process
group, but ``os.killpg`` / ``os.getpgid`` only exist on POSIX. On Windows the
handler raised ``AttributeError`` instead of exiting.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest

from livekit.agents.cli import cli

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


class _StubServer:
    """Stands in for AgentServer: run() until aclose()."""

    def __init__(self, on_run: Callable[[], None]) -> None:
        self._on_run = on_run
        self._stopped: asyncio.Event | None = None
        self.closed = False

    def once(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        return lambda fnc: fnc

    async def run(self, *, devmode: bool, unregistered: bool) -> None:
        self._stopped = asyncio.Event()
        self._on_run()
        await self._stopped.wait()

    async def aclose(self) -> None:
        self.closed = True
        if self._stopped is not None:
            self._stopped.set()


def test_second_signal_force_exits_without_process_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    # no process-group API, as on Windows
    monkeypatch.delattr(os, "killpg", raising=False)
    monkeypatch.delattr(os, "getpgid", raising=False)

    exit_codes: list[int] = []

    def _press_ctrl_c_twice() -> None:
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)  # first: graceful shutdown
        handler(signal.SIGINT, None)  # second: force exit

    server = _StubServer(on_run=_press_ctrl_c_twice)
    saved = {sig: signal.getsignal(sig) for sig in cli.HANDLED_SIGNALS}
    try:
        with (
            patch.object(cli, "setup_logging"),
            patch.object(cli.os, "_exit", side_effect=exit_codes.append),
        ):
            cli._run_tcp_console(server=server, connect_addr="127.0.0.1:0")  # type: ignore[arg-type]
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)
        asyncio.set_event_loop(None)
        cli.AgentsConsole._instance = None

    assert exit_codes == [1]
    assert server.closed
