"""Surviving Python bootstrap contracts (G-8 and G-9)."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import textwrap
from collections.abc import Coroutine
from typing import Any
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from livekit import api
from livekit.agents import AgentServer, cli
from livekit.agents._generated.cli import CliArgsStart
from livekit.protocol import models

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]


@pytest.mark.parametrize("failure", ["startup", "drain"])
def test_launcher_failure_exits_nonzero_after_cleanup(failure: str) -> None:
    code = """
        import sys
        from livekit.agents import AgentServer, cli

        class FailingServer(AgentServer):
            async def _run(self, **kwargs):
                if FAILURE == "startup":
                    raise RuntimeError("startup refused")
            async def _drain(self):
                print("drained")
                if FAILURE == "drain":
                    raise RuntimeError("drain refused")
            async def aclose(self):
                print("closed")

        server = FailingServer()
        sys.argv = ["agent.py", "start"]
    """
    code = f"FAILURE = {failure!r}\n" + textwrap.dedent(code)
    code += "\ncli.run_app(server)\n"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert f"{failure} refused" in result.stdout + result.stderr
    assert "drained" in result.stdout and "closed" in result.stdout
    assert result.returncode != 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "arguments, status", [("--help", 0), ("start --help", 0), ("download-files", 2)]
)
def test_cli_diagnostics_do_not_run_the_server(arguments: str, status: int) -> None:
    code = textwrap.dedent("""
        import sys
        from livekit.agents import AgentServer, cli

        def fail(*args):
            raise AssertionError("server must not be loaded")

        cli._run_worker = fail
        cli._run_console = fail
    """)
    code += f"sys.argv = ['agent.py', *{arguments!r}.split()]\n"
    code += "cli.run_app(AgentServer())\n"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == status, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    if status == 0:
        assert "Usage:" in result.stdout
    else:
        assert "download-files" in result.stderr


@pytest.mark.parametrize("credentials", ["explicit", "environment", "join_tokens"])
def test_notebook_starts_and_dispatches_with_available_credentials(
    monkeypatch: pytest.MonkeyPatch, credentials: str
) -> None:
    monkeypatch.setitem(sys.modules, "livekit.rtc.jupyter", Mock(display_room=lambda *_: None))
    from livekit.agents import jupyter

    monkeypatch.setattr("nest_asyncio.apply", lambda: None)
    monkeypatch.setattr(jupyter, "display_room", lambda *_: None)
    for key in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    url, key, secret = "ws://localhost:7880", "notebook-key", "notebook-secret-32-bytes-for-tests"
    if credentials == "environment":
        for name, value in zip(
            ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"),
            (url, key, secret),
            strict=True,
        ):
            monkeypatch.setenv(name, value)
    server = (
        AgentServer(ws_url=url, api_key=key, api_secret=secret)
        if credentials == "explicit"
        else AgentServer()
    )
    simulate = AsyncMock()
    monkeypatch.setattr(server, "_simulate_job", simulate)
    room_api = AsyncMock()

    async def create_room(request: api.CreateRoomRequest) -> models.Room:
        return models.Room(name=request.name, sid="RM_notebook")

    room_api.room.create_room.side_effect = create_room
    room_api.__aenter__.return_value = room_api
    api_client = Mock(return_value=room_api)
    # LiveKitAPI is a synchronous factory returning an async context manager.
    monkeypatch.setattr(api, "LiveKitAPI", api_client)

    agent_token = (
        api.AccessToken(key, secret)
        .with_identity("agent-jupyter")
        .with_grants(api.VideoGrants(room_join=True, room="notebook-room"))
        .to_jwt()
    )
    response = AsyncMock()
    response.json.return_value = {
        "livekit_url": url,
        "user_token": "user-token",
        "agent_token": agent_token,
    }
    response.__aenter__.return_value = response
    session = AsyncMock()
    session.__aenter__.return_value = session
    monkeypatch.setattr(session, "post", lambda _: response)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda: session)

    pending: list[Coroutine[Any, Any, None]] = []
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coro, _: pending.append(coro))

    def start_worker(
        server: AgentServer, args: CliArgsStart, *, unregistered: bool = False
    ) -> None:
        assert unregistered, "notebooks with join tokens cannot register as workers"
        server._update_options(
            ws_url=args.ws_url or "", api_key=args.api_key or "", api_secret=args.api_secret or ""
        )
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for callback in server._on_started:
                callback()
            for coro in pending:
                loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    monkeypatch.setattr(cli, "_run_worker", start_worker)
    # Notebook logging setup must not remove pytest's own stream handlers.
    monkeypatch.setattr(logging, "getLogger", lambda: logging.Logger("notebook-test"))
    jupyter.run_app(
        server, jupyter_url="https://join.invalid" if credentials == "join_tokens" else None
    )
    simulate.assert_awaited_once()
    call = simulate.await_args
    assert call is not None
    assert call.kwargs["room_info"].name == call.args[0]
    claims = api.TokenVerifier(key, secret).verify(call.kwargs["token"])
    assert claims.video is not None and claims.video.room == call.args[0]
    assert server._config.ws_url == url
    if credentials == "join_tokens":
        assert not api_client.mock_calls, "join tokens do not authorize admin room creation"
    else:
        api_client.assert_called_once_with(url, key, secret)
