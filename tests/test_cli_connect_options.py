from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from livekit.agents.cli._legacy import _build_cli
from livekit.agents.worker import AgentServer, ServerOptions

pytestmark = pytest.mark.unit


def _make_server(**kwargs) -> AgentServer:
    async def _fake_entrypoint(ctx):
        pass

    return AgentServer.from_server_options(ServerOptions(entrypoint_fnc=_fake_entrypoint, **kwargs))


@pytest.fixture
def runner():
    return CliRunner()


class _FakeRoomService:
    async def list_rooms(self, request):
        return SimpleNamespace(rooms=[SimpleNamespace(name=request.names[0])])

    async def create_room(self, request):
        return SimpleNamespace(name=request.name)


class _FakeLiveKitAPI:
    def __init__(self, url, api_key, api_secret):
        self.room = _FakeRoomService()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def test_connect_applies_cli_connection_options_before_simulate_job(monkeypatch, runner):
    """connect must feed --url/--api-key/--api-secret into the AgentServer, as
    start/dev do via _run_worker, so simulate_job mints the job token and join
    URL from those credentials rather than from construction-time/env values."""
    for var in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_LOG_LEVEL"):
        monkeypatch.delenv(var, raising=False)

    server = _make_server()
    simulated: list[tuple[str, str, str]] = []
    observed = asyncio.Event()

    async def fake_simulate_job(*, room, fake_job, room_info, agent_identity):
        simulated.append((server._ws_url, server._api_key, server._api_secret))
        observed.set()

    async def fake_run(*, devmode: bool, unregistered: bool) -> None:
        server.emit("worker_started")
        await asyncio.wait_for(observed.wait(), timeout=1.0)

    monkeypatch.setattr(server, "simulate_job", fake_simulate_job)
    monkeypatch.setattr(server, "run", fake_run)

    with patch("livekit.agents.cli._legacy.api.LiveKitAPI", _FakeLiveKitAPI):
        app = _build_cli(server)
        result = runner.invoke(
            app,
            [
                "connect",
                "--room",
                "myroom",
                "--url",
                "ws://flag-host:7880",
                "--api-key",
                "cli-key",
                "--api-secret",
                "cli-secret",
                "--participant-identity",
                "agent-x",
            ],
        )

    assert result.exit_code == 0, result.output
    assert simulated == [("ws://flag-host:7880", "cli-key", "cli-secret")], (
        "connect built the job connection before applying --url/--api-key/--api-secret"
    )
