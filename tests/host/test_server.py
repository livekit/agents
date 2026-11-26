"""The storage an AgentServer picks when it is given none."""

from __future__ import annotations

import pytest

from livekit.agents import AgentServer
from livekit.agents.storage import AgentDBStorage, SQLiteStorage

CLOUD = {"ws_url": "wss://demo.livekit.cloud", "api_key": "key", "api_secret": "secret"}


@pytest.fixture(autouse=True)
def no_project_env(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVEKIT_AGENT_THREADS_DIR", str(tmp_path))


def test_a_cloud_server_configured_by_options_alone_keeps_threads_in_its_project() -> None:
    server = AgentServer(**CLOUD)

    @server.a2a_session(endpoint="support", description="Answers questions.")
    async def support(ctx: object) -> None: ...

    assert isinstance(server._ensure_storage(), AgentDBStorage)


def test_the_console_keeps_threads_locally_on_a_cloud_project() -> None:
    assert isinstance(AgentServer(**CLOUD)._ensure_storage(console=True), SQLiteStorage)
