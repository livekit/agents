"""The framework serves conversation routes; application HTTP routes stay application owned."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from livekit.agents import AgentServer
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import SQLiteStorage

from .local_server import serving

pytestmark = pytest.mark.unit


async def test_endpoint_route_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LIVEKIT_URL", "LIVEKIT_AGENT_NAME_OVERRIDE", "LIVEKIT_AGENT_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    server = AgentServer(storage=SQLiteStorage(directory=tmp_path), agent_name="routes")

    @server.a2a_session(endpoint="desk", description="Answers routing test messages.")
    async def desk(ctx: A2ASessionContext) -> str:
        return "received: " + ctx.message.text_content

    @server._http.get("/application-content")
    async def application_content(_: Request) -> PlainTextResponse:
        return PlainTextResponse("application owned")

    async with serving(server) as base, httpx.AsyncClient(base_url=base) as client:
        card = await client.get(f"{base}/desk/.well-known/agent-card.json")
        assert card.status_code == 200
        assert card.json()["name"] == "desk"
        for method in ["message/send", "message/stream"]:
            response = await client.post(
                f"{base}/desk/v1",
                json={
                    "jsonrpc": "2.0",
                    "id": method,
                    "method": method,
                    "params": "invalid",
                },
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("application/json")
            assert response.json()["error"]["code"] == -32602
        for path in [
            "/content",
            "/history",
            "/tasks",
            "/threads/AT_example/history",
            "/desk/content",
            "/desk/history",
            "/desk/tasks",
        ]:
            assert (await client.get(f"{base}{path}")).status_code == 404, path
        response = await client.get(f"{base}/application-content")
        assert response.status_code == 200
        assert response.text == "application owned"
