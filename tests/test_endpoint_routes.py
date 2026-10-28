"""The framework serves conversation routes; application HTTP routes stay application owned."""

from __future__ import annotations

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from livekit.agents import AgentServer
from livekit.agents.a2a import A2ASessionContext, Credentials
from livekit.agents.storage import AgentDBStorage

pytestmark = pytest.mark.unit


async def test_endpoint_route_inventory() -> None:
    key, secret = "APIroute", "route-inventory-secret-secret-secret"
    storage = AgentDBStorage(url="wss://agents.invalid", api_key=key, api_secret=secret)
    server = AgentServer(storage=storage, agent_name="routes", api_key=key, api_secret=secret)

    @server.a2a_session(endpoint="desk", description="Answers routing test messages.")
    async def desk(ctx: A2ASessionContext) -> str:
        return "received: " + ctx.message.text_content

    @server.http.get("/application-content")
    async def application_content(_: Request) -> PlainTextResponse:
        return PlainTextResponse("application owned")

    token = Credentials(key, secret).project_token(identity="route-test")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.http.app),
        base_url="http://agents.test",
        headers={"Authorization": f"Bearer {token}"},
    ) as client:
        card = await client.get("/desk/.well-known/agent-card.json")
        assert card.status_code == 200
        assert card.json()["name"] == "desk"
        for method in ["message/send", "message/stream"]:
            response = await client.post(
                "/desk/v1",
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
            assert (await client.get(path)).status_code == 404, path
        response = await client.get("/application-content")
        assert response.status_code == 200
        assert response.text == "application owned"
