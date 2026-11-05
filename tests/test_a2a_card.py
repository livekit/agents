"""The published card identifies the registered endpoint and its deployment."""

from pathlib import Path

import httpx
import pytest

from livekit.agents import AgentServer
from livekit.agents.a2a import A2ASessionContext, Credentials
from livekit.agents.storage import SQLiteStorage

from .local_server import serving

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "override,deployment,expected",
    [(None, None, "local-42"), (None, "staging", "staging"), ("2.4.8", "staging", "2.4.8")],
)
async def test_card_publishes_registered_endpoint_identity_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str | None,
    deployment: str | None,
    expected: str,
) -> None:
    """docs/protocol/a2a.md: the card's version is the deployment's, or ``version=``; a local
    worker with no deployment falls back to ``LIVEKIT_AGENT_VERSION``."""
    for name in ("LIVEKIT_URL", "LIVEKIT_AGENT_NAME_OVERRIDE", "LIVEKIT_AGENT_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    if deployment:
        monkeypatch.setenv("LIVEKIT_AGENT_DEPLOYMENT", deployment)
    monkeypatch.setenv("LIVEKIT_AGENT_VERSION", "local-42")
    credentials = Credentials("card-key", "a-secret-of-at-least-thirty-two-bytes")
    storage = SQLiteStorage(directory=str(tmp_path))
    server = AgentServer(
        storage=storage,
        agent_name="worker-name",
        api_key="card-key",
        api_secret="a-secret-of-at-least-thirty-two-bytes",
    )
    description = "Books a flight and returns the reservation code."

    @server.a2a_session(endpoint="reservations", description=description, version=override)
    async def reservations(ctx: A2ASessionContext) -> str:
        return "ok"

    async with serving(server) as base, httpx.AsyncClient() as client:
        response = await client.get(
            f"{base}/reservations/.well-known/agent-card.json",
            headers={"Authorization": f"Bearer {credentials.project_token()}"},
        )
        assert response.status_code == 200
        card = response.json()
    assert card["name"] == "reservations"
    assert card["version"] == expected
    assert card["skills"] == [
        {
            "id": "delegate",
            "name": "delegate",
            "description": description,
            "tags": ["delegation"],
        }
    ]
    assert card["supportedInterfaces"] == [
        {
            "url": f"{base}/reservations/v1",
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
    ]
    assert "livekit" in card["securitySchemes"] and card["securityRequirements"]
    await storage.aclose()
