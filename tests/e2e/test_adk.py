"""The same frontend reaches an independently installed ADK agent over real A2A."""

import os

import pytest

from livekit.agents.a2a import TextPart

from .framework.app import E2E


async def test_a_remote_tool_runs_an_adk_agent_and_its_tool(e2e: E2E) -> None:
    python = os.environ.get("ADK_E2E_PYTHON")
    if python is None:
        pytest.skip("ADK_E2E_PYTHON must select the isolated ADK peer environment")
    peer = await e2e.start_app("tests.e2e.apps.adk_peer", python=python)
    desk = await e2e.start_app("tests.e2e.apps.travel_frontdesk", env={"REMOTE_URL": peer.url})
    async with e2e.text_client(desk) as client:
        reply = await client.send("Check flights to Rome.")
        updates = [update async for update in reply]
    assert updates[-1].state == "completed"
    text = "".join(
        part.text
        for artifact in updates[-1].artifacts
        for part in artifact.parts
        if isinstance(part, TextPart)
    )
    assert text == "Seats are available to Rome."
