"""Remote SDK content survives the native client, tool result and inspection boundaries."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from google.protobuf.json_format import MessageToDict

from livekit.agents import Agent
from livekit.agents.a2a import RemoteAgent
from livekit.agents.storage import ContextStore
from tests.fakes import FakeLLM, tool_call

from .a2a_peer import Peer


@pytest.mark.parametrize("shape", ["message", "task"])
async def test_remote_files_and_data_reach_the_model_without_losing_the_original_sdk_bytes(
    peer: Peer, shape: str
) -> None:
    raw = b"the original PDF bytes"
    encoded = base64.b64encode(raw).decode()
    parts: list[dict[str, Any]] = [
        {"text": "Receipt attached."},
        {"data": {"receipt": 42}},
        {"raw": encoded, "filename": "receipt.pdf", "mediaType": "application/pdf"},
    ]
    direct: dict[str, Any] = {
        "messageId": "receipt-message",
        "role": "ROLE_AGENT",
        "parts": parts,
        "metadata": {"vendor": {"reference": "R42"}},
    }
    if shape == "message":
        peer.direct_reply = direct
    else:
        peer.events = lambda _: [
            {
                "task": {
                    "id": "receipt-task",
                    "contextId": "trip",
                    "status": {"state": "TASK_STATE_COMPLETED"},
                    "artifacts": [
                        {
                            "artifactId": "receipt-artifact",
                            "name": "answer",
                            "parts": parts,
                            "extensions": ["https://vendor.example.test/receipt"],
                        }
                    ],
                }
            }
        ]
    model = FakeLLM([tool_call("receipts", request="Get the receipt"), "Your receipt is ready."])
    agent = Agent(
        name="desk",
        instructions="Help with receipts.",
        llm=model,
        tools=[
            RemoteAgent(
                peer.url, id="receipts", name="receipts", description="Retrieve receipts."
            ).as_tool()
        ],
    )
    store = ContextStore()
    result = await agent.run("Get the receipt", context=store.bind())
    assert result.status == "completed"
    outputs = [item for item in result.new_messages() if item.type == "function_call_output"]
    assert len(outputs) == 1 and not outputs[0].is_error
    material = json.loads(outputs[0].output)
    model_parts = material["parts"] if shape == "message" else material[0]["parts"]
    assert model_parts[0] == parts[0]
    assert model_parts[1] == parts[1]
    assert model_parts[2]["filename"] == "receipt.pdf"
    assert model_parts[2]["mediaType"] == "application/pdf"
    assert "raw" not in model_parts[2]
    assert f"{len(raw)} bytes retained" in model_parts[2]["text"]
    assert "have not been read" in model_parts[2]["text"]
    assert len(model.requests) == 2, "rendering content cannot retry completed business work"
    assert len(peer.messages) == 1

    state = await store.load_session()
    remote = state.remote_agents["receipts"]
    if shape == "message":
        assert not remote.tasks, "a direct SDK Message must not create a synthetic task"
        assert MessageToDict(remote.messages["receipt-message"]) == direct
    else:
        observed = MessageToDict(remote.tasks["receipt-task"].task)
        assert observed["artifacts"][0]["parts"] == parts
        assert observed["artifacts"][0]["extensions"] == ["https://vendor.example.test/receipt"]
