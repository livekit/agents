"""Delegation to the stock SDK over real HTTP/SSE, without LiveKit extension negotiation."""

import base64
import json

from a2a.types import a2a_pb2 as pb

from livekit.agents.a2a import TaskUpdate, TextPart

from .apps.stock_a2a import ITINERARY, ITINERARY_PDF, ITINERARY_REQUEST, NOTE, TICKET_URL
from .framework.app import E2E


def text(update: TaskUpdate) -> str:
    return "".join(
        part.text
        for artifact in update.artifacts
        for part in artifact.parts
        if isinstance(part, TextPart)
    )


async def test_a_remote_tool_accepts_a_stock_message_only_reply(e2e: E2E) -> None:
    peer = await e2e.start_app("tests.e2e.apps.stock_a2a")
    desk = await e2e.start_app("tests.e2e.apps.travel_frontdesk", env={"REMOTE_URL": peer.url})
    async with e2e.text_client(desk) as client:
        reply = await client.send("hello")
        updates = [update async for update in reply]
    assert updates[-1].state == pb.TASK_STATE_COMPLETED
    assert text(updates[-1]) == "Hello from the stock A2A peer."


async def test_a_stock_input_required_task_resumes_after_frontend_restart(e2e: E2E) -> None:
    peer = await e2e.start_app("tests.e2e.apps.stock_a2a")
    desk = await e2e.start_app("tests.e2e.apps.travel_frontdesk", env={"REMOTE_URL": peer.url})
    async with e2e.text_client(desk) as client:
        reply = await client.send("book a flight")
        updates = [update async for update in reply]
        thread = reply.thread_id
    assert updates[-1].state == pb.TASK_STATE_INPUT_REQUIRED
    assert "Which day?" in updates[-1].text
    assert [update.text for update in updates if update.progress] == ["Checking available dates."]
    assert all("draft answer" not in update.text for update in updates)

    await desk.restart()
    async with e2e.text_client(desk) as client:
        reply = await client.send("Why do you need the date?", thread_id=thread)
        explanation = [update async for update in reply]
    assert explanation[-1].state == pb.TASK_STATE_COMPLETED
    assert text(explanation[-1]) == "The travel agent needs it to check availability."

    await desk.restart()
    async with e2e.text_client(desk) as client:
        reply = await client.send("Friday", thread_id=thread)
        updates = [update async for update in reply]
    assert updates[-1].state == pb.TASK_STATE_COMPLETED
    assert " ".join(text(updates[-1]).split()) == "Booked for Friday."


async def test_stock_data_file_parts_and_partial_history_survive_a_store_reload(
    e2e: E2E,
) -> None:
    """Catches saved remote state that flattens SDK parts or rewrites the peer's task history."""
    peer = await e2e.start_app("tests.e2e.apps.stock_a2a")
    artifacts = e2e.directory / "desk"
    artifacts.mkdir()
    desk = await e2e.start_app(
        "tests.e2e.apps.travel_frontdesk",
        env={"REMOTE_URL": peer.url, "E2E_ARTIFACT_DIR": str(artifacts)},
    )
    async with e2e.text_client(desk) as client:
        reply = await client.send(ITINERARY_REQUEST)
        updates = [update async for update in reply]
        thread = reply.thread_id
    assert updates[-1].state == pb.TASK_STATE_COMPLETED

    # A new process loads the conversation from storage and reports it before running.
    await desk.restart()
    async with e2e.text_client(desk) as client:
        reply = await client.send("hello", thread_id=thread)
        assert [update async for update in reply][-1].state == pb.TASK_STATE_COMPLETED
    [remote] = json.loads((artifacts / "live-model-report.json").read_text())[
        "remote_agents"
    ].values()
    [task] = remote["tasks"].values()
    assert task["task"]["status"]["state"] == "TASK_STATE_COMPLETED"
    [artifact] = task["task"]["artifacts"]
    assert artifact["artifactId"] == "itinerary" and artifact["name"] == "Itinerary"
    data, raw, link = artifact["parts"]
    assert data == {
        "data": ITINERARY,
        "metadata": {"schema": "itinerary/v1"},
        "mediaType": "application/json",
    }
    assert base64.b64decode(raw.pop("raw")) == ITINERARY_PDF
    assert raw == {"filename": "itinerary.pdf", "mediaType": "application/pdf"}
    assert link == {"url": TICKET_URL, "filename": "ticket.pdf", "mediaType": "application/pdf"}
    # The peer omitted the request from its task history; the saved task keeps it omitted.
    [note] = task["task"]["history"]
    assert note["messageId"] == "itinerary-note" and note["parts"] == [{"text": NOTE}]
    request = next(
        message for message in remote["messages"].values() if message["role"] == "ROLE_USER"
    )
    assert request["parts"] == [{"text": ITINERARY_REQUEST}]
    assert remote["messages"]["itinerary-note"] == note
    assert [event["kind"] for event in remote["events"]] == [
        "request_recorded",
        "response_received",
        "response_received",
        "response_received",
    ]
    assert remote["events"][2]["payload"]["artifactUpdate"]["artifact"]["parts"][0] == data
