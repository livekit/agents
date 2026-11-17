"""Delegation to the stock SDK over real HTTP/SSE, without LiveKit extension negotiation."""

from livekit.agents.a2a import TaskUpdate, TextPart

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
    assert updates[-1].state == "completed"
    assert text(updates[-1]) == "Hello from the stock A2A peer."


async def test_a_stock_input_required_task_resumes_after_frontend_restart(e2e: E2E) -> None:
    peer = await e2e.start_app("tests.e2e.apps.stock_a2a")
    desk = await e2e.start_app("tests.e2e.apps.travel_frontdesk", env={"REMOTE_URL": peer.url})
    async with e2e.text_client(desk) as client:
        reply = await client.send("book a flight")
        updates = [update async for update in reply]
        thread = reply.thread_id
    assert updates[-1].state == "input-required"
    assert "Which day?" in updates[-1].text
    assert [update.text for update in updates if update.progress] == ["Checking available dates."]
    assert all("draft answer" not in update.text for update in updates)

    await desk.restart()
    async with e2e.text_client(desk) as client:
        reply = await client.send("Friday", thread_id=thread)
        updates = [update async for update in reply]
    assert updates[-1].state == "completed"
    assert " ".join(text(updates[-1]).split()) == "Booked for Friday."
