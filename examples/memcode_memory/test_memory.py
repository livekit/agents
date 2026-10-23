import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from memcode_sdk import MemcodeSDKError

from livekit.agents import llm

from .agent import ReturningCallerAgent
from .memory_service import MemoryService, MemoryUnavailable

pytestmark = pytest.mark.unit


def record(user, space, content):
    return SimpleNamespace(
        content=content, space=SimpleNamespace(id=space), metadata={"user_id": user}
    )


@pytest.mark.asyncio
async def test_two_calls_share_only_approved_same_user_facts():
    client = SimpleNamespace(
        search=AsyncMock(
            return_value=SimpleNamespace(
                results=[
                    record("alice", "space-a", "Use concise replies"),
                    record("bob", "space-a", "private-bob"),
                    record("alice", "space-b", "private-space"),
                ]
            )
        ),
        ingest=AsyncMock(return_value=SimpleNamespace(id="job", status="queued")),
    )
    first = MemoryService(client, "space-a", "alice", "actor-a")
    await first.save_approved_fact("Use concise replies")
    later = ReturningCallerAgent(MemoryService(client, "space-a", "alice", "actor-a"))
    await later.on_enter()
    ctx = llm.ChatContext()
    msg = llm.ChatMessage(role="user", content=["Hello"])
    await later.on_user_turn_completed(ctx, msg)
    text = str(ctx.to_dict())
    assert "Use concise replies" in text
    assert "private-bob" not in text and "private-space" not in text
    before = len(ctx.items)
    await later.on_user_turn_completed(ctx, msg)
    assert len(ctx.items) == before
    fresh_turn = llm.ChatContext()
    await later.on_user_turn_completed(fresh_turn, msg)
    assert "Use concise replies" in str(fresh_turn.to_dict())
    assert client.search.call_count == 1
    assert client.search.call_args.kwargs["scope"] == "context_only"
    assert client.ingest.call_args.kwargs["metadata"]["user_id"] == "alice"


@pytest.mark.asyncio
async def test_greeting_is_not_blocked_and_exit_cancels_prefetch():
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    service = MemoryService(SimpleNamespace(search=slow), "s", "u", "a", timeout=10)
    agent = ReturningCallerAgent(service)
    await asyncio.wait_for(agent.on_enter(), timeout=0.1)
    await started.wait()
    await agent.on_exit()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_timeout_and_failure_do_not_break_the_turn_or_expose_body():
    client = SimpleNamespace(search=AsyncMock(side_effect=MemcodeSDKError("secret prompt")))
    service = MemoryService(client, "s", "u", "a")
    with pytest.raises(MemoryUnavailable, match="temporarily unavailable") as error:
        await service.recall("preferences")
    assert "secret" not in str(error.value)
    agent = ReturningCallerAgent(service)
    await agent.on_enter()
    ctx = llm.ChatContext()
    await agent.on_user_turn_completed(ctx, llm.ChatMessage(role="user", content=["Hi"]))
    assert not ctx.items


@pytest.mark.asyncio
async def test_explicit_writes_are_idempotent_and_not_triggered_by_turns():
    client = SimpleNamespace(
        ingest=AsyncMock(), search=AsyncMock(return_value=SimpleNamespace(results=[]))
    )
    service = MemoryService(client, "s", "u", "a")
    await service.save_approved_fact("approved")
    key = client.ingest.call_args.kwargs["idempotency_key"]
    await service.save_approved_fact("approved")
    assert key == client.ingest.call_args.kwargs["idempotency_key"]
    agent = ReturningCallerAgent(service)
    await agent.on_enter()
    await agent.on_user_turn_completed(
        llm.ChatContext(), llm.ChatMessage(role="user", content=["private transcript"])
    )
    assert client.ingest.call_count == 2
    with pytest.raises(ValueError):
        await service.save_approved_fact(" ")
