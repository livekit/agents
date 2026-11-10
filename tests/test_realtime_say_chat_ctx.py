"""Text spoken with session.say() reaches the realtime model's chat context when it is spoken.

Without this, the realtime history only learns about the message on the next bulk sync, which
can insert it after items that were said later (#7916).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from livekit.agents import Agent, AgentSession, llm

from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mutable_chat_context", [True, False])
async def test_say_adds_spoken_message_to_realtime_chat_ctx(mutable_chat_context: bool) -> None:
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            supports_say=False, mutable_chat_context=mutable_chat_context
        )
    )
    agent = Agent(instructions="test")

    async with AgentSession(llm=model) as session:
        await session.start(agent)
        await session.say("Thanks for joining! Are you ready?")

        local_msgs = [item for item in agent.chat_ctx.items if item.type == "message"]
        remote_msgs = [
            item for item in model.active_session.chat_ctx.items if item.type == "message"
        ]

    assert [msg.text_content for msg in local_msgs] == ["Thanks for joining! Are you ready?"]
    if mutable_chat_context:
        assert [msg.id for msg in remote_msgs] == [local_msgs[0].id]
    else:
        assert remote_msgs == []


async def test_say_lets_next_speech_start_while_chat_ctx_update_is_pending() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_say=False))
    agent = Agent(instructions="test")

    async with AgentSession(llm=model) as session:
        await session.start(agent)
        rt_session = model.active_session
        release_update = asyncio.Event()
        apply_update = rt_session.update_chat_ctx

        async def unacknowledged_update(chat_ctx: llm.ChatContext) -> None:
            await release_update.wait()
            await apply_update(chat_ctx)

        rt_session.update_chat_ctx = unacknowledged_update  # type: ignore[method-assign]

        first = session.say("first")
        second = session.say("second")

        async def second_spoken() -> None:
            while not any(
                isinstance(item, llm.ChatMessage) and item.text_content == "second"
                for item in agent.chat_ctx.items
            ):
                await asyncio.sleep(0.01)

        await asyncio.wait_for(second_spoken(), timeout=5)
        assert not first.done()

        release_update.set()
        await first
        await second


async def test_say_places_spoken_message_before_later_remote_items() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(supports_say=False))
    agent = Agent(instructions="test")

    async with AgentSession(llm=model) as session:
        await session.start(agent)
        # a user turn the server added while the message was still playing
        remote_ctx = model.active_session.chat_ctx.copy()
        user_msg = remote_ctx.add_message(role="user", content="yes", created_at=time.time() + 60)
        await model.active_session.update_chat_ctx(remote_ctx)

        await session.say("Are you ready?")

        remote_ids = [
            item.id for item in model.active_session.chat_ctx.items if item.type == "message"
        ]
        spoken = next(
            item
            for item in agent.chat_ctx.items
            if isinstance(item, llm.ChatMessage) and item.text_content == "Are you ready?"
        )

    assert remote_ids == [spoken.id, user_msg.id]
