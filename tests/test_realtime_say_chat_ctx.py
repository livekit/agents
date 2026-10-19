"""Text spoken with session.say() reaches the realtime model's chat context when it is spoken.

Without this, the realtime history only learns about the message on the next bulk sync, which
can insert it after items that were said later (#7916).
"""

from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession

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
