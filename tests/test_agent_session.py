import asyncio

import pytest

from livekit.agents import Agent, VoiceSession
from livekit.agents.llm import ChatContext

pytestmark = pytest.mark.unit


async def test_agent_switch_preserves_the_conversation_history() -> None:
    history = ChatContext.empty()
    history.add_message(role="user", content="My booking reference is ABC123.")
    first = Agent(instructions="Welcome the caller.", name="reception")
    second = Agent(instructions="Help with bookings.", name="bookings")
    async with VoiceSession(vad=None, turn_handling={"turn_detection": None}) as session:
        await session.start(first, chat_ctx=history)
        session.update_agent(second)
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        messages = [item.text_content for item in session.history.items if item.type == "message"]
        assert "My booking reference is ABC123." in messages
        assert session.current_agent is second


async def test_agent_switch_preserves_session_listening_options() -> None:
    async with VoiceSession(
        vad=None,
        turn_detection="manual",
        endpointing_opts={"min_delay": 0.3, "max_delay": 2.0},
    ) as session:
        await session.start(Agent(instructions="Welcome the caller.", name="reception"))
        session.update_agent(Agent(instructions="Help with bookings.", name="bookings"))
        await asyncio.wait_for(session.wait_for_idle(), timeout=5)
        assert session.options["turn_detection_mode"] == "manual"
        assert session.options["endpointing"]["min_delay"] == 0.3


async def test_temporary_options_restore_the_base_after_an_exception() -> None:
    session = VoiceSession(vad=None, turn_handling={"turn_detection": None})
    try:
        with pytest.raises(RuntimeError, match="interaction failed"):
            async with session.override_options(turn_detection="manual"):
                session.update_options(turn_detection="vad")
                raise RuntimeError("interaction failed")
        assert session.options["turn_detection_mode"] == "vad"
    finally:
        await session.aclose()
