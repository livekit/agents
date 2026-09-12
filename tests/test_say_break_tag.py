from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.tts._provider_format import strip_chat_markup
from tests.fake_llm import FakeLLM
from tests.fake_tts import FakeTTS

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_say_strips_break_tags_from_chat_ctx() -> None:
    agent = Agent(instructions="test", llm=FakeLLM(), tts=FakeTTS(fake_audio_duration=0.01))
    session = AgentSession(
        vad=None,
        turn_handling={"turn_detection": None},
    )
    await session.start(agent)
    try:
        # Standard self-closing break tag
        handle = session.say('Hello <break time="1s"/> world!')
        await handle.wait_for_playout()

        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert len(messages) == 1
        assert messages[0].text_content == "Hello world!"
        assert "<break" not in messages[0].text_content

        # Multiple and enclosing break tags
        handle2 = session.say(
            '<break time="500ms"/> Good morning! <break time="1s"/> How can I help? </break>'
        )
        await handle2.wait_for_playout()

        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert len(messages) == 2
        assert messages[1].text_content == "Good morning! How can I help?"
        assert "<break" not in messages[1].text_content

        # Message with only a break tag does not leave an empty message
        handle3 = session.say('<break time="1s"/>')
        await handle3.wait_for_playout()

        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert len(messages) == 2

        # Break tag without surrounding whitespace preserves word boundary
        handle4 = session.say('Hello<break time="1s"/>world')
        await handle4.wait_for_playout()

        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert len(messages) == 3
        assert messages[2].text_content == "Hello world"

        # SSML tags like phoneme, prosody, and say-as unwrapped
        handle5 = session.say(
            '<speak><prosody rate="fast"><phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato</phoneme></prosody></speak>'
        )
        await handle5.wait_for_playout()

        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert len(messages) == 4
        assert messages[3].text_content == "tomato"
    finally:
        await session.aclose()


def test_strip_chat_markup_whitespace_only() -> None:
    """Whitespace-only text should return empty string."""
    assert strip_chat_markup("  ") == ""
    assert strip_chat_markup("   \t  ") == ""


def test_strip_chat_markup_structural_ssml_boundaries() -> None:
    """Adjacent <p> and <s> blocks should preserve word boundaries."""
    assert strip_chat_markup("<p>Hello</p><p>world</p>") == "Hello world"
    assert strip_chat_markup("<s>First.</s><s>Second.</s>") == "First. Second."


def test_strip_chat_markup_incomplete_ssml_tags() -> None:
    """Incomplete SSML tags from interruptions should be stripped."""
    assert strip_chat_markup(
        '<phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato'
    ) == "tomato"
    assert strip_chat_markup('<prosody rate="fast">hello') == "hello"
