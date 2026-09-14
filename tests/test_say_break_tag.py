from __future__ import annotations

from typing import Any

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

        # Plain text with HTML tags is preserved when SSML is not enabled
        handle6 = session.say("Use <p> and </p> tags")
        await handle6.wait_for_playout()

        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert len(messages) == 5
        assert messages[4].text_content == "Use <p> and </p> tags"
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_say_text_only_preserves_markup() -> None:
    agent = Agent(instructions="test", llm=FakeLLM())
    session = AgentSession(
        vad=None,
        turn_handling={"turn_detection": None},
    )
    session.output.set_audio_enabled(False)
    await session.start(agent)
    try:
        handle = session.say("Use <p> and </p> tags")
        await handle.wait_for_playout()

        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert len(messages) == 1
        assert messages[0].text_content == "Use <p> and </p> tags"
    finally:
        await session.aclose()


def test_strip_chat_markup_whitespace_only() -> None:
    """Whitespace-only text should return empty string."""
    assert strip_chat_markup("  ") == ""
    assert strip_chat_markup("   \t  ") == ""


def test_strip_chat_markup_structural_ssml_boundaries() -> None:
    """Adjacent <p> and <s> blocks should preserve word boundaries when SSML is enabled."""
    assert strip_chat_markup("<speak><p>Hello</p><p>world</p></speak>") == "Hello world"
    assert strip_chat_markup("<speak><s>First.</s><s>Second.</s></speak>") == "First. Second."
    assert strip_chat_markup("<p>Hello</p><p>world</p>", ssml=True) == "Hello world"
    assert strip_chat_markup("<s>First.</s><s>Second.</s>", ssml=True) == "First. Second."


def test_strip_chat_markup_preserves_literal_markup_when_ssml_disabled() -> None:
    """Preserve literal HTML/XML markup when SSML is not enabled."""
    assert strip_chat_markup("Use <p> and </p> tags") == "Use <p> and </p> tags"
    assert strip_chat_markup("Use <s> and </s> tags") == "Use <s> and </s> tags"
    assert strip_chat_markup("<p>Hello</p><p>world</p>") == "<p>Hello</p><p>world</p>"
    assert strip_chat_markup("<s>First.</s><s>Second.</s>") == "<s>First.</s><s>Second.</s>"
    assert (
        strip_chat_markup('Explain <phoneme alphabet="ipa">tomato</phoneme>')
        == 'Explain <phoneme alphabet="ipa">tomato</phoneme>'
    )
    assert (
        strip_chat_markup("Document <speaker> before <p> here")
        == "Document <speaker> before <p> here"
    )


def test_strip_chat_markup_sub_alias() -> None:
    """SSML <sub> tags should substitute their spoken alias in assistant history."""
    assert (
        strip_chat_markup('<sub alias="World Wide Web Consortium">W3C</sub>', ssml=True)
        == "World Wide Web Consortium"
    )
    assert (
        strip_chat_markup('<speak><sub alias="World Wide Web Consortium">W3C</sub></speak>')
        == "World Wide Web Consortium"
    )
    # sub without alias unwraps inner text
    assert strip_chat_markup("<sub>W3C</sub>", ssml=True) == "W3C"
    # When SSML disabled, sub tags are preserved literally
    assert (
        strip_chat_markup('<sub alias="World Wide Web Consortium">W3C</sub>')
        == '<sub alias="World Wide Web Consortium">W3C</sub>'
    )


def test_strip_chat_markup_quoted_attributes_with_closing_angle() -> None:
    """Handle closing angles inside quoted SSML attributes."""
    assert strip_chat_markup('<sub alias="2 > 1">comparison</sub>', ssml=True) == "2 > 1"
    assert strip_chat_markup('<speak><sub alias="2 > 1">comparison</sub></speak>') == "2 > 1"
    assert strip_chat_markup('<speak><prosody rate=">fast">hello</prosody></speak>') == "hello"
    assert strip_chat_markup('<sub alias="2 > 1">comparison', ssml=True) == "comparison"


def test_strip_chat_markup_incomplete_ssml_tags() -> None:
    """Incomplete SSML tags from interruptions should be stripped when SSML is enabled."""
    assert strip_chat_markup('<phoneme alphabet="ipa" ph="təˈmeɪtoʊ">tomato', ssml=True) == "tomato"
    assert strip_chat_markup('<prosody rate="fast">hello', ssml=True) == "hello"


def test_strip_chat_markup_provider_specific_ssml() -> None:
    """Provider-specific SSML (e.g. Amazon Polly, Azure) should be unwrapped or stripped."""
    assert (
        strip_chat_markup('<amazon:effect name="drc">Breaking news</amazon:effect>')
        == "Breaking news"
    )
    assert (
        strip_chat_markup(
            '<amazon:domain name="news"><amazon:effect name="drc">Breaking news</amazon:effect></amazon:domain>'
        )
        == "Breaking news"
    )
    assert (
        strip_chat_markup(
            'Normal speech <amazon:breath duration="medium" volume="default"/> continues.'
        )
        == "Normal speech continues."
    )
    assert (
        strip_chat_markup('<mstts:express-as style="cheerful">Have a nice day!</mstts:express-as>')
        == "Have a nice day!"
    )
    assert strip_chat_markup('<amazon:effect name="drc">Interrupted') == "Interrupted"


def test_strip_chat_markup_azure_provider_normalization() -> None:
    """Verify provider 'Azure TTS' enables SSML unwrapping for structural tags like <p>."""

    class FakeAzureTTS:
        @property
        def provider(self) -> str:
            return "Azure TTS"

    azure_tts = FakeAzureTTS()
    text = "<p>Hello</p><p>world</p>"
    assert strip_chat_markup(text, tts=azure_tts) == "Hello world"


def test_strip_chat_markup_azure_openai_does_not_enable_ssml() -> None:
    """Azure OpenAI TTS should not trigger SSML cleanup for literal <p> and <s> tags."""

    class FakeAzureOpenAITTS:
        __module__ = "livekit.plugins.openai.tts"

        @property
        def provider(self) -> str:
            return "my-resource.openai.azure.com"

    openai_tts = FakeAzureOpenAITTS()
    text = "Use <p> and </p> tags"
    assert strip_chat_markup(text, tts=openai_tts) == "Use <p> and </p> tags"


def test_strip_chat_markup_wrapped_tts_adapters() -> None:
    """Wrapped TTS instances inside adapters (FallbackAdapter, StreamAdapter) should be inspected."""

    class FakeSSMLTTS:
        @property
        def provider(self) -> str:
            return "Azure TTS"

    class FakeFallbackAdapter:
        def __init__(self, instances: list[Any]) -> None:
            self._tts_instances = instances

    class FakeStreamAdapter:
        def __init__(self, tts: Any) -> None:
            self._wrapped_tts = tts

    class FakeOptsTTS:
        def __init__(self) -> None:
            self._opts = type("Opts", (), {"enable_ssml": True})()

    azure_tts = FakeSSMLTTS()
    fallback_adapter = FakeFallbackAdapter([azure_tts])
    assert strip_chat_markup("<p>Hello</p><p>world</p>", tts=fallback_adapter) == "Hello world"

    stream_adapter = FakeStreamAdapter(azure_tts)
    assert strip_chat_markup("<p>Hello</p><p>world</p>", tts=stream_adapter) == "Hello world"

    opts_adapter = FakeFallbackAdapter([FakeOptsTTS()])
    assert strip_chat_markup("<p>Hello</p><p>world</p>", tts=opts_adapter) == "Hello world"
