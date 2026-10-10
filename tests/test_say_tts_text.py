from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings
from tests.fake_io import FakeAudioOutput, FakeTextOutput
from tests.fake_llm import FakeLLM
from tests.fake_tts import FakeTTS

pytestmark = pytest.mark.unit


class RecordingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test", llm=FakeLLM(), tts=FakeTTS(fake_audio_duration=0.01))
        self.tts_inputs: list[str] = []

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterator[rtc.AudioFrame]:
        async def record_input() -> AsyncIterator[str]:
            async for chunk in text:
                self.tts_inputs.append(chunk)
                yield chunk

        async for frame in Agent.default.tts_node(self, record_input(), model_settings):
            yield frame


@pytest.mark.asyncio
async def test_say_uses_separate_text_for_tts_and_chat_context() -> None:
    agent = RecordingAgent()
    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    transcript_output = FakeTextOutput()
    session.output.transcription = transcript_output
    await session.start(agent)
    try:
        handle = session.say("Hello world", tts_text='Hello<break time="1s"/>world')
        await handle.wait_for_playout()

        assert "".join(agent.tts_inputs) == 'Hello<break time="1s"/>world'
        assert transcript_output._messages == ["Hello world"]
        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert [msg.text_content for msg in messages] == ["Hello world"]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_say_accepts_independent_text_streams() -> None:
    async def plain_text() -> AsyncIterator[str]:
        yield "First "
        yield "second"

    async def speech_text() -> AsyncIterator[str]:
        yield 'First<break time="1s"/>'
        yield "second"

    agent = RecordingAgent()
    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        handle = session.say(plain_text(), tts_text=speech_text())
        await handle.wait_for_playout()

        assert "".join(agent.tts_inputs) == 'First<break time="1s"/>second'
        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert [msg.text_content for msg in messages] == ["First second"]
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_say_keeps_literal_markup_without_tts_text() -> None:
    agent = RecordingAgent()
    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        handle = session.say('Explain the <break time="1s"/> tag')
        await handle.wait_for_playout()

        assert "".join(agent.tts_inputs) == 'Explain the <break time="1s"/> tag'
        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert [msg.text_content for msg in messages] == ['Explain the <break time="1s"/> tag']
    finally:
        await session.aclose()
