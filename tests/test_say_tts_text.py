from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings, tokenize
from livekit.agents.tts import TTS, FallbackAdapter, StreamAdapter, TTSCapabilities
from livekit.agents.utils.aio.channel import ChanEmpty
from tests.fake_io import FakeAudioOutput, FakeTextOutput
from tests.fake_llm import FakeLLM
from tests.fake_tts import FakeTTS

pytestmark = pytest.mark.unit


class RecordingAgent(Agent):
    def __init__(self, tts: TTS | None = None) -> None:
        super().__init__(
            instructions="test", llm=FakeLLM(), tts=tts or FakeTTS(fake_audio_duration=0.01)
        )
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


class NonStreamingFakeTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__(fake_audio_duration=0.01)
        self._capabilities = TTSCapabilities(streaming=False)


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


@pytest.mark.asyncio
async def test_say_tees_shared_text_stream() -> None:
    async def shared_text() -> AsyncIterator[str]:
        yield "Hello "
        yield "world"

    agent = RecordingAgent()
    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        stream = shared_text()
        handle = session.say(stream, tts_text=stream)
        await handle.wait_for_playout()

        assert "".join(agent.tts_inputs) == "Hello world"
        messages = [msg for msg in agent.chat_ctx.messages() if msg.role == "assistant"]
        assert [msg.text_content for msg in messages] == ["Hello world"]
    finally:
        await session.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["direct", "stream", "custom_tokenizer", "fallback"])
@pytest.mark.parametrize(
    "markup",
    [
        '<prosody rate="slow">This is the first long sentence. This is the second long sentence.</prosody>',
        '<mstts:express-as style="cheerful">This is the first long sentence. This is the second long sentence.</mstts:express-as>',
    ],
)
async def test_say_keeps_ssml_scope_in_one_non_streaming_request(markup: str, adapter: str) -> None:
    tts = NonStreamingFakeTTS()
    model: TTS = tts
    if adapter == "stream":
        model = StreamAdapter(tts=tts)
    elif adapter == "custom_tokenizer":
        model = StreamAdapter(
            tts=tts, sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(retain_format=True)
        )
    elif adapter == "fallback":
        model = FallbackAdapter([tts, FakeTTS(fake_audio_duration=0.01)])
    agent = RecordingAgent(model)
    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        handle = session.say(
            "This is the first long sentence. This is the second long sentence.",
            tts_text=markup,
        )
        await handle.wait_for_playout()

        assert "".join(agent.tts_inputs) == markup
        assert tts.synthesize_ch.recv_nowait()._input_text == markup
        with pytest.raises(ChanEmpty):
            tts.synthesize_ch.recv_nowait()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_say_keeps_custom_stream_adapter_request_limit() -> None:
    tts = NonStreamingFakeTTS()
    model = StreamAdapter(
        tts=tts,
        sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(
            retain_format=True, max_token_len=100
        ),
    )
    agent = RecordingAgent(model)
    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        first = "<prosody>" + "First sentence. " * 3 + "</prosody>"
        second = "<prosody>" + "Second sentence. " * 3 + "</prosody>"
        handle = session.say("First sentence. Second sentence.", tts_text=f"{first} End. {second}")
        await handle.wait_for_playout()

        assert tts.synthesize_ch.recv_nowait()._input_text == f"{first} End."
        assert tts.synthesize_ch.recv_nowait()._input_text == second
        with pytest.raises(ChanEmpty):
            tts.synthesize_ch.recv_nowait()
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_stream_adapter_rejects_xml_scope_over_request_limit() -> None:
    tts = NonStreamingFakeTTS()
    model = StreamAdapter(
        tts=tts,
        sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(max_token_len=100),
    )
    markup = "<prosody>" + "Long sentence. " * 10 + "</prosody>"

    async with model.stream(xml_aware=True) as stream:
        stream.push_text(markup)
        stream.end_input()
        with pytest.raises(ValueError, match="TTS request exceeds max_token_len=100"):
            async for _ in stream:
                pass

    with pytest.raises(ChanEmpty):
        tts.synthesize_ch.recv_nowait()


@pytest.mark.asyncio
async def test_stream_adapter_pacing_respects_request_limit() -> None:
    tts = NonStreamingFakeTTS()
    model = StreamAdapter(
        tts=tts,
        sentence_tokenizer=tokenize.blingfire.SentenceTokenizer(max_token_len=100),
        text_pacing=True,
    )
    sentences = [
        f"Sentence {i} has enough words to make a useful text to speech request." for i in range(3)
    ]
    agent = RecordingAgent(model)
    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.audio = FakeAudioOutput()
    await session.start(agent)
    try:
        handle = session.say("Spoken sentences", tts_text=" ".join(sentences))
        await handle.wait_for_playout()

        assert [tts.synthesize_ch.recv_nowait()._input_text for _ in sentences] == sentences
        with pytest.raises(ChanEmpty):
            tts.synthesize_ch.recv_nowait()
    finally:
        await session.aclose()
