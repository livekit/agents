import asyncio
from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings

from .fake_io import FakeAudioOutput

pytestmark = pytest.mark.unit


class _TextSource:
    def __init__(self) -> None:
        self.items = 0
        self.close_count = 0

    def __aiter__(self) -> "_TextSource":
        return self

    async def __anext__(self) -> str:
        self.items += 1
        if self.items > 2:
            raise StopAsyncIteration
        return "hello "

    async def aclose(self) -> None:
        self.close_count += 1


@pytest.mark.parametrize("audio_enabled", [False, True])
@pytest.mark.parametrize("synthesize_audio", [False, True])
@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
async def test_say_closes_text_source(
    outcome: str, audio_enabled: bool, synthesize_audio: bool
) -> None:
    started = asyncio.Event()
    source = _TextSource()
    tts_started = asyncio.Event()

    class TestAgent(Agent):
        async def transcription_node(
            self, text: AsyncIterable[str], model_settings: ModelSettings
        ) -> AsyncIterator[str]:
            async for chunk in text:
                if audio_enabled and synthesize_audio:
                    await tts_started.wait()
                started.set()
                if outcome == "error":
                    raise RuntimeError("transcription failed")
                if outcome == "cancel":
                    await asyncio.Future()
                yield chunk

        async def tts_node(
            self, text: AsyncIterable[str], model_settings: ModelSettings
        ) -> AsyncIterator[rtc.AudioFrame]:
            async for _ in text:
                tts_started.set()
            if False:
                yield rtc.AudioFrame.create(16000, 1, 160)

    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    audio_output = FakeAudioOutput()
    session.output.audio = audio_output
    session.output.set_audio_enabled(audio_enabled)
    baseline_listeners = len(audio_output._events.get("playback_started", set()))

    async def audio() -> AsyncIterator[rtc.AudioFrame]:
        if False:
            yield rtc.AudioFrame.create(16000, 1, 160)

    await session.start(TestAgent(instructions="test"))
    try:
        handle = session.say(source, audio=None if synthesize_audio else audio())
        await asyncio.wait_for(started.wait(), timeout=5)
        if outcome == "cancel":
            for task in handle._tasks:
                task.cancel()
        results = await asyncio.wait_for(
            asyncio.gather(*handle._tasks, return_exceptions=True), timeout=5
        )
        if outcome == "error":
            assert any(isinstance(result, RuntimeError) for result in results)
        elif outcome == "cancel":
            assert any(isinstance(result, asyncio.CancelledError) for result in results)
        assert source.close_count == 1
        await asyncio.sleep(0)
        assert len(audio_output._events.get("playback_started", set())) == baseline_listeners
    finally:
        await session.aclose()
        await source.aclose()
