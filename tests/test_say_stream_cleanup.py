import asyncio
from collections.abc import AsyncIterable, AsyncIterator

import pytest

from livekit.agents import Agent, AgentSession, ModelSettings

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


@pytest.mark.parametrize("outcome", ["complete", "error", "cancel"])
async def test_say_closes_text_source(outcome: str) -> None:
    started = asyncio.Event()
    source = _TextSource()

    class TestAgent(Agent):
        async def transcription_node(
            self, text: AsyncIterable[str], model_settings: ModelSettings
        ) -> AsyncIterator[str]:
            async for chunk in text:
                started.set()
                if outcome == "error":
                    raise RuntimeError("transcription failed")
                if outcome == "cancel":
                    await asyncio.Future()
                yield chunk

    session = AgentSession(vad=None, turn_handling={"turn_detection": None})
    session.output.set_audio_enabled(False)
    await session.start(TestAgent(instructions="test"))
    try:
        handle = session.say(source)
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
    finally:
        await session.aclose()
        await source.aclose()
