from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Sequence
from typing import Any

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings, tts
from livekit.agents.voice.generation import perform_tts_inference
from livekit.agents.voice.transcription.text_transforms import TextTransforms

from .fake_io import FakeAudioOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


async def _text() -> AsyncIterable[str]:
    yield "hello world."


async def _wait_timed_texts(
    node: Any, *, text_transforms: Sequence[TextTransforms] | None = None
) -> Any:
    task, data = perform_tts_inference(
        node=node,
        input=_text(),
        model_settings=ModelSettings(),
        text_transforms=text_transforms,
    )
    try:
        # a hung future would stall the speech that awaits it before it forwards any audio
        return await asyncio.wait_for(data.timed_texts_fut, timeout=2.0)
    finally:
        await asyncio.gather(task, return_exceptions=True)


async def test_timed_texts_resolve_when_tts_node_raises() -> None:
    async def failing_node(
        text: AsyncIterable[str], settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        raise RuntimeError("tts node failed before streaming")

    assert await _wait_timed_texts(failing_node) is None


async def test_timed_texts_resolve_when_text_transform_is_invalid() -> None:
    async def node(
        text: AsyncIterable[str], settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        return None  # type: ignore[return-value]

    invalid: Any = ["not_a_transform"]
    assert await _wait_timed_texts(node, text_transforms=invalid) is None


class _AlignedFakeTTS(FakeTTS):
    def __init__(self) -> None:
        super().__init__(fake_audio_duration=0.1)
        self._capabilities = tts.TTSCapabilities(streaming=True, aligned_transcript=True)


class _FailingTTSNodeAgent(Agent):
    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ) -> AsyncIterable[rtc.AudioFrame]:
        raise RuntimeError("tts node failed before streaming")


async def test_reply_completes_when_tts_node_raises_with_aligned_transcript() -> None:
    llm = FakeLLM(
        fake_responses=[FakeLLMResponse(input="hi", content="hello there.", ttft=0, duration=0)]
    )
    agent = _FailingTTSNodeAgent(instructions="test", llm=llm, tts=_AlignedFakeTTS())
    session = AgentSession[None](
        vad=None, turn_handling={"turn_detection": None}, use_tts_aligned_transcript=True
    )
    session.output.audio = FakeAudioOutput(can_pause=True)
    await session.start(agent)
    try:
        handle = session.generate_reply(user_input="hi")
        await asyncio.wait_for(handle, timeout=3.0)
    finally:
        await session.aclose()
