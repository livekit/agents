from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import (
    Agent,
    AgentSession,
    APIConnectionError,
    APIConnectOptions,
    ModelSettings,
    tts,
)
from livekit.agents.metrics import TTSMetrics
from livekit.agents.voice.agent_session import SessionConnectOptions

from .fake_llm import FakeLLM
from .fake_tts import FakeTTS

pytestmark = pytest.mark.unit


class _NonStreamingFakeTTS(FakeTTS):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._capabilities = tts.TTSCapabilities(streaming=False)
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


async def _text() -> AsyncIterator[str]:
    yield "hello world."


async def _start_agent(tts_impl: tts.TTS) -> tuple[AgentSession, Agent]:
    agent = Agent(instructions="test", llm=FakeLLM(), tts=tts_impl)
    session = AgentSession(
        vad=None,
        turn_handling={"turn_detection": None},
        conn_options=SessionConnectOptions(
            tts_conn_options=APIConnectOptions(max_retry=0, timeout=120.0)
        ),
    )
    await session.start(agent)
    return session, agent


async def _consume_tts_node(agent: Agent) -> int:
    frame_count = 0
    async for _ in Agent.default.tts_node(agent, _text(), ModelSettings()):
        frame_count += 1
    return frame_count


def _metrics_listener_count(tts_impl: tts.TTS) -> int:
    return len(tts_impl._events.get("metrics_collected", set()))


async def test_temporary_stream_adapter_is_closed_after_each_turn() -> None:
    tts_impl = _NonStreamingFakeTTS(fake_audio_duration=0.01)
    session, agent = await _start_agent(tts_impl)
    metrics: list[TTSMetrics] = []
    tts_impl.on("metrics_collected", metrics.append)
    baseline = _metrics_listener_count(tts_impl)

    try:
        for _ in range(3):
            assert await _consume_tts_node(agent) > 0
            assert _metrics_listener_count(tts_impl) == baseline

        assert len(metrics) == 3
        assert tts_impl.close_count == 0
    finally:
        await session.aclose()


async def test_temporary_stream_adapter_is_closed_after_synthesis_failure() -> None:
    tts_impl = _NonStreamingFakeTTS(
        fake_audio_duration=0.01,
        fake_exception=APIConnectionError("probe failure"),
    )
    session, agent = await _start_agent(tts_impl)
    baseline = _metrics_listener_count(tts_impl)

    try:
        with pytest.raises(APIConnectionError, match="probe failure"):
            await _consume_tts_node(agent)

        assert _metrics_listener_count(tts_impl) == baseline
        assert tts_impl.close_count == 0
    finally:
        await session.aclose()


async def test_temporary_stream_adapter_is_closed_after_cancellation() -> None:
    tts_impl = _NonStreamingFakeTTS(fake_timeout=60.0, fake_audio_duration=0.01)
    session, agent = await _start_agent(tts_impl)
    baseline = _metrics_listener_count(tts_impl)
    consumer = asyncio.create_task(_consume_tts_node(agent))

    try:
        await asyncio.wait_for(anext(tts_impl.synthesize_ch), timeout=5.0)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer

        assert _metrics_listener_count(tts_impl) == baseline
        assert tts_impl.close_count == 0
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await session.aclose()
