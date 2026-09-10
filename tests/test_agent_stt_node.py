from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, stt

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _NonStreamingFakeSTT(FakeSTT):
    def __init__(self) -> None:
        super().__init__()
        self._capabilities = stt.STTCapabilities(streaming=False, interim_results=False)
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1


def _metrics_listener_count(stt_impl: stt.STT) -> int:
    return len(stt_impl._events.get("metrics_collected", set()))


async def test_temporary_stream_adapter_is_closed_with_session() -> None:
    stt_impl = _NonStreamingFakeSTT()
    baseline = _metrics_listener_count(stt_impl)
    agent = Agent(
        instructions="test",
        llm=FakeLLM(),
        stt=stt_impl,
        vad=FakeVAD(),
    )
    session = AgentSession(
        vad=None,
        turn_handling={"turn_detection": None},
        session_close_transcript_timeout=0.0,
    )

    await session.start(agent)
    try:
        assert _metrics_listener_count(stt_impl) > baseline
    finally:
        await session.aclose()

    assert _metrics_listener_count(stt_impl) == baseline
    assert stt_impl.close_count == 0
