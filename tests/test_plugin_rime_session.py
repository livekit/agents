"""Rime session scope and actual sink playback accounting."""

from __future__ import annotations

import asyncio

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession as CoreAgentSession, llm
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.plugins.rime._realtime_activity import RimeAgentActivity
from livekit.plugins.rime._realtime_session import AgentSession

from .fake_io import FakeAudioOutput
from .fake_rime_realtime import FakeRealtimeModel

pytestmark = pytest.mark.unit


async def test_rime_session_does_not_change_core_classes_or_ordinary_sessions():
    classes = (CoreAgentSession, AgentActivity, llm.RealtimeSession)
    before = [dict(cls.__dict__) for cls in classes]
    async with CoreAgentSession(llm=FakeRealtimeModel()) as ordinary:
        await ordinary.start(Agent(instructions="Ordinary session"))
        assert type(ordinary._activity) is AgentActivity
        async with AgentSession(llm=FakeRealtimeModel()) as prism:
            await prism.start(Agent(instructions="Rime session"))
            assert type(prism._activity) is RimeAgentActivity
            assert type(ordinary._activity) is AgentActivity
    assert [dict(cls.__dict__) for cls in classes] == before


async def test_custom_audio_node_reports_actual_sink_duration():
    pytest.importorskip("rimelabs_sdk.realtime")
    from .test_plugin_rime_realtime import peer_model

    class HalfAudio(Agent):
        async def realtime_audio_output_node(self, audio, model_settings):
            async for frame in audio:
                samples = frame.samples_per_channel // 2
                yield rtc.AudioFrame(
                    data=bytes(frame.data)[: samples * frame.num_channels * 2],
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                    samples_per_channel=samples,
                )

    async with peer_model() as (peer, model, _):
        async with AgentSession(llm=model) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(HalfAudio(instructions="Help"))
            speech = session.generate_reply(user_input="Question")
            response = await peer.start(await peer.next())
            await peer.message(response, seconds=0.2)
            await peer.end(response)
            await asyncio.wait_for(speech.wait_for_playout(), 3)
            receipt = await asyncio.wait_for(peer.receipts.get(), 3)
            assert receipt["response_id"] == response
            assert receipt["played_ms"] == 100
