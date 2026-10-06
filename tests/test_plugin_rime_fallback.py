"""Prism uses the public realtime hooks through the LiveKit fallback adapter."""

import asyncio

import pytest

from livekit.agents import Agent, AgentSession, llm

from .fake_io import FakeAudioOutput
from .test_plugin_rime_realtime import peer_model

pytestmark = pytest.mark.unit


async def test_prism_fallback_typed_input_playback_and_explicit_restart() -> None:
    async with peer_model() as (peer, model, _):
        adapter = llm.RealtimeModelFallbackAdapter([model])
        async with AgentSession(llm=adapter) as session:
            session.output.audio = FakeAudioOutput()
            await session.start(Agent(instructions="test"))
            for text in ("First turn", "After explicit restart"):
                speech = session.generate_reply(user_input=text)
                request = await peer.next()
                assert request["response"]["prism_input_text"] == text
                response_id = await peer.start(request)
                await peer.message(response_id)
                await peer.end(response_id)
                await asyncio.wait_for(speech.wait_for_playout(), 3)
                receipt = await asyncio.wait_for(peer.receipts.get(), 3)
                assert receipt["response_id"] == response_id
                if text == "First turn":
                    await adapter.restart_session()
                    assert [item["text"] for item in peer.items] == ["First turn", "Hello"]
