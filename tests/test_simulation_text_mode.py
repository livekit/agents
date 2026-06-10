from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.simulation import (
    SimulationContext,
    SimulationDispatch,
    SimulationMode,
)

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit]


def _dispatch(mode: int | None = None) -> SimulationDispatch:
    dispatch = SimulationDispatch(simulation_run_id="SR_test", job_id="SRJ_test")
    if mode is not None:
        dispatch.mode = mode
    return dispatch


def test_simulation_context_mode() -> None:
    # unspecified is treated as text: simulations predating the field were text-only
    ctx = SimulationContext(_dispatch(), MagicMock())
    assert ctx.mode == SimulationMode.SIMULATION_MODE_TEXT

    ctx = SimulationContext(_dispatch(SimulationMode.SIMULATION_MODE_AUDIO), MagicMock())
    assert ctx.mode == SimulationMode.SIMULATION_MODE_AUDIO


async def test_text_simulation_drops_stt_tts() -> None:
    session = AgentSession(
        stt=FakeSTT(),
        tts=FakeTTS(),
        vad=FakeVAD(),
        llm=FakeLLM(),
    )
    # agent-level models must be disabled too (AgentActivity resolves them
    # before falling back to the session)
    agent = Agent(instructions="test", stt=FakeSTT(), tts=FakeTTS(), vad=FakeVAD())

    job_ctx = MagicMock()
    job_ctx.job.enable_recording = False
    job_ctx.job.id = "job"
    job_ctx.job.agent_name = "agent"
    job_ctx.room.name = "room"
    job_ctx._primary_agent_session = None
    job_ctx.simulation_context.return_value = SimulationContext(_dispatch(), job_ctx)

    with patch("livekit.agents.voice.agent_session.get_job_context", return_value=job_ctx):
        await session.start(agent)

    try:
        activity = session._activity
        assert activity is not None
        assert activity.stt is None
        assert activity.tts is None
        assert activity.vad is None
        assert activity.llm is not None  # the LLM stays
    finally:
        await session.aclose()
