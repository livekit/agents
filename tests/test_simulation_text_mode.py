from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from livekit.agents import Agent, AgentSession
from livekit.agents.simulation import (
    SimulationContext,
    SimulationDispatch,
    SimulationMode,
)
from livekit.agents.worker import AgentServer

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit]


def _dispatch(mode: SimulationMode.ValueType | None = None) -> SimulationDispatch:
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


def test_worker_simulation_mode_bypasses_load_threshold() -> None:
    server = AgentServer()
    server._devmode = False  # prod-mode load_threshold (0.7)
    server._worker_load = 1.0
    server._reserved_slots = 0
    server._proc_pool = MagicMock(processes=[])

    assert not server._is_available()

    server._simulation = True
    assert server._is_available()

    server._draining = True
    assert not server._is_available()


async def test_text_simulation_drops_stt_tts() -> None:
    session = AgentSession(
        stt=FakeSTT(),
        tts=FakeTTS(),
        vad=FakeVAD(),
        llm=FakeLLM(),
    )

    job_ctx = MagicMock()
    job_ctx.job.enable_recording = False
    job_ctx.job.id = "job"
    job_ctx.job.agent_name = "agent"
    job_ctx.room.name = "room"
    job_ctx._primary_agent_session = None
    job_ctx.simulation_context.return_value = SimulationContext(_dispatch(), job_ctx)

    with patch("livekit.agents.voice.agent_session.get_job_context", return_value=job_ctx):
        await session.start(Agent(instructions="test"))

    try:
        assert session.stt is None
        assert session.tts is None
        assert session.vad is None
    finally:
        await session.aclose()
