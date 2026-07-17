from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit]


def test_update_models_not_running_replaces_fields() -> None:
    stt1, stt2 = FakeSTT(), FakeSTT()
    vad1, vad2 = FakeVAD(), FakeVAD()
    llm1, llm2 = FakeLLM(), FakeLLM()
    tts1, tts2 = FakeTTS(), FakeTTS()

    agent = Agent(instructions="test", stt=stt1, vad=vad1, llm=llm1, tts=tts1)
    agent.update_models(stt=stt2, vad=vad2, llm=llm2, tts=tts2)

    assert agent.stt is stt2
    assert agent.vad is vad2
    assert agent.llm is llm2
    assert agent.tts is tts2


def test_update_models_not_running_only_touches_given() -> None:
    stt1 = FakeSTT()
    llm1 = FakeLLM()
    agent = Agent(instructions="test", stt=stt1, llm=llm1)

    tts_new = FakeTTS()
    agent.update_models(tts=tts_new)

    assert agent.stt is stt1  # untouched
    assert agent.llm is llm1  # untouched
    assert agent.tts is tts_new


@pytest.mark.asyncio
async def test_update_models_running_swaps_tts() -> None:
    old_tts, new_tts = FakeTTS(), FakeTTS()
    agent = Agent(instructions="test", llm=FakeLLM(), tts=old_tts)
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.tts is old_tts

        agent.update_models(tts=new_tts)

        assert agent.tts is new_tts
        assert activity.tts is new_tts
        # metrics/error listeners moved to the new instance
        assert activity._on_metrics_collected not in old_tts._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_tts._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_models_running_swaps_llm() -> None:
    old_llm, new_llm = FakeLLM(), FakeLLM()
    agent = Agent(instructions="test", llm=old_llm)
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.llm is old_llm

        agent.update_models(llm=new_llm)

        assert agent.llm is new_llm
        assert activity.llm is new_llm
        assert activity._on_metrics_collected not in old_llm._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_llm._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_models_running_swaps_stt_rewires_pipeline() -> None:
    old_stt, new_stt = FakeSTT(), FakeSTT()
    agent = Agent(instructions="test", stt=old_stt, vad=FakeVAD(), llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.stt is old_stt
        recognition = activity._audio_recognition
        assert recognition is not None
        old_pipeline = recognition._stt_pipeline

        agent.update_models(stt=new_stt)

        assert agent.stt is new_stt
        assert activity.stt is new_stt
        # the live STT pipeline was rebuilt
        assert recognition._stt_pipeline is not old_pipeline
        assert activity._on_metrics_collected not in old_stt._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_stt._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_models_running_swaps_vad() -> None:
    old_vad, new_vad = FakeVAD(), FakeVAD()
    agent = Agent(instructions="test", stt=FakeSTT(), vad=old_vad, llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.vad is old_vad

        agent.update_models(vad=new_vad)

        assert agent.vad is new_vad
        assert activity.vad is new_vad
        assert activity._on_metrics_collected not in old_vad._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_vad._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_models_running_disable_stt() -> None:
    agent = Agent(instructions="test", stt=FakeSTT(), vad=FakeVAD(), llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None

        agent.update_models(stt=None)

        assert agent.stt is None
        assert activity.stt is None
        assert activity._audio_recognition is not None
        assert activity._audio_recognition._stt_pipeline is None
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_models_running_rejects_swap_to_realtime() -> None:
    agent = Agent(instructions="test", llm=FakeLLM())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        with pytest.raises(RuntimeError, match="RealtimeModel"):
            agent.update_models(llm=FakeRealtimeModel())
        # nothing was swapped
        assert isinstance(agent.llm, FakeLLM)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_models_running_rejects_swap_away_from_realtime() -> None:
    agent = Agent(instructions="test", llm=FakeRealtimeModel())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        with pytest.raises(RuntimeError, match="RealtimeModel"):
            agent.update_models(llm=FakeLLM())
        assert isinstance(agent.llm, FakeRealtimeModel)
    finally:
        await session.aclose()


class _LabeledSTT(FakeSTT):
    @property
    def model(self) -> str:
        return "new-model"

    @property
    def provider(self) -> str:
        return "new-provider"


@pytest.mark.asyncio
async def test_update_models_stt_swap_refreshes_model_provider_and_context() -> None:
    from pydantic import BaseModel

    class _Ctx(BaseModel):
        value: int = 1

    agent = Agent(instructions="test", stt=FakeSTT(), vad=FakeVAD(), llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        # stand in for live speaker metadata captured from the old stream
        recognition.stt_context = _Ctx()

        agent.update_models(stt=_LabeledSTT())

        # trace attributes and speaker context follow the new STT
        assert recognition._stt_model == "new-model"
        assert recognition._stt_provider == "new-provider"
        assert recognition.stt_context is None
    finally:
        await session.aclose()
