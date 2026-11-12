from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from livekit.agents import Agent, AgentSession
from livekit.agents.voice.turn import _StreamingTurnDetector

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit, pytest.mark.virtual_time, pytest.mark.no_concurrent]

_FACTORIES = {"stt": FakeSTT, "vad": FakeVAD, "llm": FakeLLM, "tts": FakeTTS}


def test_update_options_not_running_replaces_only_given_fields() -> None:
    stt1, stt2, vad1, llm1, tts1, tts2 = (
        FakeSTT(),
        FakeSTT(),
        FakeVAD(),
        FakeLLM(),
        FakeTTS(),
        FakeTTS(),
    )
    agent = Agent(instructions="test", stt=stt1, vad=vad1, llm=llm1, tts=tts1)

    agent.update_options(stt=stt2, tts=tts2)

    assert agent.stt is stt2
    assert agent.tts is tts2
    assert agent.vad is vad1
    assert agent.llm is llm1


@pytest.mark.parametrize("field", list(_FACTORIES))
async def test_update_options_running_swaps_model(field: str) -> None:
    old, new = _FACTORIES[field](), _FACTORIES[field]()
    agent = Agent(
        instructions="test", **{k: old if k == field else f() for k, f in _FACTORIES.items()}
    )
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and getattr(activity, field) is old
        assert activity._audio_recognition is not None
        old_pipeline = activity._audio_recognition._stt_pipeline

        agent.update_options(**{field: new})

        assert getattr(agent, field) is new
        assert getattr(activity, field) is new
        # metrics listeners moved to the new instance
        assert activity._on_metrics_collected not in old._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new._events.get("metrics_collected", set())
        if field == "stt":
            assert activity._audio_recognition._stt_pipeline is not old_pipeline
    finally:
        await session.aclose()


async def test_update_options_running_disable_stt() -> None:
    agent = Agent(instructions="test", stt=FakeSTT(), vad=FakeVAD(), llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None

        agent.update_options(stt=None)

        assert agent.stt is None
        assert activity.stt is None
        assert activity._audio_recognition is not None
        assert activity._audio_recognition._stt_pipeline is None
    finally:
        await session.aclose()


@pytest.mark.parametrize(
    ("old_llm", "new_llm"), [(FakeLLM, FakeRealtimeModel), (FakeRealtimeModel, FakeLLM)]
)
async def test_update_options_running_rejects_realtime_swap(old_llm: type, new_llm: type) -> None:
    agent = Agent(instructions="test", llm=old_llm())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        with pytest.raises(RuntimeError, match="RealtimeModel"):
            agent.update_options(llm=new_llm())
        assert isinstance(agent.llm, old_llm)
    finally:
        await session.aclose()


class _LabeledSTT(FakeSTT):
    @property
    def model(self) -> str:
        return "new-model"

    @property
    def provider(self) -> str:
        return "new-provider"


class _Ctx(BaseModel):
    value: int = 1


async def test_update_options_stt_swap_refreshes_model_provider_and_context() -> None:
    agent = Agent(instructions="test", stt=FakeSTT(), vad=FakeVAD(), llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        # stand in for live speaker metadata captured from the old stream
        recognition.stt_context = _Ctx()

        agent.update_options(stt=_LabeledSTT())

        assert recognition._stt_model == "new-model"
        assert recognition._stt_provider == "new-provider"
        assert recognition.stt_context is None
    finally:
        await session.aclose()


class _LowSilenceVAD(FakeVAD):
    @property
    def min_silence_duration(self) -> float:
        return 0.0


async def test_update_options_vad_check_is_atomic() -> None:
    old_stt, old_vad = FakeSTT(), FakeVAD()
    agent = Agent(instructions="test", stt=old_stt, vad=old_vad, llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        # a streaming turn detector constrains the VAD's min_silence_duration
        recognition._turn_detector = MagicMock(spec=_StreamingTurnDetector)

        with pytest.raises(ValueError, match="min_silence_duration"):
            agent.update_options(stt=FakeSTT(), vad=_LowSilenceVAD())

        # rejected before any mutation
        assert agent.stt is old_stt
        assert agent.vad is old_vad
    finally:
        await session.aclose()


def test_agent_tts_text_transforms_init_and_update() -> None:
    agent = Agent(instructions="test", tts_text_transforms=["filter_emoji"])
    assert agent.tts_text_transforms == ["filter_emoji"]

    agent.update_options(tts_text_transforms=["filter_markdown"])
    assert agent.tts_text_transforms == ["filter_markdown"]


@pytest.mark.asyncio
async def test_agent_tts_text_transforms_resolution_in_session() -> None:
    agent = Agent(instructions="test", llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(
        turn_handling={"turn_detection": None},
        tts_text_transforms=["filter_markdown"],
    )
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        # Falls back to session options when unset on agent
        assert activity.tts_text_transforms == ["filter_markdown"]

        # Updating agent overrides session options
        agent.update_options(tts_text_transforms=["filter_emoji"])
        assert activity.tts_text_transforms == ["filter_emoji"]
    finally:
        await session.aclose()
