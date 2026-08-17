from __future__ import annotations

import pytest

from livekit.agents import Agent, AgentSession, inference

from .fake_llm import FakeLLM
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_stt import FakeSTT
from .fake_tts import FakeTTS
from .fake_vad import FakeVAD

pytestmark = [pytest.mark.unit]


def test_update_options_not_running_replaces_fields() -> None:
    stt1, stt2 = FakeSTT(), FakeSTT()
    vad1, vad2 = FakeVAD(), FakeVAD()
    llm1, llm2 = FakeLLM(), FakeLLM()
    tts1, tts2 = FakeTTS(), FakeTTS()

    agent = Agent(instructions="test", stt=stt1, vad=vad1, llm=llm1, tts=tts1)
    agent.update_options(stt=stt2, vad=vad2, llm=llm2, tts=tts2)

    assert agent.stt is stt2
    assert agent.vad is vad2
    assert agent.llm is llm2
    assert agent.tts is tts2


def test_update_options_not_running_only_touches_given() -> None:
    stt1 = FakeSTT()
    llm1 = FakeLLM()
    agent = Agent(instructions="test", stt=stt1, llm=llm1)

    tts_new = FakeTTS()
    agent.update_options(tts=tts_new)

    assert agent.stt is stt1  # untouched
    assert agent.llm is llm1  # untouched
    assert agent.tts is tts_new


@pytest.mark.asyncio
async def test_update_options_running_swaps_tts() -> None:
    old_tts, new_tts = FakeTTS(), FakeTTS()
    agent = Agent(instructions="test", llm=FakeLLM(), tts=old_tts)
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.tts is old_tts

        agent.update_options(tts=new_tts)

        assert agent.tts is new_tts
        assert activity.tts is new_tts
        # metrics/error listeners moved to the new instance
        assert activity._on_metrics_collected not in old_tts._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_tts._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_options_running_swaps_llm() -> None:
    old_llm, new_llm = FakeLLM(), FakeLLM()
    agent = Agent(instructions="test", llm=old_llm)
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.llm is old_llm

        agent.update_options(llm=new_llm)

        assert agent.llm is new_llm
        assert activity.llm is new_llm
        assert activity._on_metrics_collected not in old_llm._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_llm._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_options_running_swaps_stt_rewires_pipeline() -> None:
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

        agent.update_options(stt=new_stt)

        assert agent.stt is new_stt
        assert activity.stt is new_stt
        # the live STT pipeline was rebuilt
        assert recognition._stt_pipeline is not old_pipeline
        assert activity._on_metrics_collected not in old_stt._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_stt._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_options_running_swaps_vad() -> None:
    old_vad, new_vad = FakeVAD(), FakeVAD()
    agent = Agent(instructions="test", stt=FakeSTT(), vad=old_vad, llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.vad is old_vad

        agent.update_options(vad=new_vad)

        assert agent.vad is new_vad
        assert activity.vad is new_vad
        assert activity._on_metrics_collected not in old_vad._events.get("metrics_collected", set())
        assert activity._on_metrics_collected in new_vad._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_update_options_running_rejects_disabling_text_mode_stt() -> None:
    old_stt = FakeSTT()
    agent = Agent(
        instructions="test",
        stt=old_stt,
        vad=FakeVAD(),
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
            )
        ),
    )
    session = AgentSession(turn_handling={"turn_detection": "vad", "realtime_input_mode": "text"})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.stt is old_stt
        recognition = activity._audio_recognition
        assert recognition is not None
        old_pipeline = recognition._stt_pipeline

        with pytest.raises(ValueError, match="requires an external STT"):
            agent.update_options(stt=None)

        assert agent.stt is old_stt
        assert activity.stt is old_stt
        assert recognition._stt_pipeline is old_pipeline
        assert activity._on_metrics_collected in old_stt._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_options_running_falls_back_to_stt_when_text_mode_vad_is_disabled() -> None:
    old_vad = FakeVAD()
    agent = Agent(
        instructions="test",
        stt=FakeSTT(),
        vad=old_vad,
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
            )
        ),
    )
    session = AgentSession(turn_handling={"turn_detection": "vad", "realtime_input_mode": "text"})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None and activity.vad is old_vad
        recognition = activity._audio_recognition
        assert recognition is not None
        agent.update_options(vad=None)

        assert agent.vad is None
        assert activity.vad is None
        assert recognition._vad is None
        assert activity._turn_detection == "stt"
        assert recognition._turn_detection_mode == "stt"
        assert activity._on_metrics_collected not in old_vad._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_text_mode_rejects_runtime_non_streaming_stt_without_vad() -> None:
    old_stt = FakeSTT()
    new_stt = FakeSTT(streaming=False)
    agent = Agent(
        instructions="test",
        stt=old_stt,
        vad=None,
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
            )
        ),
    )
    session = AgentSession(turn_handling={"realtime_input_mode": "text"})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        old_pipeline = recognition._stt_pipeline

        with pytest.raises(ValueError, match="non-streaming STT.*VAD"):
            agent.update_options(stt=new_stt)

        assert agent.stt is old_stt
        assert activity.stt is old_stt
        assert recognition._stt_pipeline is old_pipeline
        assert activity._on_metrics_collected in old_stt._events.get("metrics_collected", set())
        assert activity._on_metrics_collected not in new_stt._events.get("metrics_collected", set())
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_text_mode_rejects_runtime_vad_removal_from_non_streaming_stt() -> None:
    old_stt = FakeSTT(streaming=False)
    old_vad = FakeVAD()
    agent = Agent(
        instructions="test",
        stt=old_stt,
        vad=old_vad,
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
            )
        ),
    )
    session = AgentSession(turn_handling={"turn_detection": "stt", "realtime_input_mode": "text"})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        old_pipeline = recognition._stt_pipeline

        with pytest.raises(ValueError, match="non-streaming STT.*VAD"):
            agent.update_options(vad=None)

        assert agent.vad is old_vad
        assert activity.vad is old_vad
        assert recognition._stt_pipeline is old_pipeline
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_text_mode_accepts_atomic_non_streaming_stt_and_vad_update() -> None:
    old_stt = FakeSTT()
    new_stt = FakeSTT(streaming=False)
    new_vad = FakeVAD()
    agent = Agent(
        instructions="test",
        stt=old_stt,
        vad=None,
        llm=FakeRealtimeModel(
            capabilities=fake_capabilities(
                turn_detection=False,
                can_disable_turn_detection=False,
            )
        ),
    )
    session = AgentSession(turn_handling={"realtime_input_mode": "text"})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None

        agent.update_options(stt=new_stt, vad=new_vad)

        assert activity.stt is new_stt
        assert activity.vad is new_vad
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_options_running_rejects_swap_to_realtime() -> None:
    agent = Agent(instructions="test", llm=FakeLLM())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        with pytest.raises(RuntimeError, match="RealtimeModel"):
            agent.update_options(llm=FakeRealtimeModel())
        # nothing was swapped
        assert isinstance(agent.llm, FakeLLM)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_update_options_running_rejects_swap_away_from_realtime() -> None:
    agent = Agent(instructions="test", llm=FakeRealtimeModel())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        with pytest.raises(RuntimeError, match="RealtimeModel"):
            agent.update_options(llm=FakeLLM())
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
async def test_update_options_stt_swap_refreshes_model_provider_and_context() -> None:
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

        agent.update_options(stt=_LabeledSTT())

        # trace attributes and speaker context follow the new STT
        assert recognition._stt_model == "new-model"
        assert recognition._stt_provider == "new-provider"
        assert recognition.stt_context is None
    finally:
        await session.aclose()


class _LowSilenceVAD(FakeVAD):
    @property
    def min_silence_duration(self) -> float:
        return 0.0


@pytest.mark.asyncio
async def test_update_options_vad_check_is_atomic() -> None:
    from unittest.mock import MagicMock

    from livekit.agents.voice.turn import _StreamingTurnDetector

    old_stt, old_vad = FakeSTT(), FakeVAD()
    agent = Agent(instructions="test", stt=old_stt, vad=old_vad, llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": None})
    await session.start(agent)
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        # a streaming turn detector constrains the VAD's min_silence_duration
        detector = MagicMock(spec=_StreamingTurnDetector)
        agent._turn_detection = detector
        recognition._turn_detector = detector

        with pytest.raises(ValueError, match="min_silence_duration"):
            agent.update_options(stt=FakeSTT(), vad=_LowSilenceVAD())

        # rejected before any mutation — STT and VAD are untouched
        assert agent.stt is old_stt
        assert agent.vad is old_vad
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_runtime_detector_rejection_preserves_policy_and_listeners() -> None:
    detector = inference.TurnDetector(version="v1-mini")
    agent = Agent(
        instructions="test",
        stt=FakeSTT(),
        vad=_LowSilenceVAD(),
        llm=FakeLLM(),
        tts=FakeTTS(),
    )
    session = AgentSession(turn_handling={"turn_detection": "vad"})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        original_policy = activity._turn_policy
        original_timeout_policy = recognition._finalize_empty_transcript_on_timeout

        with pytest.raises(ValueError, match="min_silence_duration"):
            session.update_options(turn_detection=detector)

        assert activity._turn_policy is original_policy
        assert activity._turn_detection == "vad"
        assert recognition._turn_detection_mode == "vad"
        assert recognition._turn_detector is None
        assert recognition._finalize_empty_transcript_on_timeout is original_timeout_policy
        assert activity._turn_detection_metrics_source is None
        assert activity._on_metrics_collected not in detector._events.get(
            "metrics_collected", set()
        )
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_rejected_detector_update_preserves_combined_endpointing_options() -> None:
    detector = inference.TurnDetector(version="v1-mini")
    agent = Agent(
        instructions="test",
        stt=FakeSTT(),
        vad=_LowSilenceVAD(),
        llm=FakeLLM(),
        tts=FakeTTS(),
    )
    session = AgentSession(
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {"min_delay": 0.4, "max_delay": 1.2},
        }
    )
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        original_endpointing = session.options.endpointing.copy()
        original_overrides = session.options.endpointing_overrides.copy()
        original_recognition_endpointing = recognition._endpointing

        with pytest.raises(ValueError, match="min_silence_duration"):
            session.update_options(
                endpointing_opts={"min_delay": 2.0, "max_delay": 3.0},
                turn_detection=detector,
            )

        assert session.options.endpointing == original_endpointing
        assert session.options.endpointing_overrides == original_overrides
        assert recognition._endpointing is original_recognition_endpointing
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_text_mode_streaming_detector_falls_back_to_stt_without_vad() -> None:
    from unittest.mock import MagicMock

    from livekit.agents.voice.turn import _StreamingTurnDetector

    old_vad = FakeVAD()
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    agent = Agent(instructions="test", stt=FakeSTT(), vad=old_vad, llm=model)
    session = AgentSession(turn_handling={"turn_detection": "vad", "realtime_input_mode": "text"})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        detector = MagicMock(spec=_StreamingTurnDetector)
        agent._turn_detection = detector
        activity._turn_detection = detector
        recognition._turn_detector = detector

        agent.update_options(vad=None)

        assert agent.vad is None
        assert activity.vad is None
        assert activity._turn_detection == "stt"
        assert recognition._turn_detection_mode == "stt"
        assert activity._on_metrics_collected not in old_vad._events.get("metrics_collected", set())
    finally:
        await session.aclose()


class _LegacyTurnDetector:
    model = "legacy-text-detector"
    provider = "test"

    async def unlikely_threshold(self, language: object) -> float:
        return 0.5

    async def supports_language(self, language: object) -> bool:
        return True

    async def predict_end_of_turn(self, chat_ctx: object, *, timeout: float | None = None) -> float:
        return 0.9


@pytest.mark.asyncio
async def test_non_streaming_turn_detector_survives_policy_resolution() -> None:
    detector = _LegacyTurnDetector()
    agent = Agent(
        instructions="test",
        stt=FakeSTT(),
        vad=FakeVAD(),
        llm=FakeLLM(),
        tts=FakeTTS(),
    )
    session = AgentSession(turn_handling={"turn_detection": detector})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None

        assert activity._turn_policy.turn_detection is detector
        assert activity._turn_detection is detector
        assert recognition._turn_detector is detector
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_runtime_detector_activation_moves_metrics_listener() -> None:
    detector = inference.TurnDetector(version="v1-mini")
    agent = Agent(instructions="test", stt=FakeSTT(), vad=None, llm=FakeLLM(), tts=FakeTTS())
    session = AgentSession(turn_handling={"turn_detection": detector})
    await session.start(agent)
    try:
        activity = session._activity
        assert activity is not None
        recognition = activity._audio_recognition
        assert recognition is not None
        assert activity._turn_detection is None
        assert activity._on_metrics_collected not in detector._events.get(
            "metrics_collected", set()
        )

        agent.update_options(vad=FakeVAD())

        assert activity._turn_detection is detector
        assert recognition._turn_detector is detector
        assert activity._on_metrics_collected in detector._events.get("metrics_collected", set())

        agent.update_options(vad=None)

        assert activity._turn_detection is None
        assert recognition._turn_detector is None
        assert activity._on_metrics_collected not in detector._events.get(
            "metrics_collected", set()
        )
    finally:
        await session.aclose()
