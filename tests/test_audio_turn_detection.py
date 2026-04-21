from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from livekit.agents import Agent, AudioTurnContext, ConversationItemAddedEvent
from livekit.agents.utils.audio import calculate_audio_duration
from livekit.agents.voice.transcription.synchronizer import _SyncedAudioOutput

from .fake_io import FakeAudioInput
from .fake_session import FakeActions, create_session
from .fake_stt import FakeSTT

SESSION_TIMEOUT = 60.0


class _FakeAudioTurnDetector:
    def __init__(self, *, probability: float, threshold: float = 0.5) -> None:
        self._probability = probability
        self._threshold = threshold
        self.contexts: list[AudioTurnContext] = []

    @property
    def model(self) -> str:
        return "fake-audio-turn-detector"

    @property
    def provider(self) -> str:
        return "tests"

    async def unlikely_threshold(self, language) -> float | None:
        return self._threshold

    async def supports_language(self, language) -> bool:
        return True

    async def predict_end_of_turn_audio(
        self, turn_ctx: AudioTurnContext, *, timeout: float | None = None
    ) -> float:
        self.contexts.append(
            AudioTurnContext(
                audio=list(turn_ctx.audio),
                transcript=turn_ctx.transcript,
                chat_ctx=turn_ctx.chat_ctx.copy(),
                language=turn_ctx.language,
            )
        )
        return self._probability


async def _run_session_with_streamed_audio(
    session,
    agent: Agent,
    *,
    frame_duration: float = 0.02,
    drain_delay: float = 0.2,
) -> float:
    stt = session.stt
    audio_input = session.input.audio
    assert isinstance(stt, FakeSTT)
    assert isinstance(audio_input, FakeAudioInput)

    transcription_sync = None
    if isinstance(session.output.audio, _SyncedAudioOutput):
        transcription_sync = session.output.audio._synchronizer

    stop_audio = asyncio.Event()

    async def _pump_audio() -> None:
        while not stop_audio.is_set():
            audio_input.push(frame_duration)
            await asyncio.sleep(frame_duration)

    await session.start(agent)
    audio_task = asyncio.create_task(_pump_audio())
    t_origin = time.time()

    try:
        await stt.fake_user_speeches_done
        await asyncio.sleep(drain_delay)
        await session.drain()
        await session.aclose()
    finally:
        stop_audio.set()
        audio_task.cancel()
        with suppress(asyncio.CancelledError):
            await audio_task
        if transcription_sync is not None:
            await transcription_sync.aclose()

    return t_origin


async def test_audio_turn_detector_receives_audio_context() -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "Hello there", stt_delay=0.2)
    actions.add_llm("Hi there!", ttft=0.1, duration=0.3)
    actions.add_tts(1.0, ttfb=0.1, duration=0.2)

    detector = _FakeAudioTurnDetector(probability=0.95)
    session = create_session(actions, speed_factor=speed)
    session.update_options(turn_detection=detector)

    await asyncio.wait_for(
        _run_session_with_streamed_audio(session, Agent(instructions="You are helpful.")),
        timeout=SESSION_TIMEOUT,
    )

    assert len(detector.contexts) == 1
    turn_ctx = detector.contexts[0]
    assert turn_ctx.transcript == "Hello there"
    assert turn_ctx.chat_ctx.items[-1].type == "message"
    assert turn_ctx.chat_ctx.items[-1].role == "user"
    assert turn_ctx.chat_ctx.items[-1].text_content == "Hello there"
    assert calculate_audio_duration(turn_ctx.audio) >= 0.25


async def test_audio_turn_detector_controls_endpointing_delay() -> None:
    async def _run_turn(probability: float) -> float:
        speed = 5.0
        actions = FakeActions()
        actions.add_user_speech(0.5, 1.5, "hello", stt_delay=0.2)
        actions.add_llm("hi", ttft=0.1, duration=0.2)
        actions.add_tts(0.5, ttfb=0.1, duration=0.2)

        detector = _FakeAudioTurnDetector(probability=probability, threshold=0.5)
        session = create_session(actions, speed_factor=speed)
        session.update_options(
            endpointing_opts={"min_delay": 0.05, "max_delay": 0.25},
            turn_detection=detector,
        )

        conversation_events: list[ConversationItemAddedEvent] = []
        session.on("conversation_item_added", conversation_events.append)

        t_origin = await _run_session_with_streamed_audio(
            session, Agent(instructions="You are helpful.")
        )
        user_event = next(
            ev for ev in conversation_events if ev.item.type == "message" and ev.item.role == "user"
        )
        return user_event.created_at - t_origin

    fast_commit_time = await asyncio.wait_for(_run_turn(0.95), timeout=SESSION_TIMEOUT)
    slow_commit_time = await asyncio.wait_for(_run_turn(0.05), timeout=SESSION_TIMEOUT)

    assert slow_commit_time - fast_commit_time >= 0.12


async def test_audio_turn_detector_buffer_resets_between_turns() -> None:
    speed = 5.0
    actions = FakeActions()
    actions.add_user_speech(0.5, 1.5, "First turn", stt_delay=0.2)
    actions.add_llm("first response", ttft=0.1, duration=0.2)
    actions.add_tts(0.4, ttfb=0.1, duration=0.2)
    actions.add_user_speech(2.8, 3.8, "Second turn", stt_delay=0.2)
    actions.add_llm("second response", ttft=0.1, duration=0.2)
    actions.add_tts(0.4, ttfb=0.1, duration=0.2)

    detector = _FakeAudioTurnDetector(probability=0.95)
    session = create_session(actions, speed_factor=speed)
    session.update_options(turn_detection=detector)

    await asyncio.wait_for(
        _run_session_with_streamed_audio(session, Agent(instructions="You are helpful.")),
        timeout=SESSION_TIMEOUT,
    )

    assert [ctx.transcript for ctx in detector.contexts] == ["First turn", "Second turn"]
    first_duration = calculate_audio_duration(detector.contexts[0].audio)
    second_duration = calculate_audio_duration(detector.contexts[1].audio)

    assert first_duration < 0.6
    assert second_duration < 0.6
