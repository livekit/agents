"""Provider timestamps are relative; the recognizer adds the pipeline's ``input_started_at``.

Both sides must agree on where zero is. The default ``stt_node`` shifts provider timestamps
onto the session timeline through ``start_time_offset`` and anchors the pipeline on the same
session input start. A custom node that forwards provider timestamps untouched counts from
its first audio: the pipeline is anchored on the first frame pushed, and a stream with no
declared offset starts its timeline there too. Either way ``input_started_at + end_time`` is the wall-clock
end of the recognized speech, also after a handoff and when the first audio
arrives long after the stream opened.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator
from typing import Literal

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, APIStatusError, ModelSettings, stt
from livekit.agents.utils import aio
from livekit.agents.voice import audio_recognition
from livekit.agents.voice.audio_recognition import _compute_end_of_turn_metrics

from .fake_llm import FakeLLM
from .fake_stt import FakeSTT

pytestmark = [pytest.mark.unit, pytest.mark.no_concurrent]

NodeKind = Literal["default", "delegate", "custom"]


@pytest.mark.parametrize("node_kind", ["default", "delegate", "custom"])
@pytest.mark.parametrize("handoff_after", [0.0, 300.0])
@pytest.mark.parametrize("audio_delay", [0.0, 20.0])
async def test_speech_timestamps_follow_first_audio(
    node_kind: NodeKind,
    handoff_after: float,
    audio_delay: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stopped_speaking_at`` is the wall-clock end of the last word on every stream.

    ``default`` reuses the pipeline across the handoff, ``delegate`` overrides ``stt_node``
    and calls the default one (a new pipeline per agent), ``custom``
    forwards the provider's stream-relative timestamps as they are.
    """
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
    # the input clock measures elapsed time on the monotonic clock
    monkeypatch.setattr(time, "perf_counter", lambda: now)
    provider = FakeSTT()

    class DelegatingAgent(Agent):
        async def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncIterator[stt.SpeechEvent]:
            async for event in Agent.default.stt_node(self, audio, model_settings):
                yield event

    class CustomAgent(Agent):
        async def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncIterator[stt.SpeechEvent]:
            async with provider.stream() as stream:

                async def forward() -> None:
                    async for frame in audio:
                        stream.push_frame(frame)

                forward_task = asyncio.create_task(forward())
                try:
                    async for event in stream:
                        yield event
                finally:
                    await aio.cancel_and_wait(forward_task)

    agent_cls = {"default": Agent, "delegate": DelegatingAgent, "custom": CustomAgent}[node_kind]
    session = AgentSession(
        stt=provider,
        llm=FakeLLM(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        session_close_transcript_timeout=0.0,
        aec_warmup_duration=None,
    )
    await session.start(agent_cls(instructions="test"))
    try:
        stream = await asyncio.wait_for(provider.stream_ch.recv(), timeout=1.0)
        frame = rtc.AudioFrame.create(16000, 1, 320)

        async def push_audio(captured_at: float) -> None:
            nonlocal now
            pushed = asyncio.Event()
            push_frame = stream.push_frame

            def on_frame(frame: rtc.AudioFrame) -> None:
                push_frame(frame)
                pushed.set()

            monkeypatch.setattr(stream, "push_frame", on_frame)
            now = captured_at + frame.duration
            session._activity._audio_recognition._push_audio(frame)
            await asyncio.wait_for(pushed.wait(), timeout=1.0)
            monkeypatch.setattr(stream, "push_frame", push_frame)

        first_audio_at = now + audio_delay
        await push_audio(first_audio_at)
        if handoff_after:
            now = first_audio_at + handoff_after
            previous_activity = session._activity
            session.update_agent(agent_cls(instructions="handoff"))
            if node_kind == "default":
                # the default node is reused: same pipeline, same stream
                while session._activity is previous_activity:
                    await asyncio.sleep(0.01)
            else:
                stream = await asyncio.wait_for(provider.stream_ch.recv(), timeout=1.0)
            first_audio_at = now + audio_delay
            await push_audio(first_audio_at)

        recognition = session._activity._audio_recognition
        assert recognition is not None
        received: list[stt.SpeechEvent] = []
        on_final_transcript = recognition._hooks.on_final_transcript

        def on_final(event: stt.SpeechEvent, *, speaking: bool | None = None) -> None:
            received.append(event)
            on_final_transcript(event, speaking=speaking)

        monkeypatch.setattr(recognition._hooks, "on_final_transcript", on_final)

        now = first_audio_at + 0.2
        await recognition._on_stt_event(
            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH, created_at=now)
        )
        # The word ends 0.6s after the last frame starts. A reused stream has also
        # received the pre-handoff frame.
        while not stream._input_ch.empty():
            await asyncio.sleep(0)
        provider_end_time = stream._input_clock.dequeued_duration - frame.duration + 0.6
        now = first_audio_at + 0.9
        stream._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(language="en", text="hello", end_time=provider_end_time)
                ],
                created_at=now,
            )
        )
        await asyncio.wait_for(recognition._final_transcript_received.wait(), timeout=1.0)

        # the event reaches the session on the session's timeline
        assert recognition._input_started_at is not None
        assert recognition._input_started_at + received[0].alternatives[
            0
        ].end_time == pytest.approx(first_audio_at + 0.6)
        if node_kind == "custom":
            # no offset was declared: the stream's timeline starts with its first audio
            assert stream.start_time_offset == 0.0

        metrics = _compute_end_of_turn_metrics(
            speech_start_time=recognition._speech_start_time,
            last_speaking_time=recognition._last_speaking_time,
            last_final_transcript_time=recognition._last_final_transcript_time,
            now=first_audio_at + 1.2,
        )
        assert recognition._last_speaking_time == pytest.approx(first_audio_at + 0.6)
        assert metrics.stopped_speaking_at == pytest.approx(first_audio_at + 0.6)
        assert metrics.transcription_delay == pytest.approx(0.3)
        assert metrics.end_of_turn_delay == pytest.approx(0.6)
    finally:
        await session.aclose()


@pytest.mark.parametrize("node_kind", ["default", "custom"])
async def test_speech_timestamps_follow_the_audio_across_input_gaps(
    node_kind: Literal["default", "custom"], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve user-turn timing after gaps and for delayed pre-gap results."""
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
    # the input clock measures elapsed time on the monotonic clock
    monkeypatch.setattr(time, "perf_counter", lambda: now)
    provider = FakeSTT()

    class CustomAgent(Agent):
        async def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncIterator[stt.SpeechEvent]:
            async with provider.stream() as stream:

                async def forward() -> None:
                    async for frame in audio:
                        stream.push_frame(frame)

                forward_task = asyncio.create_task(forward())
                try:
                    async for event in stream:
                        yield event
                finally:
                    await aio.cancel_and_wait(forward_task)

    session = AgentSession(
        stt=provider,
        llm=FakeLLM(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        session_close_transcript_timeout=0.0,
        aec_warmup_duration=None,
    )
    await session.start((Agent if node_kind == "default" else CustomAgent)(instructions="test"))
    try:
        stream = await asyncio.wait_for(provider.stream_ch.recv(), timeout=1.0)
        recognition = session._activity._audio_recognition
        assert recognition is not None
        frame = rtc.AudioFrame.create(16000, 1, 1600)
        pushed = 0
        push_frame = stream.push_frame

        def on_frame(frame: rtc.AudioFrame) -> None:
            nonlocal pushed
            push_frame(frame)
            pushed += 1

        monkeypatch.setattr(stream, "push_frame", on_frame)

        async def push_audio(captured_from: float, duration: float) -> None:
            nonlocal now
            for i in range(round(duration / frame.duration)):
                now = captured_from + (i + 1) * frame.duration
                target = pushed + 1
                recognition._push_audio(frame)
                while pushed < target or not stream._input_ch.empty():
                    await asyncio.sleep(0)

        # the provider receives 4s of audio in 12s: 1010-1011, 1016-1017 and 1020-1022
        await push_audio(1010.0, 1.0)
        await push_audio(1016.0, 1.0)
        await push_audio(1020.0, 2.0)

        now = 1020.9
        await recognition._on_stt_event(
            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH, created_at=now)
        )
        # the last word ends 3.4s into the provider's audio, at 1021.4
        now = 1021.7
        stream._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="en", text="hello", end_time=3.4)],
                created_at=now,
            )
        )
        await asyncio.wait_for(recognition._final_transcript_received.wait(), timeout=1.0)

        metrics = _compute_end_of_turn_metrics(
            speech_start_time=recognition._speech_start_time,
            last_speaking_time=recognition._last_speaking_time,
            last_final_transcript_time=recognition._last_final_transcript_time,
            now=1022.0,
        )
        assert recognition._last_speaking_time == pytest.approx(1021.4)
        assert metrics.stopped_speaking_at == pytest.approx(1021.4)
        assert metrics.transcription_delay == pytest.approx(0.3)

        # results for speech before each pause, arriving after it
        input_started_at = recognition._input_started_at
        assert input_started_at is not None
        for provider_end, spoken_until in ((0.6, 1010.6), (1.6, 1016.6)):
            stream._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language="en", text="hi", end_time=provider_end)],
                )
            )
            late = stream._event_ch._queue[-1].alternatives[0]
            assert input_started_at + late.end_time == pytest.approx(spoken_until)
    finally:
        await session.aclose()


async def test_node_without_a_stream_counts_from_its_first_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node that builds events itself, relative to its first frame, is anchored there."""
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
    # the input clock measures elapsed time on the monotonic clock
    monkeypatch.setattr(time, "perf_counter", lambda: now)
    first_frame = asyncio.Event()
    events = aio.Chan[stt.SpeechEvent]()

    class ManualAgent(Agent):
        async def stt_node(
            self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
        ) -> AsyncIterator[stt.SpeechEvent]:
            async for _ in audio:
                first_frame.set()
                break
            yield await events.recv()

    session = AgentSession(
        stt=FakeSTT(),
        llm=FakeLLM(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        session_close_transcript_timeout=0.0,
        aec_warmup_duration=None,
    )
    await session.start(ManualAgent(instructions="test"))
    try:
        recognition = session._activity._audio_recognition
        assert recognition is not None
        frame = rtc.AudioFrame.create(16000, 1, 320)
        first_audio_at = 1020.0
        now = first_audio_at + frame.duration
        recognition._push_audio(frame)
        await asyncio.wait_for(first_frame.wait(), timeout=1.0)

        now = first_audio_at + 0.2
        await recognition._on_stt_event(
            stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH, created_at=now)
        )
        now = first_audio_at + 0.9
        events.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="en", text="hello", end_time=0.6)],
                created_at=now,
            )
        )
        await asyncio.wait_for(recognition._final_transcript_received.wait(), timeout=1.0)

        assert recognition._last_speaking_time == pytest.approx(first_audio_at + 0.6)
    finally:
        await session.aclose()


@pytest.mark.parametrize("step", [3600.0, -3600.0])
async def test_recreated_stream_keeps_the_session_start_across_a_wall_clock_step(
    step: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream recreated after a connection failure continues the pipeline's timeline
    from the same session start, whatever the wall clock did since."""
    monotonic = 50.0
    wall_offset = 950.0
    monkeypatch.setattr(time, "time", lambda: monotonic + wall_offset)
    monkeypatch.setattr(time, "perf_counter", lambda: monotonic)
    monkeypatch.setattr(audio_recognition, "_STT_RECONNECT_INTERVAL", 0.0)
    provider = FakeSTT()
    session = AgentSession(
        stt=provider,
        llm=FakeLLM(),
        vad=None,
        turn_handling={"turn_detection": "stt"},
        session_close_transcript_timeout=0.0,
        aec_warmup_duration=None,
    )
    await session.start(Agent(instructions="test"))
    try:
        stream = await asyncio.wait_for(provider.stream_ch.recv(), timeout=1.0)
        session_start = stream._input_clock.session_start
        assert session_start is not None

        monotonic += 20.0
        wall_offset += step
        # the connection fails for good; the pipeline recreates the stream
        provider.update_options(fake_exception=APIStatusError("Unauthorized", status_code=401))
        stream._input_ch.close()
        stream = await asyncio.wait_for(provider.stream_ch.recv(), timeout=1.0)
        provider.update_options(fake_exception=None)

        assert stream._input_clock.session_start is session_start
        recognition = session._activity._audio_recognition
        assert recognition is not None
        assert recognition._input_started_at == session_start.wall_timestamp

        frame = rtc.AudioFrame.create(16000, 1, 1600)
        captured_from = monotonic
        monotonic += frame.duration
        recognition._push_audio(frame)
        while stream._input_clock.dequeued_duration < frame.duration:
            await asyncio.sleep(0)
        stream._event_ch.send_nowait(
            stt.SpeechEvent(
                type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="en", text="hi", end_time=0.06)],
            )
        )
        placed = stream._event_ch._queue[-1].alternatives[0]
        assert placed.end_time == pytest.approx(
            captured_from + 0.06 - session_start.monotonic_timestamp
        )
    finally:
        await session.aclose()
