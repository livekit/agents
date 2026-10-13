"""Provider timestamps are relative; the recognizer adds the pipeline's ``input_started_at``.

Both sides must agree on where zero is. The default ``stt_node`` shifts provider timestamps
onto the session timeline through ``start_time_offset`` and anchors the pipeline on the same
session input start. A custom node that forwards provider timestamps untouched is anchored
on the pipeline's creation, when it opens its stream. The stream itself accounts for audio
that starts after its creation. Either way ``input_started_at + end_time`` is the wall-clock
end of the recognized speech, also after a handoff (AGT-3678) and when the first audio
arrives long after the stream opened.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator
from typing import Literal

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, ModelSettings, stt
from livekit.agents.utils import aio
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
    and calls the default one (a new pipeline per agent, the AGT-3678 setup), ``custom``
    forwards the provider's stream-relative timestamps as they are.
    """
    now = 1000.0
    monkeypatch.setattr(time, "time", lambda: now)
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

        first_audio_at = stream_audio_started_at = now + audio_delay
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
                stream_audio_started_at = now + audio_delay
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
        # the provider counts the audio pushed to its stream and adds the stream offset,
        # like every plugin; a reused stream has been fed since its first frame
        provider_end_time = (
            stream.start_time_offset + (first_audio_at - stream_audio_started_at) + 0.6
        )
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

        # the event reaches the session untouched: consumers keep the timeline they had
        assert received[0].alternatives[0].end_time == provider_end_time
        if node_kind == "custom":
            # no offset was declared; the stream only accounts for the delay to its audio
            assert stream.start_time_offset == pytest.approx(audio_delay)

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
