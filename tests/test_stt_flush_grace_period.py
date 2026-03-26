"""Tests for phantom turn suppression after turn commit.

Verifies that late FINAL_TRANSCRIPT events from streaming STT providers
are suppressed instead of creating phantom user turns.
"""

from __future__ import annotations

import asyncio

from livekit.agents import (
    Agent,
    AgentSession,
    ConversationItemAddedEvent,
    EndpointingOptions,
    InterruptionOptions,
    TurnHandlingOptions,
)
from livekit.agents.voice.transcription.synchronizer import (
    TranscriptSynchronizer,
    _SyncedAudioOutput,
)

from .fake_io import FakeAudioInput, FakeAudioOutput, FakeTextOutput
from .fake_llm import FakeLLM, FakeLLMResponse
from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_tts import FakeTTS, FakeTTSResponse
from .fake_vad import FakeVAD

SESSION_TIMEOUT = 60.0


class SimpleAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are a helpful assistant.")


def _create_split_session(
    *,
    stt_speeches: list[FakeUserSpeech],
    vad_speeches: list[FakeUserSpeech],
    llm_responses: list[FakeLLMResponse],
    tts_responses: list[FakeTTSResponse],
    min_endpointing_delay: float = 0.1,
    allow_interruptions: bool = True,
) -> AgentSession:
    """Create a session with separate STT and VAD speech configs."""
    session = AgentSession[None](
        vad=FakeVAD(
            fake_user_speeches=vad_speeches,
            min_silence_duration=0.1,
            min_speech_duration=0.05,
        ),
        stt=FakeSTT(fake_user_speeches=stt_speeches),
        llm=FakeLLM(fake_responses=llm_responses),
        tts=FakeTTS(fake_responses=tts_responses),
        turn_handling=TurnHandlingOptions(
            endpointing=EndpointingOptions(
                min_delay=min_endpointing_delay,
                max_delay=6.0,
            ),
            interruption=InterruptionOptions(
                enabled=allow_interruptions,
                min_duration=0.0,
                min_words=0,
            ),
        ),
        aec_warmup_duration=None,
    )

    audio_input = FakeAudioInput()
    audio_output = FakeAudioOutput()
    transcription_output = FakeTextOutput()

    transcript_sync = TranscriptSynchronizer(
        next_in_chain_audio=audio_output,
        next_in_chain_text=transcription_output,
    )
    session.input.audio = audio_input
    session.output.audio = transcript_sync.audio_output
    session.output.transcription = transcript_sync.text_output

    return session


async def _run_session(session: AgentSession, agent: Agent, *, drain_delay: float = 1.0) -> None:
    stt = session.stt
    audio_input = session.input.audio
    assert isinstance(stt, FakeSTT)
    assert isinstance(audio_input, FakeAudioInput)

    transcription_sync: TranscriptSynchronizer | None = None
    if isinstance(session.output.audio, _SyncedAudioOutput):
        transcription_sync = session.output.audio._synchronizer

    await session.start(agent)
    audio_input.push(0.1)

    await stt.fake_user_speeches_done
    await asyncio.sleep(drain_delay)
    await session.drain()
    await session.aclose()

    if transcription_sync is not None:
        await transcription_sync.aclose()


async def test_late_stt_transcript_suppressed() -> None:
    """A late FINAL_TRANSCRIPT arriving after turn commit should not create a phantom turn.

    Scenario: user says a long utterance. The STT provider delivers:
      - segment 1 quickly (stt_delay=0.05s)
      - segment 2 LATE (stt_delay=0.35s — arrives after the turn is committed)

    The VAD sees one continuous speech for both segments.

    With interruptions enabled the phantom would be processed as a real turn.
    The fix suppresses it because it arrives within _STT_FLUSH_GRACE_PERIOD
    after the turn commit and the user is not speaking (VAD).
    """
    # STT: two segments, second one arrives after the EOU commit
    stt_speeches = [
        FakeUserSpeech(
            start_time=0.1,
            end_time=0.3,
            transcript="Hello world",
            stt_delay=0.05,
        ),
        FakeUserSpeech(
            start_time=0.35,
            end_time=0.4,
            transcript="how are you?",
            stt_delay=0.35,  # arrives at 0.4 + 0.35 = 0.75s, after commit at ~0.6s
        ),
    ]

    # VAD: one continuous speech covering both segments
    vad_speeches = [
        FakeUserSpeech(
            start_time=0.1,
            end_time=0.4,
            transcript="",
            stt_delay=0.0,
        ),
    ]

    llm_responses = [
        FakeLLMResponse(
            input="Hello world",
            content="I'm doing great!",
            ttft=0.1,
            duration=0.1,
        ),
    ]

    tts_responses = [
        FakeTTSResponse(
            input="I'm doing great!",
            audio_duration=0.5,
            ttfb=0.05,
            duration=0.1,
        ),
    ]

    session = _create_split_session(
        stt_speeches=stt_speeches,
        vad_speeches=vad_speeches,
        llm_responses=llm_responses,
        tts_responses=tts_responses,
        min_endpointing_delay=0.1,
        allow_interruptions=True,
    )
    agent = SimpleAgent()

    conversation_events: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", conversation_events.append)

    await asyncio.wait_for(_run_session(session, agent), timeout=SESSION_TIMEOUT)

    user_messages = [
        ev
        for ev in conversation_events
        if ev.item.type == "message" and ev.item.role == "user"
    ]

    # key assertion: only ONE user message (no phantom turn)
    assert len(user_messages) == 1, (
        f"expected 1 user message but got {len(user_messages)}: "
        f"{[m.item.text_content for m in user_messages]}"
    )

    # the late transcript should be merged into the first user message
    transcript = user_messages[0].item.text_content or ""
    assert "Hello world" in transcript
    assert "how are you?" in transcript
