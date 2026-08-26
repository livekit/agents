from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from livekit.agents import Agent, AgentSession, TurnHandlingOptions, llm
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_io import FakeAudioInput
from .fake_realtime import FakeRealtimeModel, fake_capabilities
from .fake_stt import FakeSTT, FakeUserSpeech
from .fake_vad import FakeVAD

pytestmark = pytest.mark.unit


class _EditingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="test")

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        new_message.content = [f"edited: {new_message.raw_text_content}"]


async def _wait_for_generation(model: FakeRealtimeModel) -> None:
    while not model.active_session._reply_futs:
        await asyncio.sleep(0)


async def _wait_for_turn_reset(session: AgentSession) -> None:
    while True:
        activity = session._activity
        recognition = activity._audio_recognition if activity is not None else None
        completion = activity._user_turn_completed_atask if activity is not None else None
        if (
            recognition is not None
            and recognition._user_turn_start is None
            and (completion is None or completion.done())
        ):
            return
        await asyncio.sleep(0)


async def test_external_stt_text_turn_reaches_realtime_model_once_without_audio() -> None:
    speech = FakeUserSpeech(
        start_time=0.0,
        end_time=0.02,
        transcript="hello from external stt",
        stt_delay=0.0,
    )
    stt = FakeSTT(fake_user_speeches=[speech])
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=stt,
        vad=FakeVAD(
            fake_user_speeches=[speech],
            min_speech_duration=0.005,
            min_silence_duration=0.005,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
            endpointing={"min_delay": 0.0, "max_delay": 0.5},
        ),
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    agent = _EditingAgent()
    conversation_events: list[ConversationItemAddedEvent] = []
    session.on("conversation_item_added", conversation_events.append)

    async with session:
        await session.start(agent)
        audio_input.push(0.05)

        await asyncio.wait_for(stt.fake_user_speeches_done, timeout=1.0)
        await asyncio.wait_for(_wait_for_generation(model), timeout=1.0)

        provider = model.active_session
        provider_messages = provider.chat_ctx.messages()
        local_messages = agent.chat_ctx.messages()
        user_events = [
            event.item
            for event in conversation_events
            if isinstance(event.item, llm.ChatMessage) and event.item.role == "user"
        ]

        assert provider.generate_reply_calls == 1
        assert len(provider_messages) == len(local_messages) == len(user_events) == 1
        assert provider_messages[0].id == local_messages[0].id == user_events[0].id
        assert provider_messages[0].raw_text_content == "edited: hello from external stt"
        assert provider.pushed_audio == []
        assert provider.committed is False


async def test_empty_external_stt_text_turn_is_dropped_and_reset_after_timeout() -> None:
    speech = FakeUserSpeech(start_time=0.0, end_time=0.02, transcript="", stt_delay=0.0)
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech],
            min_speech_duration=0.005,
            min_silence_duration=0.005,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="text",
            endpointing={"min_delay": 0.0, "max_delay": 0.0},
        ),
        transcription_timeout=0.02,
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    timeout_seen = asyncio.Event()
    session.on("user_transcription_timeout", lambda _: timeout_seen.set())
    agent = Agent(instructions="test")

    async with session:
        await session.start(agent)
        audio_input.push(0.05)

        await asyncio.wait_for(timeout_seen.wait(), timeout=1.0)
        await asyncio.wait_for(_wait_for_turn_reset(session), timeout=1.0)

        provider = model.active_session
        recognition = session._activity._audio_recognition
        assert recognition is not None
        assert provider.generate_reply_calls == 0
        assert provider.pushed_audio == []
        assert provider.chat_ctx.messages() == []
        assert agent.chat_ctx.messages() == []
        assert recognition._audio_transcript == ""
        assert recognition._audio_interim_transcript == ""
        assert recognition._transcription_timeout_handle is None


@pytest.mark.parametrize("realtime_input_mode", ["text", "audio"])
async def test_unset_transcription_timeout_finalizes_empty_turn_without_event(
    realtime_input_mode: Literal["text", "audio"],
) -> None:
    speech = FakeUserSpeech(start_time=0.0, end_time=0.02, transcript="", stt_delay=0.0)
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech],
            min_speech_duration=0.005,
            min_silence_duration=0.005,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode=realtime_input_mode,
            endpointing={"min_delay": 0.0, "max_delay": 0.02},
        ),
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    timeout_events: list[object] = []
    speaking_seen = asyncio.Event()
    session.on("user_transcription_timeout", timeout_events.append)
    session.on(
        "user_state_changed",
        lambda event: speaking_seen.set() if event.new_state == "speaking" else None,
    )
    agent = Agent(instructions="test")

    async with session:
        await session.start(agent)
        audio_input.push(0.05)

        await asyncio.wait_for(speaking_seen.wait(), timeout=1.0)
        if realtime_input_mode == "audio":
            await asyncio.wait_for(_wait_for_generation(model), timeout=1.0)
        await asyncio.wait_for(_wait_for_turn_reset(session), timeout=1.0)

        provider = model.active_session
        recognition = session._activity._audio_recognition
        assert recognition is not None
        assert session.options.transcription_timeout is None
        assert timeout_events == []
        assert recognition._user_turn_start is None
        assert recognition._transcription_timeout_handle is None
        assert recognition._audio_transcript == ""
        assert recognition._audio_interim_transcript == ""
        if realtime_input_mode == "audio":
            assert provider.generate_reply_calls == 1
            assert provider.committed is True
            assert provider.pushed_audio
        else:
            assert provider.generate_reply_calls == 0
            assert provider.pushed_audio == []
        assert provider.chat_ctx.messages() == []
        assert agent.chat_ctx.messages() == []


async def test_empty_external_stt_audio_turn_settles_provider_once_after_timeout() -> None:
    speech = FakeUserSpeech(start_time=0.0, end_time=0.02, transcript="", stt_delay=0.0)
    model = FakeRealtimeModel(
        capabilities=fake_capabilities(
            turn_detection=False,
            can_disable_turn_detection=False,
            mutable_chat_context=True,
        )
    )
    session = AgentSession(
        llm=model,
        stt=FakeSTT(fake_user_speeches=[speech]),
        vad=FakeVAD(
            fake_user_speeches=[speech],
            min_speech_duration=0.005,
            min_silence_duration=0.005,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            realtime_input_mode="audio",
            endpointing={"min_delay": 0.0, "max_delay": 0.0},
        ),
        transcription_timeout=0.02,
        aec_warmup_duration=None,
    )
    audio_input = FakeAudioInput()
    session.input.audio = audio_input
    timeout_seen = asyncio.Event()
    session.on("user_transcription_timeout", lambda _: timeout_seen.set())

    async with session:
        await session.start(Agent(instructions="test"))
        audio_input.push(0.05)

        await asyncio.wait_for(timeout_seen.wait(), timeout=1.0)
        await asyncio.wait_for(_wait_for_generation(model), timeout=1.0)
        await asyncio.wait_for(_wait_for_turn_reset(session), timeout=1.0)

        provider = model.active_session
        recognition = session._activity._audio_recognition
        assert recognition is not None
        assert provider.user_activity_start_calls == 1
        assert provider.committed is True
        assert provider.generate_reply_calls == 1
        assert len(provider.pushed_audio) > 0
        assert provider.chat_ctx.messages() == []
        assert recognition._audio_transcript == ""
        assert recognition._audio_interim_transcript == ""
        assert recognition._transcription_timeout_handle is None
