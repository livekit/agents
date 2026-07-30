import asyncio
import time

import pytest

from livekit import rtc
from livekit.agents import Agent, AgentSession, llm, utils
from livekit.agents.voice.events import ConversationItemAddedEvent

from .fake_realtime import FakeRealtimeModel, fake_capabilities

pytestmark = pytest.mark.unit


async def test_realtime_response_id_is_available_on_assistant_message() -> None:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    conversation_events: list[ConversationItemAddedEvent] = []

    async with AgentSession(llm=model) as session:
        session.on("conversation_item_added", conversation_events.append)
        await session.start(Agent(instructions="test"))

        speech_handle = session.generate_reply()
        while not model.active_session._reply_futs:
            await asyncio.sleep(0)

        message_ch = utils.aio.Chan[llm.MessageGeneration]()
        function_ch = utils.aio.Chan[llm.FunctionCall]()
        text_ch = utils.aio.Chan[str]()
        audio_ch = utils.aio.Chan[rtc.AudioFrame]()
        modalities = asyncio.Future[list[str]]()
        modalities.set_result(["text"])

        message_ch.send_nowait(
            llm.MessageGeneration(
                message_id="message-id",
                text_stream=text_ch,
                audio_stream=audio_ch,
                modalities=modalities,
            )
        )
        message_ch.close()
        function_ch.close()
        text_ch.send_nowait("Hello")
        text_ch.close()
        audio_ch.close()

        model.active_session._reply_futs[0].set_result(
            llm.GenerationCreatedEvent(
                message_stream=message_ch,
                function_stream=function_ch,
                user_initiated=True,
                response_id="provider-response-id",
            )
        )
        await speech_handle

    assistant_messages = [
        event.item
        for event in conversation_events
        if event.item.type == "message" and event.item.role == "assistant"
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0].metrics["provider_request_ids"] == ["provider-response-id"]


async def _transcribed_user_messages(
    *transcripts: llm.InputTranscriptionCompleted,
) -> list[llm.ChatMessage]:
    model = FakeRealtimeModel(capabilities=fake_capabilities(audio_output=False))
    conversation_events: list[ConversationItemAddedEvent] = []

    async with AgentSession(llm=model) as session:
        session.on("conversation_item_added", conversation_events.append)
        await session.start(Agent(instructions="test"))

        for transcript in transcripts:
            model.active_session.emit("input_audio_transcription_completed", transcript)
        await asyncio.sleep(0)

    return [
        event.item
        for event in conversation_events
        if event.item.type == "message" and event.item.role == "user"
    ]


async def test_realtime_user_message_placed_where_the_turn_began() -> None:
    # the provider withholds the transcript until its reply is done generating
    turn_started_at = 1_000_000.0

    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1",
            transcript="what is my name?",
            is_final=True,
            turn_started_at=turn_started_at,
        )
    )

    assert len(messages) == 1
    assert messages[0].created_at == turn_started_at
    assert messages[0].metrics["started_speaking_at"] == turn_started_at


async def test_realtime_late_transcripts_keep_their_own_turn_start() -> None:
    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1", transcript="first", is_final=True, turn_started_at=1_000_000.0
        ),
        llm.InputTranscriptionCompleted(
            item_id="item_2", transcript="second", is_final=True, turn_started_at=1_005_000.0
        ),
    )

    assert [msg.created_at for msg in messages] == [1_000_000.0, 1_005_000.0]


async def test_realtime_user_message_falls_back_to_delivery_time() -> None:
    before = time.time()

    messages = await _transcribed_user_messages(
        llm.InputTranscriptionCompleted(
            item_id="item_1", transcript="text injected turn", is_final=True
        )
    )

    assert len(messages) == 1
    assert before <= messages[0].created_at <= time.time()
    assert "started_speaking_at" not in messages[0].metrics
