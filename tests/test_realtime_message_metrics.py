import asyncio

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
