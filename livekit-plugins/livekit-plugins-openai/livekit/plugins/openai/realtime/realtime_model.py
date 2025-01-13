from __future__ import annotations

import asyncio
import openai
import base64

from livekit.agents.llm.function_context import _create_ai_function_info
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime import (
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeClientEvent,
    InputAudioBufferAppendEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreateEvent,
    ConversationItemTruncateEvent,
    ResponseCreatedEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
)

from dataclasses import dataclass

from livekit.agents import multimodal, llm, utils
from livekit import rtc

from .log import logger


# When a response is created with the OpenAI Realtime API, those events are sent in this order:
# 1. response.created (contains resp_id)
# 2. response.output_item.added (contains item_id)
# 3. conversation.item.created
# 4. response.content_part.added (type audio/text)
# 5. response.audio_transcript.delta (x2, x3, x4, etc)
# 6. response.audio.delta (x2, x3, x4, etc)
# 7. response.content_part.done
# 8. response.output_item.done (contains item_status: "completed/incomplete")
# 9. response.done (contains status_details for cancelled/failed/turn_detected/content_filter)
#
# Ourcode assumes a response will generate only one item with type "message."


SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _RealtimeOptions:
    model: str


@dataclass
class _ResponseGeneration:
    response_id: str
    item_id: str
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    text_ch: utils.aio.Chan[str]
    tool_calls_ch: utils.aio.Chan[llm.FunctionCallInfo]


class RealtimeModel(multimodal.RealtimeModel):
    def __init__(
        self,
        *,
        model: str = "gpt-4o-realtime-preview-2024-12-17",
        client: openai.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=multimodal.RealtimeCapabilities(message_truncation=True)
        )

        self._opts = _RealtimeOptions(model=model)
        self._client = client or openai.AsyncClient()

    def session(self) -> "RealtimeSession":
        return RealtimeSession(self)

    async def aclose(self) -> None: ...


class RealtimeSession(multimodal.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._chat_ctx = llm.ChatContext()
        self._fnc_ctx: llm.FunctionContext | None = None
        self._msg_ch = utils.aio.Chan[RealtimeClientEvent]()

        self._conn: AsyncRealtimeConnection | None = None
        self._main_atask = asyncio.create_task(
            self._main_task(), name="RealtimeSession._main_task"
        )

        self._current_generation: _ResponseGeneration | None = None

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        self._conn = conn = await self._realtime_model._client.beta.realtime.connect(
            model=self._realtime_model._opts.model
        ).enter()

        @utils.log_exceptions(logger=logger)
        async def _listen_for_events() -> None:
            async for event in conn:
                if event.type == "input_audio_buffer.speech_started":
                    self._handle_input_audio_buffer_speech_started(event)
                elif event.type == "input_audio_buffer.speech_stopped":
                    self._handle_input_audio_buffer_speech_stopped(event)
                elif event.type == "response.created":
                    self._handle_response_created(event)
                elif event.type == "response.output_item.added":
                    self._handle_response_output_item_added(event)
                elif event.type == "response.audio_transcript.delta":
                    self._handle_response_audio_transcript_delta(event)
                elif event.type == "response.audio.delta":
                    self._handle_response_audio_delta(event)
                elif event.type == "response.audio_transcript.done":
                    self._handle_response_audio_transcript_done(event)
                elif event.type == "response.audio.done":
                    self._handle_response_audio_done(event)
                elif event.type == "response.output_item.done":
                    self._handle_response_output_item_done(event)
                elif event.type == "response.done":
                    self._handle_response_done(event)

        @utils.log_exceptions(logger=logger)
        async def _forward_input_audio() -> None:
            async for msg in self._msg_ch:
                await conn.send(msg)

        tasks = [
            asyncio.create_task(_listen_for_events(), name="_listen_for_events"),
            asyncio.create_task(_forward_input_audio(), name="_forward_input_audio"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.gracefully_cancel(*tasks)
            await conn.close()

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self.emit("input_speech_started", multimodal.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        self.emit("input_speech_stopped", multimodal.InputSpeechStoppedEvent())

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        response_id = event.response.id
        assert response_id is not None, "response.id is None"
        self._current_generation = _ResponseGeneration(
            response_id=response_id,
            item_id="",
            audio_ch=utils.aio.Chan(),
            text_ch=utils.aio.Chan(),
            tool_calls_ch=utils.aio.Chan(),
        )

    def _handle_response_output_item_added(
        self, event: ResponseOutputItemAddedEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_id = event.item.id
        assert item_id is not None, "item.id is None"

        # We assume only one "message" item in the current approach
        if self._current_generation.item_id and event.item.type == "message":
            logger.warning("Received an unexpected second item with type `message`")
            return

        if event.item.type == "function_call":
            return

        self._current_generation.item_id = item_id

    def _handle_response_audio_transcript_delta(
        self, event: ResponseAudioTranscriptDeltaEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        self._current_generation.text_ch.send_nowait(event.delta)

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        data = base64.b64decode(event.delta)
        frame = rtc.AudioFrame(
            data=data,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(data) // 2,
        )
        self._current_generation.audio_ch.send_nowait(frame)

    def _handle_response_audio_transcript_done(
        self, _: ResponseAudioTranscriptDoneEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        self._current_generation.text_ch.close()

    def _handle_response_audio_done(self, _: ResponseAudioDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        self._current_generation.audio_ch.close()

    def _handle_response_output_item_done(
        self, event: ResponseOutputItemDoneEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"

        item = event.item
        if item.type == "function_call":
            if self.fnc_ctx is None:
                logger.warning(
                    "received a function_call item without a function context",
                    extra={"item": item},
                )
                return

            assert item.call_id is not None, "call_id is None"
            assert item.name is not None, "name is None"
            assert item.arguments is not None, "arguments is None"

            fnc_call_info = _create_ai_function_info(
                self.fnc_ctx,
                item.call_id,
                item.name,
                item.arguments,
            )
            self._current_generation.tool_calls_ch.send_nowait(fnc_call_info)

    def _handle_response_done(self, _: ResponseDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        self._current_generation.tool_calls_ch.close()
        self._current_generation = None

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        # get the difference between self._chat_ctx and chat_ctx
        pass

    @property
    def fnc_ctx(self) -> llm.FunctionContext | None:
        return self._fnc_ctx

    async def update_fnc_ctx(self, fnc_ctx: llm.FunctionContext | None) -> None:
        pass

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._msg_ch.send_nowait(
            InputAudioBufferAppendEvent(
                type="input_audio_buffer.append",
                audio=base64.b64encode(frame.data).decode("ascii"),
            )
        )

    def generate_reply(self) -> None:
        self._msg_ch.send_nowait(ResponseCreateEvent(type="response.create"))

    def interrupt(self) -> None:
        self._msg_ch.send_nowait(ResponseCancelEvent(type="response.cancel"))

    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
        self._msg_ch.send_nowait(
            ConversationItemTruncateEvent(
                type="conversation.item.truncate",
                content_index=0,
                item_id=message_id,
                audio_end_ms=audio_end_ms,
            )
        )

    async def aclose(self) -> None:
        if self._conn is not None:
            await self._conn.close()
