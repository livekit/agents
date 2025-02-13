from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from pydantic import ValidationError

import openai
from openai.resources.beta.realtime.realtime import AsyncRealtimeConnection
from openai.types.beta.realtime import (
    ConversationItem,
    ConversationItemContent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemTruncateEvent,
    ErrorEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeClientEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
    ResponseAudioTranscriptDeltaEvent,
    ResponseAudioTranscriptDoneEvent,
    ResponseCancelEvent,
    ResponseCreatedEvent,
    ResponseCreateEvent,
    ResponseDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    SessionUpdateEvent,
    session_update_event,
)
from openai.types.beta.realtime.response_create_event import Response

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
# Ourcode assumes a response will generate only one item with type "message"


SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _RealtimeOptions:
    model: str
    voice: str


@dataclass
class _MessageGeneration:
    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]


@dataclass
class _ResponseGeneration:
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    function_ch: utils.aio.Chan[llm.FunctionCall]

    messages: dict[str, _MessageGeneration]


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        model: str = "gpt-4o-realtime-preview",
        voice: str = "alloy",
        client: openai.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(message_truncation=True)
        )

        self._opts = _RealtimeOptions(model=model, voice=voice)
        self._client = client or openai.AsyncClient()

    def session(self) -> "RealtimeSession":
        return RealtimeSession(self)

    async def aclose(self) -> None: ...


class RealtimeSession(llm.RealtimeSession):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._fnc_ctx = llm.FunctionContext.empty()
        self._msg_ch = utils.aio.Chan[RealtimeClientEvent]()

        self._conn: AsyncRealtimeConnection | None = None
        self._main_atask = asyncio.create_task(
            self._main_task(), name="RealtimeSession._main_task"
        )

        self._response_created_futures: dict[
            str, asyncio.Future[llm.GenerationCreatedEvent]
        ] = {}
        self._item_delete_future: dict[str, asyncio.Future] = {}
        self._item_create_future: dict[str, asyncio.Future] = {}

        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        # TODO(theomonnom): handle reconnections
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
                elif event.type == "conversation.item.created":
                    self._handle_conversion_item_created(event)
                elif event.type == "conversation.item.deleted":
                    self._handle_conversion_item_deleted(event)
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
                elif event.type == "error":
                    self._handle_error(event)

        @utils.log_exceptions(logger=logger)
        async def _forward_input() -> None:
            async for msg in self._msg_ch:
                try:
                    await conn.send(msg)
                except Exception:
                    break

        self._msg_ch.send_nowait(
            SessionUpdateEvent(
                type="session.update",
                session=session_update_event.Session(
                    model=self._realtime_model._opts.model,  # type: ignore
                    voice=self._realtime_model._opts.voice,  # type: ignore
                ),
                event_id=utils.shortuuid("session_update_"),
            )
        )

        tasks = [
            asyncio.create_task(_listen_for_events(), name="_listen_for_events"),
            asyncio.create_task(_forward_input(), name="_forward_input"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await conn.close()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def fnc_ctx(self) -> llm.FunctionContext:
        return self._fnc_ctx.copy()

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            diff_ops = llm.utils.compute_chat_ctx_diff(
                self._remote_chat_ctx.to_chat_ctx(), chat_ctx
            )

            futs = []

            for msg_id in diff_ops.to_remove:
                event_id = utils.shortuuid("chat_ctx_delete_")
                self._msg_ch.send_nowait(
                    ConversationItemDeleteEvent(
                        type="conversation.item.delete",
                        item_id=msg_id,
                        event_id=event_id,
                    )
                )
                futs.append(f := asyncio.Future())
                self._item_delete_future[msg_id] = f

            for previous_msg_id, msg_id in diff_ops.to_create:
                event_id = utils.shortuuid("chat_ctx_create_")
                chat_item = chat_ctx.get_by_id(msg_id)
                assert chat_item is not None

                self._msg_ch.send_nowait(
                    ConversationItemCreateEvent(
                        type="conversation.item.create",
                        item=_livekit_item_to_openai_item(chat_item),
                        previous_item_id=(
                            "root" if previous_msg_id is None else previous_msg_id
                        ),
                        event_id=event_id,
                    )
                )
                futs.append(f := asyncio.Future())
                self._item_create_future[msg_id] = f

            try:
                await asyncio.wait_for(
                    asyncio.gather(*futs, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                raise llm.RealtimeError("update_chat_ctx timed out.") from None

    async def update_fnc_ctx(
        self, fnc_ctx: llm.FunctionContext | list[llm.AIFunction]
    ) -> None:
        async with self._update_fnc_ctx_lock:
            if isinstance(fnc_ctx, list):
                fnc_ctx = llm.FunctionContext(fnc_ctx)

            tools: list[session_update_event.SessionTool] = []
            retained_functions: list[llm.AIFunction] = []

            for ai_fnc in fnc_ctx.ai_functions.values():
                tool_desc = llm.utils.build_legacy_openai_schema(
                    ai_fnc, internally_tagged=True
                )
                try:
                    session_tool = session_update_event.SessionTool.model_validate(
                        tool_desc
                    )
                    tools.append(session_tool)
                    retained_functions.append(ai_fnc)
                except ValidationError:
                    logger.error(
                        "OpenAI Realtime API doesn't support this tool",
                        extra={"tool": tool_desc},
                    )
                    continue

            event_id = utils.shortuuid("fnc_ctx_update_")
            # f = asyncio.Future()
            # self._response_futures[event_id] = f
            self._msg_ch.send_nowait(
                SessionUpdateEvent(
                    type="session.update",
                    session=session_update_event.Session(
                        model=self._realtime_model._opts.model,  # type: ignore (str -> Literal)
                        tools=tools,
                    ),
                    event_id=event_id,
                )
            )

            self._fnc_ctx = llm.FunctionContext(retained_functions)

    async def update_instructions(self, instructions: str) -> None:
        event_id = utils.shortuuid("instructions_update_")
        # f = asyncio.Future()
        # self._response_futures[event_id] = f
        self._msg_ch.send_nowait(
            SessionUpdateEvent(
                type="session.update",
                session=session_update_event.Session(
                    model=self._realtime_model._opts.model,  # type: ignore
                    instructions=instructions,
                ),
                event_id=event_id,
            )
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self._msg_ch.send_nowait(
            InputAudioBufferAppendEvent(
                type="input_audio_buffer.append",
                audio=base64.b64encode(frame.data).decode("utf-8"),
            )
        )

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future()
        self._response_created_futures[event_id] = fut
        self._msg_ch.send_nowait(
            ResponseCreateEvent(
                type="response.create",
                event_id=event_id,
                response=Response(
                    instructions=instructions or None,
                    metadata={"client_event_id": event_id},
                ),
            )
        )

        def _on_timeout() -> None:
            if fut and not fut.done():
                fut.set_exception(llm.RealtimeError("generate_reply timed out."))

        handle = asyncio.get_event_loop().call_later(5.0, _on_timeout)
        fut.add_done_callback(lambda _: handle.cancel())
        return fut

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

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        self.emit("input_speech_stopped", llm.InputSpeechStoppedEvent())

    def _handle_response_created(self, event: ResponseCreatedEvent) -> None:
        assert event.response.id is not None, "response.id is None"

        self._current_generation = _ResponseGeneration(
            message_ch=utils.aio.Chan(),
            function_ch=utils.aio.Chan(),
            messages={},
        )

        generation_ev = llm.GenerationCreatedEvent(
            message_stream=self._current_generation.message_ch,
            function_stream=self._current_generation.function_ch,
            user_initiated=False,
        )

        if (
            isinstance(event.response.metadata, dict)
            and (client_event_id := event.response.metadata.get("client_event_id"))
            and (fut := self._response_created_futures.pop(client_event_id, None))
        ):
            generation_ev.user_initiated = True
            fut.set_result(generation_ev)

        self.emit("generation_created", generation_ev)

    def _handle_response_output_item_added(
        self, event: ResponseOutputItemAddedEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "message":
            item_generation = _MessageGeneration(
                message_id=item_id,
                text_ch=utils.aio.Chan(),
                audio_ch=utils.aio.Chan(),
            )
            self._current_generation.message_ch.send_nowait(
                llm.MessageGeneration(
                    message_id=item_id,
                    text_stream=item_generation.text_ch,
                    audio_stream=item_generation.audio_ch,
                )
            )
            self._current_generation.messages[item_id] = item_generation

    def _handle_conversion_item_created(
        self, event: ConversationItemCreatedEvent
    ) -> None:
        assert event.item.id is not None, "item.id is None"

        self._remote_chat_ctx.insert(
            event.previous_item_id, _openai_item_to_livekit_item(event.item)
        )
        if fut := self._item_create_future.pop(event.item.id, None):
            fut.set_result(None)

    def _handle_conversion_item_deleted(
        self, event: ConversationItemDeletedEvent
    ) -> None:
        assert event.item_id is not None, "item_id is None"

        self._remote_chat_ctx.delete(event.item_id)

        if fut := self._item_delete_future.pop(event.item_id, None):
            fut.set_result(None)

    def _handle_response_audio_transcript_delta(
        self, event: ResponseAudioTranscriptDeltaEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]
        item_generation.text_ch.send_nowait(event.delta)

    def _handle_response_audio_delta(self, event: ResponseAudioDeltaEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        item_generation = self._current_generation.messages[event.item_id]

        data = base64.b64decode(event.delta)
        item_generation.audio_ch.send_nowait(
            rtc.AudioFrame(
                data=data,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=len(data) // 2,
            )
        )

    def _handle_response_audio_transcript_done(
        self, _: ResponseAudioTranscriptDoneEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_done(self, _: ResponseAudioDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_output_item_done(
        self, event: ResponseOutputItemDoneEvent
    ) -> None:
        assert self._current_generation is not None, "current_generation is None"
        assert (item_id := event.item.id) is not None, "item.id is None"
        assert (item_type := event.item.type) is not None, "item.type is None"

        if item_type == "function_call":
            item = event.item
            assert item.call_id is not None, "call_id is None"
            assert item.name is not None, "name is None"
            assert item.arguments is not None, "arguments is None"

            self._current_generation.function_ch.send_nowait(
                llm.FunctionCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=item.arguments,
                )
            )
        elif item_type == "message":
            item_generation = self._current_generation.messages[item_id]
            item_generation.text_ch.close()
            item_generation.audio_ch.close()

    def _handle_response_done(self, _: ResponseDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"
        self._current_generation.function_ch.close()
        self._current_generation.message_ch.close()
        self._current_generation = None

    def _handle_error(self, event: ErrorEvent) -> None:
        if event.error.message.startswith("Cancellation failed"):
            return

        logger.error(
            "OpenAI Realtime API returned an error",
            extra={"error": event.error},
        )
        self.emit(
            "error",
            llm.ErrorEvent(type=event.error.type, message=event.error.message),
        )

        # if event.error.event_id:
        #     fut = self._response_futures.pop(event.error.event_id, None)
        #     if fut is not None and not fut.done():
        #         fut.set_exception(multimodal.RealtimeError(event.error.message))


def _livekit_item_to_openai_item(item: llm.ChatItem) -> ConversationItem:
    conversation_item = ConversationItem(
        id=item.id,
    )

    if item.type == "function_call":
        conversation_item.type = "function_call"
        conversation_item.call_id = item.call_id
        conversation_item.name = item.name
        conversation_item.arguments = item.arguments

    elif item.type == "function_call_output":
        conversation_item.type = "function_call_output"
        conversation_item.call_id = item.call_id
        conversation_item.output = item.output

    elif item.type == "message":
        role = "system" if item.role == "developer" else item.role
        conversation_item.type = "message"
        conversation_item.role = role

        content_list: list[ConversationItemContent] = []
        for c in item.content:
            if isinstance(c, str):
                content_list.append(
                    ConversationItemContent(
                        type=("text" if role == "assistant" else "input_text"),
                        text=c,
                    )
                )

            elif isinstance(c, llm.ImageContent):
                continue  # not supported for now
            elif isinstance(c, llm.AudioContent):
                if conversation_item.role == "user":
                    encoded_audio = base64.b64encode(
                        rtc.combine_audio_frames(c.frame).data
                    ).decode("utf-8")

                    content_list.append(
                        ConversationItemContent(
                            type="input_audio",
                            audio=encoded_audio,
                            transcript=c.transcript,
                        )
                    )

        conversation_item.content = content_list

    return conversation_item


def _openai_item_to_livekit_item(item: ConversationItem) -> llm.ChatItem:
    assert item.id is not None, "id is None"

    if item.type == "function_call":
        assert item.call_id is not None, "call_id is None"
        assert item.name is not None, "name is None"
        assert item.arguments is not None, "arguments is None"

        return llm.FunctionCall(
            id=item.id,
            call_id=item.call_id,
            name=item.name,
            arguments=item.arguments,
        )

    if item.type == "function_call_output":
        assert item.call_id is not None, "call_id is None"
        assert item.output is not None, "output is None"

        return llm.FunctionCallOutput(
            id=item.id,
            call_id=item.call_id,
            output=item.output,
            is_error=False,
        )

    if item.type == "message":
        assert item.role is not None, "role is None"
        assert item.content is not None, "content is None"

        content: list[llm.ChatContent] = []

        for c in item.content:
            if c.type == "text" or c.type == "input_text":
                assert c.text is not None, "text is None"
                content.append(c.text)

        return llm.ChatMessage(
            id=item.id,
            role=item.role,
            content=content,
        )

    raise ValueError(f"unsupported item type: {item.type}")
