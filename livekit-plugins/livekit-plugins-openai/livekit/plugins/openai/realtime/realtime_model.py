from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import weakref
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Literal, Union
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import aiohttp
from pydantic import BaseModel, ValidationError

from livekit import rtc
from livekit.agents import io, llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from openai.types.beta.realtime import (
    ConversationItem,
    ConversationItemContent,
    ConversationItemCreatedEvent,
    ConversationItemCreateEvent,
    ConversationItemDeletedEvent,
    ConversationItemDeleteEvent,
    ConversationItemInputAudioTranscriptionCompletedEvent,
    ConversationItemInputAudioTranscriptionFailedEvent,
    ConversationItemTruncateEvent,
    ErrorEvent,
    InputAudioBufferAppendEvent,
    InputAudioBufferCommitEvent,
    InputAudioBufferSpeechStartedEvent,
    InputAudioBufferSpeechStoppedEvent,
    RealtimeClientEvent,
    ResponseAudioDeltaEvent,
    ResponseAudioDoneEvent,
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
from openai.types.beta.realtime.session import InputAudioTranscription, TurnDetection

from ..log import logger

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
OPENAI_BASE_URL = "https://api.openai.com/v1"

_log_oai_events = int(os.getenv("LOG_OAI_EVENTS", 0))


@dataclass
class _RealtimeOptions:
    model: str
    voice: str
    temperature: float | None
    input_audio_transcription: InputAudioTranscription | None
    turn_detection: TurnDetection | None
    api_key: str
    base_url: str


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
        temperature: float | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        input_audio_transcription: InputAudioTranscription | None = None,
        turn_detection: TurnDetection | None = None,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=True,
                turn_detection=turn_detection is not None,
                user_transcription=input_audio_transcription is not None,
            )
        )

        api_key = api_key or os.environ.get("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"  # noqa: E501
            )

        base_url = base_url or OPENAI_BASE_URL
        self._opts = _RealtimeOptions(
            model=model,
            voice=voice,
            temperature=temperature,
            input_audio_transcription=input_audio_transcription,
            turn_detection=turn_detection,
            api_key=api_key,
            base_url=base_url,
        )
        self._http_session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

    def update_options(self, *, voice: str, temperature: float) -> None:
        self._opts.voice = voice
        self._opts.temperature = temperature
        for sess in self._sessions:
            sess.update_options(voice=voice, temperature=temperature)

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._http_session:
            self._http_session = utils.http_context.http_session()

        return self._http_session

    def session(self) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None: ...


def process_base_url(url: str, model: str) -> str:
    if url.startswith("http"):
        url = url.replace("http", "ws", 1)

    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # ensure "/realtime" is added if the path is empty OR "/v1"
    if not parsed_url.path or parsed_url.path.rstrip("/") in ["", "/v1"]:
        path = parsed_url.path.rstrip("/") + "/realtime"

        if "model" not in query_params:
            query_params["model"] = [model]
    else:
        path = parsed_url.path

    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, path, "", new_query, ""))

    return new_url


class RealtimeSession(
    llm.RealtimeSession[Literal["openai_server_event_received", "openai_client_event_queued"]]
):
    """
    A session for the OpenAI Realtime API.

    This class is used to interact with the OpenAI Realtime API.
    It is responsible for sending events to the OpenAI Realtime API and receiving events from it.

    It exposes two more events:
    - openai_server_event_received: expose the raw server events from the OpenAI Realtime API
    - openai_client_event_queued: expose the raw client events sent to the OpenAI Realtime API
    """

    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model = realtime_model
        self._tools = llm.ToolContext.empty()
        self._msg_ch = utils.aio.Chan[Union[RealtimeClientEvent, dict]]()
        self._input_resampler: rtc.AudioResampler | None = None

        self._main_atask = asyncio.create_task(self._main_task(), name="RealtimeSession._main_task")

        self._response_created_futures: dict[str, asyncio.Future[llm.GenerationCreatedEvent]] = {}
        self._item_delete_future: dict[str, asyncio.Future] = {}
        self._item_create_future: dict[str, asyncio.Future] = {}

        self._current_generation: _ResponseGeneration | None = None
        self._remote_chat_ctx = llm.remote_chat_context.RemoteChatContext()

        self._update_chat_ctx_lock = asyncio.Lock()
        self._update_fnc_ctx_lock = asyncio.Lock()

        self._bstream = utils.audio.AudioByteStream(
            SAMPLE_RATE,
            NUM_CHANNELS,
            samples_per_channel=SAMPLE_RATE // 10,  # 100ms
        )
        self._pushed_duration_s = 0  # duration of audio pushed to the OpenAI Realtime API

    def send_event(self, event: RealtimeClientEvent | dict) -> None:
        with contextlib.suppress(utils.aio.channel.ChanClosed):
            self._msg_ch.send_nowait(event)

    @utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization": f"Bearer {self._realtime_model._opts.api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        url = process_base_url(
            self._realtime_model._opts.base_url, self._realtime_model._opts.model
        )

        if _log_oai_events:
            logger.debug(f"connecting to Realtime API: {url}")

        ws_conn = await self._realtime_model._ensure_http_session().ws_connect(
            url=url, headers=headers
        )

        closing = False

        @utils.log_exceptions(logger=logger)
        async def _send_task() -> None:
            nonlocal closing
            async for msg in self._msg_ch:
                try:
                    if isinstance(msg, BaseModel):
                        msg = msg.model_dump(
                            by_alias=True, exclude_unset=True, exclude_defaults=False
                        )

                    self.emit("openai_client_event_queued", msg)
                    await ws_conn.send_str(json.dumps(msg))

                    if _log_oai_events:
                        msg_copy = msg.copy()
                        if msg_copy["type"] == "input_audio_buffer.append":
                            msg_copy = {**msg_copy, "audio": "..."}

                        logger.debug(f">>> {msg_copy}")
                except Exception:
                    break

            closing = True
            await ws_conn.close()

        @utils.log_exceptions(logger=logger)
        async def _recv_task() -> None:
            while True:
                msg = await ws_conn.receive()
                if msg.type == aiohttp.WSMsgType.CLOSED:
                    if not closing:
                        raise Exception("OpenAI S2S connection closed unexpectedly")

                    return
                elif msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                event = json.loads(msg.data)

                # emit the raw json dictionary instead of the BaseModel because different
                # providers can have different event types that are not part of the OpenAI Realtime API  # noqa: E501
                self.emit("openai_server_event_received", event)

                try:
                    if _log_oai_events:
                        event_copy = event.copy()
                        if event_copy["type"] == "response.audio.delta":
                            event_copy = {**event_copy, "delta": "..."}

                        logger.debug(f"<<< {event_copy}")

                    if event["type"] == "input_audio_buffer.speech_started":
                        self._handle_input_audio_buffer_speech_started(
                            InputAudioBufferSpeechStartedEvent.construct(**event)
                        )
                    elif event["type"] == "input_audio_buffer.speech_stopped":
                        self._handle_input_audio_buffer_speech_stopped(
                            InputAudioBufferSpeechStoppedEvent.construct(**event)
                        )
                    elif event["type"] == "response.created":
                        self._handle_response_created(ResponseCreatedEvent.construct(**event))
                    elif event["type"] == "response.output_item.added":
                        self._handle_response_output_item_added(
                            ResponseOutputItemAddedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.created":
                        self._handle_conversion_item_created(
                            ConversationItemCreatedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.deleted":
                        self._handle_conversion_item_deleted(
                            ConversationItemDeletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.completed":
                        self._handle_conversion_item_input_audio_transcription_completed(
                            ConversationItemInputAudioTranscriptionCompletedEvent.construct(**event)
                        )
                    elif event["type"] == "conversation.item.input_audio_transcription.failed":
                        self._handle_conversion_item_input_audio_transcription_failed(
                            ConversationItemInputAudioTranscriptionFailedEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio_transcript.delta":
                        self._handle_response_audio_transcript_delta(event)
                    elif event["type"] == "response.audio.delta":
                        self._handle_response_audio_delta(
                            ResponseAudioDeltaEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio_transcript.done":
                        self._handle_response_audio_transcript_done(
                            ResponseAudioTranscriptDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.audio.done":
                        self._handle_response_audio_done(ResponseAudioDoneEvent.construct(**event))
                    elif event["type"] == "response.output_item.done":
                        self._handle_response_output_item_done(
                            ResponseOutputItemDoneEvent.construct(**event)
                        )
                    elif event["type"] == "response.done":
                        self._handle_response_done(ResponseDoneEvent.construct(**event))
                    elif event["type"] == "error":
                        self._handle_error(ErrorEvent.construct(**event))
                except Exception:
                    logger.exception("failed to handle event", extra={"event": event})

        input_audio_transcription: session_update_event.SessionInputAudioTranscription | None = None
        if self._realtime_model._opts.input_audio_transcription:
            input_audio_transcription = session_update_event.SessionInputAudioTranscription(
                model=self._realtime_model._opts.input_audio_transcription.model,
                language=self._realtime_model._opts.input_audio_transcription.language,
                prompt=self._realtime_model._opts.input_audio_transcription.prompt,
            )

        turn_detection: session_update_event.SessionTurnDetection | None = None
        if self._realtime_model._opts.turn_detection:
            turn_detection = session_update_event.SessionTurnDetection.model_validate(
                asdict(self._realtime_model._opts.turn_detection)
            )

        # initial session update
        self.send_event(
            SessionUpdateEvent(
                type="session.update",
                # Using model_construct since OpenAI restricts voices to those defined in the BaseModel.  # noqa: E501
                # Other providers support different voices, so we need to accommodate that.
                session=session_update_event.Session.model_construct(
                    model=self._realtime_model._opts.model,
                    voice=self._realtime_model._opts.voice,
                    temperature=self._realtime_model._opts.temperature,
                    input_audio_format="pcm16",
                    output_audio_format="pcm16",
                    modalities=["text", "audio"],
                    turn_detection=turn_detection,
                    input_audio_transcription=input_audio_transcription,
                ),
                event_id=utils.shortuuid("session_update_"),
            )
        )

        tasks = [
            asyncio.create_task(_recv_task(), name="_recv_task"),
            asyncio.create_task(_send_task(), name="_send_task"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws_conn.close()

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._remote_chat_ctx.to_chat_ctx()

    @property
    def tools(self) -> llm.ToolContext:
        return self._tools.copy()

    def update_options(
        self,
        *,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        temperature: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        kwargs = {}

        if utils.is_given(tool_choice):
            oai_tool_choice = tool_choice
            if isinstance(tool_choice, dict) and tool_choice["type"] == "function":
                oai_tool_choice = tool_choice["function"]

            kwargs["tool_choice"] = oai_tool_choice

        if utils.is_given(voice):
            kwargs["voice"] = voice

        if utils.is_given(temperature):
            kwargs["temperature"] = temperature

        if kwargs:
            self.send_event(
                SessionUpdateEvent(
                    type="session.update",
                    session=session_update_event.Session.model_construct(**kwargs),
                    event_id=utils.shortuuid("options_update_"),
                )
            )

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        async with self._update_chat_ctx_lock:
            diff_ops = llm.utils.compute_chat_ctx_diff(
                self._remote_chat_ctx.to_chat_ctx(), chat_ctx
            )

            futs = []

            for msg_id in diff_ops.to_remove:
                event_id = utils.shortuuid("chat_ctx_delete_")
                self.send_event(
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

                self.send_event(
                    ConversationItemCreateEvent(
                        type="conversation.item.create",
                        item=_livekit_item_to_openai_item(chat_item),
                        previous_item_id=("root" if previous_msg_id is None else previous_msg_id),
                        event_id=event_id,
                    )
                )
                futs.append(f := asyncio.Future())
                self._item_create_future[msg_id] = f

            try:
                await asyncio.wait_for(asyncio.gather(*futs, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                raise llm.RealtimeError("update_chat_ctx timed out.") from None

    async def update_tools(self, tools: list[llm.FunctionTool]) -> None:
        async with self._update_fnc_ctx_lock:
            oai_tools: list[session_update_event.SessionTool] = []
            retained_tools: list[llm.FunctionTool] = []

            for tool in tools:
                tool_desc = llm.utils.build_legacy_openai_schema(tool, internally_tagged=True)
                try:
                    session_tool = session_update_event.SessionTool.model_validate(tool_desc)
                    oai_tools.append(session_tool)
                    retained_tools.append(tool)
                except ValidationError:
                    logger.error(
                        "OpenAI Realtime API doesn't support this tool",
                        extra={"tool": tool_desc},
                    )
                    continue

            event_id = utils.shortuuid("tools_update_")
            # f = asyncio.Future()
            # self._response_futures[event_id] = f
            self.send_event(
                SessionUpdateEvent(
                    type="session.update",
                    session=session_update_event.Session.model_construct(
                        model=self._realtime_model._opts.model,
                        tools=oai_tools,
                    ),
                    event_id=event_id,
                )
            )

            self._tools = llm.ToolContext(retained_tools)

    async def update_instructions(self, instructions: str) -> None:
        event_id = utils.shortuuid("instructions_update_")
        # f = asyncio.Future()
        # self._response_futures[event_id] = f
        self.send_event(
            SessionUpdateEvent(
                type="session.update",
                session=session_update_event.Session.model_construct(instructions=instructions),
                event_id=event_id,
            )
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        for f in self._resample_audio(frame):
            for f in self._bstream.write(f.data.tobytes()):  # noqa: B020
                self.send_event(
                    InputAudioBufferAppendEvent(
                        type="input_audio_buffer.append",
                        audio=base64.b64encode(f.data).decode("utf-8"),
                    )
                )
                self._pushed_duration_s += f.duration

    def commit_audio(self) -> None:
        if self._pushed_duration_s > 0.1:  # OpenAI requires at least 100ms of audio
            self.send_event(InputAudioBufferCommitEvent(type="input_audio_buffer.commit"))
            self._pushed_duration_s = 0

    def generate_reply(
        self, *, instructions: NotGivenOr[str] = NOT_GIVEN
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        event_id = utils.shortuuid("response_create_")
        fut = asyncio.Future()
        self._response_created_futures[event_id] = fut
        self.send_event(
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
        self.send_event(ResponseCancelEvent(type="response.cancel"))

    def truncate(self, *, message_id: str, audio_end_ms: int) -> None:
        self.send_event(
            ConversationItemTruncateEvent(
                type="conversation.item.truncate",
                content_index=0,
                item_id=message_id,
                audio_end_ms=audio_end_ms,
            )
        )

    async def aclose(self) -> None:
        self._msg_ch.close()
        await self._main_atask

    def _resample_audio(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler:
            if frame.sample_rate != self._input_resampler._input_rate:
                # input audio changed to a different sample rate
                self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            # TODO(long): flush the resampler when the input source is changed
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def _handle_input_audio_buffer_speech_started(
        self, _: InputAudioBufferSpeechStartedEvent
    ) -> None:
        self.emit("input_speech_started", llm.InputSpeechStartedEvent())

    def _handle_input_audio_buffer_speech_stopped(
        self, _: InputAudioBufferSpeechStoppedEvent
    ) -> None:
        user_transcription_enabled = (
            self._realtime_model._opts.input_audio_transcription is not None
        )
        self.emit(
            "input_speech_stopped",
            llm.InputSpeechStoppedEvent(user_transcription_enabled=user_transcription_enabled),
        )

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

    def _handle_response_output_item_added(self, event: ResponseOutputItemAddedEvent) -> None:
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

    def _handle_conversion_item_created(self, event: ConversationItemCreatedEvent) -> None:
        assert event.item.id is not None, "item.id is None"

        try:
            self._remote_chat_ctx.insert(
                event.previous_item_id, _openai_item_to_livekit_item(event.item)
            )
        except ValueError as e:
            logger.warning(
                f"failed to insert item `{event.item.id}`: {str(e)}",
            )

        if fut := self._item_create_future.pop(event.item.id, None):
            fut.set_result(None)

    def _handle_conversion_item_deleted(self, event: ConversationItemDeletedEvent) -> None:
        assert event.item_id is not None, "item_id is None"

        try:
            self._remote_chat_ctx.delete(event.item_id)
        except ValueError as e:
            logger.warning(
                f"failed to delete item `{event.item_id}`: {str(e)}",
            )

        if fut := self._item_delete_future.pop(event.item_id, None):
            fut.set_result(None)

    def _handle_conversion_item_input_audio_transcription_completed(
        self, event: ConversationItemInputAudioTranscriptionCompletedEvent
    ) -> None:
        if remote_item := self._remote_chat_ctx.get(event.item_id):
            assert isinstance(remote_item.item, llm.ChatMessage)
            remote_item.item.content.append(event.transcript)

        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(item_id=event.item_id, transcript=event.transcript),
        )

    def _handle_conversion_item_input_audio_transcription_failed(
        self, event: ConversationItemInputAudioTranscriptionFailedEvent
    ) -> None:
        logger.error(
            "OpenAI Realtime API failed to transcribe input audio",
            extra={"error": event.error},
        )

    def _handle_response_audio_transcript_delta(self, event: dict) -> None:
        assert self._current_generation is not None, "current_generation is None"

        item_id = event["item_id"]
        delta = event["delta"]

        if (start_time := event.get("start_time")) is not None:
            delta = io.TimedString(delta, start_time=start_time)

        item_generation = self._current_generation.messages[item_id]
        item_generation.text_ch.send_nowait(delta)

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

    def _handle_response_audio_transcript_done(self, _: ResponseAudioTranscriptDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_audio_done(self, _: ResponseAudioDoneEvent) -> None:
        assert self._current_generation is not None, "current_generation is None"

    def _handle_response_output_item_done(self, event: ResponseOutputItemDoneEvent) -> None:
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
        if self._current_generation is None:
            return  # OpenAI has a race condition where we could receive response.done without any previous response.created (This happens generally during interruption)  # noqa: E501

        assert self._current_generation is not None, "current_generation is None"
        for generation in self._current_generation.messages.values():
            # close all messages that haven't been closed yet
            if not generation.text_ch.closed:
                generation.text_ch.close()
            if not generation.audio_ch.closed:
                generation.audio_ch.close()

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
                    encoded_audio = base64.b64encode(rtc.combine_audio_frames(c.frame).data).decode(
                        "utf-8"
                    )

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
